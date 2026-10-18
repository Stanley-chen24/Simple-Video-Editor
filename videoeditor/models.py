"""Core data model of the editor.

This module deliberately has no Qt or OpenCV imports, so the editing logic can
be unit-tested on its own.

Terminology
-----------
* *clip*           - a video file in the media pool (a :class:`VideoInfo`).
* *project format* - size and frame rate of the sequence, taken from the first clip
  imported. Clips that differ are fitted to it (letterboxed / frame-rate converted).
* *clip frame*     - frame index of a clip resampled to the project frame rate. When a
  clip already has the project frame rate this is simply its frame number in the file.
* *sequence frame* - index of a frame on the timeline, i.e. after the segments have
  been arranged and the deleted ones removed.

All cut points are whole frames, so every edit is frame-accurate.
"""
from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

from .crop import Crop


class EditError(ValueError):
    """An editing operation cannot be performed. The message is meant for the user."""


def format_timecode(seconds: float, show_ms: bool = True) -> str:
    """Format ``seconds`` as ``HH:MM:SS`` or ``HH:MM:SS.mmm``."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    secs, ms = divmod(total_ms, 1000)
    hours, rem = divmod(secs, 3600)
    minutes, secs = divmod(rem, 60)
    text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{text}.{ms:03d}" if show_ms else text


def frame_rate_fraction(fps: float) -> Fraction:
    """``fps`` as an exact fraction such as ``30000/1001`` (OpenCV reports 29.97002997...)."""
    return Fraction(fps).limit_denominator(1001)


def same_frame_rate(a: float, b: float) -> bool:
    return frame_rate_fraction(a) == frame_rate_fraction(b)


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


@dataclass(frozen=True)
class VideoInfo:
    """Metadata of a clip in the media pool."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass(frozen=True)
class ProjectFormat:
    """Size and frame rate every clip is conformed to. Fixed by the first clip imported."""

    width: int
    height: int
    fps: float

    @classmethod
    def of(cls, info: VideoInfo) -> "ProjectFormat":
        return cls(info.width, info.height, info.fps)


def clip_frame_count(info: VideoInfo, fps: float) -> int:
    """Number of clip frames: the file's frames resampled to the project frame rate ``fps``."""
    if same_frame_rate(info.fps, fps):
        return info.frame_count
    return max(1, round(info.frame_count * frame_rate_fraction(fps) / frame_rate_fraction(info.fps)))


def source_frame(info: VideoInfo, fps: float, clip_frame: int) -> int:
    """File frame shown for a clip frame (frames repeat or are skipped if the rates differ)."""
    if same_frame_rate(info.fps, fps):
        return clip_frame
    exact = clip_frame * frame_rate_fraction(info.fps) / frame_rate_fraction(fps)
    return min(info.frame_count - 1, int(exact))


@dataclass(frozen=True)
class Segment:
    """A run of clip frames ``[start, end)`` taken from the clip at path ``source``."""

    source: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid segment [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EditList:
    """The timeline: an ordered tuple of segments played back to back. May be empty.

    Instances are immutable; every editing method returns a new ``EditList``.
    That makes undo/redo trivial (see :mod:`videoeditor.history`) because a
    snapshot is just a reference to an old instance.
    """

    segments: Tuple[Segment, ...] = ()
    # Cached sequence-frame offset of every segment, filled in __post_init__.
    _starts: Tuple[int, ...] = field(init=False, repr=False, compare=False)
    _total: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        starts: List[int] = []
        acc = 0
        for seg in self.segments:
            starts.append(acc)
            acc += seg.length
        object.__setattr__(self, "_starts", tuple(starts))
        object.__setattr__(self, "_total", acc)

    # ------------------------------------------------------------------ queries
    @classmethod
    def whole(cls, source: str, clip_frames: int) -> "EditList":
        """A timeline holding one clip, untouched."""
        return cls((Segment(source, 0, clip_frames),))

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def total_frames(self) -> int:
        return self._total

    def seq_start(self, index: int) -> int:
        """Sequence frame at which segment ``index`` begins."""
        return self._starts[index]

    def index_at(self, seq_frame: int) -> int:
        """Index of the segment containing ``seq_frame``."""
        if not 0 <= seq_frame < self._total:
            raise IndexError(seq_frame)
        return bisect.bisect_right(self._starts, seq_frame) - 1

    def locate(self, seq_frame: int) -> Tuple[int, int]:
        """Map a sequence frame to ``(segment index, clip frame)``."""
        i = self.index_at(seq_frame)
        return i, self.segments[i].start + (seq_frame - self._starts[i])

    def nearest_seq(self, source: str, clip_frame: int) -> Optional[int]:
        """Where a clip frame is on the timeline, even if it has been deleted.

        If the frame is still there this is its (first) sequence position. If it was
        deleted, it is the position of the next kept frame of the same clip, or just
        after its last segment. None if the clip is no longer on the timeline at all.
        """
        after: Optional[int] = None
        end_of_last: Optional[int] = None
        for i, seg in enumerate(self.segments):
            if seg.source != source:
                continue
            if seg.start <= clip_frame < seg.end:
                return self._starts[i] + clip_frame - seg.start
            if seg.start > clip_frame and after is None:
                after = self._starts[i]
            end_of_last = self._starts[i] + seg.length
        return after if after is not None else end_of_last

    def sources(self) -> List[str]:
        """Clip paths used by the timeline, in order of first appearance."""
        seen: List[str] = []
        for seg in self.segments:
            if seg.source not in seen:
                seen.append(seg.source)
        return seen

    def merged_ranges(self) -> List[Segment]:
        """Segments with neighbours that are contiguous in the same clip coalesced.

        Splitting without deleting leaves segments that touch in the clip; rendering
        them as one range avoids a pointless seam in the export.
        """
        out: List[Segment] = []
        for seg in self.segments:
            if out and out[-1].source == seg.source and out[-1].end == seg.start:
                out[-1] = Segment(seg.source, out[-1].start, seg.end)
            else:
                out.append(seg)
        return out

    # ------------------------------------------------------------------ cutting
    def split_at(self, seq_frame: int) -> "EditList":
        """Cut the timeline just before ``seq_frame``."""
        if not self._total:
            raise EditError("The timeline is empty.")
        if not 0 <= seq_frame < self._total:
            raise EditError("The playhead is outside the timeline.")
        result = self._cut(seq_frame)
        if result is self:
            raise EditError("There is already a cut at the playhead.")
        return result

    def split_range(self, first: int, last_exclusive: int) -> Tuple["EditList", List[int]]:
        """Cut at both ends of the sequence range ``[first, last_exclusive)``.

        Returns the new list and the indices of the segments inside the range.
        """
        if not 0 <= first < last_exclusive <= self._total:
            raise EditError("The marked range is outside the timeline.")
        result = self._cut(first)._cut(last_exclusive)
        inside = [i for i, start in enumerate(result._starts) if first <= start < last_exclusive]
        return result, inside

    def delete(self, indices: Iterable[int]) -> "EditList":
        """Remove the segments at ``indices``; the rest closes up (ripple delete).

        Deleting everything is allowed and leaves an empty timeline: the clips stay in
        the media pool and can be placed again.
        """
        doomed = set(indices)
        if not doomed:
            raise EditError("No segment is selected.")
        if not all(0 <= i < len(self.segments) for i in doomed):
            raise EditError("The selection no longer matches the timeline.")
        return EditList(tuple(s for i, s in enumerate(self.segments) if i not in doomed))

    # --------------------------------------------------------------- arranging
    def append(self, segments: Iterable[Segment]) -> "EditList":
        """Place ``segments`` at the end of the timeline."""
        return EditList(self.segments + tuple(segments))

    def remove_source(self, source: str) -> "EditList":
        """Drop every segment that comes from the clip ``source``."""
        return EditList(tuple(s for s in self.segments if s.source != source))

    def shift(self, indices: Iterable[int], delta: int) -> Tuple["EditList", List[int]]:
        """Move the selected segments one place earlier (``delta=-1``) or later (``+1``).

        Each selected segment trades places with its unselected neighbour, so a group
        moves together. Returns the new list and the new indices of the moved segments.
        """
        selected = set(indices)
        if not selected:
            raise EditError("No segment is selected.")
        if not all(0 <= i < len(self.segments) for i in selected):
            raise EditError("The selection no longer matches the timeline.")
        segs = list(self.segments)
        marked = [i in selected for i in range(len(segs))]
        order = range(len(segs)) if delta < 0 else range(len(segs) - 1, -1, -1)
        changed = False
        for i in order:
            j = i + (-1 if delta < 0 else 1)
            if marked[i] and 0 <= j < len(segs) and not marked[j]:
                segs[i], segs[j] = segs[j], segs[i]
                marked[i], marked[j] = marked[j], marked[i]
                changed = True
        if not changed:
            raise EditError(
                "The selection is already at the start of the timeline."
                if delta < 0
                else "The selection is already at the end of the timeline."
            )
        return EditList(tuple(segs)), [i for i, m in enumerate(marked) if m]

    def move_to(self, indices: Iterable[int], gap: int) -> Tuple["EditList", List[int]]:
        """Move the selected segments (keeping their order) to just before segment ``gap``.

        ``gap`` counts positions in the current list: 0 puts them first, ``len(self)`` last.
        Returns the new list and the new indices of the moved segments.
        """
        selected = sorted(set(indices))
        if not selected:
            raise EditError("No segment is selected.")
        if not all(0 <= i < len(self.segments) for i in selected) or not 0 <= gap <= len(self):
            raise EditError("The selection no longer matches the timeline.")
        moving = [self.segments[i] for i in selected]
        staying = [s for i, s in enumerate(self.segments) if i not in set(selected)]
        position = gap - sum(1 for i in selected if i < gap)
        result = tuple(staying[:position] + moving + staying[position:])
        if result == self.segments:
            raise EditError("The segments are already there.")
        return EditList(result), list(range(position, position + len(moving)))

    # ---------------------------------------------------------------- internals
    def _cut(self, seq_frame: int) -> "EditList":
        """Split before ``seq_frame``; returns ``self`` if that is already a boundary."""
        if seq_frame <= 0 or seq_frame >= self._total:
            return self
        i, frame = self.locate(seq_frame)
        seg = self.segments[i]
        if frame == seg.start:
            return self
        pieces = (Segment(seg.source, seg.start, frame), Segment(seg.source, frame, seg.end))
        return EditList(self.segments[:i] + pieces + self.segments[i + 1:])


@dataclass(frozen=True)
class EditState:
    """Everything the user can undo: the media pool, the timeline and the crop.

    Immutable like :class:`EditList`, so :class:`~videoeditor.history.History` can keep
    snapshots of it and Ctrl+Z / Ctrl+Y restore every kind of edit together.
    """

    pool: Tuple[VideoInfo, ...] = ()
    project_format: Optional[ProjectFormat] = None  # fixed by the first clip; None while the pool is empty
    timeline: EditList = EditList()
    crop: Optional[Crop] = None  # None = "Original": the full frame (in project-format pixels)

    @classmethod
    def initial(cls) -> "EditState":
        """An empty project."""
        return cls()

    # ----------------------------------------------------------------- queries
    def clip(self, path: str) -> VideoInfo:
        for info in self.pool:
            if info.path == path:
                return info
        raise KeyError(path)

    @property
    def fps(self) -> float:
        return self.project_format.fps if self.project_format else 0.0

    def clip_segment(self, info: VideoInfo) -> Segment:
        """A segment covering the whole of ``info`` at the project frame rate."""
        assert self.project_format is not None
        return Segment(info.path, 0, clip_frame_count(info, self.project_format.fps))

    def is_pristine(self) -> bool:
        """True if nothing has been done beyond importing clips (nothing worth a 'save?' prompt)."""
        if self.crop is not None:
            return False
        if self.project_format is None:
            return not self.timeline.segments
        return self.timeline.segments == tuple(self.clip_segment(i) for i in self.pool)

    # ------------------------------------------------------------------- edits
    def with_timeline(self, timeline: EditList) -> "EditState":
        return replace(self, timeline=timeline)

    def with_crop(self, crop: Optional[Crop]) -> "EditState":
        return replace(self, crop=crop)

    def add_clips(self, infos: Sequence[VideoInfo]) -> "EditState":
        """Import clips into the pool (a file that is already there is not added twice)
        and place each of them, whole, at the end of the timeline."""
        pool = list(self.pool)
        project_format = self.project_format
        placed: List[Segment] = []
        for info in infos:
            known = next((p for p in pool if same_path(p.path, info.path)), None)
            if known is None:
                pool.append(info)
                known = info
            if project_format is None:
                project_format = ProjectFormat.of(known)
            placed.append(Segment(known.path, 0, clip_frame_count(known, project_format.fps)))
        return replace(
            self,
            pool=tuple(pool),
            project_format=project_format,
            timeline=self.timeline.append(placed),
        )

    def add_to_timeline(self, path: str) -> "EditState":
        """Place a pool clip, whole, at the end of the timeline."""
        try:
            info = self.clip(path)
        except KeyError:
            raise EditError("That clip is not in the project.") from None
        return self.with_timeline(self.timeline.append([self.clip_segment(info)]))

    def remove_clip(self, path: str) -> "EditState":
        """Take a clip out of the pool and off the timeline. An empty pool resets the project."""
        pool = tuple(p for p in self.pool if p.path != path)
        if len(pool) == len(self.pool):
            raise EditError("That clip is not in the project.")
        if not pool:
            return EditState()
        return replace(self, pool=pool, timeline=self.timeline.remove_source(path))
