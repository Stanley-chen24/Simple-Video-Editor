"""Preview playback of the edited timeline.

The player works in *sequence frames* (see :mod:`videoeditor.models`): it maps each
one to a clip and a clip frame, decodes that frame with OpenCV, fits it to the
project format and emits it. Playback is paced by the wall clock and drops frames
if decoding cannot keep up, so the timeline always advances in real time. Sound
is played by :mod:`videoeditor.audio`.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

from .audio import SequenceAudio
from .media import VideoError, VideoReader, fit_to_canvas
from .models import EditList, ProjectFormat, VideoInfo, source_frame

# Decoders hold a lot of memory, so only the most recently used clips stay open.
MAX_OPEN_READERS = 4


class Player(QObject):
    frame_ready = pyqtSignal(object)  # numpy BGR image, already the size of the project format
    frame_cleared = pyqtSignal()  # nothing to show (the timeline is empty)
    position_changed = pyqtSignal(int)  # current sequence frame
    state_changed = pyqtSignal(bool)  # True while playing
    read_failed = pyqtSignal(str)  # path of a clip that could not be opened

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._infos: Dict[str, VideoInfo] = {}
        self._format: Optional[ProjectFormat] = None
        self._edit = EditList()
        self._readers: "OrderedDict[str, VideoReader]" = OrderedDict()
        self._failed: Set[str] = set()  # clips that could not be opened; not retried
        self._cache: Optional[Tuple[str, int, np.ndarray]] = None  # last decoded (clip, file frame, image)
        self._pos = 0  # current sequence frame
        self._shown: Optional[Tuple[str, int]] = None  # (clip, clip frame) currently on screen
        self._playing = False
        self._audio = SequenceAudio()
        self._clock_start = 0.0  # wall-clock time and sequence frame playback (re)started from
        self._clock_frame = 0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------- properties
    @property
    def position(self) -> int:
        return self._pos

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def total_frames(self) -> int:
        return self._edit.total_frames

    # ---------------------------------------------------------------- project
    def set_project(
        self, pool: Sequence[VideoInfo], project_format: Optional[ProjectFormat], edit: EditList
    ) -> None:
        """Switch to a new media pool / timeline (after an import, edit, undo or redo).

        The playhead stays on the same clip frame if that is still on the timeline,
        otherwise it moves to where that frame used to be.
        """
        old_format = self._format
        self._infos = {info.path: info for info in pool}
        for path in [p for p in self._readers if p not in self._infos]:
            self._readers.pop(path).close()  # the clip left the project
        self._failed &= set(self._infos)
        if self._cache is not None and self._cache[0] not in self._infos:
            self._cache = None
        self._format = project_format
        self._audio.set_project(pool, project_format, edit)
        shown = self._shown
        self._edit = edit

        if edit.total_frames == 0 or project_format is None:
            self.pause()
            self._pos, self._shown, self._cache = 0, None, None
            self.frame_cleared.emit()
            self.position_changed.emit(0)
            return

        pos = edit.nearest_seq(*shown) if shown is not None else None
        if pos is None:
            pos = self._pos  # the clip is gone: stay at the same place on the timeline
        pos = max(0, min(pos, edit.total_frames - 1))
        if self._playing:
            self._restart_clock(pos)
            self._audio.start(pos)
        index, clip_frame = edit.locate(pos)
        if shown == (edit.segments[index].source, clip_frame) and project_format == old_format:
            self._pos = pos  # same picture, only its position on the timeline changed
            self.position_changed.emit(pos)
        else:
            self._show(pos)

    def close(self) -> None:
        self.pause()
        self._audio.stop()
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._cache = None

    # -------------------------------------------------------------- transport
    def seek(self, seq_frame: int) -> None:
        if not self._edit.total_frames:
            return
        seq_frame = max(0, min(seq_frame, self._edit.total_frames - 1))
        if self._playing:
            self._restart_clock(seq_frame)
            self._audio.start(seq_frame)
        self._show(seq_frame)

    def step(self, delta: int) -> None:
        """Pause and move by ``delta`` frames."""
        self.pause()
        self.seek(self._pos + delta)

    def play(self) -> None:
        if not self._edit.total_frames or self._playing:
            return
        if self._pos >= self._edit.total_frames - 1:
            self._show(0)  # at the end: start over
        self._restart_clock(self._pos)
        self._audio.start(self._pos)
        self._playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._audio.stop()
        self._timer.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    # --------------------------------------------------------------- internals
    def _restart_clock(self, from_frame: int) -> None:
        self._clock_start = time.perf_counter()
        self._clock_frame = from_frame

    def _on_tick(self) -> None:
        if not self._edit.total_frames or self._format is None:
            return
        elapsed = time.perf_counter() - self._clock_start
        target = self._clock_frame + int(elapsed * self._format.fps)
        last = self._edit.total_frames - 1
        if target <= self._pos:
            return
        self._show(min(target, last))
        if target >= last:
            self.pause()

    def _show(self, seq_frame: int) -> None:
        index, clip_frame = self._edit.locate(seq_frame)
        path = self._edit.segments[index].source
        self._pos, self._shown = seq_frame, (path, clip_frame)
        frame = self._decode(path, clip_frame)
        if frame is not None:  # an undecodable frame just leaves the previous picture up
            self.frame_ready.emit(frame)
        self.position_changed.emit(seq_frame)

    def _decode(self, path: str, clip_frame: int) -> Optional[np.ndarray]:
        """The picture for a clip frame, fitted to the project format."""
        info, fmt = self._infos[path], self._format
        assert fmt is not None
        file_frame = source_frame(info, fmt.fps, clip_frame)
        if self._cache is not None and self._cache[:2] == (path, file_frame):
            return self._cache[2]  # e.g. a 24 fps clip shows some frames twice at 30 fps
        reader = self._reader_for(path)
        if reader is None:
            return None
        raw = reader.read(file_frame)
        if raw is None:
            return None
        image = fit_to_canvas(raw, fmt.width, fmt.height)
        self._cache = (path, file_frame, image)
        return image

    def _reader_for(self, path: str) -> Optional[VideoReader]:
        reader = self._readers.get(path)
        if reader is not None:
            self._readers.move_to_end(path)
            return reader
        if path in self._failed:
            return None
        try:
            reader = VideoReader(path)
        except VideoError:
            self._failed.add(path)
            self.read_failed.emit(path)
            return None
        self._readers[path] = reader
        while len(self._readers) > MAX_OPEN_READERS:
            _, oldest = self._readers.popitem(last=False)
            oldest.close()
        return reader
