"""Rendering the edited project to an .mp4 file with FFmpeg (no Qt dependency).

Why FFmpeg directly instead of moviepy: one filter graph trims every kept range of
every clip, conforms them to the project format, joins them, crops and rescales in a
single decode/encode pass, with frame-exact cut points and audio kept in sync.
moviepy would push every frame through Python.

Video chain, per segment: trim -> (frame rate) -> (fit to the project size)
Then:                     concat -> constant frame rate -> crop -> scale
Audio is trimmed and joined alongside, or left out entirely when muted. A clip
without an audio track contributes silence if other clips have sound.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .crop import Crop
from .media import NO_WINDOW
from .models import (
    EditState,
    ProjectFormat,
    frame_rate_fraction,
    same_frame_rate,
    same_path,
)

# Windows allows 32767 characters on a command line; switch to a script file well before that.
MAX_COMMAND_CHARS = 24000

# Audio of different clips can differ in sample rate and channel layout; the join needs one format.
AUDIO_FORMAT = "aformat=sample_rates=48000:sample_fmts=fltp:channel_layouts=stereo"


class ExportError(Exception):
    """Export failed. The message is meant for the user."""


class ExportCancelled(Exception):
    """The user cancelled the export."""


@dataclass(frozen=True)
class ExportSettings:
    output_path: str
    width: int  # output size, i.e. after cropping and scaling
    height: int
    mute_audio: bool = False


@dataclass(frozen=True)
class RenderPlan:
    """What to hand to FFmpeg: the input files (in order) and the filter graph over them."""

    inputs: Tuple[str, ...]
    graph: str
    with_audio: bool


def even(value: int) -> int:
    """H.264 with yuv420p needs even dimensions."""
    return max(2, value - value % 2)


def frame_rate_expression(fps: float) -> str:
    """``fps`` as an FFmpeg rational such as ``30000/1001``."""
    rate = frame_rate_fraction(fps)
    return f"{rate.numerator}/{rate.denominator}"


def effective_crop(project_format: ProjectFormat, crop: Optional[Crop]) -> Optional[Crop]:
    """``crop``, or None if it would not change the picture (no crop, or the whole frame)."""
    if crop is None or crop.covers(project_format.width, project_format.height):
        return None
    return crop


def picture_size(project_format: ProjectFormat, crop: Optional[Crop]) -> Tuple[int, int]:
    """Size of the picture after cropping; also the natural output resolution."""
    crop = effective_crop(project_format, crop)
    return (crop.width, crop.height) if crop else (project_format.width, project_format.height)


def partial_path(output_path: str) -> str:
    """Temporary name used while rendering, so a failed export never clobbers an existing file."""
    base, ext = os.path.splitext(output_path)
    return f"{base}.part{ext}"


def same_file(a: str, b: str) -> bool:
    return same_path(a, b)


def _fan_out(parts: List[str], source: str, filter_name: str, prefix: str, count: int) -> List[str]:
    """One pad per use of a stream: ``split``/``asplit`` it when it is used more than once."""
    if count == 1:
        return [source]
    labels = [f"[{prefix}{i}]" for i in range(count)]
    parts.append(f"{source}{filter_name}={count}{''.join(labels)}")
    return labels


def plan_render(
    state: EditState, width: int, height: int, mute_audio: bool = False
) -> RenderPlan:
    """Build the filter graph for ``state``. ``width`` x ``height`` is the final output size."""
    fmt = state.project_format
    ranges = state.timeline.merged_ranges()
    if fmt is None or not ranges:
        raise ExportError("There is nothing to export: the timeline is empty.")

    inputs: List[str] = []  # clips in order of first use; the index is the FFmpeg input number
    for r in ranges:
        if r.source not in inputs:
            inputs.append(r.source)
    clips = {path: state.clip(path) for path in inputs}
    with_audio = not mute_audio and any(c.has_audio for c in clips.values())
    normalise_audio = len(inputs) > 1  # clips may differ in audio format
    rate = frame_rate_fraction(fmt.fps)
    rate_text = f"{rate.numerator}/{rate.denominator}"

    def seconds(frames: int) -> float:
        return float(Fraction(frames) / rate)

    parts: List[str] = []
    uses = Counter(r.source for r in ranges)
    video_pads: Dict[str, List[str]] = {}
    audio_pads: Dict[str, List[str]] = {}
    for k, path in enumerate(inputs):
        video_pads[path] = _fan_out(parts, f"[{k}:v]", "split", f"v{k}_", uses[path])
        if with_audio and clips[path].has_audio:
            audio_pads[path] = _fan_out(parts, f"[{k}:a]", "asplit", f"a{k}_", uses[path])

    joined = ""
    for i, r in enumerate(ranges):
        clip = clips[r.source]
        start, end = seconds(r.start), seconds(r.end)

        # Video: cut the range out, then conform it to the project frame rate and size.
        if same_frame_rate(clip.fps, fmt.fps):
            chain = [f"trim=start_frame={r.start}:end_frame={r.end}", "setpts=PTS-STARTPTS"]
        else:
            chain = [f"trim=start={start:.6f}:end={end:.6f}", "setpts=PTS-STARTPTS", f"fps={rate_text}"]
        if (clip.width, clip.height) != (fmt.width, fmt.height):
            chain += [
                f"scale={fmt.width}:{fmt.height}:force_original_aspect_ratio=decrease",
                f"pad={fmt.width}:{fmt.height}:(ow-iw)/2:(oh-ih)/2",
                "setsar=1",
            ]
        parts.append(f"{video_pads[r.source].pop(0)}{','.join(chain)}[sv{i}]")
        joined += f"[sv{i}]"

        if with_audio:
            if clip.has_audio:
                head = audio_pads[r.source].pop(0)
                achain = [f"atrim=start={start:.6f}:end={end:.6f}", "asetpts=PTS-STARTPTS"]
            else:  # silence of the same length keeps the sound in step with the picture
                head = ""
                achain = ["anullsrc=r=48000:cl=stereo", f"atrim=end={end - start:.6f}", "asetpts=PTS-STARTPTS"]
            if normalise_audio:
                achain.append(AUDIO_FORMAT)
            parts.append(f"{head}{','.join(achain)}[sa{i}]")
            joined += f"[sa{i}]"

    concat_out = "[cv][aout]" if with_audio else "[cv]"
    parts.append(f"{joined}concat=n={len(ranges)}:v=1:a={1 if with_audio else 0}{concat_out}")

    # The concat filter output has no frame rate of its own; without an explicit one FFmpeg
    # assumes 25 fps and silently drops frames from e.g. 30 fps material.
    tail = [f"fps={rate_text}"]
    crop = effective_crop(fmt, state.crop)
    if crop is not None:  # cropping comes before scaling, so the output size is of the cropped area
        tail.append(f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}")
    if (width, height) != picture_size(fmt, crop):
        tail.append(f"scale={width}:{height}:flags=lanczos,setsar=1")
    parts.append(f"[cv]{','.join(tail)}[vout]")
    return RenderPlan(tuple(inputs), ";".join(parts), with_audio)


def build_command(
    ffmpeg: str,
    plan: RenderPlan,
    output: str,
    graph_file: Optional[str] = None,
) -> List[str]:
    """Full FFmpeg command line. ``graph_file`` replaces the inline graph for very long graphs."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for path in plan.inputs:
        cmd += ["-i", path]
    cmd += ["-filter_complex_script", graph_file] if graph_file else ["-filter_complex", plan.graph]
    cmd += ["-map", "[vout]"]
    if plan.with_audio:
        cmd += ["-map", "[aout]"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if plan.with_audio else ["-an"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", output]
    return cmd


def _command_length(cmd: Sequence[str]) -> int:
    return sum(len(arg) + 1 for arg in cmd)


class Exporter:
    """Runs one export. :meth:`run` blocks, so call it from a worker thread."""

    def __init__(self, ffmpeg: Optional[str], state: EditState, settings: ExportSettings) -> None:
        self._ffmpeg = ffmpeg
        self._state = state
        self._settings = settings
        self._proc: Optional["subprocess.Popen[str]"] = None
        self._cancelled = False

    def cancel(self) -> None:
        """Stop the export. Safe to call from any thread."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def run(self, on_progress: Callable[[float], None] = lambda fraction: None) -> str:
        """Render the timeline and return the output path.

        Raises :class:`ExportError` on failure and :class:`ExportCancelled` if cancelled.
        """
        state, settings = self._state, self._settings
        fmt = state.project_format
        if not self._ffmpeg:
            raise ExportError(
                "FFmpeg was not found.\n\nInstall it with  pip install imageio-ffmpeg  "
                "or add ffmpeg to your PATH, then try again."
            )
        if fmt is None or not state.timeline.segments:
            raise ExportError("There is nothing to export: the timeline is empty.")
        output = os.path.abspath(settings.output_path)
        for source in state.timeline.sources():
            if same_file(output, source):
                raise ExportError("The output file must be different from the source videos.")
            if not os.path.isfile(source):
                raise ExportError(f"A source video is missing (moved or deleted?):\n{source}")
        if not os.path.isdir(os.path.dirname(output)):
            raise ExportError(f"The output folder does not exist:\n{os.path.dirname(output)}")
        if state.crop is not None and not state.crop.fits_in(fmt.width, fmt.height):
            raise ExportError("The crop area lies outside the video frame. Please adjust the crop.")

        plan = plan_render(state, even(settings.width), even(settings.height), settings.mute_audio)
        temp_output = partial_path(output)
        graph_file: Optional[str] = None
        stderr_file = tempfile.TemporaryFile()
        try:
            cmd = build_command(self._ffmpeg, plan, temp_output)
            if _command_length(cmd) > MAX_COMMAND_CHARS:
                fd, graph_file = tempfile.mkstemp(suffix=".ffgraph")
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(plan.graph)
                cmd = build_command(self._ffmpeg, plan, temp_output, graph_file)

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=NO_WINDOW,
                )
            except OSError as exc:
                raise ExportError(f"FFmpeg could not be started:\n{exc}") from exc
            if self._cancelled:  # cancel() arrived before the process existed
                self._proc.terminate()

            total_seconds = state.timeline.total_frames / fmt.fps
            on_progress(0.0)
            for line in self._proc.stdout:  # "key=value" lines written by -progress
                key, _, value = line.strip().partition("=")
                if key in ("out_time_us", "out_time_ms"):  # both are microseconds
                    try:
                        micros = int(value)
                    except ValueError:
                        continue
                    if micros >= 0:
                        on_progress(min(0.99, micros / 1e6 / total_seconds))
            returncode = self._proc.wait()

            if self._cancelled:
                raise ExportCancelled()
            if returncode != 0:
                stderr_file.seek(0)
                details = stderr_file.read().decode("utf-8", errors="replace").strip()
                tail = "\n".join(details.splitlines()[-12:]) or "(no details)"
                raise ExportError(f"FFmpeg failed (exit code {returncode}):\n\n{tail}")
            if not os.path.isfile(temp_output) or os.path.getsize(temp_output) == 0:
                raise ExportError("FFmpeg finished but did not produce an output file.")
            try:
                os.replace(temp_output, output)
            except OSError as exc:
                raise ExportError(
                    f"The video was rendered but could not be saved to:\n{output}\n\n{exc}\n\n"
                    "If the file is open in another program, close it and export again."
                ) from exc
            on_progress(1.0)
            return output
        finally:
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()
            stderr_file.close()
            for leftover in (temp_output, graph_file):
                if leftover and os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass
