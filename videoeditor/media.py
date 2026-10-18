"""Video access: locating FFmpeg, probing files and frame-accurate reading.

OpenCV is used for the preview because it decodes to plain numpy arrays and
can seek to an exact frame index, which the editing model is built on.
FFmpeg (see :mod:`videoeditor.exporter`) does the final rendering.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

from .models import VideoInfo

# Stops a console window flashing up when FFmpeg is started from a GUI app on Windows.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_MAX_COUNT_CORRECTION = 30  # how many frames we are willing to trim off a too-high frame count


class VideoError(Exception):
    """A file could not be opened or decoded. The message is meant for the user."""


@lru_cache(maxsize=1)
def find_ffmpeg() -> Optional[str]:
    """Path of an FFmpeg executable: the one on PATH, else the imageio-ffmpeg binary."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # package missing, or it has no binary for this platform
        return None


def _open_capture(path: str) -> "cv2.VideoCapture":
    # Ask for the FFmpeg backend explicitly: on Windows the default can pick Media
    # Foundation, which seeks less reliably.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path)
    return cap


def _probe_container(path: str) -> Tuple[bool, Optional[float]]:
    """Return ``(has_audio_stream, container_duration_seconds)`` using ``ffmpeg -i``."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False, None
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-i", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None
    # `ffmpeg -i` with no output file always exits with an error, but it prints the stream list first.
    text = proc.stderr.decode("utf-8", errors="replace")
    match = _DURATION_RE.search(text)
    duration = None
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return bool(_AUDIO_RE.search(text)), duration


def _verified_frame_count(cap: "cv2.VideoCapture", reported: int) -> int:
    """Container frame counts are sometimes a few frames too high; check the tail."""
    count = reported
    for _ in range(_MAX_COUNT_CORRECTION):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        ok, _frame = cap.read()
        if ok:
            return count
        count -= 1
        if count < 1:
            break
    return reported  # the tail could not be verified at all; trust the container


def probe_video(path: str) -> VideoInfo:
    """Read the metadata of ``path``. Raises :class:`VideoError` if it is not usable."""
    if not os.path.isfile(path):
        raise VideoError("The file does not exist.")
    cap = _open_capture(path)
    try:
        if not cap.isOpened():
            raise VideoError(
                "The file could not be opened. It may be corrupted or in an unsupported format."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("No video frames could be decoded from this file.")
        height, width = frame.shape[:2]
        if not 0.1 <= fps <= 1000.0:
            raise VideoError("The frame rate of this video could not be determined.")

        has_audio, container_duration = _probe_container(path)
        if reported <= 0 and container_duration:
            reported = int(round(container_duration * fps))
        if reported <= 0:
            raise VideoError("The length of this video could not be determined.")
        frame_count = _verified_frame_count(cap, reported)
    finally:
        cap.release()
    return VideoInfo(
        path=os.path.abspath(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=has_audio,
    )


def fit_to_canvas(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``frame`` to fit inside ``width`` x ``height`` and centre it on black bars.

    This is how clips whose size differs from the project format are shown (the export
    does the same with FFmpeg's scale + pad). A frame that already fits is returned as is.
    """
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame
    scale = min(width / src_w, height / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (min(new_w, width), min(new_h, height)), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


class VideoReader:
    """Random access to decoded frames (BGR numpy arrays).

    Sequential reads are cheap, short forward jumps are done by decoding and
    discarding frames, and everything else falls back to a real seek.
    """

    MAX_SKIP = 48  # a forward gap up to this many frames is cheaper to decode than to seek

    def __init__(self, path: str) -> None:
        self._cap = _open_capture(path)
        if not self._cap.isOpened():
            self._cap.release()
            raise VideoError("The video could not be opened for playback.")
        self._next: Optional[int] = 0  # index the next cap.read() returns; None = unknown

    def read(self, index: int) -> Optional[np.ndarray]:
        """Decode frame ``index``; returns None if it cannot be decoded."""
        if self._next is not None and 0 < index - self._next <= self.MAX_SKIP:
            for _ in range(index - self._next):
                if not self._cap.grab():
                    self._next = None
                    break
            else:
                self._next = index
        if self._next != index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._next = None  # position is now uncertain: force a seek next time
            return None
        self._next = index + 1
        return frame

    def close(self) -> None:
        self._cap.release()
