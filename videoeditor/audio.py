"""Audio for the preview: plays the sound of the timeline while the video plays.

Each clip's sound is decoded once by FFmpeg (in a background thread) to 44.1 kHz stereo
16-bit PCM and kept in memory. On play, a small QIODevice hands the audio output the
samples of every segment from the playhead onwards, in timeline order; a clip without a
sound track, or one that is still being decoded, plays as silence. Whether the sound is
included in the *export* is a separate choice made in the Export dialog.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QIODevice
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudioFormat, QAudioOutput

from .media import NO_WINDOW, find_ffmpeg
from .models import EditList, ProjectFormat, VideoInfo

RATE = 44100
CHANNELS = 2
_BYTES_PER_FRAME = 2 * CHANNELS


def decode_clip_audio(ffmpeg: str, path: str) -> Optional[np.ndarray]:
    """The whole sound track of ``path`` as an int16 array of shape (samples, 2)."""
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-vn",
             "-f", "s16le", "-ac", str(CHANNELS), "-ar", str(RATE), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    usable = len(proc.stdout) // _BYTES_PER_FRAME * _BYTES_PER_FRAME
    if proc.returncode != 0 or not usable:
        return None
    return np.frombuffer(proc.stdout[:usable], dtype="<i2").reshape(-1, CHANNELS)


class _Feed(QIODevice):
    """Read-only stream of the timeline's audio, produced on demand."""

    def __init__(self, pieces: Iterator[bytes]) -> None:
        super().__init__()
        self._pieces = pieces
        self._rest = b""
        self.open(QIODevice.ReadOnly)

    def isSequential(self) -> bool:  # noqa: N802 (Qt naming)
        return True

    def bytesAvailable(self) -> int:  # noqa: N802 (Qt naming)
        return len(self._rest) + super().bytesAvailable()

    def readData(self, maxlen: int) -> bytes:  # noqa: N802 (Qt naming)
        while len(self._rest) < maxlen:
            piece = next(self._pieces, None)
            if piece is None:
                break
            self._rest += piece
        data, self._rest = self._rest[:maxlen], self._rest[maxlen:]
        return data

    def writeData(self, data) -> int:  # noqa: N802 (Qt naming)
        return 0


class SequenceAudio:
    """Plays the audio of the current timeline from any sequence frame."""

    def __init__(self) -> None:
        self._ffmpeg = find_ffmpeg()
        self._clips: Dict[str, VideoInfo] = {}
        self._pcm: Dict[str, Optional[np.ndarray]] = {}  # decoded sound; None = no usable sound
        self._pending: set = set()
        self._format: Optional[ProjectFormat] = None
        self._edit = EditList()
        self._output: Optional[QAudioOutput] = None
        self._feed: Optional[_Feed] = None

        fmt = QAudioFormat()
        fmt.setSampleRate(RATE)
        fmt.setChannelCount(CHANNELS)
        fmt.setSampleSize(16)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)
        device = QAudioDeviceInfo.defaultOutputDevice()
        # No sound card (or an unsupported format) just means a silent preview.
        self._audio_format = fmt if not device.isNull() and device.isFormatSupported(fmt) else None

    @property
    def available(self) -> bool:
        return self._audio_format is not None and self._ffmpeg is not None

    def set_project(
        self, pool: Sequence[VideoInfo], project_format: Optional[ProjectFormat], edit: EditList
    ) -> None:
        self._clips = {info.path: info for info in pool}
        self._format, self._edit = project_format, edit
        for path in [p for p in self._pcm if p not in self._clips]:
            del self._pcm[path]
        if self.available:
            for info in pool:
                if info.has_audio and info.path not in self._pcm and info.path not in self._pending:
                    self._pending.add(info.path)
                    threading.Thread(target=self._decode, args=(info.path,), daemon=True).start()

    def _decode(self, path: str) -> None:
        pcm = decode_clip_audio(self._ffmpeg, path) if self._ffmpeg else None
        self._pending.discard(path)
        if path in self._clips:
            self._pcm[path] = pcm

    # ---------------------------------------------------------------- playback
    def start(self, seq_frame: int) -> None:
        """(Re)start playing at ``seq_frame``."""
        self.stop()
        if not self.available or self._format is None or not self._edit.total_frames:
            return
        self._feed = _Feed(self._pieces(seq_frame))
        self._output = QAudioOutput(self._audio_format)
        self._output.start(self._feed)

    def stop(self) -> None:
        if self._output is not None:
            self._output.stop()
            self._output.deleteLater()
        self._output = self._feed = None

    def _pieces(self, seq_frame: int) -> Iterator[bytes]:
        """Audio bytes for the timeline from ``seq_frame`` to its end, a segment at a time."""
        fmt = self._format
        assert fmt is not None
        edit = self._edit
        first = edit.index_at(min(seq_frame, edit.total_frames - 1))
        for i in range(first, len(edit)):
            seg = edit.segments[i]
            start = seg.start + (seq_frame - edit.seq_start(i) if i == first else 0)
            lo = int(round(start / fmt.fps * RATE))
            hi = int(round(seg.end / fmt.fps * RATE))
            pcm = self._pcm.get(seg.source)
            step = RATE  # hand over one second at a time
            for a in range(lo, hi, step):
                b = min(a + step, hi)
                chunk = pcm[a:b] if pcm is not None else None
                data = chunk.tobytes() if chunk is not None else b""
                missing = (b - a) * _BYTES_PER_FRAME - len(data)
                yield data + bytes(missing)  # silence for a short or missing track
