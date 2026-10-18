"""Crop region geometry (pure Python: no Qt, OpenCV or FFmpeg).

A crop is a rectangle in *source pixels*. Position and size are always even
numbers: H.264 in ``yuv420p`` needs even dimensions, and an even origin keeps
the chroma planes aligned. A crop may carry a locked aspect ratio (the 16:9 /
1:1 / 4:3 / 9:16 presets); with ``ratio=None`` it is "Free" and can be given any
shape. "Original" simply means no crop at all (``None`` in the edit state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Ratio = Tuple[int, int]

MIN_SIZE = 16  # smallest crop edge, in source pixels

# Preset name -> (width, height) ratio, in the order they appear in the UI.
RATIO_PRESETS: Dict[str, Ratio] = {
    "16:9": (16, 9),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "9:16": (9, 16),
}

# Handles are named by the sides they move: "tl" = top-left corner, "r" = right edge, ...
CORNER_HANDLES = ("tl", "tr", "bl", "br")
EDGE_HANDLES = ("t", "b", "l", "r")


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int
    ratio: Optional[Ratio] = None  # locked aspect ratio, or None for a free-form crop

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def preset_name(self) -> Optional[str]:
        """``"16:9"`` etc. for a preset ratio, None for a free crop."""
        for name, ratio in RATIO_PRESETS.items():
            if ratio == self.ratio:
                return name
        return None

    def covers(self, frame_w: int, frame_h: int) -> bool:
        """True if the crop is the whole frame, i.e. cropping would change nothing."""
        return (self.x, self.y, self.width, self.height) == (0, 0, frame_w, frame_h)

    def fits_in(self, frame_w: int, frame_h: int) -> bool:
        return (
            self.x >= 0
            and self.y >= 0
            and self.width > 0
            and self.height > 0
            and self.right <= frame_w
            and self.bottom <= frame_h
        )


def _even(value: float) -> int:
    """Round down to an even whole number (values here are never negative)."""
    whole = int(value)
    return whole - whole % 2


def _clamp_origin(value: float, size: int, limit: int) -> int:
    """Even position of a ``size``-long run that must stay inside ``[0, limit]``."""
    return _even(max(0, min(value, limit - size)))


def fit_crop(
    frame_w: int, frame_h: int, ratio: Ratio, center: Optional[Tuple[float, float]] = None
) -> Crop:
    """The largest crop with the given aspect ratio that fits the frame.

    It is centred on ``center`` (default: the middle of the frame), moved as far
    as needed to stay inside the frame.
    """
    rw, rh = ratio
    if frame_w * rh >= frame_h * rw:  # frame is wider than the ratio: height is the limit
        h = _even(frame_h)
        w = _even(round(h * rw / rh))
    else:  # frame is narrower than the ratio: width is the limit
        w = _even(frame_w)
        h = _even(round(w * rh / rw))
    w = max(2, min(w, _even(frame_w)))
    h = max(2, min(h, _even(frame_h)))
    cx, cy = center if center is not None else (frame_w / 2, frame_h / 2)
    return Crop(
        _clamp_origin(round(cx - w / 2), w, frame_w),
        _clamp_origin(round(cy - h / 2), h, frame_h),
        w,
        h,
        ratio,
    )


def default_free_crop(frame_w: int, frame_h: int) -> Crop:
    """Starting point for a free-form crop: the middle 80% of the frame."""
    w = max(2, _even(frame_w * 0.8))
    h = max(2, _even(frame_h * 0.8))
    return Crop(
        _clamp_origin((frame_w - w) / 2, w, frame_w),
        _clamp_origin((frame_h - h) / 2, h, frame_h),
        w,
        h,
        None,
    )


def move_crop(crop: Crop, dx: float, dy: float, frame_w: int, frame_h: int) -> Crop:
    """Shift the crop by ``(dx, dy)`` pixels, keeping it inside the frame."""
    return Crop(
        _clamp_origin(round(crop.x + dx), crop.width, frame_w),
        _clamp_origin(round(crop.y + dy), crop.height, frame_h),
        crop.width,
        crop.height,
        crop.ratio,
    )


def handle_point(crop: Crop, handle: str) -> Tuple[float, float]:
    """Frame position of a resize handle."""
    x = crop.x if "l" in handle else crop.right if "r" in handle else crop.x + crop.width / 2
    y = crop.y if "t" in handle else crop.bottom if "b" in handle else crop.y + crop.height / 2
    return x, y


def resize_crop(
    crop: Crop, handle: str, px: float, py: float, frame_w: int, frame_h: int
) -> Crop:
    """Drag ``handle`` of ``crop`` to the frame position ``(px, py)``.

    A free crop can be resized from any corner or edge. A crop with a locked ratio
    is resized from the corners only: the opposite corner stays put and the box
    keeps its shape. The result never leaves the frame or drops below MIN_SIZE.
    """
    px = min(max(px, 0.0), float(frame_w))
    py = min(max(py, 0.0), float(frame_h))
    left, top, right, bottom = crop.x, crop.y, crop.right, crop.bottom

    if crop.ratio is None:
        if "l" in handle:
            left = min(px, right - MIN_SIZE)
        if "r" in handle:
            right = max(px, left + MIN_SIZE)
        if "t" in handle:
            top = min(py, bottom - MIN_SIZE)
        if "b" in handle:
            bottom = max(py, top + MIN_SIZE)
        left, top, right, bottom = (_even(v) for v in (left, top, right, bottom))
        return Crop(left, top, right - left, bottom - top, None)

    if len(handle) != 2:
        return crop
    rw, rh = crop.ratio
    anchor_x = right if "l" in handle else left  # the corner opposite the dragged one
    anchor_y = bottom if "t" in handle else top
    max_w = anchor_x if "l" in handle else frame_w - anchor_x  # room towards the frame edge
    max_h = anchor_y if "t" in handle else frame_h - anchor_y

    w = max(abs(px - anchor_x), abs(py - anchor_y) * rw / rh)
    w = max(w, MIN_SIZE * max(1.0, rw / rh))
    w = min(w, max_w, max_h * rw / rh)
    w = _even(round(w))
    h = _even(round(w * rh / rw))
    while (w > max_w or h > max_h) and w > 2:  # rounding to even can overshoot by a pixel
        w -= 2
        h = _even(round(w * rh / rw))
    w, h = max(w, 2), max(h, 2)
    x = anchor_x - w if "l" in handle else anchor_x
    y = anchor_y - h if "t" in handle else anchor_y
    return Crop(x, y, w, h, crop.ratio)
