"""Widget that shows the current preview frame, with an editable crop overlay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .crop import (
    CORNER_HANDLES,
    EDGE_HANDLES,
    Crop,
    handle_point,
    move_crop,
    resize_crop,
)

_MARGIN = 8  # empty border around the picture, in widget pixels
_HANDLE_SIZE = 9  # edge length of a drawn handle, in widget pixels
_HIT_RADIUS = 10  # how close the pointer must be to grab a handle
_ACCENT = QColor(255, 196, 61)

_CURSORS = {
    "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
    "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
    "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
    "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
    "move": Qt.SizeAllCursor,
}  # fmt: skip


@dataclass
class _Drag:
    """State of a crop drag in progress."""

    mode: str  # "move" or the name of the grabbed handle
    start_crop: Crop
    start_x: float  # pointer position in frame pixels when the button went down
    start_y: float
    offset_x: float  # grabbed handle minus pointer, so the box does not jump on the first move
    offset_y: float


class VideoWidget(QWidget):
    """Draws a BGR numpy frame, letterboxed and scaled to fit.

    When a crop is set, the area outside it is dimmed and the box can be dragged to
    move it, or by its handles to resize it. The widget only reports the finished
    drag (``crop_committed``); the owner decides what to do with it, e.g. record
    it in the undo history, and hands the result back through :meth:`set_crop`.
    """

    crop_committed = pyqtSignal(object)  # the Crop the user ended up with

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._placeholder = "Import videos to begin\nFile > Import Videos...  (Ctrl+O)  or drop files here"
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)  # hover cursors over the crop handles
        self._frame: Optional[np.ndarray] = None
        self._image: Optional[QImage] = None
        self._pixels: Optional[np.ndarray] = None  # keeps the QImage's buffer alive
        self._crop: Optional[Crop] = None
        self._drag: Optional[_Drag] = None
        self._drag_crop: Optional[Crop] = None  # the crop as it looks while being dragged

    # ------------------------------------------------------------------ frame
    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._rebuild_image()
        self.update()

    def clear(self) -> None:
        """Show the placeholder instead of a picture."""
        self._frame = self._image = self._pixels = None
        self._drag = self._drag_crop = None
        self.unsetCursor()
        self.update()

    def set_placeholder(self, text: str) -> None:
        """Text shown while there is no picture."""
        if text != self._placeholder:
            self._placeholder = text
            self.update()

    def _rebuild_image(self) -> None:
        """Scale to the widget size here (cheap, in numpy) rather than in the paint event."""
        if self._frame is None:
            return
        height, width = self._frame.shape[:2]
        ratio = self.devicePixelRatioF()
        # Leave a margin so crop handles sitting on the edge of the picture are not clipped.
        room_w, room_h = self.width() - 2 * _MARGIN, self.height() - 2 * _MARGIN
        scale = min(room_w * ratio / width, room_h * ratio / height)
        new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(self._frame, (new_w, new_h), interpolation=interpolation)
        self._pixels = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self._image = QImage(
            self._pixels.data, new_w, new_h, self._pixels.strides[0], QImage.Format_RGB888
        )
        self._image.setDevicePixelRatio(ratio)

    def _image_rect(self) -> Optional[QRectF]:
        """Where the frame is drawn, in widget coordinates (logical pixels)."""
        if self._image is None:
            return None
        ratio = self._image.devicePixelRatio()
        w, h = self._image.width() / ratio, self._image.height() / ratio
        return QRectF(int((self.width() - w) / 2), int((self.height() - h) / 2), w, h)

    def _frame_size(self) -> Optional[Tuple[int, int]]:
        if self._frame is None:
            return None
        return self._frame.shape[1], self._frame.shape[0]

    # ------------------------------------------------------------------- crop
    def set_crop(self, crop: Optional[Crop]) -> None:
        """Show ``crop`` (None: no overlay). Does nothing if it is already shown."""
        if crop == self._crop:
            return
        self._crop = crop
        self._drag = self._drag_crop = None
        self._update_cursor(None)
        self.update()

    def _to_frame(self, pos) -> Optional[Tuple[float, float]]:
        """Widget position -> frame pixel position (not clamped)."""
        rect, size = self._image_rect(), self._frame_size()
        if rect is None or size is None:
            return None
        return (pos.x() - rect.x()) / rect.width() * size[0], (pos.y() - rect.y()) / rect.height() * size[1]

    def _crop_rect(self, crop: Crop) -> QRectF:
        """A crop's rectangle in widget coordinates."""
        rect, size = self._image_rect(), self._frame_size()
        assert rect is not None and size is not None
        sx, sy = rect.width() / size[0], rect.height() / size[1]
        return QRectF(rect.x() + crop.x * sx, rect.y() + crop.y * sy, crop.width * sx, crop.height * sy)

    def _handle_positions(self, crop: Crop) -> Dict[str, QPointF]:
        """Handles in widget coordinates. A locked ratio only has corner handles."""
        rect = self._crop_rect(crop)
        names = CORNER_HANDLES if crop.ratio is not None else CORNER_HANDLES + EDGE_HANDLES
        points = {}
        for name in names:
            x = rect.left() if "l" in name else rect.right() if "r" in name else rect.center().x()
            y = rect.top() if "t" in name else rect.bottom() if "b" in name else rect.center().y()
            points[name] = QPointF(x, y)
        return points

    def _hit_test(self, pos) -> Optional[str]:
        """``"move"``, a handle name, or None if the pointer is not on the crop box."""
        if self._crop is None or self._image is None:
            return None
        for name, point in self._handle_positions(self._crop).items():
            if abs(point.x() - pos.x()) <= _HIT_RADIUS and abs(point.y() - pos.y()) <= _HIT_RADIUS:
                return name
        return "move" if self._crop_rect(self._crop).contains(QPointF(pos)) else None

    def _update_cursor(self, pos) -> None:
        mode = self._hit_test(pos) if pos is not None else None
        if mode is None:
            self.unsetCursor()
        else:
            self.setCursor(_CURSORS[mode])

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or self._crop is None:
            return
        mode, pointer = self._hit_test(event.pos()), self._to_frame(event.pos())
        if mode is None or pointer is None:
            return
        if mode == "move":
            offset = (0.0, 0.0)
        else:
            hx, hy = handle_point(self._crop, mode)
            offset = (hx - pointer[0], hy - pointer[1])
        self._drag = _Drag(mode, self._crop, pointer[0], pointer[1], offset[0], offset[1])
        self._drag_crop = self._crop

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        drag, size = self._drag, self._frame_size()
        pointer = self._to_frame(event.pos())
        if drag is None or size is None or pointer is None:
            self._update_cursor(event.pos())
            return
        if drag.mode == "move":
            crop = move_crop(
                drag.start_crop, pointer[0] - drag.start_x, pointer[1] - drag.start_y, *size
            )
        else:
            crop = resize_crop(
                drag.start_crop, drag.mode, pointer[0] + drag.offset_x, pointer[1] + drag.offset_y, *size
            )
        if crop != self._drag_crop:
            self._drag_crop = crop
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or self._drag is None:
            return
        start, final = self._drag.start_crop, self._drag_crop
        self._drag = self._drag_crop = None
        if final is not None and final != start:
            self._crop = final  # the owner will normally confirm this through set_crop()
            self.crop_committed.emit(final)
        self._update_cursor(event.pos())
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._drag is None:
            self.unsetCursor()

    # --------------------------------------------------------------- painting
    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._rebuild_image()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 8, 10))
        rect = self._image_rect()
        if self._image is None or rect is None:
            painter.setPen(QColor(130, 130, 138))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        painter.drawImage(int(rect.x()), int(rect.y()), self._image)
        crop = self._drag_crop or self._crop
        if crop is not None:
            self._paint_crop(painter, rect, crop)

    def _paint_crop(self, painter: QPainter, image_rect: QRectF, crop: Crop) -> None:
        box = self._crop_rect(crop)
        outside = QPainterPath()  # the picture minus the crop box (odd-even fill leaves a hole)
        outside.addRect(image_rect)
        outside.addRect(box)
        painter.fillPath(outside, QColor(0, 0, 0, 150))

        painter.setPen(QPen(_ACCENT, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(box)

        painter.setPen(QPen(QColor(20, 20, 24), 1))
        painter.setBrush(_ACCENT)
        for point in self._handle_positions(crop).values():
            half = _HANDLE_SIZE / 2
            painter.drawRect(QRectF(point.x() - half, point.y() - half, _HANDLE_SIZE, _HANDLE_SIZE))

        label = f"{crop.width} x {crop.height}"
        text_width = painter.fontMetrics().horizontalAdvance(label)
        if box.width() > text_width + 40 and box.height() > 50:
            tag = QRectF(box.left() + 8, box.top() + 8, text_width + 12, 20)
            painter.setPen(Qt.NoPen)
            painter.fillRect(tag, QColor(0, 0, 0, 170))
            painter.setPen(QColor(240, 240, 244))
            painter.drawText(tag, Qt.AlignCenter, label)
