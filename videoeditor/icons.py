"""Transport icons drawn with QPainter.

The icons that ship with Qt styles are dark glyphs meant for light themes and are
almost invisible on the dark UI, so we draw simple light ones (plus a dimmed
variant for disabled buttons).
"""
from typing import Callable, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF

_SIZE = 22  # logical pixels; drawn at 2x so the icons stay sharp on high-DPI screens
_NORMAL = QColor(228, 228, 232)
_DISABLED = QColor(110, 110, 118)

Point = Tuple[float, float]


def _icon(draw: Callable[[QPainter], None]) -> QIcon:
    icon = QIcon()
    for mode, color in ((QIcon.Normal, _NORMAL), (QIcon.Disabled, _DISABLED)):
        pixmap = QPixmap(_SIZE * 2, _SIZE * 2)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        draw(painter)
        painter.end()
        icon.addPixmap(pixmap, mode)
    return icon


def _arrow(p: QPainter, tip: float, base: float) -> None:
    """Triangle pointing left (tip < base) or right (tip > base)."""
    p.drawPolygon(QPolygonF([QPointF(tip, 11), QPointF(base, 4), QPointF(base, 18)]))


def _bar(p: QPainter, x: float) -> None:
    p.drawRect(QRectF(x, 4, 2.5, 14))


def play_icon() -> QIcon:
    return _icon(lambda p: _arrow(p, 19, 6))


def pause_icon() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRect(QRectF(5, 4, 4.5, 14))
        p.drawRect(QRectF(12.5, 4, 4.5, 14))

    return _icon(draw)


def first_icon() -> QIcon:  # |<<  go to start
    def draw(p: QPainter) -> None:
        _bar(p, 3)
        _arrow(p, 6.5, 12.5)
        _arrow(p, 12.5, 18.5)

    return _icon(draw)


def last_icon() -> QIcon:  # >>|  go to end
    def draw(p: QPainter) -> None:
        _bar(p, 16.5)
        _arrow(p, 15.5, 9.5)
        _arrow(p, 9.5, 3.5)

    return _icon(draw)


def prev_icon() -> QIcon:  # |<  previous frame
    def draw(p: QPainter) -> None:
        _bar(p, 5)
        _arrow(p, 8.5, 17.5)

    return _icon(draw)


def next_icon() -> QIcon:  # >|  next frame
    def draw(p: QPainter) -> None:
        _bar(p, 14.5)
        _arrow(p, 13.5, 4.5)

    return _icon(draw)
