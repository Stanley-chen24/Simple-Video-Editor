"""Interactive timeline: time ruler, one block per segment, playhead and In/Out marks.

The timeline shows the *edited* sequence: segments of every clip in playing order, each
clip in its own colour. Deleting a segment closes the gap.

* Press or drag on the ruler, or on the body of a segment, to move the playhead.
* Pressing on a segment also selects it (Ctrl+click toggles, Shift+click selects a range).
* Drag the coloured header of a selected segment sideways to reorder it; a marker shows
  where it will land. (The widget only reports the request; the owner applies it.)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from PyQt5.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .models import EditList, VideoInfo

_TICK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)

_BACKGROUND = QColor(26, 26, 30)
_RULER_TEXT = QColor(150, 150, 158)
_RULER_TICK = QColor(90, 90, 98)
# One colour per clip in the media pool (cycled when there are more clips than colours).
_SOURCE_COLORS = (
    QColor(52, 112, 150),
    QColor(62, 135, 98),
    QColor(150, 92, 62),
    QColor(122, 82, 150),
    QColor(60, 130, 138),
    QColor(150, 122, 52),
    QColor(150, 72, 102),
    QColor(92, 102, 160),
)
_SELECTED_BODY = QColor(222, 160, 50)
_SELECTED_HEADER = QColor(240, 186, 90)
_SELECTED_EDGE = QColor(255, 214, 120)
_MARK = QColor(90, 170, 255)
_PLAYHEAD = QColor(255, 70, 70)
_DROP = QColor(255, 255, 255)

_DRAG_THRESHOLD = 6  # pixels the pointer must travel before a header press becomes a reorder


def source_color(pool_index: int) -> QColor:
    """Colour used for the clip at ``pool_index`` (also by the media pool list)."""
    return QColor(_SOURCE_COLORS[pool_index % len(_SOURCE_COLORS)])


def _tick_label(seconds: float, step: float) -> str:
    if step < 1:
        return f"{int(seconds // 60)}:{seconds % 60:04.1f}"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(int)  # sequence frame under the mouse
    scrub_started = pyqtSignal()  # the mouse went down on the timeline
    selection_changed = pyqtSignal()
    reorder_requested = pyqtSignal(list, int)  # (selected segment indices, gap to drop them at)

    RULER_HEIGHT = 24
    HEADER_HEIGHT = 18  # draggable strip at the top of every segment
    MARGIN = 12

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(124)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)  # hand cursor over draggable headers
        self._edit = EditList()
        self._fps = 30.0
        self._pos = 0
        self._mark_in: Optional[int] = None
        self._mark_out: Optional[int] = None  # inclusive
        self._names: Dict[str, Tuple[str, int]] = {}  # clip path -> (file name, pool index)
        self._selection: Set[int] = set()
        self._anchor: Optional[int] = None
        self._dragging = False
        self._press_x = 0
        self._reorder_ready = False  # the press landed on a selected header
        self._reordering = False
        self._drop_gap: Optional[int] = None

    # ------------------------------------------------------------------ state
    def set_fps(self, fps: float) -> None:
        self._fps = fps or 30.0
        self.update()

    def set_sources(self, pool: Sequence[VideoInfo]) -> None:
        """Tell the timeline which clips exist, for their colours and names."""
        self._names = {info.path: (info.name, i) for i, info in enumerate(pool)}
        self.update()

    def set_edit_list(self, edit: EditList) -> None:
        """Show a new timeline. Segment indices change, so the selection is dropped."""
        self._edit = edit
        had_selection = bool(self._selection)
        self._selection = set()
        self._anchor = None
        self._reordering = self._reorder_ready = False
        self.update()
        if had_selection:
            self.selection_changed.emit()

    def set_position(self, seq_frame: int) -> None:
        self._pos = seq_frame
        self.update()

    def set_marks(self, mark_in: Optional[int], mark_out: Optional[int]) -> None:
        self._mark_in, self._mark_out = mark_in, mark_out
        self.update()

    def set_selection(self, indices) -> None:
        self._selection = set(indices)
        self._anchor = min(self._selection) if self._selection else None
        self.update()
        self.selection_changed.emit()

    def selected_indices(self) -> List[int]:
        return sorted(self._selection)

    # --------------------------------------------------------------- geometry
    @property
    def _track_top(self) -> int:
        return self.RULER_HEIGHT + 4

    def _x(self, frame: float) -> float:
        usable = self.width() - 2 * self.MARGIN
        return self.MARGIN + frame / max(1, self._edit.total_frames) * usable

    def _frame_at(self, x: float) -> int:
        usable = max(1, self.width() - 2 * self.MARGIN)
        frame = int((x - self.MARGIN) / usable * self._edit.total_frames)
        return max(0, min(frame, self._edit.total_frames - 1))

    def _gap_x(self, gap: int) -> float:
        """x of the boundary before segment ``gap`` (``len(edit)`` = the end of the timeline)."""
        if gap >= len(self._edit):
            return self._x(self._edit.total_frames)
        return self._x(self._edit.seq_start(gap))

    def _nearest_gap(self, x: float) -> int:
        return min(range(len(self._edit) + 1), key=lambda g: abs(self._gap_x(g) - x))

    def _on_header(self, pos) -> bool:
        return self._track_top <= pos.y() < self._track_top + self.HEADER_HEIGHT

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or not self._edit.total_frames:
            return
        frame = self._frame_at(event.x())
        self._reorder_ready = False
        if event.y() >= self.RULER_HEIGHT:
            index = self._edit.index_at(frame)
            self._select_at(index, event.modifiers())
            self._reorder_ready = self._on_header(event.pos()) and index in self._selection
        self._dragging = True
        self._press_x = event.x()
        self.scrub_started.emit()
        self.seek_requested.emit(frame)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not self._edit.total_frames:
            return
        if not self._dragging:
            hover_header = self._on_header(event.pos()) and len(self._edit) > 1
            self.setCursor(Qt.OpenHandCursor if hover_header else Qt.ArrowCursor)
            return
        if self._reorder_ready and (self._reordering or abs(event.x() - self._press_x) >= _DRAG_THRESHOLD):
            self._reordering = True
            self._drop_gap = self._nearest_gap(event.x())
            self.setCursor(Qt.ClosedHandCursor)
            self.update()
            return
        self.seek_requested.emit(self._frame_at(event.x()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton:
            return
        reorder = self._reordering and self._drop_gap is not None
        indices, gap = self.selected_indices(), self._drop_gap
        self._dragging = self._reordering = self._reorder_ready = False
        self._drop_gap = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        if reorder and gap is not None:
            self.reorder_requested.emit(indices, gap)

    def _select_at(self, index: int, modifiers) -> None:
        if modifiers & Qt.ControlModifier:
            self._selection ^= {index}
            self._anchor = index
        elif modifiers & Qt.ShiftModifier and self._anchor is not None:
            low, high = sorted((self._anchor, index))
            self._selection = set(range(low, high + 1))
        else:
            self._selection = {index}
            self._anchor = index
        self.update()
        self.selection_changed.emit()

    # --------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self._edit.total_frames:
            painter.setPen(_RULER_TEXT)
            painter.drawText(self.rect(), Qt.AlignCenter, "The timeline is empty")
            return
        track_height = self.height() - self._track_top - 8
        self._paint_ruler(painter)
        self._paint_segments(painter, self._track_top, track_height)
        self._paint_marks(painter)
        self._paint_drop_marker(painter)
        self._paint_playhead(painter)

    def _paint_ruler(self, painter: QPainter) -> None:
        duration = self._edit.total_frames / self._fps
        usable = max(1, self.width() - 2 * self.MARGIN)
        step = next((s for s in _TICK_STEPS if s / duration * usable >= 90), _TICK_STEPS[-1])
        metrics = painter.fontMetrics()
        t = 0.0
        while t <= duration + 1e-9:
            x = int(self._x(t * self._fps))
            label = _tick_label(t, step)
            painter.setPen(_RULER_TICK)
            painter.drawLine(x, self.RULER_HEIGHT - 7, x, self.RULER_HEIGHT - 1)
            painter.setPen(_RULER_TEXT)
            text_width = metrics.horizontalAdvance(label)
            # Put the label right of its tick, or left of it when it would run off the edge.
            left = x + 4 if x + 4 + text_width < self.width() - 2 else x - 4 - text_width
            painter.drawText(left, self.RULER_HEIGHT - 9, label)
            t += step

    def _paint_segments(self, painter: QPainter, top: int, height: int) -> None:
        metrics = painter.fontMetrics()
        movable = len(self._edit) > 1
        for i, seg in enumerate(self._edit.segments):
            x0 = self._x(self._edit.seq_start(i))
            x1 = self._x(self._edit.seq_start(i) + seg.length)
            rect = QRectF(x0, top, max(1.0, x1 - x0 - 2), height)  # 2px gap makes each cut visible
            name, color_index = self._names.get(seg.source, ("?", 0))
            base = source_color(color_index)
            selected = i in self._selection
            body = _SELECTED_BODY if selected else (base.lighter(112) if i % 2 else base)
            header_color = _SELECTED_HEADER if selected else base.lighter(140)
            header = QRectF(rect.left(), rect.top(), rect.width(), min(self.HEADER_HEIGHT, rect.height()))
            painter.fillRect(rect, body)
            painter.fillRect(header, header_color)
            if selected:
                painter.setPen(QPen(_SELECTED_EDGE, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect.adjusted(1, 1, -1, -1))

            ink = QColor(24, 24, 28) if selected else QColor(240, 244, 248)
            painter.setPen(ink)
            text_left = 6.0
            if movable and rect.width() > 40:  # grip dots: this strip can be dragged
                for dx in (0, 4):
                    for dy in (5, 9, 13):
                        painter.fillRect(QRectF(rect.left() + 5 + dx, rect.top() + dy - 1, 2, 2), ink)
                text_left = 16.0
            label = f"{i + 1}. {name}  {seg.length / self._fps:.1f}s"
            room = int(rect.width() - text_left - 4)
            if room > 14:
                painter.drawText(
                    QRectF(rect.left() + text_left, rect.top(), room, header.height()),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    metrics.elidedText(label, Qt.ElideRight, room),
                )
            if self._reordering and selected:  # dim the segments being dragged
                painter.fillRect(rect, QColor(0, 0, 0, 110))

    def _paint_drop_marker(self, painter: QPainter) -> None:
        if not self._reordering or self._drop_gap is None:
            return
        x = int(self._gap_x(self._drop_gap))
        painter.setPen(QPen(_DROP, 3))
        painter.drawLine(x, self._track_top - 4, x, self.height() - 4)

    def _paint_marks(self, painter: QPainter) -> None:
        if self._mark_in is None and self._mark_out is None:
            return
        height = self.height()
        x_in = self._x(self._mark_in) if self._mark_in is not None else None
        # Mark Out is inclusive, so the marked range ends after that frame.
        x_out = self._x(self._mark_out + 1) if self._mark_out is not None else None
        if x_in is not None and x_out is not None:
            painter.fillRect(QRectF(x_in, 0, x_out - x_in, height), QColor(90, 170, 255, 55))
        painter.setPen(QPen(_MARK, 2))
        # Drawn as brackets: "[" for In and "]" for Out.
        for x, serif in ((x_in, 7), (x_out, -7)):
            if x is not None:
                x = int(x)
                painter.drawLine(x, 0, x, height)
                painter.drawLine(x, 1, x + serif, 1)
                painter.drawLine(x, height - 1, x + serif, height - 1)

    def _paint_playhead(self, painter: QPainter) -> None:
        x = int(self._x(self._pos))
        painter.setPen(QPen(_PLAYHEAD, 2))
        painter.drawLine(x, 0, x, self.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(_PLAYHEAD)
        painter.drawPolygon(QPolygon([QPoint(x - 6, 0), QPoint(x + 6, 0), QPoint(x, 9)]))
