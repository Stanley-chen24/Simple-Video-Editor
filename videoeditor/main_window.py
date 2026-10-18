"""Main window: media pool, preview, transport controls, timeline and editing commands."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .crop import RATIO_PRESETS, Crop, default_free_crop, fit_crop
from .export_dialog import ExportDialog
from .history import History
from .media import VideoError, probe_video
from .models import (
    EditError,
    EditList,
    EditState,
    VideoInfo,
    format_timecode,
    same_frame_rate,
)
from .player import Player
from .timeline_widget import TimelineWidget, source_color
from .video_widget import VideoWidget

APP_NAME = "Simple Video Editor"
VIDEO_FILTER = "Video files (*.mp4 *.m4v *.mov *.mkv *.avi *.webm *.wmv);;All files (*.*)"

EMPTY_PROJECT_HINT = "Import videos to begin\nFile > Import Videos...  (Ctrl+O)  or drop files here"
EMPTY_TIMELINE_HINT = "The timeline is empty\nSelect a clip in the media pool and click Add to Timeline"

# (combo box text, key) in display order. Keys are "original", "free" or a RATIO_PRESETS name.
CROP_CHOICES = (
    ("Original (no crop)", "original"),
    ("16:9  Widescreen", "16:9"),
    ("1:1  Square", "1:1"),
    ("4:3  Standard", "4:3"),
    ("9:16  Vertical", "9:16"),
    ("Free / Custom", "free"),
)


def _fps_text(fps: float) -> str:
    return f"{fps:.2f}".rstrip("0").rstrip(".")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # The project: media pool + timeline + crop, all undoable.
        self._history: "History[EditState]" = History(EditState.initial())
        self._shown_edit: Optional[EditList] = None  # timeline currently given to player/widget
        self._shown_pool: Optional[Tuple[str, ...]] = None  # clip paths currently in the list widget
        self._mark_in: Optional[int] = None  # sequence frames
        self._mark_out: Optional[int] = None  # inclusive
        self._last_exported: Optional[EditState] = None
        self.player = Player(self)

        self.setWindowTitle(APP_NAME)
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self._create_actions()
        self._create_menus_and_toolbar()
        self._create_central_widget()
        self._connect_signals()
        self.statusBar().showMessage("Import videos to get started (Ctrl+O, or drop files here).")
        self._update_ui()

    # ================================================================ state access
    @property
    def _state(self) -> EditState:
        return self._history.current

    @property
    def _edit(self) -> EditList:
        """The timeline of the current state."""
        return self._state.timeline

    @property
    def _fps(self) -> float:
        return self._state.fps

    # ============================================================== construction
    def _action(
        self,
        text: str,
        slot: Callable[[], None],
        shortcut: Union[str, Sequence[str], None] = None,
        tip: str = "",
        icon: Optional[QIcon] = None,
    ) -> QAction:
        act = QAction(text, self)
        if icon is not None:
            act.setIcon(icon)
        keys = [shortcut] if isinstance(shortcut, str) else list(shortcut or [])
        if keys:
            act.setShortcuts([QKeySequence(k) for k in keys])
        label = tip or text.replace("&", "").replace("...", "")
        act.setToolTip(f"{label} ({keys[0]})" if keys else label)
        # Wrap the slot so the `checked` argument of triggered() is not passed on to it.
        act.triggered.connect(lambda _checked=False, s=slot: s())
        return act

    def _create_actions(self) -> None:
        p = self.player

        # File
        self.act_import = self._action("&Import Videos...", self.import_dialog, "Ctrl+O",
                                       "Import videos into the media pool and add them to the timeline")  # fmt: skip
        self.act_export = self._action("&Export Video...", self.export_video, "Ctrl+E",
                                       "Render the timeline to an .mp4 file")  # fmt: skip
        self.act_quit = self._action("&Quit", self.close, "Ctrl+Q")

        # Media pool
        self.act_add_clip = self._action("Add to Timeline", self.add_clip_to_timeline, None,
                                         "Place the selected pool clip at the end of the timeline")  # fmt: skip
        self.act_remove_clip = self._action("Remove", self.remove_clip, None,
                                            "Remove the selected clip from the project (and from the timeline)")  # fmt: skip

        # Edit
        self.act_undo = self._action("&Undo", self.undo, "Ctrl+Z", "Undo")
        self.act_redo = self._action("&Redo", self.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], "Redo")
        self.act_split = self._action("Split at Playhead", self.split_at_playhead, "S",
                                      "Cut the segment under the playhead in two")  # fmt: skip
        self.act_mark_in = self._action("Mark In", self.mark_in, "I",
                                        "Set the start of the marked range")  # fmt: skip
        self.act_mark_out = self._action("Mark Out", self.mark_out, "O",
                                         "Set the end of the marked range (this frame is included)")  # fmt: skip
        self.act_split_marks = self._action("Split at Marks", self.split_at_marks, "Shift+S",
                                            "Cut at Mark In and Mark Out and select what is between")  # fmt: skip
        self.act_clear_marks = self._action("Clear Marks", self.clear_marks, "X",
                                            "Remove Mark In / Mark Out")  # fmt: skip
        self.act_delete = self._action("Delete Selected", self.delete_selected,
                                       ["Delete", "Backspace"], "Delete the selected segments")  # fmt: skip
        self.act_move_earlier = self._action("Move Earlier", lambda: self._shift_selected(-1), "Alt+Left",
                                             "Move the selected segments one place earlier")  # fmt: skip
        self.act_move_later = self._action("Move Later", lambda: self._shift_selected(1), "Alt+Right",
                                           "Move the selected segments one place later")  # fmt: skip

        # Playback
        self._play_icon, self._pause_icon = icons.play_icon(), icons.pause_icon()
        self.act_play = self._action("Play", p.toggle, "Space", "Play / Pause", self._play_icon)
        self.act_first = self._action("Go to Start", lambda: p.seek(0), "Home",
                                      "Go to start", icons.first_icon())  # fmt: skip
        self.act_prev = self._action("Previous Frame", lambda: p.step(-1), "Left",
                                     "Previous frame", icons.prev_icon())  # fmt: skip
        self.act_next = self._action("Next Frame", lambda: p.step(1), "Right",
                                     "Next frame", icons.next_icon())  # fmt: skip
        self.act_last = self._action("Go to End", lambda: p.seek(p.total_frames - 1), "End",
                                     "Go to end", icons.last_icon())  # fmt: skip
        self.act_back_1s = self._action("Back 1 Second", lambda: self._jump_seconds(-1), "Shift+Left")
        self.act_fwd_1s = self._action("Forward 1 Second", lambda: self._jump_seconds(1), "Shift+Right")

        # Everything that needs at least one frame on the timeline.
        self._timeline_actions = (
            self.act_export, self.act_split, self.act_mark_in, self.act_mark_out,
            self.act_play, self.act_first, self.act_prev, self.act_next, self.act_last,
            self.act_back_1s, self.act_fwd_1s,
        )  # fmt: skip

    def _create_menus_and_toolbar(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addActions([self.act_import, self.act_export])
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit_menu = bar.addMenu("&Edit")
        edit_menu.addActions([self.act_undo, self.act_redo])
        edit_menu.addSeparator()
        edit_menu.addActions([self.act_split, self.act_mark_in, self.act_mark_out,
                              self.act_split_marks, self.act_clear_marks])  # fmt: skip
        edit_menu.addSeparator()
        edit_menu.addActions([self.act_delete, self.act_move_earlier, self.act_move_later])

        media_menu = bar.addMenu("&Media")
        media_menu.addActions([self.act_import, self.act_add_clip, self.act_remove_clip])

        play_menu = bar.addMenu("&Playback")
        play_menu.addActions([self.act_play, self.act_prev, self.act_next,
                              self.act_back_1s, self.act_fwd_1s, self.act_first, self.act_last])  # fmt: skip

        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        toolbar.addActions([self.act_import, self.act_export])
        toolbar.addSeparator()
        toolbar.addActions([self.act_undo, self.act_redo])
        self.addToolBar(toolbar)

    def _tool_button(self, action: QAction, icon_only: bool = False) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setFocusPolicy(Qt.NoFocus)  # keep Space / arrow keys for the shortcuts
        if icon_only:
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setIconSize(QSize(22, 22))
            button.setAutoRaise(True)
        else:
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        return button

    def _create_central_widget(self) -> None:
        self.video = VideoWidget()
        self.timeline = TimelineWidget()

        self.timecode = QLabel("00:00:00.000 / 00:00:00.000")
        mono = QFont("Consolas", 15)
        mono.setStyleHint(QFont.Monospace)
        self.timecode.setFont(mono)
        self.frame_label = QLabel("")
        self.frame_label.setMinimumWidth(130)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        transport = QHBoxLayout()
        for act in (self.act_first, self.act_prev, self.act_play, self.act_next, self.act_last):
            transport.addWidget(self._tool_button(act, icon_only=True))
        transport.addSpacing(16)
        transport.addWidget(self.timecode)
        transport.addStretch(1)
        transport.addWidget(self.frame_label)

        edit_row = QHBoxLayout()
        for act in (self.act_split, self.act_mark_in, self.act_mark_out,
                    self.act_split_marks, self.act_clear_marks):  # fmt: skip
            edit_row.addWidget(self._tool_button(act))
        edit_row.addSpacing(24)
        edit_row.addWidget(self._tool_button(self.act_delete))
        edit_row.addSpacing(24)
        edit_row.addWidget(self._tool_button(self.act_move_earlier))
        edit_row.addWidget(self._tool_button(self.act_move_later))
        edit_row.addStretch(1)

        hint = QLabel(
            "Click the timeline to seek  |  click a segment to select it (Ctrl / Shift for several)  |  "
            "drag a selected segment's top strip to reorder  |  Space play  |  S split  |  "
            "I / O marks  |  Del delete"
        )
        hint.setStyleSheet("color: #8a8a94;")
        hint.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addLayout(transport)
        left.addWidget(self.timeline)
        left.addLayout(edit_row)
        left.addWidget(hint)

        central = QWidget()
        root = QHBoxLayout(central)
        root.addLayout(left, 1)
        root.addWidget(self._create_side_panel())
        self.setCentralWidget(central)

    def _create_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(310)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._create_pool_box(), 1)
        layout.addWidget(self._create_crop_box())
        layout.addWidget(self._create_status_box())
        return panel

    def _create_pool_box(self) -> QGroupBox:
        box = QGroupBox("Media pool")
        layout = QVBoxLayout(box)
        self.pool_list = QListWidget()
        self.pool_list.setMinimumHeight(140)
        self.pool_list.setIconSize(QSize(14, 14))
        self.pool_list.setFocusPolicy(Qt.NoFocus)  # keep Space / arrow keys for the shortcuts
        self.pool_list.setToolTip("Imported clips. Double-click one to add it to the timeline again.")
        buttons = QHBoxLayout()
        for act in (self.act_import, self.act_add_clip, self.act_remove_clip):
            buttons.addWidget(self._tool_button(act))
        buttons.addStretch(1)
        self.project_label = QLabel("No clips imported yet")
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet("color: #8a8a94;")
        layout.addWidget(self.pool_list, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.project_label)
        return box

    def _create_crop_box(self) -> QGroupBox:
        box = QGroupBox("Crop")
        form = QFormLayout(box)
        self.crop_combo = QComboBox()
        for text, key in CROP_CHOICES:
            self.crop_combo.addItem(text, key)
        self.crop_combo.setFocusPolicy(Qt.NoFocus)  # keep Space / arrow keys for the shortcuts
        self.crop_combo.setToolTip("Choose an aspect ratio; 'Original' removes the crop")
        self.crop_region = QLabel("-")
        self.crop_region.setWordWrap(True)
        hint = QLabel("Drag the box on the preview to move it, or its handles to resize it.")
        hint.setStyleSheet("color: #8a8a94;")
        hint.setWordWrap(True)
        form.addRow("Aspect ratio:", self.crop_combo)
        form.addRow("Region:", self.crop_region)
        form.addRow(hint)
        return box

    def _create_status_box(self) -> QGroupBox:
        box = QGroupBox("Timeline status")
        form = QFormLayout(box)
        self._edit_labels = {}
        for key, caption in (("segments", "Segments"), ("selected", "Selected"),
                             ("length", "Output length"), ("mark_in", "Mark In"),
                             ("mark_out", "Mark Out")):  # fmt: skip
            label = QLabel("-")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(caption + ":", label)
            self._edit_labels[key] = label
        return box

    def _connect_signals(self) -> None:
        self.crop_combo.activated[int].connect(self._on_crop_choice)  # user picks only
        self.video.crop_committed.connect(self._on_crop_dragged)
        self.pool_list.currentItemChanged.connect(lambda *_: self._update_ui())
        self.pool_list.itemDoubleClicked.connect(lambda *_: self.add_clip_to_timeline())
        self.player.frame_ready.connect(self.video.set_frame)
        self.player.frame_cleared.connect(self.video.clear)
        self.player.position_changed.connect(self._on_position_changed)
        self.player.state_changed.connect(self._on_play_state_changed)
        self.player.read_failed.connect(self._on_read_failed)
        self.timeline.scrub_started.connect(self.player.pause)
        self.timeline.seek_requested.connect(self.player.seek)
        self.timeline.selection_changed.connect(self._update_ui)
        self.timeline.reorder_requested.connect(self._on_reorder)

    # ============================================================== media pool
    def import_dialog(self) -> None:
        pool = self._state.pool
        start = os.path.dirname(pool[-1].path) if pool else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Videos", start, VIDEO_FILTER)
        if paths:
            self.import_paths(paths)

    def import_paths(self, paths: Sequence[str]) -> int:
        """Probe ``paths`` and add the usable ones to the pool and the end of the timeline.

        Files that cannot be used are reported together in one message; the others are
        still imported. Returns how many were imported.
        """
        infos: List[VideoInfo] = []
        problems: List[str] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    infos.append(probe_video(path))
                except VideoError as exc:
                    problems.append(f"{path}\n    {exc}")
                except Exception as exc:  # e.g. a corrupt file that trips up the decoder
                    problems.append(f"{path}\n    Unexpected error: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

        if infos:
            self.player.pause()
            label = f"Import {infos[0].name}" if len(infos) == 1 else f"Import {len(infos)} videos"
            self._apply(self._state.add_clips(infos), label)
            fmt = self._state.project_format
            assert fmt is not None
            odd = [i for i in infos if (i.width, i.height) != (fmt.width, fmt.height)
                   or not same_frame_rate(i.fps, fmt.fps)]  # fmt: skip
            if odd:
                self.statusBar().showMessage(
                    f"{label}. {len(odd)} clip{'s differ' if len(odd) != 1 else ' differs'} from the project "
                    f"format ({fmt.width} x {fmt.height}, {_fps_text(fmt.fps)} fps) and will be fitted to it.",
                    8000,
                )
        if problems:
            QMessageBox.warning(
                self,
                "Some files could not be imported",
                "These files were skipped:\n\n" + "\n\n".join(problems),
            )
        return len(infos)

    def _selected_clip(self) -> Optional[str]:
        """Path of the clip selected in the media pool list."""
        item = self.pool_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def add_clip_to_timeline(self) -> None:
        path = self._selected_clip()
        if path is None:
            self.statusBar().showMessage("Select a clip in the media pool first.", 4000)
            return
        try:
            new_state = self._state.add_to_timeline(path)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._apply(new_state, f"Add {os.path.basename(path)} to the timeline")

    def remove_clip(self) -> None:
        path = self._selected_clip()
        if path is None:
            self.statusBar().showMessage("Select a clip in the media pool first.", 4000)
            return
        try:
            new_state = self._state.remove_clip(path)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self.player.pause()
        self._apply(new_state, f"Remove {os.path.basename(path)} from the project")

    def _on_read_failed(self, path: str) -> None:
        self.statusBar().showMessage(f"Cannot read {os.path.basename(path)} (moved or deleted?).", 8000)

    # ================================================================== editing
    def _apply(self, new_state: EditState, label: str, keep_marks: bool = False,
               select: Sequence[int] = ()) -> None:  # fmt: skip
        """Record ``new_state`` in the undo history and show it."""
        self._history.push(new_state, label)
        self._state_changed(keep_marks, select)
        self.statusBar().showMessage(label, 4000)

    def _apply_timeline(self, new_edit: EditList, label: str, keep_marks: bool = False,
                        select: Sequence[int] = ()) -> None:  # fmt: skip
        self._apply(self._state.with_timeline(new_edit), label, keep_marks, select)

    def _state_changed(self, keep_marks: bool = False, select: Sequence[int] = ()) -> None:
        """Propagate ``self._state`` (after an edit, undo or redo) to the player and widgets."""
        state, edit = self._state, self._edit
        self.timeline.set_sources(state.pool)
        self.timeline.set_fps(state.fps)
        self.player.set_project(state.pool, state.project_format, edit)
        if edit != self._shown_edit:  # a crop-only change leaves the timeline, selection and marks alone
            self._shown_edit = edit
            self.timeline.set_edit_list(edit)  # also drops the old selection
            if select:
                self.timeline.set_selection(select)
            if not keep_marks:
                self._mark_in = self._mark_out = None
                self.timeline.set_marks(None, None)
        self._update_ui()
        self._on_position_changed(self.player.position)

    # ---------------------------------------------------------------- cropping
    def _on_crop_choice(self, index: int) -> None:
        """The user picked an entry in the aspect-ratio box."""
        fmt = self._state.project_format
        if fmt is None or not self._edit.total_frames:
            return
        key = self.crop_combo.itemData(index)
        current = self._state.crop
        if key == "original":
            new_crop = None
        elif key == "free":
            # Keep the box where it is and just unlock its shape.
            new_crop = replace(current, ratio=None) if current else default_free_crop(fmt.width, fmt.height)
        else:
            # Largest box of that ratio, centred on where the previous box was.
            new_crop = fit_crop(fmt.width, fmt.height, RATIO_PRESETS[key],
                                current.center if current else None)  # fmt: skip
        label = "Remove crop" if new_crop is None else f"Crop: {self.crop_combo.itemText(index).split('  ')[0]}"
        self._apply(self._state.with_crop(new_crop), label)  # a no-op choice just re-syncs the combo

    def _on_crop_dragged(self, crop: Crop) -> None:
        """The user finished moving or resizing the crop box on the preview."""
        old = self._state.crop
        resized = old is None or (old.width, old.height) != (crop.width, crop.height)
        self._apply(self._state.with_crop(crop), "Resize crop" if resized else "Move crop")

    # ----------------------------------------------------------------- cutting
    def split_at_playhead(self) -> None:
        pos = self.player.position
        try:
            new_edit = self._edit.split_at(pos)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        stamp = format_timecode(pos / self._fps)
        self._apply_timeline(new_edit, f"Split at {stamp}", keep_marks=True)

    def mark_in(self) -> None:
        pos = self.player.position
        if self._mark_out is not None and pos > self._mark_out:
            self._mark_out = None  # In must not come after Out
        self._mark_in = pos
        self._marks_changed()

    def mark_out(self) -> None:
        pos = self.player.position
        if self._mark_in is not None and pos < self._mark_in:
            self._mark_in = None  # Out must not come before In
        self._mark_out = pos
        self._marks_changed()

    def clear_marks(self) -> None:
        self._mark_in = self._mark_out = None
        self._marks_changed()

    def _marks_changed(self) -> None:
        self.timeline.set_marks(self._mark_in, self._mark_out)
        self._update_ui()

    def split_at_marks(self) -> None:
        """Cut at Mark In and Mark Out and select the segments in between."""
        if self._mark_in is None or self._mark_out is None:
            self.statusBar().showMessage("Set both Mark In and Mark Out first.", 4000)
            return
        try:
            new_edit, inside = self._edit.split_range(self._mark_in, self._mark_out + 1)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._apply_timeline(new_edit, "Split at marks", select=inside)

    def delete_selected(self) -> None:
        indices = self.timeline.selected_indices()
        if not indices:
            self.statusBar().showMessage("Select one or more segments first.", 4000)
            return
        try:
            new_edit = self._edit.delete(indices)
        except EditError as exc:
            QMessageBox.warning(self, "Cannot delete", str(exc))
            return
        count = len(indices)
        self._apply_timeline(new_edit, f"Deleted {count} segment{'s' if count != 1 else ''}")

    # --------------------------------------------------------------- arranging
    def _shift_selected(self, delta: int) -> None:
        """Move the selected segments one place earlier (-1) or later (+1)."""
        indices = self.timeline.selected_indices()
        if not indices:
            self.statusBar().showMessage("Select one or more segments first.", 4000)
            return
        try:
            new_edit, moved = self._edit.shift(indices, delta)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._apply_timeline(new_edit, "Move earlier" if delta < 0 else "Move later", select=moved)

    def _on_reorder(self, indices: List[int], gap: int) -> None:
        """The user dragged the selected segments to the boundary before segment ``gap``."""
        try:
            new_edit, moved = self._edit.move_to(indices, gap)
        except EditError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._apply_timeline(new_edit, "Reorder segments", select=moved)

    # -------------------------------------------------------------- undo / redo
    def undo(self) -> None:
        if self._history.can_undo:
            label = self._history.undo()
            self._state_changed()
            self.statusBar().showMessage(f"Undid: {label}", 4000)

    def redo(self) -> None:
        if self._history.can_redo:
            label = self._history.redo()
            self._state_changed()
            self.statusBar().showMessage(f"Redid: {label}", 4000)

    def _jump_seconds(self, seconds: int) -> None:
        if self._edit.total_frames:
            self.player.step(int(round(seconds * self._fps)))

    # ================================================================== export
    def export_video(self) -> None:
        if not self._edit.total_frames:
            self.statusBar().showMessage("There is nothing to export yet: the timeline is empty.", 4000)
            return
        self.player.pause()
        state = self._state
        dialog = ExportDialog(state, parent=self)
        dialog.exported.connect(lambda path, s=state: self._on_exported(path, s))
        dialog.exec_()

    def _on_exported(self, path: str, state: EditState) -> None:
        self._last_exported = state
        self.statusBar().showMessage(f"Exported to {path}", 8000)

    # ============================================================== UI updating
    def _on_position_changed(self, seq_frame: int) -> None:
        self.timeline.set_position(seq_frame)
        fps, total = self._fps, self._edit.total_frames
        if not total:
            self.timecode.setText("00:00:00.000 / 00:00:00.000")
            self.frame_label.setText("")
            return
        self.timecode.setText(f"{format_timecode(seq_frame / fps)} / {format_timecode(total / fps)}")
        self.frame_label.setText(f"Frame {seq_frame + 1} / {total}")

    def _on_play_state_changed(self, playing: bool) -> None:
        self.act_play.setIcon(self._pause_icon if playing else self._play_icon)
        self.act_play.setText("Pause" if playing else "Play")

    def _update_ui(self) -> None:
        state = self._state
        has_frames = self._edit.total_frames > 0
        self._update_pool_list()
        selected = self.timeline.selected_indices()
        has_marks = self._mark_in is not None or self._mark_out is not None
        for act in self._timeline_actions:
            act.setEnabled(has_frames)
        clip_selected = self._selected_clip() is not None
        self.act_add_clip.setEnabled(clip_selected)
        self.act_remove_clip.setEnabled(clip_selected)
        self.act_undo.setEnabled(self._history.can_undo)
        self.act_redo.setEnabled(self._history.can_redo)
        self.act_clear_marks.setEnabled(has_marks)
        self.act_split_marks.setEnabled(has_frames and self._mark_in is not None and self._mark_out is not None)
        self.act_delete.setEnabled(bool(selected))
        self.act_move_earlier.setEnabled(bool(selected) and len(self._edit) > 1)
        self.act_move_later.setEnabled(bool(selected) and len(self._edit) > 1)
        self.act_undo.setToolTip(f"Undo: {self._history.undo_label or 'nothing to undo'} (Ctrl+Z)")
        self.act_redo.setToolTip(f"Redo: {self._history.redo_label or 'nothing to redo'} (Ctrl+Y)")
        self.video.set_placeholder(EMPTY_TIMELINE_HINT if state.pool else EMPTY_PROJECT_HINT)
        self._update_title()
        self._update_crop_ui()
        self._update_status_panel(selected)

    def _update_title(self) -> None:
        pool = self._state.pool
        if not pool:
            self.setWindowTitle(APP_NAME)
        elif len(pool) == 1:
            self.setWindowTitle(f"{pool[0].name} - {APP_NAME}")
        else:
            self.setWindowTitle(f"{len(pool)} clips - {APP_NAME}")

    def _update_pool_list(self) -> None:
        """Show the media pool in the list (rebuilt only when the set of clips changed)."""
        pool = self._state.pool
        paths = tuple(info.path for info in pool)
        fmt = self._state.project_format
        if fmt is None:
            self.project_label.setText("No clips imported yet")
        else:
            self.project_label.setText(
                f"Project format: {fmt.width} x {fmt.height} at {_fps_text(fmt.fps)} fps "
                "(set by the first clip; others are fitted to it)"
            )
        if paths == self._shown_pool:
            return
        self._shown_pool = paths
        keep = self._selected_clip()
        self.pool_list.blockSignals(True)
        self.pool_list.clear()
        for i, info in enumerate(pool):
            chip = QPixmap(14, 14)
            chip.fill(source_color(i))  # same colour as the clip's segments on the timeline
            audio = "audio" if info.has_audio else "no audio"
            item = QListWidgetItem(
                QIcon(chip),
                f"{info.name}\n{format_timecode(info.duration, show_ms=False)}  |  "
                f"{info.width}x{info.height}  |  {_fps_text(info.fps)} fps  |  {audio}",
            )
            item.setData(Qt.UserRole, info.path)
            item.setToolTip(f"{info.path}\n{info.frame_count} frames")
            self.pool_list.addItem(item)
            if info.path == keep:
                self.pool_list.setCurrentItem(item)
        if self.pool_list.currentItem() is None and pool:
            self.pool_list.setCurrentRow(self.pool_list.count() - 1)  # newest clip
        self.pool_list.blockSignals(False)

    def _update_crop_ui(self) -> None:
        """Make the preview overlay, the combo box and the region label show the current crop."""
        crop = self._state.crop
        self.video.set_crop(crop)
        self.crop_combo.setEnabled(self._edit.total_frames > 0)
        key = "original" if crop is None else (crop.preset_name or "free")
        self.crop_combo.blockSignals(True)  # only user picks may trigger an edit
        self.crop_combo.setCurrentIndex(max(0, self.crop_combo.findData(key)))
        self.crop_combo.blockSignals(False)
        self.crop_region.setText(
            "-" if crop is None else f"{crop.width} x {crop.height}  at ({crop.x}, {crop.y})"
        )

    def _update_status_panel(self, selected: List[int]) -> None:
        edit, fps = self._edit, self._fps
        has_frames = edit.total_frames > 0
        self._edit_labels["segments"].setText(str(len(edit)))
        self._edit_labels["selected"].setText(str(len(selected)))
        self._edit_labels["length"].setText(format_timecode(edit.total_frames / fps) if has_frames else "-")
        for key, mark in (("mark_in", self._mark_in), ("mark_out", self._mark_out)):
            self._edit_labels[key].setText("-" if mark is None else format_timecode(mark / fps))

    # ============================================================ window events
    def _has_unsaved_edits(self) -> bool:
        state = self._state
        return not state.is_pristine() and state != self._last_exported

    def _confirm_discard(self, action: str) -> bool:
        if not self._has_unsaved_edits():
            return True
        answer = QMessageBox.question(
            self, "Discard edits?",
            f"Your edits have not been exported.\n\n{action} anyway and discard them?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )  # fmt: skip
        return answer == QMessageBox.Yes

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not self._confirm_discard("Quit"):
            event.ignore()
            return
        self.player.close()
        event.accept()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.import_paths(paths)
