"""The "Export Video" dialog: output file, resolution, audio mute and rendering progress."""
from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .exporter import (
    ExportCancelled,
    ExportError,
    Exporter,
    ExportSettings,
    effective_crop,
    even,
    picture_size,
    same_file,
)
from .media import find_ffmpeg
from .models import EditState, format_timecode


class _ExportThread(QThread):
    """Runs an :class:`Exporter` off the GUI thread and reports back through signals."""

    progress = pyqtSignal(float)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, exporter: Exporter, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._exporter = exporter

    def cancel(self) -> None:
        self._exporter.cancel()

    def run(self) -> None:
        try:
            path = self._exporter.run(self.progress.emit)
        except ExportCancelled:
            self.cancelled.emit()
        except ExportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # never let a worker thread die silently
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.succeeded.emit(path)


class ExportDialog(QDialog):
    exported = pyqtSignal(str)  # path of a successfully written file

    def __init__(
        self,
        state: EditState,
        ffmpeg: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        fmt = state.project_format
        if fmt is None or not state.timeline.segments:
            raise ValueError("nothing to export")  # the main window never opens the dialog then
        self.setWindowTitle("Export Video")
        self.setMinimumWidth(520)
        self._state = state
        self._format = fmt
        self._sources = [state.clip(path) for path in state.timeline.sources()]
        self._ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        # The picture after cropping is the starting point for the output resolution.
        self._base_w, self._base_h = picture_size(fmt, state.crop)
        self._aspect = self._base_w / self._base_h
        self._has_audio = any(clip.has_audio for clip in self._sources)
        self._thread: Optional[_ExportThread] = None

        self._build_ui()
        first = self._sources[0]
        stem = os.path.splitext(first.name)[0]
        suffix = "combined" if len(self._sources) > 1 else "edited"
        self.path_edit.setText(os.path.join(os.path.dirname(first.path), f"{stem}_{suffix}.mp4"))

    # --------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        state, fmt = self._state, self._format
        layout = QVBoxLayout(self)

        duration = format_timecode(state.timeline.total_frames / fmt.fps)
        crop = effective_crop(fmt, state.crop)
        picture = (
            f"cropped to {crop.width} x {crop.height} (from {fmt.width} x {fmt.height})"
            if crop
            else f"project {fmt.width} x {fmt.height}"
        )
        clips = len(self._sources)
        summary = QLabel(
            f"{clips} clip{'s' if clips != 1 else ''}, {len(state.timeline.merged_ranges())} part(s) "
            f"joined  |  output length {duration}  |  {picture}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form = QFormLayout()
        layout.addLayout(form)

        # Output file
        self.path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_btn)
        form.addRow("Output file (.mp4):", row)

        # Resolution
        self.width_spin = self._make_spin(self._base_w)
        self.height_spin = self._make_spin(self._base_h)
        self.lock_check = QCheckBox("Lock aspect ratio")
        self.lock_check.setChecked(True)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Restore the native size of the (cropped) picture")
        self.reset_btn.clicked.connect(self._reset_size)
        size_row = QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height_spin)
        size_row.addWidget(self.lock_check)
        size_row.addWidget(self.reset_btn)
        size_row.addStretch(1)
        form.addRow("Resolution (W x H):", size_row)
        self.width_spin.valueChanged.connect(self._width_changed)
        self.height_spin.valueChanged.connect(self._height_changed)

        # Audio
        self.mute_check = QCheckBox("Mute Audio on Export")
        if not self._has_audio:
            self.mute_check.setChecked(True)
            self.mute_check.setEnabled(False)
            self.mute_check.setText("Mute Audio on Export (no audio track in the clips used)")
        form.addRow("Audio:", self.mute_check)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.export_btn = QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self.start_export)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._close_clicked)
        buttons.addWidget(self.export_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

        self._inputs = (
            self.path_edit, self.browse_btn, self.width_spin, self.height_spin,
            self.lock_check, self.reset_btn, self.mute_check,
        )  # fmt: skip

    @staticmethod
    def _make_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(2, 16384)
        spin.setSingleStep(2)
        spin.setKeyboardTracking(False)  # only react once typing is finished
        spin.setValue(value)
        return spin

    # ------------------------------------------------------------- resolution
    def _set_silently(self, spin: QSpinBox, value: int) -> None:
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _width_changed(self, value: int) -> None:
        width = even(value)
        self._set_silently(self.width_spin, width)
        if self.lock_check.isChecked():
            self._set_silently(self.height_spin, even(round(width / self._aspect)))
        else:
            self._set_silently(self.height_spin, even(self.height_spin.value()))

    def _height_changed(self, value: int) -> None:
        height = even(value)
        self._set_silently(self.height_spin, height)
        if self.lock_check.isChecked():
            self._set_silently(self.width_spin, even(round(height * self._aspect)))
        else:
            self._set_silently(self.width_spin, even(self.width_spin.value()))

    def _reset_size(self) -> None:
        self._set_silently(self.width_spin, self._base_w)
        self._set_silently(self.height_spin, self._base_h)

    # ------------------------------------------------------------ output file
    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export video as",
            self.path_edit.text(),
            "MP4 video (*.mp4)",
            options=QFileDialog.DontConfirmOverwrite,  # we ask once, in start_export()
        )
        if path:
            self.path_edit.setText(path)

    def _collect_settings(self) -> Optional[ExportSettings]:
        """Validate the form. Shows a message and returns None if something is wrong."""
        if not self._ffmpeg:
            self._warn(
                "FFmpeg was not found.\n\nInstall it with  pip install imageio-ffmpeg  "
                "or add ffmpeg to your PATH, then try again."
            )
            return None
        path = self.path_edit.text().strip()
        if not path:
            self._warn("Please choose an output file.")
            return None
        root, ext = os.path.splitext(path)
        if ext.lower() != ".mp4":
            path = (root if ext else path) + ".mp4"  # the output is always an .mp4
            self.path_edit.setText(path)
        path = os.path.abspath(path)
        if not os.path.isdir(os.path.dirname(path)):
            self._warn(f"The output folder does not exist:\n{os.path.dirname(path)}")
            return None
        if any(same_file(path, clip.path) for clip in self._sources):
            self._warn("The output file must be different from the source videos.")
            return None
        if os.path.exists(path):
            answer = QMessageBox.question(
                self, "Overwrite file?", f"{path}\n\nalready exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )  # fmt: skip
            if answer != QMessageBox.Yes:
                return None
        return ExportSettings(
            output_path=path,
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            mute_audio=self.mute_check.isChecked(),
        )

    # -------------------------------------------------------------- exporting
    def start_export(self) -> None:
        settings = self._collect_settings()
        if settings is None:
            return
        exporter = Exporter(self._ffmpeg, self._state, settings)
        self._thread = _ExportThread(exporter, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.succeeded.connect(self._on_succeeded)
        self._thread.failed.connect(self._on_failed)
        self._thread.cancelled.connect(self._on_cancelled)
        self._set_running(True)
        self.progress.setValue(0)
        self.status.setText("Rendering...")
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _set_running(self, running: bool) -> None:
        for widget in self._inputs:
            widget.setEnabled(not running)
        if not running and not self._has_audio:
            self.mute_check.setEnabled(False)  # nothing to choose without an audio track
        self.export_btn.setEnabled(not running)
        self.close_btn.setText("Cancel" if running else "Close")

    def _on_progress(self, fraction: float) -> None:
        percent = int(fraction * 100)
        self.progress.setValue(percent)
        self.status.setText(f"Rendering... {percent}%")

    def _on_succeeded(self, path: str) -> None:
        self._finish()
        self.progress.setValue(100)
        self.status.setText(f"Export complete:\n{path}")
        self.exported.emit(path)

    def _on_failed(self, message: str) -> None:
        self._finish()
        self.progress.setValue(0)
        self.status.setText("Export failed.")
        QMessageBox.critical(self, "Export failed", message)

    def _on_cancelled(self) -> None:
        self._finish()
        self.progress.setValue(0)
        self.status.setText("Export cancelled.")

    def _finish(self) -> None:
        if self._thread is not None:
            self._thread.wait()
        self._set_running(False)

    # ------------------------------------------------------ closing / cancel
    def _close_clicked(self) -> None:
        if self.is_running():
            self._thread.cancel()  # the thread reports back through cancelled()
        else:
            self.reject()

    def reject(self) -> None:  # Esc and the window's close button end up here
        if self.is_running():
            answer = QMessageBox.question(
                self, "Cancel export?", "An export is in progress. Stop it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )  # fmt: skip
            if answer != QMessageBox.Yes:
                return
            self._thread.cancel()
            self._thread.wait(10000)
        super().reject()

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "Export Video", message)
