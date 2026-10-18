"""Entry point:  python main.py [video.mp4 ...]"""
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from videoeditor.main_window import MainWindow
from videoeditor.theme import apply_dark_theme


def _install_excepthook() -> None:
    """PyQt5 aborts the whole process on an unhandled exception in a slot; show it instead."""

    def hook(exc_type, exc, tb) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(details)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Unexpected error", details)

    sys.excepthook = hook


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    _install_excepthook()

    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.import_paths(sys.argv[1:])
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
