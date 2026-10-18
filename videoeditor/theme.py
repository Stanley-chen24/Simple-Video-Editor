"""Dark Fusion theme, the usual look for a video editor."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    text = QColor(228, 228, 232)
    muted = QColor(125, 125, 132)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(38, 38, 43))
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, QColor(26, 26, 30))
    palette.setColor(QPalette.AlternateBase, QColor(38, 38, 43))
    palette.setColor(QPalette.ToolTipBase, QColor(26, 26, 30))
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, QColor(54, 54, 60))
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, QColor(255, 90, 90))
    palette.setColor(QPalette.Link, QColor(90, 170, 255))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, muted)
    app.setPalette(palette)
