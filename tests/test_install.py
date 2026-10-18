"""Checks that everything the program needs is installed."""
import unittest


class InstallTests(unittest.TestCase):
    def test_python_packages(self):
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from PyQt5 import QtCore, QtMultimedia, QtWidgets  # noqa: F401

    def test_ffmpeg_is_available(self):
        from videoeditor.media import find_ffmpeg

        self.assertIsNotNone(find_ffmpeg(), "找不到 FFmpeg：請確認已執行 pip install -r requirements.txt")


if __name__ == "__main__":
    unittest.main()
