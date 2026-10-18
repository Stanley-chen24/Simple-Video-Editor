"""End-to-end tests: probe generated videos, edit them, export and inspect the result.

Needs FFmpeg and OpenCV; the whole module is skipped if either is missing.
The sample videos carry known pictures (frame numbers, quadrant brightness, flat colours)
so we can verify exactly which frames, which region and which clips ended up in the export.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from array import array
from math import sqrt
from unittest import mock

try:
    import cv2
    import numpy as np

    from videoeditor.crop import Crop, fit_crop
    from videoeditor.exporter import ExportCancelled, ExportError, Exporter, ExportSettings
    from videoeditor.media import VideoError, VideoReader, find_ffmpeg, fit_to_canvas, probe_video
    from videoeditor.models import EditList, EditState, Segment

    FFMPEG = find_ffmpeg()
except ImportError:  # pragma: no cover
    FFMPEG = None

FRAMES = 300  # 10 s at 30 fps


def project(*infos):
    """A project with ``infos`` imported, each placed whole on the timeline."""
    return EditState.initial().add_clips(list(infos))


def luma_code(frame):
    """Recover (frame_number mod 40) from a decoded sample frame.

    The sample's luma is 16 + 5 * (n mod 40), inside the limited video range [16, 235];
    decoding to full-range gray stretches that by 255/219.
    """
    gray = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    return int(round(gray * 219 / 255 / 5))


def luma_value(region):
    """Mean video luma (16..235) of a decoded BGR image."""
    return float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()) * 219 / 255 + 16


def make_sample(path):
    """320x240, 10 s at 30 fps: the luma of frame n is 16 + 5 * (n mod 40). With audio."""
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30,geq=lum='16+mod(N,40)*5':cb=128:cr=128",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "10", "-c:v", "libx264", "-crf", "12", "-g", "15", "-pix_fmt", "yuv420p",
            "-c:a", "aac", path,
        ],
        check=True,
    )  # fmt: skip


# Brightness of the four quadrants of the spatial sample (top-left, top-right, ...).
QUADRANTS = {"tl": 16, "tr": 116, "bl": 66, "br": 166}


def make_quadrant_sample(path):
    """320x240, 60 frames: each quadrant has its own flat brightness, plus an audio track."""
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            "color=c=black:s=320x240:r=30,geq=lum='16+if(gte(X,W/2),100,0)+if(gte(Y,H/2),50,0)':cb=128:cr=128",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "2", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", path,
        ],
        check=True,
    )  # fmt: skip


def make_flat_clip(path, size, fps, luma, audio_rate=None):
    """A 2 s clip of one flat brightness. ``audio_rate`` adds a mono sine track at that sample rate."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={size}:r={fps},geq=lum={luma}:cb=128:cr=128",
    ]  # fmt: skip
    if audio_rate:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={audio_rate}"]
    cmd += ["-t", "2", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    if audio_rate:
        cmd += ["-c:a", "aac"]
    subprocess.run(cmd + [path], check=True)


def decode_audio(path):
    """All audio of ``path`` as mono float samples at 48 kHz."""
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout  # fmt: skip
    samples = array("f")
    samples.frombytes(raw)
    return samples


def rms(samples, start_s, end_s):
    window = samples[int(start_s * 48000):int(end_s * 48000)]
    return sqrt(sum(x * x for x in window) / len(window))


@unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A non-ASCII folder name makes sure unicode paths work end to end.
        cls._tmp = tempfile.TemporaryDirectory(prefix="影片_")
        cls.dir = cls._tmp.name
        cls.src = os.path.join(cls.dir, "測試.mp4")
        make_sample(cls.src)
        cls.info = probe_video(cls.src)
        cls.state = project(cls.info)
        cls.whole = cls.state.timeline

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def export(self, edit, name="out.mp4", **kwargs):
        settings = ExportSettings(
            output_path=os.path.join(self.dir, name),
            width=kwargs.pop("width", self.info.width),
            height=kwargs.pop("height", self.info.height),
            mute_audio=kwargs.pop("mute", False),
        )
        progress = []
        out = Exporter(FFMPEG, self.state.with_timeline(edit), settings).run(progress.append)
        self.assertTrue(progress and progress[-1] == 1.0)
        return out

    def test_probe(self):
        self.assertEqual((self.info.width, self.info.height), (320, 240))
        self.assertAlmostEqual(self.info.fps, 30.0, places=2)
        self.assertEqual(self.info.frame_count, FRAMES)
        self.assertTrue(self.info.has_audio)

    def test_probe_rejects_bad_files(self):
        with self.assertRaises(VideoError):
            probe_video(os.path.join(self.dir, "missing.mp4"))
        bad = os.path.join(self.dir, "garbage.mp4")
        with open(bad, "wb") as handle:
            handle.write(b"this is not a video" * 100)
        with self.assertRaises(VideoError):
            probe_video(bad)

    def test_reader_is_frame_accurate(self):
        reader = VideoReader(self.src)
        try:
            for index in (0, 1, 2, 77, 76, 200, 201, 299):  # sequential, backward and far jumps
                self.assertEqual(luma_code(reader.read(index)), index % 40, f"frame {index}")
        finally:
            reader.close()

    def test_cut_and_join(self):
        # Keep source frames 0-59 and 150-209; drop the rest.
        edit = self.whole.split_at(60).split_at(150).split_at(210).delete([1, 3])
        out = self.export(edit, "cut.mp4")
        info = probe_video(out)
        self.assertEqual(info.frame_count, 120)
        self.assertTrue(info.has_audio)
        reader = VideoReader(out)
        try:
            for out_index, src_index in ((0, 0), (59, 59), (60, 150), (119, 209)):
                self.assertEqual(
                    luma_code(reader.read(out_index)), src_index % 40, f"output frame {out_index}"
                )
        finally:
            reader.close()

    def test_resize_and_mute(self):
        edit = self.whole.split_at(100).delete([0])
        out = self.export(edit, "small.mp4", width=160, height=120, mute=True)
        info = probe_video(out)
        self.assertEqual((info.width, info.height), (160, 120))
        self.assertEqual(info.frame_count, 200)
        self.assertFalse(info.has_audio)

    def test_many_segments(self):
        # 150 one-frame pieces with gaps, i.e. a very long filter graph.
        edit = self.whole
        for point in range(1, FRAMES):
            edit = edit.split_at(point)
        edit = edit.delete(range(1, FRAMES, 2))
        out = self.export(edit, "many.mp4")
        self.assertEqual(probe_video(out).frame_count, 150)

    def test_long_command_uses_a_graph_file(self):
        edit = self.whole.split_at(100).split_at(200).delete([1])
        with mock.patch("videoeditor.exporter.MAX_COMMAND_CHARS", 100):
            out = self.export(edit, "script.mp4")
        self.assertEqual(probe_video(out).frame_count, 200)

    def test_errors_and_cancel(self):
        def exporter(path, ffmpeg=FFMPEG, state=self.state):
            return Exporter(ffmpeg, state, ExportSettings(path, 320, 240))

        with self.assertRaises(ExportError):  # output over a source
            exporter(self.src).run()
        with self.assertRaises(ExportError):  # missing folder
            exporter(os.path.join(self.dir, "nope", "x.mp4")).run()
        with self.assertRaises(ExportError):  # no ffmpeg
            exporter(os.path.join(self.dir, "y.mp4"), ffmpeg=None).run()
        with self.assertRaises(ExportError):  # nothing on the timeline
            exporter(os.path.join(self.dir, "z.mp4"), state=self.state.with_timeline(EditList())).run()
        cancelled = exporter(os.path.join(self.dir, "c.mp4"))
        cancelled.cancel()
        with self.assertRaises(ExportCancelled):
            cancelled.run()
        self.assertFalse(os.path.exists(os.path.join(self.dir, "c.mp4")))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "c.part.mp4")))


@unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
class CropExportTests(unittest.TestCase):
    """Cropping must pick the right region and work together with cuts, resize and mute."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="裁切_")
        cls.dir = cls._tmp.name
        cls.src = os.path.join(cls.dir, "quadrants.mp4")
        make_quadrant_sample(cls.src)
        cls.info = probe_video(cls.src)
        cls.state = project(cls.info)
        cls.whole = cls.state.timeline

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def export(self, edit, crop, width, height, mute=False, name="out.mp4"):
        state = self.state.with_timeline(edit).with_crop(crop)
        settings = ExportSettings(os.path.join(self.dir, name), width, height, mute)
        return Exporter(FFMPEG, state, settings).run()

    def mean_luma(self, path):
        reader = VideoReader(path)
        try:
            return luma_value(reader.read(5))
        finally:
            reader.close()

    def test_each_quadrant(self):
        for name, (x, y) in {"tl": (0, 0), "tr": (160, 0), "bl": (0, 120), "br": (160, 120)}.items():
            with self.subTest(quadrant=name):
                out = self.export(self.whole, Crop(x, y, 160, 120), 160, 120, name=f"{name}.mp4")
                info = probe_video(out)
                self.assertEqual((info.width, info.height), (160, 120))
                self.assertEqual(info.frame_count, self.info.frame_count)
                self.assertAlmostEqual(self.mean_luma(out), QUADRANTS[name], delta=8)

    def test_preset_ratio_crop(self):
        crop = fit_crop(320, 240, (16, 9))  # 320x180, centred vertically: straddles both rows
        self.assertEqual((crop.width, crop.height, crop.y), (320, 180, 30))
        out = self.export(self.whole, crop, 320, 180)
        info = probe_video(out)
        self.assertEqual((info.width, info.height), (320, 180))

    def test_crop_with_cuts_resize_and_mute(self):
        # Keep frames 0-19 and 40-59, crop the left square, shrink it, drop the audio.
        edit = self.whole.split_at(20).split_at(40).delete([1])
        crop = fit_crop(320, 240, (1, 1), (80, 120))  # 240x240 at the left edge
        self.assertEqual((crop.x, crop.width, crop.height), (0, 240, 240))
        out = self.export(edit, crop, 120, 120, mute=True)
        info = probe_video(out)
        self.assertEqual((info.width, info.height), (120, 120))
        self.assertEqual(info.frame_count, 40)
        self.assertFalse(info.has_audio)

    def test_crop_keeps_audio_when_not_muted(self):
        out = self.export(self.whole, Crop(160, 0, 160, 240), 160, 240)
        info = probe_video(out)
        self.assertTrue(info.has_audio)
        self.assertEqual((info.width, info.height), (160, 240))

    def test_crop_outside_the_frame_is_rejected(self):
        with self.assertRaises(ExportError):
            self.export(self.whole, Crop(200, 0, 160, 120), 160, 120)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out.part.mp4")))


class FitToCanvasTests(unittest.TestCase):
    @unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
    def test_same_size_is_returned_untouched(self):
        frame = np.full((240, 320, 3), 7, np.uint8)
        self.assertIs(fit_to_canvas(frame, 320, 240), frame)

    @unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
    def test_wide_clip_gets_bars_above_and_below(self):
        out = fit_to_canvas(np.full((90, 160, 3), 200, np.uint8), 320, 240)  # fits as 320x180
        self.assertEqual(out.shape, (240, 320, 3))
        self.assertEqual(int(out[:29].max()), 0)
        self.assertEqual(int(out[211:].max()), 0)
        self.assertGreaterEqual(int(out[40:200].min()), 199)

    @unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
    def test_tall_clip_gets_bars_left_and_right(self):
        out = fit_to_canvas(np.full((160, 90, 3), 200, np.uint8), 320, 240)  # fits as 135x240
        self.assertEqual(out.shape, (240, 320, 3))
        self.assertEqual(int(out[:, :91].max()), 0)
        self.assertEqual(int(out[:, 229:].max()), 0)
        self.assertGreaterEqual(int(out[:, 100:220].min()), 199)


@unittest.skipUnless(FFMPEG, "FFmpeg / OpenCV not available")
class MultiClipExportTests(unittest.TestCase):
    """Clips with different sizes, frame rates and audio are joined into one video."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="多段_")
        d = cls.dir = cls._tmp.name
        # Every clip is 2 s long and has its own flat brightness.
        make_flat_clip(os.path.join(d, "a.mp4"), "320x240", 30, 66, audio_rate=44100)
        make_flat_clip(os.path.join(d, "b.mp4"), "160x90", 25, 166)  # smaller, 16:9, 25 fps, no sound
        make_flat_clip(os.path.join(d, "c.mp4"), "320x240", 24, 116, audio_rate=22050)
        make_flat_clip(os.path.join(d, "d.mp4"), "320x240", 30, 200)  # same format as a, no sound
        cls.a, cls.b, cls.c, cls.d = (probe_video(os.path.join(d, f"{n}.mp4")) for n in "abcd")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def export(self, state, name="out.mp4", size=(320, 240), mute=False):
        settings = ExportSettings(os.path.join(self.dir, name), size[0], size[1], mute)
        return Exporter(FFMPEG, state, settings).run()

    def luma_at(self, path, index, region=None):
        reader = VideoReader(path)
        try:
            frame = reader.read(index)
        finally:
            reader.close()
        return luma_value(frame if region is None else frame[region])

    def test_probe_of_the_clips(self):
        self.assertEqual((self.b.width, self.b.height, self.b.fps, self.b.frame_count), (160, 90, 25.0, 50))
        self.assertEqual((self.c.fps, self.c.frame_count), (24.0, 48))
        self.assertEqual([c.has_audio for c in (self.a, self.b, self.c, self.d)], [True, False, True, False])

    def test_three_different_clips_become_one_video(self):
        state = project(self.a, self.b, self.c)  # the project format is a's: 320x240 at 30 fps
        self.assertEqual(state.timeline.total_frames, 180)  # 60 + 60 + 60 project frames
        out = self.export(state)
        info = probe_video(out)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.frame_count, 180, delta=3)
        self.assertTrue(info.has_audio)

        # Picture: a, then b letterboxed (black bars above and below), then c.
        self.assertAlmostEqual(self.luma_at(out, 30), 66, delta=8)
        self.assertAlmostEqual(self.luma_at(out, 90, (slice(70, 170), slice(None))), 166, delta=8)
        self.assertAlmostEqual(self.luma_at(out, 90, (slice(0, 20), slice(None))), 16, delta=6)
        self.assertAlmostEqual(self.luma_at(out, 90, (slice(220, 240), slice(None))), 16, delta=6)
        self.assertAlmostEqual(self.luma_at(out, 150), 116, delta=8)

        # Sound: a and c are audible; b has no audio track, so its 2 s must be silent, not skipped.
        samples = decode_audio(out)
        self.assertAlmostEqual(len(samples) / 48000, 6.0, delta=0.2)
        self.assertGreater(rms(samples, 0.2, 1.8), 0.02)
        self.assertLess(rms(samples, 2.2, 3.8), 0.002)
        self.assertGreater(rms(samples, 4.2, 5.8), 0.02)

    def test_reordered_and_trimmed_segments(self):
        state = project(self.a, self.c)
        # c's project frames 10-39, then a's frames 0-19.
        edit = EditList((Segment(self.c.path, 10, 40), Segment(self.a.path, 0, 20)))
        out = self.export(state.with_timeline(edit))
        info = probe_video(out)
        self.assertAlmostEqual(info.frame_count, 50, delta=2)
        self.assertAlmostEqual(self.luma_at(out, 5), 116, delta=8)
        self.assertAlmostEqual(self.luma_at(out, 44), 66, delta=8)

    def test_same_format_clips_join_frame_exactly(self):
        out = self.export(project(self.a, self.d))
        info = probe_video(out)
        self.assertEqual(info.frame_count, 120)
        self.assertAlmostEqual(self.luma_at(out, 59), 66, delta=8)
        self.assertAlmostEqual(self.luma_at(out, 60), 200, delta=8)
        samples = decode_audio(out)  # d is silent, a is not
        self.assertGreater(rms(samples, 0.2, 1.8), 0.02)
        self.assertLess(rms(samples, 2.2, 3.8), 0.002)

    def test_one_clip_used_twice(self):
        state = project(self.a, self.d).add_to_timeline(self.a.path)
        out = self.export(state)
        self.assertEqual(probe_video(out).frame_count, 180)
        self.assertAlmostEqual(self.luma_at(out, 150), 66, delta=8)

    def test_mute_and_all_silent_clips(self):
        self.assertFalse(probe_video(self.export(project(self.a, self.c), "muted.mp4", mute=True)).has_audio)
        self.assertFalse(probe_video(self.export(project(self.b, self.d), "silent.mp4")).has_audio)

    def test_crop_and_resize_work_on_a_multi_clip_project(self):
        state = project(self.a, self.b).with_crop(fit_crop(320, 240, (1, 1)))  # 240x240 in project pixels
        out = self.export(state, size=(120, 120))
        info = probe_video(out)
        self.assertEqual((info.width, info.height), (120, 120))
        self.assertAlmostEqual(info.frame_count, 120, delta=3)
        self.assertTrue(info.has_audio)

    def test_a_missing_source_is_reported(self):
        gone = os.path.join(self.dir, "gone.mp4")
        shutil.copy(self.a.path, gone)
        info = probe_video(gone)
        os.remove(gone)
        with self.assertRaises(ExportError) as caught:
            self.export(project(self.d, info), "never.mp4")
        self.assertIn("missing", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "never.mp4")))

    def test_output_may_not_overwrite_any_clip(self):
        with self.assertRaises(ExportError):
            Exporter(FFMPEG, project(self.a, self.d), ExportSettings(self.d.path, 320, 240)).run()


if __name__ == "__main__":
    unittest.main()
