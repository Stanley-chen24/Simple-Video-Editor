"""Tests for the crop geometry (pure Python)."""
import random
import unittest

from videoeditor.crop import (
    CORNER_HANDLES,
    EDGE_HANDLES,
    MIN_SIZE,
    RATIO_PRESETS,
    Crop,
    default_free_crop,
    fit_crop,
    handle_point,
    move_crop,
    resize_crop,
)


def box(c):
    return c.x, c.y, c.width, c.height


class CropModelTests(unittest.TestCase):
    def test_properties(self):
        c = Crop(10, 20, 100, 50, (2, 1))
        self.assertEqual((c.right, c.bottom, c.center), (110, 70, (60.0, 45.0)))
        self.assertTrue(Crop(0, 0, 1920, 1080).covers(1920, 1080))
        self.assertFalse(c.covers(1920, 1080))
        self.assertTrue(c.fits_in(110, 70))
        self.assertFalse(c.fits_in(109, 70))
        self.assertFalse(Crop(-2, 0, 10, 10).fits_in(100, 100))

    def test_preset_name(self):
        self.assertEqual(Crop(0, 0, 2, 2, (16, 9)).preset_name, "16:9")
        self.assertEqual(Crop(0, 0, 2, 2, (1, 1)).preset_name, "1:1")
        self.assertIsNone(Crop(0, 0, 2, 2, None).preset_name)


class FitCropTests(unittest.TestCase):
    def test_landscape_frame(self):
        self.assertEqual(box(fit_crop(1920, 1080, (16, 9))), (0, 0, 1920, 1080))
        self.assertEqual(box(fit_crop(1920, 1080, (1, 1))), (420, 0, 1080, 1080))
        self.assertEqual(box(fit_crop(1920, 1080, (4, 3))), (240, 0, 1440, 1080))
        self.assertEqual(box(fit_crop(1920, 1080, (9, 16))), (656, 0, 608, 1080))

    def test_portrait_frame(self):
        self.assertEqual(box(fit_crop(1080, 1920, (1, 1))), (0, 420, 1080, 1080))
        c = fit_crop(1080, 1920, (16, 9))
        self.assertEqual((c.x, c.width), (0, 1080))
        self.assertAlmostEqual(c.width / c.height, 16 / 9, delta=0.01)

    def test_keeps_ratio_on_the_crop(self):
        self.assertEqual(fit_crop(1920, 1080, (4, 3)).ratio, (4, 3))

    def test_centre_is_respected_and_clamped(self):
        self.assertEqual(fit_crop(1920, 1080, (1, 1), (960, 540)).x, 420)
        self.assertEqual(fit_crop(1920, 1080, (1, 1), (1800, 540)).x, 840)  # pushed back inside
        self.assertEqual(fit_crop(1920, 1080, (1, 1), (0, 0)).x, 0)

    def test_odd_frame_sizes_still_give_even_boxes(self):
        c = fit_crop(1919, 1079, (1, 1))
        self.assertTrue(all(v % 2 == 0 for v in box(c)))
        self.assertTrue(c.fits_in(1919, 1079))

    def test_default_free_crop(self):
        c = default_free_crop(1000, 500)
        self.assertEqual(box(c), (100, 50, 800, 400))
        self.assertIsNone(c.ratio)


class MoveTests(unittest.TestCase):
    def test_move_and_clamp(self):
        c = fit_crop(1920, 1080, (1, 1))  # 1080x1080 at x=420
        self.assertEqual(move_crop(c, 100, 0, 1920, 1080).x, 520)
        self.assertEqual(move_crop(c, 5000, 0, 1920, 1080).x, 840)
        self.assertEqual(move_crop(c, -5000, 0, 1920, 1080).x, 0)
        self.assertEqual(move_crop(c, 0, 300, 1920, 1080).y, 0)  # cannot leave a full-height box
        moved = move_crop(c, 101, 0, 1920, 1080)
        self.assertEqual(moved.x % 2, 0)
        self.assertEqual((moved.width, moved.height, moved.ratio), (1080, 1080, (1, 1)))


class ResizeTests(unittest.TestCase):
    FREE = Crop(100, 100, 200, 200, None)

    def test_free_corner_and_edges(self):
        self.assertEqual(box(resize_crop(self.FREE, "br", 400, 350, 1920, 1080)), (100, 100, 300, 250))
        self.assertEqual(box(resize_crop(self.FREE, "tl", 50, 60, 1920, 1080)), (50, 60, 250, 240))
        self.assertEqual(box(resize_crop(self.FREE, "l", 50, 999, 1920, 1080)), (50, 100, 250, 200))
        self.assertEqual(box(resize_crop(self.FREE, "b", 999, 500, 1920, 1080)), (100, 100, 200, 400))

    def test_free_resize_stops_at_the_frame_and_the_minimum(self):
        self.assertEqual(resize_crop(self.FREE, "br", 9999, 9999, 1920, 1080).right, 1920)
        self.assertEqual(resize_crop(self.FREE, "tl", -50, -50, 1920, 1080).x, 0)
        small = resize_crop(self.FREE, "br", 101, 101, 1920, 1080)
        self.assertEqual((small.width, small.height), (MIN_SIZE, MIN_SIZE))
        crossed = resize_crop(self.FREE, "r", 10, 0, 1920, 1080)  # dragged past the left edge
        self.assertEqual((crossed.x, crossed.width), (100, MIN_SIZE))

    def test_locked_corner_keeps_the_ratio_and_the_opposite_corner(self):
        start = Crop(100, 100, 320, 180, (16, 9))
        out = resize_crop(start, "br", 500, 200, 1920, 1080)
        self.assertEqual(box(out), (100, 100, 400, 224))  # 400 wide, 225 -> 224 (even)
        out = resize_crop(start, "tl", 20, 30, 1920, 1080)  # pulled up-left
        self.assertEqual((out.right, out.bottom), (start.right, start.bottom))
        self.assertAlmostEqual(out.width / out.height, 16 / 9, delta=0.03)
        self.assertEqual(out.ratio, (16, 9))

    def test_locked_resize_cannot_leave_the_frame(self):
        start = Crop(100, 100, 320, 180, (16, 9))
        out = resize_crop(start, "br", 9999, 9999, 1920, 1080)
        self.assertTrue(out.fits_in(1920, 1080))
        out = resize_crop(start, "tl", -999, -999, 1920, 1080)
        self.assertEqual(out.fits_in(1920, 1080), True)

    def test_locked_crop_ignores_edge_handles(self):
        start = Crop(100, 100, 320, 180, (16, 9))
        self.assertEqual(resize_crop(start, "r", 700, 100, 1920, 1080), start)

    def test_handle_point(self):
        c = Crop(10, 20, 100, 50)
        self.assertEqual(handle_point(c, "tl"), (10, 20))
        self.assertEqual(handle_point(c, "br"), (110, 70))
        self.assertEqual(handle_point(c, "t"), (60, 20))
        self.assertEqual(handle_point(c, "l"), (10, 45))


class InvariantFuzzTests(unittest.TestCase):
    """Whatever the user drags, the crop stays even, inside the frame and (if locked) in shape."""

    FRAMES = [(1920, 1080), (1080, 1920), (1919, 1079), (640, 480), (100, 100), (3840, 2160)]

    def check(self, c, fw, fh, locked_ratio):
        self.assertTrue(all(v % 2 == 0 for v in box(c)), box(c))
        self.assertTrue(c.fits_in(fw, fh), (box(c), fw, fh))
        self.assertGreaterEqual(min(c.width, c.height), 2)
        if locked_ratio and min(c.width, c.height) >= MIN_SIZE:
            rw, rh = locked_ratio
            self.assertLessEqual(abs(c.width * rh - c.height * rw), 2 * max(rw, rh), box(c))

    def test_random_edits(self):
        rng = random.Random(1234)
        for _ in range(4000):
            fw, fh = rng.choice(self.FRAMES)
            ratio = rng.choice([None, *RATIO_PRESETS.values()])
            crop = default_free_crop(fw, fh) if ratio is None else fit_crop(fw, fh, ratio)
            self.check(crop, fw, fh, ratio)
            for _ in range(5):
                if rng.random() < 0.3:
                    crop = move_crop(crop, rng.uniform(-3000, 3000), rng.uniform(-3000, 3000), fw, fh)
                else:
                    handles = CORNER_HANDLES if ratio else CORNER_HANDLES + EDGE_HANDLES
                    crop = resize_crop(
                        crop, rng.choice(handles),
                        rng.uniform(-500, fw + 500), rng.uniform(-500, fh + 500), fw, fh,
                    )  # fmt: skip
                self.check(crop, fw, fh, ratio)


if __name__ == "__main__":
    unittest.main()
