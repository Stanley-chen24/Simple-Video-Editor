"""Tests for the pure editing logic (no Qt, OpenCV or FFmpeg needed)."""
import unittest

from videoeditor.crop import Crop, fit_crop
from videoeditor.exporter import (
    ExportError,
    effective_crop,
    even,
    frame_rate_expression,
    picture_size,
    plan_render,
)
from videoeditor.history import History
from videoeditor.models import (
    EditError,
    EditList,
    EditState,
    ProjectFormat,
    Segment,
    VideoInfo,
    clip_frame_count,
    format_timecode,
    frame_rate_fraction,
    same_frame_rate,
    source_frame,
)

A, B, C, D = "a.mp4", "b.mp4", "c.mp4", "d.mp4"


def info(name, width=1920, height=1080, fps=30.0, frames=300, audio=True):
    return VideoInfo(path=name, width=width, height=height, fps=fps, frame_count=frames, has_audio=audio)


def whole(n, source=A):
    return EditList.whole(source, n)


def spans(edit):
    return [(s.source, s.start, s.end) for s in edit.segments]


def four():
    """Four 10-frame segments a, b, c, d (different clips, so they are told apart)."""
    return EditList(tuple(Segment(name, 0, 10) for name in (A, B, C, D)))


def order(edit):
    return [s.source for s in edit.segments]


class TimecodeTests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_timecode(0), "00:00:00.000")
        self.assertEqual(format_timecode(83.456), "00:01:23.456")
        self.assertEqual(format_timecode(3661, show_ms=False), "01:01:01")
        self.assertEqual(format_timecode(-5), "00:00:00.000")


class FrameRateTests(unittest.TestCase):
    def test_exact_fractions(self):
        self.assertEqual(frame_rate_fraction(29.97002997002997), frame_rate_fraction(30000 / 1001))
        self.assertEqual(frame_rate_expression(29.97002997002997), "30000/1001")
        self.assertEqual(frame_rate_expression(23.976023976), "24000/1001")
        self.assertEqual(frame_rate_expression(25.0), "25/1")

    def test_same_frame_rate(self):
        self.assertTrue(same_frame_rate(30.0, 30.0))
        self.assertTrue(same_frame_rate(30000 / 1001, 29.97002997002997))  # NTSC, however it is rounded
        self.assertFalse(same_frame_rate(30.0, 29.97002997))
        self.assertFalse(same_frame_rate(24.0, 30.0))

    def test_clip_frames_follow_the_project_rate(self):
        self.assertEqual(clip_frame_count(info(A, fps=30, frames=300), 30.0), 300)  # untouched
        self.assertEqual(clip_frame_count(info(A, fps=25, frames=100), 30.0), 120)  # 4 s either way
        self.assertEqual(clip_frame_count(info(A, fps=60, frames=600), 30.0), 300)
        self.assertEqual(clip_frame_count(info(A, fps=24, frames=10), 60.0), 25)
        self.assertEqual(clip_frame_count(info(A, fps=60, frames=1), 24.0), 1)  # never zero

    def test_source_frame_mapping(self):
        slow, fast = info(A, fps=24, frames=48), info(A, fps=60, frames=120)
        self.assertEqual([source_frame(slow, 30.0, v) for v in range(6)], [0, 0, 1, 2, 3, 4])  # repeats one
        self.assertEqual([source_frame(fast, 30.0, v) for v in range(4)], [0, 2, 4, 6])  # skips frames
        self.assertEqual(source_frame(slow, 30.0, 59), 47)  # never past the last frame
        self.assertEqual(source_frame(info(A, fps=30), 30.0, 123), 123)  # same rate: identity


class EditListTests(unittest.TestCase):
    def test_whole_and_empty(self):
        edit = whole(300)
        self.assertEqual((edit.total_frames, len(edit)), (300, 1))
        empty = EditList()
        self.assertEqual((empty.total_frames, len(empty)), (0, 0))
        with self.assertRaises(EditError):
            empty.split_at(0)
        with self.assertRaises(IndexError):
            empty.index_at(0)

    def test_split_and_locate(self):
        edit = whole(300).split_at(100)
        self.assertEqual(spans(edit), [(A, 0, 100), (A, 100, 300)])
        self.assertEqual(edit.locate(99), (0, 99))
        self.assertEqual(edit.locate(100), (1, 100))
        self.assertEqual(edit.seq_start(1), 100)

    def test_split_keeps_the_clip(self):
        edit = EditList((Segment(A, 0, 50), Segment(B, 10, 90))).split_at(70)
        self.assertEqual(spans(edit), [(A, 0, 50), (B, 10, 30), (B, 30, 90)])
        self.assertEqual(edit.locate(70), (2, 30))

    def test_split_rejects_boundaries_and_out_of_range(self):
        edit = whole(300).split_at(100)
        for bad in (0, 100, -1, 300):
            with self.assertRaises(EditError):
                edit.split_at(bad)

    def test_delete_closes_the_gap(self):
        edit = whole(300).split_at(100).split_at(200).delete([1])
        self.assertEqual(spans(edit), [(A, 0, 100), (A, 200, 300)])
        self.assertEqual(edit.total_frames, 200)
        self.assertEqual(edit.locate(100), (1, 200))

    def test_delete_errors_and_deleting_everything(self):
        edit = whole(300).split_at(100)
        with self.assertRaises(EditError):
            edit.delete([])  # nothing selected
        with self.assertRaises(EditError):
            edit.delete([5])  # stale index
        self.assertEqual(edit.delete([0, 1]).total_frames, 0)  # allowed: the clips stay in the pool

    def test_nearest_seq(self):
        edit = whole(300).split_at(100).split_at(200).delete([1])
        self.assertEqual(edit.nearest_seq(A, 50), 50)
        self.assertEqual(edit.nearest_seq(A, 150), 100)  # deleted: lands on the next kept frame
        self.assertEqual(edit.nearest_seq(A, 250), 150)
        self.assertEqual(edit.nearest_seq(A, 299), 199)
        self.assertIsNone(edit.nearest_seq(B, 0))  # that clip is not on the timeline

    def test_nearest_seq_with_several_clips(self):
        edit = EditList((Segment(A, 0, 50), Segment(B, 0, 50), Segment(A, 100, 150)))
        self.assertEqual(edit.nearest_seq(B, 10), 60)
        self.assertEqual(edit.nearest_seq(A, 120), 120)  # in the second A segment
        self.assertEqual(edit.nearest_seq(A, 75), 100)  # deleted: the next A frame that is kept
        self.assertEqual(edit.nearest_seq(A, 500), 150)  # beyond everything: just after the last A

    def test_split_range_selects_the_middle(self):
        edit, inside = whole(300).split_range(50, 120)
        self.assertEqual(spans(edit), [(A, 0, 50), (A, 50, 120), (A, 120, 300)])
        self.assertEqual(inside, [1])

    def test_split_range_at_existing_boundaries_and_ends(self):
        edit, inside = whole(300).split_range(0, 300)
        self.assertEqual(spans(edit), [(A, 0, 300)])
        self.assertEqual(inside, [0])
        with self.assertRaises(EditError):
            whole(300).split_range(10, 301)

    def test_split_range_across_segments(self):
        edit, inside = whole(300).split_at(100).split_range(50, 150)
        self.assertEqual(spans(edit), [(A, 0, 50), (A, 50, 100), (A, 100, 150), (A, 150, 300)])
        self.assertEqual(inside, [1, 2])

    def test_merged_ranges_joins_touching_segments_of_one_clip(self):
        edit = whole(300).split_at(100).split_at(200)
        self.assertEqual(edit.merged_ranges(), [Segment(A, 0, 300)])
        self.assertEqual(edit.delete([1]).merged_ranges(), [Segment(A, 0, 100), Segment(A, 200, 300)])
        # Touching in frame numbers is not enough: the clips differ.
        mixed = EditList((Segment(A, 0, 100), Segment(B, 100, 200)))
        self.assertEqual(len(mixed.merged_ranges()), 2)

    def test_sources_in_order_of_first_use(self):
        edit = EditList((Segment(B, 0, 5), Segment(A, 0, 5), Segment(B, 5, 9)))
        self.assertEqual(edit.sources(), [B, A])

    def test_append_and_remove_source(self):
        edit = whole(10).append([Segment(B, 0, 20), Segment(A, 10, 15)])
        self.assertEqual(spans(edit), [(A, 0, 10), (B, 0, 20), (A, 10, 15)])
        self.assertEqual(spans(edit.remove_source(A)), [(B, 0, 20)])
        self.assertEqual(edit.remove_source(B).total_frames, 15)

    def test_value_equality(self):
        self.assertEqual(whole(10).split_at(5), whole(10).split_at(5))
        self.assertNotEqual(whole(10), whole(10).split_at(5))
        self.assertNotEqual(whole(10, A), whole(10, B))


class ArrangeTests(unittest.TestCase):
    def test_shift_swaps_with_the_neighbour(self):
        edit, moved = four().shift([1], -1)
        self.assertEqual(order(edit), [B, A, C, D])
        self.assertEqual(moved, [0])
        edit, moved = four().shift([1], 1)
        self.assertEqual(order(edit), [A, C, B, D])
        self.assertEqual(moved, [2])

    def test_shift_moves_a_group_together(self):
        edit, moved = four().shift([1, 2], 1)
        self.assertEqual(order(edit), [A, D, B, C])
        self.assertEqual(moved, [2, 3])
        edit, moved = four().shift([2, 3], -1)
        self.assertEqual(order(edit), [A, C, D, B])
        self.assertEqual(moved, [1, 2])

    def test_shift_at_the_ends(self):
        with self.assertRaises(EditError):
            four().shift([0], -1)
        with self.assertRaises(EditError):
            four().shift([3], 1)
        with self.assertRaises(EditError):
            four().shift([0, 1, 2, 3], 1)
        with self.assertRaises(EditError):
            four().shift([], 1)

    def test_move_to_a_gap(self):
        edit, moved = four().move_to([0], 3)  # before segment 3
        self.assertEqual(order(edit), [B, C, A, D])
        self.assertEqual(moved, [2])
        edit, moved = four().move_to([0], 4)  # the very end
        self.assertEqual(order(edit), [B, C, D, A])
        self.assertEqual(moved, [3])
        edit, moved = four().move_to([3], 0)  # the very start
        self.assertEqual(order(edit), [D, A, B, C])
        self.assertEqual(moved, [0])

    def test_move_to_keeps_the_order_of_a_group(self):
        edit, moved = four().move_to([2, 3], 0)
        self.assertEqual(order(edit), [C, D, A, B])
        self.assertEqual(moved, [0, 1])
        edit, moved = four().move_to([0, 2], 4)  # not adjacent: they end up together
        self.assertEqual(order(edit), [B, D, A, C])
        self.assertEqual(moved, [2, 3])

    def test_move_to_where_it_already_is(self):
        for indices, gap in (([1], 1), ([1], 2), ([0, 1], 2), ([3], 4)):
            with self.assertRaises(EditError, msg=f"{indices} -> {gap}"):
                four().move_to(indices, gap)

    def test_move_to_rejects_bad_input(self):
        with self.assertRaises(EditError):
            four().move_to([], 1)
        with self.assertRaises(EditError):
            four().move_to([7], 1)
        with self.assertRaises(EditError):
            four().move_to([1], 9)


class EditStateTests(unittest.TestCase):
    def setUp(self):
        self.a = info(A, 1920, 1080, 30.0, 300, True)
        self.b = info(B, 1280, 720, 25.0, 100, False)

    def test_initial_is_empty(self):
        state = EditState.initial()
        self.assertEqual((state.pool, state.project_format, state.crop), ((), None, None))
        self.assertEqual(state.timeline.total_frames, 0)
        self.assertTrue(state.is_pristine())

    def test_first_clip_sets_the_project_format(self):
        state = EditState.initial().add_clips([self.a, self.b])
        self.assertEqual(state.project_format, ProjectFormat(1920, 1080, 30.0))
        self.assertEqual(state.pool, (self.a, self.b))
        # b is 100 frames at 25 fps = 4 s = 120 frames at the project's 30 fps.
        self.assertEqual(spans(state.timeline), [(A, 0, 300), (B, 0, 120)])

    def test_the_format_does_not_follow_later_imports(self):
        state = EditState.initial().add_clips([self.b]).add_clips([self.a])
        self.assertEqual(state.project_format, ProjectFormat(1280, 720, 25.0))
        self.assertEqual(spans(state.timeline), [(B, 0, 100), (A, 0, 250)])  # 10 s at 25 fps

    def test_importing_a_known_file_again_only_places_it_again(self):
        state = EditState.initial().add_clips([self.a]).add_clips([self.a])
        self.assertEqual(len(state.pool), 1)
        self.assertEqual(len(state.timeline), 2)

    def test_add_to_timeline(self):
        state = EditState.initial().add_clips([self.a, self.b]).add_to_timeline(B)
        self.assertEqual(spans(state.timeline)[-1], (B, 0, 120))
        with self.assertRaises(EditError):
            state.add_to_timeline("nope.mp4")

    def test_remove_clip(self):
        state = EditState.initial().add_clips([self.a, self.b])
        state = state.with_timeline(state.timeline.split_at(100))
        removed = state.remove_clip(A)
        self.assertEqual(removed.pool, (self.b,))
        self.assertEqual(spans(removed.timeline), [(B, 0, 120)])
        self.assertEqual(removed.project_format, state.project_format)  # fixed once set
        with self.assertRaises(EditError):
            removed.remove_clip(A)

    def test_removing_the_last_clip_resets_the_project(self):
        state = EditState.initial().add_clips([self.a]).with_crop(fit_crop(1920, 1080, (1, 1)))
        self.assertEqual(state.remove_clip(A), EditState.initial())

    def test_pristine_means_nothing_beyond_importing(self):
        state = EditState.initial().add_clips([self.a, self.b])
        self.assertTrue(state.is_pristine())
        self.assertFalse(state.with_timeline(state.timeline.split_at(10)).is_pristine())
        self.assertFalse(state.with_timeline(state.timeline.delete([1])).is_pristine())
        self.assertFalse(state.with_crop(fit_crop(1920, 1080, (1, 1))).is_pristine())
        self.assertFalse(state.add_to_timeline(A).is_pristine())  # placed twice
        self.assertTrue(state.remove_clip(B).is_pristine())  # still just "what was imported"

    def test_crop_and_timeline_are_independent(self):
        state = EditState.initial().add_clips([self.a])
        crop = fit_crop(1920, 1080, (1, 1))
        cropped = state.with_crop(crop)
        self.assertEqual((cropped.crop, cropped.timeline), (crop, state.timeline))
        self.assertEqual(cropped.with_timeline(state.timeline.split_at(10)).crop, crop)

    def test_clip_lookup(self):
        state = EditState.initial().add_clips([self.a])
        self.assertIs(state.clip(A), self.a)
        with self.assertRaises(KeyError):
            state.clip("nope.mp4")


class HistoryTests(unittest.TestCase):
    def test_undo_redo(self):
        h = History(whole(100))
        a = h.current.split_at(10)
        b = a.split_at(20)
        h.push(a, "split a")
        h.push(b, "split b")
        self.assertEqual(h.undo(), "split b")
        self.assertEqual(h.current, a)
        self.assertEqual(h.undo(), "split a")
        self.assertEqual(h.current, whole(100))
        self.assertFalse(h.can_undo)
        self.assertEqual(h.redo(), "split a")
        self.assertEqual(h.redo(), "split b")
        self.assertEqual(h.current, b)
        self.assertFalse(h.can_redo)

    def test_new_edit_clears_redo(self):
        h = History(whole(100))
        h.push(h.current.split_at(10), "one")
        h.undo()
        h.push(h.current.split_at(20), "two")
        self.assertFalse(h.can_redo)

    def test_unchanged_state_is_not_recorded(self):
        h = History(whole(100))
        self.assertFalse(h.push(whole(100), "noop"))
        self.assertFalse(h.can_undo)

    def test_limit(self):
        h = History(0, limit=3)
        for i in range(1, 8):
            h.push(i, str(i))
        undone = 0
        while h.can_undo:
            h.undo()
            undone += 1
        self.assertEqual(undone, 3)

    def test_undo_covers_imports_arranging_and_crop(self):
        a, b = info(A), info(B, frames=100)
        h = History(EditState.initial())
        h.push(h.current.add_clips([a]), "import a")
        h.push(h.current.add_clips([b]), "import b")
        h.push(h.current.with_timeline(h.current.timeline.shift([1], -1)[0]), "move earlier")
        h.push(h.current.with_crop(fit_crop(1920, 1080, (1, 1))), "crop")
        h.push(h.current.remove_clip(A), "remove a")
        self.assertEqual(h.current.pool, (b,))
        h.undo()  # remove
        self.assertEqual(h.current.pool, (a, b))
        self.assertEqual(order(h.current.timeline), [B, A])
        h.undo()  # crop
        self.assertIsNone(h.current.crop)
        h.undo()  # move
        self.assertEqual(order(h.current.timeline), [A, B])
        h.undo()  # import b
        self.assertEqual(h.current.pool, (a,))
        h.undo()  # import a
        self.assertEqual(h.current, EditState.initial())
        h.redo()
        h.redo()
        self.assertEqual(h.current.pool, (a, b))


class RenderPlanTests(unittest.TestCase):
    """The FFmpeg filter graph, built without running FFmpeg."""

    def state(self, *clips, edit=None):
        state = EditState.initial().add_clips(list(clips))
        return state.with_timeline(edit) if edit is not None else state

    def test_single_clip_is_a_plain_trim_and_join(self):
        plan = plan_render(self.state(info(A)), 1920, 1080)
        g = plan.graph
        self.assertEqual(plan.inputs, (A,))
        self.assertTrue(plan.with_audio)
        self.assertIn("[0:v]trim=start_frame=0:end_frame=300,setpts=PTS-STARTPTS[sv0]", g)
        self.assertIn("[0:a]atrim=start=0.000000:end=10.000000,asetpts=PTS-STARTPTS[sa0]", g)
        self.assertIn("[sv0][sa0]concat=n=1:v=1:a=1[cv][aout]", g)
        self.assertTrue(g.endswith("[cv]fps=30/1[vout]"))
        for absent in ("split", "scale", "crop", "aformat", "anullsrc"):
            self.assertNotIn(absent, g)

    def test_two_ranges_of_one_clip_split_the_input(self):
        state = self.state(info(A))
        edit = state.timeline.split_at(100).split_at(200).delete([1])
        g = plan_render(state.with_timeline(edit), 1920, 1080).graph
        self.assertIn("[0:v]split=2[v0_0][v0_1]", g)
        self.assertIn("[0:a]asplit=2[a0_0][a0_1]", g)
        self.assertIn("[v0_1]trim=start_frame=200:end_frame=300", g)
        self.assertIn("concat=n=2:v=1:a=1", g)

    def test_clips_are_inputs_in_order_of_first_use(self):
        a, b = info(A), info(B)
        state = self.state(a, b)
        reordered = state.with_timeline(state.timeline.shift([1], -1)[0])
        plan = plan_render(reordered, 1920, 1080)
        self.assertEqual(plan.inputs, (B, A))
        self.assertIn("[0:v]trim", plan.graph)  # b is input 0 now

    def test_a_clip_used_twice_is_one_input(self):
        state = self.state(info(A), info(B)).add_to_timeline(A)
        plan = plan_render(state, 1920, 1080)
        self.assertEqual(plan.inputs, (A, B))
        self.assertIn("[0:v]split=2", plan.graph)

    def test_other_size_and_rate_are_conformed(self):
        a = info(A, 1920, 1080, 30.0, 300)
        b = info(B, 1280, 720, 25.0, 100, audio=False)
        g = plan_render(self.state(a, b), 1920, 1080).graph
        # The 25 fps clip is cut by time (4 s = 120 project frames) and converted to 30 fps ...
        self.assertIn("[1:v]trim=start=0.000000:end=4.000000,setpts=PTS-STARTPTS,fps=30/1,", g)
        # ... and letterboxed into the project size.
        self.assertIn(
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[sv1]", g
        )
        self.assertNotIn("[0:v]trim=start_frame=0:end_frame=300,setpts=PTS-STARTPTS,fps", g)  # a is untouched

    def test_a_silent_clip_gets_silence_when_others_have_sound(self):
        a = info(A, audio=True)
        b = info(B, 1280, 720, 30.0, 150, audio=False)
        plan = plan_render(self.state(a, b), 1920, 1080)
        self.assertTrue(plan.with_audio)
        self.assertIn("anullsrc=r=48000:cl=stereo,atrim=end=5.000000,asetpts=PTS-STARTPTS,aformat=", plan.graph)
        self.assertNotIn("[1:a]", plan.graph)
        self.assertEqual(plan.graph.count("aformat="), 2)  # both clips are brought to one audio format

    def test_no_audio_when_no_clip_has_any_or_when_muted(self):
        silent = plan_render(self.state(info(A, audio=False), info(B, audio=False)), 1920, 1080)
        self.assertFalse(silent.with_audio)
        self.assertNotIn("atrim", silent.graph)
        muted = plan_render(self.state(info(A), info(B)), 1920, 1080, mute_audio=True)
        self.assertFalse(muted.with_audio)
        self.assertIn("concat=n=2:v=1:a=0[cv]", muted.graph)
        self.assertNotIn("[aout]", muted.graph)

    def test_crop_comes_after_fps_and_before_scale(self):
        state = self.state(info(A)).with_crop(Crop(480, 0, 960, 1080, (1, 1)))
        g = plan_render(state, 480, 540).graph
        self.assertTrue(g.endswith("[cv]fps=30/1,crop=960:1080:480:0,scale=480:540:flags=lanczos,setsar=1[vout]"))

    def test_crop_without_resize_has_no_scale(self):
        state = self.state(info(A)).with_crop(Crop(480, 0, 960, 1080))
        self.assertTrue(plan_render(state, 960, 1080).graph.endswith("[cv]fps=30/1,crop=960:1080:480:0[vout]"))

    def test_resize_to_project_size_still_scales_a_cropped_picture(self):
        state = self.state(info(A)).with_crop(Crop(480, 0, 960, 1080))
        self.assertIn("scale=1920:1080", plan_render(state, 1920, 1080).graph)

    def test_full_frame_crop_is_ignored(self):
        state = self.state(info(A)).with_crop(Crop(0, 0, 1920, 1080, (16, 9)))
        self.assertNotIn("crop", plan_render(state, 1920, 1080).graph)

    def test_effective_crop_and_picture_size(self):
        fmt = ProjectFormat(1920, 1080, 30.0)
        full = Crop(0, 0, 1920, 1080, (16, 9))
        self.assertIsNone(effective_crop(fmt, full))
        self.assertIsNone(effective_crop(fmt, None))
        self.assertEqual(picture_size(fmt, full), (1920, 1080))
        self.assertEqual(picture_size(fmt, Crop(0, 0, 1080, 1080)), (1080, 1080))

    def test_nothing_to_export(self):
        with self.assertRaises(ExportError):
            plan_render(EditState.initial(), 100, 100)
        empty_timeline = self.state(info(A)).with_timeline(EditList())
        with self.assertRaises(ExportError):
            plan_render(empty_timeline, 1920, 1080)

    def test_even(self):
        self.assertEqual((even(1281), even(720), even(1)), (1280, 720, 2))


if __name__ == "__main__":
    unittest.main()
