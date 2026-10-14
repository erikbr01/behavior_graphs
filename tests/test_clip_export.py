"""The clip export: geometry, the bar's reading, and one real encode.

What these guard is the two ways an export can lie about the page it came from.
The bands can land in the wrong place (frame → fraction has two conventions in
this codebase and only one of them is right for a VLA clip), and the views can
be laid out so that ffmpeg refuses the stack or silently letterboxes a rotated
camera. Both are arithmetic, so both are tested without encoding anything; one
test at the end actually runs ffmpeg, because a filter graph that is wrong by a
pixel is only wrong at encode time.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from behavior_graphs.viz import clip_figure
from behavior_graphs.viz.clip_figure import (
    HORIZONTAL,
    VERTICAL,
    Band,
    ClipInfo,
)
from behavior_graphs.viz.streamlit.components import clip_export

# ── Where the bands go ───────────────────────────────────────────────────────


def test_bands_are_placed_by_policy_step_not_by_fps():
    """The one conversion a VLA clip makes visible.

    40 policy steps stepped at 10 Hz is 4 s of *policy*, but the clip is 20 s of
    wall clock. Divide by fps and the whole annotation crams into the first
    fifth of the video; divide by total_frames and the last behavior ends where
    the clip does — which is what the player's JS computes and what the reader
    sees on the page.
    """
    bands = [Band(0, 20, "reaching", "#1f77b4"), Band(20, 40, "grasping", "#ff7f0e")]
    spans = clip_figure.band_spans(bands, total_frames=40, duration=20.0, fps=10)
    assert spans == [(0.0, 0.5), (0.5, 1.0)]


def test_bands_fall_back_to_fps_without_a_frame_count():
    spans = clip_figure.band_spans(
        [Band(0, 50, "reaching", "#1f77b4")], total_frames=None, duration=10.0, fps=10
    )
    assert spans == [(0.0, 0.5)]


def test_band_spans_never_run_off_the_clip():
    """A segment extended past the last frame still ends at the end of the bar."""
    spans = clip_figure.band_spans(
        [Band(0, 500, "reaching", "#1f77b4")], total_frames=100, duration=10.0, fps=10
    )
    assert spans == [(0.0, 1.0)]


def test_the_live_band_at_a_boundary_is_the_one_starting_there():
    """The panels close the gaps, so one band's end is the next one's start.

    Testing both ends inclusively named the behavior that had just finished —
    exactly the frame a terminal selection opens on. Same rule as the player's
    ``_mhl``.
    """
    spans = [(0.0, 0.5), (0.5, 1.0)]
    assert clip_figure.live_band(spans, 0.0) == 0
    assert clip_figure.live_band(spans, 0.49) == 0
    assert clip_figure.live_band(spans, 0.5) == 1
    # The very end belongs to the last band, or the final frame matches nothing.
    assert clip_figure.live_band(spans, 1.0) == 1


def test_nothing_is_live_before_the_first_band():
    assert clip_figure.live_band([(0.25, 0.5)], 0.1) is None


# ── The bar itself ───────────────────────────────────────────────────────────


def test_the_bar_grows_a_row_per_line_of_legend_and_stops():
    """A twenty-behavior path must not give a strip taller than the video."""
    long_labels = [f"behavior number {i} doing something at some length" for i in range(40)]
    layout = clip_figure.bar_layout(long_labels, 1280)
    assert 0 < len(layout.rows) <= clip_figure.MAX_LEGEND_ROWS
    assert layout.height < 200


def test_a_band_with_no_legend_entry_still_draws_its_frames():
    """The legend truncates; the bands do not, and the playhead crosses them.

    `bar_layout` stops after MAX_LEGEND_ROWS, so a long path has bands with a
    coloured span on the track and no legend entry — and therefore no "live"
    backdrop. Indexing the backdrops raised KeyError the moment the playhead
    reached the 13th behavior, which `export_clip` does not catch: the panel
    showed a traceback instead of a clip. Exactly the case the module docstring
    claims to handle, and the layout test above never fed it to `bar_frames`.
    """
    bands = [
        Band(i * 10, (i + 1) * 10, f"behavior {i} doing something at sentence length", "#1f77b4")
        for i in range(40)
    ]
    layout = clip_figure.bar_layout([b.label for b in bands], 1280)
    with_slots = sum(len(row) for row in layout.rows)
    assert with_slots < len(bands), "this test needs the legend to actually truncate"

    frames = list(
        clip_figure.bar_frames(bands, layout, total_frames=400, duration=40.0, fps=10, out_fps=5.0)
    )
    assert len(frames) == 200
    assert all(frame.shape == (layout.height, layout.width, 3) for frame in frames)


def test_a_long_path_exports_rather_than_raising(tmp_path, _ffmpeg):
    """The same case end to end, since the KeyError escaped export_clip."""
    clip = _make_clip(tmp_path / "front.mp4", width=160, height=120, seconds=2)
    bands = [
        Band(i * 10, (i + 1) * 10, f"behavior {i} doing something at sentence length", "#1f77b4")
        for i in range(40)
    ]
    assert clip_figure.export_clip([("front", clip)], bands, total_frames=400)


def test_a_legend_that_stopped_early_says_so():
    """Six of a path's twenty behaviors, listed with nothing to mark the gap,
    reads as a complete key to the bands — in a file whose point is to be
    believed. One entry gives way to "+N more"."""
    labels = [f"behavior {i} doing something at sentence length" for i in range(40)]
    layout = clip_figure.bar_layout(labels, 1280)

    assert layout.dropped > 0
    assert layout.rows[-1][-1] == clip_figure.MORE_SENTINEL
    # It accounts for every behavior the legend does not name, the displaced
    # entry included.
    named = sum(1 for row in layout.rows for i in row if i != clip_figure.MORE_SENTINEL)
    assert named + layout.dropped == len(labels)


def test_a_legend_that_fits_says_nothing():
    layout = clip_figure.bar_layout(["reaching", "grasping"], 1280)

    assert layout.dropped == 0
    assert clip_figure.MORE_SENTINEL not in [i for row in layout.rows for i in row]


def test_the_marker_is_drawn_without_claiming_a_behavior_colour():
    """It stands for several bands, so a colour chip would attribute all of
    them to whichever one lent its colour."""
    bands = [
        Band(i * 10, (i + 1) * 10, f"behavior {i} doing something at sentence length", "#ff00ff")
        for i in range(40)
    ]
    layout = clip_figure.bar_layout([b.label for b in bands], 1280)
    spans = clip_figure.band_spans(bands, total_frames=400, duration=40.0, fps=10)
    backdrops = clip_figure.bar_backdrops(bands, spans, layout)

    # No backdrop is keyed on the sentinel, so nothing can look it up as live.
    assert clip_figure.MORE_SENTINEL not in backdrops
    marker_row = backdrops[None][
        layout.legend_top + (len(layout.rows) - 1) * layout.line_height + 2
    ]
    assert not any(tuple(pixel) == (255, 0, 255) for pixel in marker_row[-40:]), (
        "the marker drew a colour chip"
    )


def test_the_bar_is_an_even_number_of_pixels_tall():
    """yuv420p has no odd dimensions, and ffmpeg refuses them."""
    for n_labels in range(1, 8):
        layout = clip_figure.bar_layout([f"behavior {i}" for i in range(n_labels)], 640)
        assert layout.height % 2 == 0


def test_one_backdrop_per_live_behavior_plus_one_with_none():
    """The per-frame work is a copy, so the variants are built once up front."""
    bands = [Band(0, 10, "reaching", "#1f77b4"), Band(10, 20, "grasping", "#ff7f0e")]
    layout = clip_figure.bar_layout([b.label for b in bands], 640)
    spans = clip_figure.band_spans(bands, total_frames=20, duration=2.0, fps=10)
    backdrops = clip_figure.bar_backdrops(bands, spans, layout)
    assert set(backdrops) == {None, 0, 1}
    assert all(frame.shape == (layout.height, layout.width, 3) for frame in backdrops.values())
    # The live variants differ from the neutral one only in the legend text.
    assert not np.array_equal(backdrops[0], backdrops[None])


def test_each_band_draws_at_least_a_pixel():
    """A behavior the policy passed through for one step is still a band."""
    bands = [Band(0, 1, "a blink of a behavior", "#ff00ff")]
    layout = clip_figure.bar_layout(["a blink of a behavior"], 640)
    spans = clip_figure.band_spans(bands, total_frames=1000, duration=10.0, fps=10)
    backdrop = clip_figure.bar_backdrops(bands, spans, layout)[None]
    track = backdrop[layout.track_top + 2]
    assert any(tuple(pixel) == (255, 0, 255) for pixel in track)


def test_the_playhead_sweeps_the_whole_bar():
    bands = [Band(0, 20, "reaching", "#1f77b4")]
    layout = clip_figure.bar_layout(["reaching"], 640)
    frames = list(
        clip_figure.bar_frames(bands, layout, total_frames=20, duration=2.0, fps=10, out_fps=10.0)
    )
    assert len(frames) == 20

    def head_x(frame):
        row = frame[layout.track_top + 1]
        white = [x for x, pixel in enumerate(row) if tuple(pixel) == (255, 255, 255)]
        return white[0] if white else -1

    assert head_x(frames[0]) < head_x(frames[10]) < head_x(frames[-1])


# ── Laying the views out ─────────────────────────────────────────────────────


def test_a_row_matches_heights_and_a_column_matches_widths():
    sizes = [(640, 480), (320, 240)]
    row, (row_w, row_h) = clip_figure.stack_geometry(sizes, HORIZONTAL)
    assert {h for _, h in row} == {240}, "hstack needs one height or ffmpeg refuses it"
    assert row_w == sum(w for w, _ in row) and row_h == 240

    column, (col_w, col_h) = clip_figure.stack_geometry(sizes, VERTICAL)
    assert {w for w, _ in column} == {320}, "vstack needs one width"
    assert col_w == 320 and col_h == sum(h for _, h in column)


def test_nothing_is_upscaled_to_match_a_bigger_camera():
    """The smaller view's measure wins: a 240p wrist camera blown up to 720p is
    a blurry half of the figure, and the reader cannot tell it was ever small."""
    row, _ = clip_figure.stack_geometry([(1280, 720), (320, 240)], HORIZONTAL)
    assert max(h for _, h in row) == 240


def test_a_wide_row_is_scaled_down_whole():
    """Three real-robot views must fit the cap without one being cropped."""
    sizes = [(1280, 720)] * 3
    row, (out_w, _) = clip_figure.stack_geometry(sizes, HORIZONTAL, max_width=1280)
    assert out_w <= 1280
    assert len({(w, h) for w, h in row}) == 1, "equal views keep equal shares"


def test_every_stacked_dimension_is_even():
    for sizes in ([(641, 481)], [(641, 481), (323, 241)], [(100, 33), (77, 91)]):
        for orientation in (HORIZONTAL, VERTICAL):
            scaled, out = clip_figure.stack_geometry(sizes, orientation)
            assert all(w % 2 == 0 and h % 2 == 0 for w, h in scaled)
            assert out[0] % 2 == 0 and out[1] % 2 == 0


def test_a_quarter_turned_camera_is_laid_out_on_its_turned_size():
    """Otherwise the view is letterboxed inside a box the shape it is not."""
    info = ClipInfo(width=1280, height=720, duration=5.0, fps=30.0)
    assert clip_figure.oriented_size(info, {"rotate": 90}) == (720, 1280)
    assert clip_figure.oriented_size(info, {"rotate": 180}) == (1280, 720)
    assert clip_figure.oriented_size(info, None) == (1280, 720)


def test_the_filter_graph_puts_the_bar_under_the_stack():
    graph = clip_figure.filter_graph(
        [(320, 240), (320, 240)],
        transforms=[[], ["hflip"]],
        orientation=HORIZONTAL,
        out_fps=30.0,
        bar_input=2,
    )
    assert "[v0][v1]hstack=inputs=2[stacked]" in graph
    assert "[stacked][2:v]vstack=inputs=2[out]" in graph
    # The mirror belongs to the second view alone, and comes before its scale.
    assert "[1:v]fps=30,hflip,scale=320:240" in graph
    assert "[0:v]fps=30,scale=320:240" in graph


def test_a_single_view_needs_no_stack():
    graph = clip_figure.filter_graph(
        [(320, 240)], transforms=[[]], orientation=HORIZONTAL, out_fps=10.0, bar_input=1
    )
    assert "hstack" not in graph and "[v0][1:v]vstack=inputs=2[out]" in graph


def test_without_a_bar_the_views_are_the_whole_frame():
    graph = clip_figure.filter_graph(
        [(320, 240)], transforms=[[]], orientation=HORIZONTAL, out_fps=10.0, bar_input=None
    )
    assert "vstack" not in graph and graph.endswith("[v0]null[out]")


# ── Turning a selection into bands ───────────────────────────────────────────


def test_the_players_segment_tuples_become_bands():
    bands = clip_export.bands_from_segments(
        [(0, 10, "reaching", "#1f77b4"), (10, 25, "grasping", "#ff7f0e")]
    )
    assert bands == [Band(0, 10, "reaching", "#1f77b4"), Band(10, 25, "grasping", "#ff7f0e")]


def test_an_empty_segment_is_dropped_rather_than_drawn():
    assert clip_export.bands_from_segments([(10, 10, "nothing", "#fff")]) == []
    assert clip_export.bands_from_segments(None) == []


def test_a_single_behavior_selection_exports_one_named_band():
    """The page draws this case as one orange block, and so does the file."""
    bands = clip_export.slice_band(30, 95, "lifting the pumpkin from the bowl")
    assert bands == [Band(30, 95, "lifting the pumpkin from the bowl", clip_export.SLICE_COLOR)]


def test_a_selection_with_no_span_exports_no_bar():
    assert clip_export.slice_band(None, None, "reaching") == []
    assert clip_export.slice_band(40, 40, "reaching") == []


def test_a_whole_episode_is_banded_from_its_run_length_encoding():
    """The Figures page has no graph selection, so the bar is the whole
    decomposition: consecutive windows of one behavior are one band, and the
    last one runs to the end of the clip."""
    metadata = [
        {"rollout_idx": 0, "window_start": ws, "window_end": ws + 5, "success": True}
        for ws in range(0, 40, 5)
    ]
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    bands = clip_export.episode_bands(
        labels,
        metadata,
        0,
        cluster_names={0: "reaching", 1: "grasping", 2: "lifting"},
        frame_count=60,
    )
    assert [(b.start, b.end, b.label) for b in bands] == [
        (0, 10, "reaching"),
        (10, 25, "grasping"),
        (25, 35, "lifting"),
        (35, 60, "reaching"),
    ]
    # Coloured by cluster id, so the two "reaching" bands match each other and
    # the node the graph draws for that behavior.
    assert bands[0].color == bands[3].color != bands[1].color


def test_an_unnamed_cluster_still_gets_a_label():
    metadata = [{"rollout_idx": 0, "window_start": 0, "window_end": 5}]
    bands = clip_export.episode_bands(np.array([7]), metadata, 0, frame_count=10)
    assert bands[0].label == "Behavior 7"


# ── Degrading rather than raising ────────────────────────────────────────────


def test_a_missing_clip_exports_nothing_rather_than_raising():
    assert clip_figure.export_clip([("front", "/nonexistent/ep0.mp4")]) is None


def test_no_views_at_all_exports_nothing():
    assert clip_figure.export_clip([]) is None
    assert clip_figure.export_size([]) is None


def test_stack_geometry_refuses_an_empty_stack():
    with pytest.raises(ValueError, match="at least one view"):
        clip_figure.stack_geometry([], HORIZONTAL)


def test_the_bar_pipe_is_not_flushed_after_it_is_closed(tmp_path, monkeypatch, _ffmpeg):
    """The deployed container is Python 3.10, and this only breaks there.

    ``Popen.communicate`` flushes ``self.stdin``; on 3.10 flushing an
    already-closed handle raises ``ValueError``, which is not a
    ``SubprocessError`` and so escaped the error handling entirely — the export
    died *after* ffmpeg had encoded the clip perfectly well. 3.13, which the
    tests run on, tolerates it, so the interpreter difference is emulated here
    rather than trusted to show up.
    """
    clip = _make_clip(tmp_path / "front.mp4", width=160, height=120)
    real_popen = subprocess.Popen
    seen = {}

    class Strict310Popen(real_popen):
        def communicate(self, input=None, timeout=None):  # noqa: A002 - match Popen
            seen["stdin_was"] = (
                "None" if self.stdin is None else "closed" if self.stdin.closed else "open"
            )
            if self.stdin is not None and self.stdin.closed:
                raise ValueError("flush of closed file")
            return super().communicate(input=input, timeout=timeout)

    monkeypatch.setattr(subprocess, "Popen", Strict310Popen)
    data = clip_figure.export_clip(
        [("front", clip)], [Band(0, 20, "reaching", "#1f77b4")], total_frames=20
    )

    assert seen["stdin_was"] == "None", "the closed handle was handed to communicate"
    assert data, "the clip encoded but the bytes were lost on the way out"


def test_a_failing_encode_returns_none_and_says_why(tmp_path, monkeypatch, caplog, _ffmpeg):
    """No exception reaches the page, and the reason is not simply swallowed."""
    clip = _make_clip(tmp_path / "front.mp4", width=160, height=120)
    # An impossible filter graph: ffmpeg starts, reads the strip, and fails.
    monkeypatch.setattr(clip_figure, "filter_graph", lambda *a, **k: "[0:v]nosuchfilter[out]")

    with caplog.at_level("WARNING"):
        assert (
            clip_figure.export_clip(
                [("front", clip)], [Band(0, 20, "reaching", "#1f77b4")], total_frames=20
            )
            is None
        )
    assert any("clip export failed" in record.message for record in caplog.records)


# ── One real encode ──────────────────────────────────────────────────────────


def _make_clip(path, *, width, height, seconds=2, fps=10):
    exe = clip_figure.ffmpeg_exe()
    subprocess.run(
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        timeout=120,
    )
    return path


@pytest.fixture(scope="module")
def _ffmpeg():
    exe = clip_figure.ffmpeg_exe()
    if exe is None:
        pytest.skip("no ffmpeg on this host")
    return exe


def test_probe_reads_a_clip_it_just_wrote(tmp_path_factory, _ffmpeg):
    clip = _make_clip(tmp_path_factory.mktemp("probe") / "ep0.mp4", width=320, height=240)
    info = clip_figure.probe_clip(clip)
    assert info is not None
    assert (info.width, info.height) == (320, 240)
    assert info.duration == pytest.approx(2.0, abs=0.2)
    assert info.fps == pytest.approx(10.0, abs=0.5)


def test_probe_works_without_ffprobe_on_the_path(tmp_path_factory, _ffmpeg, monkeypatch):
    """The banner parse is the fallback that keeps this working against the
    binary imageio-ffmpeg ships, which comes with no ffprobe beside it."""
    clip = _make_clip(tmp_path_factory.mktemp("probe2") / "ep0.mp4", width=160, height=120)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    info = clip_figure.probe_clip(clip)
    assert info is not None and (info.width, info.height) == (160, 120)
    assert info.duration == pytest.approx(2.0, abs=0.2)


def test_probe_says_nothing_about_a_file_that_is_not_a_video(tmp_path):
    junk = tmp_path / "ep0.mp4"
    junk.write_bytes(b"not a video")
    assert clip_figure.probe_clip(junk) is None


def test_two_views_and_a_bar_encode_into_one_playable_file(tmp_path, _ffmpeg):
    """The end-to-end shape check: a stack whose members disagree by a pixel,
    or a bar whose declared size is a byte off the frames written to the pipe,
    fails here and nowhere earlier."""
    left = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    right = _make_clip(tmp_path / "wrist.mp4", width=240, height=240)
    bands = [Band(0, 10, "reaching for the can", "#1f77b4"), Band(10, 20, "grasping", "#ff7f0e")]

    data = clip_figure.export_clip(
        [("front", left), ("wrist", right)],
        bands,
        total_frames=20,
        fps=10,
        orientation=HORIZONTAL,
    )
    assert data and data[4:8] == b"ftyp", "not an MP4"

    out = tmp_path / "out.mp4"
    out.write_bytes(data)
    info = clip_figure.probe_clip(out)
    assert info is not None
    # The views stack to 560×240, and the bar adds its own height underneath.
    row, (stack_w, stack_h) = clip_figure.stack_geometry([(320, 240), (240, 240)], HORIZONTAL)
    bar = clip_figure.bar_layout([b.label for b in bands], stack_w)
    assert (info.width, info.height) == (stack_w, stack_h + bar.height)
    assert info.duration == pytest.approx(2.0, abs=0.3)


def test_stacking_vertically_is_the_same_clip_a_different_shape(tmp_path, _ffmpeg):
    top = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    bottom = _make_clip(tmp_path / "wrist.mp4", width=320, height=240)
    data = clip_figure.export_clip(
        [("front", top), ("wrist", bottom)],
        [Band(0, 20, "reaching", "#1f77b4")],
        total_frames=20,
        orientation=VERTICAL,
    )
    assert data
    out = tmp_path / "out.mp4"
    out.write_bytes(data)
    info = clip_figure.probe_clip(out)
    assert info is not None and info.width == 320
    bar = clip_figure.bar_layout(["reaching"], 320)
    assert info.height == 480 + bar.height


def test_the_bar_can_be_left_off(tmp_path, _ffmpeg):
    clip = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    data = clip_figure.export_clip(
        [("front", clip)],
        [Band(0, 20, "reaching", "#1f77b4")],
        total_frames=20,
        with_bar=False,
    )
    assert data
    out = tmp_path / "out.mp4"
    out.write_bytes(data)
    info = clip_figure.probe_clip(out)
    assert info is not None and (info.width, info.height) == (320, 240)


def test_a_selection_with_no_bands_exports_the_video_alone(tmp_path, _ffmpeg):
    clip = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    data = clip_figure.export_clip([("front", clip)], [])
    assert data
    out = tmp_path / "out.mp4"
    out.write_bytes(data)
    info = clip_figure.probe_clip(out)
    assert info is not None and (info.width, info.height) == (320, 240)


def test_a_rotated_camera_exports_rotated(tmp_path, _ffmpeg):
    """The transform is baked into the export the same way it is baked into the
    web proxy — from `media.transform_filters`, so the two cannot disagree."""
    clip = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    data = clip_figure.export_clip(
        [("front", clip)],
        [],
        view_transforms={"front": {"rotate": 90}},
    )
    assert data
    out = tmp_path / "out.mp4"
    out.write_bytes(data)
    info = clip_figure.probe_clip(out)
    assert info is not None and (info.width, info.height) == (240, 320)


def test_export_size_predicts_the_video_part_without_encoding(tmp_path, _ffmpeg):
    left = _make_clip(tmp_path / "front.mp4", width=320, height=240)
    right = _make_clip(tmp_path / "wrist.mp4", width=240, height=240)
    assert clip_figure.export_size([("front", left), ("wrist", right)]) == (560, 240)
    assert clip_figure.export_size([("front", left), ("wrist", right)], orientation=VERTICAL) == (
        240,
        420,
    )


# ── The panel ────────────────────────────────────────────────────────────────
# Driven through Streamlit's own test runner: everything here is plain widgets
# (the players and the graph are custom components, which AppTest cannot
# execute, and none of them is in this panel).

_PANEL_SCRIPT = """
from pathlib import Path
import streamlit as st
from behavior_graphs.viz.clip_figure import Band
from behavior_graphs.viz.streamlit.components.clip_export import clip_export_panel

clip_export_panel(
    [("front", Path("FRONT")), ("wrist", Path("WRIST"))],
    [Band(0, 10, "reaching for the can", "#1f77b4"),
     Band(10, 20, "grasping the can", "#ff7f0e")] if BANDS else [],
    key="t",
    filename="clip_ep0",
    total_frames=20,
    fps=10,
    expanded=True,
)
"""


_LONG_PATH_SCRIPT = """
from pathlib import Path
import streamlit as st
from behavior_graphs.viz.clip_figure import Band
from behavior_graphs.viz.streamlit.components.clip_export import clip_export_panel

clip_export_panel(
    [("front", Path("FRONT"))],
    [Band(i * 10, (i + 1) * 10, f"behavior {i} doing something at sentence length", "#1f77b4")
     for i in range(40)],
    key="t",
    filename="clip_ep0",
    total_frames=400,
    fps=10,
    expanded=True,
)
"""


def test_a_forty_behavior_path_builds_through_the_panel(_two_views):
    """The KeyError's actual path to the user: it reached the page through this
    panel, not through `export_clip` in isolation. AppTest runs the real
    widgets and the real build, so a regression here fails as a broken panel
    rather than as a unit-test detail."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(
        _LONG_PATH_SCRIPT.replace("FRONT", str(_two_views[0])), default_timeout=180
    ).run()
    assert not at.exception

    at = at.button[0].click().run()
    assert not at.exception, "the long-path export raised instead of encoding"
    assert _downloads(at), "no clip was offered"


def _panel(front, wrist, *, bands=True):
    from streamlit.testing.v1 import AppTest

    script = (
        _PANEL_SCRIPT.replace("FRONT", str(front))
        .replace("WRIST", str(wrist))
        .replace("BANDS", str(bands))
    )
    return AppTest.from_string(script, default_timeout=120).run()


@pytest.fixture(scope="module")
def _two_views(tmp_path_factory, _ffmpeg):
    directory = tmp_path_factory.mktemp("views")
    return (
        _make_clip(directory / "front.mp4", width=320, height=240),
        _make_clip(directory / "wrist.mp4", width=240, height=240),
    )


def test_the_panel_asks_only_what_a_figure_has_to_choose(_two_views):
    """Which views, how they are stacked, and whether the bar comes with them.
    Nothing else: the clip, the episode and the bands are what the page already
    decided, and re-asking is how an export stops matching the page."""
    at = _panel(*_two_views)

    assert not at.exception
    assert [m.label for m in at.multiselect] == ["Camera views to export"]
    assert [r.label for r in at.radio] == ["Stack the views"]
    assert [c.label for c in at.checkbox] == ["Behavior bar below the video"]
    # The export starts at what the player is showing, which is every view.
    assert at.multiselect[0].value == ["front", "wrist"]


def test_one_view_is_offered_no_orientation(_two_views):
    """With a single camera selected there is nothing to stack."""
    at = _panel(*_two_views)
    at = at.multiselect[0].set_value(["front"]).run()

    assert not at.exception
    assert not at.radio


def _downloads(at) -> list[str]:
    """The download buttons on screen, by label.

    Through ``get`` because AppTest has no accessor for ``st.download_button``,
    and by label because its proto carries a media URL rather than the bytes —
    what the file *is* is tested directly against :func:`export_clip` above.
    """
    return [element.proto.label for element in at.get("download_button")]


def test_nothing_is_encoded_until_the_button_is_pressed(_two_views):
    """A panel that built on every rerun would re-encode on every click in the
    graph above it."""
    at = _panel(*_two_views)

    assert not _downloads(at)
    assert [b.label for b in at.button] == ["Build the MP4"]


def test_building_offers_the_clip_the_options_describe(_two_views):
    at = _panel(*_two_views)
    at = at.button[0].click().run()

    assert not at.exception
    labels = _downloads(at)
    assert len(labels) == 1
    assert labels[0].startswith("⬇ clip_ep0.mp4")
    # The size is on the button, so a reader knows what they are about to pull
    # before they pull it.
    assert "MB" in labels[0]


def test_changing_an_option_withdraws_the_download(_two_views):
    """What is approved above is what is downloaded. A stale file behind a
    changed orientation is the export quietly disagreeing with the panel."""
    at = _panel(*_two_views)
    at = at.button[0].click().run()
    assert _downloads(at)

    at = at.radio[0].set_value(VERTICAL).run()
    assert not _downloads(at)
    assert [b.label for b in at.button] == ["Build the MP4"]


def test_an_emptied_camera_picker_asks_rather_than_exporting_nothing(_two_views):
    at = _panel(*_two_views)
    at = at.multiselect[0].set_value([]).run()

    assert not at.exception
    assert not at.button
    assert any("at least one view" in c.value for c in at.caption)


def test_with_no_bands_the_panel_says_so_and_still_exports(_two_views):
    """A reader who has clicked nothing in the graph gets the video, and is told
    where the bar comes from rather than left wondering."""
    at = _panel(*_two_views, bands=False)

    assert not at.checkbox, "nothing to toggle when there is no bar"
    assert any("No behavior segments" in c.value for c in at.caption)

    at = at.button[0].click().run()
    assert not at.exception and _downloads(at)


def test_a_missing_clip_says_so_instead_of_offering_a_button(tmp_path):
    at = _panel(tmp_path / "gone.mp4", tmp_path / "also_gone.mp4")

    assert not at.exception
    assert not at.button
    assert any("nothing to export" in info.value for info in at.info)


# ── Who is allowed to offer it ───────────────────────────────────────────────


def _repo_python_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in (*sorted((root / "apps").rglob("*.py")), *sorted((root / "src").rglob("*.py"))):
        yield path.relative_to(root).as_posix(), path.read_text()


def test_only_the_two_figure_pages_offer_the_export():
    """The constraint the panel exists under, as a tripwire rather than a note.

    A study app must not offer a participant a download — it is an action the
    instrument does not measure — and the reviewer page is anonymous, where a
    file named after a run directory hands out exactly what the page withholds.
    Both get it by *not asking*, so a page that starts asking should have to
    come through here.
    """
    turned_on = {
        path
        for path, source in _repo_python_files()
        if "clip_export=True" in source or "clip_export_panel(" in source
    }
    assert turned_on == {
        # Passes the flag to the tree component…
        "apps/demo/_pages/graph_demo.py",
        # …and its own section calls the panel directly, the trees there being
        # static figures with no player to hang it off.
        "apps/demo/_pages/figures.py",
        # The component that renders it, and the panel itself.
        "src/behavior_graphs/viz/streamlit/components/trajectory_tree_view.py",
        "src/behavior_graphs/viz/streamlit/components/clip_export.py",
    }


def test_a_caller_that_says_nothing_gets_no_export():
    """Which is how the study apps and the reviewer page are excluded."""
    import inspect

    from behavior_graphs.viz.streamlit.components import trajectory_tree_view as ttv

    for function in (ttv.render_trajectory_tree, ttv._show_one_video_panel):
        assert inspect.signature(function).parameters["clip_export"].default is False
