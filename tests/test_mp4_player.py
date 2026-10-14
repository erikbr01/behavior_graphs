"""Tests for where the player's cursor lands when it opens.

Clicking a behavior should start the video at that behavior. Two things used to
get in the way: a multi-segment selection (a path, or a src→tgt transition)
passed no ``slice_start`` at all and opened at frame 0, and the seek that did
happen converted frames with ``frames / fps`` while the bars underneath were
positioned against the clip's real duration — so on any clip where those two
disagree the playhead sat outside its own bar.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from behavior_graphs.viz.streamlit.components import mp4_player as mod  # noqa: E402


def test_start_frame_prefers_an_explicit_slice():
    assert mod.start_frame(40, None) == 40
    # A single-behavior selection carries both; the slice is the behavior.
    assert mod.start_frame(40, [(10, 20, "a", "#fff")]) == 40


def test_start_frame_uses_the_earliest_segment_when_there_is_no_slice():
    segments = [(60, 90, "tgt", "#111"), (30, 60, "src", "#222")]

    # Segment order follows the path, not the clock — take the earliest.
    assert mod.start_frame(None, segments) == 30


def test_start_frame_is_none_when_nothing_is_highlighted():
    assert mod.start_frame(None, None) is None
    assert mod.start_frame(None, []) is None


def test_seek_frame_overrides_the_highlight():
    """A terminal leaf highlights the whole path but opens on its last behavior.

    The earlier segments still belong on the timeline as context, so the frame to
    open on cannot be derived from them — it is passed in.
    """
    segments = [(0, 30, "B4", "#111"), (30, 90, "B11", "#222")]

    assert mod.start_frame(None, segments, 30) == 30
    # Wins over an explicit slice too, not just over the segment fallback.
    assert mod.start_frame(0, segments, 30) == 30


def _render_html(monkeypatch, **kwargs) -> str:
    """The HTML mp4_player emits, with media resolution and the iframe stubbed."""
    captured: dict[str, str] = {}

    monkeypatch.setattr(mod, "video_src", lambda *a, **k: "/media/fake.mp4")
    monkeypatch.setattr(mod, "poster_src", lambda *a, **k: None)
    monkeypatch.setattr(mod.components, "html", lambda html, **k: captured.setdefault("html", html))

    mod.mp4_player("/nonexistent/ep0.mp4", key="t", **kwargs)
    return captured["html"]


def test_seek_scales_frames_through_the_clips_duration(monkeypatch):
    # 300 policy steps over a clip that is not 300/10 = 30s long: the frame ->
    # seconds mapping has to come from v.duration, like the bars do.
    html = _render_html(monkeypatch, slice_start=150, slice_end=200, total_frames=300, fps=10)

    assert "var _seekF=150;" in html
    assert "t=_sec(_seekF" in html
    # Not the old fixed-seconds seek (150/10 = 15.00).
    assert "v.currentTime=15.00" not in html


def test_multi_segment_selection_opens_at_its_first_segment(monkeypatch):
    html = _render_html(
        monkeypatch,
        segments=[(120, 180, "src", "#f5a623"), (180, 240, "tgt", "#38bdf8")],
        total_frames=300,
        fps=10,
    )

    assert "var _seekF=120;" in html


def test_whole_episode_player_opens_at_the_top(monkeypatch):
    html = _render_html(monkeypatch, total_frames=300, fps=10)

    # The video browser's grid highlights nothing — -1 means "do not seek".
    assert "var _seekF=-1;" in html
