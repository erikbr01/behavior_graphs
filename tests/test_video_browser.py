"""Tests for the user study's rollout-video browser.

The browser is the whole of the video-only study step, so what matters is that
a page's episodes all render at once (a grid, not expanders the participant has
to open), that the filter narrows the set, and that paging walks the filtered
list without dropping or repeating episodes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.user_study.video_browser import (  # noqa: E402
    episode_status,
    filter_episodes,
)

# 7 episodes: alternating outcomes, with one unknown, so a page_size of 3 leaves
# a short last row to lay out.
_OUTCOMES = [True, False, True, True, False, None, False]


def _index() -> dict:
    return {
        "episodes": [
            {"index": i * 10, "path": f"ep{i * 10}.mp4", "success": ok}
            for i, ok in enumerate(_OUTCOMES)
        ]
    }


def test_filter_episodes_selects_by_outcome():
    episodes = _index()["episodes"]

    assert len(filter_episodes(episodes, "All")) == 7
    assert [ep["index"] for ep in filter_episodes(episodes, "Success only")] == [0, 20, 30]
    assert [ep["index"] for ep in filter_episodes(episodes, "Failure only")] == [10, 40, 60]
    # The unknown-outcome episode belongs to neither side.
    assert 50 not in [ep["index"] for ep in filter_episodes(episodes, "Failure only")]


def test_episode_status_labels():
    assert episode_status({"success": True}) == "✓ Success"
    assert episode_status({"success": False}) == "✗ Failure"
    assert episode_status({}) == "Unknown"


def _browser_script(page_size: int, show_true_index: bool) -> None:
    """Script body for AppTest: the browser over clips that do not exist.

    Missing files are deliberate — ``mp4_player`` then renders its "Video
    unavailable" notice per cell, which counts the cells without needing real
    video files or ffmpeg in the test environment.
    """
    from pathlib import Path

    from behavior_graphs.viz.streamlit.user_study.video_browser import render_video_browser

    render_video_browser(
        Path("/nonexistent"),
        {
            "episodes": [
                {"index": i * 10, "path": f"ep{i * 10}.mp4", "success": ok}
                for i, ok in enumerate([True, False, True, True, False, None, False])
            ]
        },
        page_size=page_size,
        show_true_index=show_true_index,
    )


def _run(page_size=3, show_true_index=True) -> AppTest:
    at = AppTest.from_function(
        _browser_script,
        kwargs={"page_size": page_size, "show_true_index": show_true_index},
        default_timeout=60,
    )
    return at.run()


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_page_renders_every_episode_without_expanders():
    at = _run(page_size=3)

    assert not at.exception
    # No collapsed containers: every player on the page is already visible.
    assert at.get("expandable") == []
    assert _captions(at) == [
        "**Episode 0** · ✓ Success",
        "**Episode 10** · ✗ Failure",
        "**Episode 20** · ✓ Success",
    ]
    assert len(at.info) == 3  # one player per episode


def test_paging_walks_the_whole_list():
    at = _run(page_size=3)
    seen = list(_captions(at))

    # 7 episodes at 3 per page -> 3 pages, the last one short.
    at.button(key="vbrow_next").click().run()
    seen += _captions(at)
    at.button(key="vbrow_next").click().run()
    seen += _captions(at)

    assert len(seen) == 7
    assert len(set(seen)) == 7
    assert seen[-1] == "**Episode 60** · ✗ Failure"
    assert at.button(key="vbrow_next").disabled
    assert not at.button(key="vbrow_prev").disabled


def test_there_is_one_pager_and_not_two():
    """It used to be drawn above the grid and again below it. Four clips in two
    columns is about a screen, so the second copy saved no scrolling and put a
    second pair of controls on the page doing the same thing as the first."""
    at = _run(page_size=3)

    assert [button.key for button in at.button if "prev" in button.key] == ["vbrow_prev"]
    assert [button.key for button in at.button if "next" in button.key] == ["vbrow_next"]


def test_filter_narrows_the_grid_and_resets_the_page():
    at = _run(page_size=3)
    at.button(key="vbrow_next").click().run()

    at.radio(key="vbrow_filter").set_value("Failure only").run()

    # Back to page 1 of the failures — not page 2 of a now-shorter list.
    assert _captions(at) == [
        "**Episode 10** · ✗ Failure",
        "**Episode 40** · ✗ Failure",
        "**Episode 60** · ✗ Failure",
    ]
    assert at.button(key="vbrow_next").disabled


def _no_match_script() -> None:
    """A rollout set with no failures, so "Failure only" matches nothing."""
    from pathlib import Path

    from behavior_graphs.viz.streamlit.user_study.video_browser import render_video_browser

    render_video_browser(
        Path("/nonexistent"),
        {"episodes": [{"index": i, "path": f"ep{i}.mp4", "success": True} for i in range(3)]},
        page_size=3,
        key_prefix="vbrow",
    )


def test_an_empty_filter_says_so_and_leaves_a_way_back():
    at = AppTest.from_function(_no_match_script, default_timeout=60).run()

    at.radio(key="vbrow_filter").set_value("Failure only").run()

    assert not at.exception
    assert [i.value for i in at.info] == ["No episodes match this filter."]
    assert _captions(at) == []
    # The filter and the nav row still render, so the participant can get back
    # to "All" instead of being stranded on an empty page.
    assert at.radio(key="vbrow_filter").value == "Failure only"
    assert at.button(key="vbrow_next").disabled
    assert at.button(key="vbrow_prev").disabled


def test_display_numbers_hide_the_true_episode_index():
    at = _run(page_size=3, show_true_index=False)

    # Positions in the (shuffled) full list, 1-based — not the episode indices.
    assert _captions(at) == [
        "**Episode 1** · ✓ Success",
        "**Episode 2** · ✗ Failure",
        "**Episode 3** · ✓ Success",
    ]
