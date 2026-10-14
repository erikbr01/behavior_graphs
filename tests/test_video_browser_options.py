"""Tests for the clip grid's display options.

All three exist because the grid is what a participant browses before answering:
where a clip opens, whether its number means anything, and whether the page says
so are part of what they are being asked to judge.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_SCRIPT = """
import streamlit as st
from pathlib import Path
from behavior_graphs.viz.streamlit.user_study.video_browser import render_video_browser

index = {
    "fps": 10,
    "cameras": ["exterior"],
    "episodes": [
        {"index": i, "success": i % 2 == 0, "frame_count": 400,
         "path": f"media/ep{i}.mp4", "views": {}}
        for i in range(4)
    ],
    "by_index": {},
}
render_video_browser(
    Path("/nonexistent"), index, page_size=4, key_prefix="vb",
    show_true_index=SHOW_TRUE, lead_seconds=LEAD, shuffle_indices=SHUFFLE,
)
"""


def _run(*, show_true: bool = False, lead: float = 0.0, shuffle: bool = False) -> AppTest:
    script = (
        _SCRIPT.replace("SHOW_TRUE", str(show_true))
        .replace("LEAD", str(lead))
        .replace("SHUFFLE", str(shuffle))
    )
    return AppTest.from_string(script, default_timeout=30).run()


def _numbers(at: AppTest) -> list[int]:
    out = []
    for caption in at.caption:
        text = caption.value
        if text.startswith("**Episode "):
            out.append(int(text.split("**Episode ")[1].split("**")[0]))
    return out


def test_the_grid_does_not_announce_the_lead_in_itself():
    """A clip that starts mid-task reads as a rollout that started mid-task, so
    it has to be said — but by the page, with the rest of what it says about the
    clips, not as a note of its own above the grid."""
    at = _run(lead=6)

    assert not at.exception
    assert not [c for c in at.caption if "before it ends" in c.value]


def test_the_study_page_says_it_instead():
    """Said in the task line the studies share, above the grid on every layout."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies import evidence

    src = inspect.getsource(evidence.render_task_restatement)
    assert "seconds before it ends" in src
    assert 'graph_params.get("video_lead_seconds"' in src


def test_display_numbers_are_sequential_unless_shuffled():
    assert _numbers(_run(shuffle=False)) == [1, 2, 3, 4]


def test_shuffled_numbers_are_a_permutation_and_stay_put():
    at = _run(shuffle=True)
    first = _numbers(at)

    assert sorted(first) == [1, 2, 3, 4]
    # Turning a page must not renumber a clip someone has written about.
    at = at.radio[0].set_value("Success only").run()
    kept = _numbers(at)
    assert set(kept) <= set(first)
    assert len(kept) == 2


def test_true_indices_ignore_the_shuffle():
    """`show_true_index` means the real episode id, whatever the display option."""
    assert _numbers(_run(show_true=True, shuffle=True)) == [0, 1, 2, 3]


# ── The graph's +/- badges ────────────────────────────────────────────────────


def test_the_study_hides_the_fold_and_subcluster_badges():
    """Folding a rare branch back in, or splitting a leaf into subclusters, are
    things to do to a graph you already read fluently. A participant meeting one
    for the first time has a reading task, and two controls per node is a second
    one on top of it."""
    from pathlib import Path as _Path

    import yaml

    config = (
        _Path(__file__).resolve().parents[1]
        / "configs"
        / "viz"
        / "study_variants"
        / "three_stage_v1.yaml"
    )
    assert yaml.safe_load(config.read_text())["graph"]["show_fold_toggles"] is False


def test_the_flag_reaches_the_component_as_an_empty_fold_map():
    """The component draws one badge per `fold_state` entry, so withholding the
    map is the whole switch — nothing about which branches are folded changes.
    """
    import inspect

    from behavior_graphs.viz.streamlit.components import trajectory_tree_view as ttv

    assert "show_fold_toggles" in inspect.signature(ttv.render_trajectory_tree).parameters
    source = inspect.getsource(ttv._render_native_svg)
    assert "fold_state=fold_state if show_fold_toggles else {}" in source


def test_it_defaults_to_showing_them_for_the_lab_app():
    """The demo app explores graphs; it is the study that wants them gone."""
    import inspect

    from behavior_graphs.viz.streamlit.components import trajectory_tree_view as ttv

    parameter = inspect.signature(ttv.render_trajectory_tree).parameters["show_fold_toggles"]
    assert parameter.default is True


def test_the_video_page_is_framed_like_the_graph_page():
    """Header, a paragraph saying what the unit is, a key, and a heading over
    the artifact. The graph page had all four and the video page had one, which
    tells a participant which display the study is about before they read
    either."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies import evidence

    source = inspect.getsource(evidence.render_phase_view)
    videos, graph = source.split('else:  # "graph"')

    for piece in ("st.header(", "st.markdown(", "_intro()", "st.subheader("):
        assert piece in videos, f"the video branch is missing {piece}"
        assert piece in graph, f"the graph branch is missing {piece}"


def test_both_displays_explain_themselves_in_the_same_number_of_lines():
    """The grid's key is a list and the graph's was only an annotated figure.
    A study asking which of two displays a participant prefers cannot hand one
    of them the explaining."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies import evidence
    from behavior_graphs.viz.streamlit.user_study import graph_intro

    def bullets(source: str) -> int:
        return source.count('"- ')

    grid = bullets(inspect.getsource(evidence.render_rollout_grid_intro))
    graph = bullets(inspect.getsource(graph_intro.render_graph_intro))

    assert grid > 0 and graph > 0, f"grid={grid} graph={graph}"
    assert abs(grid - graph) <= 1, f"grid has {grid} lines, graph has {graph}"


def test_the_graph_numbers_a_clip_the_way_the_grid_does():
    """Both displays show the same rollouts, so they must call a clip the same
    thing. The graph's panel used to caption with the true episode index while
    the grid showed a number drawn from a permutation, which both contradicted
    the grid and handed back the recording order `shuffle_rollout_indices`
    exists to withhold."""
    from behavior_graphs.viz.streamlit.user_studies.evidence import _display_numbers
    from behavior_graphs.viz.streamlit.user_study.video_browser import shuffled_numbers

    index = {"episodes": [{"index": i} for i in (3, 17, 4, 92, 5)]}
    params = {"shuffle_rollout_indices": True}

    grid = shuffled_numbers(index["episodes"])
    graph = _display_numbers(index, params, hide_episode_index=True)

    assert graph == grid
    # And it really is shuffled: a permutation of 1..N that is not simply the
    # episodes numbered down the page, which is what it replaces.
    assert sorted(graph.values()) == [1, 2, 3, 4, 5]
    sequential = {ep["index"]: i for i, ep in enumerate(index["episodes"], start=1)}
    assert graph != sequential


def test_true_indices_are_left_alone_when_the_study_shows_them():
    """A display map replaces the real index rather than sitting beside it, so
    a study that shows real indices must get no map at all — otherwise the grid
    would print the true index and the graph a permuted one."""
    from behavior_graphs.viz.streamlit.user_studies.evidence import _display_numbers

    index = {"episodes": [{"index": i} for i in range(4)]}

    assert _display_numbers(index, {"shuffle_rollout_indices": True}, False) is None
    assert _display_numbers(index, {"shuffle_rollout_indices": False}, True) is None


def test_every_number_the_graph_panel_prints_goes_through_the_map():
    """Four places print an episode number: the caption, the no-video warning,
    the thumbnail strip and the name of the exported clip. A new one added
    without the map is the same bug again, in a corner nobody looks at.

    The filename counts even though the export is off in the study apps, where
    the map is the only place a permutation exists: a downloaded file is read by
    a person, and "it cannot be reached from there" is a property of today's
    callers rather than of this function.
    """
    import inspect
    import re

    from behavior_graphs.viz.streamlit.components import trajectory_tree_view

    source = inspect.getsource(trajectory_tree_view._show_one_video_panel)
    # Only what reaches the participant. A widget key also interpolates the
    # episode index and must keep the true one: it has to be unique and stable,
    # and nobody reads it.
    visible = [
        line
        for line in source.splitlines()
        if re.search(r"st\.(caption|warning|info)\(|\"label\":", line)
    ]
    leaks = [line.strip() for line in visible if re.search(r"\{(ep_idx|ep)\}", line)]
    assert not leaks, f"episode number shown without the display map: {leaks}"
    assert source.count("_shown(") == 4
    assert "_ep{_shown(ep_idx, episode_numbers)}" in source
