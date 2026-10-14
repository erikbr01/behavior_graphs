"""Tests for the graph video-panel layout switch.

The choice reaches two very different render paths (the behavior-graph panels
and the trajectory tree's selection panel) and changes how much a panel tries to
fit across. What is worth pinning down is the observable consequence — beside the
graph, a panel shows one clip per page instead of three — and that the two ways
of making the choice agree: the demo's live sidebar control, and a study
session's config, which is where it belongs when a participant is driving.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.components.panel_layout import (  # noqa: E402
    BELOW,
    DEFAULT_LAYOUT,
    RIGHT,
    videos_per_page,
)

# AppTest.from_function runs each script function standalone, so every name it
# uses has to be imported inside it — this module's imports are not in scope.


def test_videos_per_page_shrinks_beside_the_graph():
    assert videos_per_page(BELOW) == 3
    assert videos_per_page(RIGHT) == 1


def _selector_script() -> None:
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.panel_layout import panel_layout_selector

    st.write(f"layout={panel_layout_selector(key='pl')}")


def test_selector_offers_both_layouts_and_returns_the_choice():
    at = AppTest.from_function(_selector_script, default_timeout=60).run()

    radio = at.sidebar.radio(key="pl")
    # The default is offered first, because st.radio starts on its first option.
    assert list(radio.options) == ["Right of graph", "Below graph"]
    assert at.markdown[0].value == f"layout={DEFAULT_LAYOUT}"
    assert DEFAULT_LAYOUT == RIGHT

    at.sidebar.radio(key="pl").set_value(BELOW).run()
    assert at.markdown[0].value == f"layout={BELOW}"


def _page_default_script(default: str) -> None:
    """A page with its own opinion, the way the demo reads one from its config."""
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.panel_layout import (
        panel_layout_selector,
        remembered_panel_layout,
    )

    st.write(f"selector={panel_layout_selector(key='pl', default=default)}")
    # What a view with no panel to place reads back on the same run.
    st.write(f"remembered={remembered_panel_layout(key='pl2', default=default)}")


def test_a_page_can_open_on_its_own_default_layout():
    """The demo asks for `below`; a study session that says nothing still gets RIGHT."""
    at = AppTest.from_function(_page_default_script, kwargs={"default": BELOW}).run()

    assert at.markdown[0].value == f"selector={BELOW}"
    assert at.markdown[1].value == f"remembered={BELOW}"
    assert DEFAULT_LAYOUT == RIGHT, "the shared default is deliberately untouched"

    # And the page default is only a starting point — the control still wins.
    at.sidebar.radio(key="pl").set_value(RIGHT).run()
    assert at.markdown[0].value == f"selector={RIGHT}"


def test_an_unknown_page_default_falls_back_rather_than_propagating():
    """A typo in a viz config cannot put a page in a layout with no slot-maker."""
    at = AppTest.from_function(_page_default_script, kwargs={"default": "beneath"}).run()

    assert at.markdown[0].value == f"selector={DEFAULT_LAYOUT}"
    assert at.markdown[1].value == f"remembered={DEFAULT_LAYOUT}"


def _from_config_script(graph_params: dict) -> None:
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.panel_layout import panel_layout_from_config

    st.write(f"layout={panel_layout_from_config(graph_params)}")


def _from_config(graph_params: dict) -> AppTest:
    return AppTest.from_function(
        _from_config_script, kwargs={"graph_params": graph_params}, default_timeout=60
    ).run()


@pytest.mark.parametrize(
    ("graph_params", "expected"),
    [
        ({}, RIGHT),  # the default: clips beside the graph
        ({"video_panel": "below"}, BELOW),
        ({"video_panel": "Below"}, BELOW),  # case/space tolerant
        ({"video_panel": " right "}, RIGHT),
        (None, RIGHT),  # a session with no graph: block at all
    ],
)
def test_study_config_chooses_the_layout(graph_params, expected):
    at = _from_config(graph_params)

    assert at.markdown[0].value == f"layout={expected}"
    assert not at.warning


def test_a_typo_in_the_session_config_is_loud():
    at = _from_config({"video_panel": "besides"})

    # Falls back rather than crashing a run, but says so: this should be caught
    # while setting the session up, not inferred from the responses afterwards.
    assert at.markdown[0].value == f"layout={DEFAULT_LAYOUT}"
    assert "besides" in at.warning[0].value

    # No sidebar widget in the study apps — a participant cannot change it.
    assert not at.sidebar.radio


def _graph_script(layout: str | None) -> None:
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from behavior_graphs.core.graph import BehaviorGraph
    from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
    from behavior_graphs.viz.streamlit.components.panel_layout import panel_layout_selector
    from behavior_graphs.viz.streamlit.user_study.graph_explorer import render_graph_full_width

    # layout=None puts the demo's sidebar control in the loop, so one session
    # can switch layouts the way a user does.
    if layout is None:
        layout = panel_layout_selector(key="pl")

    seqs = [[0, 1, 2], [0, 2, 1], [0, 1, 2], [0, 2, 1]]
    outcomes = [True, False, True, True]
    episodes = episodes_from_lengths([len(s) for s in seqs], outcomes)
    windows = [
        WindowInfo(episode_idx=ep, start=t, end=t + 1)
        for ep, seq in enumerate(seqs)
        for t in range(len(seq))
    ]
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    window_set = WindowSet(
        features=np.zeros((len(windows), 2), dtype=np.float32),
        windows=windows,
        episodes=episodes,
    )
    metadata = [
        {"episode_idx": ep, "window_start": t, "window_end": t + 1, "success": outcomes[ep]}
        for ep, seq in enumerate(seqs)
        for t in range(len(seq))
    ]
    eps = [
        {"index": i, "path": f"ep{i}.mp4", "success": outcomes[i], "frame_count": 3, "fps": 10}
        for i in range(len(seqs))
    ]

    # Normally written by a click in the SVG component, which AppTest cannot
    # drive — seed it so a panel renders.
    if st.session_state.pop("select_node", True):
        st.session_state["t_graph_selected"] = 1

    render_graph_full_width(
        graph=BehaviorGraph.from_window_labels(labels, window_set),
        labels=labels,
        metadata=metadata,
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": eps, "by_index": {e["index"]: e for e in eps}},
        key_prefix="t",
        panel_layout=layout,
    )


def _run_graph(layout: str, *, select_node: bool = True) -> AppTest:
    at = AppTest.from_function(_graph_script, kwargs={"layout": layout}, default_timeout=60)
    at.session_state["select_node"] = select_node
    return at.run()


def _episode_captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption if c.value.startswith("Episode ")]


def test_node_panel_fits_three_clips_below_the_graph():
    at = _run_graph(BELOW)

    assert not at.exception
    assert _episode_captions(at) == [
        "Episode 0 — ✓ Success",
        "Episode 1 — ✗ Failure",
        "Episode 2 — ✓ Success",
    ]


def test_node_panel_shows_one_clip_beside_the_graph():
    at = _run_graph(RIGHT)

    assert not at.exception
    # Same node, same episodes — one clip per page at a third of the width.
    assert _episode_captions(at) == ["Episode 0 — ✓ Success"]


@pytest.mark.parametrize(
    ("layout", "where"),
    [(BELOW, "graph above"), (RIGHT, "graph to the left")],
)
def test_empty_state_points_at_the_graph_it_is_next_to(layout, where):
    at = _run_graph(layout, select_node=False)

    assert not at.exception
    # A caption, not st.info: nothing selected is the page's resting state, so the
    # prompt is written as a quiet line rather than a full-width alert.
    assert [c.value for c in at.caption] == [f"Click a node or edge in the {where} to explore it."]


def test_switching_layout_while_on_a_late_page_does_not_crash():
    """Clips-per-page changes at runtime, so a stored page index can go stale.

    Beside the graph, a node with 4 clips has 4 pages; below it, 2. Being on
    page 4 and switching used to slice to an empty page and reach
    `st.columns(0)` — an exception where the panel should be.
    """
    at = AppTest.from_function(_graph_script, kwargs={"layout": None}, default_timeout=90)
    at.session_state["select_node"] = True
    at.run()
    at.sidebar.radio(key="pl").set_value(RIGHT).run()
    at.session_state["t_vid_page_1"] = 3  # the 4th clip, one per page
    at.run()
    assert _episode_captions(at) == ["Episode 3 — ✓ Success"]

    at.sidebar.radio(key="pl").set_value(BELOW).run()  # now three per page

    assert not at.exception
    # Clamped to the last page that exists, showing its clips rather than none.
    assert at.session_state["t_vid_page_1"] == 1
    assert _episode_captions(at) == ["Episode 3 — ✓ Success"]


def test_tree_renders_its_selection_panel_in_either_layout():
    """The tree wraps a graph plus its own selection panel — both layouts run.

    Checked by selecting a node and looking for its clips: with nothing selected
    the panel draws nothing at all, so an empty page would otherwise pass.
    """
    for layout in (BELOW, RIGHT):
        at = AppTest.from_function(
            _tree_script, kwargs={"layout": layout}, default_timeout=90
        ).run()
        assert not at.exception, layout
        paths = {p: nid for p, nid in at.session_state["tt_path_to_id"].items() if p}
        at.session_state["tt_graph_selected"] = next(iter(paths.values()))
        at.run()
        assert not at.exception, layout
        assert [c for c in at.caption if c.value.startswith("Episode ")], layout


def _tree_script(layout: str) -> None:
    from pathlib import Path

    import numpy as np

    from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree

    seqs = [[0, 1, 2], [0, 2, 1], [0, 1, 2], [0, 2, 1]]
    outcomes = [True, False, True, True]
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    metadata = [
        {
            "episode_idx": ep,
            "rollout_idx": ep,
            "window_start": t * 10,
            "window_end": t * 10 + 10,
            "success": outcomes[ep],
        }
        for ep, seq in enumerate(seqs)
        for t in range(len(seq))
    ]
    eps = [
        {"index": i, "path": f"ep{i}.mp4", "success": outcomes[i], "frame_count": 30, "fps": 10}
        for i in range(len(seqs))
    ]
    render_trajectory_tree(
        labels=labels,
        metadata=metadata,
        view_mode="native_svg",
        min_branch=1,
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": eps, "by_index": {e["index"]: e for e in eps}},
        level="rollout",
        key_prefix="tt",
        panel_layout=layout,
    )


def test_the_graph_frame_can_be_given_extra_room():
    """`height_extra` adds to what the depth works out to, after the fit's own
    cap — room the page asked for, not room the tree happened to need."""
    import inspect

    from behavior_graphs.viz.streamlit.components import trajectory_tree_view as ttv

    assert "height_extra" in inspect.signature(ttv.render_trajectory_tree).parameters
    assert "_canvas_height(nodes_f, height) + height_extra" in inspect.getsource(
        ttv._render_native_svg
    )
