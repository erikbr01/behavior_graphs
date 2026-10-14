"""Tests for dropping a graph selection when the graph itself is replaced.

The bug this guards: pick a node, change K, and the freshly loaded graph came up
greyed out around a node id belonging to the clustering you just left. The other
half matters just as much — folding a branch or pruning changes which nodes are
drawn without being a new graph, and must keep the selection.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.components.graph_selection import (  # noqa: E402
    _SELECTION_SUFFIXES,
    graph_identity,
    labels_identity,
)

_META = [
    {"episode_idx": ep, "window_start": t, "window_end": t + 1, "success": True}
    for ep in range(2)
    for t in range(3)
]


def test_labels_identity_tracks_the_clustering_not_the_view():
    a = np.array([0, 1, 2, 0, 2, 1], dtype=np.int64)
    b = np.array([0, 1, 3, 0, 3, 1], dtype=np.int64)

    assert labels_identity(a, _META) == labels_identity(a.copy(), _META)
    assert labels_identity(a, _META) != labels_identity(b, _META)


def test_reset_keeps_the_already_handled_markers():
    """The dedup markers are not selections and must survive the reset.

    The SVG component re-sends its last click on every rerun; `graph_last_seq`
    and the `*_last_processed` pair are how that gets ignored. Clearing one
    makes the click before the reset look new — the dropped node re-selects
    itself, or the last +/- badge click is re-applied through the previous
    graph's id map, collapsing a branch nobody touched.
    """
    for marker in ("graph_last_seq", "toggle_last_processed", "expand_last_processed"):
        assert marker not in _SELECTION_SUFFIXES

    # What the reset does own: the selection itself and the fold sets, whose
    # keys are cluster-id paths and mean something else after a relabel.
    assert "graph_selected" in _SELECTION_SUFFIXES
    assert "manually_collapsed" in _SELECTION_SUFFIXES
    assert "expanded_buckets" in _SELECTION_SUFFIXES


def test_graph_identity_also_tracks_the_node_set():
    labels = np.array([0, 1, 2, 0, 2, 1], dtype=np.int64)

    same = graph_identity(labels, _META, [0, 1, 2])
    assert same == graph_identity(labels, _META, [2, 1, 0])  # order-insensitive
    # State merging rebuilds node ids from unchanged labels — a different graph.
    assert same != graph_identity(labels, _META, [0, 1, 2, 7])


def _graph_script() -> None:
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from behavior_graphs.core.graph import BehaviorGraph
    from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
    from behavior_graphs.viz.streamlit.user_study.graph_explorer import render_graph_full_width

    # Stands in for the sidebar's K selector: a different clustering of the
    # same episodes, which in the app means a different run and a new graph.
    k = st.radio("K", options=[3, 4], key="k")
    seqs = [[0, 1, 2], [0, 2, 1], [0, 1, 2]] if k == 3 else [[0, 1, 3], [0, 3, 1], [0, 1, 3]]
    outcomes = [True, False, True]
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
    render_graph_full_width(
        graph=BehaviorGraph.from_window_labels(labels, window_set),
        labels=labels,
        metadata=metadata,
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": eps, "by_index": {e["index"]: e for e in eps}},
        key_prefix="t",
        # min_edge_count stands in for the pruning slider: a view knob, not a
        # new graph.
        min_edge_count=st.slider("prune", 0, 5, 0, key="prune"),
    )


def _selected(at: AppTest):
    return at.session_state["t_graph_selected"] if "t_graph_selected" in at.session_state else None


def _prompt(at: AppTest) -> bool:
    """Whether the panel is showing its "nothing selected" prompt.

    A caption, not st.info: the empty state is where the page rests, so it is
    written as a quiet line rather than an alert.
    """
    return any("Click a node or edge" in c.value for c in at.caption)


def _select_node(at: AppTest, node_id: int) -> AppTest:
    """Stand in for a click in the SVG component, which AppTest cannot drive."""
    at.session_state["t_graph_selected"] = node_id
    return at.run()


def test_selection_is_dropped_when_the_clustering_changes():
    at = _select_node(AppTest.from_function(_graph_script, default_timeout=90).run(), 1)
    assert not _prompt(at)  # the node panel is open
    token_key = "t_graph_render_token"
    token_before = at.session_state[token_key] if token_key in at.session_state else 0

    at.radio(key="k").set_value(4).run()

    assert _selected(at) is None
    assert _prompt(at)
    # Bumped so the iframe, which holds its own copy of the selection, repaints
    # instead of staying greyed out around the node that is gone.
    assert at.session_state[token_key] > token_before


def test_selection_does_not_come_back_when_the_old_clustering_does():
    at = _select_node(AppTest.from_function(_graph_script, default_timeout=90).run(), 1)
    at.radio(key="k").set_value(4).run()
    at.radio(key="k").set_value(3).run()

    assert _selected(at) is None


def test_pruning_keeps_the_selection():
    at = _select_node(AppTest.from_function(_graph_script, default_timeout=90).run(), 1)

    at.slider(key="prune").set_value(3).run()

    # Same labels, same graph — only what is drawn changed.
    assert _selected(at) == 1
    assert not _prompt(at)


def _tree_script() -> None:
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree

    k = st.radio("K", options=[3, 4], key="k")
    seqs = [[0, 1, 2], [0, 2, 1], [0, 1, 2]] if k == 3 else [[0, 1, 3], [0, 3, 1], [0, 1, 3]]
    outcomes = [True, False, True]
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
    )


def _tree_selected(at: AppTest, path_ids: dict) -> AppTest:
    """Select one of the tree's synthetic nodes, as a node click would."""
    at.session_state["tt_graph_selected"] = next(iter(path_ids.values()))
    return at.run()


def test_tree_selection_survives_a_fold_but_not_a_relabel():
    at = AppTest.from_function(_tree_script, default_timeout=120).run()
    path_ids = {p: nid for p, nid in at.session_state["tt_path_to_id"].items() if p}
    at = _tree_selected(at, path_ids)
    picked = at.session_state["tt_graph_selected"]
    assert picked is not None

    # Folding a branch changes which synthetic ids exist without changing the
    # clustering — the tree is named by its labels precisely so this survives.
    at.session_state["tt_manually_collapsed"] = {next(iter(path_ids))}
    at.run()
    assert at.session_state["tt_graph_selected"] == picked

    at.radio(key="k").set_value(4).run()
    assert "tt_graph_selected" not in at.session_state
    # And the panel is empty rather than still showing the dropped node's clips
    # (it used to be checked through the "click a node" prompt, which is gone).
    assert not [c for c in at.caption if c.value.startswith("Episode ")]
