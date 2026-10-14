"""Clicking an outcome terminal opens on the behavior that led into it.

A ✗ FAILURE node used to be a dead end in two different ways: the Markov view's
node panel showed "no video clips for terminal nodes" at all, and the trajectory
tree's panel did show clips but opened them at frame 0 — so the answer to "what
went wrong in these seven episodes" was a full run's playback away.

Both now start on the last behavior run before the outcome, which is the span the
failure happens in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from behavior_graphs.core.graph import (  # noqa: E402
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths  # noqa: E402
from behavior_graphs.viz.streamlit.user_study.graph_explorer import (  # noqa: E402
    _OUTCOME_TERMINALS,
    _episodes_for_terminal,
)

#: Two failures and one success, each three behaviors long. Every window is 10
#: frames wide, so the last behavior of episode 0 starts at frame 20.
SEQS = [[0, 1, 2], [0, 1, 3], [0, 4, 5]]
OUTCOMES = [False, False, True]
WINDOW = 10


def _run():
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in SEQS])
    metadata = [
        {
            "rollout_idx": ep,
            "window_start": t * WINDOW,
            "window_end": (t + 1) * WINDOW,
            "success": OUTCOMES[ep],
        }
        for ep, seq in enumerate(SEQS)
        for t in range(len(seq))
    ]
    windows = WindowSet(
        features=np.zeros((len(metadata), 2), dtype=np.float32),
        windows=[
            WindowInfo(episode_idx=ep, start=t * WINDOW, end=(t + 1) * WINDOW)
            for ep, seq in enumerate(SEQS)
            for t in range(len(seq))
        ],
        episodes=episodes_from_lengths([len(s) for s in SEQS], OUTCOMES),
    )
    return labels, metadata, BehaviorGraph.from_window_labels(labels, windows)


def test_failure_terminal_opens_on_the_last_behavior():
    labels, metadata, graph = _run()

    starts = _episodes_for_terminal(graph.nodes[FAILURE_NODE_ID], labels, metadata)

    # Only the failures, each at the start of its third (last) behavior.
    assert starts == {0: 2 * WINDOW, 1: 2 * WINDOW}


def test_success_terminal_gets_the_same_treatment():
    labels, metadata, graph = _run()

    assert _episodes_for_terminal(graph.nodes[SUCCESS_NODE_ID], labels, metadata) == {2: 2 * WINDOW}


def test_start_is_not_an_outcome_terminal():
    """START carries every episode but is not a place any of them ended.

    So it stays the one special node with nothing to watch, while the panel now
    renders clips for the rest — see the ``is_outcome_terminal`` branch in
    ``_render_node_panel``.
    """
    _labels, _metadata, graph = _run()

    assert START_NODE_ID not in _OUTCOME_TERMINALS
    assert {SUCCESS_NODE_ID, FAILURE_NODE_ID} <= _OUTCOME_TERMINALS
    # And the graph really does mark it special, so it takes the START branch.
    assert graph.nodes[START_NODE_ID].is_special
