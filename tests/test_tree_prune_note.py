""" "17 nodes pruned." is a note about how the drawing was made.

It belongs with the rest of the stats: it is addressed to whoever chose the
``min_branch`` that pruned them. A study participant chose nothing and can act on
nothing, so to them it only says the graph in front of them is incomplete.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

#: Six episodes take behavior 1 and all succeed; four take behavior 2 and split
#: two-two on the outcome. At ``min_branch=3`` both of *those* leaves fall below
#: the threshold, which leaves behavior 2 reaching no terminal at all — so it is
#: pruned, and there is a note to make.
SEQS = [[0, 1]] * 6 + [[0, 2]] * 4
OUTCOMES = [True] * 6 + [True, True, False, False]
MIN_BRANCH = 3


def _tree_script() -> None:
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (
        render_trajectory_tree,
    )

    # Through session state, not module globals: AppTest ships the function's
    # own source to a scratch script, where this module's names do not exist.
    seqs = st.session_state["seqs"]
    outcomes = st.session_state["outcomes"]
    labels = np.concatenate([np.asarray(seq, dtype=np.int64) for seq in seqs])
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

    @st.fragment
    def _view() -> None:
        render_trajectory_tree(
            labels=labels,
            metadata=metadata,
            view_mode="native_svg",
            min_branch=st.session_state["min_branch"],
            level="rollout",
            key_prefix="pn",
            show_stats=st.session_state["show_stats"],
            show_paths=False,
            mp4_dir=Path("/nonexistent"),
        )

    _view()


def _captions(*, show_stats: bool) -> str:
    at = AppTest.from_function(_tree_script, default_timeout=30)
    at.session_state["seqs"] = SEQS
    at.session_state["outcomes"] = OUTCOMES
    at.session_state["min_branch"] = MIN_BRANCH
    at.session_state["show_stats"] = show_stats
    at.run()
    assert not at.exception, at.exception
    return "\n".join(element.value for element in at.caption)


def test_the_note_is_there_for_whoever_set_the_threshold():
    assert "nodes pruned" in _captions(show_stats=True)


def test_the_study_does_not_tell_a_participant_the_graph_was_pruned():
    assert "nodes pruned" not in _captions(show_stats=False)
