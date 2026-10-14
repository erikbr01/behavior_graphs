"""Tests for the "Top success / failure paths" block under the graph.

The list is a second way to read the same graph, so a caller (the study, for
now) may want it off. What must not happen is the switch also taking the
node-count line with it, or leaving the buttons behind while hiding the heading.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_SCRIPT = """
from pathlib import Path
import numpy as np
import streamlit as st
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree

SEQS = [[0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 1, 3]]
OUTCOMES = [True, True, False, False]
labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in SEQS])
metadata = [
    {"episode_idx": ep, "rollout_idx": ep, "window_start": t * 10,
     "window_end": t * 10 + 10, "success": OUTCOMES[ep]}
    for ep, seq in enumerate(SEQS) for t in range(len(seq))
]
render_trajectory_tree(
    labels=labels, metadata=metadata, view_mode="native_svg", min_branch=1,
    mp4_dir=Path("/nonexistent"), mp4_index={"fps": 10, "episodes": [], "by_index": {}},
    level="rollout", key_prefix="t", show_paths=SHOW_PATHS,
)
"""


def _run(show_paths: bool) -> AppTest:
    at = AppTest.from_string(_SCRIPT.replace("SHOW_PATHS", str(show_paths)), default_timeout=30)
    return at.run()


def test_the_paths_block_is_there_by_default():
    at = _run(True)

    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    assert "Top success paths" in body and "Top failure paths" in body
    assert [b for b in at.button if "%" in b.label]


def test_hiding_it_takes_the_buttons_with_it():
    at = _run(False)

    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    assert "Top success paths" not in body and "Top failure paths" not in body
    assert not [b for b in at.button if "%" in b.label]


def test_the_node_count_line_is_gone():
    """ "N of M graph nodes shown" told a participant about the drawing's own
    pruning, which is not a fact about the robot and not theirs to act on."""
    for show_paths in (True, False):
        at = _run(show_paths)
        assert not [c for c in at.caption if "nodes shown" in c.value], show_paths


def test_the_study_variants_have_it_off_for_now():
    from behavior_graphs.viz.streamlit.user_studies.registry import list_variants

    for variant in list_variants():
        if variant.id.startswith("contrastive_"):
            assert variant.graph_params.get("show_paths") is False, variant.id
