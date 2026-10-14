"""A trajectory tree plus the layout switch, for measuring the tree's width.

The tree renders its own selection panel, so it makes the two layout slots
itself and asks the graph renderer to suppress the panel. If the renderer split
the row *again* the tree would draw in two thirds of its slot and leave the rest
blank — a full-width layout that isn't. Only a browser can measure that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="tree layout width", layout="wide")

from behavior_graphs.viz.streamlit.components.panel_layout import (  # noqa: E402
    panel_layout_selector,
)
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)

SEQS = [[0, 1, 2]] * 6 + [[0, 3, 4]] * 2 + [[0, 5, 6]]
OUTCOMES = [True] * 6 + [False, True, False]
NAMES = {0: "Approach", 1: "Grasp", 2: "Place", 3: "Rotate", 4: "Nudge", 5: "Miss", 6: "Retract"}

labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in SEQS])
metadata = [
    {
        "episode_idx": ep,
        "rollout_idx": ep,
        "window_start": t * 10,
        "window_end": t * 10 + 10,
        "success": OUTCOMES[ep],
    }
    for ep, seq in enumerate(SEQS)
    for t in range(len(seq))
]

layout = panel_layout_selector(key="tl")


@st.fragment
def _view() -> None:
    render_trajectory_tree(
        labels=labels,
        metadata=metadata,
        view_mode="native_svg",
        min_branch=1,
        cluster_names=NAMES,
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": [], "by_index": {}},
        level="rollout",
        key_prefix="tl",
        show_stats=False,
        panel_layout=layout,
    )


_view()
