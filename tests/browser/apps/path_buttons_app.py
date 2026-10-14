"""A trajectory tree whose paths carry sentence-length behavior names.

The "Top success/failure paths" buttons put one step per line, which only
holds up in a browser: a bare newline in a button label collapses to a space
and markdown list syntax renders literally, so the line breaks are markdown
hard breaks and the left alignment is CSS scoped to the block's container.
AppTest sees the label string either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="path buttons smoke test", layout="wide")

from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)

SEQS = [[0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 1, 3], [0, 3, 2]]
OUTCOMES = [True, True, False, True, False]
NAMES = {
    0: "picking up the ladle from the pot and placing it with its head on the towel",
    1: "picking up the carrot from the metal bowl and placing it in the pot",
    2: "picking up the corn from the metal bowl and placing it in the pot",
    3: "hovering the gripper over the chili on the top shelf",
}
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
mp4_index = {"fps": 10, "episodes": [], "by_index": {}}


@st.fragment
def _view() -> None:
    render_trajectory_tree(
        labels=labels,
        metadata=metadata,
        view_mode="native_svg",
        min_branch=1,
        cluster_names=NAMES,
        mp4_dir=Path("/nonexistent"),
        mp4_index=mp4_index,
        level="rollout",
        key_prefix="probe",
    )


_view()
