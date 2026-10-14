"""A branch point whose siblings' labels compete for horizontal room.

Three runs of the same episodes, so one page covers what the width logic has to
get right:

* ``long`` — long names, but with spaces: they wrap onto a second line, and the
  wrapped tree fits the frame.
* ``unbreakable`` — the same names as single tokens, so wrapping cannot help.
  Nothing fits, and the frame's width is redistributed rather than grown: the
  graph still draws at frame width and the crowding shows.
* ``short`` — terse names that fit with room to spare and must not buy width.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="wide labels", layout="wide")

from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)

SEQS = [[0, 1, 2]] * 12 + [[0, 3, 4]] * 2 + [[0, 5, 6]] + [[0, 7, 8]] + [[0, 9, 2]]
OUTCOMES = [True] * 12 + [False, True, False, False, True]
NAMES = {
    0: "Approach the shelf",
    1: "Grasp the milk carton firmly",
    2: "Place it beside the plate",
    3: "Rotate the wrist outward",
    4: "Nudge the carton over",
    5: "Miss the carton entirely",
    6: "Retract without the carton",
    7: "Knock the carton sideways",
    8: "Recover and re-approach",
    9: "Grasp the carton by its lid",
}
#: The same names with the spaces closed up, so no wrapping is possible.
UNBREAKABLE_NAMES = {cid: name.replace(" ", "_") for cid, name in NAMES.items()}
#: A third run with terse names, for the fits-easily contrast case.
SHORT_NAMES = {cid: f"B{cid}" for cid in NAMES}
NAMES_BY_MODE = {"long": NAMES, "unbreakable": UNBREAKABLE_NAMES, "short": SHORT_NAMES}

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

mode = st.radio("Labels", options=list(NAMES_BY_MODE), horizontal=True, key="labels")


@st.fragment
def _view() -> None:
    render_trajectory_tree(
        labels=labels,
        metadata=metadata,
        view_mode="native_svg",
        min_branch=1,
        cluster_names=NAMES_BY_MODE[mode],
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": [], "by_index": {}},
        level="rollout",
        key_prefix="wide",
        show_stats=False,
    )


_view()
