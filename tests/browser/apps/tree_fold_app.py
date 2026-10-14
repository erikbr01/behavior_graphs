"""A trajectory tree with fold badges, plus a knob that rebuilds it.

Companion to graph_selection_app.py for the other half of the reset: a +/- badge
click is remembered as a *pending* value in session state alongside an
"already handled" marker, and the tree re-applies the fold whenever those two
disagree. Clearing the marker without clearing the pending value makes the last
badge click replay against the new graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="tree fold smoke test", layout="wide")

from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)

KEY_PREFIX = "smoke"
# Two clusterings that branch, so every interior node carries a fold badge.
SEQS = {
    3: [[0, 1, 2], [0, 2, 1], [0, 1, 2], [0, 2, 1]],
    4: [[0, 1, 3], [0, 3, 1], [0, 1, 3], [0, 3, 1]],
}
OUTCOMES = [True, False, True, True]

k = st.radio("K", options=sorted(SEQS), horizontal=True, key="smoke_k")
seqs = SEQS[k]

labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
metadata = [
    {
        "episode_idx": ep,
        "rollout_idx": ep,
        "window_start": t * 10,
        "window_end": t * 10 + 10,
        "success": OUTCOMES[ep],
    }
    for ep, seq in enumerate(seqs)
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
        mp4_dir=Path("/nonexistent"),
        mp4_index=mp4_index,
        level="rollout",
        key_prefix=KEY_PREFIX,
        show_stats=False,
    )
    collapsed = st.session_state.get(f"{KEY_PREFIX}_manually_collapsed") or set()
    st.write(f"COLLAPSED={len(collapsed)}")
    # The two halves of the badge-click guard, published so a failure says
    # which one let a stale toggle through.
    st.write(f"TOGGLE={st.session_state.get(f'{KEY_PREFIX}_graph_fold_toggle')}")
    st.write(f"PROCESSED={st.session_state.get(f'{KEY_PREFIX}_toggle_last_processed')}")
    st.write(f"IDENTITY={st.session_state.get(f'{KEY_PREFIX}_graph_identity')}")


_view()
