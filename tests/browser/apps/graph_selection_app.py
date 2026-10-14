"""A behavior graph plus a knob that rebuilds it, for the browser smoke test.

Deliberately synthetic and self-contained: no run dir, no clips, no config. The
point is the real SVG component in a real browser, so that a click, the value
the component keeps re-sending afterwards, and the selection reset are all
exercised together — the part no AppTest can reach.

The "K" radio stands in for any hyperparameter that rebuilds the graph: it
relabels the same episodes, which is what the app sees when you pick a
different K in the sidebar and a new run loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="graph selection smoke test", layout="wide")

from behavior_graphs.core.graph import BehaviorGraph  # noqa: E402
from behavior_graphs.core.types import (  # noqa: E402
    WindowInfo,
    WindowSet,
    episodes_from_lengths,
)
from behavior_graphs.viz.streamlit.user_study.graph_explorer import (  # noqa: E402
    render_graph_full_width,
)

KEY_PREFIX = "smoke"
# Distinct behaviors per K so the two graphs share node 1 — the id a replayed
# click would land on, which is exactly the case worth catching.
SEQS = {3: [[0, 1, 2], [0, 2, 1], [0, 1, 2]], 4: [[0, 1, 3], [0, 3, 1], [0, 1, 3]]}
OUTCOMES = [True, False, True]

k = st.radio("K", options=sorted(SEQS), horizontal=True, key="smoke_k")
seqs = SEQS[k]

episodes = episodes_from_lengths([len(s) for s in seqs], OUTCOMES)
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
    {"episode_idx": ep, "window_start": t, "window_end": t + 1, "success": OUTCOMES[ep]}
    for ep, seq in enumerate(seqs)
    for t in range(len(seq))
]
mp4_index = {"fps": 10, "episodes": [], "by_index": {}}

graph = BehaviorGraph.from_window_labels(labels, window_set)


# Inside a fragment, as both real apps wrap their graph: the component's click
# path calls st.rerun(scope="fragment"), which raises anywhere else. The K radio
# stays outside it, so changing K re-runs the whole script the way a new run
# loading does.
@st.fragment
def _view() -> None:
    render_graph_full_width(
        graph=graph,
        labels=labels,
        metadata=metadata,
        mp4_dir=Path("/nonexistent"),
        mp4_index=mp4_index,
        key_prefix=KEY_PREFIX,
    )
    # The assertion surface: session state is not reachable from the browser, so
    # publish what the app thinks is selected.
    st.write(f"SELECTED={st.session_state.get(f'{KEY_PREFIX}_graph_selected')}")
    st.write(f"EDGE={st.session_state.get(f'{KEY_PREFIX}_graph_selected_edge')}")


_view()
