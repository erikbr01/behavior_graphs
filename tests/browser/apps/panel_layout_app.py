"""A graph plus the sidebar layout switch, for measuring the graph's width.

Switching the video panel from beside the graph to below it changes the width
of the container the component is in without changing any of the component's
args — so Streamlit resizes the iframe and sends no render message. Only a real
browser can show whether the SVG followed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="panel layout width", layout="wide")

from behavior_graphs.core.graph import BehaviorGraph  # noqa: E402
from behavior_graphs.core.types import (  # noqa: E402
    WindowInfo,
    WindowSet,
    episodes_from_lengths,
)
from behavior_graphs.viz.streamlit.components.panel_layout import (  # noqa: E402
    panel_layout_selector,
)
from behavior_graphs.viz.streamlit.user_study.graph_explorer import (  # noqa: E402
    render_graph_full_width,
)

SEQS = [[0, 1, 2], [0, 2, 1], [0, 1, 2]]
OUTCOMES = [True, False, True]

episodes = episodes_from_lengths([len(s) for s in SEQS], OUTCOMES)
windows = [
    WindowInfo(episode_idx=ep, start=t, end=t + 1)
    for ep, seq in enumerate(SEQS)
    for t in range(len(seq))
]
labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in SEQS])
window_set = WindowSet(
    features=np.zeros((len(windows), 2), dtype=np.float32),
    windows=windows,
    episodes=episodes,
)
metadata = [
    {"episode_idx": ep, "window_start": t, "window_end": t + 1, "success": OUTCOMES[ep]}
    for ep, seq in enumerate(SEQS)
    for t in range(len(seq))
]

layout = panel_layout_selector(key="pl")


@st.fragment
def _view() -> None:
    render_graph_full_width(
        graph=BehaviorGraph.from_window_labels(labels, window_set),
        labels=labels,
        metadata=metadata,
        mp4_dir=Path("/nonexistent"),
        mp4_index={"fps": 10, "episodes": [], "by_index": {}},
        key_prefix="lw",
        panel_layout=layout,
    )


_view()
