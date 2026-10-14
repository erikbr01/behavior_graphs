"""Bridges between the legacy viz data model and the behavior_graphs APIs.

The ported visualization stack (apps + ``viz.streamlit`` + ``viz.plotly``)
was written against policy_doctor's *metadata list* convention: one dict per
window with keys ``rollout_idx``, ``window_start``, ``window_end`` and
``success``. Rather than rewriting every render function, the integration
layer converts once between that schema and :class:`WindowSet`:

- :func:`meta_from_windows` — WindowSet → legacy metadata list (load path).
- :func:`windowset_from_meta` — legacy metadata list → WindowSet (for code
  that re-derives graphs from possibly-relabeled windows).
- :func:`graph_from_labels_meta` — replacement for the old
  ``BehaviorGraph.from_cluster_assignments(labels, metadata, level=...)``
  (rollout level assumed throughout).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet

EP_KEY = "rollout_idx"


def meta_from_windows(windows: WindowSet) -> list[dict]:
    """Legacy per-window metadata list from a :class:`WindowSet`."""
    success_by_ep = {ep.index: ep.success for ep in windows.episodes}
    meta: list[dict] = []
    for w in windows.windows:
        meta.append(
            {
                EP_KEY: int(w.episode_idx),
                "window_start": int(w.start),
                "window_end": int(w.end),
                "success": success_by_ep.get(w.episode_idx),
            }
        )
    return meta


def windowset_from_meta(metadata: list[dict], features: np.ndarray | None = None) -> WindowSet:
    """Build a :class:`WindowSet` from a legacy metadata list.

    ``features`` defaults to a zero placeholder — sufficient for graph
    construction and the Markov test, which only read window/episode info.
    """
    windows: list[WindowInfo] = []
    ep_lengths: dict[int, int] = defaultdict(int)
    ep_success: dict[int, bool | None] = {}
    for m in metadata:
        ep = int(m.get(EP_KEY, m.get("demo_idx", 0)))
        start = int(m.get("window_start", m.get("timestep", 0)))
        end = int(m.get("window_end", start + 1))
        windows.append(WindowInfo(episode_idx=ep, start=start, end=end))
        ep_lengths[ep] = max(ep_lengths[ep], end)
        if ep not in ep_success and "success" in m:
            ep_success[ep] = m.get("success")

    episodes: list[EpisodeInfo] = []
    offset = 0
    for ep in sorted(ep_lengths):
        n = ep_lengths[ep]
        episodes.append(
            EpisodeInfo(
                index=ep,
                num_samples=n,
                sample_start_idx=offset,
                sample_end_idx=offset + n,
                success=ep_success.get(ep),
            )
        )
        offset += n

    if features is None:
        features = np.zeros((len(windows), 1), dtype=np.float32)
    return WindowSet(features=features, windows=windows, episodes=episodes)


def graph_from_labels_meta(
    labels: np.ndarray,
    metadata: list[dict],
    cluster_names: dict[int, str] | None = None,
) -> BehaviorGraph:
    """Drop-in for the old ``BehaviorGraph.from_cluster_assignments``."""
    return BehaviorGraph.from_window_labels(
        np.asarray(labels), windowset_from_meta(metadata), cluster_names=cluster_names
    )
