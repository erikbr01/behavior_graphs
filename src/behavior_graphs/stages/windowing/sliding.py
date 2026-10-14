"""Episode-aware sliding-window aggregation of per-timestep features."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.stages import Windowing
from behavior_graphs.core.types import FeatureSet, WindowInfo, WindowSet

_AGGREGATIONS = {
    "mean": np.mean,
    "sum": np.sum,
    "max": np.max,
}


class SlidingWindow(Windowing):
    """Aggregates consecutive timesteps into fixed-width windows per episode.

    Windows never cross episode boundaries. An episode shorter than ``width``
    yields one window spanning the whole episode. ``width=1, stride=1``
    degenerates to one window per timestep.

    Args:
        width: Number of timesteps per window.
        stride: Step between consecutive window starts.
        aggregation: How to pool timesteps within a window:
            ``"mean"``, ``"sum"``, or ``"max"``.
    """

    def __init__(self, width: int = 1, stride: int = 1, aggregation: str = "mean") -> None:
        if width < 1 or stride < 1:
            raise ValueError(f"width and stride must be >= 1, got {width}, {stride}")
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {sorted(_AGGREGATIONS)}, got {aggregation!r}"
            )
        self.width = width
        self.stride = stride
        self.aggregation = aggregation

    def transform(self, features: FeatureSet) -> WindowSet:
        """Aggregate per-timestep features into per-window features."""
        agg = _AGGREGATIONS[self.aggregation]
        rows = []
        windows = []
        for ep in features.episodes:
            ep_features = features.features[ep.sample_start_idx : ep.sample_end_idx]
            n = ep.num_samples
            if n == 0:
                continue
            if n < self.width:
                spans = [(0, n)]
            else:
                spans = [
                    (start, start + self.width)
                    for start in range(0, n - self.width + 1, self.stride)
                ]
            for start, end in spans:
                rows.append(agg(ep_features[start:end], axis=0))
                windows.append(WindowInfo(episode_idx=ep.index, start=start, end=end))
        return WindowSet(
            features=np.asarray(rows, dtype=np.float32),
            windows=windows,
            episodes=features.episodes,
        )
