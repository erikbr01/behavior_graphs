"""Container for all intermediate and final outputs of one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.subclusters import SubClusterSet
from behavior_graphs.core.types import FeatureSet, WindowSet


@dataclass
class PipelineResult:
    """Accumulates stage outputs as the pipeline progresses.

    Evaluators receive this object and may consume any subset of it; fields
    not yet produced are ``None``.
    """

    features: FeatureSet | None = None
    windows: WindowSet | None = None
    reduced: np.ndarray | None = None  # (n_windows, n_components)
    coords_2d: np.ndarray | None = None  # (n_windows, 2) for visualization
    labels: np.ndarray | None = None  # (n_windows,) int, -1 = noise
    graph: BehaviorGraph | None = None
    node_values: dict[int, float] | None = None
    #: Set only when the optional subclustering stage ran (cfg.subclustering).
    subclusters: SubClusterSet | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
