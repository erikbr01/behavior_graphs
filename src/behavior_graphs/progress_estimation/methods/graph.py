"""Graph-derived progress methods: fit a behavior graph on the train split, then
read progress off the labels of held-out episodes.

The three methods (stage / absorb / frame-index) share a single
:class:`GraphBackbone` — the window -> reduction -> clustering -> BehaviorGraph
pipeline — which the runner instantiates once and injects into each method
(``needs_backbone``). ``GraphBackbone.fit`` is idempotent, so selecting both
graph-stage and graph-absorb fits the backbone only once per split.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.types import FeatureSet, episodes_from_lengths
from behavior_graphs.progress_estimation.estimation import (
    absorption_progress,
    episode_label_sequences,
    frame_index_progress,
    stage_progress,
    stage_schedule,
    success_absorption,
)
from behavior_graphs.progress_estimation.methods.base import ProgressMethod
from behavior_graphs.stages import IdentityReduction, KMeansClustering, SlidingWindow


class GraphBackbone:
    """Shared, fit-once behavior-graph backbone for the graph methods.

    ``reduction`` and ``clustering`` are instantiated stage objects (from config,
    reusing ``behavior_graphs.stages.*``); defaults reproduce the identity/K16
    settings of the original benchmark.
    """

    def __init__(
        self,
        window_width: int = 2,
        window_stride: int = 3,
        reduction=None,
        clustering=None,
    ):
        self.window_width = window_width
        self.window_stride = window_stride
        self.reduction = (
            reduction if reduction is not None else IdentityReduction(normalize="standard")
        )
        self.clustering = clustering if clustering is not None else KMeansClustering(n_clusters=16)
        self.fitted = False
        self.red = self.km = self.graph = None
        self.schedule = None
        self.absorb: dict[int, float] = {}

    def _windows(self, obs_list, flags_list):
        lens = [len(o) for o in obs_list]
        fs = FeatureSet(np.concatenate(obs_list), episodes_from_lengths(lens, flags_list))
        return SlidingWindow(width=self.window_width, stride=self.window_stride).transform(fs)

    def fit(self, train) -> GraphBackbone:
        if self.fitted:
            return self
        win = self._windows(train.embeddings(), train.flags)
        self.red = self.reduction.fit(win.features)
        red_feat = self.red.transform(win.features)
        self.km = self.clustering.fit(red_feat)
        lab = self.km.predict(red_feat)
        self.graph = BehaviorGraph.from_window_labels(lab, win)
        self.absorb = {c: v for c, v in success_absorption(self.graph).items() if c >= 0}
        self.schedule = stage_schedule(episode_label_sequences(lab, win), train.flags)
        self.fitted = True
        return self

    def label(self, view) -> list[np.ndarray]:
        """Per-episode window label sequences for an eval view (graph must be fit)."""
        win = self._windows(view.embeddings(), view.flags)
        lab = self.km.predict(self.red.transform(win.features))
        return episode_label_sequences(lab, win)

    def window_lengths(self, view) -> list[int]:
        """Per-episode window count (no fit needed) — for frame-index progress."""
        win = self._windows(view.embeddings(), view.flags)
        seqs = episode_label_sequences(np.zeros(len(win.windows), dtype=int), win)
        return [len(s) for s in seqs]


class _GraphBase(ProgressMethod):
    modality = "embedding"
    needs_backbone = True

    def __init__(self, name: str, backbone: GraphBackbone | None = None):
        self.name = name
        self.backbone = backbone

    def fit(self, train) -> None:
        self.backbone.fit(train)

    def reset(self) -> None:
        # shared backbone: first reset in a fold clears it, later ones are no-ops
        if self.backbone is not None:
            self.backbone.fitted = False


class GraphStage(_GraphBase):
    """SARM-style stage progress: cumulative stage base + linear ramp within a segment."""

    def __init__(self, name: str = "graph_stage", backbone=None, normalize_visited: bool = False):
        super().__init__(name, backbone)
        self.normalize_visited = normalize_visited

    def predict(self, eval) -> list[np.ndarray]:
        return [
            stage_progress(s, self.backbone.schedule, normalize_visited=self.normalize_visited)
            for s in self.backbone.label(eval)
        ]


class GraphAbsorb(_GraphBase):
    """Absorption progress: P(eventually reach SUCCESS | current cluster)."""

    def __init__(self, name: str = "graph_absorb", backbone=None):
        super().__init__(name, backbone)

    def predict(self, eval) -> list[np.ndarray]:
        return [absorption_progress(s, self.backbone.absorb) for s in self.backbone.label(eval)]


class FrameIndex(_GraphBase):
    """The baseline SARM argues against: progress = normalized window index.

    Shares the backbone only for window geometry — no graph fit is required.
    """

    def __init__(self, name: str = "frame_index", backbone=None):
        super().__init__(name, backbone)

    def fit(self, train) -> None:  # frame index needs no fitted graph
        pass

    def predict(self, eval) -> list[np.ndarray]:
        return [frame_index_progress(n) for n in self.backbone.window_lengths(eval)]
