"""Markov behavior-graph construction with optional smoothing and pruning."""

from __future__ import annotations

import logging

import numpy as np

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.stages import GraphConstruction
from behavior_graphs.core.types import WindowSet
from behavior_graphs.stages.graph.simplify import (
    apply_merge_map,
    degree_one_prune_to_fixed_point,
    prune_edges_by_count,
)
from behavior_graphs.stages.graph.smoothing import (
    SMOOTHING_METHODS,
    cluster_centroids,
    smooth_labels,
)

logger = logging.getLogger(__name__)


class TransitionGraphBuilder(GraphConstruction):
    """Builds a :class:`BehaviorGraph` from per-window cluster labels.

    Steps: per-episode label smoothing, graph construction via
    :meth:`BehaviorGraph.from_window_labels`, optional degree-one node
    pruning (which relabels and rebuilds), and optional low-count edge
    pruning with probability renormalization.

    The input labels are never mutated. When smoothing or pruning relabels
    windows, the final labels are exposed as ``last_labels_`` after
    :meth:`build` (the pipeline runner saves the clustering-stage labels;
    consumers that need the graph-stage relabeling read this attribute).

    Args:
        smoothing: ``"none"``, ``"median"`` (per-episode mode filter), or
            ``"sticky"`` (DP decoder with a transition penalty).
        smoothing_window: Full window width for ``"median"`` smoothing.
        sticky_lambda: Transition penalty for ``"sticky"`` smoothing, in
            whatever feature space the caller put in ``windows.features`` —
            which has to be the space the labels' clusters were fitted in (see
            :func:`~behavior_graphs.stages.graph.smoothing.in_clustering_space`).
        degree_one_prune: Merge cluster nodes with a single in- or
            out-neighbor to a fixed point.
        min_edge_count: Drop transitions seen fewer than this many times and
            renormalize the remaining probabilities (0 disables).
    """

    def __init__(
        self,
        smoothing: str = "none",
        smoothing_window: int = 5,
        sticky_lambda: float = 2.0,
        degree_one_prune: bool = False,
        min_edge_count: int = 0,
    ) -> None:
        if smoothing not in SMOOTHING_METHODS:
            raise ValueError(f"smoothing must be one of {SMOOTHING_METHODS}, got {smoothing!r}")
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.sticky_lambda = sticky_lambda
        self.degree_one_prune = degree_one_prune
        self.min_edge_count = min_edge_count
        self.last_labels_: np.ndarray | None = None
        self._merge_map: dict[int, int] = {}
        self._sticky_centroids: dict[int, np.ndarray] | None = None

    def build(self, labels: np.ndarray, windows: WindowSet) -> BehaviorGraph:
        """Construct the graph; ``labels[i]`` corresponds to ``windows.windows[i]``."""
        labels = np.asarray(labels, dtype=np.int64).copy()
        if self.smoothing == "sticky":
            self._sticky_centroids = cluster_centroids(labels, windows.features)
        if self.smoothing != "none":
            labels = smooth_labels(
                labels,
                windows,
                self.smoothing,
                self.smoothing_window,
                self.sticky_lambda,
                centroids=self._sticky_centroids,
            )
        graph = BehaviorGraph.from_window_labels(labels, windows)
        self._merge_map = {}
        if self.degree_one_prune:
            graph, labels, _, _, self._merge_map = degree_one_prune_to_fixed_point(labels, windows)
        if self.min_edge_count > 0:
            graph = prune_edges_by_count(graph, self.min_edge_count)
        self.last_labels_ = labels
        return graph

    def relabel(self, labels: np.ndarray, windows: WindowSet) -> np.ndarray:
        """Map new raw cluster labels the same way :meth:`build` mapped train labels.

        Reapplies smoothing and then the degree-one-prune merge map learned
        from training. Smoothing is a pure function of ``(labels,
        windows)``, safe to rerun on new data, EXCEPT for ``"sticky"``'s
        cluster centroids: those are estimated from whatever data they're
        given, so relabeling reuses the centroids :meth:`build` computed
        from training data rather than re-deriving noisy ones from the
        (typically much smaller) new data. A raw id never seen during
        training (so absent from the merge map) passes through unchanged:
        there is nothing learned to remap it to. ``min_edge_count`` pruning
        only drops graph edges, not window labels, so it needs no
        relabeling step. Must be called after :meth:`build`.
        """
        if self.last_labels_ is None:
            raise RuntimeError("TransitionGraphBuilder.relabel called before build")
        labels = np.asarray(labels, dtype=np.int64).copy()
        if self.smoothing != "none":
            labels = smooth_labels(
                labels,
                windows,
                self.smoothing,
                self.smoothing_window,
                self.sticky_lambda,
                centroids=self._sticky_centroids,
            )
        if self._merge_map:
            labels = apply_merge_map(labels, self._merge_map)
        return labels
