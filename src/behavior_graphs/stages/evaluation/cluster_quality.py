"""Intrinsic cluster-quality metrics on the reduced window features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator

logger = logging.getLogger(__name__)


def _capped_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    # silhouette_score is O(n²); cap samples to keep it tractable.
    if len(features) > 5_000:
        return silhouette_score(features, labels, sample_size=5_000)
    return silhouette_score(features, labels)


# Every metric is callable as fn(features, labels); per-metric quirks (like
# the silhouette sampling cap) live here, not at the call site.
_METRICS = {
    "silhouette": _capped_silhouette,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}


class ClusterQualityEvaluator(Evaluator):
    """Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    Noise windows (label ``-1``) are excluded; metrics that need at least two
    clusters are skipped gracefully and never emit NaN.
    """

    name = "cluster_quality"

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        labels = np.asarray(result.labels)
        mask = labels != -1
        out: dict[str, Any] = {
            "n_clusters": int(len(np.unique(labels[mask]))) if mask.any() else 0,
            "noise_fraction": float((~mask).mean()),
        }
        features = result.reduced[mask]
        kept = labels[mask]
        if out["n_clusters"] < 2:
            logger.info("cluster_quality: fewer than 2 clusters, skipping scores")
            return out
        for name, fn in _METRICS.items():
            try:
                value = float(fn(features, kept))
            except ValueError as e:
                logger.warning("cluster_quality: %s failed: %s", name, e)
                continue
            if np.isfinite(value):
                out[name] = value
        return out
