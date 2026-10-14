"""Ranking / regression progress metrics (TOPReward + ROBOMETER convention).

Each wraps the shared kernels in :mod:`behavior_graphs.progress_estimation.estimation`
(``progress_regression_metrics`` for the set-level mean over successful episodes,
``per_episode_progress_metrics`` for the per-episode value), so the class-based
selectable metrics are numerically identical to the original bundled report.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.progress_estimation.estimation import (
    per_episode_progress_metrics,
    progress_regression_metrics,
)
from behavior_graphs.progress_estimation.metrics.base import ProgressMetric


class _RankingMetric(ProgressMetric):
    key: str = ""

    def compute(self, curves: list[np.ndarray], flags: list[bool]) -> float:
        return progress_regression_metrics(list(curves), list(flags))[self.key]

    def per_episode(self, curve: np.ndarray, flag: bool) -> float:
        if not flag or len(curve) < 3:
            return float("nan")
        return per_episode_progress_metrics(np.asarray(curve, float))[self.key]


class VOC(_RankingMetric):
    """Spearman rank correlation of predicted progress vs frame order."""

    name = "voc"
    key = "voc"
    higher_better = True


class KendallTau(_RankingMetric):
    """Kendall tau-b of predicted progress vs frame order."""

    name = "kendall"
    key = "kendall"
    higher_better = True


class Pearson(_RankingMetric):
    """Pearson correlation of predicted progress vs the linear GT ramp."""

    name = "pearson"
    key = "pearson"
    higher_better = True


class MAE(_RankingMetric):
    """Mean absolute error vs the linear GT ramp (both clipped to [0, 1])."""

    name = "mae"
    key = "mae"
    higher_better = False
