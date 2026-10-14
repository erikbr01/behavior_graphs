"""Success-detection metric: AUROC of terminal progress vs the success flag."""

from __future__ import annotations

import numpy as np

from behavior_graphs.progress_estimation.estimation import success_auroc
from behavior_graphs.progress_estimation.metrics.base import ProgressMetric


class SuccessAUROC(ProgressMetric):
    """AUROC separating successful from failed episodes by terminal progress.

    Each episode is reduced to the mean of its last ``tail`` progress values
    (clipped to [0, 1]); the AUROC of that score against the success flag is the
    threshold-free success-detection number. NaN when outcomes are one-sided.
    """

    name = "auroc"
    higher_better = True

    def __init__(self, tail: int = 3):
        self.tail = tail

    def _terminal(self, curves: list[np.ndarray]) -> np.ndarray:
        return np.array(
            [
                float(np.mean(np.clip(np.asarray(c, float), 0, 1)[-self.tail :]))
                if len(c)
                else np.nan
                for c in curves
            ]
        )

    def compute(self, curves: list[np.ndarray], flags: list[bool]) -> float:
        fl = np.asarray(flags, dtype=bool)
        if not (fl.any() and (~fl).any()):
            return float("nan")
        return success_auroc(self._terminal(curves), fl)

    def per_episode(self, curve: np.ndarray, flag: bool) -> float:
        c = np.clip(np.asarray(curve, float), 0, 1)
        return float(np.mean(c[-self.tail :])) if len(c) else float("nan")
