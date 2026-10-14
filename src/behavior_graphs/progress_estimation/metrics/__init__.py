"""Selectable progress-estimation metrics (one class per metric).

All ranking metrics wrap the shared kernels in
:mod:`behavior_graphs.progress_estimation.estimation`, so results match the original
bundled report exactly; :class:`SuccessAUROC` is the success-detection metric.
"""

from behavior_graphs.progress_estimation.metrics.base import ProgressMetric
from behavior_graphs.progress_estimation.metrics.detection import SuccessAUROC
from behavior_graphs.progress_estimation.metrics.ranking import (
    MAE,
    VOC,
    KendallTau,
    Pearson,
)

__all__ = [
    "ProgressMetric",
    "VOC",
    "KendallTau",
    "Pearson",
    "MAE",
    "SuccessAUROC",
]
