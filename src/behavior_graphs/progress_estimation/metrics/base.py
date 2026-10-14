"""Base class for progress-estimation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ProgressMetric(ABC):
    """One paper-comparable progress metric.

    ``compute`` reduces a set of per-episode progress curves + success flags to a
    single scalar (the number reported in the benchmark table). ``per_episode``
    returns the per-episode statistic used for paired bootstrap CIs; it defaults to
    NaN for metrics that are only defined set-wide.
    """

    name: str = ""
    higher_better: bool = True

    @abstractmethod
    def compute(self, curves: list[np.ndarray], flags: list[bool]) -> float: ...

    def per_episode(self, curve: np.ndarray, flag: bool) -> float:
        return float("nan")
