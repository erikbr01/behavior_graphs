"""Pass-through dimensionality reduction (optional normalization only)."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.stages import DimReduction
from behavior_graphs.stages.reduction.normalize import make_normalizer


class IdentityReduction(DimReduction):
    """Keeps features unchanged, optionally applying a fitted scaler.

    Args:
        normalize: ``"none"``, ``"standard"``, ``"minmax"``, or ``"robust"``.
    """

    def __init__(self, normalize: str = "none") -> None:
        self.normalize = normalize
        self._scaler = make_normalizer(normalize)

    def fit(self, features: np.ndarray) -> IdentityReduction:
        if self._scaler is not None:
            self._scaler.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            features = self._scaler.transform(features)
        return np.asarray(features, dtype=np.float32)
