"""PCA dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from behavior_graphs.core.stages import DimReduction
from behavior_graphs.stages.reduction.normalize import make_normalizer


class PCAReduction(DimReduction):
    """PCA with optional pre-scaling.

    ``n_components`` is clamped at fit time to ``min(n_components, n_samples,
    n_features)`` so small inputs never fail.

    Args:
        n_components: Target dimensionality.
        normalize: Pre-PCA scaling: ``"none"``, ``"standard"``, ``"minmax"``,
            or ``"robust"``.
        whiten: Whether to whiten the principal components.
        random_state: Seed for randomized solvers.
    """

    def __init__(
        self,
        n_components: int = 32,
        normalize: str = "none",
        whiten: bool = False,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.normalize = normalize
        self.whiten = whiten
        self.random_state = random_state
        self._scaler = make_normalizer(normalize)
        self._pca: PCA | None = None

    def fit(self, features: np.ndarray) -> PCAReduction:
        if self._scaler is not None:
            features = self._scaler.fit_transform(features)
        # n_components must be strictly less than n_samples for scipy's sparse
        # eigensolver (k >= N raises TypeError). Also cap at n_features.
        n_components = min(self.n_components, features.shape[0] - 1, features.shape[1])
        self._pca = PCA(
            n_components=n_components, whiten=self.whiten, random_state=self.random_state
        )
        self._pca.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReduction.transform called before fit")
        if self._scaler is not None:
            features = self._scaler.transform(features)
        return self._pca.transform(features).astype(np.float32)
