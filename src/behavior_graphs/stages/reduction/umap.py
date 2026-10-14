"""UMAP dimensionality reduction (requires the optional ``umap-learn`` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np

from behavior_graphs.core.stages import DimReduction
from behavior_graphs.stages.reduction.normalize import make_normalizer


def _import_umap() -> Any:
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAPReduction; install it with "
            "`pip install umap-learn` or `pip install behavior_graphs[umap]`"
        ) from e
    return umap


class UMAPReduction(DimReduction):
    """UMAP with optional pre-scaling.

    Args:
        n_components: Target dimensionality.
        n_neighbors: UMAP neighborhood size.
        min_dist: Minimum distance between embedded points.
        metric: Distance metric in the input space.
        normalize: Pre-UMAP scaling: ``"none"``, ``"standard"``, ``"minmax"``,
            or ``"robust"``.
        random_state: Seed (makes UMAP deterministic and single-threaded).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        normalize: str = "none",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.normalize = normalize
        self.random_state = random_state
        self._scaler = make_normalizer(normalize)
        self._umap: Any = None

    def fit(self, features: np.ndarray) -> UMAPReduction:
        umap = _import_umap()
        if self._scaler is not None:
            features = self._scaler.fit_transform(features)
        n_samples = features.shape[0]
        # n_neighbors must be < n_samples.
        n_neighbors = min(self.n_neighbors, n_samples - 1)
        # UMAP's spectral init calls scipy ARPACK with k = n_components + 1
        # (it skips the trivial eigenvector), so k >= N when
        # n_components >= n_samples - 1. Fall back to random init in that case.
        init = "spectral" if self.n_components < n_samples - 1 else "random"
        self._umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            init=init,
            random_state=self.random_state,
        )
        self._umap.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._umap is None:
            raise RuntimeError("UMAPReduction.transform called before fit")
        if self._scaler is not None:
            features = self._scaler.transform(features)
        return self._umap.transform(features).astype(np.float32)
