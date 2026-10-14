"""GPU UMAP via RAPIDS cuML (requires the optional ``cuml-cu12`` extra).

A drop-in alternative to :class:`~behavior_graphs.stages.reduction.umap.UMAPReduction`
with the same constructor arguments and the same pre-scaling, so a sweep can name
either one and change nothing else.

**It is not numerically equivalent.** cuML implements UMAP's fuzzy simplicial set
and layout optimisation on device with its own kNN and its own SGD schedule, so
the embedding differs from ``umap-learn``'s on the same input and seed. Anything
comparing results across the two backends has to keep a CPU arm as a bridge
rather than assume the numbers transfer.

What it buys, measured on ABC bottles ``rgrl_cond`` (110,360 x 512 windows, one CV
fold, RTX 4090):

    n_components     cuML fit     umap-learn fit
    10                   2.7 s          ~1,114 s
    100                  6.4 s          ~1,114 s+

roughly 280-400x, and the transform onto a held-out fold is 0.3-0.6 s. The CPU
figure is per-fold, derived from a sweep combo that reduced six folds in 6,683 s.
The gap is that large because ``random_state`` forces ``umap-learn`` to run
single-threaded (it warns as much), which is the right reproducibility trade on
CPU and simply is not a constraint on device.

Determinism was verified rather than assumed: two fits at ``random_state=0`` on
the array above agree to ``atol=1e-4`` at both 10 and 100 components.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from behavior_graphs.core.stages import DimReduction
from behavior_graphs.stages.reduction.normalize import make_normalizer


def _import_cuml_umap() -> Any:
    try:
        from cuml.manifold import UMAP as cuUMAP
    except ImportError as e:
        raise ImportError(
            "cuml is required for CuMLUMAPReduction; install it with "
            "`uv pip install --extra-index-url https://pypi.nvidia.com cuml-cu12`"
        ) from e
    return cuUMAP


class CuMLUMAPReduction(DimReduction):
    """cuML UMAP with optional pre-scaling.

    Arguments mirror :class:`UMAPReduction` exactly so the two are substitutable.
    ``random_state`` is passed through; unlike the CPU backend it costs no
    parallelism to set it.
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

    def fit(self, features: np.ndarray) -> CuMLUMAPReduction:
        cuUMAP = _import_cuml_umap()
        if self._scaler is not None:
            features = self._scaler.fit_transform(features)
        features = np.ascontiguousarray(features, dtype=np.float32)
        n_samples = features.shape[0]
        n_neighbors = min(self.n_neighbors, max(2, n_samples - 1))
        # cuML has no ARPACK spectral-init failure mode to guard against (it
        # falls back internally), but n_components >= n_samples is still
        # meaningless, so refuse it the same way the CPU path effectively does.
        if self.n_components >= n_samples:
            raise ValueError(f"n_components={self.n_components} with only {n_samples} samples")
        self._umap = cuUMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        self._umap.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._umap is None:
            raise RuntimeError("CuMLUMAPReduction.transform called before fit")
        if self._scaler is not None:
            features = self._scaler.transform(features)
        features = np.ascontiguousarray(features, dtype=np.float32)
        # cuML returns a device array (cupy/cudf-backed); np.asarray pulls it
        # back to host so every downstream stage sees a plain ndarray.
        return np.asarray(self._umap.transform(features)).astype(np.float32)
