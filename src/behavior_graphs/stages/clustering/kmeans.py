"""K-Means clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from behavior_graphs.core.stages import Clustering


class KMeansClustering(Clustering):
    """K-Means over reduced window features.

    Args:
        n_clusters: Number of clusters.
        n_init: Number of k-means++ restarts.
        random_state: Seed for centroid initialization.
    """

    def __init__(self, n_clusters: int = 10, n_init: int = 10, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self._model: KMeans | None = None

    def fit(self, features: np.ndarray) -> KMeansClustering:
        self._model = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("KMeansClustering.predict called before fit")
        return self._model.predict(features).astype(np.int64)
