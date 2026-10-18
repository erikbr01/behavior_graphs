"""K-Means clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from behavior_graphs.core.stages import Clustering


class KMeansClustering(Clustering):
    """K-Means over reduced window features.

    Both ``fit`` and ``predict`` run under ``threadpool_limits(1)``, which is
    the difference between a label array that depends only on the data and the
    seed and one that also depends on how many cores the machine offered.
    K-means computes its distances in parallel chunks, so the thread count sets
    the floating-point summation order and a window sitting near a cluster
    boundary can land on either side of it. Measured on the progress benchmark,
    same clone, same seed, libero_object:

        metric     published   threads auto   threads 1
        voc         0.965915      0.965863    0.965915
        kendall     0.934581      0.934449    0.934581

    One run at each thread count was internally consistent, so this was never
    noise -- it was a result that silently varied with the host. The limit
    belongs here rather than in each entry point because the stage is reached
    from the pipeline, the sweep runner, the apps and the tests, and because
    the thread count is not part of the stage cache key: without it a cached
    label array and a freshly computed one are different artifacts filed under
    the same key.

    The parallelism worth having is across jobs, not inside one k-means.

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
        with threadpool_limits(limits=1):
            self._model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("KMeansClustering.predict called before fit")
        with threadpool_limits(limits=1):
            return self._model.predict(features).astype(np.int64)
