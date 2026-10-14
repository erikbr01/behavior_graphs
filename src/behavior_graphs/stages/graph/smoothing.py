"""Per-episode label-sequence smoothing.

Both smoothers are pure functions over one episode's time-ordered label array;
they reduce the A→B→A→B flicker that survives run-length collapse when
building the behavior graph. Ported from policy_doctor's
``behavior_graph.graph_simplification``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from behavior_graphs.core.types import WindowSet

SMOOTHING_METHODS = ("none", "median", "sticky")


def in_clustering_space(windows: WindowSet, reduced: np.ndarray | None) -> WindowSet:
    """``windows`` carrying the features the clustering was actually fitted on.

    The pipeline reduces (UMAP, median 10 dims) and then clusters the *reduced*
    array, but a ``WindowSet`` keeps the raw features (median 512 dims). Anything
    that measures distance to a cluster centroid — sticky-decode smoothing above
    all — has to do it where the clusters are compact, and after UMAP that is not
    the raw space: UMAP does not preserve distances, so a KMeans cluster found in
    the embedding is an arbitrary shape in the original features.

    Measured over the 176 best-per-K runs in ``sweep_all``, this one substitution
    is the difference between the smoother working and not. At λ=1 it relabels
    0.4% of windows and shrinks the trajectory tree by a median 11%, helping
    119 runs and hurting 9; the same call on raw features relabels 13.7% and
    *grows* the tree by a median 56%, hurting 117 of 176. Given each run its own
    best λ in this space, the median tree is 29% smaller and only 2 of 176 runs
    come out bigger. Raw-space "smoothing" at a low λ is not smoothing at all —
    it re-assigns an eighth of the windows by a geometry the clusters were never
    defined in, and the prefix tree charges for every path that diverges as a
    result.

    ``reduced`` of None means the run has no reduction stage (or predates the
    artifact), and raw features are the clustering space; ``windows`` is then
    returned unchanged.
    """
    if reduced is None:
        return windows
    features = np.asarray(reduced, dtype=np.float32)
    if features.shape[0] != windows.num_windows:
        raise ValueError(
            f"reduced has {features.shape[0]} rows but there are {windows.num_windows} windows"
        )
    return replace(windows, features=features)


def cluster_centroids(labels: np.ndarray, features: np.ndarray) -> dict[int, np.ndarray]:
    """Mean feature vector per cluster id (noise ``-1`` is ignored)."""
    out: dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        out[int(c)] = features[labels == c].mean(axis=0)
    return out


def median_filter(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Mode filter over a sliding window on one episode's label sequence.

    Each position is replaced by the most frequent non-noise label in
    ``[t - window//2, t + window//2]``; positions whose window contains only
    noise keep their original label.

    Args:
        labels: (T,) int labels of one episode, in time order.
        window: Full filter width (odd values are symmetric).

    Returns:
        New (T,) label array; the input is not modified.
    """
    out = labels.copy()
    if window <= 1:
        return out
    half = window // 2
    for t in range(len(labels)):
        lo = max(0, t - half)
        hi = min(len(labels), t + half + 1)
        vals = labels[lo:hi]
        vals = vals[vals != -1]
        if len(vals) == 0:
            continue
        uniq, counts = np.unique(vals, return_counts=True)
        out[t] = int(uniq[np.argmax(counts)])
    return out


def sticky_decode(
    labels: np.ndarray,
    features: np.ndarray,
    centroids: dict[int, np.ndarray],
    lambda_stick: float = 2.0,
) -> np.ndarray:
    """Sticky DP decoder for one episode: min Σ d(x_t, μ_{z_t}) + λ·1[z_t ≠ z_{t-1}].

    Re-assigns every window to a cluster using the squared distance of its
    feature vector to each centroid plus a flat penalty ``lambda_stick`` per
    transition. Higher λ → fewer transitions, longer runs. The decoder can
    only re-assign to existing centroids; it cannot invent new clusters.

    Args:
        labels: (T,) int labels of one episode, in time order.
        features: (T, d) feature vectors aligned with ``labels``.
        centroids: Cluster id → (d,) centroid (e.g. from
            :func:`cluster_centroids` over all episodes).
        lambda_stick: Transition penalty, on the scale of normalized unary costs.

    Returns:
        New (T,) label array; the input is not modified.
    """
    if not centroids or len(labels) == 0:
        return labels.copy()
    cluster_ids = sorted(centroids)
    C = np.stack([centroids[c] for c in cluster_ids])  # (K, d)
    K = len(cluster_ids)

    diffs = features[:, None, :] - C[None, :, :]
    unary = np.sum(diffs * diffs, axis=-1)  # (T, K)
    unary = unary / float(unary.std() + 1e-6)

    T = len(labels)
    dp = np.empty((T, K))
    bp = np.empty((T, K), dtype=np.int64)
    dp[0] = unary[0]
    bp[0] = -1
    arange = np.arange(K)
    for t in range(1, T):
        prev = dp[t - 1]
        order = np.argsort(prev)
        best, second = order[0], order[1] if K > 1 else order[0]
        other_idx = np.where(arange == best, second, best)
        other_cost = prev[other_idx] + lambda_stick
        choose_same = prev <= other_cost
        dp[t] = unary[t] + np.where(choose_same, prev, other_cost)
        bp[t] = np.where(choose_same, arange, other_idx)

    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return np.array([cluster_ids[k] for k in path], dtype=labels.dtype)


def episode_order(windows: WindowSet) -> dict[int, list[int]]:
    """Window indices per episode, in time order."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, w in enumerate(windows.windows):
        groups[w.episode_idx].append(i)
    return {ep: sorted(idxs, key=lambda i: windows.windows[i].start) for ep, idxs in groups.items()}


def smooth_labels(
    labels: np.ndarray,
    windows: WindowSet,
    method: str = "none",
    smoothing_window: int = 5,
    sticky_lambda: float = 2.0,
    centroids: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply per-episode smoothing across a whole (labels, windows) pair.

    Shared by every graph builder that offers a ``smoothing`` option, so the
    method stays consistent regardless of which builder is downstream of it.

    ``centroids`` (only used for ``"sticky"``) lets a caller reuse
    already-computed centroids instead of deriving them fresh from
    ``labels``/``windows``. Pass the training centroids here when
    relabeling new (e.g. held-out) data through an already-fit sticky
    decoder, so it decodes against the same reference the graph was built
    with rather than centroids re-estimated from the new data itself --
    see ``TransitionGraphBuilder.relabel``/``StateMergingGraphBuilder.relabel``.
    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(f"method must be one of {SMOOTHING_METHODS}, got {method!r}")
    if method == "none":
        return labels.copy()
    if method == "sticky" and centroids is None:
        centroids = cluster_centroids(labels, windows.features)
    out = labels.copy()
    for idxs in episode_order(windows).values():
        ep_labels = labels[idxs]
        if method == "median":
            out[idxs] = median_filter(ep_labels, smoothing_window)
        else:
            out[idxs] = sticky_decode(ep_labels, windows.features[idxs], centroids, sticky_lambda)
    return out
