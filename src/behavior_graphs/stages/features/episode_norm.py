"""Remove each episode's own offset from per-timestep features.

Why this exists, concretely. UMAP builds its embedding from each point's
``n_neighbors`` nearest neighbours, and that is an absolute count (15) rather
than a fraction of the data. On the ABC datasets an episode contributes ~405
windows at stride 4 against ~228 for the older DROID collections, and measured
on bottles ``dinov3`` two thirds of a point's 15 nearest neighbours come from
its own episode. UMAP faithfully embeds that as one component per episode, so
the clustering downstream returns episode identity: the biggest single segment
covers 83% of an episode on average and 100% at the 90th percentile, and one
cluster owns 97.5% of the windows.

Subtracting the episode's own statistics removes the offset the neighbourhood
search is locking onto, leaving only within-episode variation -- which is the
only thing a behavior segmentation should be reading anyway.

Applied between features and windowing (see :mod:`behavior_graphs.pipeline.run`)
rather than as a wrapper extractor, deliberately: a wrapper would nest the
extractor's own config one level deeper and break every ``features.layer=``
grid override in the study. As a pipeline-level key it composes with any
extractor, including ``ConcatFeatures``, and sweeps as a plain dial.

**No leakage.** Every variant uses only the episode's own frames, so a held-out
episode is normalized by its own statistics and nothing crosses the split. That
is what makes this legitimate where standardizing against dataset statistics
would not be.

The transforms themselves come from
:func:`behavior_graphs.wgo_bench.data.normalize_episode`, which is where they
were first written and tested (``episode_center`` subtracts the episode mean,
``episode_zscore`` also divides by its per-dimension standard deviation with a
floor so a dead dimension cannot be amplified into noise, ``episode_rank``
replaces each value by its within-episode quantile). Imported rather than
reimplemented so the two cannot drift.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.types import FeatureSet

#: The per-episode normalizations :func:`normalize_episode` understands.
FEATURE_NORMS = ("none", "episode_center", "episode_zscore", "episode_rank")


def normalize_episode(x: np.ndarray, how: str) -> np.ndarray:
    """Remove the episode-level component of a per-frame feature block.

    A high-dimensional visual embedding of an episode varies far more with
    *which scene* it is than with *when in the episode* it is. K-Means over the
    raw features therefore partitions episodes, not behaviors -- every episode
    lands in a single cluster, the label sequence never changes, and the graph
    emits no boundaries at all. Our usual setting hides this: a rollout set is
    many attempts at ONE task, so the between-episode component is small and
    ignoring it costs nothing. Across tasks and scenes, it does not.

    ``episode_center`` subtracts the episode's own mean. ``episode_zscore`` also
    divides by its per-dimension standard deviation, floored at 1e-3 of the mean
    standard deviation so that a dead dimension cannot be amplified into noise.
    ``episode_rank`` replaces each value by its within-episode quantile, which
    removes the episode's scale AND the shape of its marginal, leaving only the
    temporal ordering each dimension induces.

    All three use only the episode's own frames -- no labels, no other episodes,
    and nothing that is unavailable when annotating a single recorded video.
    """
    if how == "none":
        return x
    if how not in FEATURE_NORMS:
        raise ValueError(f"feature_norm must be one of {FEATURE_NORMS}, got {how!r}")
    if how == "episode_rank":
        n = x.shape[0]
        if n < 2:
            return np.zeros_like(x)
        order = np.argsort(x, axis=0, kind="stable")
        ranks = np.empty_like(order, dtype=np.float32)
        np.put_along_axis(
            ranks, order, np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], x.shape), axis=0
        )
        return ranks / (n - 1)
    centred = x - x.mean(axis=0, keepdims=True)
    if how == "episode_center":
        return centred
    sd = x.std(axis=0, keepdims=True)
    floor = 1e-3 * float(sd.mean()) if sd.size else 1.0
    return centred / np.maximum(sd, max(floor, 1e-12))


__all__ = ["FEATURE_NORMS", "apply_episode_norm", "normalize_episode"]


def apply_episode_norm(features: FeatureSet, how: str) -> FeatureSet:
    """*features* with each episode's own statistics removed.

    ``how="none"`` returns the input unchanged and untouched, so the default
    path allocates nothing.

    Raises:
        ValueError: if *how* is not one of :data:`FEATURE_NORMS`.
    """
    if how == "none":
        return features
    if how not in FEATURE_NORMS:
        raise ValueError(f"episode_norm must be one of {list(FEATURE_NORMS)}, got {how!r}")

    x = np.asarray(features.features)
    # float64 for the accumulation: episode_zscore divides by a per-dimension
    # standard deviation, and on a 9,216-d block in float32 the small
    # denominators are where precision actually bites.
    out = np.empty_like(x, dtype=np.float32)
    for ep in features.episodes:
        a, b = ep.sample_start_idx, ep.sample_end_idx
        if b <= a:
            continue
        out[a:b] = normalize_episode(x[a:b].astype(np.float64), how).astype(np.float32)
    return FeatureSet(out, features.episodes)
