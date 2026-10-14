"""Feature extractor that concatenates the outputs of other extractors."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.env import TrajectorySet
from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.core.types import FeatureSet


class ConcatFeatures(FeatureExtractor):
    """Concatenates sub-extractor features along the feature dimension.

    Args:
        extractors: Feature extractors whose per-timestep outputs are joined;
            hydra instantiates them from a nested ``_target_`` list.
        block_norm: How to put the blocks on a comparable scale before joining.
            ``none`` concatenates raw. ``invsqrtd`` divides each block by the
            square root of its width. ``l2`` scales each block to unit norm per
            timestep.

    On ``block_norm``: joining 14 joint angles in radians to a 9,216-d DINOv3
    embedding raw means the embedding decides every distance, and the arm just
    reproduces the embedding-only arm.

    Measured, the choice barely matters. At fixed dials on bottles -- PCA to 10
    components, k-means at K=16, width and stride 4 -- the three score 0.532
    (none), 0.534 (l2) and 0.532 (invsqrtd), against 0.530 for the embedding
    alone and 0.361 for proprioception alone. So the concat is worth having and
    the scaling is not the lever it looks like.

    Worth recording because the opposite was assumed: the arm first reported
    0.111 in a cross-task table and the scaling was the obvious suspect. It was
    not. That arm had 2 to 10 of its 72 cells swept, on three tasks, and every
    swept cell used UMAP -- the worse reduction here by a wide margin. The
    number was an argmax over the arm's worst corner, not a broken
    representation. `invsqrtd` is kept because it is the more defensible
    default on principle -- a constant per block, no dataset statistics, and it
    preserves magnitude, which for joint angles carries real signal -- but
    `l2` is what the existing jobs used and the two are interchangeable at this
    precision.
    """

    def __init__(self, extractors: list[FeatureExtractor], block_norm: str = "none") -> None:
        if not extractors:
            raise ValueError("ConcatFeatures needs at least one sub-extractor")
        allowed = ("none", "l2", "invsqrtd")
        if block_norm not in allowed:
            raise ValueError(f"block_norm must be one of {allowed}, got {block_norm!r}")
        self.extractors = list(extractors)
        self.block_norm = block_norm

    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return ``(total_steps, sum of sub-extractor dims)`` features.

        Written to hold two copies of the largest block rather than three.
        The obvious form -- scale each block, then ``np.concatenate`` -- keeps
        the source, the scaled copy and the joined result alive at once. On
        legos' 836,312 x 9,216 DINOv3 block that is 3 x 31 GB, and the sweep
        worker was measured at 81 GB RSS before it was stopped for crowding the
        machine to 12 GB free.

        Instead the output is allocated once, each block is copied straight
        into its own columns, normalised in place, and its source released
        immediately. Peak is the output plus the one block being copied.
        """
        feature_sets = [extractor.extract(trajectories) for extractor in self.extractors]
        episodes = feature_sets[0].episodes
        widths = [int(fs.features.shape[1]) for fs in feature_sets]
        n_rows = int(feature_sets[0].features.shape[0])

        out = np.empty((n_rows, sum(widths)), dtype=np.float32)
        lo = 0
        for i, width in enumerate(widths):
            block = out[:, lo : lo + width]
            block[:] = feature_sets[i].features
            # Dropped before the next block is copied, so the joined array and
            # one source are the only large allocations at any moment.
            feature_sets[i] = None  # type: ignore[call-overload]
            if self.block_norm == "l2":
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                # A timestep whose block is all zeros stays zero rather than
                # becoming NaN and poisoning every downstream distance.
                np.divide(block, np.maximum(norms, 1e-8), out=block)
            elif self.block_norm == "invsqrtd":
                block /= np.sqrt(float(width))
            lo += width
        return FeatureSet(out, episodes)
