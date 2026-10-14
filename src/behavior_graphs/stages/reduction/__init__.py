"""Dimensionality-reduction stage implementations."""

from behavior_graphs.stages.reduction.identity import IdentityReduction
from behavior_graphs.stages.reduction.normalize import make_normalizer
from behavior_graphs.stages.reduction.pca import PCAReduction
from behavior_graphs.stages.reduction.umap import UMAPReduction

__all__ = [
    "IdentityReduction",
    "PCAReduction",
    "UMAPReduction",
    "make_normalizer",
]
