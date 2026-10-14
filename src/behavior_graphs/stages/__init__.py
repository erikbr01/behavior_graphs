"""Concrete pipeline-stage implementations.

Every class here implements one of the :mod:`behavior_graphs.core.stages` ABCs
and is instantiable from a hydra config via ``_target_``.
"""

from behavior_graphs.stages.clustering import KMeansClustering
from behavior_graphs.stages.evaluation import (
    ClusterQualityEvaluator,
    GraphStatsEvaluator,
    SubtaskBoundaryEvaluator,
    ValueEvaluator,
)
from behavior_graphs.stages.features import (
    ConcatFeatures,
    PolicyEmbeddings,
    StateActionFeatures,
    StateFeatures,
    VisionBackboneFeatures,
)
from behavior_graphs.stages.graph import TransitionGraphBuilder
from behavior_graphs.stages.reduction import (
    IdentityReduction,
    PCAReduction,
    UMAPReduction,
    make_normalizer,
)
from behavior_graphs.stages.windowing import SlidingWindow

__all__ = [
    "ConcatFeatures",
    "PolicyEmbeddings",
    "StateActionFeatures",
    "StateFeatures",
    "VisionBackboneFeatures",
    "SlidingWindow",
    "IdentityReduction",
    "PCAReduction",
    "UMAPReduction",
    "make_normalizer",
    "KMeansClustering",
    "TransitionGraphBuilder",
    "GraphStatsEvaluator",
    "ClusterQualityEvaluator",
    "SubtaskBoundaryEvaluator",
    "ValueEvaluator",
]
