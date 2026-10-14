"""Core interfaces and data types of the behavior-graph pipeline."""

from behavior_graphs.core.env import EnvAdapter, Policy, RolloutSet, TrainingData
from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    SPECIAL_NODE_IDS,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
    BehaviorNode,
)
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import (
    Clustering,
    DimReduction,
    Evaluator,
    FeatureExtractor,
    GraphConstruction,
    Windowing,
)
from behavior_graphs.core.types import (
    EpisodeInfo,
    FeatureSet,
    WindowInfo,
    WindowSet,
    episodes_from_lengths,
)

__all__ = [
    "EnvAdapter",
    "Policy",
    "RolloutSet",
    "TrainingData",
    "BehaviorGraph",
    "BehaviorNode",
    "START_NODE_ID",
    "END_NODE_ID",
    "SUCCESS_NODE_ID",
    "FAILURE_NODE_ID",
    "TERMINAL_NODE_IDS",
    "SPECIAL_NODE_IDS",
    "PipelineResult",
    "FeatureExtractor",
    "Windowing",
    "DimReduction",
    "Clustering",
    "GraphConstruction",
    "Evaluator",
    "EpisodeInfo",
    "FeatureSet",
    "WindowInfo",
    "WindowSet",
    "episodes_from_lengths",
]
