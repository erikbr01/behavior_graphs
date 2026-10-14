"""Progress-estimation methods (one class per method, config-instantiable).

Graph methods (fit a behavior graph, label held-out episodes) and VLM methods
(score episode videos) both implement :class:`ProgressMethod`.
"""

from behavior_graphs.progress_estimation.methods.base import ProgressMethod
from behavior_graphs.progress_estimation.methods.graph import (
    FrameIndex,
    GraphBackbone,
    GraphStage,
)
from behavior_graphs.progress_estimation.methods.mlp import MLPRegressorProgress
from behavior_graphs.progress_estimation.methods.tuned import TunedGraphBackbone
from behavior_graphs.progress_estimation.methods.vlm import (
    Robometer,
    TopRewardProbe,
)

__all__ = [
    "ProgressMethod",
    "GraphBackbone",
    "TunedGraphBackbone",
    "GraphStage",
    "FrameIndex",
    "MLPRegressorProgress",
    "TopRewardProbe",
    "Robometer",
]
