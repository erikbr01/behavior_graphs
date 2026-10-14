"""Node values and per-window advantages from the behavior graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator


class ValueEvaluator(Evaluator):
    """Solves the graph's Markov reward process and summarizes advantages.

    Args:
        gamma: Discount factor of the Markov reward process.
        reward_success: Reward at the SUCCESS terminal.
        reward_failure: Reward at the FAILURE terminal.
        reward_end: Reward at the END terminal (unknown outcome).
    """

    name = "values"

    def __init__(
        self,
        gamma: float = 0.99,
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
        reward_end: float = 0.0,
    ) -> None:
        self.gamma = gamma
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.reward_end = reward_end

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        node_values = result.graph.compute_values(
            gamma=self.gamma,
            reward_success=self.reward_success,
            reward_failure=self.reward_failure,
            reward_end=self.reward_end,
        )
        result.node_values = node_values
        labels = np.asarray(result.labels)
        _, advantages, _ = result.graph.compute_window_values(
            labels, result.windows, node_values, gamma=self.gamma
        )
        adv = advantages[labels != -1]
        return {
            "node_values": {str(nid): float(v) for nid, v in node_values.items()},
            "mean_advantage": float(adv.mean()) if adv.size else 0.0,
            "mean_abs_advantage": float(np.abs(adv).mean()) if adv.size else 0.0,
            "frac_negative_advantage": float((adv < 0).mean()) if adv.size else 0.0,
        }
