"""Feature extractor that concatenates raw observations with executed actions."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.env import TrajectorySet
from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.core.types import FeatureSet


class StateActionFeatures(FeatureExtractor):
    """Per-timestep ``[obs, action_weight * action]`` feature vectors.

    Requires trajectories that store both ``observations/<key>.npz`` and
    ``actions.npz`` (the action executed at each step).

    Args:
        key: Name of the observation array to load (e.g. ``"state"``).
        action_weight: Scale applied to the action block before concatenation.
    """

    def __init__(self, key: str = "state", action_weight: float = 1.0) -> None:
        self.key = key
        self.action_weight = action_weight

    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return ``(total_steps, d_obs + d_action)`` state-action features."""
        available = trajectories.observation_keys()
        if self.key not in available:
            raise FileNotFoundError(
                f"observation key '{self.key}' not found; available: {available}"
            )
        actions = trajectories.actions()
        if actions is None:
            raise FileNotFoundError(
                f"trajectories at {trajectories.path} store no actions; "
                "state-action features need the executed action per step"
            )
        obs = trajectories.observations(self.key)
        features = np.concatenate([obs, self.action_weight * actions], axis=1).astype(np.float32)
        return FeatureSet(features, trajectories.episodes())
