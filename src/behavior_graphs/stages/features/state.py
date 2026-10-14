"""Feature extractor that reads raw observations stored alongside the trajectories."""

from __future__ import annotations

from behavior_graphs.core.env import TrajectorySet
from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.core.types import FeatureSet


class StateFeatures(FeatureExtractor):
    """Loads per-timestep raw observations stored alongside the trajectories.

    Args:
        key: Name of the observation array to load (e.g. ``"state"``).
    """

    def __init__(self, key: str = "state") -> None:
        self.key = key

    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return the stored ``(total_steps, d_obs)`` observations as a FeatureSet."""
        available = trajectories.observation_keys()
        if self.key not in available:
            raise FileNotFoundError(
                f"observation key '{self.key}' not found; available: {available}"
            )
        return FeatureSet(trajectories.observations(self.key), trajectories.episodes())
