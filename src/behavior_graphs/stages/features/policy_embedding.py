"""Feature extractor that loads policy embeddings stored alongside the trajectories."""

from __future__ import annotations

from behavior_graphs.core.env import TrajectorySet
from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.core.types import FeatureSet


class PolicyEmbeddings(FeatureExtractor):
    """Loads per-timestep policy embeddings stored alongside the trajectories.

    Args:
        layer: Name of the embedding layer to load (e.g. ``"pi05"``).
    """

    def __init__(self, layer: str = "pi05") -> None:
        self.layer = layer

    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return the stored ``(total_steps, d)`` embeddings as a FeatureSet."""
        return FeatureSet(trajectories.embeddings(self.layer), trajectories.episodes())
