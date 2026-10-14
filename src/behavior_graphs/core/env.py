"""Abstract interfaces to test environments.

Every environment (robomimic, mimicgen, robocasa, droid, libero, libero-pro)
exposes exactly three things to the behavior-graph pipeline — training data,
a policy, and rollouts — through an :class:`EnvAdapter`. All policy-learning
code and configs live in the corresponding ``third_party/`` submodule; the
adapter's job is to invoke it and to translate its outputs into the standard
interfaces below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.types import EpisodeInfo


class TrainingData(ABC):
    """Interface to an environment's demonstration dataset."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Location of the dataset on disk."""

    @property
    @abstractmethod
    def num_episodes(self) -> int:
        """Number of demonstration episodes."""

    def open_store(self) -> Any:
        """Open the dataset as a ``robo_dataset.TrajectoryStore`` if supported.

        Optional; adapters whose data is consumable through robo_dataset
        should override this. Raises ``NotImplementedError`` otherwise.
        """
        raise NotImplementedError(f"{type(self).__name__} has no robo_dataset backend")


class Policy(ABC):
    """Handle to a trained policy checkpoint."""

    @property
    @abstractmethod
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint usable by the owning environment."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Free-form info: backbone, task, training config, etc."""


class CheckpointPolicy(Policy):
    """Generic policy handle: a checkpoint path plus free-form metadata."""

    def __init__(self, checkpoint_path: Path, metadata: dict[str, Any] | None = None):
        self._checkpoint_path = Path(checkpoint_path)
        self._metadata = metadata or {}

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class TrajectorySet(ABC):
    """A set of episodes with aligned per-timestep feature arrays.

    The shared input type of the behavior-graph pipeline's front half
    (feature extraction and windowing): a flat stack of per-timestep rows,
    partitioned into episodes. Whether the trajectories were produced by a
    policy (:class:`RolloutSet`) or by a demonstrator only matters as
    provenance, recorded in the episode metadata — no pipeline stage branches
    on it.

    The canonical on-disk layout is implemented by
    ``behavior_graphs.envs.trajectory_dir.TrajectoryDirectory``::

        <dir>/
            episodes/metadata.yaml          # lengths, success flags, provenance
            policy_embeddings/<layer>.npz   # (total_steps, d) float32, key "embeddings"
            observations/<key>.npz          # optional, (total_steps, d_obs) float32
            actions.npz                     # optional, (total_steps, d_action) float32
            media/ep{idx:04d}_{success|failure}.mp4   # optional

    ``observations/<key>.npz`` and ``actions.npz`` (the action executed at each
    step) are optional per-timestep arrays aligned with the episode ranges.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """Root directory of the trajectory artifacts."""

    @abstractmethod
    def episodes(self) -> list[EpisodeInfo]:
        """Per-episode metadata with contiguous global sample ranges."""

    @abstractmethod
    def embedding_layers(self) -> list[str]:
        """Names of available embedding layers."""

    @abstractmethod
    def embeddings(self, layer: str) -> np.ndarray:
        """(total_steps, d) float32 embeddings, aligned with episode ranges."""

    def observation_keys(self) -> list[str]:
        """Names of available raw per-timestep observation arrays."""
        return []

    def observations(self, key: str) -> np.ndarray:
        """(total_steps, d_obs) float32 observations, aligned with episode ranges."""
        raise FileNotFoundError(
            f"observation key '{key}' not found; available: {self.observation_keys()}"
        )

    def actions(self) -> np.ndarray | None:
        """(total_steps, d_action) float32 executed actions, or None if absent."""
        return None

    def video_path(self, episode_idx: int) -> Path | None:
        """Path to the episode's video, if recorded."""
        return None


class RolloutSet(TrajectorySet):
    """Trajectories produced by rolling out a policy.

    Same data contract as :class:`TrajectorySet`; episode ``success`` reflects
    task outcome and one embedding row corresponds to one policy inference
    call. Adapters' ``collect_rollouts`` returns this type.
    """


class EnvAdapter(ABC):
    """One environment's gateway between its submodule and the pipeline."""

    name: str = "env"

    @abstractmethod
    def training_data(self) -> TrainingData:
        """The demonstration dataset used to train the policy."""

    @abstractmethod
    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Train a policy by delegating to the environment's submodule.

        ``overrides`` are forwarded to the submodule's own config system;
        the main repo defines no policy-learning hyperparameters.
        """

    def train_policy_multi_gpu(self, output_dir: Path, num_gpus: int, **overrides: Any) -> Policy:
        """Train on multiple GPUs via distributed data-parallel.

        ``num_gpus`` is the number of local GPUs (ranks) to use.
        ``overrides`` are forwarded to the submodule's own config system.
        Raises ``NotImplementedError`` for adapters that do not support DDP.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-GPU training; "
            f"use train_policy() for single-GPU training"
        )

    @abstractmethod
    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Roll out the policy and write the standard rollout layout."""

    def load_policy(self, checkpoint_path: Path) -> Policy:
        """Wrap an existing checkpoint as a :class:`Policy` handle."""
        return CheckpointPolicy(checkpoint_path, {"env": self.name})

    def load_rollouts(self, rollout_dir: Path) -> RolloutSet:
        """Load previously collected rollouts (standard layout by default)."""
        from behavior_graphs.envs.trajectory_dir import RolloutDirectory

        return RolloutDirectory(rollout_dir)

    def extract_demo_features(
        self, policy: Policy, output_dir: Path, **kwargs: Any
    ) -> TrajectorySet:
        """Featurize the training demonstrations with a trained policy.

        Runs the policy forward over the demonstration dataset and writes the
        canonical trajectory layout (one embedding row per policy-dataset
        sample), so demos and rollouts share one representation space — the
        prerequisite for featurizing demonstrations. Optional: adapters whose
        stack has a demo featurizer override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no demo featurizer; see "
            f"DiffusionPolicyEnvAdapter.extract_demo_features for the exemplar"
        )

    def extract_rollout_features(
        self, policy: Policy, rollout_dir: Path, output_dir: Path, **kwargs: Any
    ) -> TrajectorySet:
        """Featurize existing rollouts by replaying saved observations through the policy.

        Reads observations from ``rollout_dir`` (canonical layout), runs the
        policy forward pass, and writes embeddings to ``output_dir`` in the
        same canonical layout — so rollouts collected without embeddings can
        be retrofitted. Adapters whose stack supports observation replay
        override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no rollout featurizer; implement "
            f"extract_rollout_features to replay saved observations through the policy"
        )

    def load_demo_features(self, demo_dir: Path) -> TrajectorySet:
        """Load previously featurized demonstrations (standard layout)."""
        from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

        return TrajectoryDirectory(demo_dir)
