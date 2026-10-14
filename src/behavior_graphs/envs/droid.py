"""DROID environment adapter (real-robot platform, diffusion-policy backed).

All DROID support lives in the ``diffusion_policy`` submodule — no separate
robot-stack dependency:

* training: the droid image configs (``task="image/droid/diffusion_policy_cnn"``)
  on data in robo_dataset formats (raw sessions via ``DroidStore`` or a
  converted ``VideoStore``);
* on-robot rollout collection: ``scripts/droid_collect_rollouts.py`` in the
  submodule (runs on the robot workstation, where the DROID stack is
  installed, and writes the canonical rollout layout);
* offline embedding extraction for already-recorded episodes:
  ``scripts/droid_extract_embeddings.py`` in the submodule.

Recorded rollout directories are loaded here with :meth:`load_rollouts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from behavior_graphs.core.env import Policy, RolloutSet, TrainingData
from behavior_graphs.envs.diffusion_policy_env import DiffusionPolicyEnvAdapter


class DroidAdapter(DiffusionPolicyEnvAdapter):
    """DROID real-robot tasks trained via diffusion_policy."""

    name = "droid"

    # The droid config has no env_runner dataset node (no sim evaluation).
    dataset_override_keys = ("task.dataset.dataset_path", "task.dataset_path")

    def training_data(self) -> TrainingData:
        return DroidTrainingData(self.dataset_path)

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        raise NotImplementedError(
            "DROID is a real-robot platform: collect rollouts on the robot "
            "workstation with third_party/diffusion_policy/scripts/"
            f"droid_collect_rollouts.py (serving {policy.checkpoint_path}), "
            "then load the recorded directory here with load_rollouts(dir). "
            "For already-recorded episodes without embeddings, use "
            "scripts/droid_extract_embeddings.py in the same submodule."
        )


class DroidTrainingData(TrainingData):
    """DROID demonstrations: raw session folders or a converted VideoStore."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def num_episodes(self) -> int:
        store = self.open_store()
        return store.num_episodes

    def open_store(self) -> Any:
        try:
            from robo_dataset import VideoStore
            from robo_dataset.backends.droid import DroidStore
        except ImportError as exc:
            raise NotImplementedError(
                "robo_dataset is not installed; cannot open DROID data"
            ) from exc
        if (self._path / "lowdim.zarr").exists() or self._path.suffix == ".video":
            return VideoStore(str(self._path))
        return DroidStore(str(self._path))
