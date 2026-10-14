"""Shared adapter for diffusion-policy-based environments.

robomimic, mimicgen, robocasa, and droid all learn policies through the
vendored `real-stanford/diffusion_policy` workspace mounted at
``third_party/diffusion_policy``. This adapter delegates training and rollout
collection to that submodule via subprocesses in the submodule's own (uv-managed)
environment, so the main repo never imports the simulator or policy-learning
stack.

Training invokes the submodule's ``train.py`` with a task config dir
(``configs/<task>/config.yaml``); checkpoints land in
``<output_dir>/checkpoints/`` as ``epoch=XXXX-test_mean_score=Y.ckpt`` plus
``latest.ckpt``. Rollout collection invokes ``scripts/eval_save_rollouts.py``,
which writes the canonical rollout layout consumed by
:class:`behavior_graphs.envs.trajectory_dir.RolloutDirectory`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from behavior_graphs.core.env import (
    EnvAdapter,
    Policy,
    RolloutSet,
    TrainingData,
    TrajectorySet,
)
from behavior_graphs.envs._process import default_repo_root, run_streaming
from behavior_graphs.envs.trajectory_dir import RolloutDirectory, TrajectoryDirectory

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"test_mean_score=(-?[\d.]+)")


class DiffusionPolicyCheckpoint(Policy):
    """Handle to a checkpoint produced by the diffusion_policy submodule."""

    def __init__(self, checkpoint_path: Path, metadata: dict[str, Any] | None = None) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._metadata = dict(metadata or {})

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class HDF5TrainingData(TrainingData):
    """Robomimic-format HDF5 demonstration dataset (``data/demo_{i}/...``)."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def num_episodes(self) -> int:
        import h5py

        with h5py.File(self._path, "r") as f:
            return len(f["data"])

    def open_store(self) -> Any:
        try:
            from robo_dataset import RobomimicHDF5Store
        except ImportError as exc:
            raise NotImplementedError(
                "robo_dataset is not installed; cannot open a TrajectoryStore"
            ) from exc
        import h5py

        with h5py.File(self._path, "r") as f:
            obs_keys = sorted(f["data/demo_0/obs"].keys())
        return RobomimicHDF5Store(str(self._path), obs_keys=obs_keys)


class DiffusionPolicyEnvAdapter(EnvAdapter):
    """Environment adapter backed by the diffusion_policy submodule.

    Args:
        task: Config dir inside the submodule, relative to ``configs/``
            (e.g. ``"low_dim/lift_mh/diffusion_policy_cnn"``).
        dataset_path: Demonstration dataset (robomimic-format HDF5).
        repo_root: Root of the behavior_graphs checkout; auto-detected by
            walking up from this file when omitted.
        submodule: Submodule path relative to ``repo_root``.
        python_path: Interpreter of the submodule's training environment.
            Defaults to the uv-managed venv inside the submodule
            (``<submodule>/.venv/bin/python``, created by the submodule's
            ``scripts/setup_env_uv.sh``); any interpreter path works.
        embedding_layer: Diffusion-U-Net layer for per-timestep policy
            embeddings (``{bottleneck|decoder|encoder}[_{plan|plan8|exec}][_t{T}]``).
    """

    name = "diffusion_policy"

    #: Hydra keys that receive ``dataset_path``; subclasses whose configs lack
    #: an env-runner dataset path (e.g. droid's NoOp runner) override this.
    dataset_override_keys: tuple[str, ...] = (
        "task.dataset.dataset_path",
        "task.dataset_path",
        "task.env_runner.dataset_path",
    )

    def __init__(
        self,
        task: str,
        dataset_path: str,
        repo_root: str | None = None,
        submodule: str = "third_party/diffusion_policy",
        python_path: str | None = None,
        embedding_layer: str = "bottleneck_plan_t0",
    ) -> None:
        self.task = task
        self.dataset_path = str(dataset_path)
        self.repo_root = Path(repo_root) if repo_root is not None else default_repo_root()
        self.submodule = submodule
        self.python_path = python_path
        self.embedding_layer = embedding_layer

    @property
    def submodule_dir(self) -> Path:
        """Absolute path to the diffusion_policy submodule checkout."""
        path = self.repo_root / self.submodule
        if not (path / "train.py").exists():
            raise FileNotFoundError(
                f"diffusion_policy submodule not found at {path}; is the submodule checked out?"
            )
        return path

    def training_data(self) -> TrainingData:
        return HDF5TrainingData(self.dataset_path)

    def _train_args(self, output_dir: Path, **overrides: Any) -> list:
        """Hydra CLI args for train.py (no interpreter prefix)."""
        args = [
            "train.py",
            "--config-path",
            f"configs/{self.task}",
            "--config-name",
            "config",
            f"hydra.run.dir={output_dir}",
            f"multi_run.run_dir={output_dir}",
        ]
        args += [f"{key}={self.dataset_path}" for key in self.dataset_override_keys]
        args += [f"{key}={value}" for key, value in overrides.items()]
        return args

    def _train_metadata(self, output_dir: Path) -> dict[str, Any]:
        return {
            "env": self.name,
            "task": self.task,
            "dataset_path": self.dataset_path,
            "python": str(self._python()),
            "train_output_dir": str(output_dir),
        }

    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Train via the submodule's ``train.py``; returns the best checkpoint.

        ``overrides`` are forwarded verbatim as Hydra overrides (e.g.
        ``training__num_epochs`` is not translated — pass real dotted keys via
        ``**{"training.num_epochs": 100}``).
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_in_submodule(
            self._train_args(output_dir, **overrides),
            log_path=output_dir / "subprocess.log",
        )
        checkpoint = self._find_checkpoint(output_dir / "checkpoints")
        return DiffusionPolicyCheckpoint(checkpoint, metadata=self._train_metadata(output_dir))

    def train_policy_multi_gpu(self, output_dir: Path, num_gpus: int, **overrides: Any) -> Policy:
        """Train on ``num_gpus`` local GPUs via ``torch.distributed.run`` (DDP).

        Equivalent to ``torchrun --standalone --nnodes=1 --nproc_per_node=N train.py ...``.
        All Hydra overrides apply; validation and checkpointing run on rank 0 only.
        """
        if num_gpus < 2:
            return self.train_policy(output_dir, **overrides)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        launcher = [
            str(self._python()),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={num_gpus}",
        ]
        from behavior_graphs.envs._process import run_streaming

        run_streaming(
            launcher + self._train_args(output_dir, **overrides),
            cwd=self.submodule_dir,
            log_path=output_dir / "subprocess.log",
        )
        checkpoint = self._find_checkpoint(output_dir / "checkpoints")
        meta = self._train_metadata(output_dir)
        meta["num_gpus"] = num_gpus
        return DiffusionPolicyCheckpoint(checkpoint, metadata=meta)

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Roll out via ``scripts/eval_save_rollouts.py`` in the submodule.

        ``seed`` is the test start seed (episode ``i`` uses ``seed + i``).
        Extra ``kwargs`` map to script flags (``num_vis=5`` → ``--num-vis 5``).
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "scripts/eval_save_rollouts.py",
            "--checkpoint",
            str(policy.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            str(num_episodes),
            "--seed",
            str(seed),
            "--layer",
            self.embedding_layer,
            "--dataset-path",
            self.dataset_path,
        ]
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args += [flag, str(value)]
        self._run_in_submodule(args, log_path=output_dir / "subprocess.log")
        return RolloutDirectory(output_dir)

    def extract_demo_features(
        self, policy: Policy, output_dir: Path, **kwargs: Any
    ) -> TrajectorySet:
        """Featurize demos via ``scripts/extract_demo_embeddings.py``.

        Runs the policy forward over its own training dataset (one embedding
        row per policy-dataset sample) and writes the canonical trajectory
        layout. Extra ``kwargs`` map to script flags
        (``batch_size=256`` → ``--batch-size 256``).
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "scripts/extract_demo_embeddings.py",
            "--checkpoint",
            str(policy.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--layer",
            self.embedding_layer,
            "--dataset-path",
            self.dataset_path,
        ]
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args += [flag, str(value)]
        self._run_in_submodule(args, log_path=output_dir / "subprocess.log")
        return TrajectoryDirectory(output_dir)

    @staticmethod
    def _find_checkpoint(checkpoint_dir: Path) -> Path:
        """Best checkpoint by ``test_mean_score`` in the filename, else latest."""
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"no checkpoint directory at {checkpoint_dir}")
        best: Path | None = None
        best_score = float("-inf")
        for ckpt in checkpoint_dir.glob("*.ckpt"):
            match = _SCORE_RE.search(ckpt.stem)
            if match and float(match.group(1)) > best_score:
                best, best_score = ckpt, float(match.group(1))
        if best is not None:
            return best
        latest = checkpoint_dir / "latest.ckpt"
        if latest.exists():
            return latest
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    def _python(self) -> Path:
        """Interpreter of the submodule's environment (uv venv by default)."""
        if self.python_path is not None:
            return Path(self.python_path)
        venv_python = self.submodule_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():
            raise FileNotFoundError(
                f"no environment at {venv_python}; create it with "
                f"'cd {self.submodule_dir} && scripts/setup_env_uv.sh' or pass "
                f"python_path explicitly"
            )
        return venv_python

    def _run_in_submodule(self, args: Sequence[str], log_path: Path) -> None:
        """Run a python script in the submodule's environment, cwd at its root."""
        run_streaming([str(self._python()), *args], cwd=self.submodule_dir, log_path=log_path)
