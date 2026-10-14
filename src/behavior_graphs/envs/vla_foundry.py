"""VLA Foundry environment adapter (TRI LBM eval via the vla_foundry submodule).

VLA Foundry (TRI-ML/vla_foundry; fork erikbr01/vla_foundry) is Toyota
Research's VLA training framework; its policies are evaluated on the LBM
benchmark — a Drake simulator shipped as the ``toyotaresearch/lbm-eval-oss``
Docker image that drives a gRPC policy server hosting the checkpoint.

The fork's ``vla_foundry/eval/run_eval_save_rollouts.py`` owns the whole
eval lifecycle (embedding-capturing policy server + Docker simulator) and
writes the canonical rollout layout directly, so this adapter shells out to
that one entry point inside the submodule's uv environment. Embeddings are
the vision-language backbone's action-conditioning output (layer
``action_conditioning``), one row per inference call; episodes align to
embedding segments via the per-episode gRPC reset seed.

Setup (one-off, inside the submodule)::

    cd third_party/vla_foundry
    uv sync --group inference && uv pip install -e .
    docker pull toyotaresearch/lbm-eval-oss:vla-foundry
    hf download TRI-ML/Foundry-VLA-1.7B-sim   # or any Foundry checkpoint

Requires Docker with the NVIDIA container toolkit on the host. Checkpoints
are HF repo ids (e.g. ``TRI-ML/Foundry-VLA-1.7B-sim``), ``hf://``/``s3://``
URIs, or local training output dirs. Tasks are the LBM benchmark names
(e.g. ``PutMugOnSaucer``; full list in the fork's
``vla_foundry/eval/run_evaluation.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from behavior_graphs.core.env import EnvAdapter, Policy, RolloutSet, TrainingData
from behavior_graphs.envs._hf_policy import HFCheckpointPolicy
from behavior_graphs.envs._process import default_repo_root, run_streaming
from behavior_graphs.envs.trajectory_dir import RolloutDirectory

log = logging.getLogger(__name__)


class VlaFoundryAdapter(EnvAdapter):
    """TRI VLA Foundry policies on the LBM (Drake) benchmark."""

    name = "vla_foundry"

    def __init__(
        self,
        task: str = "PutMugOnSaucer",
        checkpoint: str = "TRI-ML/Foundry-VLA-1.7B-sim",
        repo_root: str | None = None,
        submodule: str = "third_party/vla_foundry",
        port: int = 50051,
        num_flow_steps: int = 8,
        open_loop_steps: int = 8,
        gpu: int | None = None,
    ) -> None:
        """
        Args:
            task: LBM benchmark task name (PascalCase).
            checkpoint: HF repo id, ``hf://``/``s3://`` URI, or local
                checkpoint directory (must hold config.yaml + checkpoints/).
            repo_root: main repo root; auto-detected when ``None``.
            submodule: vla_foundry checkout, relative to ``repo_root``.
            port: localhost gRPC port for the policy server.
            num_flow_steps: diffusion/flow denoising steps per inference call.
            open_loop_steps: env steps executed per inference call (action
                chunk length before re-planning).
            gpu: pin policy server + simulator to one GPU.
        """
        self.task = task
        self.checkpoint = checkpoint
        self.repo_root = Path(repo_root) if repo_root else default_repo_root()
        self.submodule = submodule
        self.port = port
        self.num_flow_steps = num_flow_steps
        self.open_loop_steps = open_loop_steps
        self.gpu = gpu

    # ------------------------------------------------------------------ paths

    @property
    def foundry_dir(self) -> Path:
        return self.repo_root / self.submodule

    def _require_setup(self) -> None:
        if not (self.foundry_dir / "vla_foundry" / "eval" / "run_eval_save_rollouts.py").exists():
            raise FileNotFoundError(
                f"vla_foundry submodule not set up at {self.foundry_dir}; "
                f"run `git submodule update --init {self.submodule}` and the "
                f"setup steps in this module's docstring."
            )

    # -------------------------------------------------------------- interface

    def training_data(self) -> TrainingData:
        raise NotImplementedError(
            "LBM sim training data is public but not wired here: download raw "
            "Spartan episodes per task with the fork's "
            "vla_foundry/data/scripts/download_dataset.py (--list for tasks) "
            "and preprocess to WebDataset shards per docs/guides/lbm-pipeline.md."
        )

    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Return the pretrained checkpoint (local training not wired yet).

        Training IS possible upstream: public raw sim data (see
        ``training_data``) + ``vla_foundry/main.py`` (torchrun, FSDP2) with
        e.g. ``config_presets/training_jobs/diffusion_policy_bellpepper.yaml``
        (CLIP-backbone diffusion policy). Until that pipeline is wired into
        this adapter, only ``pretrained=True`` (the default) is supported.
        """
        if not bool(overrides.pop("pretrained", True)) or overrides:
            raise NotImplementedError(
                "Local VLA Foundry training is not wired into this adapter "
                "yet; follow the fork's docs/guides/lbm-pipeline.md "
                "(download_dataset.py -> preprocess_robotics_to_tar.py -> "
                "torchrun main.py), then evaluate the checkpoint dir via "
                "checkpoint=<dir>."
            )
        log.info("using pretrained checkpoint %s (no training)", self.checkpoint)
        return HFCheckpointPolicy(self.checkpoint, {"env": self.name})

    def load_policy(self, checkpoint_path) -> Policy:
        """Wrap a checkpoint reference (HF id, URI, or local dir) verbatim."""
        return HFCheckpointPolicy(str(checkpoint_path), {"env": self.name})

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Run the fork's eval orchestrator; it writes the canonical layout.

        ``seed`` offsets the LBM scenario-index range: episodes
        ``seed:seed+num_episodes`` are evaluated. Extra ``kwargs`` are
        forwarded to ``run_eval_save_rollouts.py`` as flags
        (e.g. ``max_retries=0``, ``video_fps=20``).
        """
        self._require_setup()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = policy.metadata.get("checkpoint", str(policy.checkpoint_path))

        cmd = [
            "uv",
            "run",
            "--no-sync",
            "--group",
            "inference",
            "python",
            "vla_foundry/eval/run_eval_save_rollouts.py",
            "--checkpoint",
            checkpoint,
            "--task",
            self.task,
            "--num-episodes",
            f"{seed}:{seed + num_episodes}",
            "--output-dir",
            str(output_dir),
            "--port",
            str(self.port),
            "--num_flow_steps",
            str(self.num_flow_steps),
            "--open_loop_steps",
            str(self.open_loop_steps),
        ]
        if self.gpu is not None:
            cmd += ["--gpu", str(self.gpu)]
        for key, value in kwargs.items():
            cmd += [f"--{key}", str(value)]
        run_streaming(cmd, cwd=self.foundry_dir, log_path=output_dir / "collect.log")
        return RolloutDirectory(output_dir)
