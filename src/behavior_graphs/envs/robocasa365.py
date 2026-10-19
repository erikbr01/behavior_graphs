"""RoboCasa365 environment adapter (pretrained multitask policies).

RoboCasa 1.0 ("RoboCasa365", robocasa/robocasa@main) benchmarks pretrained
multitask policies — pi0 (openpi), GR00T N1.5, diffusion policy — over 365
kitchen tasks. Checkpoints: ``robocasa/robocasa365_checkpoints`` on HF.

Each policy stack lives in its own fork/submodule with an in-process eval
script that writes the canonical rollout layout directly (one embedding per
env step, the active chunk's embedding repeated — same convention as the
other diffusion-policy envs):

- ``pi0``: ``third_party/robocasa_openpi`` (robocasa-benchmark/openpi fork)
  ``examples/robocasa/eval_save_with_embeddings.py`` — mean-pooled
  action-expert ``suffix_out`` (layer ``pi0``, 1024-d).

Setup (one-off)::

    # simulator (third_party/robocasa365, branch robocasa365 of erikbr01/robocasa)
    cd third_party/robocasa365
    uv venv --python 3.11 .venv
    uv pip install -e . "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@master"
    .venv/bin/python robocasa/scripts/setup_macros.py
    yes | .venv/bin/python robocasa/scripts/download_kitchen_assets.py --type all  # ~22 GB

    # pi0 stack
    cd third_party/robocasa_openpi
    uv sync && uv pip install -e ../robocasa365 \
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@master" \
        "huggingface-hub>=0.30.0,<1.0"
    hf download robocasa/robocasa365_checkpoints --include "pi0/*"

Task names are RoboCasa365 PascalCase env names (e.g.
``PickPlaceCounterToCabinet``; registry in
``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from behavior_graphs.core.env import EnvAdapter, Policy, RolloutSet, TrainingData
from behavior_graphs.envs._hf_policy import HFCheckpointPolicy
from behavior_graphs.envs._process import default_repo_root, run_streaming
from behavior_graphs.envs.trajectory_dir import RolloutDirectory

log = logging.getLogger(__name__)

_POLICY_STACKS = {
    "pi0": {
        "submodule": "third_party/robocasa_openpi",
        "script": "examples/robocasa/eval_save_with_embeddings.py",
        "runner": ["uv", "run", "--no-sync", "python"],
        "extra_args": lambda self: [
            "--config-name",
            "pi0_robocasa_pretrain_human300",
            "--replan-steps",
            str(self.replan_steps),
        ],
    },
    "gr00t": {
        "submodule": "third_party/robocasa_gr00t",
        "script": "scripts/eval_save_with_embeddings.py",
        # plain venv python: the repo is pip-managed, not a uv project
        "runner": [".venv/bin/python"],
        "extra_args": lambda self: [],
    },
    "dp": {
        # our diffusion_policy fork, robocasa365 runner. Its venv is NOT the
        # submodule's `.venv`: that one is the python 3.9 training stack from
        # setup_env_uv.sh, while robocasa365 needs 3.10 with robosuite master
        # and pinned mujoco/numpy. `setup_env_robocasa365_uv.sh` builds
        # `.venv-robocasa365` beside it, which is what this has to name --
        # pointing at `.venv` silently runs the training interpreter.
        # BEHAVIOR_GRAPHS_ROBOCASA365_PYTHON overrides it (the script takes the
        # matching VENV= to build elsewhere).
        "submodule": "third_party/diffusion_policy",
        "script": "scripts/eval_save_rollouts.py",
        "runner": [
            os.environ.get("BEHAVIOR_GRAPHS_ROBOCASA365_PYTHON", ".venv-robocasa365/bin/python")
        ],
        "extra_args": lambda self: [
            "--layer",
            "obs_encoder",
            "--num-vis",
            "-1",
            "--runner-target",
            "diffusion_policy.env_runner.robocasa365_image_runner.Robocasa365ImageRunner",
            "--runner-set",
            f"env_name={self.task}",
            "--runner-set",
            f"split={self.split}",
        ],
    },
}


class Robocasa365Adapter(EnvAdapter):
    """Pretrained multitask policies on RoboCasa365 (robosuite 1.5 kitchens)."""

    name = "robocasa365"

    def __init__(
        self,
        task: str = "PickPlaceCounterToCabinet",
        policy_stack: str = "pi0",
        checkpoint: str = "",
        repo_root: str | None = None,
        split: str = "pretrain",
        replan_steps: int = 5,
        gpu: int | None = None,
    ) -> None:
        """
        Args:
            task: RoboCasa365 env name (PascalCase).
            policy_stack: which pretrained stack to serve (``pi0``; gr00t and
                diffusion policy pending).
            checkpoint: checkpoint step dir (e.g. the HF snapshot's
                ``pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000``).
            repo_root: main repo root; auto-detected when ``None``.
            split: scene split (``pretrain`` | ``target``).
            replan_steps: env steps executed per action chunk.
            gpu: pin the eval to one GPU (JAX/torch servers preallocate).
        """
        if policy_stack not in _POLICY_STACKS:
            raise ValueError(
                f"unknown policy_stack {policy_stack!r}; expected one of {sorted(_POLICY_STACKS)}"
            )
        self.task = task
        self.policy_stack = policy_stack
        self.checkpoint = checkpoint
        self.repo_root = Path(repo_root) if repo_root else default_repo_root()
        self.split = split
        self.replan_steps = replan_steps
        self.gpu = gpu

    @property
    def stack_dir(self) -> Path:
        return self.repo_root / _POLICY_STACKS[self.policy_stack]["submodule"]

    def _require_setup(self) -> None:
        script = self.stack_dir / _POLICY_STACKS[self.policy_stack]["script"]
        if not script.exists():
            raise FileNotFoundError(
                f"{self.policy_stack} stack not set up at {self.stack_dir}; see this module's docstring."
            )

    # -------------------------------------------------------------- interface

    def training_data(self) -> TrainingData:
        raise NotImplementedError(
            "RoboCasa365 training data (482h human demos) is not mirrored here; "
            "download per task via robocasa/scripts/download_datasets.py."
        )

    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Return the pretrained multitask checkpoint (training not wired)."""
        if not bool(overrides.pop("pretrained", True)) or overrides:
            raise NotImplementedError(
                "Training the RoboCasa365 multitask policies is upstream work "
                "(robocasa-benchmark forks); use the released "
                "robocasa/robocasa365_checkpoints."
            )
        log.info("using pretrained checkpoint %s (no training)", self.checkpoint)
        return HFCheckpointPolicy(
            self.checkpoint, {"env": self.name, "policy_stack": self.policy_stack}
        )

    def load_policy(self, checkpoint_path) -> Policy:
        return HFCheckpointPolicy(
            str(checkpoint_path), {"env": self.name, "policy_stack": self.policy_stack}
        )

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Run the stack's in-process eval; it writes the canonical layout."""
        self._require_setup()

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = policy.metadata.get("checkpoint", str(policy.checkpoint_path))
        stack = _POLICY_STACKS[self.policy_stack]

        ckpt_flag = {"pi0": "--checkpoint", "gr00t": "--model-path", "dp": "--checkpoint"}[
            self.policy_stack
        ]
        if self.policy_stack == "dp":
            # our eval_save_rollouts.py CLI differs from the in-process scripts
            cmd = [
                *stack["runner"],
                stack["script"],
                ckpt_flag,
                checkpoint,
                "--output-dir",
                str(output_dir),
                "--num-episodes",
                str(num_episodes),
                "--seed",
                str(100000 + seed),
                *stack["extra_args"](self),
            ]
        else:
            cmd = [
                *stack["runner"],
                stack["script"],
                "--env-name",
                self.task,
                "--num-trials",
                str(num_episodes),
                ckpt_flag,
                checkpoint,
                "--output-dir",
                str(output_dir),
                "--split",
                self.split,
                *stack["extra_args"](self),
            ]
            if self.policy_stack == "pi0":
                cmd += ["--seed", str(seed)]
        for key, value in kwargs.items():
            cmd += [f"--{key.replace('_', '-')}", str(value)]
        extra_env = {"MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if self.gpu is not None:
            extra_env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        run_streaming(
            cmd, cwd=self.stack_dir, log_path=output_dir / "collect.log", extra_env=extra_env
        )
        return RolloutDirectory(output_dir)
