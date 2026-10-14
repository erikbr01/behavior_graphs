"""LIBERO environment adapter (pi0.5 via the openpi submodule).

Policy learning and inference live in the openpi fork mounted at
``third_party/openpi`` (erikbr01/openpi — Physical-Intelligence/openpi plus
policy-embedding extraction). The fork ships
``examples/libero/eval_save_with_embeddings.py``, which rolls out the
pi05_libero policy and writes ``episodes/metadata.yaml``, ``media/*.mp4`` and
``policy_embeddings/pi05.npz`` ((total_steps, 1024) float32). This adapter
shells into the submodule's uv environment and translates that output into
the canonical rollout layout (``behavior_graphs.envs.trajectory_dir``).

Setup (one-off, inside the submodule)::

    cd third_party/openpi
    uv sync
    uv pip install -e third_party/libero --config-settings editable_mode=compat
    uv pip install "robosuite==1.4.1" "bddl==1.0.1" easydict future "gym==0.25.2" \\
        "thop==0.1.1.post2209072238"   # LIBERO sim deps (its own pins are stale)

The pretrained pi05_libero checkpoint (~10 GB) downloads to ``~/.cache/openpi``
on first use.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import yaml

from behavior_graphs.core.env import EnvAdapter, Policy, RolloutSet, TrainingData
from behavior_graphs.envs._process import default_repo_root, run_streaming
from behavior_graphs.envs.trajectory_dir import RolloutDirectory

log = logging.getLogger(__name__)


def _cli_flags(overrides: dict[str, Any]) -> list[str]:
    """Translate keyword overrides into tyro-style CLI flags."""
    flags: list[str] = []
    for key, value in overrides.items():
        name = key.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{name}" if value else f"--no-{name}")
        else:
            flags.extend([f"--{name}", str(value)])
    return flags


class LiberoTrainingData(TrainingData):
    """LIBERO demonstration HDF5s (one ``<task>_demo.hdf5`` per task)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.is_dir():
            raise FileNotFoundError(f"LIBERO demo directory not found: {self._path}")
        self._files = sorted(self._path.rglob("*.hdf5"))
        if not self._files:
            raise FileNotFoundError(f"no .hdf5 demo files under {self._path}")
        self._num_episodes: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def num_episodes(self) -> int:
        if self._num_episodes is None:
            import h5py  # lazy: only needed when episode counts are queried

            total = 0
            for f in self._files:
                with h5py.File(f, "r") as h5:
                    total += len(h5["data"])
            self._num_episodes = total
        return self._num_episodes


class OpenpiPolicy(Policy):
    """Handle to an openpi checkpoint (local directory or ``gs://`` URI).

    ``checkpoint_path`` goes through :class:`Path`, which mangles ``gs://``
    URIs (``gs:/...``); the verbatim string is always kept in
    ``metadata["checkpoint"]`` and is what the adapter passes to openpi.
    """

    def __init__(self, checkpoint: str, metadata: dict[str, Any] | None = None) -> None:
        self._checkpoint = str(checkpoint)
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("checkpoint", self._checkpoint)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self._checkpoint)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class LiberoAdapter(EnvAdapter):
    """LIBERO via the openpi submodule (pi0.5 with policy-embedding capture)."""

    name = "libero"

    _EVAL_SCRIPT = "examples/libero/eval_save_with_embeddings.py"
    _TRAIN_SCRIPT = "scripts/train.py"

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero",
        config_name: str = "pi05_libero",
        repo_root: str | None = None,
        submodule: str = "third_party/openpi",
        num_trials_per_task: int = 20,
        replan_steps: int = 5,
        resize_size: int = 224,
        demo_data_dir: str | None = None,
        gpu: int | None = None,
    ) -> None:
        """
        Args:
            task_suite: libero_spatial | libero_object | libero_goal |
                libero_10 | libero_90.
            checkpoint: default checkpoint (GCS URI or local dir) used for the
                pretrained policy and as the eval default.
            config_name: openpi ``TrainConfig`` name matching the checkpoint.
            repo_root: main repo root; auto-detected by walking up from this
                file when ``None``.
            submodule: openpi checkout, relative to ``repo_root``.
            num_trials_per_task: eval trials per task when ``num_episodes``
                is not positive in :meth:`collect_rollouts`.
            replan_steps: env steps executed per pi0.5 action chunk.
            resize_size: image size fed to the model.
            demo_data_dir: directory of LIBERO demo HDF5s for
                :meth:`training_data` (optional; only needed for that method).
            gpu: pin the eval to one GPU (JAX preallocates on every visible
                device otherwise); sets CUDA_VISIBLE_DEVICES for subprocesses.
        """
        self.task_suite = task_suite
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.repo_root = Path(repo_root) if repo_root else default_repo_root()
        self.submodule = submodule
        self.num_trials_per_task = num_trials_per_task
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.demo_data_dir = demo_data_dir
        self.gpu = gpu

    # ------------------------------------------------------------------ paths

    @property
    def openpi_dir(self) -> Path:
        """The openpi submodule checkout."""
        return self.repo_root / self.submodule

    def _require_eval_script(self) -> None:
        script = self.openpi_dir / self._EVAL_SCRIPT
        if not script.exists():
            raise FileNotFoundError(
                f"openpi eval script not found: {script}\n"
                f"Initialize and set up the submodule first:\n"
                f"  git submodule update --init {self.submodule}\n"
                f"  cd {self.openpi_dir} && uv sync && uv pip install -e third_party/libero"
            )

    # -------------------------------------------------------------- interface

    def training_data(self) -> TrainingData:
        """The LIBERO demo HDF5s used for pi05_libero fine-tuning."""
        if self.demo_data_dir is None:
            raise ValueError(
                "demo_data_dir is not set. Point it at a local LIBERO demo "
                "directory (per-task <task>_demo.hdf5 files); download via "
                "LIBERO's benchmark_scripts/download_libero_datasets.py or see "
                "third_party/openpi/examples/libero/README.md for the "
                "LeRobot-format pipeline used for fine-tuning."
            )
        return LiberoTrainingData(Path(self.demo_data_dir))

    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Return the pretrained pi05 checkpoint, or fine-tune via openpi.

        With ``pretrained=True`` (default) no training runs; the returned
        policy references ``self.checkpoint``. With ``pretrained=False`` this
        shells into ``uv run scripts/train.py <config_name>
        --exp-name=<exp_name>`` inside the openpi submodule, writing
        checkpoints under ``output_dir/<config_name>/<exp_name>``. Remaining
        ``overrides`` (e.g. ``exp_name``, ``overwrite=True``, ``batch_size``)
        are forwarded as tyro CLI flags to openpi's ``TrainConfig``.
        """
        if bool(overrides.pop("pretrained", True)):
            if overrides:
                raise ValueError(f"training overrides {sorted(overrides)} require pretrained=False")
            log.info("using pretrained checkpoint %s (no training)", self.checkpoint)
            return OpenpiPolicy(
                self.checkpoint,
                {"config_name": self.config_name, "env": self.name, "pretrained": True},
            )

        self._require_eval_script()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_name = str(overrides.pop("exp_name", output_dir.name))
        cmd = [
            "uv",
            "run",
            "--no-sync",
            self._TRAIN_SCRIPT,
            self.config_name,
            f"--exp-name={exp_name}",
            f"--checkpoint-base-dir={output_dir}",
        ] + _cli_flags(overrides)
        run_streaming(cmd, cwd=self.openpi_dir, extra_env=self._subprocess_env)
        checkpoint_dir = output_dir / self.config_name / exp_name
        return OpenpiPolicy(
            str(checkpoint_dir),
            {
                "config_name": self.config_name,
                "exp_name": exp_name,
                "env": self.name,
                "pretrained": False,
            },
        )

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Roll out the policy via the openpi eval script.

        ``num_episodes`` is translated into per-task trials
        (``ceil(num_episodes / n_tasks)``); pass ``num_trials_per_task=...``
        to set it directly, or a non-positive ``num_episodes`` to use the
        adapter default. Remaining ``kwargs`` are forwarded to the eval
        script as CLI flags.
        """
        self._require_eval_script()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if "num_trials_per_task" in kwargs:
            trials = int(kwargs.pop("num_trials_per_task"))
        elif num_episodes > 0:
            n_tasks = self._num_tasks_in_suite()
            trials = math.ceil(num_episodes / n_tasks)
            if trials * n_tasks != num_episodes:
                log.warning(
                    "num_episodes=%d is not a multiple of the %d tasks in %s; "
                    "collecting %d episodes (%d per task)",
                    num_episodes,
                    n_tasks,
                    self.task_suite,
                    trials * n_tasks,
                    trials,
                )
        else:
            trials = self.num_trials_per_task

        checkpoint = policy.metadata.get("checkpoint") or str(policy.checkpoint_path)
        config_name = policy.metadata.get("config_name", self.config_name)
        cmd = [
            "uv",
            "run",
            "--no-sync",
            self._EVAL_SCRIPT,
            "--task-suite-name",
            self.task_suite,
            "--output-dir",
            str(output_dir),
            "--num-trials-per-task",
            str(trials),
            "--replan-steps",
            str(self.replan_steps),
            "--resize-size",
            str(self.resize_size),
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint,
            "--config-name",
            config_name,
        ] + _cli_flags(kwargs)
        run_streaming(cmd, cwd=self.openpi_dir, extra_env=self._subprocess_env)

        self._finalize_rollout_dir(output_dir)
        return RolloutDirectory(output_dir)

    def load_policy(self, checkpoint_path) -> Policy:
        """Wrap a checkpoint reference (local dir or ``gs://`` URI) verbatim."""
        return OpenpiPolicy(
            str(checkpoint_path), {"config_name": self.config_name, "env": self.name}
        )

    def load_rollouts(self, rollout_dir: Path) -> RolloutDirectory:
        """Load rollouts, translating openpi's native metadata if needed."""
        self._finalize_rollout_dir(Path(rollout_dir))
        return RolloutDirectory(rollout_dir)

    @property
    def _subprocess_env(self) -> dict | None:
        if self.gpu is None:
            return None
        return {"CUDA_VISIBLE_DEVICES": str(self.gpu)}

    # ---------------------------------------------------------------- helpers

    def _num_tasks_in_suite(self) -> int:
        """Tasks per suite without importing libero (90 for libero_90, else 10)."""
        return 90 if self.task_suite.startswith("libero_90") else 10

    @staticmethod
    def _finalize_rollout_dir(output_dir: Path) -> None:
        """Translate openpi's eval output into the canonical rollout layout.

        The openpi eval script writes ``episodes/metadata.yaml`` with
        ``episode_lengths``/``episode_successes`` lists; the canonical layout
        expects an ``episodes`` list of ``{index, length, success}``. The other
        artifacts already line up: ``policy_embeddings/pi05.npz`` stores a
        single ``rollout_embeddings`` array (the canonical reader falls back
        to the first npz key) and media files ``ep{idx:04d}_{succ|fail}.mp4``
        match the ``ep{idx:04d}_*.mp4`` lookup glob.
        """
        meta_path = output_dir / "episodes" / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"openpi eval produced no metadata at {meta_path}")
        raw = yaml.safe_load(meta_path.read_text()) or {}
        if "episodes" in raw:
            return
        lengths = raw.get("episode_lengths")
        successes = raw.get("episode_successes")
        if lengths is None or successes is None:
            raise ValueError(
                f"unrecognized eval metadata schema at {meta_path}: keys={sorted(raw)}"
            )
        raw["episodes"] = [
            {"index": i, "length": int(n), "success": bool(s)}
            for i, (n, s) in enumerate(zip(lengths, successes))
        ]
        meta_path.write_text(yaml.safe_dump(raw, sort_keys=False))
