"""Environment adapter backed by the rgrl submodule.

rgrl (third_party/rgrl) is a from-scratch, GPU-resident imitation-learning
stack built for iteration speed — minutes, not hours, per training run (see
its README). This adapter mirrors DiffusionPolicyEnvAdapter's subprocess-
based pattern, but against rgrl's own hydra surface: five composed config
groups (data/target/policy/training/env), not one nested task config, and
separate train.py/eval.py entrypoints, not train.py/eval_save_rollouts.py.

Three entrypoints, not two. ``train_policy`` and ``eval_policy`` drive
train.py and eval.py; ``collect_rollouts`` drives
``rgrl.inference.collect_rollouts``, which records rollouts against any
``rollout_env`` backend -- robomimic sim or the real DROID robot -- with
per-step activation capture, and is then folded into this repo's canonical
layout by ``scripts/utils/assemble_droid_sync_recordings.py``.

``eval_policy`` is not part of the EnvAdapter ABC: eval.py reports
``evals/<label>/summary.json`` (success_rate, wall time, a policy/env timing
breakdown), whose shape does not fit RolloutSet. It is the cheap way to get a
success rate without recording anything.

NOTE on submodule pins: the collector was generalized beyond DROID (and renamed
from ``collect_droid_rollouts.py``) on the branch this repo pins. A working tree
left on an older branch will not have it, and ``collect_rollouts`` says so
rather than failing obscurely.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from behavior_graphs.core.env import EnvAdapter, Policy, RolloutSet, TrainingData
from behavior_graphs.envs._process import default_repo_root, run_streaming
from behavior_graphs.envs.trajectory_dir import RolloutDirectory

logger = logging.getLogger(__name__)


#: Filesystem mtimes are not always finer than a second, and eval.py can
#: finish inside one, so a summary written in the second the subprocess
#: started still counts as fresh.
_MTIME_SLACK_NS = 2_000_000_000


class RgrlCheckpoint(Policy):
    """Handle to a checkpoint (``latest.pt``) produced by the rgrl submodule."""

    def __init__(self, checkpoint_path: Path, metadata: dict[str, Any] | None = None) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._metadata = dict(metadata or {})

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RgrlStoreTrainingData(TrainingData):
    """A robo_dataset store consumed directly by rgrl's ``store_path`` source kind.

    Self-describing (unlike a raw robomimic HDF5, no obs-key sniffing needed):
    ``robo_dataset.store_factory.open_store`` reads the store's own meta.json.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def num_episodes(self) -> int:
        return len(self.open_store().episode_ends)

    def open_store(self) -> Any:
        from robo_dataset.store_factory import open_store

        return open_store(str(self._path))


class RgrlAdapter(EnvAdapter):
    """Environment adapter backed by the rgrl submodule.

    Unlike :class:`DiffusionPolicyEnvAdapter`, this adapter is constructed
    from hydra **group-selector names**, not dataset paths -- rgrl resolves
    ``data``/``target``/``policy``/``training``/``env`` to real configs under
    ``configs/`` itself (see ``third_party/rgrl/configs/train.yaml``).

    Args:
        data: hydra ``data`` group selector (e.g. ``"transport_mh_success_only"``).
        target: hydra ``target`` group selector.
        env: hydra ``env`` group selector used by :meth:`eval_policy`; ``None``
            if this adapter is only ever used for training.
        policy: hydra ``policy`` group selector.
        training: hydra ``training`` group selector.
        repo_root: root of the behavior_graphs checkout; auto-detected when omitted.
        submodule: submodule path relative to ``repo_root``.
        python_path: interpreter of the submodule's environment. Defaults to
            the uv-managed venv inside the submodule
            (``<submodule>/.venv/bin/python``, created by ``uv sync --extra eval``).
    """

    name = "rgrl"

    def __init__(
        self,
        data: str,
        target: str = "transport_mh_state",
        env: str | None = "robomimic_transport",
        policy: str = "diffusion",
        training: str = "default",
        repo_root: str | None = None,
        submodule: str = "third_party/rgrl",
        python_path: str | None = None,
    ) -> None:
        self.data = data
        self.target = target
        self.env = env
        self.policy = policy
        self.training = training
        self.repo_root = Path(repo_root) if repo_root is not None else default_repo_root()
        self.submodule = submodule
        self.python_path = python_path

    @property
    def submodule_dir(self) -> Path:
        """Absolute path to the rgrl submodule checkout."""
        path = self.repo_root / self.submodule
        if not (path / "train.py").exists():
            raise FileNotFoundError(
                f"rgrl submodule not found at {path}; is the submodule checked out?"
            )
        return path

    def training_data(self) -> TrainingData:
        import yaml

        data_cfg = self.submodule_dir / "configs" / "data" / f"{self.data}.yaml"
        cfg = yaml.safe_load(data_cfg.read_text())
        source = cfg.get("store_path") or cfg.get("path")
        if source is None:
            raise ValueError(f"{data_cfg} has neither store_path nor path")
        return RgrlStoreTrainingData(source)

    def _train_args(self, output_dir: Path, **overrides: Any) -> list:
        """Hydra CLI args for train.py (no interpreter prefix)."""
        args = [
            "train.py",
            f"data={self.data}",
            f"target={self.target}",
            f"policy={self.policy}",
            f"training={self.training}",
            f"out={output_dir}",
        ]
        args += [f"{key}={value}" for key, value in overrides.items()]
        return args

    def _train_metadata(self, output_dir: Path) -> dict[str, Any]:
        return {
            "env": self.name,
            "data": self.data,
            "target": self.target,
            "policy": self.policy,
            "training": self.training,
            "python": str(self._python()),
            "train_output_dir": str(output_dir),
        }

    def train_policy(self, output_dir: Path, **overrides: Any) -> Policy:
        """Train via the submodule's ``train.py``; returns the final checkpoint.

        Refuses if ``output_dir/latest.pt`` already exists: rgrl itself
        auto-resumes in place on relaunch against the same ``out`` dir (see
        ``train_loop.py``), which would silently continue a stale run under a
        reused name instead of failing loudly -- use a fresh ``output_dir``
        per run, matching :class:`DiffusionPolicyEnvAdapter`'s own
        refuse-if-``checkpoints/``-exists convention.
        """
        output_dir = Path(output_dir).resolve()
        if (output_dir / "latest.pt").exists():
            raise FileExistsError(
                f"REFUSING: {output_dir / 'latest.pt'} exists -- rgrl would auto-resume "
                f"from it rather than start fresh. Use a new output_dir."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_in_submodule(
            self._train_args(output_dir, **overrides), log_path=output_dir / "subprocess.log"
        )
        return RgrlCheckpoint(output_dir / "latest.pt", metadata=self._train_metadata(output_dir))

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir: Path,
        num_episodes: int,
        seed: int = 0,
        *,
        rollout_env: str | None = None,
        capture_layers: Sequence[str] | None = None,
        assemble: bool = True,
        **kwargs: Any,
    ) -> RolloutSet | Path:
        """Collect rollouts via the submodule's ``collect_rollouts`` CLI.

        Two subprocesses, because rgrl's collector and this repo's canonical
        layout meet at a conversion step rather than directly:

        1. ``python -m rgrl.inference.collect_rollouts`` writes
           ``<output_dir>/recordings/ep_<ts>/`` — per-camera MP4s,
           ``embeddings_<layer>.npz`` per ``capture_layers`` entry, and a
           per-episode ``metadata.yaml`` carrying the success label. (A real
           DROID session also writes ``trajectory.h5``; a sim session has no
           analog and skips it.)
        2. ``scripts/utils/assemble_droid_sync_recordings.py`` folds that tree
           into ``episodes/`` + ``media/`` + ``policy_embeddings/`` via
           ``write_rollout_dir``. It reads only timestamps, the online
           embeddings, the videos and the per-episode metadata — never
           ``trajectory.h5`` — which is why one assembler serves both the real
           robot and sim.

        ``rollout_env`` selects the backend from the submodule's
        ``configs/rollout_env/`` (``robomimic`` for sim, ``droid`` for the real
        robot); it defaults to ``robomimic`` when this adapter has a sim ``env``
        configured and ``droid`` otherwise. ``capture_layers`` names the policy
        taps to record; without it the collector runs but writes no embeddings,
        so a behavior graph built from the result would have nothing to cluster.

        Only ``sync`` and ``async_chunk`` modes pair an activation with the
        action it produced; ``async_streaming`` can attribute a future chunk's
        activation to the current step, so pass ``mode`` accordingly and
        re-extract offline if you need embeddings from a streaming session.

        Pass ``assemble=False`` to stop after step 1 and keep the raw
        ``recordings/`` tree — the right choice for an async-streaming session
        that needs offline extraction before assembly.
        """
        output_dir = Path(output_dir).resolve()
        module = "rgrl.inference.collect_rollouts"
        if not (self.submodule_dir / "rgrl" / "inference" / "collect_rollouts.py").exists():
            raise FileNotFoundError(
                f"{self.submodule_dir} has no {module}: this checkout predates the "
                "collector being generalized beyond DROID. The recorded submodule "
                "pin carries it; the working tree is on a branch that does not. "
                f"Check out the pin (git -C {self.submodule_dir} checkout "
                "<gitlink sha>) or a branch containing it."
            )

        backend = rollout_env or ("robomimic" if self.env else "droid")
        if not (self.submodule_dir / "configs" / "rollout_env" / f"{backend}.yaml").exists():
            available = sorted(
                p.stem for p in (self.submodule_dir / "configs" / "rollout_env").glob("*.yaml")
            )
            raise ValueError(f"no rollout_env {backend!r} (available: {available})")

        layers = list(capture_layers or [])
        args = [
            "-m",
            module,
            f"ckpt={policy.checkpoint_path}",
            f"rollout_env={backend}",
            f"output_dir={output_dir}",
            f"num_episodes={num_episodes}",
            f"rollout_env.seed={seed}",
        ]
        if layers:
            args.append("capture_layers=[" + ",".join(layers) + "]")
        args += [f"{key}={value}" for key, value in kwargs.items()]
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_in_submodule(args, log_path=output_dir / "collect_subprocess.log")

        if not assemble:
            # Deliberately a Path: see the docstring. Constructing a
            # RolloutDirectory here raised FileNotFoundError on the metadata
            # assembly had not written yet.
            return output_dir

        if not layers:
            logger.warning(
                "no capture_layers requested: assembling %s with videos and outcomes "
                "but no policy_embeddings, so the pipeline's default feature source "
                "will find nothing",
                output_dir,
            )
        assembler = self.repo_root / "scripts" / "utils" / "assemble_droid_sync_recordings.py"
        assemble_args = [str(assembler), str(output_dir)]
        for layer in layers:
            assemble_args += ["--layer", layer]
        # The assembler imports this repo, not the submodule's env, so it runs
        # under the interpreter that imported this module rather than
        # self._python(). Its cwd is the repo root for the same reason.
        run_streaming(
            [sys.executable, *assemble_args],
            cwd=self.repo_root,
            log_path=output_dir / "assemble.log",
        )
        return RolloutDirectory(output_dir)

    def eval_policy(
        self,
        policy: Policy,
        episodes: int | None = None,
        env: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Run the submodule's ``eval.py`` and return the parsed ``summary.json``.

        Not part of the :class:`EnvAdapter` ABC: rgrl's eval output
        (success_rate, wall time, policy/env timing) doesn't fit
        :class:`RolloutSet`'s per-episode trajectory contract, so this is a
        narrower, adapter-specific accessor for the numbers this experiment
        actually needs. Results accumulate under
        ``<checkpoint_dir>/evals/<label>/`` (one dir per distinct eval
        config) -- the freshest one written by *this* invocation is returned.
        """
        env_name = env or self.env
        if env_name is None:
            raise ValueError("no env group selector configured; pass env=... explicitly")
        ckpt_dir = policy.checkpoint_path.parent
        args = ["eval.py", f"ckpt={policy.checkpoint_path}", f"env={env_name}"]
        if episodes is not None:
            args.append(f"env.episodes={episodes}")
        args += [f"{key}={value}" for key, value in overrides.items()]
        # Newest-by-mtime alone cannot tell this run's output from a previous
        # one's: if eval.py exits 0 having written nothing -- an early return, a
        # different label dir than expected -- the freshest *pre-existing*
        # summary comes back as though it were the result. This adapter is the
        # source of the success-rate numbers, so require a file that did not
        # exist, or was not this recent, before the subprocess started.
        before = {p: p.stat().st_mtime_ns for p in (ckpt_dir / "evals").glob("*/summary.json")}
        started = time.time_ns()
        self._run_in_submodule(args, log_path=ckpt_dir / "eval_subprocess.log")
        fresh = [
            p
            for p in (ckpt_dir / "evals").glob("*/summary.json")
            if p.stat().st_mtime_ns > max(before.get(p, 0), started - _MTIME_SLACK_NS)
        ]
        if not fresh:
            stale = len(before)
            raise FileNotFoundError(
                f"eval.py exited 0 but wrote no summary.json under {ckpt_dir / 'evals'} "
                f"({stale} from earlier runs left untouched); refusing to report one of "
                f"those as this run's result. See {ckpt_dir / 'eval_subprocess.log'}."
            )
        return json.loads(max(fresh, key=lambda p: p.stat().st_mtime_ns).read_text())

    def _python(self) -> Path:
        """Interpreter of the submodule's environment (uv venv by default)."""
        if self.python_path is not None:
            return Path(self.python_path)
        venv_python = self.submodule_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():
            raise FileNotFoundError(
                f"no environment at {venv_python}; create it with "
                f"'cd {self.submodule_dir} && uv sync --extra eval' or pass "
                f"python_path explicitly"
            )
        return venv_python

    def _run_in_submodule(self, args: Sequence[str], log_path: Path) -> None:
        """Run a python script in the submodule's environment, cwd at its root."""
        run_streaming([str(self._python()), *args], cwd=self.submodule_dir, log_path=log_path)
