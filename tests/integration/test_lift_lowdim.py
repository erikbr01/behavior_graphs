"""End-to-end integration test on robomimic lift (state-based).

Trains a tiny diffusion policy (2 epochs), collects rollouts with
per-timestep policy embeddings, runs the behavior-graph pipeline, and checks
the artifact directory. Slow and GPU-bound::

    pytest -m integration tests/integration/test_lift_lowdim.py

Requires the diffusion_policy submodule env (scripts/setup_env_uv.sh) and a
robomimic lift dataset; set BEHAVIOR_GRAPHS_LIFT_DATASET to its low_dim_abs.hdf5.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        return len([ln for ln in out.strip().splitlines() if ln.strip()])
    except Exception:
        return 0


REPO = Path(__file__).resolve().parents[2]
# os.path.expandvars, because argparse/os.environ.get expand nothing: a
# literal "${DATA_PATH}/..." default makes the skip guard below fire and the
# whole tier report success by never running. The old default also pointed at
# a directory that exists on no machine here.
DATASET = os.path.expandvars(
    os.environ.get(
        "BEHAVIOR_GRAPHS_LIFT_DATASET",
        "${DATA_PATH}/datasets/robomimic/lift/ph/low_dim.hdf5",
    )
)

pytestmark = pytest.mark.integration


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run(args, cwd=REPO, env=env, check=True)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("lift_lowdim")


def test_train_collect_pipeline(workdir):
    if not Path(DATASET).exists():
        pytest.skip(f"lift dataset not found at {DATASET}")
    train_dir = workdir / "train"
    rollout_dir = workdir / "rollouts"
    run_dir = workdir / "graph_run"

    _run(
        [
            sys.executable,
            "scripts/train_policy.py",
            "env=robomimic_lift_lowdim",
            f"env.dataset_path={DATASET}",
            "++train_overrides={training: {num_epochs: 2, rollout_every: 1, "
            "checkpoint_every: 1, sample_every: 10}, logging: {mode: offline}, "
            "task: {env_runner: {n_train: 0, n_train_vis: 0, n_test: 10, "
            "n_test_vis: 0, n_envs: 10}}}",
            f"hydra.run.dir={train_dir}",
        ]
    )
    checkpoints = list((train_dir / "checkpoints").glob("*.ckpt"))
    assert checkpoints, "training produced no checkpoint"

    _run(
        [
            sys.executable,
            "scripts/collect_rollouts.py",
            "env=robomimic_lift_lowdim",
            f"env.dataset_path={DATASET}",
            f'checkpoint="{checkpoints[0]}"',
            "num_episodes=20",
            "seed=100000",
            "++rollout_kwargs={n_envs: 10, num_vis: 0}",
            f"hydra.run.dir={rollout_dir}",
        ]
    )
    assert (rollout_dir / "episodes" / "metadata.yaml").exists()
    assert list((rollout_dir / "policy_embeddings").glob("*.npz"))

    _run(
        [
            sys.executable,
            "-m",
            "behavior_graphs.pipeline.run",
            f"rollout_dir={rollout_dir}",
            "features.layer=bottleneck_plan_t0",
            "windowing.width=5",
            "windowing.stride=2",
            "pipeline/reduction@reduction=pca",
            "reduction.n_components=16",
            "pipeline/clustering@clustering=kmeans",
            "clustering.n_clusters=5",
            "viz/viz_reduction@viz_reduction=pca_2d",
            f"hydra.run.dir={run_dir}",
        ]
    )
    for artifact in ("graph.json", "labels.npy", "coords_2d.npy", "metrics.json", "windows.npz"):
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    sys.path.insert(0, str(REPO / "src"))
    from behavior_graphs.pipeline.artifacts import RunArtifacts

    result = RunArtifacts(run_dir).load_result()
    assert result.graph is not None and len(result.graph.cluster_nodes) > 0
    assert result.labels.shape[0] == result.windows.num_windows
    assert "graph_stats" in result.metrics


def test_train_multi_gpu(workdir):
    n_gpus = _cuda_device_count()
    if n_gpus < 2:
        pytest.skip(f"need ≥2 GPUs, found {n_gpus}")
    if not Path(DATASET).exists():
        pytest.skip(f"lift dataset not found at {DATASET}")
    train_dir = workdir / "train_multi_gpu"
    _run(
        [
            sys.executable,
            "scripts/train_policy.py",
            "env=robomimic_lift_lowdim",
            f"env.dataset_path={DATASET}",
            "num_gpus=2",
            "++train_overrides={training: {num_epochs: 2, rollout_every: 1, "
            "checkpoint_every: 1, sample_every: 10}, logging: {mode: offline}, "
            "task: {env_runner: {n_train: 0, n_train_vis: 0, n_test: 2, "
            "n_test_vis: 0, n_envs: 2}}}",
            f"hydra.run.dir={train_dir}",
        ]
    )
    checkpoints = list((train_dir / "checkpoints").glob("*.ckpt"))
    assert checkpoints, "multi-GPU training produced no checkpoint"
