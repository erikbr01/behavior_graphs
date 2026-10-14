"""End-to-end integration test for VLA Foundry (TRI LBM eval).

Launches the embedding-capturing gRPC policy server, runs the LBM Drake
simulator in Docker, and builds a behavior graph from the rollouts.
GPU- and Docker-bound and slow::

    pytest -m integration tests/integration/test_vla_foundry.py

Requires the vla_foundry submodule env (uv sync --group inference), the
toyotaresearch/lbm-eval-oss:vla-foundry Docker image, and the
TRI-ML/Foundry-VLA-1.7B-sim checkpoint in the HF cache.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
CHECKPOINT_CACHED = any(Path(HF_HOME).glob("hub/models--TRI-ML--Foundry-VLA-1.7B-sim/snapshots/*"))


def _docker_image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", "toyotaresearch/lbm-eval-oss:vla-foundry"],
        capture_output=True,
    )
    return result.returncode == 0


pytestmark = pytest.mark.integration


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run(args, cwd=REPO, env=env, check=True)


def test_collect_and_pipeline(tmp_path):
    if not CHECKPOINT_CACHED:
        pytest.skip("TRI-ML/Foundry-VLA-1.7B-sim not in the HF cache")
    if not _docker_image_present():
        pytest.skip("toyotaresearch/lbm-eval-oss:vla-foundry Docker image not present")
    rollout_dir = tmp_path / "rollouts"
    run_dir = tmp_path / "graph_run"

    _run(
        [
            sys.executable,
            "scripts/collect_rollouts.py",
            "env=vla_foundry",
            "++env.gpu=0",
            "checkpoint=TRI-ML/Foundry-VLA-1.7B-sim",
            "num_episodes=2",
            f"hydra.run.dir={rollout_dir}",
        ]
    )
    assert (rollout_dir / "episodes" / "metadata.yaml").exists()
    assert (rollout_dir / "policy_embeddings" / "action_conditioning.npz").exists()
    assert list((rollout_dir / "media").glob("*.mp4"))

    _run(
        [
            sys.executable,
            "-m",
            "behavior_graphs.pipeline.run",
            f"rollout_dir={rollout_dir}",
            "features.layer=action_conditioning",
            "windowing.width=3",
            "pipeline/reduction@reduction=pca",
            "reduction.n_components=16",
            "pipeline/clustering@clustering=kmeans",
            "clustering.n_clusters=4",
            "viz/viz_reduction@viz_reduction=pca_2d",
            f"hydra.run.dir={run_dir}",
        ]
    )
    from behavior_graphs.pipeline.artifacts import RunArtifacts

    result = RunArtifacts(run_dir).load_result()
    assert result.graph is not None and len(result.graph.cluster_nodes) > 0
    assert result.labels.shape[0] == result.windows.num_windows
