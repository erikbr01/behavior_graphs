"""End-to-end integration test on robomimic lift (image-based).

Same flow as the state-based test but with the hybrid image policy and
video recording. Requires an absolute-action image dataset; convert with::

    .venv/bin/python diffusion_policy/scripts/robomimic_dataset_conversion.py \
        -i .../lift/mh/image.hdf5 -o <datasets>/robomimic/lift/mh/image_abs.hdf5

(the path must contain ``robomimic/<task>/<type>/``) and set
BEHAVIOR_GRAPHS_LIFT_IMAGE_DATASET accordingly.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATASET = os.environ.get(
    "BEHAVIOR_GRAPHS_LIFT_IMAGE_DATASET",
    "${DATA_PATH}/datasets/robomimic/lift/mh/image_abs.hdf5",
)

pytestmark = pytest.mark.integration


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run(args, cwd=REPO, env=env, check=True)


def test_train_collect_pipeline_image(tmp_path):
    if not Path(DATASET).exists():
        pytest.skip(f"lift image_abs dataset not found at {DATASET}")
    train_dir = tmp_path / "train"
    rollout_dir = tmp_path / "rollouts"
    run_dir = tmp_path / "graph_run"

    _run(
        [
            sys.executable,
            "scripts/train_policy.py",
            "env=robomimic_lift_image",
            f"env.dataset_path={DATASET}",
            "++train_overrides={training: {num_epochs: 1, rollout_every: 1, "
            "checkpoint_every: 1, sample_every: 10}, logging: {mode: offline}, "
            "task: {env_runner: {n_train: 0, n_train_vis: 0, n_test: 5, "
            "n_test_vis: 0, n_envs: 5}}}",
            f"hydra.run.dir={train_dir}",
        ]
    )
    checkpoints = list((train_dir / "checkpoints").glob("*.ckpt"))
    assert checkpoints, "training produced no checkpoint"

    _run(
        [
            sys.executable,
            "scripts/collect_rollouts.py",
            "env=robomimic_lift_image",
            f"env.dataset_path={DATASET}",
            f'checkpoint="{checkpoints[0]}"',
            "num_episodes=10",
            "seed=200000",
            # episode saving requires sequential execution in the image runner
            "++rollout_kwargs={n_envs: 1, num_vis: 3}",
            f"hydra.run.dir={rollout_dir}",
        ]
    )
    assert (rollout_dir / "episodes" / "metadata.yaml").exists()
    assert list((rollout_dir / "policy_embeddings").glob("*.npz"))
    assert list((rollout_dir / "media").glob("*.mp4")), "no rollout videos recorded"

    _run(
        [
            sys.executable,
            "-m",
            "behavior_graphs.pipeline.run",
            f"rollout_dir={rollout_dir}",
            "features.layer=bottleneck_plan_t0",
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
    assert "values" in result.metrics
