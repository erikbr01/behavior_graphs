"""behavior_graphs.envs.safe_import: shared SAFE-import helpers."""

import logging

import numpy as np

from behavior_graphs.envs.safe_import import slugify, write_safe_task
from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory


def test_slugify_lowercases_and_collapses_punctuation():
    assert slugify("Pick up the Knife!") == "pick_up_the_knife"
    assert slugify("  extra   spaces  ") == "extra_spaces"


def test_write_safe_task_writes_provenance_and_returns_reader(tmp_path):
    log = logging.getLogger("test_safe_import")
    emb = np.zeros((7, 4), dtype=np.float32)
    dest = tmp_path / "safe_widowx_lift_red_bottle"

    result = write_safe_task(
        log,
        dest,
        lengths=[3, 4],
        successes=[True, False],
        embeddings={"openvla_hidden_mean": emb},
        dataset_label="SAFE WidowX/OpenVLA (arXiv:2506.09937)",
        task_id=2,
        task_description="Lift the red bottle",
        episode_extras=[{"safe_episode_idx": 0}, {"safe_episode_idx": 1}],
    )

    assert isinstance(result, TrajectoryDirectory)
    reread = TrajectoryDirectory(dest)
    assert [ep.success for ep in reread.episodes()] == [True, False]
    assert reread.source == {
        "kind": "safe_release",
        "dataset": "SAFE WidowX/OpenVLA (arXiv:2506.09937)",
        "url": "https://github.com/vla-safe/SAFE",
        "task_id": 2,
        "task_description": "Lift the red bottle",
    }
    assert reread.episodes()[1].extra["safe_episode_idx"] == 1
