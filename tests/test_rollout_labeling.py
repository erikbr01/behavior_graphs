"""Post-hoc success labeling + session merging for real-robot rollouts.

Outcomes are normally labeled inline by the on-robot collector
(``third_party/diffusion_policy/scripts/droid_collect_rollouts.py``);
``set_episode_success`` is the primitive for fixing a label afterwards, and
``merge_trajectory_dirs`` joins per-session directories into one pipeline
input.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from behavior_graphs.envs.trajectory_dir import (
    TrajectoryDirectory,
    merge_trajectory_dirs,
    set_episode_success,
    write_rollout_dir,
)

REPO = Path(__file__).resolve().parents[1]


def _load_script(name, path=None):
    if path is None:
        path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


merge_droid_dirs_script = _load_script(
    "merge_droid_dirs", REPO / "scripts" / "utils" / "merge_droid_dirs.py"
)

DIM = 4
LAYER = "emb"


def _make_rollouts(path, lengths, successes, with_videos=True, extras=None):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(sum(lengths), DIM)).astype(np.float32)
    videos = None
    if with_videos:
        src = path.parent / f"_vid_{path.name}.mp4"
        src.write_bytes(b"fake mp4")
        videos = {i: src for i in range(len(lengths))}
    return write_rollout_dir(
        path,
        lengths,
        successes,
        {LAYER: emb},
        video_paths=videos,
        episode_extras=extras,
    )


class TestSetEpisodeSuccess:
    def test_label_and_media_rename(self, tmp_path):
        d = tmp_path / "rollouts"
        _make_rollouts(d, [5, 6], [None, None])
        assert (d / "media" / "ep0000_unknown.mp4").exists()

        set_episode_success(d, 0, True)
        set_episode_success(d, 1, False, note="dropped the ball")

        eps = TrajectoryDirectory(d).episodes()
        assert eps[0].success is True
        assert eps[1].success is False
        assert eps[1].extra["label_note"] == "dropped the ball"
        assert (d / "media" / "ep0000_success.mp4").exists()
        assert (d / "media" / "ep0001_failure.mp4").exists()
        assert not (d / "media" / "ep0000_unknown.mp4").exists()

    def test_relabel_to_unknown(self, tmp_path):
        d = tmp_path / "rollouts"
        _make_rollouts(d, [5], [True])
        set_episode_success(d, 0, None)
        assert TrajectoryDirectory(d).episodes()[0].success is None
        assert (d / "media" / "ep0000_unknown.mp4").exists()

    def test_preserves_source_and_extras(self, tmp_path):
        d = tmp_path / "rollouts"
        rng = np.random.default_rng(0)
        write_rollout_dir(
            d,
            [4],
            [None],
            {LAYER: rng.normal(size=(4, DIM)).astype(np.float32)},
            source={"kind": "droid_session"},
            episode_extras=[{"instruction": "pick up the kendama"}],
        )
        set_episode_success(d, 0, True)
        raw = yaml.safe_load((d / "episodes" / "metadata.yaml").read_text())
        assert raw["source"] == {"kind": "droid_session"}
        assert raw["episodes"][0]["instruction"] == "pick up the kendama"

    def test_unknown_episode_raises(self, tmp_path):
        d = tmp_path / "rollouts"
        _make_rollouts(d, [5], [None], with_videos=False)
        with pytest.raises(KeyError):
            set_episode_success(d, 7, True)


class TestMerge:
    def test_merge_two_sessions(self, tmp_path):
        a = tmp_path / "sess_a"
        b = tmp_path / "sess_b"
        _make_rollouts(a, [5, 6], [True, False])
        _make_rollouts(b, [7], [None], extras=[{"instruction": "x"}])

        merged = merge_trajectory_dirs([a, b], tmp_path / "merged")
        eps = merged.episodes()
        assert [ep.num_samples for ep in eps] == [5, 6, 7]
        assert [ep.success for ep in eps] == [True, False, None]
        assert eps[2].extra["instruction"] == "x"
        assert merged.embeddings(LAYER).shape == (18, DIM)
        assert merged.video_path(2).name == "ep0002_unknown.mp4"
        assert merged.source["kind"] == "merged"
        assert len(merged.source["sources"]) == 2

    def test_layer_mismatch_rejected(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_rollouts(a, [3], [True], with_videos=False)
        rng = np.random.default_rng(0)
        write_rollout_dir(b, [3], [True], {"other": rng.normal(size=(3, DIM)).astype(np.float32)})
        with pytest.raises(ValueError, match="embedding layers differ"):
            merge_trajectory_dirs([a, b], tmp_path / "merged")

    def test_merge_script_with_recordings(self, tmp_path, capsys):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_rollouts(a, [3, 4], [True, False])
        _make_rollouts(b, [5], [None])
        # DROID raw recordings next to the canonical layout.
        for src, eps in ((a, 2), (b, 1)):
            for i in range(eps):
                rec = src / "recordings" / f"ep{i:04d}"
                rec.mkdir(parents=True)
                (rec / "trajectory.h5").write_bytes(b"h5")

        out = tmp_path / "merged"
        rc = merge_droid_dirs_script.main([str(a), str(b), "--output", str(out)])
        assert rc == 0
        assert "3 episodes" in capsys.readouterr().out
        assert (out / "recordings" / "ep0002" / "trajectory.h5").exists()
        assert len(TrajectoryDirectory(out).episodes()) == 3

    def test_merge_script_refuses_nonempty_output(self, tmp_path):
        a = tmp_path / "a"
        _make_rollouts(a, [3], [True], with_videos=False)
        out = tmp_path / "merged"
        out.mkdir()
        (out / "junk").write_text("x")
        with pytest.raises(SystemExit):
            merge_droid_dirs_script.main([str(a), "--output", str(out)])
