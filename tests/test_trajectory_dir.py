"""Round-trip and contract tests for the canonical rollout layout."""

import numpy as np
import pytest

from behavior_graphs.core.env import CheckpointPolicy, EnvAdapter, RolloutSet
from behavior_graphs.envs.trajectory_dir import (
    EpisodeSubset,
    PooledTrajectorySet,
    RolloutDirectory,
    TrajectoryDirectory,
    add_embedding_layer,
    write_rollout_dir,
    write_trajectory_dir,
)


@pytest.fixture
def rollout_dir(tmp_path):
    rng = np.random.default_rng(0)
    lengths = [30, 45, 20]
    successes = [True, False, None]
    emb = {"pi05": rng.normal(size=(sum(lengths), 16)).astype(np.float32)}
    write_rollout_dir(tmp_path / "r", lengths, successes, emb)
    return tmp_path / "r"


def test_roundtrip(rollout_dir):
    rollouts = RolloutDirectory(rollout_dir)
    episodes = rollouts.episodes()
    assert [ep.num_samples for ep in episodes] == [30, 45, 20]
    assert [ep.success for ep in episodes] == [True, False, None]
    assert episodes[1].sample_start_idx == 30
    assert episodes[2].sample_end_idx == 95
    assert rollouts.embedding_layers() == ["pi05"]
    emb = rollouts.embeddings("pi05")
    assert emb.shape == (95, 16)
    assert emb.dtype == np.float32
    assert rollouts.video_path(0) is None


def test_unknown_layer_raises(rollout_dir):
    rollouts = RolloutDirectory(rollout_dir)
    with pytest.raises(FileNotFoundError, match="pi05"):
        rollouts.embeddings("nope")


def test_misaligned_embeddings_rejected(tmp_path):
    emb = {"x": np.zeros((10, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="rows"):
        write_rollout_dir(tmp_path / "r", [5, 6], [True, True], emb)


def test_env_adapter_default_loaders(rollout_dir, tmp_path):
    class DummyAdapter(EnvAdapter):
        name = "dummy"

        def training_data(self):
            raise NotImplementedError

        def train_policy(self, output_dir, **overrides):
            raise NotImplementedError

        def collect_rollouts(self, policy, output_dir, num_episodes, seed=0, **kwargs):
            raise NotImplementedError

    adapter = DummyAdapter()
    policy = adapter.load_policy(tmp_path / "ckpt.pt")
    assert isinstance(policy, CheckpointPolicy)
    assert policy.metadata["env"] == "dummy"
    rollouts = adapter.load_rollouts(rollout_dir)
    assert isinstance(rollouts, RolloutSet)
    assert len(rollouts.episodes()) == 3


class TestTrajectoryDirectory:
    """Base reader: provenance block, per-episode extras, subclassing."""

    def test_source_and_extras_roundtrip(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import (
            TrajectoryDirectory,
            write_trajectory_dir,
        )

        rng = np.random.default_rng(1)
        lengths = [10, 12]
        emb = {"bottleneck_plan_t0": rng.normal(size=(22, 8)).astype(np.float32)}
        write_trajectory_dir(
            tmp_path / "demos",
            lengths,
            [True, True],
            emb,
            source={"kind": "demos", "dataset": "/data/lift.hdf5"},
            episode_extras=[
                {"buffer_episode_idx": 3, "split": "train"},
                {"buffer_episode_idx": 7, "split": "train"},
            ],
        )
        demos = TrajectoryDirectory(tmp_path / "demos")
        assert demos.source == {"kind": "demos", "dataset": "/data/lift.hdf5"}
        assert demos.episodes()[0].extra["buffer_episode_idx"] == 3
        assert demos.episodes()[1].extra["split"] == "train"
        assert not isinstance(demos, RolloutSet)

    def test_rollout_directory_is_trajectory_directory(self, rollout_dir):
        from behavior_graphs.core.env import TrajectorySet
        from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

        rollouts = RolloutDirectory(rollout_dir)
        assert isinstance(rollouts, TrajectoryDirectory)
        assert isinstance(rollouts, TrajectorySet)
        assert isinstance(rollouts, RolloutSet)
        assert rollouts.source == {}

    def test_extras_length_mismatch_rejected(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import write_trajectory_dir

        emb = {"l": np.zeros((10, 4), dtype=np.float32)}
        with pytest.raises(ValueError, match="episode_extras"):
            write_trajectory_dir(tmp_path / "bad", [10], [True], emb, episode_extras=[{}, {}])


class TestEpisodeSubset:
    """The one in-memory (non-disk-backed) TrajectorySet implementation."""

    def test_renumbers_episodes_and_preserves_orig_index(self, rollout_dir):
        rollouts = RolloutDirectory(rollout_dir)
        subset = EpisodeSubset(rollouts, [2, 0])

        eps = subset.episodes()
        assert [ep.index for ep in eps] == [0, 1]
        assert [ep.extra["orig_index"] for ep in eps] == [2, 0]
        assert [ep.success for ep in eps] == [None, True]

    def test_slices_and_concatenates_embeddings_in_new_order(self, tmp_path):
        rng = np.random.default_rng(2)
        lengths = [3, 4, 5]
        base_emb = rng.normal(size=(sum(lengths), 6)).astype(np.float32)
        write_rollout_dir(tmp_path / "r", lengths, [True, False, True], {"pi05": base_emb})
        rollouts = RolloutDirectory(tmp_path / "r")

        subset = EpisodeSubset(rollouts, [2, 0])
        emb = subset.embeddings("pi05")
        assert emb.shape[0] == lengths[2] + lengths[0]
        for ep in subset.episodes():
            orig = rollouts.episodes()[ep.extra["orig_index"]]
            expected = base_emb[orig.sample_start_idx : orig.sample_end_idx]
            actual = emb[ep.sample_start_idx : ep.sample_end_idx]
            assert np.array_equal(actual, expected)

    def test_path_and_layers_delegate_to_base(self, rollout_dir):
        rollouts = RolloutDirectory(rollout_dir)
        subset = EpisodeSubset(rollouts, [1])
        assert subset.path == rollouts.path
        assert subset.embedding_layers() == rollouts.embedding_layers()


class TestPooledTrajectorySet:
    """Pooling episodes from several distinct TrajectorySets into one."""

    def _make_rollout_dir(self, tmp_path, name, lengths, successes, seed):
        rng = np.random.default_rng(seed)
        emb = {"pi05": rng.normal(size=(sum(lengths), 6)).astype(np.float32)}
        write_rollout_dir(tmp_path / name, lengths, successes, emb)
        return RolloutDirectory(tmp_path / name), emb["pi05"]

    def test_renumbers_and_concatenates_across_sources(self, tmp_path):
        a, emb_a = self._make_rollout_dir(tmp_path, "a", [3, 4], [True, False], seed=0)
        b, emb_b = self._make_rollout_dir(tmp_path, "b", [5], [True], seed=1)

        pooled = PooledTrajectorySet([a, b])
        eps = pooled.episodes()
        assert [ep.index for ep in eps] == [0, 1, 2]
        assert [ep.num_samples for ep in eps] == [3, 4, 5]
        assert [ep.success for ep in eps] == [True, False, True]
        assert [ep.extra["source_index"] for ep in eps] == [0, 0, 1]
        assert [ep.extra["orig_index"] for ep in eps] == [0, 1, 0]

        emb = pooled.embeddings("pi05")
        assert emb.shape == (12, 6)
        assert np.array_equal(emb[:7], emb_a)
        assert np.array_equal(emb[7:], emb_b)

    def test_rejects_mismatched_embedding_layers(self, tmp_path):
        a, _ = self._make_rollout_dir(tmp_path, "a", [3], [True], seed=0)
        rng = np.random.default_rng(2)
        write_rollout_dir(
            tmp_path / "b", [4], [True], {"other_layer": rng.normal(size=(4, 6)).astype(np.float32)}
        )
        b = RolloutDirectory(tmp_path / "b")
        with pytest.raises(ValueError, match="embedding layers differ"):
            PooledTrajectorySet([a, b])

    def test_rejects_empty_sources(self):
        with pytest.raises(ValueError, match="no source"):
            PooledTrajectorySet([])

    def test_pools_episode_subsets_not_just_full_directories(self, tmp_path):
        """The intended use: pool a per-task TRAIN split, not whole directories."""
        a, emb_a = self._make_rollout_dir(tmp_path, "a", [3, 4], [True, False], seed=0)
        b, emb_b = self._make_rollout_dir(tmp_path, "b", [5, 2], [True, False], seed=1)
        train_a = EpisodeSubset(a, [0])  # just the first episode of "a"
        train_b = EpisodeSubset(b, [1])  # just the second episode of "b"

        pooled = PooledTrajectorySet([train_a, train_b])
        assert [ep.num_samples for ep in pooled.episodes()] == [3, 2]
        emb = pooled.embeddings("pi05")
        assert np.array_equal(emb[:3], emb_a[:3])
        assert np.array_equal(emb[3:], emb_b[5:7])

    def test_concatenates_observations_and_actions_across_sources(self, tmp_path):
        rng = np.random.default_rng(3)
        lengths_a, lengths_b = [3, 4], [5]
        state_a = rng.normal(size=(sum(lengths_a), 2)).astype(np.float32)
        state_b = rng.normal(size=(sum(lengths_b), 2)).astype(np.float32)
        actions_a = rng.normal(size=(sum(lengths_a), 1)).astype(np.float32)
        actions_b = rng.normal(size=(sum(lengths_b), 1)).astype(np.float32)
        emb = {"pi05": rng.normal(size=(sum(lengths_a), 6)).astype(np.float32)}
        write_rollout_dir(
            tmp_path / "a",
            lengths_a,
            [True, False],
            emb,
            observations={"state": state_a},
            actions=actions_a,
        )
        write_rollout_dir(
            tmp_path / "b",
            lengths_b,
            [True],
            {"pi05": rng.normal(size=(sum(lengths_b), 6)).astype(np.float32)},
            observations={"state": state_b},
            actions=actions_b,
        )
        a = RolloutDirectory(tmp_path / "a")
        b = RolloutDirectory(tmp_path / "b")

        pooled = PooledTrajectorySet([a, b])
        assert pooled.observation_keys() == ["state"]
        obs = pooled.observations("state")
        assert np.array_equal(obs[:7], state_a)
        assert np.array_equal(obs[7:], state_b)
        actions = pooled.actions()
        assert np.array_equal(actions[:7], actions_a)
        assert np.array_equal(actions[7:], actions_b)

    def test_rejects_mismatched_observation_keys(self, tmp_path):
        rng = np.random.default_rng(4)
        write_rollout_dir(
            tmp_path / "a",
            [3],
            [True],
            {"pi05": rng.normal(size=(3, 6)).astype(np.float32)},
            observations={"state": rng.normal(size=(3, 2)).astype(np.float32)},
        )
        write_rollout_dir(
            tmp_path / "b", [4], [True], {"pi05": rng.normal(size=(4, 6)).astype(np.float32)}
        )
        a = RolloutDirectory(tmp_path / "a")
        b = RolloutDirectory(tmp_path / "b")
        with pytest.raises(ValueError, match="observation keys differ"):
            PooledTrajectorySet([a, b])

    def test_actions_none_when_no_source_has_actions(self, tmp_path):
        a, _ = self._make_rollout_dir(tmp_path, "a", [3], [True], seed=0)
        b, _ = self._make_rollout_dir(tmp_path, "b", [4], [True], seed=1)
        assert PooledTrajectorySet([a, b]).actions() is None

    def test_rejects_inconsistent_action_availability(self, tmp_path):
        rng = np.random.default_rng(5)
        write_rollout_dir(
            tmp_path / "a",
            [3],
            [True],
            {"pi05": rng.normal(size=(3, 6)).astype(np.float32)},
            actions=rng.normal(size=(3, 1)).astype(np.float32),
        )
        b, _ = self._make_rollout_dir(tmp_path, "b", [4], [True], seed=1)
        a = RolloutDirectory(tmp_path / "a")
        with pytest.raises(ValueError, match="inconsistent action availability"):
            PooledTrajectorySet([a, b]).actions()

    def test_video_path_delegates_to_originating_source(self, tmp_path):
        rng = np.random.default_rng(6)
        video_a = tmp_path / "vid_a.mp4"
        video_a.write_bytes(b"a")
        video_b = tmp_path / "vid_b.mp4"
        video_b.write_bytes(b"b")
        write_rollout_dir(
            tmp_path / "a",
            [3],
            [True],
            {"pi05": rng.normal(size=(3, 6)).astype(np.float32)},
            video_paths={0: video_a},
        )
        write_rollout_dir(
            tmp_path / "b",
            [4],
            [True],
            {"pi05": rng.normal(size=(4, 6)).astype(np.float32)},
            video_paths={0: video_b},
        )
        a = RolloutDirectory(tmp_path / "a")
        b = RolloutDirectory(tmp_path / "b")
        pooled = PooledTrajectorySet([a, b])
        assert pooled.video_path(0) == a.video_path(0)
        assert pooled.video_path(1) == b.video_path(0)


class TestCameraViews:
    """``media/<camera>/`` alongside the flat single-view layout."""

    @staticmethod
    def _write(tmp_path, video_paths, successes=(True, False)):
        emb = {"l": np.zeros((sum((10, 12)), 4), dtype=np.float32)}
        return write_rollout_dir(
            tmp_path / "r", [10, 12], list(successes), emb, video_paths=video_paths
        )

    @staticmethod
    def _clip(tmp_path, name):
        src = tmp_path / name
        src.write_bytes(b"\x00" * 16)
        return src

    def test_flat_layout_is_the_default_camera(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA

        clip = self._clip(tmp_path, "a.mp4")
        rollouts = self._write(tmp_path, {0: clip, 1: clip})

        assert rollouts.camera_names() == [DEFAULT_CAMERA]
        assert rollouts.video_path(0) == tmp_path / "r" / "media" / "ep0000_success.mp4"
        assert rollouts.video_path(0, DEFAULT_CAMERA) == rollouts.video_path(0)
        assert rollouts.video_paths(1) == {DEFAULT_CAMERA: rollouts.video_path(1)}
        assert rollouts.video_path(0, "wrist") is None

    def test_named_views_get_their_own_directories(self, tmp_path):
        wrist, ext = self._clip(tmp_path, "w.mp4"), self._clip(tmp_path, "e.mp4")
        rollouts = self._write(
            tmp_path, {0: {"wrist": wrist, "exterior": ext}, 1: {"exterior": ext}}
        )

        media = tmp_path / "r" / "media"
        assert rollouts.camera_names() == ["exterior", "wrist"]
        assert rollouts.video_paths(0) == {
            "exterior": media / "exterior" / "ep0000_success.mp4",
            "wrist": media / "wrist" / "ep0000_success.mp4",
        }
        # Episode 1 only has the one view; the other is absent, not None.
        assert rollouts.video_paths(1) == {"exterior": media / "exterior" / "ep0001_failure.mp4"}
        # camera=None keeps the single-view contract: the first view that has it.
        assert rollouts.video_path(0) == media / "exterior" / "ep0000_success.mp4"

    def test_flat_and_named_views_coexist(self, tmp_path):
        """A dataset assembled before per-camera media can gain views in place."""
        from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA

        clip = self._clip(tmp_path, "a.mp4")
        rollouts = self._write(tmp_path, {0: {DEFAULT_CAMERA: clip, "wrist": clip}, 1: clip})

        assert rollouts.camera_names() == [DEFAULT_CAMERA, "wrist"]
        assert (tmp_path / "r" / "media" / "ep0000_success.mp4").is_file()
        assert (tmp_path / "r" / "media" / "wrist" / "ep0000_success.mp4").is_file()
        assert list(rollouts.video_paths(0)) == [DEFAULT_CAMERA, "wrist"]

    def test_a_named_primary_camera_comes_first(self, tmp_path):
        """Which view is "the" view is otherwise alphabetical — see primary_camera."""
        from behavior_graphs.envs.trajectory_dir import RolloutDirectory

        wrist, ext = self._clip(tmp_path, "w.mp4"), self._clip(tmp_path, "e.mp4")
        self._write(tmp_path, {0: {"wrist": wrist, "exterior": ext}, 1: {"wrist": wrist}})
        media = tmp_path / "r" / "media"

        rollouts = RolloutDirectory(tmp_path / "r", primary_camera="wrist")

        assert rollouts.camera_names() == ["wrist", "exterior"]
        assert rollouts.video_path(0) == media / "wrist" / "ep0000_success.mp4"
        assert list(rollouts.video_paths(0)) == ["wrist", "exterior"]
        # Every view is still there, and still reachable by name.
        assert rollouts.video_path(0, "exterior") == media / "exterior" / "ep0000_success.mp4"

    @pytest.mark.parametrize("primary", [None, "", "not_a_camera"])
    def test_an_absent_primary_leaves_the_order_alphabetical(self, tmp_path, primary):
        """A group-wide default is safe for a dataset that lacks that camera."""
        from behavior_graphs.envs.trajectory_dir import RolloutDirectory

        wrist, ext = self._clip(tmp_path, "w.mp4"), self._clip(tmp_path, "e.mp4")
        self._write(tmp_path, {0: {"wrist": wrist, "exterior": ext}, 1: {"wrist": wrist}})

        rollouts = RolloutDirectory(tmp_path / "r", primary_camera=primary)

        assert rollouts.camera_names() == ["exterior", "wrist"]

    def test_relabeling_success_renames_every_view(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import set_episode_success

        clip = self._clip(tmp_path, "a.mp4")
        self._write(tmp_path, {0: {"wrist": clip, "exterior": clip}})
        set_episode_success(tmp_path / "r", 0, False)

        rollouts = RolloutDirectory(tmp_path / "r")
        assert [p.name for p in rollouts.video_paths(0).values()] == [
            "ep0000_failure.mp4",
            "ep0000_failure.mp4",
        ]

    def test_merge_carries_every_view(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import merge_trajectory_dirs

        clip = self._clip(tmp_path, "a.mp4")
        emb = {"l": np.zeros((5, 4), dtype=np.float32)}
        for name in ("s1", "s2"):
            write_rollout_dir(
                tmp_path / name,
                [5],
                [True],
                emb,
                video_paths={0: {"wrist": clip, "exterior": clip}},
            )
        merged = merge_trajectory_dirs([tmp_path / "s1", tmp_path / "s2"], tmp_path / "m")

        assert merged.camera_names() == ["exterior", "wrist"]
        # Re-indexed in source order, both views intact.
        assert set(merged.video_paths(1)) == {"exterior", "wrist"}
        assert merged.video_paths(1)["wrist"].name == "ep0001_success.mp4"

    def test_empty_view_directory_is_not_a_camera(self, tmp_path):
        clip = self._clip(tmp_path, "a.mp4")
        rollouts = self._write(tmp_path, {0: clip})
        (tmp_path / "r" / "media" / "wrist").mkdir()

        assert "wrist" not in RolloutDirectory(tmp_path / "r").camera_names()
        assert rollouts.camera_names()  # the flat view is still there


class TestAddEmbeddingLayer:
    """Adding a representation to a directory that already exists.

    The write is what a video-feature extraction materializes so that
    everything addressing representations by layer name (the app's picker, the
    sweep runner's per-layer passes, the outcome classifiers) can read it.
    """

    def _dir(self, tmp_path):
        return write_trajectory_dir(
            tmp_path / "t",
            [3, 2],
            [True, False],
            embeddings={"pi05": np.zeros((5, 4), dtype=np.float32)},
        )

    def test_adds_a_layer_beside_the_existing_ones(self, tmp_path):
        out = self._dir(tmp_path)
        new = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)

        add_embedding_layer(out.path, "dino_patch", new)

        reread = TrajectoryDirectory(out.path)
        assert reread.embedding_layers() == ["dino_patch", "pi05"]
        assert np.array_equal(reread.embeddings("dino_patch"), new)

    def test_no_temp_file_is_left_registered_as_a_layer(self, tmp_path):
        """`embedding_layers()` globs the directory, so a stray temp file is a layer."""
        out = self._dir(tmp_path)
        add_embedding_layer(out.path, "dino_patch", np.zeros((5, 2), dtype=np.float32))

        assert TrajectoryDirectory(out.path).embedding_layers() == ["dino_patch", "pi05"]
        assert not list((out.path / "policy_embeddings").glob("*tmp*"))

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path, monkeypatch):
        out = self._dir(tmp_path)

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(np, "savez_compressed", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            add_embedding_layer(out.path, "dino_patch", np.zeros((5, 2), dtype=np.float32))

        assert TrajectoryDirectory(out.path).embedding_layers() == ["pi05"]
        assert not list(out.path.glob(".*tmp*"))

    def test_existing_layer_is_kept_unless_overwrite(self, tmp_path):
        out = self._dir(tmp_path)
        first = np.ones((5, 2), dtype=np.float32)
        add_embedding_layer(out.path, "dino_patch", first)

        with pytest.raises(FileExistsError, match="overwrite"):
            add_embedding_layer(out.path, "dino_patch", np.zeros((5, 2), dtype=np.float32))
        assert np.array_equal(TrajectoryDirectory(out.path).embeddings("dino_patch"), first)

        add_embedding_layer(
            out.path, "dino_patch", np.zeros((5, 2), dtype=np.float32), overwrite=True
        )
        assert not TrajectoryDirectory(out.path).embeddings("dino_patch").any()

    def test_overwriting_a_layer_changes_the_data_fingerprint(self, tmp_path):
        """Otherwise the pipeline serves features computed from the old contents.

        The stage cache keys features on `data_fingerprint`, which hashes
        `episodes/metadata.yaml` rather than the arrays -- sound for every other
        writer of this layout, because they all rewrite that file. Replacing a
        layer in place is the one operation that changes what a directory means
        without touching it.
        """
        from behavior_graphs.pipeline.cache import data_fingerprint

        out = self._dir(tmp_path)
        add_embedding_layer(out.path, "dino_patch", np.ones((5, 2), dtype=np.float32))
        before = data_fingerprint(out.path)

        add_embedding_layer(
            out.path, "dino_patch", np.zeros((5, 2), dtype=np.float32), overwrite=True
        )

        assert data_fingerprint(out.path) != before

    def test_adding_a_new_layer_leaves_the_fingerprint_alone(self, tmp_path):
        """A new name is a new features config, so a new cache key already --
        invalidating here would discard every other layer's cached work."""
        from behavior_graphs.pipeline.cache import data_fingerprint

        out = self._dir(tmp_path)
        before = data_fingerprint(out.path)

        add_embedding_layer(out.path, "dino_patch", np.ones((5, 2), dtype=np.float32))

        assert data_fingerprint(out.path) == before

    def test_misaligned_rows_are_refused(self, tmp_path):
        """Caught here rather than on the next read, which is a different session."""
        out = self._dir(tmp_path)
        with pytest.raises(ValueError, match="rows"):
            add_embedding_layer(out.path, "dino_patch", np.zeros((4, 2), dtype=np.float32))
