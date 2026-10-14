"""Terminal-leaf sub-clustering as a trajectory-tree overlay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from behavior_graphs.core.graph import FAILURE_NODE_ID, BehaviorGraph
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.subclusters import SubClusterSet
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.stages.subclustering.terminal import TerminalSubClustering


def _result() -> PipelineResult:
    ep_labels = {
        0: [1, 4],
        1: [1, 4],
        2: [3, 4],
        3: [3, 4],
        4: [1, 2],
    }
    successes = {0: False, 1: False, 2: False, 3: False, 4: True}
    rng = np.random.default_rng(0)
    episodes = [
        EpisodeInfo(
            index=ep,
            num_samples=len(seq),
            sample_start_idx=0,
            sample_end_idx=len(seq),
            success=successes[ep],
        )
        for ep, seq in ep_labels.items()
    ]
    windows: list[WindowInfo] = []
    labels: list[int] = []
    for ep, seq in ep_labels.items():
        for t, label in enumerate(seq):
            windows.append(WindowInfo(episode_idx=ep, start=t, end=t + 1))
            labels.append(label)
    label_array = np.asarray(labels, dtype=np.int64)
    window_set = WindowSet(
        features=rng.normal(size=(len(windows), 4)).astype(np.float32),
        windows=windows,
        episodes=episodes,
    )
    result = PipelineResult()
    result.windows = window_set
    result.labels = label_array
    result.graph = BehaviorGraph.from_window_labels(label_array, window_set)
    return result


class TestTerminalSubClustering:
    def test_split_is_a_pure_overlay(self):
        result = _result()
        graph_before = result.graph.to_dict()
        split = TerminalSubClustering(
            k=2,
            min_episodes=2,
            reduction=None,
        ).split(result)

        assert result.graph.to_dict() == graph_before
        assert len(split.groups) == 2
        assert {group.path for group in split.groups} == {
            (1, 4, FAILURE_NODE_ID),
            (3, 4, FAILURE_NODE_ID),
        }

    def test_overlay_round_trips_without_graph_node_identity(self):
        split = TerminalSubClustering(
            k=2,
            min_episodes=2,
            reduction=None,
        ).split(_result())
        payload = json.loads(json.dumps(split.to_dict()))
        restored = SubClusterSet.from_dict(payload)

        assert restored == split
        assert all("node_id" not in group for group in payload["groups"])

    def test_window_detail_is_optional(self):
        split = TerminalSubClustering(
            k=2,
            min_episodes=2,
            reduction=None,
            keep_window_detail=False,
        ).split(_result())
        assert all(not group.window_subclusters for group in split.groups)

    def test_max_leaves_caps_largest_first(self):
        split = TerminalSubClustering(
            k=2,
            min_episodes=2,
            max_leaves=1,
            reduction=None,
        ).split(_result())
        assert len(split.groups) == 1
        assert split.groups[0].path == (1, 4, FAILURE_NODE_ID)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"k": 1}, "k must be >= 2"),
            ({"n_components": 0}, "n_components must be >= 1"),
            ({"min_episodes": 0}, "min_episodes must be >= 1"),
            ({"max_leaves": 0}, "max_leaves must be null or >= 1"),
        ],
    )
    def test_invalid_parameters_fail_at_construction(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TerminalSubClustering(**kwargs)


class TestPipelineOverlay:
    @pytest.fixture(scope="class")
    def rollouts(self, tmp_path_factory):
        from behavior_graphs.envs.trajectory_dir import write_rollout_dir

        rng = np.random.default_rng(3)
        dim = 24
        regimes = rng.normal(0, 1, size=(3, dim)) * 6.0
        lengths = [int(n) for n in rng.integers(20, 31, size=9)]
        successes = [i % 3 != 0 for i in range(9)]
        chunks = []
        for n in lengths:
            thirds = [n // 3, n // 3, n - 2 * (n // 3)]
            for regime, size in zip(regimes, thirds):
                chunks.append(regime + rng.normal(0, 0.5, size=(size, dim)))
        return write_rollout_dir(
            tmp_path_factory.mktemp("rollouts"),
            lengths,
            successes,
            {"pi05": np.concatenate(chunks).astype(np.float32)},
        )

    def _run(self, rollouts, run_dir, *overrides):
        from behavior_graphs.pipeline.run import compose_pipeline_config, run_pipeline

        cfg = compose_pipeline_config(
            [
                f"rollout_dir={rollouts.path}",
                "features.layer=pi05",
                "clustering.n_clusters=3",
                # PCA to two components, not the default UMAP: nothing here is
                # about the reduction (every assertion compares an overlay-on
                # run against an overlay-off one), and at width two the
                # pipeline reuses `reduced` for the 2-D coords instead of
                # fitting a second reducer. UMAP costs ~20s of numba JIT the
                # first time any test in a worker fits one.
                "pipeline/reduction@reduction=pca",
                "reduction.n_components=2",
                "cache_dir=null",
                *overrides,
            ]
        )
        return run_pipeline(cfg, rollouts, run_dir)

    def test_enabled_stage_persists_overlay_without_changing_graph(self, rollouts, tmp_path):
        from behavior_graphs.pipeline.artifacts import RunArtifacts

        off = self._run(rollouts, tmp_path / "off")
        on_dir = tmp_path / "on"
        on = self._run(
            rollouts,
            on_dir,
            "pipeline/subclustering@subclustering=terminal",
            "subclustering.k=2",
            "subclustering.min_episodes=2",
            "subclustering.reduction=null",
        )

        assert on.subclusters is not None
        assert (on_dir / "subclusters.json").exists()
        assert on.graph.to_dict() == off.graph.to_dict()
        assert on.graph.enumerate_paths() == off.graph.enumerate_paths()
        assert on.graph.compute_values() == pytest.approx(off.graph.compute_values())

        loaded = RunArtifacts(on_dir).load_result()
        assert loaded.graph.to_dict() == off.graph.to_dict()
        assert loaded.subclusters == on.subclusters

    def test_graph_metrics_are_invariant(self, rollouts, tmp_path):
        off = self._run(rollouts, tmp_path / "metrics-off")
        on = self._run(
            rollouts,
            tmp_path / "metrics-on",
            "pipeline/subclustering@subclustering=terminal",
            "subclustering.k=2",
            "subclustering.min_episodes=2",
            "subclustering.reduction=null",
        )

        for metric in ("graph_stats", "values", "cluster_quality"):
            assert on.metrics[metric] == off.metrics[metric]

    def test_overlay_is_served_from_stage_cache(self, rollouts, tmp_path):
        overrides = (
            "pipeline/subclustering@subclustering=terminal",
            "subclustering.k=2",
            "subclustering.min_episodes=2",
            "subclustering.reduction=null",
            f"cache_dir={tmp_path / 'cache'}",
        )
        first = self._run(rollouts, tmp_path / "cached-1", *overrides)
        second = self._run(rollouts, tmp_path / "cached-2", *overrides)

        assert first.metrics["cache_hits"]["subclusters"] is False
        assert second.metrics["cache_hits"]["subclusters"] is True
        assert second.subclusters == first.subclusters
        assert second.graph.to_dict() == first.graph.to_dict()
