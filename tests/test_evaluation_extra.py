"""Tests for the label-stats and outcome-MI evaluators (and metric porting)."""

import json

import numpy as np
import pytest

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.stages.evaluation import (
    ClusterQualityEvaluator,
    GraphStatsEvaluator,
    LabelStatsEvaluator,
    ValueEvaluator,
)
from behavior_graphs.stages.graph import TransitionGraphBuilder

WIDTH = 5  # frames per window; windows are non-overlapping (stride = width)


def make_result(episode_labels: list[list[int]], successes: list[bool | None]) -> PipelineResult:
    """Synthetic result: one non-overlapping window per label, WIDTH frames each."""
    windows: list[WindowInfo] = []
    episodes: list[EpisodeInfo] = []
    labels: list[int] = []
    start = 0
    rng = np.random.default_rng(0)
    for ep_idx, (labs, success) in enumerate(zip(episode_labels, successes)):
        n_frames = WIDTH * len(labs)
        episodes.append(
            EpisodeInfo(
                index=ep_idx,
                num_samples=n_frames,
                sample_start_idx=start,
                sample_end_idx=start + n_frames,
                success=success,
            )
        )
        start += n_frames
        for k, lab in enumerate(labs):
            windows.append(WindowInfo(episode_idx=ep_idx, start=k * WIDTH, end=(k + 1) * WIDTH))
            labels.append(lab)
    # Cluster-separated 2-d features so geometric metrics are well defined.
    arr = np.asarray(labels, dtype=np.int64)
    features = arr[:, None] * 10.0 + rng.normal(0.0, 0.1, size=(len(labels), 2))
    ws = WindowSet(features=features.astype(np.float32), windows=windows, episodes=episodes)
    return PipelineResult(windows=ws, reduced=features, labels=arr)


@pytest.fixture()
def known_result():
    """4 episodes with hand-counted swaps / distinct clusters / segments.

    ep0: [0,0,1,1] -> 1 swap, 2 distinct, segments (2,2)
    ep1: [0,2,2,2] -> 1 swap, 2 distinct, segments (1,3)
    ep2: [0,1,0,1] -> 3 swaps, 2 distinct, segments (1,1,1,1)
    ep3: [2,2,2,2] -> 0 swaps, 1 distinct, segments (4,)
    """
    return make_result(
        [[0, 0, 1, 1], [0, 2, 2, 2], [0, 1, 0, 1], [2, 2, 2, 2]],
        [True, False, True, False],
    )


class TestLabelStats:
    def test_known_values(self, known_result):
        m = LabelStatsEvaluator().evaluate(known_result)
        assert m["n_episodes"] == 4
        assert m["n_windows"] == 16
        assert m["n_clusters"] == 3
        # 5 swaps over 4 episodes * 4 windows * 5 frames = 80 frames.
        assert m["swap_rate_per_frame"] == pytest.approx(5 / 80)
        # 5 swaps over 4 episodes * 3 adjacent pairs = 12 pairs.
        assert m["swap_rate_per_window"] == pytest.approx(5 / 12)
        assert m["distinct_per_episode"] == pytest.approx((2 + 2 + 2 + 1) / 4)
        assert m["distinct_per_episode_median"] == pytest.approx(2.0)
        # Segment lengths: (2,2), (1,3), (1,1,1,1), (4,) -> mean 16/9.
        assert m["mean_segment_length"] == pytest.approx(16 / 9)
        # Coverage: (2/3 + 2/3 + 2/3 + 1/3) / 4.
        assert m["cluster_coverage_per_episode"] == pytest.approx(7 / 12)

    def test_noise_windows_excluded(self):
        result = make_result([[0, -1, 0, 0], [1, 1, -1, 1]], [True, False])
        m = LabelStatsEvaluator().evaluate(result)
        # Noise windows drop out before swap counting: both episodes are
        # constant-label after exclusion.
        assert m["n_windows"] == 6
        assert m["n_clusters"] == 2
        assert m["swap_rate_per_frame"] == 0.0
        assert m["distinct_per_episode"] == pytest.approx(1.0)
        assert m["mean_segment_length"] == pytest.approx(3.0)

    def test_all_noise_is_graceful_and_serializable(self):
        result = make_result([[-1, -1]], [True])
        m = LabelStatsEvaluator().evaluate(result)
        assert m["n_episodes"] == 0
        assert m["swap_rate_per_frame"] is None
        json.dumps(m)

    def test_single_cluster_has_zero_swap_rate(self):
        result = make_result([[0, 0, 0], [0, 0, 0]], [True, False])
        m = LabelStatsEvaluator().evaluate(result)
        assert m["swap_rate_per_frame"] == 0.0
        assert m["distinct_per_episode"] == pytest.approx(1.0)
        assert m["cluster_coverage_per_episode"] == pytest.approx(1.0)


class TestAllEvaluatorsJSONSerializable:
    def test_json_dumps_succeeds(self, known_result):
        known_result.graph = TransitionGraphBuilder().build(
            known_result.labels, known_result.windows
        )
        evaluators = [
            GraphStatsEvaluator(),
            ClusterQualityEvaluator(),
            ValueEvaluator(),
            LabelStatsEvaluator(),
        ]
        for evaluator in evaluators:
            metrics = evaluator.evaluate(known_result)
            assert isinstance(metrics, dict)
            payload = json.dumps(metrics)
            assert "NaN" not in payload, evaluator.name
