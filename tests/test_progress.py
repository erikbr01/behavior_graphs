"""Tests for behavior_graphs.progress_estimation — SARM-style labels and demo scoring on synthetic data."""

import numpy as np
import pytest

from behavior_graphs.core.graph import (
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
from behavior_graphs.progress_estimation import (
    ProgressEvaluator,
    absorption_progress,
    dwell_fraction_stats,
    episode_label_sequences,
    ess,
    expand_to_steps,
    rabc_weights,
    seg_trim_scores,
    stage_balance_scores,
    stage_progress,
    stage_schedule,
    success_absorption,
    zero_frac,
)

# Toy task: stages 0 -> 1 -> 2, successes traverse all three, failures stall
# in 1 or bounce back to 0.
SUCC_A = np.array([0, 0, 1, 1, 1, 2, 2])
SUCC_B = np.array([0, 1, 1, 2, 2, 2])
FAIL_STALL = np.array([0, 0, 1, 1, 1, 1, 1, 1])
FAIL_REGRESS = np.array([0, 1, 1, 0, 0, 0])

SEQS = [SUCC_A, SUCC_B, FAIL_STALL, FAIL_REGRESS]
FLAGS = [True, True, False, False]


def make_window_set(label_seqs, successes):
    """One window per timestep (width 1) so labels map 1:1 onto windows."""
    lengths = [len(s) for s in label_seqs]
    episodes = episodes_from_lengths(lengths, successes)
    windows = [
        WindowInfo(episode_idx=ep.index, start=i, end=i + 1)
        for ep, seq in zip(episodes, label_seqs)
        for i in range(len(seq))
    ]
    features = np.zeros((sum(lengths), 2), dtype=np.float32)
    return WindowSet(features=features, windows=windows, episodes=episodes)


class TestStageSchedule:
    def test_order_follows_time(self):
        sched = stage_schedule(SEQS, FLAGS)
        assert sched.order == (0, 1, 2)

    def test_spans_and_bases_cumulative(self):
        sched = stage_schedule(SEQS, FLAGS)
        assert sum(sched.span.values()) == pytest.approx(1.0)
        assert sched.base[0] == 0.0
        assert sched.base[1] == pytest.approx(sched.span[0])
        assert sched.base[2] == pytest.approx(sched.span[0] + sched.span[1])

    def test_requires_a_success(self):
        with pytest.raises(ValueError):
            stage_schedule([FAIL_STALL], [False])


class TestStageProgress:
    def test_monotone_on_success(self):
        sched = stage_schedule(SEQS, FLAGS)
        p = stage_progress(SUCC_A, sched)
        assert np.all(np.diff(p) > 0)
        assert p[0] >= 0 and p[-1] <= 1

    def test_regression_drops_label(self):
        sched = stage_schedule(SEQS, FLAGS)
        p = stage_progress(FAIL_REGRESS, sched)
        # back in stage 0 after stage 1 -> label falls back below stage-1 base
        assert p[3] < p[1]


class TestSuccessAbsorption:
    def two_path_graph(self):
        # START -> 0, then 0 branches: via 1 to SUCCESS (p=0.6) or via 2 to FAILURE.
        probs = {
            START_NODE_ID: {0: 1.0},
            0: {1: 0.6, 2: 0.4},
            1: {SUCCESS_NODE_ID: 1.0},
            2: {FAILURE_NODE_ID: 1.0},
        }
        return BehaviorGraph(
            nodes={}, transition_counts={}, transition_probs=probs, num_episodes=10
        )

    def test_bounds_and_separation(self):
        absorb = success_absorption(self.two_path_graph())
        assert all(0.0 <= p <= 1.0 for p in absorb.values())
        assert absorb[1] == pytest.approx(1.0)  # success path
        assert absorb[2] == pytest.approx(0.0)  # failure path
        assert absorb[2] < absorb[0] < absorb[1]
        assert absorb[0] == pytest.approx(0.6)

    def test_special_ids_excluded(self):
        absorb = success_absorption(self.two_path_graph())
        assert set(absorb) == {0, 1, 2}

    def test_absorption_progress_lookup(self):
        absorb = success_absorption(self.two_path_graph())
        p = absorption_progress(np.array([0, 1, 7]), absorb)
        assert p == pytest.approx([0.6, 1.0, 0.0])  # unknown cluster reads 0


class TestRABCWeights:
    def test_clip_range_and_ordering(self):
        seqs = [np.linspace(0, 1, 10), np.full(10, 0.3)]
        w_prog, w_stall = rabc_weights(seqs, delta=2)
        assert np.all((w_prog >= 0.0) & (w_prog <= 1.0))
        assert np.all((w_stall >= 0.0) & (w_stall <= 1.0))
        assert w_prog[:-2].mean() > w_stall.mean()

    def test_negative_delta_is_zeroed_not_soft_clipped(self):
        """SARM Eq.9's r<0 -> 0 branch. Without it a backtrack lands mid-ramp (~0.5),
        which flattens the weight distribution and blunts the scheme."""
        # progress falls then rises: the early chunk deltas are negative.
        (w,) = rabc_weights([np.array([1.0, 0.6, 0.2, 0.4, 0.9])], delta=1)
        assert w[0] == 0.0 and w[1] == 0.0, "backtracking chunks must get zero weight"
        assert w[2] > 0.0, "a recovering chunk still earns weight"

    def test_kappa_defaults_to_the_mean_delta(self):
        seqs = [np.linspace(0, 1, 12)]
        explicit = rabc_weights(
            seqs, delta=2, kappa=float(np.diff(np.linspace(0, 1, 12), 2).mean())
        )
        auto = rabc_weights(seqs, delta=2)
        assert auto[0].shape == explicit[0].shape

    def test_kappa_override_forces_full_weight(self):
        (w,) = rabc_weights([np.linspace(0, 1, 10)], delta=2, kappa=0.1)
        # interior deltas are 2/9 > kappa -> exactly 1
        assert np.all(w[:-2] == 1.0)


class TestSummaryStats:
    def test_ess(self):
        assert ess(np.ones(10)) == pytest.approx(1.0)
        one_hot = np.zeros(10)
        one_hot[3] = 1.0
        assert ess(one_hot) == pytest.approx(0.1)
        assert ess(np.zeros(4)) == 0.0

    def test_zero_frac(self):
        assert zero_frac(np.array([0.0, 0.01, 0.5, 1.0])) == pytest.approx(0.5)
        assert zero_frac(np.ones(5)) == 0.0


class TestSegTrim:
    def test_fractional_caps_trim_long_dwell(self):
        # Successful rollouts dwell 2/6 of their windows in each stage.
        rollout_seqs = [np.array([0, 0, 1, 1, 2, 2])] * 2
        mu, sd = dwell_fraction_stats(rollout_seqs, [True, True])
        assert mu[0] == pytest.approx(1 / 3)
        # Demo spends 8/12 of its windows in stage 0: cap = ceil((1/3 + sd) * 12) = 5.
        demo = np.array([0] * 8 + [1] * 2 + [2] * 2)
        (wt,) = seg_trim_scores([demo], mu, sd)
        assert np.all(wt[:5] == 1.0)
        assert np.all(wt[5:8] == 0.0)
        assert np.all(wt[8:] == 1.0)

    def test_third_revisit_dropped(self):
        rollout_seqs = [np.array([0, 1, 0, 1])]
        mu, sd = dwell_fraction_stats(rollout_seqs, [True])
        demo = np.array([0, 1, 0, 1, 0])  # third visit to cluster 0 at the end
        (wt,) = seg_trim_scores([demo], mu, sd)
        assert wt[-1] == 0.0
        assert np.all(wt[:-1] == 1.0)

    def test_unknown_cluster_uncapped(self):
        demo = np.array([7] * 6)
        (wt,) = seg_trim_scores([demo], {}, {})
        assert np.all(wt == 1.0)


class TestNoiseWindows:
    """A `-1` window is unlabelled, and must not be transparent to the things
    that count visits, dwells or class mass."""

    def test_noise_breaks_the_run_it_interrupts(self):
        # Skipping the -1 merged these into one dwell of 2, so a cluster the
        # demo visited twice looked like one long stay.
        mu, sd = dwell_fraction_stats([np.array([3, -1, 3])], [True])
        assert mu[3] == pytest.approx(1 / 3)  # two dwells of 1/3, not one of 2/3

    def test_a_revisit_across_noise_still_counts_as_a_revisit(self):
        # `seg_trim_scores` drops the third visit to a cluster; with the noise
        # window transparent, these three visits read as one and it never fired.
        demo = np.array([0, -1, 0, -1, 0])
        (wt,) = seg_trim_scores([demo], {}, {}, max_revisits=2)
        assert wt[-1] == 0.0
        assert np.all(wt[:-1] == 1.0)

    def test_noise_is_not_a_class_to_balance(self):
        # Counting -1 in the mass histogram made one pseudo-class whose size
        # then boosted or suppressed every unlabelled window.
        (w,) = stage_balance_scores([np.array([0] * 8 + [1] * 2 + [-1] * 40)])
        labelled = w[:10]
        assert labelled[-1] / labelled[0] == pytest.approx(2.0)  # sqrt(8/2), noise ignored
        assert len(set(np.round(w[10:], 12))) == 1  # every noise window alike


class TestStageBalance:
    def test_sqrt_tempering(self):
        (w,) = stage_balance_scores([np.array([0] * 8 + [1] * 2)])
        # mass ratio 4 -> weight ratio sqrt(4) = 2 (cap not engaged here)
        assert w[-1] / w[0] == pytest.approx(2.0)
        assert np.concatenate([w]).mean() == pytest.approx(1.0)

    def test_median_cap_limits_rare_clusters(self):
        (w,) = stage_balance_scores([np.array([0] * 100 + [1])])
        # raw ratio would be sqrt(100) = 10; the 5x-median cap clips it to 5
        assert w[-1] / w[0] == pytest.approx(5.0)


class TestExpandToSteps:
    @pytest.mark.parametrize(
        "width,stride,length,n_windows",
        [(2, 3, 10, 3), (5, 2, 30, 13), (1, 1, 4, 4), (5, 2, 3, 1)],
    )
    def test_length_correctness(self, width, stride, length, n_windows):
        out = expand_to_steps(np.arange(n_windows, dtype=float), width, stride, length)
        assert len(out) == length

    def test_constant_scores_stay_constant(self):
        out = expand_to_steps(np.full(3, 0.7), 2, 3, 10)
        assert out == pytest.approx(np.full(10, 0.7))

    def test_window_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            expand_to_steps(np.ones(4), 2, 3, 10)


class TestEpisodeLabelSequences:
    def test_split_and_temporal_order(self):
        ws = make_window_set([np.array([0, 1, 2]), np.array([5, 6])], [True, False])
        # scramble window order to prove sequences are rebuilt temporally
        perm = [3, 0, 4, 1, 2]
        ws_scrambled = WindowSet(
            features=ws.features[perm],
            windows=[ws.windows[i] for i in perm],
            episodes=ws.episodes,
        )
        labels = np.array([5, 0, 6, 1, 2])
        seqs = episode_label_sequences(labels, ws_scrambled)
        assert len(seqs) == 2
        np.testing.assert_array_equal(seqs[0], [0, 1, 2])
        np.testing.assert_array_equal(seqs[1], [5, 6])


class TestProgressEvaluator:
    def test_metrics_on_synthetic_pipeline(self):
        label_seqs = [
            np.array([0, 0, 1, 1, 2]),
            np.array([0, 1, 1, 2, 2]),
            np.array([0, 0, 1, 1, 1]),
        ]
        successes = [True, True, False]
        ws = make_window_set(label_seqs, successes)
        labels = np.concatenate(label_seqs)
        graph = BehaviorGraph.from_window_labels(labels, ws)
        result = PipelineResult(windows=ws, labels=labels, graph=graph)

        metrics = ProgressEvaluator().evaluate(result)
        assert metrics["n_episodes"] == 3
        assert metrics["n_success"] == 2
        assert metrics["n_stages"] == 3
        assert metrics["stage_order"] == [0, 1, 2]
        assert 0.0 <= metrics["absorption_min"] <= metrics["absorption_max"] <= 1.0
        assert metrics["absorption_max"] == pytest.approx(1.0)  # stage 2 always succeeds
        assert metrics["stage_progress_monotonicity"] > 0.9
        assert metrics["tail_progress_auroc"] == pytest.approx(1.0)
