"""Tests for the policy-comparison statistics, figure and page.

The statistics are the part worth pinning: a compact letter display that gets a
pair backwards, or a p-value computed off a transposed table, is wrong in a way
that looks entirely plausible on screen.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import barnard_exact, boschloo_exact, fisher_exact

from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.stats.policy_comparison import (
    DEFAULT_PRIOR,
    JEFFREYS_PRIOR,
    MIRROR_SINGLE_TAIL_ALPHA,
    UNIFORM_PRIOR,
    PolicyOutcomes,
    _pvalue_for_counts,
    adjust_pvalues,
    cld_is_valid,
    compact_letter_display,
    compare_policies,
    credible_interval,
    display_order,
    pairwise_pvalues,
    posterior_median,
    posterior_params,
    posterior_sample_grid,
    unique_labels,
)


def _sig(pairs: list[tuple[int, int]], n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        m[i, j] = m[j, i] = True
    return m


class TestPolicyOutcomes:
    def test_rate_and_failures(self):
        arm = PolicyOutcomes("p", 30, 50)
        assert arm.failures == 20
        assert arm.rate == pytest.approx(0.6)

    def test_no_trials_has_nan_rate(self):
        assert np.isnan(PolicyOutcomes("p", 0, 0).rate)

    def test_more_successes_than_trials_is_rejected(self):
        with pytest.raises(ValueError, match="51 successes out of 50"):
            PolicyOutcomes("p", 51, 50)


class TestPosterior:
    def test_params_add_counts_to_prior(self):
        assert posterior_params(PolicyOutcomes("p", 30, 50), JEFFREYS_PRIOR) == (30.5, 20.5)
        assert posterior_params(PolicyOutcomes("p", 30, 50), UNIFORM_PRIOR) == (31.0, 21.0)

    def test_the_default_prior_is_the_uniform_one_tri_draws_with(self):
        assert DEFAULT_PRIOR == UNIFORM_PRIOR
        assert posterior_params(PolicyOutcomes("p", 30, 50)) == (31.0, 21.0)

    def test_median_tracks_the_observed_rate(self):
        arm = PolicyOutcomes("p", 300, 500)
        assert posterior_median(arm) == pytest.approx(arm.rate, abs=0.005)

    def test_more_rollouts_narrow_the_interval(self):
        small = credible_interval(PolicyOutcomes("p", 6, 10))
        large = credible_interval(PolicyOutcomes("p", 600, 1000))
        assert (large[1] - large[0]) < (small[1] - small[0]) / 5

    def test_sample_grid_is_deterministic_and_in_range(self):
        arm = PolicyOutcomes("p", 30, 50)
        first = posterior_sample_grid(arm, resolution=64)
        assert np.array_equal(first, posterior_sample_grid(arm, resolution=64))
        assert first.shape == (64,)
        assert np.all((first > 0.0) & (first < 1.0))
        assert np.all(np.diff(first) > 0)

    @pytest.mark.parametrize("arm", [PolicyOutcomes("z", 0, 50), PolicyOutcomes("f", 50, 50)])
    def test_sample_grid_is_finite_at_the_boundaries(self, arm):
        """0/n and n/n put a pole on the Jeffreys posterior — quantiles survive it."""
        grid = posterior_sample_grid(arm, JEFFREYS_PRIOR, resolution=256)
        assert np.all(np.isfinite(grid))
        assert np.all((grid >= 0.0) & (grid <= 1.0))

    def test_unscored_dataset_gets_the_prior_back(self):
        """And Jeffreys says so far more loudly than Laplace does."""
        lo, hi = credible_interval(PolicyOutcomes("p", 0, 0), prior=JEFFREYS_PRIOR)
        assert lo < 0.01 and hi > 0.99
        flat_lo, flat_hi = credible_interval(PolicyOutcomes("p", 0, 0), prior=UNIFORM_PRIOR)
        assert flat_lo == pytest.approx(0.025, abs=1e-3)
        assert flat_hi == pytest.approx(0.975, abs=1e-3)


def _tri_tails(a: PolicyOutcomes, b: PolicyOutcomes, test: str) -> tuple[float, float]:
    """Both one-sided p-values, exactly as TRI's ``MirroredBarnardExactTest`` gets them.

    Transcribed from ``sequentialized_barnard_tests.batch``: the table is
    outcomes-by-policy, each direction is a separate one-sided call, and
    Barnard's pools only when the two arms have the same n.
    """
    table = [[a.successes, b.successes], [a.failures, b.failures]]
    if test == "barnard":
        pooled = a.trials == b.trials
        return (
            float(barnard_exact(table, alternative="less", pooled=pooled).pvalue),
            float(barnard_exact(table, alternative="greater", pooled=pooled).pvalue),
        )
    if test == "boschloo":
        return (
            float(boschloo_exact(table, alternative="less").pvalue),
            float(boschloo_exact(table, alternative="greater").pvalue),
        )
    return (
        float(fisher_exact(table, alternative="less")[1]),
        float(fisher_exact(table, alternative="greater")[1]),
    )


def _tri_separates(a: PolicyOutcomes, b: PolicyOutcomes, test: str, alpha: float) -> bool:
    """TRI's mirrored decision: separated iff exactly one direction fires."""
    fired = [p <= alpha for p in _tri_tails(a, b, test)]
    return sum(fired) == 1


class TestPairwiseTests:
    def test_matrix_is_symmetric_with_an_empty_diagonal(self):
        arms = [PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 20, 50), PolicyOutcomes("c", 5, 9)]
        p = pairwise_pvalues(arms, "fisher")
        assert np.allclose(p, p.T, equal_nan=True)
        assert np.all(np.isnan(np.diag(p)))

    def test_fisher_matches_the_one_sided_scipy_call_on_that_table(self):
        """One-sided in the direction the rates point — a is ahead, so "greater"."""
        a, b = PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 25, 50)
        expected = fisher_exact([[40, 25], [10, 25]], alternative="greater")[1]
        assert pairwise_pvalues([a, b], "fisher")[0, 1] == pytest.approx(expected)

    def test_the_tail_follows_the_data_not_the_argument_order(self):
        """Swapping the two policies must not change the pair's p-value."""
        ahead, behind = PolicyOutcomes("ahead", 40, 50), PolicyOutcomes("behind", 25, 47)
        for test in ("barnard", "boschloo", "fisher"):
            forward = pairwise_pvalues([ahead, behind], test)[0, 1]
            backward = pairwise_pvalues([behind, ahead], test)[0, 1]
            assert forward == pytest.approx(backward)
            # ...and it is the smaller tail, not the near-1 one.
            assert forward == pytest.approx(min(_tri_tails(ahead, behind, test)))

    def test_barnard_pools_only_when_the_arms_have_the_same_n(self):
        """TRI's rule; SciPy would pool always, giving a different p-value."""
        equal = [PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 25, 50)]
        unequal = [PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 25, 47)]
        for arms, pooled in ((equal, True), (unequal, False)):
            table = [
                [arms[0].successes, arms[1].successes],
                [arms[0].failures, arms[1].failures],
            ]
            expected = barnard_exact(table, alternative="greater", pooled=pooled).pvalue
            assert pairwise_pvalues(arms, "barnard")[0, 1] == pytest.approx(expected)
        # The rule has teeth: on unequal n the two settings disagree.
        table = [[40, 25], [10, 22]]
        assert barnard_exact(table, alternative="greater", pooled=True).pvalue != pytest.approx(
            barnard_exact(table, alternative="greater", pooled=False).pvalue
        )

    def test_unconditional_tests_are_more_powerful_than_fisher(self):
        """Barnard's and Boschloo's should not be *less* significant than Fisher's."""
        arms = [PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 25, 50)]
        fisher = pairwise_pvalues(arms, "fisher")[0, 1]
        assert pairwise_pvalues(arms, "barnard")[0, 1] < fisher
        assert pairwise_pvalues(arms, "boschloo")[0, 1] < fisher

    def test_identical_policies_are_not_separated(self):
        arms = [PolicyOutcomes("a", 30, 50), PolicyOutcomes("b", 30, 50)]
        # One-sided, so the ceiling is ~½ rather than 1 — the point is that no
        # α anyone would set comes near it.
        assert pairwise_pvalues(arms, "barnard")[0, 1] > 0.4
        assert not compare_policies(arms, alpha=0.1).significant.any()

    def test_a_policy_with_no_scored_rollouts_gives_nan(self):
        arms = [PolicyOutcomes("a", 30, 50), PolicyOutcomes("empty", 0, 0)]
        assert np.isnan(pairwise_pvalues(arms)[0, 1])

    def test_unknown_test_is_rejected(self):
        with pytest.raises(ValueError, match="unknown test"):
            pairwise_pvalues([PolicyOutcomes("a", 1, 2), PolicyOutcomes("b", 1, 2)], "ttest")


class TestParityWithTRI:
    """The decisions must match TRI's reference implementation, not merely resemble it."""

    #: The four set_table DIT policies, the comparison that motivated the port.
    REAL = [
        PolicyOutcomes("baseline", 32, 55),
        PolicyOutcomes("plusleft", 46, 60),
        PolicyOutcomes("plusmiddle", 28, 60),
        PolicyOutcomes("plusright", 26, 60),
    ]

    @pytest.mark.parametrize("test", ["barnard", "boschloo", "fisher"])
    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
    def test_separations_match_the_mirrored_test_at_global_alpha(self, test, alpha):
        """Bonferroni: our adjusted p ≤ α must agree pair-for-pair with raw ≤ α/m."""
        result = compare_policies(self.REAL, alpha=alpha, test=test, correction="bonferroni")
        individual = alpha / result.n_comparisons
        for i, j in itertools.combinations(range(len(self.REAL)), 2):
            expected = _tri_separates(self.REAL[i], self.REAL[j], test, individual)
            assert bool(result.significant[i, j]) is expected, (
                f"{self.REAL[i].name} vs {self.REAL[j].name} at α={alpha} ({test})"
            )

    def test_the_four_way_comparison_reproduces_tris_letters(self):
        """Checked against a run of TRI's own CLD helper on this data."""
        result = compare_policies(self.REAL, alpha=0.05, correction="bonferroni")
        letters = {result.arms[i].name: result.letters[i] for i in range(len(self.REAL))}
        assert letters == {
            "plusleft": "a",
            "baseline": "ab",
            "plusmiddle": "b",
            "plusright": "b",
        }

    def test_the_pair_alone_separates_where_the_family_of_six_does_not(self):
        """The behaviour that looks like a bug and is not: α is spent per family."""
        pair = [self.REAL[0], self.REAL[1]]
        assert compare_policies(pair, alpha=0.05).significant[0, 1]
        assert not compare_policies(self.REAL, alpha=0.05).significant[0, 1]

    def test_a_pair_exactly_at_alpha_is_separated(self):
        """TRI decides at ``p <= alpha``; a discrete test hits the boundary exactly."""
        arms = [PolicyOutcomes("a", 9, 10), PolicyOutcomes("b", 3, 10)]
        p = pairwise_pvalues(arms, "fisher")[0, 1]
        assert compare_policies(arms, alpha=p, test="fisher", correction="none").significant[0, 1]

    @pytest.mark.parametrize("test", ["barnard", "boschloo", "fisher"])
    def test_both_tails_cannot_fire_below_the_documented_alpha(self, test):
        """Why one tail reproduces the mirrored test: the other is never small too.

        Re-measures the bound behind :data:`MIRROR_SINGLE_TAIL_ALPHA` over a grid
        of small tables, where discreteness is at its worst.
        """
        smallest_far_tail = min(
            max(_tri_tails(PolicyOutcomes("a", sa, na), PolicyOutcomes("b", sb, nb), test))
            for na, nb in itertools.product((1, 2, 3, 6, 10), repeat=2)
            for sa in range(na + 1)
            for sb in range(nb + 1)
        )
        assert smallest_far_tail > MIRROR_SINGLE_TAIL_ALPHA


class TestCorrections:
    def _matrix(self, upper: list[float]) -> np.ndarray:
        n = 3
        m = np.full((n, n), np.nan)
        for (i, j), v in zip([(0, 1), (0, 2), (1, 2)], upper):
            m[i, j] = m[j, i] = v
        return m

    def test_bonferroni_multiplies_by_the_number_of_pairs(self):
        adj = adjust_pvalues(self._matrix([0.01, 0.02, 0.5]), "bonferroni")
        assert adj[0, 1] == pytest.approx(0.03)
        assert adj[1, 2] == pytest.approx(1.0)  # clipped, not 1.5

    def test_holm_steps_down_and_never_exceeds_bonferroni(self):
        raw = self._matrix([0.01, 0.02, 0.5])
        holm, bonf = adjust_pvalues(raw, "holm"), adjust_pvalues(raw, "bonferroni")
        assert holm[0, 1] == pytest.approx(0.03)  # smallest: x3
        assert holm[0, 2] == pytest.approx(0.04)  # next: x2
        upper = np.triu_indices(3, k=1)
        assert np.all(holm[upper] <= bonf[upper] + 1e-12)

    def test_holm_is_monotone_in_the_sorted_order(self):
        adj = adjust_pvalues(self._matrix([0.04, 0.001, 0.002]), "holm")
        assert adj[0, 2] <= adj[1, 2] <= adj[0, 1]

    def test_none_passes_the_p_values_through(self):
        raw = self._matrix([0.01, 0.02, 0.5])
        assert np.allclose(adjust_pvalues(raw, "none"), raw, equal_nan=True)

    def test_nan_pairs_are_left_out_of_the_family(self):
        raw = self._matrix([0.01, np.nan, 0.02])
        adj = adjust_pvalues(raw, "bonferroni")
        assert adj[0, 1] == pytest.approx(0.02)  # 2 real tests, not 3
        assert np.isnan(adj[0, 2])

    def test_unknown_correction_is_rejected(self):
        with pytest.raises(ValueError, match="unknown correction"):
            adjust_pvalues(self._matrix([0.01, 0.02, 0.5]), "fdr")


class TestCompactLetterDisplay:
    def test_nothing_separated_is_one_shared_letter(self):
        assert compact_letter_display(_sig([], 3)) == ["a", "a", "a"]

    def test_everything_separated_is_all_distinct_letters(self):
        letters = compact_letter_display(_sig([(0, 1), (0, 2), (1, 2)], 3))
        assert sorted(letters) == ["a", "b", "c"]

    def test_the_classic_overlapping_case(self):
        """A vs C separated, B overlapping both — B must carry both letters."""
        letters = compact_letter_display(_sig([(0, 2)], 3), order=[0, 1, 2])
        assert letters == ["a", "ab", "b"]

    def test_letters_follow_the_display_order(self):
        """The best policy (first in *order*) always reads 'a'."""
        letters = compact_letter_display(_sig([(0, 2)], 3), order=[2, 1, 0])
        assert letters[2] == "a"

    def test_empty_input(self):
        assert compact_letter_display(np.zeros((0, 0), dtype=bool)) == []

    # Eight seeds, not twenty-five: the property is checked by the body, and
    # the extra seventeen cases re-walk the same two branches.
    @pytest.mark.parametrize("seed", range(8))
    def test_random_matrices_produce_a_faithful_display(self, seed):
        """The property that matters: shares a letter iff not separated."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 8))
        upper = rng.random((n, n)) < 0.35
        sig = np.triu(upper, k=1)
        sig = sig | sig.T
        letters = compact_letter_display(sig, order=list(rng.permutation(n)))
        assert cld_is_valid(letters, sig)
        assert all(letters)  # every policy is in at least one group

    def test_validator_catches_a_broken_display(self):
        assert not cld_is_valid(["a", "a"], _sig([(0, 1)], 2))


class TestUniqueLabels:
    def test_repeats_are_numbered(self):
        assert unique_labels(["x", "y", "x", "x"]) == ["x", "y", "x (2)", "x (3)"]

    def test_a_name_that_looks_like_a_suffix_is_not_collided_with(self):
        # "x (2)" is a legal display_name; the generated suffix has to step
        # over it rather than duplicating it.
        assert unique_labels(["x", "x (2)", "x"]) == ["x", "x (2)", "x (3)"]

    @pytest.mark.parametrize(
        "names",
        [
            ["x", "x (2)", "x", "x (3)", "x"],
            ["a (2)", "a", "a", "a"],
            ["", "", ""],
        ],
    )
    def test_output_is_always_unique(self, names):
        out = unique_labels(names)
        assert len(set(out)) == len(out)


class TestDisplayOrder:
    def _arms(self):
        return [
            PolicyOutcomes("baseline", 30, 50),
            PolicyOutcomes("plus left", 45, 50),
            PolicyOutcomes("plus right", 20, 50),
        ]

    def test_best_first_without_a_pin(self):
        names = [a.name for a in self._arms()]
        order = display_order(self._arms())
        assert [names[i] for i in order] == ["plus left", "baseline", "plus right"]

    def test_a_pinned_arm_leads_and_the_rest_stay_ranked(self):
        arms = self._arms()
        order = display_order(arms, pin_first="baseline")
        assert [arms[i].name for i in order] == ["baseline", "plus left", "plus right"]

    def test_a_pin_that_matches_nothing_is_ignored(self):
        arms = self._arms()
        assert display_order(arms, pin_first="control") == display_order(arms)

    def test_several_pins_lead_in_the_order_named(self):
        arms = self._arms()
        order = display_order(arms, pin_first=["plus right", "baseline"])
        assert [arms[i].name for i in order] == ["plus right", "baseline", "plus left"]

    def test_a_repeated_pin_does_not_draw_an_arm_twice(self):
        arms = self._arms()
        order = display_order(arms, pin_first=["baseline", "baseline"])
        assert sorted(order) == list(range(len(arms)))

    def test_an_unscored_arm_sorts_last_even_pinned_behind(self):
        arms = [*self._arms(), PolicyOutcomes("unscored", 0, 0)]
        order = display_order(arms, pin_first="baseline")
        assert arms[order[0]].name == "baseline"
        assert arms[order[-1]].name == "unscored"

    def test_an_unscored_arm_can_still_be_pinned(self):
        arms = [*self._arms(), PolicyOutcomes("unscored", 0, 0)]
        order = display_order(arms, pin_first="unscored")
        assert arms[order[0]].name == "unscored"


class TestComparePolicies:
    def _arms(self):
        return [
            PolicyOutcomes("weak", 20, 50),
            PolicyOutcomes("strong", 45, 50),
            PolicyOutcomes("middling", 33, 50),
        ]

    def test_order_is_best_first(self):
        result = compare_policies(self._arms())
        assert [result.arms[i].name for i in result.order] == ["strong", "middling", "weak"]
        assert result.letters[result.order[0]].startswith("a")

    def test_a_pinned_arm_leads_the_row_without_taking_the_first_letter(self):
        """`a` is the best-performing policy, wherever it sits. A baseline is
        pinned left so it is in the same place in every panel, and it is not
        usually the best arm — handing it `a` would leave the letters running
        up and down the row while the violins above them descend."""
        arms = [*self._arms(), PolicyOutcomes("baseline", 30, 50)]
        result = compare_policies(arms, pin_first="baseline")

        assert [result.arms[i].name for i in result.order] == [
            "baseline",
            "strong",
            "middling",
            "weak",
        ]
        by_name = dict(zip([a.name for a in result.arms], result.letters, strict=True))
        assert by_name["strong"].startswith("a")
        assert not by_name["baseline"].startswith("a")

    def test_the_letters_ascend_with_success_rate(self):
        """Read left to right in rate order, the first letter of each group
        never goes backwards — which is what makes the letters legible as a
        ranking rather than as arbitrary names."""
        arms = [*self._arms(), PolicyOutcomes("baseline", 30, 50)]
        for pin in (None, "baseline"):
            result = compare_policies(arms, pin_first=pin)
            firsts = [
                result.letters[i][0]
                for i in sorted(range(len(result.arms)), key=lambda i: -result.arms[i].rate)
            ]
            assert firsts == sorted(firsts), f"pin_first={pin!r}: {firsts}"

    def test_pinning_regroups_no_letters(self):
        arms = [*self._arms(), PolicyOutcomes("baseline", 30, 50)]
        pinned = compare_policies(arms, pin_first="baseline")
        ranked = compare_policies(arms)
        # The same policies share a letter either way, and now they are called
        # the same letters too: naming follows the rate ranking, which pinning
        # does not touch.
        assert cld_is_valid(pinned.letters, pinned.significant)
        assert np.array_equal(pinned.significant, ranked.significant)
        assert sorted(pinned.order) == sorted(ranked.order)
        assert pinned.letters == ranked.letters

    def test_letters_agree_with_the_significance_matrix(self):
        result = compare_policies(self._arms())
        assert cld_is_valid(result.letters, result.significant)

    def test_a_clear_winner_is_separated(self):
        result = compare_policies(
            [PolicyOutcomes("weak", 10, 100), PolicyOutcomes("strong", 90, 100)]
        )
        assert result.significant[0, 1]
        assert not set(result.letters[0]) & set(result.letters[1])

    def test_the_same_rate_on_few_rollouts_is_not_separated(self):
        result = compare_policies([PolicyOutcomes("a", 6, 10), PolicyOutcomes("b", 4, 10)])
        assert not result.significant[0, 1]
        assert result.letters == ["a", "a"]

    def test_a_stricter_alpha_separates_fewer_pairs(self):
        arms = self._arms()
        loose = compare_policies(arms, alpha=0.1).significant.sum()
        strict = compare_policies(arms, alpha=0.001).significant.sum()
        assert strict <= loose

    def test_correction_only_ever_costs_separations(self):
        arms = self._arms()
        uncorrected = compare_policies(arms, correction="none").significant.sum()
        corrected = compare_policies(arms, correction="bonferroni").significant.sum()
        assert corrected <= uncorrected

    def test_nan_pairs_do_not_separate(self):
        result = compare_policies([PolicyOutcomes("real", 45, 50), PolicyOutcomes("empty", 0, 0)])
        assert not result.significant.any()
        assert result.letters == ["a", "a"]

    def test_repeat_calls_are_identical_and_memoised(self):
        """Re-running with a different α must not re-run the exact tests."""
        arms = self._arms()
        first = compare_policies(arms)
        before = _pvalue_for_counts.cache_info().misses
        second = compare_policies(arms, alpha=0.1, credible_level=0.5)
        assert _pvalue_for_counts.cache_info().misses == before
        assert np.array_equal(first.adjusted, second.adjusted, equal_nan=True)

    def test_result_summaries_are_aligned_with_arms(self):
        result = compare_policies(self._arms())
        assert len(result.medians) == len(result.intervals) == len(result.arms)
        assert result.labels == ["weak", "strong", "middling"]
        assert result.n_comparisons == 3
        for i, arm in enumerate(result.arms):
            lo, hi = result.intervals[i]
            assert lo < result.medians[i] < hi
            assert lo < arm.rate < hi


class TestFigure:
    def test_it_draws_one_violin_and_one_letter_per_policy(self):
        pytest.importorskip("plotly")
        from behavior_graphs.viz.plotly.policy_comparison import violin_cld_figure

        result = compare_policies(
            [
                PolicyOutcomes("weak", 20, 50),
                PolicyOutcomes("strong", 45, 50),
                PolicyOutcomes("never scored", 0, 0),
            ]
        )
        fig = violin_cld_figure(result, resolution=64)
        violins = [t for t in fig.data if t.type == "violin"]
        assert len(violins) == 3
        # Best first, and the unscored policy last rather than dropped.
        assert [v.x0 for v in violins] == ["strong", "weak", "never scored"]
        letters = {a.text for a in fig.layout.annotations if a.xref == "x"}
        assert letters == {f"<b>{s}</b>" for s in result.letters}

    def test_duplicate_display_names_stay_separate_categories(self):
        pytest.importorskip("plotly")
        from behavior_graphs.viz.plotly.policy_comparison import violin_cld_figure

        result = compare_policies([PolicyOutcomes("same", 5, 10), PolicyOutcomes("same", 9, 10)])
        fig = violin_cld_figure(result, resolution=64)
        assert len({t.x0 for t in fig.data if t.type == "violin"}) == 2


class TestStandardErrorBars:
    def test_standard_error_of_a_proportion(self):
        assert PolicyOutcomes("p", 30, 50).standard_error == pytest.approx(
            math.sqrt(0.6 * 0.4 / 50)
        )

    @pytest.mark.parametrize("arm", [PolicyOutcomes("z", 0, 40), PolicyOutcomes("f", 40, 40)])
    def test_it_collapses_to_zero_at_the_boundaries(self, arm):
        """Documented, and the reason the violins stay the default."""
        assert arm.standard_error == 0.0

    def test_no_scored_rollouts_has_no_standard_error(self):
        assert np.isnan(PolicyOutcomes("p", 0, 0).standard_error)

    def test_the_bar_figure_draws_a_rate_and_a_whisker_per_policy(self):
        pytest.importorskip("plotly")
        from behavior_graphs.viz.plotly.policy_comparison import bar_se_cld_figure

        arms = [
            PolicyOutcomes("weak", 20, 50),
            PolicyOutcomes("strong", 45, 50),
            PolicyOutcomes("never scored", 0, 0),
        ]
        result = compare_policies(arms)
        fig = bar_se_cld_figure(result)
        (bars,) = [t for t in fig.data if t.type == "bar"]
        assert list(bars.x) == ["strong", "weak", "never scored"]  # best first
        assert bars.y[0] == pytest.approx(0.9)
        assert bars.error_y.array[0] == pytest.approx(math.sqrt(0.9 * 0.1 / 50))
        # The unscored policy keeps its slot with nothing drawn in it.
        assert np.isnan(bars.y[2]) and np.isnan(bars.error_y.array[2])
        letters = {a.text for a in fig.layout.annotations if a.xref == "x"}
        assert letters == {f"<b>{s}</b>" for s in result.letters}

    def test_the_bars_start_at_zero_and_keep_a_tall_whisker_in_frame(self):
        pytest.importorskip("plotly")
        from behavior_graphs.viz.plotly.policy_comparison import bar_se_cld_figure

        result = compare_policies([PolicyOutcomes("a", 49, 50), PolicyOutcomes("b", 20, 50)])
        fig = bar_se_cld_figure(result, n_se=3.0)
        lo, hi = fig.layout.yaxis.range
        assert lo == 0.0
        assert hi >= 0.98 + 3 * PolicyOutcomes("a", 49, 50).standard_error

    def test_it_carries_the_same_letter_caption_as_the_violins(self):
        pytest.importorskip("plotly")
        from behavior_graphs.viz.plotly.policy_comparison import (
            bar_se_cld_figure,
            violin_cld_figure,
        )

        result = compare_policies([PolicyOutcomes("a", 40, 50), PolicyOutcomes("b", 20, 50)])
        assert fig_captions(bar_se_cld_figure(result)) == fig_captions(
            violin_cld_figure(result, resolution=64)
        )


def fig_captions(fig) -> list[str]:
    """The paper-space annotations — everything that is not a CLD letter."""
    return [a.text for a in fig.layout.annotations if a.xref == "paper"]


@pytest.fixture
def ro():
    """The discovery module with its three caches cleared.

    Every cache is process-global and keyed on paths under ``tmp_path``, so a
    test that forgets one reads the previous test's answer for a directory that
    no longer exists. Imported here rather than at module scope so the pure
    statistics above still run on an interpreter without Streamlit.
    """
    pytest.importorskip("streamlit")
    from behavior_graphs.viz.streamlit import rollout_outcomes as module

    for cache in (module._scan_rollouts_root, module._scan_indexed_runs, module._outcomes_cached):
        cache.clear()
    return module


@pytest.fixture
def fallback_index(monkeypatch, ro):
    """Force the runs-index fallback route with a given set of ``rollout_dir``s."""

    def install(*raw_dirs: str) -> None:
        monkeypatch.setattr(ro, "display_run_dirs", lambda: [])
        monkeypatch.setattr(
            ro, "runs_index", lambda _dirs: pd.DataFrame({"rollout_dir": list(raw_dirs)})
        )

    return install


def _dataset(root: Path, name: str, successes: list[bool | None]) -> Path:
    """A canonical rollout dataset, written by the same writer the app reads."""
    path = root / name
    write_rollout_dir(path, [10] * len(successes), successes, {})
    return path


class TestRolloutOutcomes:
    """The dataset-discovery/tally layer, against a canonical rollout dir."""

    def test_tally_splits_scored_from_unscored(self, tmp_path, ro):
        d = _dataset(tmp_path, "ds", [True, True, False, None])
        out = ro.load_outcomes("ds", d)
        assert (out.successes, out.failures, out.unscored, out.scored) == (2, 1, 1, 3)
        arm = out.as_arm()
        assert (arm.successes, arm.trials) == (2, 3)  # the null is not a failure

    def test_a_page_can_say_what_to_call_the_policy(self, tmp_path, ro):
        """The arm's name is what every reading of the comparison shows — the
        chart's axis, the tables, the exported figure — so a label given here
        renames the policy in all of them at once, and changes no count."""
        d = _dataset(tmp_path, "set_table_baseline_pretrained_dit", [True, True, False])
        out = ro.load_outcomes("set_table_baseline_pretrained_dit", d)

        named = out.as_arm("Ours")

        assert named.name == "Ours"
        assert (named.successes, named.trials) == (out.as_arm().successes, out.as_arm().trials)
        # And with nothing said, the dataset's own display name.
        assert out.as_arm().name == ro.dataset_display_name("set_table_baseline_pretrained_dit")

    def test_an_unreadable_dataset_tallies_to_zero_instead_of_raising(self, tmp_path, ro):
        out = ro.load_outcomes("missing", tmp_path / "nope")
        assert (out.successes, out.failures, out.unscored) == (0, 0, 0)

    @pytest.mark.parametrize(
        ("case", "text"),
        [
            # write_trajectory_dir emits source: before episodes:, so this is
            # what a page load lands on mid-write.
            ("truncated_before_episodes", "source:\n  kind: rollouts\n"),
            ("empty", ""),
            ("unparseable", "episodes: [\n"),
            ("episode_missing_length", "episodes:\n- index: 0\n  success: true\n"),
        ],
    )
    def test_a_partial_metadata_file_tallies_to_zero_instead_of_raising(
        self, tmp_path, ro, case, text
    ):
        """A dataset being written must not take the page down with it."""
        d = tmp_path / case / "episodes"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(text)
        out = ro.load_outcomes(case, d.parent)
        assert (out.successes, out.failures, out.unscored) == (0, 0, 0)

    def test_discovery_finds_datasets_under_the_rollouts_root(self, tmp_path, monkeypatch, ro):
        rollouts = tmp_path / "rollouts"
        _dataset(rollouts, "alpha", [True, False])
        _dataset(rollouts, "beta", [True])
        (rollouts / "not_a_dataset").mkdir()
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        assert sorted(ro.discover_datasets()) == ["alpha", "beta"]

    def test_discovery_falls_back_to_the_runs_index(
        self, tmp_path, monkeypatch, ro, fallback_index
    ):
        """No ``$DATA_PATH/rollouts``: find datasets via each run's rollout_dir."""
        _dataset(tmp_path / "elsewhere", "gamma", [True, False])
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        fallback_index("elsewhere/gamma")
        assert list(ro.discover_datasets()) == ["gamma"]

    def test_fallback_keeps_same_named_datasets_under_different_parents(
        self, tmp_path, monkeypatch, ro, fallback_index
    ):
        """Two runs rolled out ``lift`` in different sweeps — both stay selectable."""
        _dataset(tmp_path / "sweep_a", "lift", [True, False])
        _dataset(tmp_path / "sweep_b", "lift", [True])
        _dataset(tmp_path / "sweep_b", "square", [True])
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        fallback_index("sweep_a/lift", "sweep_b/lift", "sweep_b/square")
        found = ro.discover_datasets()
        # Colliding names are qualified; the unique one stays bare.
        assert sorted(found) == ["square", "sweep_a/lift", "sweep_b/lift"]
        assert len({str(p) for p in found.values()}) == 3
        assert ro.load_outcomes("sweep_a/lift", found["sweep_a/lift"]).scored == 2
        assert ro.load_outcomes("sweep_b/lift", found["sweep_b/lift"]).scored == 1

    def test_a_qualified_key_is_still_matched_against_the_allowlist(
        self, tmp_path, monkeypatch, ro, fallback_index
    ):
        """The allowlist is keyed by dataset name, so qualification must not hide one."""
        _dataset(tmp_path / "sweep_a", "lift", [True])
        _dataset(tmp_path / "sweep_b", "lift", [True])
        _dataset(tmp_path / "sweep_b", "secret", [True])
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        monkeypatch.setattr(ro, "display_dataset_names", lambda: {"lift"})
        fallback_index("sweep_a/lift", "sweep_b/lift", "sweep_b/secret")
        assert sorted(ro.discover_datasets()) == ["sweep_a/lift", "sweep_b/lift"]

    def test_discovery_respects_the_display_allowlist(self, tmp_path, monkeypatch, ro):
        rollouts = tmp_path / "rollouts"
        _dataset(rollouts, "shown", [True])
        _dataset(rollouts, "hidden", [True])
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        monkeypatch.setattr(ro, "display_dataset_names", lambda: {"shown"})
        assert list(ro.discover_datasets()) == ["shown"]


class TestPage:
    """Drive the real page with AppTest, against a temporary data root."""

    @staticmethod
    def _run(tmp_path, monkeypatch, ro):
        pytest.importorskip("plotly")
        from streamlit.testing.v1 import AppTest

        rollouts = tmp_path / "rollouts"
        _dataset(rollouts, "policy_a", [True] * 45 + [False] * 5)
        _dataset(rollouts, "policy_b", [True] * 10 + [False] * 40)
        _dataset(rollouts, "policy_c", [True] * 30 + [False] * 20)
        monkeypatch.setenv("DATA_PATH", str(tmp_path))

        page = Path(__file__).parent.parent / "apps" / "demo" / "_pages" / "policy_comparison.py"
        at = AppTest.from_file(str(page), default_timeout=120)
        at.run()
        return at

    def test_it_renders_the_chart_and_the_tables(self, tmp_path, monkeypatch, ro):
        at = self._run(tmp_path, monkeypatch, ro)
        assert not at.exception
        assert len(at.multiselect) == 1
        assert sorted(at.multiselect[0].value) == ["policy_a", "policy_b", "policy_c"]
        assert len(at.dataframe) == 2  # summary + pairwise p-values

    def test_deselecting_down_to_one_policy_asks_for_more(self, tmp_path, monkeypatch, ro):
        at = self._run(tmp_path, monkeypatch, ro)
        at.multiselect[0].set_value(["policy_a"]).run()
        assert not at.exception
        assert any("at least 2" in i.value for i in at.info)

    @staticmethod
    def _chart_spec(at) -> dict:
        """The rendered figure. ``.proto.spec`` rather than ``.value``: a
        plotly_chart's value is its selection state, which an un-clicked chart
        never puts in session state."""
        return json.loads(at.get("plotly_chart")[0].proto.spec)

    def test_the_credibility_level_is_fixed_rather_than_offered_as_a_control(
        self, tmp_path, monkeypatch, ro
    ):
        at = self._run(tmp_path, monkeypatch, ro)
        assert [w.label for w in at.select_slider] == ["Family-wise α"]
        # Still reported, at the fixed level.
        assert "95% CrI" in at.dataframe[0].value.columns

    def test_the_chart_can_be_switched_to_bars_with_standard_error(self, tmp_path, monkeypatch, ro):
        at = self._run(tmp_path, monkeypatch, ro)
        (chart,) = [r for r in at.radio if r.label == "Chart"]
        assert chart.value == "Posterior violins"
        assert "violin" in {t["type"] for t in self._chart_spec(at)["data"]}

        at = chart.set_value("Bars ± standard error").run()
        assert not at.exception
        spec = self._chart_spec(at)
        assert {t["type"] for t in spec["data"]} == {"bar"}
        (bars,) = spec["data"]
        assert bars["error_y"]["array"][0] == pytest.approx(math.sqrt(0.9 * 0.1 / 50))
        # The letters come from the tests, so they survive the switch.
        letters = {a["text"] for a in spec["layout"]["annotations"] if a.get("xref") == "x"}
        assert letters == {"<b>a</b>", "<b>b</b>", "<b>c</b>"}

    def test_the_clear_winner_is_separated_from_the_worst(self, tmp_path, monkeypatch, ro):
        at = self._run(tmp_path, monkeypatch, ro)
        at.multiselect[0].set_value(["policy_a", "policy_b"]).run()
        assert not at.exception
        # Read off the compact letter display in the summary table: 45/50 against
        # 10/50 is a separated pair, so the two carry different letters.
        assert sorted(at.dataframe[0].value["Group"]) == ["a", "b"]
