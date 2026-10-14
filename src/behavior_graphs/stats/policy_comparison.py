"""A/B (and A/B/C/...) testing over policy rollout outcomes.

Implements the batch-testing recipe from TRI's *Statistical thinking for robot
policy evaluation* (https://medium.com/toyotaresearch/statistical-thinking-for-
robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d):
each policy is ``s`` successes out of ``n`` rollouts, every pair gets an exact
test, family-wise error is controlled across the ``N(N-1)/2`` pairs, and what
survives is summarised as a compact letter display. ``docs/policy_comparison.md``
walks through the method and the reasoning; three points are load-bearing here:

  - **Unconditional tests by default.** Neither margin of the 2x2 table is
    fixed by the experiment — we chose how many rollouts to run per policy, not
    how many successes to observe in total — so Barnard's and Boschloo's are
    both correct and more powerful than Fisher's.
  - **Mirrored one-sided tests, not two-sided ones.** Each pair is TRI's
    mirrored test: two one-sided tests with the null and the alternative
    swapped, *each* at α. No correction is owed between the two directions
    because one test's null is the other's alternative, so a two-sided p
    against α would be about twice as strict as the recipe calls for and would
    quietly lose separations the article's own code reports.
  - **Batch, not sequential.** These tests assume the rollout counts were fixed
    before collection. Adaptive stopping needs TRI's STEP (a separate package
    under a non-commercial licence, tri-ml.github.io/step), which is not what
    the finished rollout sets this repo indexes call for.

The reference implementation is TRI's ``sequentialized_barnard_tests`` — the
``MirroredBarnardExactTest`` class and ``compare_success_and_get_cld`` — with
the compact letter display from ``statistical_comparison_helpers``. This module
reproduces its decisions for the batch case, and adds Holm as a strictly more
powerful alternative to the Bonferroni correction it applies.

Everything here is pure: no Streamlit, no plotting, no filesystem.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import barnard_exact, boschloo_exact, fisher_exact
from scipy.stats import beta as beta_dist

#: Reference prior for a binomial proportion. Proper for every (s, n)
#: including 0/n and n/n, unlike Haldane's Beta(0, 0), and it keeps a dataset
#: with no scored rollouts looking unscored: its 95% interval is [0.002, 0.998]
#: where Laplace's is [0.025, 0.975].
JEFFREYS_PRIOR = (0.5, 0.5)
#: Laplace's "flat" prior, and **the default**: it is what TRI's notebook draws
#: its violins with. Pulls the posterior toward 1/2 harder than Jeffreys at
#: small n.
UNIFORM_PRIOR = (1.0, 1.0)
#: Prior used wherever the caller does not name one.
DEFAULT_PRIOR = UNIFORM_PRIOR

#: Pairwise tests, in decreasing order of power for this design.
TESTS: dict[str, str] = {
    "barnard": "Barnard's exact (unconditional)",
    "boschloo": "Boschloo's exact (unconditional)",
    "fisher": "Fisher's exact (conditional)",
}

#: Family-wise error corrections across the pairwise tests.
CORRECTIONS: dict[str, str] = {
    "bonferroni": "Bonferroni",
    "holm": "Holm–Bonferroni",
    "none": "None (per-comparison α)",
}


@dataclass(frozen=True)
class PolicyOutcomes:
    """One policy's rollout tally: *successes* out of *trials*.

    Episodes with an unknown outcome must be excluded by the caller before
    constructing this — a rollout that was never scored is not a failure.
    """

    name: str
    successes: int
    trials: int

    def __post_init__(self) -> None:
        if self.trials < 0 or self.successes < 0:
            raise ValueError(f"{self.name}: counts must be non-negative")  # noqa: TRY003
        if self.successes > self.trials:
            raise ValueError(  # noqa: TRY003
                f"{self.name}: {self.successes} successes out of {self.trials} trials"
            )

    @property
    def failures(self) -> int:
        return self.trials - self.successes

    @property
    def rate(self) -> float:
        """Observed success rate; NaN for a policy with no scored rollouts."""
        return self.successes / self.trials if self.trials else float("nan")

    @property
    def standard_error(self) -> float:
        """``sqrt(p(1-p)/n)`` of the observed rate; NaN with no scored rollouts.

        Collapses to exactly 0 at 0/n and n/n, claiming perfect certainty from
        the samples that have the least of it. That is one of the reasons the
        page draws posterior violins by default and offers this only as an
        alternative view.
        """
        if not self.trials:
            return float("nan")
        return math.sqrt(self.rate * (1.0 - self.rate) / self.trials)


# ── Bayesian posterior over the success rate ─────────────────────────────────


def posterior_params(arm: PolicyOutcomes, prior: tuple[float, float] = DEFAULT_PRIOR) -> tuple:
    """``(a, b)`` of the Beta posterior for *arm* under a Beta *prior*."""
    return (prior[0] + arm.successes, prior[1] + arm.failures)


def posterior_sample_grid(
    arm: PolicyOutcomes,
    prior: tuple[float, float] = DEFAULT_PRIOR,
    resolution: int = 1000,
) -> np.ndarray:
    """Evenly-spaced posterior quantiles — a violin's worth of "samples".

    Quantiles rather than draws from an RNG: the result is an exact, seedless,
    reproducible stand-in for a sample of that size, so two renders of the same
    page cannot disagree and a test can assert on it.

    The plot wants samples rather than the analytic density because a Jeffreys
    posterior with 0 or N successes has an integrable pole at the boundary. The
    density there is unplottable — normalising the shape by its maximum
    collapses every other violin to a hairline — whereas the same posterior as
    quantiles is perfectly ordinary, and the violin's KDE renders it as the
    lopsided bump against the boundary that it is.
    """
    a, b = posterior_params(arm, prior)
    # Open interval: the closed one would put a quantile at exactly 0 and 1.
    probs = (np.arange(resolution) + 0.5) / resolution
    return beta_dist.ppf(probs, a, b)


def credible_interval(
    arm: PolicyOutcomes,
    level: float = 0.95,
    prior: tuple[float, float] = DEFAULT_PRIOR,
) -> tuple[float, float]:
    """Equal-tailed posterior credible interval at *level*."""
    a, b = posterior_params(arm, prior)
    tail = (1.0 - level) / 2.0
    return (float(beta_dist.ppf(tail, a, b)), float(beta_dist.ppf(1.0 - tail, a, b)))


def posterior_median(arm: PolicyOutcomes, prior: tuple[float, float] = DEFAULT_PRIOR) -> float:
    a, b = posterior_params(arm, prior)
    return float(beta_dist.median(a, b))


# ── Pairwise testing ─────────────────────────────────────────────────────────


#: Concurrency for the pair loop. Capped rather than scaled to the machine:
#: the page tests at most 66 pairs, and a Streamlit server is already serving
#: other sessions on the same box.
_MAX_TEST_THREADS = 8

#: Memoised pair results. The unconditional tests are a grid search — ~0.9 s
#: for a single pair of 1000-rollout policies — and a p-value is a pure
#: function of four counts and the test, so a repeat is always waste. Sized for
#: many sessions' worth of pairs; each entry is one float.
_PVALUE_CACHE_SIZE = 4096


#: The largest α at which one tail is enough to reproduce the mirrored test.
#: TRI's mirrored test reports "undecided" when *both* one-sided tests fire, so
#: strictly it needs both tails. Measured exhaustively over every 2x2 table with
#: n ≤ 12 per arm, the smaller of the two tails is at worst 0.470 (Boschloo;
#: 0.658 Fisher, 1.0 Barnard), so below this bound the branch cannot be reached
#: and the second call — which doubles the cost of the expensive grid search —
#: buys nothing. ``test_both_tails_cannot_fire_below_the_documented_alpha``
#: re-measures it.
MIRROR_SINGLE_TAIL_ALPHA = 0.4


@lru_cache(maxsize=_PVALUE_CACHE_SIZE)
def _pvalue_for_counts(sa: int, fa: int, sb: int, fb: int, test: str) -> float:
    """One-sided p-value for "these two policies have the same success rate".

    The tail is the one the observed rates point at, which is the tail TRI's
    mirrored test decides on: asking whether the *worse*-scoring policy is
    secretly better returns a p-value near 1 and is never the question.

    SciPy's unconditional tests take each *column* of the 2x2 table as one
    binomial experiment, so the table is outcomes-by-policy, not the other way
    round — transposing it silently computes a different (wrong) test.

    Keyed on counts rather than on :class:`PolicyOutcomes` so that two datasets
    with the same tally share an entry, and renaming one does not miss.
    """
    na, nb = sa + fa, sb + fb
    if na == 0 or nb == 0:
        return float("nan")
    table = [[sa, sb], [fa, fb]]
    # Column 0 is policy A, so "greater" is the tail to take when A scored
    # higher. A tie takes either — both tails are then ≥ ½ and no α separates.
    alternative = "less" if sa / na < sb / nb else "greater"
    if test == "barnard":
        # ``pooled`` picks the Wald statistic that orders the sample space, and
        # TRI pools only when the two arms have the same n — the case where a
        # common-variance statistic is the natural one. SciPy would pool always.
        return float(barnard_exact(table, alternative=alternative, pooled=na == nb).pvalue)
    if test == "boschloo":
        return float(boschloo_exact(table, alternative=alternative).pvalue)
    if test == "fisher":
        return float(fisher_exact(table, alternative=alternative)[1])
    raise ValueError(f"unknown test {test!r}; expected one of {sorted(TESTS)}")  # noqa: TRY003


def _pvalue(a: PolicyOutcomes, b: PolicyOutcomes, test: str) -> float:
    """:func:`_pvalue_for_counts` on a canonically ordered pair.

    Swapping the two policies swaps the columns *and* the tail the rates point
    at, so the value is unchanged; ordering the key means A-then-B and B-then-A
    share a cache entry.
    """
    left, right = sorted(((a.successes, a.failures), (b.successes, b.failures)))
    return _pvalue_for_counts(*left, *right, test)


def pairwise_pvalues(arms: Sequence[PolicyOutcomes], test: str = "barnard") -> np.ndarray:
    """Symmetric ``(N, N)`` matrix of raw one-sided p-values; NaN on the diagonal.

    Pairs are evaluated concurrently: they are independent, and SciPy drops the
    GIL inside the exact tests, so threads buy roughly 2-3x on the selections
    that are slow enough to notice. Results are order-independent, so this
    changes only the wall clock.
    """
    n = len(arms)
    out = np.full((n, n), np.nan)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return out
    with ThreadPoolExecutor(max_workers=min(len(pairs), _MAX_TEST_THREADS)) as pool:
        for (i, j), p in zip(
            pairs, pool.map(lambda ij: _pvalue(arms[ij[0]], arms[ij[1]], test), pairs)
        ):
            out[i, j] = out[j, i] = p
    return out


def adjust_pvalues(pvalues: np.ndarray, correction: str = "bonferroni") -> np.ndarray:
    """Family-wise adjusted copy of a pairwise p-value matrix.

    Adjusting the p-values rather than shrinking α keeps one number per pair
    that can be read against the α the user actually chose, which is what the
    table on the page shows. The family is the upper triangle — each unordered
    pair is one test, not two.
    """
    n = pvalues.shape[0]
    iu = np.triu_indices(n, k=1)
    raw = pvalues[iu]
    tested = ~np.isnan(raw)
    m = int(tested.sum())
    adj = raw.copy()
    if m and correction == "bonferroni":
        adj[tested] = np.minimum(1.0, raw[tested] * m)
    elif m and correction == "holm":
        vals = raw[tested]
        order = np.argsort(vals)
        # Step-down: the k-th smallest is scaled by (m - k), then made
        # monotone so a pair can never be more significant than a smaller one.
        stepped = np.maximum.accumulate(np.minimum(1.0, vals[order] * (m - np.arange(m))))
        holm = np.empty(m)
        holm[order] = stepped
        adj[tested] = holm
    elif correction not in CORRECTIONS:
        raise ValueError(  # noqa: TRY003
            f"unknown correction {correction!r}; expected one of {sorted(CORRECTIONS)}"
        )
    out = np.full_like(pvalues, np.nan)
    out[iu] = adj
    out.T[iu] = adj
    return out


# ── Compact letter display ───────────────────────────────────────────────────


def _letters() -> list[str]:
    """``a..z`` then ``aa..zz`` — enough for any comparison a page can render."""
    single = list(string.ascii_lowercase)
    return single + [x + y for x in single for y in single]


def _absorb(groups: list[set[int]]) -> list[set[int]]:
    """Drop empty groups and any group contained in another.

    Absorption is what keeps the display compact, and it is safe: a group is
    only removed when a *superset* survives, so every member keeps a letter and
    every pair that shared a letter still shares one.
    """
    kept: list[set[int]] = []
    for g in sorted((g for g in groups if g), key=len, reverse=True):
        if not any(g <= h for h in kept):
            kept.append(g)
    return kept


def compact_letter_display(
    significant: np.ndarray, order: Sequence[int] | None = None
) -> list[str]:
    """Letters per policy from a boolean "differs significantly" matrix.

    Piepho's insert-and-absorb algorithm (J. Comput. Graph. Stat. 13(2), 2004):
    start with every policy in one group, and each time a significant pair is
    found sharing a group, split that group into two — one without each member
    — then absorb. What comes out satisfies both directions by construction:
    a significant pair shares no letter, and a non-significant pair shares at
    least one.

    *order* is the display order (e.g. by descending success rate); letters are
    handed out so that the first column reads ``a``. Returns one string per
    policy in the *input* order, each a concatenation like ``"ab"``.
    """
    n = int(significant.shape[0])
    if n == 0:
        return []
    groups: list[set[int]] = [set(range(n))]
    for i in range(n):
        for j in range(i + 1, n):
            if not significant[i, j]:
                continue
            split: list[set[int]] = []
            for g in groups:
                if i in g and j in g:
                    split.append(g - {i})
                    split.append(g - {j})
                else:
                    split.append(g)
            groups = _absorb(split)

    rank = {idx: pos for pos, idx in enumerate(order if order is not None else range(n))}
    groups.sort(key=lambda g: (min(rank[i] for i in g), sorted(rank[i] for i in g)))
    alphabet = _letters()
    out = [""] * n
    for letter, group in zip(alphabet, groups):
        for i in group:
            out[i] += letter
    return out


def cld_is_valid(letters: Sequence[str], significant: np.ndarray) -> bool:
    """Whether *letters* faithfully encode *significant* — used by the tests.

    Sharing a letter must mean "not separated" in both directions; a display
    that only gets one direction right is worse than no display at all.
    """
    n = len(letters)
    for i in range(n):
        for j in range(i + 1, n):
            shares = bool(set(letters[i]) & set(letters[j]))
            if shares == bool(significant[i, j]):
                return False
    return True


def unique_labels(names: Sequence[str]) -> list[str]:
    """*names* made unique by numbering repeats, order preserved.

    The suffix is checked against the labels already handed out, not just
    counted: a name that literally ends in ``" (2)"`` is a legal
    ``display_name``, and colliding with it would put two policies on one plot
    axis category — the failure this function exists to prevent.
    """
    counts: dict[str, int] = {}
    taken: set[str] = set()
    out = []
    for name in names:
        label = name
        while label in taken:
            counts[name] = counts.get(name, 1) + 1
            label = f"{name} ({counts[name]})"
        taken.add(label)
        out.append(label)
    return out


# ── Display order ────────────────────────────────────────────────────────────


def display_order(
    arms: Sequence[PolicyOutcomes], *, pin_first: Sequence[str] | str | None = None
) -> list[int]:
    """Indices into *arms*, best success rate first.

    *pin_first* names arms that lead the order regardless of their rate, in the
    order named — a reference arm the rest are read against, which is what a
    figure of one baseline plus its variants wants: the baseline in the same
    place in every panel, and everything after it ranked. Arms not in *arms*
    are ignored, so a group that has no such arm simply sorts by rate.

    NaN rates (no scored rollouts) sort last rather than poisoning the order.
    Ties break on input position, so the order is total and reproducible.
    """
    pinned = [pin_first] if isinstance(pin_first, str) else list(pin_first or ())
    lead: list[int] = []
    for name in pinned:
        lead += [i for i in range(len(arms)) if arms[i].name == name and i not in lead]
    rest = sorted(
        (i for i in range(len(arms)) if i not in set(lead)),
        key=lambda i: (np.isnan(arms[i].rate), -np.nan_to_num(arms[i].rate, nan=0.0), i),
    )
    return lead + rest


# ── Everything a page needs, in one call ─────────────────────────────────────


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing N policies. Rows/columns follow ``arms``."""

    arms: list[PolicyOutcomes]
    #: Family-wise adjusted one-sided p-values. The raw matrix is not kept:
    #: with a correction chosen it is never the number to read off, and
    #: ``adjusted`` is what both the table and ``significant`` are built from.
    adjusted: np.ndarray
    significant: np.ndarray
    letters: list[str]
    #: ``arms`` names made unique, aligned with ``arms``. Two datasets can carry
    #: the same ``display_name``, and every downstream consumer needs one key
    #: per policy — a plot axis category, a table row, a table column.
    labels: list[str]
    #: Indices into ``arms`` — the plotting order. Best success rate first,
    #: except for any arm ``pin_first`` held out of the sort.
    order: list[int]
    alpha: float
    test: str
    correction: str
    prior: tuple[float, float]
    #: Posterior summaries, aligned with ``arms``.
    medians: list[float]
    intervals: list[tuple[float, float]]
    credible_level: float

    @property
    def n_comparisons(self) -> int:
        """Pairs tested — the family the correction was applied over."""
        n = len(self.arms)
        return n * (n - 1) // 2


def compare_policies(
    arms: Sequence[PolicyOutcomes],
    *,
    alpha: float = 0.05,
    test: str = "barnard",
    correction: str = "bonferroni",
    prior: tuple[float, float] = DEFAULT_PRIOR,
    credible_level: float = 0.95,
    pin_first: Sequence[str] | str | None = None,
) -> ComparisonResult:
    """Run the full batch comparison: pairwise tests, correction, CLD, posteriors.

    A pair is separated when its adjusted one-sided p-value is ``<= alpha``,
    which is TRI's mirrored-test decision: see :func:`_pvalue_for_counts` and
    :data:`MIRROR_SINGLE_TAIL_ALPHA` for why one tail suffices to reproduce it.

    A pair involving a policy with no scored rollouts yields a NaN p-value and
    is treated as *not* separated — absence of evidence, so it should not split
    a letter group.

    *pin_first* names arms to lead the display order ahead of the ranking —
    see :func:`display_order`. It moves no number and it renames no letter:
    the letters are handed out by success rate, so ``a`` is the best-performing
    policy in every figure, including one that leads each panel with its
    baseline. Pinning a mid-ranked baseline to the left would otherwise hand it
    ``a`` and leave the letters running up and down the row, which reads as a
    ranking that disagrees with the violins above it.

    Cheap to call repeatedly with the same *arms*: the exact tests are the only
    expensive step and they memoise per pair, so re-running with a different α,
    correction or prior costs microseconds.
    """
    arms = list(arms)
    adjusted = adjust_pvalues(pairwise_pvalues(arms, test), correction)
    significant = np.zeros_like(adjusted, dtype=bool)
    # ``<=``, not ``<``: TRI's test decides at ``pvalue <= alpha``, and on the
    # discrete tables these tests produce an exact tie is reachable, not a
    # measure-zero curiosity.
    np.less_equal(adjusted, alpha, out=significant, where=~np.isnan(adjusted))
    order = display_order(arms, pin_first=pin_first)
    # Two orders, on purpose: *order* is where the arms sit, and the ranking
    # below is what the letters are named for. They are the same list whenever
    # nothing is pinned.
    ranked = order if pin_first is None else display_order(arms)
    return ComparisonResult(
        arms=arms,
        adjusted=adjusted,
        significant=significant,
        letters=compact_letter_display(significant, ranked),
        labels=unique_labels([a.name for a in arms]),
        order=order,
        alpha=alpha,
        test=test,
        correction=correction,
        prior=prior,
        medians=[posterior_median(a, prior) for a in arms],
        intervals=[credible_interval(a, credible_level, prior) for a in arms],
        credible_level=credible_level,
    )
