# Policy comparison

Is policy A actually better than policy B, or did it just get a luckier set of
rollouts? The **Policy Comparison** page of the demo app answers that for any N
of the rollout datasets the app can see, following TRI's
[Statistical thinking for robot policy evaluation][article].

[article]: https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d

## The moving parts

| Piece | Where | What it does |
|---|---|---|
| Statistics | `stats/policy_comparison.py` | Beta posteriors, pairwise exact tests, FWER correction, compact letter display. Pure — no Streamlit, no plotting, no filesystem. |
| Figure | `viz/plotly/policy_comparison.py` | Posterior violins ordered best-first, credible interval + median marks, CLD letters as direct labels — or the same result as bars ± 1 SE. |
| Data | `viz/streamlit/rollout_outcomes.py` | Finds rollout datasets and tallies each one's successes/failures. |
| Page | `apps/demo/_pages/policy_comparison.py` | Selector, controls, summary table, pairwise p-value matrix. |

## What counts as a policy

One **rollout dataset** — one policy rolled out on one task, with per-episode
outcomes in `episodes/metadata.yaml`. Pipeline runs are not the unit: a sweep
holds hundreds of runs over the same rollouts, and their success rates are all
identical because the graph pipeline never touches the outcomes.

Datasets are found by scanning `$DATA_PATH/rollouts`, falling back to the
`rollout_dir` of every indexed run when that directory does not exist. Either
way the result is cut to the same `BEHAVIOR_GRAPHS_DISPLAY_DATA` allowlist as the
run selectors — matched on the directory name, since the fallback route names a
dataset `parent/name` when two runs rolled out same-named directories under
different parents.

Episodes with a null `success` are **excluded**, not counted as failures; the
page reports how many it dropped. A dataset with no scored episodes at all is
still drawn — its posterior is just the prior — so an unlabelled dataset looks
obviously unlabelled rather than silently zero.

## The method

1. **Each policy is a binomial sample**: `s` successes out of `n` rollouts.
2. **Every pair gets an exact test.** Barnard's by default. Neither margin of
   the 2x2 table is fixed by the experiment — we chose the rollout counts, not
   the total number of successes — so the *unconditional* tests (Barnard's,
   Boschloo's) are both correct and more powerful than Fisher's, which is
   offered only for comparison against the conditional convention.
   Each pair is TRI's **mirrored test**: two one-sided tests with the null and
   the alternative swapped, each at α, rather than one two-sided test. Nothing
   is owed between the two directions, because one test's null is the other's
   alternative — a two-sided p against α would be roughly twice as strict as
   the recipe and would lose separations the article's own code reports.
3. **Family-wise error is controlled** across the `N(N-1)/2` pairs. Bonferroni
   by default (TRI's `global_alpha / num_comparisons`); Holm–Bonferroni
   controls the same error rate with more power.
4. **A compact letter display** summarises what survived, via Piepho's
   insert-and-absorb algorithm: **policies sharing a letter are not
   statistically separated**. `cld_is_valid` asserts the property in both
   directions and the tests check it on random significance matrices. The
   letters are named by success rate, so `a` is the best-performing policy —
   including in a figure that pins a baseline to the left of every panel,
   where the leftmost violin is deliberately not the highest one.
5. **Uncertainty is a Beta posterior**, drawn as a violin. Beta(1, 1) by
   default, which is what TRI's notebook draws with; Jeffreys' Beta(½, ½) is
   available and says "unlabelled" far more loudly for a dataset with no scored
   rollouts (95% interval [0.2%, 99.8%] against Laplace's [2.5%, 97.5%]).

### Parity with TRI's implementation

The reference code is [`sequentialized_barnard_tests`][sbt] (`batch.py`'s
`MirroredBarnardExactTest`, driven by `tools/plotting.compare_success_and_get_cld`)
plus the CLD in [`statistical_comparison_helpers`][sch]. `TestParityWithTRI`
pins our decisions against a transcription of it, pair for pair, over three
tests and four α values.

Two details are easy to get wrong and are therefore tested directly:

- **`pooled=(n0 == n1)`.** The pooled-variance Wald statistic orders Barnard's
  sample space, and TRI pools only when the arms have equal n. SciPy pools
  always, which gives a different p-value on unequal n.
- **`p <= alpha`, not `p < alpha`.** These are discrete tests; the boundary is
  reachable.

One deliberate shortcut: the mirrored test formally needs *both* tails, since
it reports "undecided" when both fire. We compute only the tail the observed
rates point at. Measured exhaustively over every 2x2 table with n ≤ 12, the
larger tail is never below 0.470, so that branch is unreachable for any
family-wise α this page can produce — and the second call would double the
cost of the grid search. `MIRROR_SINGLE_TAIL_ALPHA` records the bound and a
test re-measures it.

What we add on top: Holm as a correction, Boschloo's and Fisher's as
alternative tests, and credible intervals on the violins.

[sbt]: https://github.com/TRI-ML/sequentialized_barnard_tests
[sch]: https://github.com/TRI-ML/statistical_comparison_helpers

### Why violins instead of error bars

Two 95% intervals that overlap can still belong to policies that are
significantly different — the difference has its own, narrower, sampling
distribution. Error bars invite exactly the comparison they cannot support, so
the page shows the posterior over each rate and puts the pairwise decisions in
the letters instead.

The **Chart** control offers `Bars ± standard error` anyway, because that is
the convention most audiences read. It draws the same letters over bars of the
observed rate with `sqrt(p(1-p)/n)` whiskers. Note what the bars do at the extremes:
a policy at 0/n or n/n gets no whisker at all, since the SE of an observed
proportion is exactly zero there — the sample with the least information drawn
as if it had the most.

### The study's choices under the violins

Every policy in these comparisons but the baseline is a policy retrained on data
a study participant asked for, so the paper figure has a second version that
says how many asked for it: a **row of bars per display** (graph, videos) under
the rate axis, at the violins' own x positions. **Figures** →
*Policy comparison — violins over the study's choices*, exported by the same
panel as the plain violins and downloaded under a `_violins_choices` name, so a
paper can carry both.

```
Success rate   ▇ baseline   ▇ plus milk left   ▇ plus milk middle
Graph              —          50% (2)            25% (1)
Videos             —          60% (3)             0%
```

What the marks mean, and why each is not the obvious alternative:

- **A share with its count.** `50% (2)` — the denominators here are single
  figures (four graph answers for set table, five for videos), and a share
  without its count invites a reader to assume a study an order of magnitude
  larger. Say the denominator in the LaTeX caption too.
- **A dash, not a zero, for the baseline.** Nobody was offered "collect
  nothing", and a policy nobody *could* pick is not a policy nobody picked. An
  option that was offered and went unchosen gets a stub on the axis and `0%`.
- **The heavier outline** marks where that display's answers most went — ink
  rather than a second hue, so it survives greyscale. Ties are all outlined.
- **Randomized stages only**, by default: the third stage let the participant
  choose their display, so a share pooled over it describes the people who
  reached for the graph rather than what the graph led people to. The page has
  the toggle, and the restriction is the one every display comparison in the
  paper is made under (`figure_export.study_figure(..., randomized_only=True)`).

The shares come from `regret.choice_share_rows` — the same option-to-policy
mapping the regret scoring uses, so the bars and the regret panels cannot come
to disagree about which option named which retrained policy — and are turned
into bars by `figure_export.choice_series`, which is also where the two
displays' colours and names live.

### What it does not do

These are **batch** tests: they assume the rollout counts were fixed before
collection. Re-running an evaluation, watching the p-value and stopping when it
dips below α inflates the false-positive rate however good the test is. Adaptive
stopping needs a sequential test built for it — TRI's [STEP][step] (a separate
package, non-commercial licence), not anything in this repo.

[step]: https://tri-ml.github.io/step/

The page also does not know whether two datasets are comparable. Two policies on
*different* tasks will separate on task difficulty; that is a real difference in
success rate, and reading it as "policy A is better" is the user's error, not
the test's.

## Using the statistics without the app

```python
from behavior_graphs.stats.policy_comparison import PolicyOutcomes, compare_policies

result = compare_policies(
    [
        PolicyOutcomes("diffusion policy", 45, 50),
        PolicyOutcomes("pi0", 30, 50),
        PolicyOutcomes("baseline", 12, 50),
    ],
    alpha=0.05,
    test="barnard",        # or "boschloo", "fisher"
    correction="holm",     # or "bonferroni" (TRI's), "none"
)
for i in result.order:                       # best success rate first
    print(result.letters[i], result.arms[i].name, result.intervals[i])
```
