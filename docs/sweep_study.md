# Sweeps & model selection

How to sweep every hyperparameter / design decision of the graph-creation
pipeline, measure sensitivity against the proxy metrics, and select good
configurations — per task, per model, and per complexity budget K.

## The moving parts

| Piece | Where | What it does |
|---|---|---|
| Stage cache | `pipeline/cache.py` | Content-addressed reuse of features / windows / reduced / labels across runs. Keyed by data fingerprint + chained stage configs; a run that differs only in clustering recomputes only clustering. |
| Sweep runner | `pipeline/sweep.py` | Grouped grids (each stage option carries its own parameter grid — no invalid combos like `n_clusters` on HDBSCAN), cache-friendly job order, per-sweep reports. Library behind the entry point. |
| Entry point | `scripts/run_sweep.py` + `configs/sweep.yaml` | A hydra entry point: pick a grid with `grid=quick\|full`, point at data, run; prints best-per-K + sensitivity. |
| Grid configs | `configs/grid/` | The `grid` config group: `quick` (start here), `full` (windowing × reduction × K), and the paper's ablation grids `ablation` / `ablation_droid`. |
| Model selection lib | `model_selection/` | Runs index (config + metrics → one row per run), composite scoring, Pareto fronts, best-per-K, sensitivity tables. `CompositeSelection` / `Objective` are `_target_`-instantiable. |
| GUI | Graph Demo page | Model-selection panel (sortable score table, Pareto plot, best-per-K, sensitivity). Sweep Analysis page reads the same index. |

## Running a sweep

A hydra entry point — pick a grid (`grid=quick|full`, default `full`), point
at data with `datasets=` (an inline `{name: rollout_dir}` map) or `data=`
(a registered `configs/data/*.yaml` group), embedding layer auto-detected,
and run. It prints best-per-K + the most sensitive knobs and saves the
resolved grid to `<sweep_dir>/spec.yaml`:

```bash
# Quick scan on one dataset (~54 jobs):
python scripts/run_sweep.py grid=quick 'datasets={lift: /data/rollouts/lift_mh}'

# Full grid across several tasks/models in one sweep:
python scripts/run_sweep.py \
    'datasets={lift: /data/rollouts/lift_mh, square: /data/rollouts/square_pi05}'

# Inspect the job list without running anything:
python scripts/run_sweep.py 'datasets={lift: /data/lift}' dry_run=true
```

Everything is a plain hydra override: `sweep_dir=…`, `limit=N` (first N jobs
per dataset, a smoke test), `layer=…` (force an embedding layer instead of
auto-detect), `'extra_overrides=["cache_dir=/scratch/cache"]'` (applied to
every job — quote list elements containing `=`). Customise the grid by
editing `configs/grid/*.yaml` or overriding inline — a list-valued grid
directly, a `_target_` generator through its `_args_`:

```bash
python scripts/run_sweep.py 'datasets={lift: /data/lift}' \
    'grid.stages.windowing.sliding.width=[1,5]' \
    'grid.stages.clustering.kmeans.n_clusters._args_=[6,16,2]'   # irange(6,16,2)
```

Outputs under the sweep dir: one self-contained run dir per job, plus
`spec.yaml` (the resolved grid + selection), `sweep_index.csv` (hyperparams
× metrics × composite score), `best_per_k.csv`, and
`sensitivity_<metric>.csv`. Re-running is resumable — completed jobs are
skipped. Hydra `-m` multirun still works for flat per-stage sweeps and
benefits from the same cache.

### Typical workflow

1. `python scripts/run_sweep.py grid=quick 'datasets={task: <rollouts>}'` —
   look at the printed best-per-K and sensitivity ranking.
2. Open the GUI (below), sort the table, drag the weights, eyeball the
   Pareto frontier, and *look at the actual graphs* of the top candidates —
   the metrics are proxies.
3. Zoom into the interesting region — edit a `configs/grid/*.yaml` grid (or
   override inline) and rerun into the same `sweep_dir=`; completed jobs are
   skipped, the cache makes new ones cheap.
4. For "is this rule general?", pass several `datasets=` and compare the
   per-dataset sensitivity tables.

### Exploring in the GUI

`streamlit run apps/demo/Home.py` → **Graph Demo** (viz only — the app reads
existing runs; use `scripts/run_sweep.py` to produce new ones):

- **Model selection** (expander, below the graph): pick the dataset, set weights + the
  silhouette floor (flags only — nothing is discarded), sort the
  hyperparams × scores table by any column, read the best-per-K table,
  inspect the Pareto frontier (current run is starred; ringed points are
  below the floor), and open the sensitivity expander for which-knobs-matter.
- **Sweep Analysis** (separate page): per-metric winners, metric-vs-parameter
  scatter with best-envelope overlays, joint Pareto fronts, and the
  per-cluster Markov drill-down.

### Writing a grid config

A `grid` config (`configs/grid/*.yaml`) has `stages` (per-stage-option
parameter grids) and `selection` (a `_target_` `CompositeSelection`). It is
validated before anything runs — a misspelled option or parameter fails
immediately with the valid choices listed. Grid values come in three forms:

```yaml
stages:
  windowing:
    sliding:
      stride: 2                            # scalar — fixed value
      width: [1, 3, 5, 8]                  # explicit list
  clustering:
    kmeans:                                # _target_ generator (returns a list)
      n_clusters: {_target_: behavior_graphs.pipeline.grid.irange, _args_: [4, 20, 2]}
  reduction:
    umap:
      min_dist: {_target_: behavior_graphs.pipeline.grid.linspace, _args_: [0.0, 0.5, 3]}
    pca:
      n_components: {_target_: behavior_graphs.pipeline.grid.irange, _args_: [10, 100, 10]}

selection:
  _target_: behavior_graphs.model_selection.scoring.CompositeSelection
  weights: {silhouette: 1.0, boundary_gain: 1.0}
  constraints: {silhouette: {min: 0.05}}
```

`irange` is an inclusive integer range; `linspace`/`logspace` are
endpoint-inclusive float grids (see `behavior_graphs/pipeline/grid.py`).
Generators are instantiated with `hydra.utils.instantiate`, the same
mechanism every pipeline stage uses — there is no bespoke grid DSL.

### Sweeping tasks and models too

To test whether the same hyperparameters win everywhere, list several
rollout sources; the grid runs once per dataset:

```yaml
datasets:
  lift_mh:   /data/rollouts/lift_mh
  square_pi05: /data/rollouts/square_pi05
```

`sweep_index.csv` gains a `dataset` column, composite scores are
normalized *within* each dataset (an easy task can't dominate a hard one),
`best_per_k.csv` has one block per dataset, and the sensitivity tables
include `dataset` as an axis — its eta² versus the knobs' eta² is the
direct answer to "is this rule generally applicable or task-specific?".

### Adding a new dataset to an existing sweep

Sweep dirs accumulate: pointing a new dataset at an existing `sweep_dir=`
adds its jobs next to the old ones without touching them. The steps:

1. **Prepare the rollouts** in the canonical layout under the data root —
   `${DATA_PATH}/rollouts/<name>/` with `episodes/metadata.yaml`, one
   `policy_embeddings/<layer>.npz` per embedding layer (concatenated
   `(total_steps, D)` float32), and `media/ep####_{success,failure,unknown}.mp4`.
   Online collection with `--embed-layer` writes this directly; for stores
   in other layouts, assemble via
   `behavior_graphs.envs.trajectory_dir.write_rollout_dir` or the offline
   embedding-extraction scripts.

2. **Register the path** — either inline for a one-off, or permanently in
   the `configs/data/` group (e.g. `configs/data/droid.yaml`:
   `my_task: ${oc.env:DATA_PATH}/rollouts/my_task`) so `data=all`
   includes it from then on. The key is what the app displays; to relabel a
   dataset already swept under an outgrown name, see
   [droid_workflow.md](droid_workflow.md#relabelling-a-dataset-in-the-app).

3. **Rerun the sweep into the same `sweep_dir`** with the *same grid
   options the sweep was built with* — jobs are keyed by (overrides,
   dataset path), so a different grid would add a second grid's worth of
   jobs instead of reusing the existing one. For `sweep_all`
   (plan-bottleneck layer) that recipe is:

   ```bash
   python scripts/run_sweep.py grid=full \
       layers=[bottleneck_plan_t0] \
       sweep_dir=${DATA_PATH}/sweeps/sweep_all \
       'datasets={my_task: ${DATA_PATH}/rollouts/my_task}'
   ```

   When unsure which grid that was, check `<sweep_dir>/spec.yaml` — it
   records the grid the sweep was created with — and
   `<sweep_dir>/datasets.yaml` for the accumulated dataset map so far.
   Completed jobs (those with a `metrics.json`) are skipped, so only the new
   dataset's jobs actually run, and the reports (`sweep_index.csv`,
   `best_per_k.csv`, `sensitivity_*.csv`) are regenerated over *all*
   datasets at the end. Run with `$DATA_PATH` set: runs record their
   `rollout_dir` relative to it, which is what keeps them portable across
   machines and into the deployed web app.

4. **Check the results** — the printed best-per-K block for the new
   dataset, then the Sweep Analysis page. On the live-data deployment
   (see `deploy/README.md`) the sweep dir is mounted into the app
   containers, so the new runs appear on the next browser reload — no
   redeploy.

If a run of the sweep was interrupted, just rerun the same command — the
`metrics.json` skip makes it resumable; `failures.json` lists jobs that
errored (entries clear automatically once a job recovers).

## Proxy metrics

All directions are encoded once, in `model_selection/scoring.py::METRICS`:

- **silhouette** (↑), **calinski_harabasz** (↑), **davies_bouldin** (↓) —
  geometric cluster quality in the reduced space;
- **swap_rate** (↓) — label changes per frame: temporal coherence;
- diagnostics (no direction, never weighted by default): noise fraction,
  segment length, distinct clusters/episode, transition entropy,
  Markov violation, realized cluster count.

**Caveat:** the geometric metrics are computed in each run's own reduced
space, so comparisons across different `reduction` settings conflate
clustering quality with how easy the space makes the metric. Treat
cross-reduction silhouette gaps as a screening signal and lean on the
space-independent metrics (outcome MI, swap rate) plus visual inspection
for the final call.

How much this matters, measured: UMAP took **173 of 176** best-per-K slots in
`sweep_all`, and UMAP's objective is precisely to make tight, well-separated
neighbourhoods. A reduction is rewarded for reshaping the space, not for
representing behavior. Anything comparing reductions — a new reduction row, say —
has to be judged on the label-space metrics.

## What drives graph complexity

Measured over all 176 best-per-K runs in `sweep_all` (44 datasets, K ∈
5/10/15/20, KMeans on UMAP). Trajectory-tree node count against each candidate
cause, Spearman:

| cause | ρ |
|---|---|
| distinct paths the episodes take | **+0.93** |
| K | +0.63 |
| mean path length | +0.44 |
| episode count | +0.32 |
| A→B→A label flicker | ~0 |

The median run is 46 nodes from 17 distinct paths over 60 episodes, and **51% of
leaves are reached by exactly one episode** (in 88/176 runs, most leaves are).
A trajectory tree is therefore half a record of individual rollouts — inherent
to a prefix tree over paths, not noise to be filtered.

Consequences:

- **Aggregation is the remedy, not filtering.** Rare-branch folding, ALERGIA
  and a smaller K both attack path diversity. Folding
  single-episode leaves into a summary node is semantically right, not a display
  trick.
- **The Markov view has K + terminals nodes by construction** and is immune to
  all of this. The tree's unboundedness is what showing order costs.
- **Label smoothing is a refinement, not a remedy** — it addresses flicker,
  which is ~0 correlated with tree size (median 1% of path steps; 100/176 runs
  under 2%). It concentrates in particular feature layers: `pi05` median 20%
  against `bottleneck_plan_t0`'s 0%.

### Smoothing, if you enable it

Two things to know. First, it must measure distance in the space the clustering
was fitted in — `run.py` passes `in_clustering_space(windows, reduced)` to the
graph stage for exactly this. Scored on raw features instead, a low λ re-assigns
an eighth of the windows by a geometry the clusters were never defined in and
*grows* the median tree 56%; in the clustering space the same λ touches 0.4% and
shrinks it 11%.

Second, whether it helps is a property of the run, so there is no good default.
At λ=25 it shrinks the tree on 60% of runs and enlarges it on 26%, and the
oscillation rate barely predicts which (ρ = +0.10). λ is a flat penalty against
per-episode-normalized costs, so its strength moves with K and cluster spread —
1–2 is the useful band once the space is right. The graph demo prices it per run
beside the control. Outcomes are never at risk: 0/176 runs changed one.

## Model selection

The composite score is a linear combination over **population-normalized**
metrics: each metric is min-max scaled across the runs being compared and
flipped so 1 = best, then weighted and divided by Σ|w| (score stays in
[0, 1]). That makes weights comparable across metrics with incompatible
scales, at the cost of being relative to the run population (a score is
meaningless for a single run in isolation).

Silhouette is **necessary but not sufficient**: it screens out degenerate
clusterings but maximizing it tends to pick trivially separable, coarse
structure. It therefore enters twice — as a *floor* on the raw value and
optionally as one weighted term among several. The floor only **flags**
runs (`feasible=False` in the tables, ring markers in the plots): every
run is already computed, so nothing is ever discarded — the flag is a
reading aid, not a filter. Defaults: `silhouette ≥ 0.05`, weights
`silhouette 1.0, davies_bouldin 0.25, swap_rate 0.25`.
Tune both live in the GUI.

**K is the user's choice, not an optimization target.** More clusters ⇒
more granular behaviors ⇒ geometrically worse scores, so optimizing over K
always collapses to small K. The deliverable is `best_per_k`: for each K,
the run with the best composite score over the *remaining*
hyperparameters (with its silhouette flag shown alongside).

Per-run, the pipeline can also emit a raw (unnormalized) objective — an
`objective:` config (a `_target_` `Objective` with `weights` / `constraints`)
in the config — which the hydra entry point returns, so optuna-style
sweepers can optimize a linear combination directly.

## Sensitivity analysis

`model_selection/sensitivity.py` reports per hyperparameter, per metric:

- **eta²** — variance explained by the knob across all runs (one-way
  effect; confounded if the grid isn't full-factorial);
- **mean / max spread** — direction-corrected best-vs-worst gap the knob
  induces within *matched groups* (all other varying hyperparameters held
  fixed). This is the causal-ish number: the gain available from tuning
  just this knob;
- **best value & wins** — which setting wins how many matched groups. A
  knob whose winner flips between tasks/models has no generally applicable
  rule; a consistent winner across sweeps is a default worth adopting.

To look for generally applicable rules, sweep several tasks/models in one
spec (`datasets:`) and read `sensitivity_*.csv`: a knob whose `best_value`
holds across datasets and whose eta² dwarfs the `dataset` axis is a default
worth adopting; one whose winner flips per dataset must stay a user-facing
knob.

## Caching notes

- The cache key chains `data fingerprint → features → windowing →
  reduction → clustering` configs; `viz_reduction` hangs off the windows
  key. Graph construction and evaluation are cheap and always recomputed.
- The data fingerprint is the rollout dir's resolved path + size/mtime of
  `episodes/metadata.yaml` — re-collecting rollouts invalidates everything.
- Default location is `${DATA_PATH}/cache`, falling back to `./cache` when
  `DATA_PATH` is unset (`cache_dir=null` disables). Hanging it off the data
  volume means every worktree shares one cache instead of growing its own
  beside the repo. Entries are
  written atomically; concurrent sweep workers at worst duplicate work.
- Run dirs remain fully self-contained (the cache dedupes compute, not
  artifacts), so `rm -rf cache` is always safe.
