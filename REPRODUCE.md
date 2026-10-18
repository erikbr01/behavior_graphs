# Reproducing the paper

Every number in *Behavior Graphs: Unsupervised Graph Structures for Interpreting
Robot Deployment Data*, and how to re-derive it.

The scripts in [`reproduce/`](reproduce/) **check** rather than render: each one
compares what it computes against the value the paper reports and exits non-zero
if they disagree. A green run is a reproduction, not a re-print.

```bash
export DATA_PATH=/path/to/data                 # rollouts/, sweeps/
export STUDY_RESULTS_DIR=/path/to/study_results   # the participant store -- see below
./reproduce/run_all.sh
```

```
== summary ==
  preflight                  reproduces
  Table III -- ablation      reproduces
  Table II  -- progress      reproduces
  Figure 3  -- policies      reproduces
  Figure 4  -- study answers reproduces
  Table I   -- study effects reproduces
```

---

## 0. Setup

```bash
git clone --recurse-submodules <this repo> && cd behavior_graphs
uv sync --all-extras          # the locked versions; `uv pip install` ignores uv.lock
```

Two steps need extras beyond the default install, and `00_preflight.sh` tells
you which are missing before anything long starts:

| extra | needed by | why |
|---|---|---|
| `analysis` | Table I | the Bayesian mixed-effects fit (bambi, arviz) |
| `viz` | Table I, Figure 4 | the regret read-out imports the app's outcome reader |

```bash
uv pip install -e ".[analysis,viz]"
```

If your interpreter for those differs from `python3`, pass it through:

```bash
PYTHON=/path/to/venv/bin/python ./reproduce/run_all.sh
```

**Check your setup first.** It is read-only and takes a second:

```bash
./reproduce/00_preflight.sh
```

### The data, and a trap in one environment variable

`DATA_PATH` is the data root holding `rollouts/` and `sweeps/`.

Everything on this page reads per-episode **outcomes** plus one rescored sweep
— no video, no features, no checkpoints — so the whole input is about 60 MB.
If you have a full data root, carve the bundle and work from that:

```bash
SRC=/path/to/full/data SRC_STUDY=/path/to/study_results \
  ./reproduce/make_data_bundle.sh data_bundle
export DATA_PATH=$PWD/data_bundle
export STUDY_RESULTS_DIR=$PWD/data_bundle/study_results
```

`STUDY_RESULTS_DIR` is the **participant** response store, and it is the one
variable worth setting by hand. Two directories look like it and are not:

- a development checkout grows its own `study_results/`, holding the lab app's
  *previews* — every record `origin: lab`. Point at it and the study steps
  correctly report no participants, which reads like missing data rather than
  the wrong directory;
- it is **not** necessarily under `DATA_PATH`. On the machine the paper was
  produced on it sits beside the data root, not inside it.

The filter behind this matters more than it looks. Of 84 stored sessions, 84 are
participant-origin, 20 ran to completion, and **18** are on the layout the paper
reports — the n it quotes. The other two completed an earlier pilot layout: the
same questions they were asked are not the questions the paper analyses. Counting
them would turn "16 of 18 preferred the graph" into 80% of 20.

The store also holds `raffle_entries.jsonl` — participant email addresses.
Nothing here reads it, and `make_data_bundle.sh` copies responses by explicit
glob so it cannot reach a bundle.

---

## 1. Table III — the ablation (representation × reduction)

```bash
./reproduce/01_table3_ablation.sh           # prints and verifies
./reproduce/01_table3_ablation.sh --latex   # the table body, to paste
```

**Needs** `$DATA_PATH/sweeps/{abc_settable,abc_preparetray,abc_cooking_v2}/rescored.json`.
**Takes** seconds — it reads the rescored sweep and recomputes nothing.

Computes exactly the twenty-four numbers the paper reports: nine cells of
representation × reduction, each a mean and standard deviation over the three
DROID tasks, plus a mean row per representation. It then checks all twenty-four
against the published values and both bolded maxima, and **refuses to print at
all** if the cell counts behind the rows differ — because a cell that reached its
value over a larger nuisance grid won on search budget rather than merit.

Expect `0.226 / 0.132 / 0.113` for the policy-embedding, proprioception and
DINOv3 means, and `All 24 published values reproduce`.

To rebuild the sweep from rollouts instead of reading the rescore, see
`scripts/ablation_study/study.py` and `launch_sweep.sh`. That is a GPU job of
hours, and `build_gripper_gt.py` regenerates the gripper-derived ground truth.

## 2. Table II — progress estimation over 43 collections

```bash
./reproduce/02_table2_progress.sh verify          # consistency of the published table
./reproduce/02_table2_progress.sh one libero_goal # re-run one dataset
./reproduce/02_table2_progress.sh all             # the whole zoo (hours)
```

**This is the one result whose intermediate artifacts this repo does not
carry.** The benchmark writes a `{dataset}_metrics.json` per collection and the
export reads `$PROGRESS_RESULTS_ROOT`; those files are not in the checkout, so
`verify` cannot recompute the numbers. What it does instead is check the
published table against itself: every per-domain mean, and the all-43 Average
row, against the per-dataset rows behind them — 80 checks. A stale export or a
transcription slip shows up; a wrong measurement would not.

A genuine re-derivation means re-running the benchmark. One dataset is cheap
and CPU-only; the full zoo is 43 collections × 4 methods and the two VLM
baselines dominate it.

**[docs/progress_estimation.md](docs/progress_estimation.md) is the full
recipe** — raw data and preprocessing, which method reads what, the three
benchmark phases, how to set up the two VLM baselines (including which of the
two TOPReward implementations is the paper's), and the export.

## 3. Figure 3 — retrained-policy success rates

```bash
./reproduce/03_fig3_policies.sh --stdout   # compute without touching the checked-in file
python reproduce/verify_against_published.py policies
```

**Needs** `$DATA_PATH/rollouts/` — the scored rollout collections of the baseline
and retrained policies. **Takes** a minute or two.

Each violin is a Beta posterior over that arm's successes and trials; the letters
above the columns come from pairwise exact tests with family-wise error held at
5%. The statistics run here rather than in the figure, so the website and the
paper read the same numbers.

This recomputes from the rollouts and diffs every value against the published
one. It reproduces exactly.

## 4. Table I and Figure 4 — the user study

```bash
./reproduce/04_table1_study.sh --stdout                  # recompute both sections
python reproduce/verify_against_published.py study       # Figure 4: exact
python reproduce/check_study_claims.py                   # Table I: claims
```

**Needs** `$STUDY_RESULTS_DIR` (see the trap above) and the `analysis` + `viz`
extras. **Takes** minutes — four NUTS chains.

**Figure 4** — the raw ratings, the display choices and the regret — reproduces
exactly, and is diffed value by value.

**Table I is checked differently, on purpose.** It is a Bayesian mixed-effects
fit with unseeded chains, so re-running it moves the third significant figure
every time. Diffing digits would fail a perfectly good reproduction. So
`check_study_claims.py` checks what the paper actually claims:

- temporal demand and effort are **reduced**, intervals excluding zero
- confidence and success-rate regret are **unchanged**, intervals including zero
- every point estimate agrees with the published one within 2 rating points

A representative re-fit:

| outcome | published | re-fit |
|---|---|---|
| temporal demand | −26.1 | −25.9 |
| effort | −22.9 | −23.0 |
| confidence | +2.5 | +2.4 |
| regret | +0.9 | +0.8 |

If you add an outcome to the table, the script **fails** on the unrecognised
label rather than silently checking nothing for it.

## 5. Figures 1 and 2, and the graph figures

The behavior-graph figures (including the cooking tree in Fig. 5) are exported
from the demo app's **Figures** page, which draws them from the run directories
themselves so a figure cannot drift from the graph it depicts:

```bash
BEHAVIOR_GRAPHS_RUNS_ROOT=$DATA_PATH/sweeps streamlit run apps/demo/Home.py
# -> Figures -> choose a run -> Export as a paper figure (PDF)
```

Figures 1 and 2 are composed diagrams rather than plotted output. The paper
source and the exact figure files it uses live in the paper repository,
[corl_2026_behavior_graphs](https://github.com/erikbr01/corl_2026_behavior_graphs).

The export itself is verified: rendering a run directory from
`$DATA_PATH/sweeps` through `graph_figure.behavior_graph_figure_files` — the
same call the page's download buttons make — produces the PDF and PNG, and the
figure is the graph that run describes. What is not driven end to end here is
the Streamlit navigation around it, which needs a browser; `pytest -m browser`
covers that.

---

## What reproduces, and how exactly

| result | status | from |
|---|---|---|
| Table III — ablation | **exact**, all 24 values | sweeps on disk |
| Figure 3 — policy success rates | **exact**, value by value | raw rollouts |
| Figure 4 — study answers | **exact**, value by value | response store |
| Table I — adjusted effects | **claims hold**, digits move | re-fit (unseeded MCMC) |
| Table II — progress estimation | **self-consistent**; re-derivation needs a re-run | — |

## Building a behavior graph on your own data

Reproducing the paper is one thing; using the method is another. The shortest
path from a rollout collection to a graph:

```bash
# one graph
python -m behavior_graphs.pipeline.run rollout_dir=$DATA_PATH/rollouts/<name>

# or sweep the construction choices and pick by a proxy metric
python scripts/run_sweep.py grid=quick "datasets={<name>: $DATA_PATH/rollouts/<name>}"

# then look at it
BEHAVIOR_GRAPHS_RUNS_ROOT=<sweep_dir> streamlit run apps/demo/Home.py
```

Rollouts need per-timestep features. Collect them with
`scripts/collect_rollouts.py`, or build a graph with no policy checkpoint at all
from a frozen vision backbone over the recorded video — see
[`docs/vision_backbone_features.md`](docs/vision_backbone_features.md).
