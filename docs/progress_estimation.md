# Progress estimation (Table II)

Behavior graphs are also progress predictors: fit a piecewise-linear schedule
over the graph's behaviors, and the stage a rollout is in reads out as a
progress estimate. Table II compares that against three baselines across 43
rollout collections.

This is the most expensive result in the paper and the only one whose
intermediate artifacts are not in the repository. This page is the whole path:
raw data, preprocessing, the benchmark, the baselines, and the export.

| | |
|---|---|
| **methods** | `graph_stage` (ours), `mlp`, `robometer_4b`, `topreward_ll` |
| **metrics** | VOC, Kendall τ-b, MAE, success-AUROC |
| **datasets** | 43 collections — LIBERO 6, Robomimic 10, MimicGen 8, real-world 19 |
| **cost** | graph + MLP: CPU, minutes per dataset. The two VLMs: a 24 GB GPU, hours. |

---

## 0. What each method reads

Everything is read out of the [canonical rollout layout](data.md#the-canonical-rollout-layout).

| method | reads | hardware |
|---|---|---|
| `graph_stage`, `frame_index`, `graph_absorb` | `policy_embeddings/<layer>.npz` + `episodes/metadata.yaml` | CPU |
| `mlp` | the same policy embedding | CPU |
| `robometer_4b` | `media/` video | GPU, robometer venv |
| `topreward_ll` | `media/` video | GPU, robometer venv |

The MLP is the comparison that matters: it regresses progress from **the same
embedding the graph is built on**, so the difference between the two is the
graph's structure rather than the representation.

**The comparison is not input-matched, and should be read accordingly.** The
graph and the MLP consume the policy's internal embeddings; the two VLMs see
only raw RGB frames. The fair reading is conditional: *where policy embeddings
are available — and at rollout time they are — the graph is a far cheaper and
more accurate progress model than an 8B VLM on pixels.* It is not a claim that
the graph is the better model given the same input. The clearest failure mode
is a weak embedding: the SAFE sets on pi0fast prelogits score VOC 0.53 against
0.91-0.98 elsewhere.

### Which embedding layer

`layer: auto` resolves against `policy_embeddings/` only. That guard is not
cosmetic — the generic loader's own `auto` ordering falls through to
`observations/state.npz`, so a dataset with no embedding file silently gets its
graph built on raw simulator state while the report still labels the method
"π0.5 emb". An earlier cross-dataset pass was published wrong exactly that way.

Layers differ by family: DROID real-robot sets are `cond_proj`, the sim
families `bottleneck_plan_t0`, LIBERO `pi05`, the generalist-VLA sets
mean-pooled hidden states. `configs/progress_estimation/data/zoo.yaml` groups
the zoo by the layer on disk rather than by task name, and explains why.

## 1. Getting the data

The 43 collections are produced exactly like any other rollout collection —
see [data.md](data.md#producing-rollout-collections) for sim and
[droid_workflow.md](droid_workflow.md) for the real-robot path from raw `.mcap`
captures.

Two things specific to this benchmark:

- **Video is required for the VLM baselines.** 21 of the zoo's candidate
  datasets (the SAFE franka/widowx sets) have embeddings and labels but no
  clips, which is why they are excluded from the 43 rather than scored
  graph-only — a row comparing two methods on one subset and four on another
  is not a comparison. They are still listed, with the reasoning, at the bottom
  of `zoo.yaml`.
- **A dataset needs at least 2 successful episodes.** The stage schedule is fit
  on successes, so collections below that are skipped at runtime
  (`nut_assembly_d0` has zero and is left out of the zoo entirely).

### The dataset count, reconciled

`zoo.yaml` groups by family; the paper's table groups by domain, and the two do
not line up on one dataset. `transport_mh_salvage250` sits under
`long_horizon:` in the zoo but is a Robomimic transport collection, and
`PROGRESS_DOMAINS` in `scripts/export_paper_results.py` counts it as Robomimic.
That is the whole difference between the zoo's 9 + 20 and the table's 10 + 19.

## 2. One dataset, to check the setup

Start here. It is CPU-only and takes a few minutes:

```bash
export DATA_PATH=/path/to/data
PYTHONPATH=src python scripts/run_progress_bench.py \
    data.name=libero_goal backbone.clustering.n_clusters=128 \
    out=progress_results/libero_goal
```

```
| method | VOC ↑ | KENDALL ↑ | PEARSON ↑ | MAE ↓ | AUROC ↑ |
| graph_stage | 0.955 | 0.913 | 0.949 | 0.063 | 0.931 |
```

against a published `libero_goal` row of VOC 0.962, Kendall 0.925, MAE 0.061,
AUROC 0.928. The small gap is the split: this entry point uses the `aligned`
split with a fixed K, while the published run uses 3-fold `KFoldSplit` with K
chosen per fold on an inner validation split (§4).

Methods are selected with `+method@methods.<key>=<config>` and dropped with
`~methods.<key>`:

```bash
# the paper's four rows, on one dataset
PYTHONPATH=src python scripts/run_progress_bench.py data.name=libero_goal \
    '~methods.graph_absorb' '~methods.frame_index' \
    +method@methods.mlp=mlp \
    +method@methods.robometer_4b=robometer_4b \
    +method@methods.topreward_ll=topreward_ll
```

Each run writes `{dataset}_metrics.json`, `{dataset}_report.md` and one
`*_curves.npz` per method into `out`.

## 3. The baselines

### ROBOMETER-4B

A zero-shot 4B reward model, run from the vendored `third_party/robometer`
submodule in **its own venv** — its runtime deps conflict with this project's.
`configs/progress_estimation/method/robometer_4b.yaml` names the checkpoint;
model *code* comes from the submodule, so it must be initialised and built:

```bash
git submodule update --init third_party/robometer
(cd third_party/robometer && uv sync)      # builds third_party/robometer/.venv
HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 \
  third_party/robometer/.venv/bin/python scripts/run_progress_bench.py ...
```

The same venv serves `topreward_ll` below: both reach their model code through
this submodule.

A LIBERO-only variant, `robometer_fft`, is available through
`scripts/bench_allds_with_vlms.py --methods robometer_fft`. It is not a row in
Table II.

### TOPReward — use `topreward_ll`, not `topreward`

**Two implementations are in the tree and they disagree substantially.**

| config | class | what it is |
|---|---|---|
| `topreward_ll` | `TopRewardLL` | the authors' own code, reached through `third_party/robometer`, whose core is byte-identical to `topreward.clients.qwen.QwenClient.compute_instruction_reward`. **This is the Table II row.** |
| `topreward` | `TopRewardProbe` | our in-house reimplementation (teacher-forced P(yes)) |

On `libero_goal` they score 0.604 and 0.811 VOC. Running the reimplementation
by mistake produces a far stronger baseline than the paper reports and makes
the published comparison look wrong. The demo app hides it for that reason.

### Frame budget

Every VLM sees 64 linspace-subsampled frames per episode, and ROBOMETER's
frames are capped at a 448 px long side. Neither is tuning: ROBOMETER pushes a
whole trajectory through the vision tower in one forward and OOMs a 24 GB card
past a few hundred frames, while these clips range from 224×224 to 1920×960 —
a 37× spread in vision tokens. Both VLMs get the same budget, so they at least
see the same thing as each other. This does deviate from the published LIBERO
rows for ROBOMETER, which used whole episodes because they were short enough to
fit.

## 4. The full table

Three phases. Phase A is the cross-dataset graph result; phase B is the
head-to-head that Table II reports; phase C consolidates.

### Phase A — graph across the whole zoo (CPU)

```bash
DATA_PATH=/path/to/data PYTHONPATH=src \
  python scripts/sweep_progress_allds.py --out analysis/out/progress_allds
```

Sweeps K per dataset — K is not universal (LIBERO wants high K, MimicGen low)
and candidates are capped by the number of training windows. Writes
`sweep_results.json`.

If the sim families were first run on raw state and re-run on policy
embeddings, `scripts/merge_progress_sweep.py` folds the re-run over the base so
that every row uses the policy embedding.

### Phase B — all methods on identical folds (GPU)

```bash
HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 \
  third_party/robometer/.venv/bin/python scripts/bench_allds_with_vlms.py \
    --out analysis/out/progress_allmethods --methods graph,mlp
# one VLM per process -- the checkpoints do not share a 24 GB card
... --methods robometer_4b --shard 0 --num-shards 2
... --methods topreward_ll --shard 1 --num-shards 2
```

Sharded **by method, not by dataset**, and this matters for correctness as well
as memory: `KFoldSplit` is seeded and depends only on the success flags, so
separate processes scoring the same dataset produce identical folds and
identical episode ids. The paired per-episode bootstrap in phase C compares
like with like only because of that. An earlier set of artifacts came from
separate passes that left each method a different episode order, and the
"paired" comparison was silently pairing different episodes.

K is chosen by `TunedGraphBackbone` on a validation split of each fold's
**train** half, never on the fold being scored.

### Phase C — consolidate

```bash
PYTHONPATH=src python scripts/consolidate_progress_results.py \
    --curves-dir analysis/out/progress_allmethods \
    --sweep-json analysis/out/progress_allds/sweep_results.json \
    --out-dir analysis/out/progress_consolidated
```

Reads the saved curves — no model re-runs — and emits the aggregate and
per-dataset breakdown as markdown and CSV, with a 2000-sample paired bootstrap.

### Phase D — export

```bash
PROGRESS_RESULTS_ROOT=analysis/out/progress_allmethods \
  PYTHONPATH=src python scripts/export_paper_results.py --sections progress
```

Reads every `{dataset}_metrics.json` under `$PROGRESS_RESULTS_ROOT` (colon
separated for several) and writes the `progress` section of
`src/behavior_graphs/results/paper_results.json`, which is what the demo site
serves.

Only datasets where **every** compared method has a score are counted. A mean
over "whatever each method happened to be run on" is not a comparison, and
coverage is uneven in exactly the way that matters. Per-metric means are taken
over the datasets that define that metric, which is why a domain's AUROC can
rest on fewer datasets than its VOC — AUROC is undefined on a collection whose
rollouts all succeeded.

## 5. What reproduces, and how exactly

**The graph and MLP rows re-derive bit-exactly.** Everything in the benchmark
is seeded — the KFold split depends only on the success flags, K-means and the
MLP take a seed — so a re-run on the same embeddings returns the published
numbers to the last digit, not merely close to them. Verified on four datasets
spanning all four domains:

| dataset | method | metrics checked | largest difference |
|---|---|---|---|
| libero_goal | graph_stage, mlp | VOC, Kendall, MAE, AUROC | 0 |
| robomimic_square_mh | graph_stage, mlp | " | 0 |
| mimicgen_square_d0 | graph_stage, mlp | " | 0 |
| prepare_tray_baseline | graph_stage, mlp | " | 0 |

**The two VLM rows need more than this repository**: a 24 GB GPU, the
`third_party/robometer` submodule built with its own venv, and the Qwen3-VL-8B
and ROBOMETER-4B checkpoints from the hub. The submodule pin is correct and
carries the baseline code; the rows have not been re-derived here.

### Checking without re-running

`reproduce/02_table2_progress.sh verify` needs no data at all. It checks the
published table against itself — every per-domain mean and the all-43 average,
against the per-dataset rows behind them, 80 checks. A stale export or a
transcription slip shows up; a wrong measurement would not.

```bash
./reproduce/02_table2_progress.sh verify           # 80 consistency checks
./reproduce/02_table2_progress.sh one libero_goal  # re-run one dataset
./reproduce/02_table2_progress.sh all              # the whole zoo (hours)
```
