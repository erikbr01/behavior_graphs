# Running the experiments

Every experiment in the paper, as a command. Each section says what it needs,
roughly what it costs, and which document carries the detail — this page is the
runbook, not the reference.

**To check the published numbers rather than re-run anything**, use
[REPRODUCE.md](../REPRODUCE.md): it verifies all five results from a 60 MB
bundle in minutes. Everything below *produces* data instead.

```bash
export DATA_PATH=/path/to/data        # rollouts/, sweeps/
export MUJOCO_GL=egl                  # headless rendering
export UV_CACHE_DIR=/big/disk/uv HF_HOME=/big/disk/hf
```

---

## 1. Train a policy

Delegates to the environment's submodule, whose venv must exist
([envs.md](envs.md)).

```bash
python scripts/train_policy.py env=robomimic_lift_lowdim \
    env.dataset_path=/path/to/low_dim.hdf5

# submodule-side knobs go through train_overrides
python scripts/train_policy.py env=robomimic_square_lowdim \
    env.dataset_path=... \
    '+train_overrides={training.num_epochs: 50, training.compile: true}'
```

**Cost** minutes (lift, low-dim) to many hours (transport, image). **Real
robot** — DROID training, conversion and inline outcome labelling — is
[droid_workflow.md](droid_workflow.md).

## 2. Collect rollouts

The input to everything else. Writes the
[canonical layout](data.md#the-canonical-rollout-layout) with per-timestep
policy embeddings.

```bash
python scripts/collect_rollouts.py env=robomimic_lift_lowdim \
    env.dataset_path=/path/to/low_dim.hdf5 \
    checkpoint=runs/train/.../checkpoints/latest.ckpt \
    num_episodes=50
```

**Eval and collection are not the same measurement.** They run under different
horizons and seeds — on one `transport_mh` policy, 0.46 against 0.88. See
[known_issues.md](known_issues.md) before comparing a success rate to a graph.

## 3. Build one behavior graph

```bash
python -m behavior_graphs.pipeline.run rollout_dir=$DATA_PATH/rollouts/<name>
```

The embedding tap resolves itself from the collection. A collection carrying
two taps is an error naming both; pass `features.layer=<name>`.

**With no policy checkpoint at all**, from recorded video through a frozen
vision backbone — one of the three representations Tab. III compares:

```bash
python -m behavior_graphs.pipeline.run rollout_dir=... \
    'pipeline/features@features=dino_patch'
```

See [vision_backbone_features.md](vision_backbone_features.md).

## 4. Sweep the construction choices

```bash
python scripts/run_sweep.py grid=quick \
    "datasets={lift: $DATA_PATH/rollouts/robomimic_lift_mh}"

# several collections at once; each resolves its own tap
python scripts/run_sweep.py grid=full \
    "datasets={a: $DATA_PATH/rollouts/a, b: $DATA_PATH/rollouts/b}"
```

Grids are in `configs/grid/`: `quick` 54 jobs, `full` 1080, `ablation_droid`
432 (the paper's Tab. III design), `ablation` 3420. Writes `sweep_index.csv`,
`best_per_k.csv` and sensitivity reports; stages are shared through a
content-addressed store, so K and smoothing reuse a fitted reduction.
[sweep_study.md](sweep_study.md) covers model selection.

**Cost** minutes for `quick` on one collection, hours for `full` on several.
Resumable — a job with `metrics.json` is skipped.

---

## 5. Table III — the ablation

Nine cells of representation × reduction over three DROID tasks, scored against
gripper-derived subtask boundaries.

```bash
# (a) sweep each task on the paper's grid                      GPU, hours
cd scripts/ablation_study && ./launch_sweep.sh settable "" 3 ablation_droid

# (b) re-score the swept jobs into the artifacts the reports read
python scripts/ablation_study/study.py rescore --dataset settable
python scripts/ablation_study/study.py rescore --dataset settable --core

# (c) the table itself
python scripts/ablation_study/study.py report --dataset settable --which latex
python scripts/ablation_study/study.py report --dataset settable --which crosstask

# (d) the paper's nine cells, checked against the published values
./reproduce/01_table3_ablation.sh
```

The cells are maxima over a nuisance grid, so the design has to be **balanced**
— 144 configurations per representation, identical across arms. `(d)` refuses
to print if the counts differ.

Rebuilding from raw DROID captures (`.mcap` → frames → DiT embeddings) needs
`abc_minimal` from an ABC checkout, which is not vendored here; the rescored
sweeps are the earliest reproducible point.

## 6. Table II — progress estimation

Four methods over 43 collections. **[progress_estimation.md](progress_estimation.md)
is the full recipe**; the short form:

```bash
# one dataset, CPU, minutes
PYTHONPATH=src python scripts/run_progress_bench.py data.name=libero_goal

# the paper's four rows on one dataset
PYTHONPATH=src python scripts/run_progress_bench.py data.name=libero_goal \
    '~methods.graph_absorb' '~methods.frame_index' \
    +method@methods.mlp=mlp \
    +method@methods.robometer_4b=robometer_4b \
    +method@methods.topreward_ll=topreward_ll

# all 43, graph + MLP (CPU; shard across cores)
python scripts/bench_allds_with_vlms.py --out analysis/out/progress \
    --methods graph,mlp --shard 0 --num-shards 10

# the VLM baselines, one model per process, from robometer's venv
HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0 \
  third_party/robometer/.venv/bin/python scripts/bench_allds_with_vlms.py \
    --out analysis/out/progress --methods robometer_4b

# consolidate and export
PYTHONPATH=src python scripts/consolidate_progress_results.py \
    --curves-dir analysis/out/progress --out-dir analysis/out/consolidated
PROGRESS_RESULTS_ROOT=analysis/out/progress \
  PYTHONPATH=src python scripts/export_paper_results.py --sections progress
```

**Cost** graph+MLP about an hour across ten shards; ROBOMETER-4B a couple of
hours; TOPReward (8B) the better part of a day. Shard **by method, not by
dataset** when running the VLMs — the checkpoints do not share a 24 GB card,
and the seeded folds are what make the paired bootstrap comparable across
processes.

Use `topreward_ll`, not `topreward`: the latter is an in-house
reimplementation that scores 0.2 VOC higher and is not the published baseline.

## 7. Figure 3 — policy success rates

Pairwise exact tests with family-wise error control, over the scored rollout
collections of the baseline and retrained arms.

```bash
PYTHONPATH=src python scripts/export_paper_results.py --sections policies
python reproduce/verify_against_published.py policies
```

**Cost** a minute or two. The statistics run here rather than in the figure, so
the site and the paper read the same numbers. [policy_comparison.md](policy_comparison.md)
covers the tests; note the stated α in [known_issues.md](known_issues.md).

## 8. Table I and Figure 4 — the user study

Administering it needs participants; the analysis needs only the response
store.

```bash
# serve the study (one variant per session config)
streamlit run apps/study/Home.py

# the figures and the adjusted effects
STUDY_RESULTS_DIR=/path/to/participant/store \
  PYTHONPATH=src python scripts/export_paper_results.py --sections study,adjusted
python reproduce/check_study_claims.py
```

**Cost** minutes — four NUTS chains. `STUDY_RESULTS_DIR` is the one variable
worth setting by hand: a dev checkout's own `study_results/` holds lab previews
and yields zero participants. [data.md](data.md) and
[user_study_design.md](user_study_design.md).

## 9. The figures

```bash
BEHAVIOR_GRAPHS_RUNS_ROOT=$DATA_PATH/sweeps streamlit run apps/demo/Home.py
# -> Figures -> choose a run -> export as PDF
```

Exported from the run directories themselves, so a figure cannot drift from the
graph it depicts. PDFs embed Type 42 fonts at the paper's column width.

## 10. VLM node labelling

Optional. Auto-names behaviors from contact sheets, in the paper's two-pass
scheme.

```bash
python scripts/label_nodes.py <run_dir> --model gemini-3.5-flash
python scripts/label_nodes.py <run_dir> --list-cameras   # what views exist
```

Needs an API key. [vlm_annotations.md](vlm_annotations.md).

---

## Order of operations

```
train ──► collect ──► graph ──► sweep ──► select
                        │                   │
                        ├──► progress estimation (Tab. II)
                        ├──► ablation re-score   (Tab. III)
                        └──► policy statistics   (Fig. 3)
```

Everything after `collect` reads the canonical layout and nothing else, so a
collection from any environment — or from no policy at all, through a vision
backbone — enters at the same point.
