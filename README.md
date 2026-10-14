# Behavior Graphs

Code, data tooling and the interactive demo for **Behavior Graphs: Unsupervised
Graph Structures for Interpreting Robot Deployment Data**.

A behavior graph compresses a policy's rollouts into one interpretable directed
graph: rollouts are embedded with the policy's own representations, windowed,
reduced, and clustered into discrete behaviors; the empirical transitions
between behaviors — and between behaviors and outcomes — form the graph.

The paper is in [`paper/`](paper/) ([PDF](paper/behavior_graphs.pdf)), and the
interactive demo is at [try.behavior-graphs.com](https://try.behavior-graphs.com).

## What is here

```
Visualization              Behavior graphs               Adapters        Environments
─────────────              ───────────────               ────────        ────────────
apps/demo    ◄─┐           Feature extraction   ◄─┐                   ┌─ robomimic
apps/study   ◄─┤           Windowing/Aggregation  │   Training data ◄─┼─ mimicgen
apps/reviewer◄─┤           Dimensionality reduction├── Policy        ◄─┼─ droid
               ├─ run dir  Clustering              │   Rollouts      ◄─┼─ libero
               │           Graph construction      │                   ├─ libero-pro
               └───────────Evaluation            ◄─┘                   └─ vla_foundry
```

- **Environments** live in `third_party/` as submodules. All policy learning
  stays there: robomimic/mimicgen/droid train diffusion and flow-matching
  policies via `third_party/diffusion_policy` and `third_party/rgrl`;
  libero/libero-pro use π0.5 via `third_party/openpi`. The main repo contains no
  policy-learning configs.
- **Adapters** (`src/behavior_graphs/envs/`) implement the abstract interfaces
  `TrainingData` / `Policy` / `RolloutSet` (`core/env.py`) per environment and
  write rollouts in one canonical layout (`envs/rollout_dir.py`).
- **Feature sources** are interchangeable: the policy's own per-timestep
  embeddings (the default), raw state/actions (proprioception), or — with no
  checkpoint at all — a frozen vision backbone over the recorded episode video
  (`pipeline/features@features=dino_patch`,
  [`docs/vision_backbone_features.md`](docs/vision_backbone_features.md)).
  These are the three representations the paper's ablation compares.
- **Pipeline stages** (`src/behavior_graphs/stages/`) implement the abstract
  base classes in `core/stages.py`; every hyperparameter of every stage is a
  constructor argument bound to a hydra config group, so the whole stack is
  sweepable.
- **Visualization** (`src/behavior_graphs/viz/`, `apps/`) renders any run
  directory: clicking a node or a transition plays the episodes that went
  through it, starting where the behavior began.
- **Policy statistics** (`src/behavior_graphs/stats/`) compare policies on task
  success rather than on the graph: pairwise exact tests, family-wise error
  control, compact letter displays and Beta posteriors — the machinery behind
  Fig. 3. See [`docs/policy_comparison.md`](docs/policy_comparison.md).

## Setup

```bash
git clone --recurse-submodules <this repo>
cd behavior_graphs
uv venv && uv pip install -e ".[all]"
```

Environment-specific stacks (simulators, torch, JAX) are uv-managed per
submodule: `third_party/diffusion_policy/scripts/setup_env_uv.sh` builds the
training env for robomimic/mimicgen/droid; openpi manages its own via `uv sync`
(see `third_party/openpi/README.md`).

## Workflow

```bash
# 1. Train a policy (delegates to the env's submodule)
python scripts/train_policy.py env=robomimic_lift_lowdim

# 2. Collect rollouts with per-timestep policy embeddings
python scripts/collect_rollouts.py env=robomimic_lift_lowdim \
    checkpoint=runs/train/.../checkpoints/latest.ckpt num_episodes=50

# 3. Build the behavior graph
python -m behavior_graphs.pipeline.run rollout_dir=runs/rollouts/<...>

# 3b. ...or sweep the whole stack -- note the pipeline/<stage>@<stage>=
#     form to swap which config is selected (bare reduction=pca silently
#     sets a literal string instead of erroring, since each stage's group
#     is package-renamed via @<stage>; see configs/config.yaml's comment)
python -m behavior_graphs.pipeline.run -m rollout_dir=<...> \
    windowing.width=1,5,8 'pipeline/reduction@reduction=pca,umap' \
    clustering.n_clusters=5,10,20

# 3c. ...or run a sweep (hydra): per-stage grids, stage caching,
#     model-selection + sensitivity reports (docs/sweep_study.md;
#     grids in configs/grid/)
python scripts/run_sweep.py grid=quick 'datasets={lift: <rollout_dir>}'

# 4. Inspect
streamlit run apps/demo/Home.py
```

Each run (and each sweep job) writes a self-contained artifact directory —
resolved config, window features and metadata, reduced coordinates, labels,
`graph.json`, and `metrics.json` (see `pipeline/artifacts.py`). Intermediate
stage outputs are reused across runs through a content-addressed cache
(`pipeline/cache.py`, `cache_dir=null` to disable).

Real-robot (DROID) workflow — training, on-robot inference with inline outcome
labeling, session merging (`scripts/utils/merge_droid_dirs.py`), π0.5
fine-tuning: see [`docs/droid_workflow.md`](docs/droid_workflow.md).

## Reproducing the paper

| Paper element | How to reproduce |
|---|---|
| Tab. III — ablation over representation × reduction | `python scripts/ablation_study/paper_ablation.py` prints exactly the twenty-four numbers the table reports, and exits non-zero if any has drifted. The sweep behind it: `scripts/ablation_study/launch_sweep.sh`; ground truth: `scripts/ablation_study/build_gripper_gt.py`. |
| Tab. II — progress estimation over 43 datasets | `python scripts/run_progress_bench.py` for one dataset, `scripts/sweep_progress_allds.py` for the whole zoo (`configs/progress_estimation/data/zoo.yaml`). |
| Fig. 3 — retrained-policy success rates | `scripts/analysis/cld_analysis.py`, and the demo's **Policy Comparison** page. |
| Fig. 4 + Tab. I — user study | `scripts/analysis/user_study_mixed_effects.py` over the responses; the study itself is `apps/study` serving `three_stage_v1`. |
| Figs. 1–2 and the graph figures | the demo's **Figures** page, which exports them as the paper's PDFs. |

The numbers those produce are also checked in as data, in
`src/behavior_graphs/results/paper_results.json` — written by
`scripts/export_paper_results.py` from the artifacts, never typed in by hand —
which is what the reviewer page serves.

## VLM node labeling

Behavior graph nodes can be auto-labeled with a VLM (Gemini, Claude, Ollama,
LM Studio, …) using contact sheets sampled from rollout videos, in the two-pass
scheme the paper describes.

See **[docs/vlm_annotations.md](docs/vlm_annotations.md)** for setup, usage,
model options, HTML reports, and architecture details.

## Tests

```bash
pytest                  # unit tests (interfaces, stages, viz) — parallel, ~24s
pytest -m integration   # end-to-end: trains a cheap lift/square policy
pytest -m browser       # drives the app in headless Chrome (needs Chrome)

pytest -m 'not integration'   # everything but the simulators
```

`-n 8 --dist loadfile` is in `addopts`, and `integration` and `browser` are both
deselected by default, so `pytest` is the fast loop.

A `-m` on the command line replaces the one in `addopts` rather than adding to
it, which is why the last line above has to name the exclusion it wants.

`-m browser` covers what `AppTest` cannot see: it does not execute custom
components, so clicking a node or a fold badge in the SVG behavior graph — and
everything that follows from the value the component keeps re-sending — is only
reachable through a real browser. See `tests/browser/`.

**[docs/testing.md](docs/testing.md)** has the cost breakdown, which tests carry
which guarantee, and the test shapes to avoid.

## Citation

```bibtex
@inproceedings{bauer2026behavior,
  title     = {Behavior Graphs: Unsupervised Graph Structures for
               Interpreting Robot Deployment Data},
  author    = {Bauer, Erik and Thumm, Jakob and Agia, Christopher and
               Sinha, Rohan and Xu, Chen and Nishimura, Haruki and
               Bohg, Jeannette and Pavone, Marco and Itkina, Masha},
  year      = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE). The submodules under `third_party/` carry their
own licenses.
