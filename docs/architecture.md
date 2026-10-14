# Architecture

The repo separates three concerns, mirroring the system diagram:

```
Visualization        Behavior graphs            Adapters             Environments
apps/, viz/          core/ + stages/ +          envs/                third_party/
deploy/              pipeline/                  (interfaces only)    (all policy learning)
```

## Core contracts (`src/behavior_graphs/core/`)

| File | Contract |
|---|---|
| `stages.py` | One ABC per pipeline stage: `FeatureExtractor → Windowing → DimReduction → Clustering → GraphConstruction → Evaluator → NodeLabeler` (the last runs as an optional step after evaluation, `cfg.labeling`) |
| `env.py` | Environment interfaces: `TrainingData`, `Policy`, `TrajectorySet` (base: any episodes + per-timestep arrays, used directly for featurized demos) with `RolloutSet` as its policy-rollout subclass, `EnvAdapter` |
| `types.py` | `EpisodeInfo`, `FeatureSet`, `WindowInfo`, `WindowSet` — the only types stages exchange |
| `graph.py` | `BehaviorGraph` (empirical transition graph + value estimates + path enumeration), JSON-serializable |
| `result.py` | `PipelineResult` accumulator consumed by evaluators |

Rules that keep the repo clean:

1. **The main repo contains no policy-learning code or configs.** Training,
   rollout collection, and embedding extraction live in `third_party/`
   submodules; adapters only shell into them and translate outputs.
2. **Stages only see core types.** A stage never touches an environment,
   a file format, or another stage's internals.
3. **Everything crosses the env boundary through one on-disk format** — the
   canonical trajectory layout (`envs/trajectory_dir.py`), read by
   `TrajectoryDirectory` for any source (e.g. demos featurized by
   `extract_trajectory_features`) and by its subclass `RolloutDirectory` for
   policy rollouts:

```
<dir>/
    episodes/metadata.yaml            source block (provenance) +
                                      episodes: [{index, length, success, ...}]
    policy_embeddings/<layer>.npz     (total_steps, d) float32
    observations/<key>.npz            optional per-timestep arrays (e.g. state)
    actions.npz                       optional executed actions
    media/ep{idx:04d}_{tag}.mp4       optional videos, single view
    media/<camera>/ep{idx:04d}_{tag}.mp4
                                      optional videos, one dir per camera view
```

`media/<camera>/` is how a multi-camera platform (DROID records a wrist plus
one or more exterior views) stores every view of an episode, and it is what
writers emit — one view or six — so a clip keeps the name it was recorded under.
The flat form is a *read* compatibility layout for the datasets that predate
named views; it reads back as the single view `DEFAULT_CAMERA` ("default"), and
the two can coexist in one dataset. Readers go through `camera_names()` /
`video_paths()`, and the demo app's sidebar picks which views the video panels
show.

Orientation and view order are configured, not stored: a `cameras:` block in
`configs/data/<group>.yaml` declares `rotate`/`mirror` per camera name (DROID's
wrist camera hangs upside down) and flags one camera `default: true` as the view
shown first — otherwise views are ordered alphabetically, which decides the
default by accident of naming. It is baked into that view's web proxy and
poster, which key the media cache by it, so one orientation reaches the player,
its poster and the thumbnail strip alike — and the recorded clips are still never
rewritten, so a wrong guess costs a re-transcode of the derived cache.

4. **Value estimation is an optional add-on, not part of graph
   construction.** The graph is built purely from empirical transitions;
   per-node values (MRP approximation) only appear when the `values`
   evaluator is configured, and everything downstream (artifacts, demo app)
   works without them. `GraphConstruction` is `TransitionGraphBuilder`
   (`stages/graph/transition.py`), selected via `pipeline/graph@graph=`:
   empirical transition counts, with optional count-based edge pruning and
   sticky smoothing.
5. **Every stage hyperparameter is a constructor argument bound to a hydra
   config group**, so the entire stack is sweepable from the CLI and every
   run (or sweep job) writes a self-contained artifact directory
   (`pipeline/artifacts.py`).
6. **Sweeps are first-class** (see `docs/sweep_study.md`):
   intermediate stage outputs are reused across runs through a
   content-addressed cache (`pipeline/cache.py`); `pipeline/sweep.py` runs
   validated, grouped hyperparameter grids over one or more datasets, driven
   by the hydra entry point `scripts/run_sweep.py` (+ `configs/sweep.yaml`,
   `configs/grid/*.yaml`); `model_selection/` turns the
   resulting run population into composite scores, Pareto fronts,
   best-per-K recommendations, and per-knob sensitivity tables.

## Run artifact directory

```
<run_dir>/
    config.yaml      resolved config          labels.npy      cluster per window
    windows.npz/json window features+meta     graph.json      behavior graph
    reduced.npz      reduced features         node_values.json MRP values
    coords_2d.npy    2-D viz coordinates      metrics.json    all evaluator outputs
```

The visualization apps (`apps/demo`, `apps/study`) read only this layout
plus the rollout dir referenced in `metrics.json["rollout_dir"]`. That value is
recorded *relative to the data root* (`rollouts/<dataset>`) by
`RunArtifacts.save_metrics`, and joined back by `resolve_rollout_dir`, so one
run dir resolves both on the machine that produced it and in a container that
mounts the data volume elsewhere. The root itself (`$DATA_PATH`) is read only
at entrypoints — a CLI's `main()`, an app page, or a hydra config — via
`core/io.py`'s `resolve_data_path`, and passed down explicitly; library code
takes it as an argument.

## Extending

**New environment**: subclass `EnvAdapter` in `envs/<name>.py`, register it in
`envs/__init__.py`, add `configs/env/<name>.yaml`. Implement `training_data`,
`train_policy`, `collect_rollouts` by delegating to a `third_party/` submodule
that writes the canonical rollout layout. See `envs/diffusion_policy_env.py`
(sim envs) and `envs/libero.py` (uv-based submodule) for the two patterns.

**Graphs without a policy checkpoint**: the features group also has a
policy-free source. `pipeline/features@features=dino_patch`
(`stages/features/vision_backbone.py`) embeds the episode *videos* every
rollout directory already stores with a frozen DINOv2, keeping the patch
tokens as a coarse spatial map rather than the CLS/mean summary, so a graph
can be built for an uninstrumented policy, an unreachable stack, or human
demonstrations. `scripts/utils/extract_video_features.py` bakes the same
features into `policy_embeddings/<layer>.npz` for everything that addresses
representations by layer name. See `docs/vision_backbone_features.md`.

**New stage implementation**: subclass the stage ABC under
`stages/<group>/<name>.py` with plain-typed constructor args, add
`configs/<group>/<name>.yaml` with a `_target_`. The config test
(`tests/test_configs.py`) picks it up automatically.
