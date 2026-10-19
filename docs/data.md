# Data

Nothing in this repository ships data. This page says what the code expects on
disk, which parts of it each result needs, and how to produce the parts you do
not have.

## Three kinds of data, three environment variables

| variable | holds | read by |
|---|---|---|
| `DATA_PATH` | `rollouts/` and `sweeps/` | everything |
| `STUDY_RESULTS_DIR` | participant response records | Fig. 4, Tab. I |
| `PROGRESS_RESULTS_ROOT` | `{dataset}_metrics.json` per collection | Tab. II |

`BEHAVIOR_GRAPHS_RUNS_ROOT` is separate: it points the apps at a directory of
pipeline *outputs* (run or sweep directories), not inputs.

### `STUDY_RESULTS_DIR` is not necessarily under `DATA_PATH`

Point it at the **participant** store. Two other directories look like it and
are not it:

- a development checkout grows its own `study_results/` when you run the study
  app locally; every record in it is `origin: lab`, and the study steps will
  correctly report zero participants, which reads like missing data rather than
  the wrong directory;
- the working store on the machine the paper was produced on sits *beside* the
  data root, not inside it.

So set it explicitly rather than deriving it, and check the count: of 84 stored
sessions, 84 are participant-origin, 20 ran to completion, and **18** are on the
layout the paper reports. The other two completed an earlier pilot layout — the
questions they answered are not the questions the paper analyses, and counting
them would turn "16 of 18 preferred the graph" into 80% of 20.

The participant store also holds **`raffle_entries.jsonl`, which is participant
email addresses.** It is not read by anything here, and it must not travel with
any copy of the response data. `reproduce/make_data_bundle.sh` copies response
files by explicit glob so that it cannot.

## The reproduction bundle (~60 MB)

Every published number except Tab. II comes from per-episode **outcomes** and
one rescored sweep. No video, no features, no checkpoints. That makes the
reproduction input small enough to publish on its own:

```
data_bundle/
  rollouts/<collection>/episodes/metadata.yaml   75 files   3.4 MB
  sweeps/<task>/rescored.json                     3 files    56 MB
  study_results/*.json                           85 files   350 KB
```

Carve one out of a full data root:

```bash
SRC=/path/to/full/data SRC_STUDY=/path/to/study_results \
  ./reproduce/make_data_bundle.sh data_bundle

DATA_PATH=$PWD/data_bundle STUDY_RESULTS_DIR=$PWD/data_bundle/study_results \
  ./reproduce/run_all.sh
```

That run is the check that the bundle is complete — Tab. III, Fig. 3, Fig. 4 and
Tab. I all pass against the 60 MB tree, with no access to the ~1 TB it came
from.

**Tab. II is not in the bundle.** Its benchmark reads policy embeddings and
video; see [progress_estimation.md](progress_estimation.md).

## The canonical rollout layout

Every environment adapter writes the same shape, and the pipeline only ever
reads this. See `src/behavior_graphs/envs/trajectory_dir.py` for the full
contract.

```
<collection>/
  episodes/metadata.yaml           per-episode length + success + provenance
  policy_embeddings/<layer>.npz    (total_steps, d) float32 — the graph's input
  observations/<key>.npz           optional raw observations
  actions.npz                      optional, action executed at each step
  media/ep0000_success.mp4         optional, single view
  media/<camera>/ep0000_success.mp4  optional, one directory per view
```

What each consumer needs:

| consumer | needs |
|---|---|
| success rates, study regret | `episodes/metadata.yaml` |
| behavior graph, progress estimation (graph, MLP) | `+ policy_embeddings/<layer>.npz` |
| VLM baselines, the app's video panes | `+ media/` |
| vision-backbone features (no checkpoint) | `episodes/` + `media/` only |

## Producing rollout collections

### Simulated (robomimic, mimicgen, LIBERO, RoboCasa)

Train in the submodule, then collect through the adapter — one command each,
and the output is already canonical:

```bash
python scripts/train_policy.py env=robomimic_lift_lowdim \
    env.dataset_path=/path/to/low_dim.hdf5
python scripts/collect_rollouts.py env=robomimic_lift_lowdim \
    env.dataset_path=/path/to/low_dim.hdf5 \
    checkpoint=runs/train/.../checkpoints/latest.ckpt num_episodes=50
```

Per-environment setup (which venv, which submodule, which embedding layer) is
in [envs.md](envs.md).

### Real robot (DROID)

Raw captures are per-episode `.mcap` logs, which need decoding and assembling
before anything here can read them. That path — conversion, trimming, session
merging, inline outcome labelling, and `assemble_droid_sync_recordings.py` —
is documented end to end in [droid_workflow.md](droid_workflow.md), under
*Assembling the canonical layout*.

Reading raw captures needs the `droid` extra (`mcap`, `mcap-protobuf-support`,
`opencv-python`); it is not part of the default install because nothing but
preprocessing uses it.

```bash
uv pip install -e ".[droid]"
```

One preprocessing step has a dependency that is not on PyPI:
`scripts/ablation_study/extract_dit_embeddings.py` imports `abc_minimal`,
which comes from the ABC policy checkout and must be on `PYTHONPATH`. It is
needed only to regenerate the DiT embeddings behind Tab. III from raw
captures; the rescored sweep in the bundle already has them.

### Without a policy checkpoint

A behavior graph can be built from recorded video alone, using a frozen vision
backbone instead of policy embeddings — see
[vision_backbone_features.md](vision_backbone_features.md). This is one of the
three representations Tab. III compares.
