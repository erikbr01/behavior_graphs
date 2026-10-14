# Real-world DROID workflow

End to end: demonstrations → policy training → on-robot inference →
labeled rollouts → behavior graphs. Two policy stacks:

- **Diffusion policy** (`third_party/diffusion_policy`) — fully wired:
  training, on-robot collection, embedding extraction, and the behavior
  graph pipeline all connect through the adapters. Use this path first.
- **π0.5** (`third_party/openpi`) — fine-tuning and on-robot inference are
  supported by the submodule and documented below; embedding capture for
  behavior graphs is wired for LIBERO but not yet for DROID (see status).

The typical loop:

```
1. collect demos     raw DROID sessions on the robot
2. convert           robo_dataset → VideoStore (compact, fast to train on)
3. train             train.py (submodule) → checkpoint
4. collect + label   droid_collect_rollouts.py (labels each episode inline)
   rollouts          → one canonical session dir per run
       └ scripts/utils/merge_droid_dirs.py if you ran several sessions
5. behavior graph    run_sweep.py / pipeline.run → GUI
```

## 0. One-off setup

```bash
git submodule update --init third_party/diffusion_policy third_party/robo_dataset
cd third_party/diffusion_policy && scripts/setup_env_uv.sh   # training env (uv)
```

The robot workstation needs the DROID robot stack (`~/src_droid`) installed.
`droid_collect_rollouts.py --dry-run` exercises the full loop with synthetic
observations without the NUC (use it to test the plumbing before robot time).

### Always launch training via `source .venv/bin/activate` on machines with system CUDA

If your GPU box has CUDA installed via the NVIDIA apt repo (GCE AI/DL images
do this by default), `/etc/ld.so.conf.d/cuda-XX-Y.conf` registers
`/usr/local/cuda-XX.Y/…/lib` with `ldconfig` system-wide. On the first
convolution, cuDNN 9 lazily `dlopen`s `libcublas.so.12` / `libcublasLt.so.12`
by unversioned soname, which resolves through `ldconfig` to the system CUDA
libs — not the venv's `nvidia-cublas-cu12==12.1.3.1` from
`site-packages/nvidia/cublas/lib/`. Torch 2.4 is ABI-tied to cu12.1, so an
`ldconfig`-registered cu12.9 (or newer) install shadows the venv and training
aborts with:

```
Invalid handle. Cannot load symbol cublasLtCreate
Aborted (core dumped)
```

Fix: always launch training through an activated venv shell:

```bash
conda deactivate                                        # if base is active
source third_party/diffusion_policy/.venv/bin/activate
python third_party/diffusion_policy/train.py …
```

`setup_env_uv.sh` patches `.venv/bin/activate` to prepend the venv's
`nvidia/*/lib` dirs to `LD_LIBRARY_PATH`, so the unversioned `dlopen` picks
up the venv's cu12.1 libs first. Invoking `.venv/bin/python` directly
(without `source`) skips that prepend and re-triggers the crash. Same reason
`(base)` conda shells work by accident — conda's activate also sets
`LD_LIBRARY_PATH` to `$CONDA_PREFIX/lib`, coincidentally shadowing system
CUDA.

Symptom-free machines are usually ones where CUDA was installed via the
`.run` file without a `/etc/ld.so.conf.d/` entry — no `ldconfig` shadowing,
so torch's own `_load_global_deps()` preload wins by default and the
`LD_LIBRARY_PATH` prepend is redundant (but harmless).

See diffusion_policy PR #7 for the full debug log; parent-repo tracking is
in behavior_graphs PR #28.

## 1. Configs for a new task

Two files, one per layer of the stack:

1. **Submodule task config** — copy
   `third_party/diffusion_policy/configs/image/droid/set_table_baseline/`
   to `configs/image/droid/<my_task>/` (inside the submodule) and adjust
   image keys / horizons there. All policy hyperparameters live in the
   submodule, never in the main repo.
2. **Main-repo env config** — copy `configs/env/droid.yaml`:

```yaml
# configs/env/kendama.yaml
_target_: behavior_graphs.envs.droid.DroidAdapter
task: image/droid/<my_task>          # submodule config dir
dataset_path: ???                     # set per-run or here
embedding_layer: bottleneck_plan_t0
```

## 2. Demonstration data

`DroidAdapter` reads demonstrations through `robo_dataset`, which accepts
**raw DROID session folders directly** — no conversion required:

```
<dataset_root>/success/<date>/<timestamp>/
    trajectory.h5                  robot states + actions
    recordings/MP4/<serial>.mp4    per-camera videos (or SVO/)
```

(`DroidStore` drops `movement_enabled=False` teleop idle steps and aligns
camera frames automatically.)

**Recommended: convert once for training speed.** Raw sessions decode SVO
video on every epoch; a converted `VideoStore` (H.264 + zarr lowdim) is the
compact, fast format the task configs were benchmarked with.

### Converting to VideoStore

Requires a Python 3.9 env with `pyzed` (installed alongside the ZED SDK) and
`robo_dataset` (`pip install -e third_party/robo_dataset`):

```bash
LD_LIBRARY_PATH=/path/to/zed_sdk/lib \
python scripts/utils/convert_droid_data.py \
    --data-dir ${DATA_ROOT}/droid_raw \
    --output   ${DATA_ROOT}/droid_converted \
    --camera wrist_image_left=14313307 \
    --camera exterior_image_1_left=36716034 \
    --zed-settings-path ${DATA_ROOT}/zed_settings \
    --workers 5 \
    --skip-empty --skip-corrupt
```

Key flags:

| Flag | Purpose |
|---|---|
| `--workers N` | Parallel conversion via N subprocesses (~13 min with 5 workers for ~740 sessions) |
| `--skip-empty` | Skip sessions where the movement mask selects zero steps (robot was idle the whole time) |
| `--skip-corrupt` | Skip sessions where camera decode fails (e.g. truncated SVO) |
| `--camera OBS_KEY=ZED_SERIAL` | Map an observation key to a ZED serial; repeat for each camera |
| `--zed-settings-path` | Dir containing `SN<serial>.conf` calibration files (required for offline SVO decode) |

Camera serials for the DROID workstation:

| Key | Serial |
|---|---|
| `wrist_image_left` | `14313307` |
| `exterior_image_1_left` | `36716034` |

The script writes a `tasks.json` sidecar alongside the VideoStore that maps
each episode index to its language instruction (`current_task` from
`trajectory.h5` attrs). This is used for task filtering at training time.

**Regenerating tasks.json without re-converting** — if the VideoStore exists
but `tasks.json` is missing (e.g. converted before the sidecar was added):

```bash
LD_LIBRARY_PATH=${DATA_ROOT}/zed_sdk_extracted/lib \
python scripts/utils/write_droid_tasks_sidecar.py \
    --raw-data-dir ${DATA_ROOT}/droid_raw \
    --store-path   ${DATA_ROOT}/droid_converted \
    --zed-settings-path ${DATA_ROOT}/zed_settings
```

This uses header-only SVO reads (no image decode, ~seconds per session) to
apply the same corrupt-session filter without a full re-conversion.

### Running conversion on a remote GPU box (ZED NVDEC)

Bulk conversions are heavy (SVO decode + H.264 encode per camera per frame)
and are best run off the training GPUs — e.g. a spare cloud A100. The
converter uses the ZED SDK for SVO decode, which offloads to the GPU's NVDEC
engines *only if the NVIDIA video libraries match the running driver*.

One-time setup on the box (Ubuntu, ZED SDK 4.2 extracted under
`~/zed_sdk_extracted`):

```bash
# 1. Driver-matched NVIDIA video codec libs (the driver install often omits
#    them). Match the version to `nvidia-smi --query-gpu=driver_version`:
sudo apt-get install -y libnvidia-decode-<VER>-server libnvidia-encode-<VER>-server
sudo ldconfig                      # e.g. VER=580 for driver 580.173.02

# 2. pyzed in the conversion venv (Python 3.9). Copy from a working box:
#    cp -r <working>/pyzed <working>/pyzed-4.2.dist-info \
#          .venv/lib/python3.9/site-packages/

# 3. Run with only the ZED libs on LD_LIBRARY_PATH (libnvcuvid/libnvidia-encode
#    resolve via ldconfig from step 1):
export LD_LIBRARY_PATH=$HOME/zed_sdk_extracted/lib
python scripts/utils/convert_droid_data.py \
    --data-dir ~/droid_raw/success --output ~/data/<store> \
    --dates 2026-07-27,2026-07-28 \
    --camera wrist_image_left=14313307 \
    --camera wrist_image_right=14313307:right \
    --camera exterior_image_1_left=36716034 \
    --zed-settings-path ~/zed_settings \
    --workers 8 --skip-empty --skip-corrupt
python scripts/utils/trim_droid_idle_beginnings.py \
    --store ~/data/<store> --output ~/data/<store>_trimmed
```

**Hiccups (and fixes):**

- **ZED decodes on CPU (slow, `nvidia-smi` decoder util ~0%).** `libnvcuvid`
  did not match the driver (a stale hand-staged copy shadowed it). Install
  the driver-matched `libnvidia-decode-*` as above. The ZED SDK also links
  `libnvidia-encode` at load, so install both — even on an A100, which has
  **NVDEC but no NVENC**. (Encode is always CPU x264; fine, since frames are
  a tiny 180×320.)
- **Every right-eye episode silently skipped** with
  `No SVO recording for serial '<serial>:right'`. The `:right` stereo-view
  decode support was missing from the checked-out `robo_dataset` commit.
  Ensure the submodule is current (`:right` merged in robo_dataset PR #4;
  parent pin bumped in behavior_graphs PR #47) — a stale pin produces a store
  that silently drops half the cameras.
- **Worker count.** The box is CPU-bound on encode; ~`vcpus / 1.4` workers
  saturates without thrashing (e.g. 8 on a 12-vCPU box). More just
  oversubscribes.
- **Getting the raw data there.** From the GPU box, `gsutil -m rsync` straight
  from the bucket is far faster than a local→box `rsync` over a home uplink.

The converter is **not** incremental — it refuses a non-empty `--output`, and
each worker writes into `<output>_shards/shard_NN`, merged only at the end
(so the finalized episode count stays 0 until the run is nearly done).

### Adding a new instruction variant to an existing store

Say a store already holds 200 "cooking" episodes and new sessions for a
related instruction (e.g. "cooking - carrot") show up later. Re-converting
everything from scratch would duplicate the 200 existing episodes' videos on
disk and burn the conversion time again. Instead, convert only the new
sessions into their own small store, then append it onto the existing one:

```bash
# 1. see what's there and how many successes exist per instruction
python scripts/utils/fetch_droid_language_stats.py --data-dir ${DATA_ROOT}/droid_raw

# 2. symlink just that task's successful raw sessions into a filtered tree
#    (task string must match fetch_droid_language_stats.py's output exactly)
python scripts/utils/filter_droid_task_sessions.py \
    --data-dir ${DATA_ROOT}/droid_raw \
    --output   /tmp/cooking_carrot_sessions \
    --task "cooking - carrot"

# 3. convert just those sessions into a separate, throwaway store -- use the
#    *same* --camera/--action-space/--image-size/--fps/--crf as the existing
#    store, since append_droid_store.py checks they match
python scripts/utils/convert_droid_data.py \
    --data-dir /tmp/cooking_carrot_sessions \
    --output   /tmp/cooking_carrot_store \
    --camera wrist_image_left=14313307 \
    --camera exterior_image_1_left=36716034 \
    --zed-settings-path ${DATA_ROOT}/zed_settings \
    --skip-empty --skip-corrupt

# 4. trim idle beginnings, the same as the target store went through (the
#    *_trimmed stores the configs train on). Trimming leaves no trace in meta.json,
#    so this is the one mismatch append_droid_store.py cannot catch: appending
#    an untrimmed store onto a trimmed one yields a store whose episodes are
#    preprocessed two different ways. Skip this step only if the target store
#    is itself untrimmed.
python scripts/utils/trim_droid_idle_beginnings.py \
    --store  /tmp/cooking_carrot_store \
    --output /tmp/cooking_carrot_store_trimmed

# 5. append it onto the existing store in place -- copies the new episodes'
#    videos/lowdim at new, contiguous episode ids, merges tasks.json, bumps
#    n_episodes; every existing episode's files are left untouched. --dry-run
#    runs every compatibility/consistency check and writes nothing.
python scripts/utils/append_droid_store.py --dry-run \
    --src /tmp/cooking_carrot_store_trimmed \
    --dst ${DATA_ROOT}/droid_converted_cooking_trimmed
python scripts/utils/append_droid_store.py \
    --src /tmp/cooking_carrot_store_trimmed \
    --dst ${DATA_ROOT}/droid_converted_cooking_trimmed
```

The target store must use the per-episode layout the converter writes today.
The older flat layout (`data/<key>/` arrays plus `episode_ends.npy`, e.g.
`droid_converted` and `droid_converted_absjoint`) is deprecated and rejected:
its readers take the episode count from `episode_ends.npy`, so appended
episodes would be invisible. Pass such a store through
`trim_droid_idle_beginnings.py` first — that rewrites it into the current
layout — and append onto the result.

The store's `tasks.json` now has both "cooking" and "cooking - carrot" labels
over disjoint episode-id ranges, so a training config can draw from both (see
"Task-specific training configs" below):

```yaml
task:
  dataset:
    store_path: ${DATA_ROOT}/droid_converted_cooking_trimmed
    episodes_per_task:      # frozen index lists — unaffected by the append
      cooking:
        [3, 7, 11, ...]     # the baseline's exact 200 episodes
      cooking - carrot:
        [200, 201, ...]     # the 30 appended episodes, at the new ids
```

Use frozen `episodes_per_task` lists, not `task_filter` +
`max_episodes_per_task`, if the variant run has to train on the baseline's
exact episodes. The seed-based path draws
`rng.permutation(<the task's episode ids>)[:cap]`, so it is only stable while
that id list is: appending episodes under a *new* task label leaves the base
draw alone, but appending more "cooking" episodes re-draws it entirely. The
train/val split is also recomputed over `n_episodes`, so a plain `val_ratio`
split shifts after any append regardless of task labels.

### Discovering task language instructions

Before writing a training config, inspect what tasks and how many successful
demonstrations are in the dataset:

```bash
python scripts/utils/fetch_droid_language_stats.py \
    --data-dir ${DATA_ROOT}/droid_raw
```

Prints a table of language instructions sorted by successful episode count.
Example output for the set-table dataset:

```
task                              total  success  failure  unknown
prepare plate - milk right           71       59       12        0
prepare plate - milk middle          71       59       12        0
prepare plate - milk left            65       52       13        0
```

### Task-specific training configs

Task configs live in
`third_party/diffusion_policy/configs/image/droid/<task_name>/config.yaml`.
See `configs/image/droid/set_table_baseline/config.yaml` for a complete
example.

**Resolution and crop.** DROID cameras record at 180×320 (9:16). ResNet uses
global average pooling so arbitrary input sizes work — no need to resize to
224×224. Keep native resolution and apply a proportional crop:

```yaml
policy:
  crop_shape: [162, 288]   # 9:16 ratio, ~10% inset — preserves aspect ratio
  eval_fixed_crop: true    # center-crop at inference (random crop at training)
shape_meta:
  obs:
    wrist_image_left:
      shape: [3, 180, 320]
      type: rgb
    exterior_image_1_left:
      shape: [3, 180, 320]
      type: rgb
```

`eval_fixed_crop: true` is required: it replaces the robomimic
`CropRandomizer` with a deterministic center-crop at inference time so the
encoder receives the same crop geometry it saw during training (random during
training, center at eval). Without it, the encoder runs on the full 180×320
at inference, which is a distribution shift.

**Task filtering — frozen episode lists.** For reproducible ablation
comparisons, episode indices are frozen explicitly in the config using
`episodes_per_task`, bypassing all runtime sampling so baseline datasets are
guaranteed identical across runs:

```yaml
task:
  dataset:
    _target_: diffusion_policy.dataset.robo_store_dataset.RoboStoreImageDataset
    store_path: ${DATA_ROOT}/droid_converted
    episodes_per_task:
      prepare plate - milk right:
        [10, 13, 14, 15, 21, 30, ...]   # explicit episode indices from tasks.json
      prepare plate - milk middle:
        [2, 5, 8, 11, ...]
    use_shm: true
```

Generate and write these index lists with the helper script (run once per
store, commit the resulting configs):

```bash
python scripts/utils/generate_set_table_episode_selection.py \
    --store ${DATA_ROOT}/droid_converted_set_table_trimmed \
    --config-dir third_party/diffusion_policy/configs/image/droid \
    --seed 42

# dry-run (print only, no writes):
python scripts/utils/generate_set_table_episode_selection.py ... --dry-run

# verify existing configs without regenerating:
python scripts/utils/generate_set_table_episode_selection.py ... --verify-only
```

The script selects a base set via `rng.choice(seed)` and any extra episodes
via `rng.permutation(remaining)[:n]`, which guarantees the base set is a
strict subset of any extended set. It verifies uniqueness and index bounds
before writing.

**Task filtering — seed-based sampling.** Older configs (e.g.
`set_table_baseline`) instead set `task_filter` and cap per task at runtime:

```yaml
task:
  dataset:
    task_filter:
      - prepare plate - milk right
      - prepare plate - milk middle
    max_episodes_per_task: 50   # cap per instruction; omit for no cap
    use_shm: true               # load filtered episodes into RAM (see below)
```

`task_filter` values must match the strings returned by
`fetch_droid_language_stats.py`. Note: `choice(size=60)` and
`choice(size=120)` with the same seed are independent draws, so two configs
capped at different sizes will not have a nested relationship — that's what
the frozen `episodes_per_task` lists above are for.

**In-RAM caching (`use_shm: true`).** With 100 filtered episodes (two cameras
at 180×320, ~31k frames), uncompressed RAM footprint is ~10.7 GB. Setting
`use_shm: true` decodes all filtered training episodes from H.264 into shared
memory once at startup (~34 s for 91 training episodes), then serves every
batch from RAM with zero disk I/O. GPU utilization goes from bursty to
sustained 99%. Use when:

- dataset fits in RAM (10–20 GB for typical task-specific subsets)
- training run is long enough to amortize the startup decode (~34 s here)

For the full 725-episode DROID store (~78 GB uncompressed), omit `use_shm`
and rely on the VideoStore's per-worker LRU decoder cache with 12+ workers.

## 3. Train

Training invokes the submodule's `train.py` directly (the `scripts/train_policy.py`
adapter works for configs with a `dataset_path` key, but `set_table_baseline`
hardcodes `store_path`):

```bash
SUBMODULE=third_party/diffusion_policy
OUTPUT=${OUTPUT_PATH}/set_table_baseline

$SUBMODULE/.venv/bin/python $SUBMODULE/train.py \
    --config-path $SUBMODULE/configs/image/droid/set_table_baseline \
    --config-name config \
    "hydra.run.dir=$OUTPUT" \
    "multi_run.run_dir=$OUTPUT"
```

For tasks whose config has `dataset_path` or `store_path` as a top-level
override key (i.e. not hardcoded), use the adapter:

```bash
python scripts/train_policy.py env=droid \
    env.task=image/droid/<my_task> \
    env.dataset_path=${DATA_ROOT}/droid_converted
```

Checkpoints land in `<output>/checkpoints/` (`latest.ckpt` + top-k by
`test_mean_score`). For DROID there is no sim env runner, so
`test_mean_score` stays 0; `latest.ckpt` is the checkpoint to use.

**Throughput** (RTX 4090, batch 256, 180×320 images, use_shm=true): ~4 it/s
(≈1024 samples/s), 99% GPU utilization. At 3000 epochs × 136 steps/epoch ≈
~35 hours.

## 4. Collect and label rollouts (one step)

`scripts/droid_collect_rollouts.py` **in the submodule** is the interactive
runner: sync / async-chunk / async-streaming inference modes, per-episode
instruction. It records each episode, **prompts you for the outcome right
then**, and writes the canonical layout — no separate labeling pass.

Three processes across two machines:

**NUC** — robot server (Franka + ZED cameras):
```bash
python droid/scripts/server/run_server.py
```

**GPU workstation** — policy server (diffusion-policy venv, no DROID stack needed):
```bash
cd third_party/diffusion_policy
source .venv/bin/activate
python -m diffusion_policy.real_world.droid.policy_server \
    --checkpoint ../../checkpoints/set_table_baseline.ckpt \
    --device cuda:0 --host 0.0.0.0 --port 5001 \
    --embed-layer bottleneck_plan_t0 \
    --num-inference-steps 75
```

`--num-inference-steps` overrides the checkpoint's default (100 DDPM steps).
75 is a good balance between quality and latency; 16 is too noisy for this
checkpoint.

**Robot workstation** — control loop + data recording (`zed_env_spacemouse`, talks to both):
```bash
conda activate zed_env_spacemouse
cd third_party/diffusion_policy
python scripts/droid_collect_rollouts.py \
    --policy http://<gpu_box_ip>:5001 \
    --output-dir ../../rollouts/set_table_baseline_test_jun22 \
    --num-episodes 20 \
    --instruction "prepare plate - milk middle" \
    --mode async_chunk --control-hz 15 --max-timesteps 800 \
    --embed-layer bottleneck_plan_t0 \
    --open-loop-horizon 4 --warmup-steps 20 --warmup-horizon 16
```

**Controls:**
- **SPACE** — start episode / stop episode early
- **q / Ctrl-C** — quit at any prompt
- After each episode: `s` success, `f` failure, `u` unknown, `r` retry

**Horizon tuning:**
- `--open-loop-horizon 4` — steady-state: re-queries policy every ~267ms for
  reactivity near goals
- `--warmup-horizon 16` / `--warmup-steps 20` — uses a longer horizon for the
  first 20 steps (~1.3s) so the policy commits to an action before the
  dead-zone at episode start causes it to get stuck

**Sync mode** — re-queries the policy every control step instead of acting
open-loop for `--open-loop-horizon` steps; use `--compile` on the policy
server to keep inference fast enough for this mode:

```bash
# GPU workstation
python -m diffusion_policy.real_world.droid.policy_server \
    --checkpoint ../../checkpoints/set_table_v2_baseline_plusleft.ckpt \
    --device cuda:0 --host 0.0.0.0 --port 5001 \
    --embed-layer bottleneck_plan_t0 \
    --num-inference-steps 75 --compile

# Robot workstation
python scripts/droid_collect_rollouts.py \
    --policy http://127.0.0.1:5001 \
    --output-dir ../../rollouts/set_table_baseline_v2_demo_videos_sync \
    --num-episodes 20 \
    --instruction "set table - milk right" \
    --mode sync --control-hz 15 --max-timesteps 1000 \
    --embed-layer bottleneck_plan_t0 \
    --open-loop-horizon 12
```

This keeps the diffusion-policy env on the GPU box and the DROID stack on the
robot workstation cleanly separated. The runner script has no diffusion-policy
dependency — it communicates with the policy server over HTTP.

**Output layout** — each episode dir is timestamped and self-contained:

```
<output-dir>/
    recordings/ep_YYYYMMDDTHHMMSS/
        trajectory.h5                   robot state + actions (DROID format)
        wrist_image_left.mp4            CRF-18 H.264 at control Hz
        exterior_image_1_left.mp4
        embeddings_bottleneck_plan_t0.npz   (n_steps, D) float32
        metadata.yaml                   instruction, success, length, all CLI args
```

One MP4 per camera, named after the observation key it came from
(`for k, frames in image_buffers.items()` in
`diffusion_policy/real_world/droid/env.py`) — nothing has to be enabled to record
more than one, and this is what the multi-camera review players downstream are
built on (see "Camera views" below).

Which cameras those are is the **intersection of a fixed candidate list in the
recorder with the obs keys actually present** — not whatever the checkpoint's
`shape_meta.obs` names:

```python
image_keys = [k for k in ("wrist_image", "wrist_image_right",
                          "exterior_image_1_left", "exterior_image_2_left")
              if episode_data and k in episode_data[0]["obs"]]
```

So a genuinely new camera has to be added to that tuple *in the submodule*, via
its own PR there before the parent pin moves, or it is silently never written and
no amount of downstream config will conjure it. It is also why the
wrist camera appears under two names across collections (`wrist_image` and
`wrist_image_left`, depending on the submodule commit a session was collected
with) and why `configs/data/droid.yaml` lists both.

Multiple sessions with different instructions can share the same `--output-dir`;
each episode is globally unique by timestamp.

Embeddings are extracted **online** by the policy server during inference and
saved per episode. The MP4s are saved for **offline re-extraction** of any
other layer on the GPU box:

```bash
uv run python scripts/droid_extract_embeddings.py \
    --store ../../rollouts/set_table_baseline_test_jun22/recordings \
    --checkpoint ../../checkpoints/set_table_baseline.ckpt \
    --output-dir ../../rollouts/set_table_baseline_test_jun22 \
    --layer bottleneck_plan_t0
```

After each episode:

```
episode 0000 outcome — [s]uccess / [f]ailure / [u]nknown / [r]etry
outcome> f dropped on release
```

- `r` re-runs the same episode without saving (hardware glitch / bad reset).
- The prompt is active only when stdin is a terminal; headless/batch runs
  record `success: null` and continue automatically.
- `--policy` takes `http://host:port` (recommended), `ws://host:port`
  (openpi), or a local checkpoint path (only if the DROID stack and
  diffusion-policy share an env).
- `--embed-layer` must match what the server was started with; omit on both
  sides to skip online embedding extraction.

```bash
uv run python scripts/droid_extract_embeddings.py \
    --store /data/rollouts/set_table_jun19/recordings \
    --checkpoint ${OUTPUT_PATH}/set_table_baseline/checkpoints/latest.ckpt \
    --output-dir /data/rollouts/set_table_jun19 \
    --layer bottleneck_plan_t0
```

### Assembling the canonical layout

`droid_collect_rollouts.py` only writes `recordings/ep_<timestamp>/` — the
pipeline reads a canonical rollout dir with `episodes/metadata.yaml`,
`policy_embeddings/<layer>.npz`, and `media/` at the *root*, not inside
`recordings/`. Build that root layout with whichever of these two matches how
the session was collected:

- **Sync / async-chunk mode (the common case)** — each episode already has an
  online `embeddings_<layer>.npz` saved next to it. Assemble the root layout
  directly from those, no checkpoint needed:

  ```bash
  python scripts/utils/assemble_droid_sync_recordings.py \
      data/rollouts/<new_rollout_dir> --layer bottleneck_plan_t0
  ```

  This sorts `recordings/ep_*` by timestamp, concatenates their embeddings,
  copies every camera the collector recorded into `media/`, and preserves each
  episode's `note`/collector `args` in `episodes/metadata.yaml`. Pass
  `--overwrite` to replace an already-assembled `episodes/`/`media/`/
  `policy_embeddings/`.

  **Camera views.** A DROID session records a wrist and one or more exterior
  views per episode (one MP4 per obs image key — see "Output layout" above).
  Each becomes its own `media/<camera>/` directory, named after the source MP4
  (`media/wrist_image_left/ep0000_success.mp4`), and the demo app's sidebar picks
  which of them the video panels show — side by side and in sync.

  **Nothing has to be passed to get all of them: exporting every camera is the
  default.** The view names are discovered as the *union* of the MP4 names across
  `recordings/ep_*`, so a camera that only starts recording partway through a
  collection still gets its own view, and the episodes missing it simply
  contribute no clip there (the players render the views each episode has).

  **The default view is the left exterior camera**, and it says so in
  `configs/data/droid.yaml` — the `cameras:` entry flagged `default: true`:

  ```yaml
  cameras:
    exterior_image_1_left:
      default: true      # shown first; the one view a single-view panel shows
    wrist_image_left:
      rotate: 180
  ```

  That view leads everywhere one view is picked: the leftmost of the side-by-side
  players, the first option in the sidebar picker, the thumbnail strip's tiles,
  `entry["path"]` in the app's episode index, and `RolloutDirectory.video_path()`
  with no camera. The exterior camera frames the whole task; the wrist view is a
  close-up that only reads next to it.

  Flag it rather than rely on the fallback. Unflagged, views are ordered
  alphabetically by directory name (`_scan_media` in `envs/trajectory_dir.py`),
  which happens to put `exterior_…` before `wrist_…` — so a future camera whose
  name sorts earlier (`back_…`, `agentview`) would silently take the default. A
  flagged camera a given dataset does not have changes nothing for it, so one
  group-wide flag is safe across a family whose members carry different cameras.
  Flag two and the first declared wins, with a warning.

  `--camera NAME` (repeatable) narrows the export. A one-camera export still
  keeps that camera's name — `media/exterior_image_1_left/`, not a bare
  `media/*.mp4`:

  ```bash
  # exterior view only
  python scripts/utils/assemble_droid_sync_recordings.py \
      data/rollouts/<new_rollout_dir> --camera exterior_image_1_left
  ```

  This script used to collapse a single view into the flat `media/` form, which
  reads back as an unnamed default view — and anything keyed by camera goes with
  the name, the `cameras:` block below most of all, so a `--camera
  wrist_image_left` export rendered upside down with no way to say otherwise.
  Readers still accept the flat form, so datasets assembled before this keep
  working; only new assemblies differ. A pre-existing flat dataset that needs a
  rotation can still name the view it reads back as: `cameras: {default: {rotate:
  180}}`.

  **Camera orientation.** The wrist camera hangs upside down on the gripper, so
  its frames are stored 180° off. `configs/data/droid.yaml` says so once, for the
  whole group:

  ```yaml
  cameras:
    wrist_image_left:
      rotate: 180        # 0, 90, 180 or 270; `mirror: true` flips left-right
    wrist_image:
      rotate: 180        # the same camera under the older obs-key name
  ```

  Both current wrist keys are listed, so a newly assembled dataset gets the
  rotation the moment it is registered in that group — there is nothing per
  dataset to add. A run that introduces a *new* rotated view (a second exterior,
  `wrist_image_right`) needs one more entry here; an unlisted camera is shown
  exactly as recorded.

  The rotation is baked into that view's web proxy and poster rather than applied
  as a CSS transform on the `<video>` — transforming the element turns its native
  control bar upside down with the picture, and leaves every frame decoded from
  the clip (poster, strip tiles) disagreeing with it. The recorded clips are
  untouched: the orientation is part of the media cache key, so changing it
  re-transcodes the derived proxies and nothing else. Re-run
  `scripts/prewarm_media.py` after an edit, or the affected views transcode on
  first click.

  Precedence runs from the most specific outwards: a dataset's own `cameras:`
  block (under its `path:`, alongside `display_name:`) beats its group's, which
  beats the groups that group inherits — and a group's block reaches only the
  datasets that group includes, so this one never follows `all.yaml` onto the sim
  datasets. `rotate: 0` on a dataset cancels the group default for one camera; a
  spec that fails validation does *not* (it is dropped, logged, and the group
  default still applies).

  **Pre-warming the rotated views.** A view's orientation is part of its cache
  key, so the rotated variant is a different entry from the unrotated one and has
  to be warmed as the app will ask for it. `BEHAVIOR_GRAPHS_DISPLAY_DATA` is what
  tells the script which group's `cameras:` blocks to resolve, so set it:

  ```bash
  BEHAVIOR_GRAPHS_DISPLAY_DATA=droid python scripts/prewarm_media.py \
      /path/to/rollouts/<new_rollout_dir> --strip-thumbs
  ```

  It prints how many clips it found rotated, which is the quick check that the
  config reached them (`55 of them rotated/mirrored: {'rotate': 180}`). Run it
  again after any `cameras:` edit, or the affected views transcode on first click.
  On the deployment it has to run *inside* the demo container, so it writes into
  the shared media-cache volume the apps read:

  ```bash
  docker exec deploy-demo-1 python scripts/prewarm_media.py \
      /data/rollouts/<new_rollout_dir> --strip-thumbs
  ```

  `BEHAVIOR_GRAPHS_DISPLAY_DATA` is already set on that container, and `scripts/` is
  bind-mounted alongside `src/`/`configs/`, so this runs the checkout's copy. That
  mount is load-bearing: without it the command silently ran the older copy baked
  into the image, which reported `done: 240/240 in 0.1s` while warming the raw
  `recordings/` captures and skipping `media/<camera>/` entirely.

  **Video transcoding.** By default the copy step also re-encodes each
  video via `ffmpeg`: scaled to a 640px-wide cap, H.264 at CRF 28, with
  `+faststart` for seekable progressive playback. Raw DROID captures run
  ~11 Mbps at 720p (~25–30 MB per episode) — far more bitrate than a review
  player needs, and large enough to crash a browser tab when the study/demo
  apps render several at once (the video browser used to base64-embed every
  video into the page on each Streamlit rerun before switching to
  `st.video`; see `video_browser.py`). Transcoding typically shrinks a
  session's `media/` dir by ~100x with no visible quality loss at
  review-player size — e.g. `set_table_baseline_v2_sync` went from 1.7 GB
  to 12 MB. Override with:

  | Flag | Purpose |
  |---|---|
  | `--no-transcode` | Copy source videos as-is (e.g. already-small input) |
  | `--max-width N` | Cap transcoded width in pixels, aspect-preserving (default: 640) |
  | `--crf N` | x264 constant-rate-factor; higher = smaller/lower quality (default: 28) |

  To re-transcode a session assembled before this default existed, just
  re-run with `--overwrite` — as long as its raw `recordings/` tree is
  still on disk, this rebuilds `media/` with the smaller encode without
  needing to re-collect anything:

  ```bash
  python scripts/utils/assemble_droid_sync_recordings.py \
      data/rollouts/<already_assembled_dir> --layer bottleneck_plan_t0 --overwrite
  ```

- **Async-streaming mode, or extracting a different layer** — no online
  embedding was saved (a known gap in `_run_async_streaming`/`_run_sync`
  policy-embedding collection), so run the offline extractor shown above
  (`droid_extract_embeddings.py --store <dir>/recordings --checkpoint ...`)
  instead — it re-runs the checkpoint over the saved MP4s.

  **This path is single-view.** The store-based export takes one camera key and
  writes the flat `media/` layout (`export_media(store, …, camera_key: str)` in
  `diffusion_policy/real_world/droid/store_embeddings.py`), so a collection
  assembled this way has one view however many cameras recorded. Multi-camera
  export exists only on the sync/async-chunk `recordings/` path above. If a run
  collected in async-streaming mode needs both views, the raw MP4s are still in
  `recordings/ep_*` — that is unbuilt work in the store exporter, not a config.

Before assembling, it's worth checking every `recordings/ep_*/` dir is
non-empty (`find <dir>/recordings -type f -size 0`) — a collection session
that stalls or is killed mid-write can leave zero-byte placeholder files for
episodes that never actually recorded.

### Worked example: a new sync-mode session end to end

Starting right after the raw session lands under `data/rollouts/` (e.g.
copied/rsynced from the robot workstation as
`recordings/ep_<timestamp>/{trajectory.h5, wrist_image_left.mp4,
exterior_image_1_left.mp4, embeddings_bottleneck_plan_t0.npz, metadata.yaml}`,
with no `episodes/`/`media/`/`policy_embeddings/` yet):

```bash
cd <repo>

# 1. sanity-check the copy landed intact — no zero-byte files from a
#    stalled/interrupted transfer:
find data/rollouts/set_table_baseline_v2_plusmiddle_sync/recordings -type f -size 0

# 2. assemble the canonical layout in place. No --camera: every recorded view
#    is exported, one media/<camera>/ each.
python scripts/utils/assemble_droid_sync_recordings.py \
    data/rollouts/set_table_baseline_v2_plusmiddle_sync --layer bottleneck_plan_t0
# → exporting 2 camera view(s): exterior_image_1_left, wrist_image_left
# → data/rollouts/set_table_baseline_v2_plusmiddle_sync: 60 episodes — 32 success, 28 failure, 0 unlabeled; layer bottleneck_plan_t0

# 3. spot-check it reads back correctly — episode count, embeddings, and that
#    the views are there with the left exterior camera as the default one:
python -c "
import sys; sys.path.insert(0, 'src')
from behavior_graphs.envs.trajectory_dir import RolloutDirectory
r = RolloutDirectory('data/rollouts/set_table_baseline_v2_plusmiddle_sync')
print(len(r.episodes()), 'episodes,', r.embeddings('bottleneck_plan_t0').shape, 'embeddings')
print('views:', r.camera_names(), '— default:', r.video_path(0).parent.name)
"
# → views: ['exterior_image_1_left', 'wrist_image_left'] — default: exterior_image_1_left

# 4. append it to the existing sweep (resumable — only this dataset's jobs run):
python scripts/run_sweep.py grid=full \
    layers=[bottleneck_plan_t0] \
    sweep_dir=${DATA_PATH}/sweeps/sweep_all \
    'datasets={set_table_baseline_v2_plusmiddle_sync: ${DATA_PATH}/rollouts/set_table_baseline_v2_plusmiddle_sync}'
```

The dataset shows up in the demo app the moment step 4 writes job output
under `sweep_dir` — no separate registration step needed for that. It's
still worth a fifth step if the new session belongs to an existing
`data=<group>` family (see below):

```bash
# 5. register it alongside its siblings for future `data=droid` sweeps —
#    and, for a DROID dataset, this is also what gives its wrist view the
#    group's `rotate: 180` (transforms are looked up by dataset key):
$EDITOR configs/data/droid.yaml
#   set_table_baseline_v2_plusmiddle_sync: ${oc.env:DATA_PATH}/rollouts/set_table_baseline_v2_plusmiddle_sync

# 6. warm the proxies, including the rotated wrist ones:
BEHAVIOR_GRAPHS_DISPLAY_DATA=droid python scripts/prewarm_media.py \
    ${DATA_PATH}/rollouts/set_table_baseline_v2_plusmiddle_sync \
    --strip-thumbs
# → 60 of them rotated/mirrored: {'rotate': 180}
```

Step 5 matters more for a multi-camera dataset than it used to: transforms are
looked up by dataset key, so an unregistered dataset gets no `cameras:` block. In
a local checkout (no `BEHAVIOR_GRAPHS_DISPLAY_DATA`, so the `all` group is read) it
is listed but its wrist view renders upside down; in the deployment
(`BEHAVIOR_GRAPHS_DISPLAY_DATA=visualize`) it stays hidden until registered.

### Adding a dataset to an existing sweep

No dataset registry file is *required* to sweep a dataset — the inline
`datasets={...}` override always works standalone, and the Streamlit demo app
finds runs by scanning the sweep's output dirs for `labels.npy` + window
metadata (local `windows.json`, or a `shared.json` pointing into `_shared/`)
rather than reading `data/rollouts/`. A newly swept dataset therefore shows up
on its own **unless `BEHAVIOR_GRAPHS_DISPLAY_DATA` is set** (it is in the
deployment, to `visualize`) — that names a `configs/data/` group to restrict
the selector to, so a dataset missing from the group stays hidden however much
job output it has. To add a freshly-assembled dataset into an existing sweep,
pass just that one dataset — the sweep is resumable and skips any job dir that
already has a `metrics.json`, so there's no need to re-list datasets already in
there:

```bash
python scripts/run_sweep.py grid=full \
    sweep_dir=${DATA_PATH}/sweeps/sweep_all \
    'datasets={<new_rollout_dir>: ${DATA_PATH}/rollouts/<new_rollout_dir>}'
```

No `layers=` needed: with one dataset there is trivially a common layer, so
auto-detection picks it (`cond_proj` for the rgrl policies). Pass it
explicitly only to force a layer the dataset also carries under another name.

`configs/data/droid.yaml` exists purely as a convenience on top of that —
one `name: ${oc.env:DATA_PATH}/rollouts/<name>` line per dataset, grouped
into families (`set_table`, `prepare_tray`, `cooking`). Add a new
same-family entry there instead of hand-writing `datasets={...}` every time,
then sweep the whole family in one command. Every entry in the group is a
flow-matching rgrl policy carrying `cond_proj`, so that one layer is common to
all of them and auto-detection resolves it — no `layers=` override:

```bash
python scripts/run_sweep.py data=droid grid=full \
    sweep_dir=${DATA_PATH}/sweeps/sweep_all
```

#### Relabelling a dataset in the app

**For a new dataset, pick a good key and stop there.** The key becomes the
sweep task-dir name (`sweep_dir/<key>/<hash>/`), and that dir name is what the
app displays — so the key already *is* the label. Nothing below applies.

What follows is for a dataset **already swept** under a name it has since
outgrown. By then the key is a historical fact, not a label: the app reads the
dir on disk and never the key, so editing the key alone makes the dataset
*disappear* (the allowlist matches keys against task-dir names, and a key with
no matching dir filters everything out). Making the key the label for real
would mean renaming the task dirs — 672 run dirs for the four `set_table` runs
— plus every sweep-root file that names them (`jobs.json`, `datasets.yaml`,
`best_per_k.csv`, `sweep_index.csv`, `sensitivity_*.csv`), or the next resumed
sweep stops recognizing that work as done. And it still would not finish the
job: the sweep-analysis and model-selection dataset pickers label by the
`rollout_dir` recorded in each run's `metrics.json`, so those keep the old name
until the `rollouts/` dir is renamed and all 672 `metrics.json` rewritten. Not
worth a re-sweep for a label. So instead, expand the one-line entry into the
mapping form:

```yaml
set_table_baseline_pretrained_dit:
  path: ${oc.env:DATA_PATH}/rollouts/set_table_baseline_pretrained_dit
  display_name: set_table_baseline_dit
```

The `display_name` applies only where the app renders a task or dataset label:
the Task selector and metric, the chat header, and the two dataset pickers.
Recorded paths, run dirs, cache keys and the URL query params all keep the real
name, so a relabelled dataset is still selected and linked to by the same key —
and swept work stays valid, since a job's identity is `_job_hash(dataset path,
sorted overrides)` (`pipeline/sweep.py`) and neither the path nor the key
changed. `configs/` is bind-mounted into the deployment, so an edit lands on the
next rerun with no rebuild (within the labels' 60 s cache TTL). A typo in an
entry — a missing `path:`, a misspelled `display_name:`, a half-saved file —
leaves the dataset listed under its real name and logs why (`docker compose logs
demo`), rather than either hiding it or taking the page down.

Name a layer only to sweep something other than the detected one. Where a group
*does* span policy families (`data=all` mixes six layers), auto-detection falls
back to one pass per layer present, and `run_sweep.py` skips the datasets that
lack the layer of the pass being run rather than failing their jobs — it logs
which ones and sweeps the rest. The same skip covers a dataset with no
`policy_embeddings/` at all, e.g. one still sitting as a raw `recordings/` tree.

### Inference transform chain

The observation preprocessing at inference matches training exactly:

| Step | Training | Inference |
|---|---|---|
| Camera output | 180×320 uint8, BGR | 180×320 uint8, BGR |
| Color convert | `DroidStore` BGR→RGB | `env._preprocess_image` BGR→RGB |
| Resize | stored at 180×320 (native) | `ObsWindowBuffer`: INTER_AREA to shape_meta H×W (no-op at native res) |
| uint8→float | `ImageToTensorDevice` → CHW float32 [0,1] | `frames_to_policy_format` → CHW float32 [0,1] |
| Image normalizer | identity (`fixed_range` [0,1]→[0,1]) | identity (same checkpoint normalizer) |
| Crop | random 180×320 → 162×288 (in-encoder `CropRandomizer`) | center 180×320 → 162×288 (`eval_fixed_crop: true` replaces randomizer) |

The `eval_fixed_crop: true` config key is load-bearing: it swaps the
robomimic `CropRandomizer` for `dmvc.CropRandomizer` (deterministic center
crop) when the checkpoint is loaded for inference. If it is absent or false,
the policy runs on the full uncropped 180×320 images at inference — a
resolution the encoder was never trained on.

## 5. Merge sessions

One directory per robot session, one directory for the pipeline:

```bash
python scripts/utils/merge_droid_dirs.py \
    /data/rollouts/set_table_jun19 /data/rollouts/set_table_jun20 \
    --output /data/rollouts/set_table_all
```

Carries over labels, notes, videos, embeddings, and the raw
`recordings/` (so embeddings can still be (re-)extracted from the merged
dir later).

## 6. Behavior graphs

The droid embedding layer is `bottleneck_plan_t0` (the features default is
`pi05`, so override it):

```bash
python -m behavior_graphs.pipeline.run \
    rollout_dir=/data/rollouts/set_table_all \
    features.layer=bottleneck_plan_t0

# or a full sweep + model selection (docs/sweep_study.md):
python scripts/run_sweep.py 'datasets={set_table: /data/rollouts/set_table_all}'  # layer auto-detected

# or sweep a whole registered family at once (see configs/data/droid.yaml):
python scripts/run_sweep.py data=droid layers=[bottleneck_plan_t0]

streamlit run apps/demo/Home.py     # Graph Demo → explore / model selection
```

## π0.5 (openpi) on DROID

Status: **training + robot inference are supported by the submodule**;
what is *not* yet wired is per-timestep π0.5 embedding capture on DROID
(the fork ships it for LIBERO via
`examples/libero/eval_save_with_embeddings.py`), so DROID behavior graphs
currently come from the diffusion-policy path. Wiring it means porting the
same embedding hook to the DROID client and writing the canonical layout —
the labeling / merging / pipeline tooling above is policy-agnostic and
applies unchanged.

```bash
git submodule update --init third_party/openpi
cd third_party/openpi && uv sync
```

**Fine-tune on your own DROID task** (LeRobot path — recommended for
task-sized datasets): convert the raw sessions, compute norm stats, train:

```bash
uv run python examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir /data/droid/kendama_demos
# add a TrainConfig for your dataset in src/openpi/training/config.py
uv run python scripts/compute_norm_stats.py --config-name <my_pi05_droid_cfg>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train.py \
    <my_pi05_droid_cfg> --exp-name kendama_pi05 --overwrite
```

**Full-DROID fine-tune** (RLDS path, 1.8 TB dataset, ~2 days on 8×H100):
follow `third_party/openpi/examples/droid/README_train.md` (Python 3.11 +
`uv sync --group rlds`, `pi05_full_droid_finetune` config; prefer joint
*position* actions).

**Robot inference**: serve the checkpoint and run the DROID client on the
workstation (see `third_party/openpi/docs/remote_inference.md`):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config <my_pi05_droid_cfg> --policy.dir <ckpt_dir>   # GPU box
uv run python examples/droid/main.py --remote-host <gpu_box_ip>   # robot ws
```

Episodes recorded this way are raw DROID sessions → label and build graphs
once embedding capture is wired (or run them through
`droid_extract_embeddings.py` with a diffusion checkpoint to analyze them
in that embedding space meanwhile).
