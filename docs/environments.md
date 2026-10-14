# Environments

All policy learning lives in `third_party/` submodules. The main repo's
adapters (`src/behavior_graphs/envs/`) expose each environment as
`TrainingData` / `Policy` / `RolloutSet`.

| Env | Policy stack | Submodule(s) | Rollouts |
|---|---|---|---|
| robomimic | diffusion policy | `diffusion_policy` (stock robomimic 0.3.0) | simulated, lowdim + image |
| mimicgen | diffusion policy | `diffusion_policy` (+`mimicgen`) | simulated |
| droid | diffusion policy | `diffusion_policy` (+`robo_dataset`) | real robot; recorded sessions |
| libero | pi0.5 (openpi) | `openpi` (+ its LIBERO) | simulated |
| libero-pro | pi0.5 (openpi) | `openpi` + `libero_pro` | simulated, perturbed suites |
| vla_foundry | TRI LBM policies (Foundry-VLA/Qwen3VLA) | `vla_foundry` | simulated (Drake in Docker), served model |

## Environment setup (uv, no conda)

```bash
# diffusion-policy stack (robomimic / mimicgen / droid)
# installs stock robomimic==0.3.0, exactly as policy_doctor's training env;
# the third_party/robomimic fork is only used by the robocasa venv (0.5-era,
# robosuite 1.5 compatible)
cd third_party/diffusion_policy && scripts/setup_env_uv.sh

# openpi stack (libero / libero-pro)
cd third_party/openpi
uv sync
uv pip install -e third_party/libero --config-settings editable_mode=compat
uv pip install "robosuite==1.4.1" "bddl==1.0.1" easydict future "gym==0.25.2" \
    "thop==0.1.1.post2209072238"
# libero-pro (replaces the libero package; perturbed suites + assets)
uv pip install -e ../libero_pro --config-settings editable_mode=compat

# robocasa needs robosuite master (PandaOmron) — a separate venv;
# pass it via ++env.python_path=<venv>/bin/python

# vla_foundry stack (LBM eval: gRPC policy server + Drake sim in Docker)
cd third_party/vla_foundry
uv sync --group inference && uv pip install -e .
docker pull toyotaresearch/lbm-eval-oss:vla-foundry   # ~18 GB
hf download TRI-ML/Foundry-VLA-1.7B-sim               # ~7.4 GB
```

`MUJOCO_GL=egl` for headless sim rendering. Set `UV_CACHE_DIR` to a large
disk.

## Workflow per env

```bash
# train (delegates to the submodule's config system via train_overrides)
python scripts/train_policy.py env=robomimic_lift_lowdim env.dataset_path=... \
    '++train_overrides={training: {num_epochs: 500, compile: true, tf32: true}}'

# collect rollouts with per-timestep policy embeddings
python scripts/collect_rollouts.py env=robomimic_lift_lowdim env.dataset_path=... \
    checkpoint="..." num_episodes=50 '++rollout_kwargs={n_envs: 10}'

# behavior graph
python -m behavior_graphs.pipeline.run rollout_dir=<...>
```

Env-specific notes:

- **robomimic lowdim** evals must run with `n_*_vis=0` (the lowdim runner has
  no offscreen renderer); image evals record a video for every episode and
  run sequentially (`n_envs=1`, enforced by the eval script).
- **robomimic image** configs expect absolute actions; convert datasets with
  `diffusion_policy/scripts/robomimic_dataset_conversion.py` into a path
  containing `robomimic/<task>/<type>/` (the dataset code parses the path).
- **droid**: training and offline embedding extraction work anywhere;
  real rollouts are collected on the robot workstation with
  `third_party/diffusion_policy/scripts/droid_collect_rollouts.py`, and
  recorded sessions get embeddings via
  `scripts/droid_extract_embeddings.py --store <VideoStore|hdf5|raw sessions>`.
  Full real-world walkthrough — configs, data conversion, training,
  inference, outcome labeling, session merging, π0.5 fine-tuning:
  `docs/droid_workflow.md`.
- **libero**: the pretrained `gs://openpi-assets/checkpoints/pi05_libero`
  checkpoint is the default policy (`train_policy()` returns it without
  training; `pretrained=False` fine-tunes via openpi).
- **vla_foundry**: the fork's `vla_foundry/eval/run_eval_save_rollouts.py`
  owns the whole eval (embedding-capturing gRPC policy server + the
  `toyotaresearch/lbm-eval-oss` Drake simulator in Docker) and writes the
  canonical layout itself. Embeddings: the vision-language backbone's
  action-conditioning output, one row per `generate_actions` call (layer
  `action_conditioning`; episodes align to embedding segments via the
  per-episode gRPC reset seed). Pick the task via `env.task` (LBM names,
  e.g. `PutMugOnSaucer`); `seed=N` shifts the scenario-index range.
  LBM sim training data is public (fork: data/scripts/download_dataset.py);
  local DP training per docs/guides/lbm-pipeline.md is not wired into the
  adapter yet — or use the released
  `TRI-ML/Foundry-*` checkpoints.
- **libero-pro**: select a perturbed suite via `env.task_suite` +
  `env.perturbation` (object|swap|lan|task|env|temp), or pass a full suite
  name (e.g. `task_suite=libero_spatial_with_mug perturbation=null` for the
  object-replacement suites whose assets ship with the repo). The
  `_env`/`_swap`/... composite suites need the bddl/init assets from
  https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro.

## Training speed optimizations (carried over from policy_doctor)

All in the `diffusion_policy` submodule and reachable via `train_overrides`:

- `training.compile=true` — `torch.compile` of the policy
- `training.tf32=true` — TensorFloat-32 matmuls/cudnn
- `dataloader.num_workers/persistent_workers/pin_memory`
- robo_dataset `VideoStore` training data (`dataset/robo_store_dataset.py`):
  H.264 per-camera MP4s + lz4 zarr for lowdim — fast random access
- multi-env parallel evaluation (`AsyncVectorEnv`, `n_envs` in the runner
  config / `--n-envs` in `scripts/eval_save_rollouts.py`)
