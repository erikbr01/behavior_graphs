# Environments

Every environment is an **adapter** implementing `TrainingData` / `Policy` /
`RolloutSet` (`core/env.py`), a **config** under `configs/env/`, and a
**submodule venv** that owns the simulator and the policy stack. The adapter
shells out to the submodule and translates whatever it writes into the one
[canonical rollout layout](data.md#the-canonical-rollout-layout). Nothing
downstream — graph construction, sweeps, the apps, progress estimation — knows
which environment it is looking at.

This page is the reference for all three parts. For *what to run* see
[experiments.md](experiments.md).

## The configs

`env=<name>` selects one. Every config leaves `dataset_path: ???` mandatory
where it needs demonstrations, so pass `env.dataset_path=...`.

| `env=` | adapter | embedding layer | venv |
|---|---|---|---|
| `robomimic_lift_lowdim` | `RobomimicAdapter` | `bottleneck_plan_t0` | diffusion_policy |
| `robomimic_square_lowdim` | " | " | " |
| `robomimic_can_lowdim` | " | " | " |
| `robomimic_transport_lowdim` | " | " | " |
| `robomimic_tool_hang_lowdim` | " | " | " |
| `robomimic_lift_image` | " | " | " |
| `mimicgen`, `mimicgen_square` | `MimicgenAdapter` | `bottleneck_plan_t0` | diffusion_policy |
| `droid` | `DroidAdapter` | `bottleneck_plan_t0` | diffusion_policy |
| `rgrl_lift`, `rgrl_transport` | `RgrlAdapter` | `enc` / `cond_proj` | rgrl |
| `libero` | `LiberoAdapter` | `pi05` | openpi |
| `libero_pro` | `LiberoProAdapter` | `pi05` | openpi + supplied checkout |
| `vla_foundry` | `VlaFoundryAdapter` | `action_conditioning` | vla_foundry + Docker |
| `robocasa_atomic` | `RobocasaAdapter` | `bottleneck_plan_t0` | supplied venv |
| `robocasa365_{dp,gr00t,pi0}` | `Robocasa365Adapter` | `obs_encoder` | supplied checkouts |

The embedding layer is the activation tap the graph is built on, and it is
**resolved from the collection** rather than assumed: `features.layer: null`
asks the rollout directory which tap it carries. Only pass `features.layer=` to
disambiguate a collection holding more than one.

## Why these are separate venvs

They pin conflicting versions of the same libraries, and not by accident:

| stack | Python | the pin that forces it |
|---|---|---|
| diffusion_policy | 3.9 | `gym==0.21`, which needs `setuptools<66` |
| robocasa365 | 3.10 | robosuite master, `mujoco==3.3.1`, `numpy==2.2.5` (asserted) |
| robocasa (0.5-era) | 3.11 | robosuite 1.5 |
| robometer | 3.10 | torch 2.8 against a transformers that wants newer |
| rgrl, openpi, vla_foundry | own | uv projects with their own locks |

This repository itself needs ≥3.10 and is tested on 3.13. There is no single
environment in which all of this installs, which is why the submodules are not
extras of this package.

## Building them

```bash
# robomimic / mimicgen / DROID                                 (python 3.9)
git submodule update --init third_party/{diffusion_policy,mimicgen,robo_dataset}
cd third_party/diffusion_policy && scripts/setup_env_uv.sh     # -> .venv

# LIBERO via pi0.5
git submodule update --init third_party/{openpi,libero}
cd third_party/openpi && uv sync
uv pip install -e ../libero --config-settings editable_mode=compat
uv pip install "robosuite==1.4.1" "bddl==1.0.1" easydict future "gym==0.25.2" \
    "thop==0.1.1.post2209072238"

# rgrl flow-matching policies
git submodule update --init third_party/rgrl
cd third_party/rgrl && uv sync --extra video        # --extra eval to score in sim

# the two VLM progress baselines                               (python 3.10)
git submodule update --init third_party/robometer
cd third_party/robometer && uv sync
uv pip uninstall torchao     # otherwise `import transformers` fails: see below

# LBM eval through vla_foundry                      (needs Docker, ~25 GB)
git submodule update --init third_party/vla_foundry
cd third_party/vla_foundry && uv sync --group inference && uv pip install -e .
docker pull toyotaresearch/lbm-eval-oss:vla-foundry
hf download TRI-ML/Foundry-VLA-1.7B-sim
```

`MUJOCO_GL=egl` for headless rendering. Point `UV_CACHE_DIR` and `HF_HOME` at a
disk with room — the two VLM checkpoints alone are ~25 GB.

The setup scripts reach for **sibling** submodules (`../mimicgen`,
`../robo_dataset`, `../robomimic`), so initialise those alongside rather than
afterwards; `setup_env_uv.sh` aborts with a message naming the one it cannot
find.

### Two that bite

**robometer's own `uv sync` produces an environment that cannot import its
baseline code.** transformers pulls torchao 0.18, which needs torch ≥ 2.11
against robometer's pinned 2.8, and `import transformers` dies with
`ImportError: cannot import name 'ScalingType'`. Nothing here uses the
quantizer it is wanted for. There is a second defect in the same path — see
[progress_estimation.md](progress_estimation.md).

**`.venv` is not one venv.** Inside `third_party/diffusion_policy`, `.venv` is
the python 3.9 training stack; `.venv-robocasa365` and `.venv-robocasa` are
separate interpreters built by separate scripts. Pointing an adapter at the
wrong one fails deep inside robosuite rather than as a missing environment.

## Environments whose checkouts are not vendored

Three adapters ship here with their submodules dropped from this repository.
They work as plug-ins: supply the checkout, point the adapter at it.

- **RoboCasa** (`robocasa_atomic`, `robocasa365_*`) — see
  [robocasa.md](robocasa.md).
- **LIBERO-Pro** (`libero_pro`) — installs as a package into openpi's venv
  (`uv pip install -e <your libero_pro> --config-settings editable_mode=compat`),
  replacing the `libero` package with the perturbed suites and their assets.

Each adapter reads its submodule path from a table, so a clone or a symlink at
that path works as well as a submodule.

## Training throughput

All in the `diffusion_policy` submodule, reachable through `train_overrides`:

- `training.compile=true` — `torch.compile` of the policy
- `training.tf32=true` — TensorFloat-32 matmuls and cudnn
- `dataloader.num_workers` / `persistent_workers` / `pin_memory`
- `robo_dataset`'s `VideoStore` training data
  (`dataset/robo_store_dataset.py`): H.264 per-camera MP4s plus lz4 zarr for
  low-dim, for fast random access
- parallel evaluation (`AsyncVectorEnv`; `n_envs` in the runner config, or
  `--n-envs` for `scripts/eval_save_rollouts.py`)

A note on `tf32`: it changes results, not only speed. The clustering stage pins
its thread pool for the same reason — see
[progress_estimation.md](progress_estimation.md).

## LIBERO task suites

`task_suite=` selects a suite by name (`task_suite=libero_spatial_with_mug
perturbation=null` for the object-replacement suites, whose assets ship with
the LIBERO checkout). The `_env` / `_swap` composite suites need the bddl and
init assets from
<https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro>.

## Adding one

1. `src/behavior_graphs/envs/<name>.py` implementing the three interfaces, and
   an entry in `envs/__init__.py` (`_ADAPTERS` **and** the `TYPE_CHECKING`
   block — static resolution breaks if you add only the first).
2. `configs/env/<name>.yaml` with `_target_` and the tap name. `tests/
   test_configs.py` resolves every `_target_` to an `EnvAdapter` subclass, so a
   typo fails there rather than in someone's first command.
3. Write rollouts in the canonical layout, or translate at the end of
   `collect_rollouts` the way `rgrl_env.py` does with its assembler.

See [architecture.md](architecture.md) for the interfaces themselves.
