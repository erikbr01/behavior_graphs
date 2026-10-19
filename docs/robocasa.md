# RoboCasa

Behavior graphs and progress estimation are wanted for RoboCasa, and the
adapters, configs and zoo entry are all here. What is **not** here are the four
RoboCasa submodules: they were dropped from this repository rather than
published, because they are ~95 MB of simulator and checkpoint forks whose
provenance belongs upstream, not with us.

So RoboCasa works the way a plug-in works: supply the checkouts yourself, point
the adapter at them, and everything downstream — graph construction, the sweep,
the app, progress estimation — is unchanged.

## What the adapter expects

`src/behavior_graphs/envs/robocasa365.py` drives three policy stacks, each in
its own checkout under `third_party/` and its own venv:

| `policy_stack` | checkout | runner |
|---|---|---|
| `dp` | `third_party/diffusion_policy` (vendored) + `third_party/robocasa365` | `.venv-robocasa365/bin/python` |
| `gr00t` | `third_party/robocasa_gr00t` | `.venv/bin/python` |
| `pi0` | `third_party/robocasa_openpi` | `uv run --no-sync python` |

Each stack's `submodule` path is read from `_POLICY_STACKS`, so a checkout at
that path is all the adapter needs — it does not care whether it is a git
submodule, a clone, or a symlink.

```bash
cd third_party
git clone <your robocasa365 fork> robocasa365
git clone <your robocasa fork> robocasa          # the 0.5-era simulator
git clone <your robomimic fork> robomimic        # robosuite 1.5 compatible
```

## Building the venvs

The RoboCasa stacks need a *different* interpreter from the main training env:
robosuite master with pinned mujoco and numpy, which the python 3.9 training
stack cannot host. Two scripts in the diffusion_policy submodule build them:

```bash
cd third_party/diffusion_policy

# robocasa365 diffusion-policy eval          -> .venv-robocasa365  (python 3.10)
scripts/setup_env_robocasa365_uv.sh
# or build it elsewhere:
VENV=/big/disk/robocasa365_venv scripts/setup_env_robocasa365_uv.sh

# the 0.5-era robocasa simulator             -> .venv-robocasa     (python 3.11)
scripts/setup_env_robocasa_uv.sh
```

**The venv name matters.** `.venv` inside that submodule is the python 3.9
training stack from `setup_env_uv.sh`; the RoboCasa one is
`.venv-robocasa365`. Pointing the adapter at `.venv` runs the training
interpreter against a simulator it cannot import, and the failure surfaces deep
inside robosuite rather than as a missing environment. If you build the venv
somewhere else, name it:

```bash
export BEHAVIOR_GRAPHS_ROBOCASA365_PYTHON=/big/disk/robocasa365_venv/bin/python
```

## Collecting rollouts and building a graph

Once a stack is in place, RoboCasa is an environment like any other:

```bash
export DATA_PATH=/path/to/data MUJOCO_GL=egl
python scripts/collect_rollouts.py env=robocasa365_dp \
    checkpoint=/path/to/robocasa365_checkpoint \
    num_episodes=50

python -m behavior_graphs.pipeline.run \
    rollout_dir=$DATA_PATH/rollouts/robocasa365_dp_pickplace
```

The embedding layer resolves itself — RoboCasa's tap is `obs_encoder`, where
the sim families use `bottleneck_plan_t0` and LIBERO uses `pi05`. Verified on
`robocasa365_dp_pickplace`: 10 behaviors, 26 transitions, from the bare command
above.

## Progress estimation

`robocasa365_dp_pickplace` is in the zoo under `other_sim`, and deliberately
outside the 43 collections the paper's Tab. II reports — it reads a
mean-pooled representation rather than a policy trained on the task, which is a
different thing to compare. It scores like any other collection:

```bash
PYTHONPATH=src python scripts/run_progress_bench.py \
    data.name=robocasa365_dp_pickplace
```

See [progress_estimation.md](progress_estimation.md) for the method set and
[data.md](data.md) for the rollout layout.
