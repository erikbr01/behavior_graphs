# Known issues

Defects and rough edges we know about and have not fixed. Listed here rather
than left for a reader to rediscover; each says what is wrong, how it shows up,
and why it is still open.

## Eval and collection disagree on the same policy

`RgrlAdapter.eval_policy` runs the submodule's evaluator, configured by
`third_party/rgrl/configs/env/robomimic_transport.yaml` — `max_steps 700`,
`parallel_envs 25`, `seed 10000`. `RgrlAdapter.collect_rollouts` runs the
submodule's collector, configured by
`third_party/rgrl/configs/rollout_env/robomimic.yaml` plus
`third_party/rgrl/configs/collect_rollouts.yaml` — `num_envs 1`, `seed 0`,
`max_timesteps 1000`. Measured on one `transport_mh` policy:
eval reports **0.46**, the collected rollouts are **44/50 = 0.88**.

Both are honest under their own settings — a longer horizon and a different
seed stream are a different measurement, not a broken one. But a reader who
sees `success_rate 0.46` and then opens a graph built from 44 successes will
reasonably think something is wrong. Aligning them is a decision about which
harness is canonical, which is why it is still open. **If you compare a success
rate to a graph, check that both came from the same harness.**

## `topreward` is not the paper's TOPReward

Two TOPReward implementations are in the tree:

| config | class | what it is |
|---|---|---|
| `method/topreward_ll.yaml` | `TopRewardLL` | the authors' own code, through `third_party/robometer`. **This is the Tab. II row.** |
| `method/topreward.yaml` | `TopRewardProbe` | our in-house reimplementation (teacher-forced P(yes)) |

They disagree substantially — 0.604 vs 0.811 VOC on `libero_goal` — so running
the wrong one produces a much stronger baseline than the paper reports and
makes the published comparison look wrong. The demo app hides the
reimplementation for this reason (`_HIDDEN_METHODS` in
`apps/demo/_pages/progress_estimation.py`). See
[progress_estimation.md](progress_estimation.md).

## The VLM baselines have not been re-run here

`third_party/robometer` and `third_party/warp_rm` now check out at their
recorded commits, and the robometer pin carries
`robometer/evals/baselines/topreward.py` — the code `TopRewardLL` reaches for.
So the pins are good.

What has not been done is running the two VLM rows of Tab. II end to end from
this repository: that needs a 24 GB GPU, robometer's own venv, and the
Qwen3-VL-8B and ROBOMETER-4B checkpoints. The graph and MLP rows re-derive
bit-exactly (see [progress_estimation.md](progress_estimation.md)); the VLM
rows are taken from the published artifacts.

## Weighted sampling is inconsistent across the submodules

`third_party/diffusion_policy` merged weighted behavior cloning; the equivalent
change in `third_party/rgrl` was closed as conflicting. Which repos support it
is therefore an accident of which pull request happened to be mergeable rather
than a decision. Not used by any paper result.

## Submodules are private

`git clone --recurse-submodules` fails for anyone outside the project while the
`third_party/` repositories stay private. Everything in
[REPRODUCE.md](../REPRODUCE.md) works without them — the submodules are needed
to *train* policies and to run the two VLM baselines, not to build graphs or
check the published numbers from rollouts.
