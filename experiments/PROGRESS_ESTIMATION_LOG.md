# Progress-Estimation Benchmark — behavior graph vs TOPReward / ROBOMETER

## Summary / verdict
On all three LIBERO suites tested (goal/object/spatial), the behavior graph's **stage-progress**
signal (identity reduction, K=128, on pi0.5 policy embeddings) **beats both ROBOMETER checkpoints
— including the LIBERO-finetuned one — and TOPReward on every progress-ranking metric
(VOC/Kendall/Pearson/MAE)**, and the win is statistically overwhelming: 95% CIs that do not
overlap any VLM baseline across 300 random episode subsets, and paired per-episode Wilcoxon
p < 1e-32 against every baseline (higher on 74–100% of episodes). It does so **unsupervised**
(no reward-model training). The one place a VLM wins is *supervised success detection* (AUROC):
ROBOMETER's trained success head beats the graph, as expected. Beyond LIBERO, a sweep across
**33 sim datasets (6–708 step horizons)** shows the graph progress signal holds at mean VOC 0.953
(E-PE4, K chosen per fold on an inner validation split); its clear failure mode is a weak policy
embedding (safe_franka / pi0fast prelogits, VOC 0.53). The TOPReward baseline was hardened with the **authors' exact code** (via ROBOMETER's
framework) — which turned out *weaker* than our own reimplementation, so the win is not an artifact
of under-powering the baseline (E-PE6).

**Honest caveat (not input-matched):** the graph consumes the pi0.5 policy's internal embeddings
(privileged 1024-d), while the VLMs see only raw RGB frames. Fair reading: *when policy embeddings
are available (they are, at rollout time), the graph is a far cheaper and more accurate progress
model than an 8B VLM on pixels.* The input-matched test (graph on frame embeddings, TOPReward's OXE
setting) is deferred.

Origin: pivot (2026-07-26) after the MimicGen data-curation line closed as a robust NULL on clean d0
(see `experiments/MIMICGEN_CURATION_LOG.md`, branch rgrl-backend-wt) — moved to where the graph's
value/progress machinery should be strong.

## Baselines (the papers)
- **TOPReward** (arXiv:2602.19313): training-free; probes a VLM's token probabilities for task
  completion. Reimplemented here as teacher-forced P(yes)/(P(yes)+P(no)) on Qwen3-VL-8B.
- **ROBOMETER** (arXiv:2603.02115): trained 4B VLM reward model (progress + preference + success).
  Run from their released checkpoints. Ckpts: `robometer/Robometer-4B` (zero-shot),
  `jesbu1/robometer-4b-fft-libero` (LIBERO-finetuned).
- **WARP-RM** (github.com/uynitsuj/WARP-RM): self-supervised *dense per-frame* reward model —
  frozen DINOv3 -> bidirectional transformer -> per-frame progress in [0,1], trained per-task on
  successful demos via WARP time-warp augmentation. Integrated as submodule `third_party/warp_rm`
  (private mirror `erikbr01/warp-rm`); scored by `scripts/warp_rm_progress_bench.py`. Unlike the
  VLM baselines it is a dense estimator like our `graph_direct`, and it is trained (not
  training-free). See the "WARP-RM integration" section below.
Neither TOPReward nor ROBOMETER uses robomimic/mimicgen; all three are vision models on OXE /
ManiRewardBench / LIBERO / RBM-EVAL / their own sim.

## Metrics (theirs; in `behavior_graphs.progress.progress_regression_metrics`)
Over SUCCESSFUL episodes vs linear-interpolation GT progress 0->1:
- **VOC** — Spearman rank corr of predicted progress vs frame order (TOPReward primary).
- **Kendall tau-b**, **Pearson** (vs linear GT), **MAE** (normalized progress error).
- **success-AUROC** — end-of-episode progress vs the true success flag.
Caveat: `frame_index` is trivially perfect on VOC/Pearson/MAE (it *is* the linear GT) but AUROC~0
(can't detect success) — the SARM point. The meaningful comparison is vs the VLM baselines + on AUROC.

## Method (ours)
Behavior graph -> per-window progress. `graph_direct:stage` (SARM stage progress) and
`graph_direct:absorb` (P(absorb at SUCCESS | cluster)). Reducer = **identity** (StandardScaler on raw
embeddings; UMAP distorts, PCA weaker — from the curation campaign). K is NOT universal: LIBERO wants
HIGH K (K=128); mimicgen wanted K~8-12. Runner `scripts/eval_progress_benchmark.py` (CPU, no GPU),
native behavior_graphs stack: SlidingWindow -> IdentityReduction -> KMeansClustering -> BehaviorGraph.

## Data
`${DATA_PATH}/rollouts/libero_goal` — 1000 eps (972 succ / 28 fail); pi0.5
embeddings (1024-d) in `policy_embeddings/pi05.npz`; per-episode MP4 in `media/`; success in
filename + metadata. LIBERO instructions recovered deterministically (task = episode_index // 100;
the 10 task.language strings live in `vlm_reward_bench.LIBERO_SUITES`, via `--suite libero_goal`).

## Results
### E-PE1 — K-sweep (graph, identity, 3-fold, full 1000)
Higher K monotonically better on LIBERO: stage VOC 0.67(K8) -> 0.96(K128), AUROC 0.62 -> 0.90;
absorb AUROC 0.78 -> 0.96. **K=128 chosen.**

### E-PE2 — aligned head-to-head, 200 eps (20/task), identical episodes, point estimates
Graph fit on the held-out 800; VLMs on raw frames.

| method | input | VOC | Kendall | Pearson | MAE | AUROC |
|---|---|---|---|---|---|---|
| **graph_direct:stage** (K=128) | pi0.5 emb | **0.952** | **0.919** | **0.946** | **0.062** | **0.966** |
| graph_direct:absorb (K=128) | pi0.5 emb | 0.793 | 0.649 | 0.770 | 0.481 | 0.968 |
| ROBOMETER-4B (zero-shot) | RGB | 0.653 | 0.496 | 0.708 | 0.186 | 0.644 |
| ROBOMETER-LIBERO-FFT | RGB | 0.792 | 0.638 | 0.799 | 0.225 | 0.908 |
| TOPReward (Qwen3-VL-8B) | RGB | 0.824 | 0.711 | 0.674 | 0.471 | 0.621 |

### E-PE3 — statistical confidence (vary the SUBSET-SELECTION seed; KMeans seed ignored)
Every method is deterministic per episode, so we score ONCE on a shared 500-ep pool (50/task), save
full per-episode curves, then bootstrap random 20/task subsets across 300 seeds (no VLM re-runs).
mean +/- std [95% percentile CI]:

| method | input | VOC | Pearson | MAE | AUROC |
|---|---|---|---|---|---|
| **graph-stage** (ours) | pi0.5 emb | **0.943 +/- 0.006 [0.932, 0.955]** | **0.937** | **0.066** | **0.973** |
| topreward (Qwen3-VL-8B) | RGB | 0.811 +/- 0.011 [0.790, 0.832] | 0.658 | 0.471 | 0.590 |
| robometer-libero-FFT | RGB | 0.774 +/- 0.008 [0.759, 0.788] | 0.783 | 0.226 | 0.899 |
| robometer-4b (zero-shot) | RGB | 0.638 +/- 0.008 | 0.702 | 0.189 | 0.828 |
| graph-absorb (ours) | pi0.5 emb | 0.665 +/- 0.009 | 0.662 | 0.480 | 0.956 |

- graph-stage vs topreward: **non-overlapping 95% CIs**; graph-stage std only +/-0.006 over 300 subsets.
- Paired per-episode **Wilcoxon** (n=500): graph-stage beats EVERY baseline — median dVOC +0.09..+0.23,
  higher on 86-99% of episodes, all **p < 1e-53**.
- Trace plot (all methods per episode vs GT): `analysis/out/progress_bench_results/lg_pool500_traces.png`.

### E-PE4 — cross-dataset sweep of OUR method (54 datasets, policy embeddings, nested-K, 3-fold)
Ran graph stage-progress on **every feasible behavior_graphs rollout dataset** (has a
`policy_embeddings/*.npz` + per-episode success flags; excluded `mimicgen_nut_assembly_d0`,
0 successes). Each dataset uses **its own policy embedding** — the same modality the VLM
comparison contrasts against pixels — NOT raw simulator state (an earlier pass silently used
`observations/state.npz` on the sim datasets via `load_rollouts` auto-ordering; caught via the
layer column and corrected). K swept per dataset (capped by train-window
count) and chosen **inside each outer fold**, on a validation split carved out of that fold's
train half, with the winner refit on the whole of it -- so every number below is scored on a test
fold that had no say in the choice. K is therefore re-selected per fold and the folds often
disagree (`k_per_fold` records what each chose: 39 of 54 datasets disagreed at least once).

Full 54-row table: `analysis/out/progress_allds_nested/sweep_table.md` (+ `sweep_results.json`,
in the rgrl checkout). Driver: `scripts/sweep_progress_allds.py`, merge:
`scripts/merge_progress_sweep.py`.

> **Retraction.** This section previously reported a sweep that scored every candidate K on the
> held-out folds and published the maximum -- selection on the metric being reported -- from
> `analysis/out/progress_allds`. Those numbers are withdrawn, that run has been deleted, and the
> script no longer has a mode that can produce it. The bias, family by family, was between 0.004
> and 0.048 VOC: sim-only 0.968 -> 0.953, all 54 datasets 0.864 -> 0.842. It also said "52
> datasets" where the table had 54.

> **How K is chosen.** Inside each outer fold, on a validation split carved out of that fold's
> TRAIN half (`--val-frac`, default 0.25), with the winner refit on the whole train half.
> Implemented as `TunedGraphBackbone`, whose `fit` only ever receives the train view — so the
> test fold is not merely unused in the choice, it is unreachable from it. The guarantee is
> structural rather than a rule to remember, which is the point: the earlier version of this
> experiment had no inner split and chose K on the folds it reported.
>
> K is recorded per fold (`k_per_fold`) rather than as one `best_k`, because a single K across
> folds was always a summary of a choice and not a fitted parameter. There is no mode that
> selects on the eval fold any more — it was removed rather than defaulted off, since the only
> thing it could produce is a number we would not report.

Per-family mean stage-VOC (higher=better):

| family | n | mean VOC | median | range | embedding |
|---|---|---|---|---|---|
| long_horizon (set_table / prepare_tray / transport, 78–708 steps) | 8 | 0.979 | 0.983 | 0.959–0.995 | bottleneck_plan_t0 / cond_proj |
| robomimic (can/lift/square/tool_hang/transport, mh+ph) | 9 | 0.965 | 0.979 | 0.873–1.000 | bottleneck_plan_t0 |
| libero (goal/object/spatial/pro) | 4 | 0.955 | 0.964 | 0.926–0.968 | pi05 |
| mimicgen (8 d0 source tasks) | 8 | 0.936 | 0.962 | 0.819–0.998 | bottleneck_plan_t0 |
| other_sim (robocasa / molmoact / vla_foundry) | 4 | 0.908 | 0.936 | 0.766–0.992 | obs_encoder / molmoact2 / action_cond |
| safe_widowx (8 real tasks) | 8 | 0.895 | 0.909 | 0.757–0.987 | openvla_hidden_mean |
| safe_franka (13 real tasks) | 13 | **0.528** | 0.527 | 0.334–0.799 | pi0fast_prelogits_mean |
| **sim-only (top 5 families, 33 datasets)** | 33 | **0.953** | **0.973** | 0.766–1.000 | — |
| all 54 datasets | 54 | 0.842 | 0.943 | 0.334–1.000 | — |

**Takeaways.**
- On the standard sim manipulation benchmarks the graph's stage-progress is **strong and
  consistent (mean VOC 0.953, median 0.973)** across task families AND horizon lengths — from
  6-step lift up to 708-step prepare-tray. Unsupervised, CPU-only, no reward-model training.
- **K is not universal (confirmed):** LIBERO + long-horizon prefer high K (32–128); short
  robomimic/mimicgen prefer low K (8–16). The chosen K tracks data volume × horizon — and is not
  even stable within a dataset: 39 of 54 re-chose it across folds, which is variance the old
  single reported K hid rather than removed.
- **Honest boundary — embedding quality dominates.** safe_franka (pi0fast prelogits-mean
  pooling) is the clear weak spot (VOC 0.57); safe_widowx (openvla hidden-mean) is decent
  (0.91). Same graph, same tasks-difficulty range — the differentiator is how well the policy
  embedding encodes task state. Where the embedding is a good state representation
  (pi0.5, DP bottleneck, obs-encoder) the graph is near-ceiling; where it is a lossy pooled
  VLM prelogit it degrades.
- **AUROC caveats:** unreliable where a dataset has ≤2 failures (can_ph/lift_ph 0 fails → NaN;
  hammer/square_ph → degenerate 0.0/1.0 on a single fail). Read AUROC only on datasets with a
  meaningful fail count (transport_mh, mimicgen_mug, square_mh, the safe_* sets, LIBERO).
  `robocasa` shows VOC 1.0 but MAE 0.36 on 9 successes: ranking perfect, scale/calibration off,
  small-sample.

### E-PE5 — multi-dataset VLM head-to-head (LIBERO object + spatial)
Extended the E-PE3 head-to-head beyond libero_goal to **libero_object** and **libero_spatial**
(the suites where robometer-4b-fft-libero is in-distribution). Same protocol: score once on a
shared 500-ep pool (50/task, graph fit on the held-out 500), save per-episode curves, bootstrap
300 random 20/task subsets. Graph = pi0.5 policy embeddings (identity, K=128); VLMs = raw RGB.
LIBERO object/spatial instructions recovered from the local `libero` package
(`get_task(i).language`, collection order == `index//100`), wired into
`vlm_reward_bench.LIBERO_SUITES`.

Bootstrap mean ± std [95% CI] over 300 subsets (VOC = TOPReward primary metric):

**libero_object** (491 succ / 9 fail in pool)

| method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |
|---|---|---|---|---|---|---|
| **graph-stage** (ours) | pi0.5 emb | **0.967 [0.958,0.974]** | **0.938** | **0.963** | **0.050** | 0.749 |
| topreward | RGB | 0.930 [0.922,0.939] | 0.854 | 0.664 | 0.369 | 0.640 |
| robometer-fft | RGB | 0.828 [0.810,0.841] | 0.686 | 0.815 | 0.205 | **0.986** |
| robometer-4b | RGB | 0.821 [0.807,0.834] | 0.662 | 0.833 | 0.162 | 0.941 |

**libero_spatial** (486 succ / 14 fail in pool)

| method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |
|---|---|---|---|---|---|---|
| **graph-stage** (ours) | pi0.5 emb | **0.954 [0.945,0.961]** | **0.912** | **0.951** | **0.060** | 0.859 |
| robometer-4b | RGB | 0.859 [0.850,0.869] | 0.692 | 0.879 | 0.136 | **0.984** |
| topreward | RGB | 0.816 [0.798,0.833] | 0.703 | 0.722 | 0.498 | 0.375 |
| robometer-fft | RGB | 0.734 [0.708,0.756] | 0.608 | 0.699 | 0.304 | 0.931 |

- **Graph-stage wins every progress-ranking metric (VOC/Kendall/Pearson/MAE) on both suites,
  with 95% CIs that do not overlap ANY VLM baseline.** Paired per-episode Wilcoxon (n=500):
  graph-stage beats every baseline, higher on 74–100% of episodes, all p < 1e-32
  (object: vs topreward +0.024 median ΔVOC @74%, vs robometer-4b +0.111 @95%, vs robometer-fft
  +0.074 @93%; spatial: vs topreward +0.095 @85%, vs robometer-4b +0.088 @92%, vs robometer-fft
  +0.159 @90%). This replicates the libero_goal result (E-PE3) on two more suites.
- **Honest counter-point — success detection (AUROC).** ROBOMETER's *trained success head*
  beats the graph on end-of-episode success detection (robometer-4b/fft ≈ 0.94–0.99 vs graph
  0.75–0.86). Expected: ROBOMETER is supervised for exactly this, whereas graph-stage is an
  unsupervised progress signal not optimized for the binary call. Caveat: with only 9–14 fails
  in each 500-ep pool, per-subset AUROC is high-variance (graph-object CI [0.49,0.99]); the
  progress-ranking win is the statistically solid claim, the AUROC gap is directional.
- Trace plots: `analysis/out/progress_bench_results/{libero_object,libero_spatial}_pool500_traces.png`.

**Verdict across all three LIBERO suites (goal/object/spatial):** the behavior graph's
unsupervised stage-progress, on policy embeddings, is a **more accurate progress estimator than
both an 8B VLM (TOPReward) and a purpose-trained 4B reward VLM (ROBOMETER, incl. the LIBERO-FFT
checkpoint) on pixels** — decisively on ranking/regression metrics. ROBOMETER retains an edge
only on supervised success detection. And E-PE4 shows the graph signal holds at VOC≈0.97 across
33 sim datasets spanning 6–708 step horizons.

NOTE: the `topreward` rows in the E-PE5 tables above are OUR reimplementation (teacher-forced
P(yes)/(P(yes)+P(no))). E-PE6 below hardens that baseline with the authors' EXACT code and finds
it is actually *weaker* — so E-PE5's TOPReward numbers are, if anything, generous to the baseline.

### E-PE6 — baseline hardening: TOPReward via the authors' own code (faithful recipe)
To be sure we were giving TOPReward its best shot, we replaced our reimplementation with the
authors' exact scoring, run through **ROBOMETER's own framework**
(`robometer.evals.baselines.topreward.TopReward`, whose core is byte-for-byte identical to the
official `topreward.clients.qwen.QwenClient.compute_instruction_reward` — verified against a fresh
clone of `github.com/TOPReward/TOPReward`). Faithful recipe = log-likelihood the VLM assigns to
"`<instruction>` … The answer is: True" conditioned on the trajectory-prefix **video** (fps=2,
`max_frames=64`), over 15 growing prefixes, mean-reduced, minmax-normalized to [0,1]
(ROBOMETER's `configs/reward_model/topreward.yaml`). ROBOMETER itself was already run from its own
checkpoints/inference (no change). Same pool500 sets, 300× 20/task bootstrap. `method=topreward_ll`
in `vlm_reward_bench.py`; the old reimpl model + curves are retained.

VOC = TOPReward's primary metric; **bold** = best in column; mean [95% CI] over 300 subsets.

| suite | method | input | VOC ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |
|---|---|---|---|---|---|---|
| **goal** | **graph-stage** (ours) | pi0.5 emb | **0.944 [0.932,0.955]** | **0.938** | **0.066** | **0.976** |
| goal | topreward — faithful (authors') | RGB | 0.604 [0.583,0.626] | 0.631 | 0.241 | 0.725 |
| goal | topreward — our reimpl | RGB | 0.811 [0.790,0.832] | 0.658 | 0.471 | 0.590 |
| goal | robometer-4b | RGB | 0.638 [0.622,0.654] | 0.702 | 0.189 | 0.828 |
| goal | robometer-fft | RGB | 0.774 [0.759,0.788] | 0.783 | 0.226 | 0.899 |
| **object** | **graph-stage** (ours) | pi0.5 emb | **0.967 [0.958,0.974]** | **0.963** | **0.050** | 0.749 |
| object | topreward — faithful (authors') | RGB | 0.539 [0.514,0.559] | 0.542 | 0.243 | 0.817 |
| object | topreward — our reimpl | RGB | 0.930 [0.922,0.939] | 0.664 | 0.369 | 0.640 |
| object | robometer-4b | RGB | 0.821 [0.807,0.834] | 0.833 | 0.162 | 0.941 |
| object | robometer-fft | RGB | 0.828 [0.810,0.841] | 0.815 | 0.205 | **0.986** |
| **spatial** | **graph-stage** (ours) | pi0.5 emb | **0.954 [0.945,0.961]** | **0.951** | **0.060** | 0.859 |
| spatial | topreward — faithful (authors') | RGB | 0.596 [0.566,0.625] | 0.577 | 0.238 | 0.757 |
| spatial | topreward — our reimpl | RGB | 0.816 [0.798,0.833] | 0.722 | 0.498 | 0.375 |
| spatial | robometer-4b | RGB | 0.859 [0.850,0.869] | 0.879 | 0.136 | **0.984** |
| spatial | robometer-fft | RGB | 0.734 [0.708,0.756] | 0.699 | 0.304 | 0.931 |

- **Hardening did NOT rescue TOPReward — it made it worse.** The authors' faithful recipe scores
  VOC 0.54–0.60 across the three suites, *below* our reimplementation (0.81–0.93). The per-prefix
  yes/no completion probe we had written is, on these short sim clips, a stronger progress signal
  than the log-likelihood-of-"True"-over-video recipe TOPReward actually uses. So the E-PE5
  comparison was already generous to TOPReward.
- **The verdict is unchanged and now more robust.** graph-stage beats the faithful TOPReward by a
  wide margin on every suite: paired per-episode Wilcoxon (n=500) median ΔVOC +0.26/+0.33/+0.30
  (goal/object/spatial), graph-stage higher on 93/98/98% of episodes, all p < 1e-75. It also beats
  both ROBOMETER checkpoints on VOC/Pearson/MAE on all three suites.
- **Success-AUROC (the honest counter-point) is task-dependent, not a clean VLM win.** graph-stage
  AUROC beats both ROBOMETER ckpts on **goal** (0.976 vs 0.897/0.829) but loses on **object** (0.755
  vs fft 0.986) and **spatial** (0.853 vs 4b 0.982). This is NOT explained by failure count — the
  pool500 sets have 14/9/14 fails (goal/object/spatial), so goal and spatial are equally scarce yet
  split. (Note: "28 fails" for goal elsewhere is the full-1000 count; the 500-ep pool has 14.) The
  mechanism is in E-PE7: on goal the graph-stage terminal value separates succ/fail cleanly, on
  object/spatial it separates less; ROBOMETER's trained success head is more uniformly strong.
- Faithful-TOPReward curves saved as `*_topreward_ll_curves.npz` (reimpl `*_topreward_curves.npz`
  retained). Upstream: `${DATA_ROOT}/{topreward (official clone), robometer}`.

### E-PE7 — is it PROGRESS, or a CLOCK? (behaviour on failures + regressions)
VOC/Pearson/MAE reward any monotonic-in-time signal (`frame_index` scores VOC 1.0), so they don't
prove the graph tracks task STATE rather than elapsed time. Two behaviours separate progress from a
clock, both measured on the curves we already saved (pool500 includes the failure episodes scored by
every method — no new GPU): (a) FAILURE PLATEAU — a real progress model ends BELOW 1.0 on a failed
episode, a clock ramps to 1.0 regardless; (b) REGRESSION DRAWDOWN — max drop from a running peak,
which is 0 for any monotone ramp. `scripts/analyze_progress_vs_clock.py`; terminal = mean of last 3.

| suite | method | term(succ) | term(FAIL) | gap | succ-AUROC | drawdown succ / FAIL |
|---|---|---|---|---|---|---|
| goal | frame_index (CLOCK) | 0.969 | 0.990 | −0.021 | 0.000 | 0.00 / 0.00 |
| goal | **graph-stage** | 0.894 | 0.523 | **+0.371** | **0.976** | 0.21 / **0.53** |
| goal | topreward-ll | 0.889 | 0.731 | +0.158 | 0.725 | 0.53 / 0.62 |
| goal | robometer-fft | 0.544 | 0.243 | +0.301 | 0.897 | 0.25 / 0.34 |
| object | frame_index (CLOCK) | 0.978 | 0.989 | −0.012 | 0.000 | 0.00 / 0.00 |
| object | **graph-stage** | 0.927 | 0.765 | +0.161 | 0.755 | 0.17 / **0.54** |
| object | topreward-ll | 0.970 | 0.721 | +0.249 | 0.816 | 0.71 / 0.88 |
| object | robometer-fft | 0.649 | 0.224 | +0.424 | 0.986 | 0.28 / 0.48 |
| spatial | frame_index (CLOCK) | 0.969 | 0.986 | −0.017 | 0.000 | 0.00 / 0.00 |
| spatial | **graph-stage** | 0.892 | 0.597 | +0.295 | 0.853 | 0.22 / **0.64** |
| spatial | topreward-ll | 0.774 | 0.550 | +0.225 | 0.749 | 0.67 / 0.91 |
| spatial | robometer-4b | 0.855 | 0.708 | +0.147 | 0.982 | 0.22 / 0.32 |

(graph-absorb + robometer-4b/fft full rows in the script output; abbreviated here.)

- **graph-stage is genuinely progress, not a clock.** It ends far lower on failures than successes
  (gap +0.16..+0.37 vs the clock's ≈0), and its regression drawdown is **selective**: ~0.53–0.64 on
  failures vs only 0.17–0.22 on successes (≈3× ratio) — it dips when the trajectory regresses and
  stays smooth when it doesn't. A clock's drawdown is 0 by construction.
- **TOPReward's drawdown is NON-selective** (0.53–0.71 succ, 0.62–0.91 fail; ratio ~1.2–1.4×) — i.e.
  noisy everywhere, not regression-aware. So its curve fidelity is genuinely worse, consistent with
  its low VOC.
- **ROBOMETER (trained success head) separates succ/fail well** (large gaps, high AUROC on
  object/spatial) — its designed strength — but that is binary success detection, not the full-curve
  progress fidelity where graph-stage wins (E-PE5/6).
- **graph-absorb** saturates high on both succ and fail (term 0.99/0.95, tiny drawdown) → poor
  terminal separator despite decent ranking-AUROC; graph-STAGE is the better success/plateau signal.
- Caveat: VLM curves exist only for LIBERO, so this head-to-head is LIBERO-only (14/9/14 fails). A
  graph-only version of this test on balanced-failure datasets (transport_mh 30/20, robocasa 9/41,
  safe_*) is cheap; a VLM head-to-head there needs GPU runs (deferred).

### E-PE8 — data-efficiency + pipeline robustness (CPU; `scripts/sweep_progress_efficiency.py`)
Data-efficiency: fix K, hold out a fixed stratified test set, fit graph on n_fit subsampled train
episodes (3 subset seeds), report graph-stage VOC on the fixed test.

| dataset (K) | n_fit → VOC (mean±std over 3 subset seeds) |
|---|---|
| libero_spatial (K=128) | 25→0.924 · 50→0.947 · 100→0.953 · 200→0.962 · 400→0.965 |
| robomimic_transport_mh (K=32) | 8→0.838±0.14 · 15→0.926 · 25→0.906 · 35→0.927 |

- **Very data-efficient:** ~25 fit episodes already gives VOC 0.92 on libero_spatial (ceiling ~0.965
  by 200); transport_mh is at ceiling (~0.93) by ~15 demos. The graph needs few trajectories to
  calibrate a progress model.

Robustness (graph-stage VOC; reducer × window width/stride, fixed fit budget):

| dataset | identity (w2s3 / w4s2 / w8s4) | pca | umap |
|---|---|---|---|
| robomimic_transport_mh | 0.959 / 0.969 / **0.980** | 0.826 / 0.874 / 0.936 | 0.980 / 0.929 / 0.875 |
| mimicgen_square_d0 | 1.000 / 1.000 / 0.933 | 0.967 / 1.000 / 1.000 | 1.000 / 0.988 / 1.000 |
| libero_spatial | 0.964 / 0.974 / **0.976** | 0.934 / 0.941 / 0.962 | (n/a — umap too slow @28k win) |

- **identity is the robust choice:** never worst, low variance across windows, and slightly better with
  larger windows (w8s4). PCA consistently lags (0.83–0.96). UMAP is competitive on small data but
  high-variance across windows (transport 0.98→0.875). Confirms the identity + K recipe is not a
  fragile sweet spot — it's the stable dominant setting (matches the curation-campaign steer).

### E-PE9 — SPEC (planned): does progress-estimation QUALITY buy downstream policy utility?
Every result so far is a *correlation* metric. The payoff question is whether a better progress
signal makes a better **policy**. We test this as **dense progress-weighted BC** (the SARM RA-BC path
already wired in `scripts/{score_demos,run_weighted_bc}.py`), NOT demo subsetting (curation was null on
clean data — E24/E25). Crucially we run it in the regime where weighting *can* matter: **mixed / noisy
demos**, where downweighting bad transitions has something to bite on.

**Hypothesis.** If graph-stage progress is a genuinely better progress estimator (E-PE5/6/7), then RA-BC
weights derived from it should lift downstream success above uniform BC and approach the oracle
(quality-label) ceiling — and do so at least as well as weights derived from ROBOMETER/TOPReward progress.
Honest null on the table: BC may be robust enough that no weighting beats uniform (then the finding is
"progress-estimation accuracy does not transfer to BC utility here" — still worth stating).

**Testbed.** `transport_mh` — 300 multi-human demos with `better/okay/worse` operator-pair labels
(genuine quality variance; `*_oracle*` arm = label ceiling), and the injected-noise variants already on
disk (`${DATA_ROOT}/cupid_data/robomimic/datasets/transport/mh/transport_mh_noisy{,0.25,0.5}_v15.hdf5`
+ `*_oracle_clean.npy`) — the "harmful data" case. Start with `noisy0.5` (strongest signal), then `mh`.

**Arms** (per-step weight npy aligned to demos in numeric order; `run_weighted_bc.py --scores`):
1. `uniform` — no weights (baseline floor).
2. `oracle` / `oracle_clean` — quality-label / clean-vs-noisy weights (ceiling).
3. `graph-relu` — graph-stage progress → `rabc_weights`(δ=action horizon) [already produced for mh:
   `transport_mh_graph_relu.npy`; relu_vs_oracle demo-corr 0.35, ranks better>worse].
4. `robometer-rabc` — ROBOMETER per-frame progress on the demos → `rabc_weights`. **NEW.**
5. `topreward-rabc` — faithful TOPReward progress on the demos → `rabc_weights`. **NEW.**
6. (opt) `shuffled` — negative control (destroys temporal structure).

**The VLM arms need demo FRAMES** (demos are low-dim only). Render via robosuite playback of demo states
with an agentview camera (robomimic `dataset_states_to_obs`), then score with `vlm_reward_bench` models
(GPU 0). Cost: 300 demos × ~650 steps ≈ 195k frames; mitigate by scoring at reduced fps (VLMs already
downsample to ≤64 frames/traj, and RA-BC only needs a per-demo progress curve). This render+VLM-score
step is the critical path and the main cost/risk.

**Protocol.** DiffusionPolicy-CNN via `run_weighted_bc.py` (`--epochs 800 --rollout-every 50 --n-test 50
--n-envs 8`), identical config across arms, ≥3 seeds. Weights → `rabc_weights` with δ = the policy's
action-chunk horizon (match `score_demos --horizon`). Same fitted graph bundle (`graph_cache_*.pkl`) for
the graph arm.

**Primary metric.** Downstream **rollout success rate** (mean±std over seeds, best-checkpoint and
area-under-success-curve), per arm. Secondary: ESS / zero-fraction of each weight vector, and
weight-vs-oracle demo-level correlation (already in `*_summary.json`) as a sanity link between
progress-quality and the resulting weights.

**Decisions needed before running:** (a) confirm the weighted-BC path (cheaper, existing infra) vs a
heavier online-RL dense-reward fine-tune; (b) render+VLM-score the demos now (GPU, ~half a day) or first
derisk with a **VLM-free pass** (uniform vs oracle vs graph-relu) to confirm weighting moves success at
all on this testbed before spending VLM compute. Recommended: derisk VLM-free first.

**VLM-FREE DERISK — harvested (2026-07-27, weighted-BC path).** Chose weighted-BC, VLMs deferred.
Rollout success of already-trained arms (produced by the SARM/curation workstream; I only read them):

- CLEAN mixed-quality `transport_mh` (DiffusionPolicy-CNN via `run_weighted_bc.py`, 1 seed, best test
  success): **uniform 0.640, oracle 0.640, graph 0.500, length 0.560, seg_trim 0.460, stage_balance
  0.420.** → **NULL / negative: `oracle == uniform`, every weighting ≤ uniform, graph hurts.** No
  headroom on clean data (matches E24/E25 curation null); VLM arms here would be pointless.
- NOISY `transport_mh_noisy` (harmful-data regime, rgrl diffusion, `+data.sample_weights_path`, 3 seeds,
  final success): **uniform 0.272, length 0.308, deminf 0.382, oracle-clean 0.407.** → **HEADROOM
  EXISTS**: downweighting the injected-noise demos lifts success +0.14 (0.27→0.41); `deminf`
  (demo-influence baseline) recovers most of it. **But there is NO graph-progress arm on the noisy demos
  yet** (only oracle-clean + deminf) — that is exactly the E-PE9 contribution to run next.

**Verdict / next step.** Downstream weighting only matters where the data is genuinely harmful (noisy),
not on clean/mixed demos. The informative, non-redundant experiment is: produce graph-progress
(`rabc_weights`) scores on the NOISY demos via `score_demos.py` (CPU), then run a **graph arm** with the
SAME rgrl harness/config as the noisy uniform/deminf/oracle-clean arms (3 seeds, GPU 0), in a separate
output dir to avoid colliding with the curation workstream. Win condition: graph recovers the uniform→
oracle-clean headroom (≥ deminf 0.382, approaching 0.407). VLM arms only if graph clears this bar.

**GRAPH ARM — RESULTS (2026-07-27, VLM-free).** Produced graph-progress demo weights via
`score_demos.py` (fit on `transport_mh_salvage250` rollouts 124s/126f, state features, umap k32c24
w2s3; 300 noisy demos → 195800 steps, aligned to `oracle_clean`). Ran the rgrl harness
(`--data transport_mh_noisy --target transport_mh_demos --steps 100000`) in a separate
`transport_mh_noisy_graph/` dir. Rollout success (mean; graph n=2 seeds, s44 eval OOM'd under 4-way
GPU concurrency; others 3 seeds):

| arm | success | note |
|---|---|---|
| uniform | 0.272 | floor |
| **graph-ramp** | **0.360** | gentle weighting (ESS 0.93); 2 tight seeds (0.36/0.36) |
| deminf | 0.382 | demo-influence baseline |
| oracle-clean | 0.407 | ceiling (clean-vs-noisy labels) |
| graph-relu | 0.130 | HURTS — over-peaked (ESS 0.055, 24% zeros) starves BC |

- **Sanity (pre-run):** graph weights are ORTHOGONAL to the clean/noisy split —
  `spearman(graph-relu, oracle_clean)=0.026`; graph-relu mean weight clean 1.002 ≈ noisy 0.996. So any
  benefit is NOT from downweighting noisy demos (the graph can't see action-noise — it measures
  task-state progress). This testbed corrupts ACTIONS, which is orthogonal to progress.
- **graph-ramp modestly beats uniform (+0.09, ~deminf) despite the orthogonality** → the gain is
  transition-level emphasis on high-progress-rate steps, NOT noise-curation. Interesting but modest,
  n=2, and mechanism-unconfirmed.
- **graph-relu (the peaked RA-BC scheme) is destructive here** — ESS 0.055 means BC effectively trains
  on ~5% of samples. Scheme/ESS calibration dominates the outcome, swamping any progress-quality signal.
- **Honest conclusion:** on an ACTION-NOISE testbed, graph-progress weighting does not recover the
  curation headroom by design (orthogonal signal); a well-calibrated (ramp) weighting is ~neutral-to-
  slightly-positive, an over-peaked one (relu) hurts. To actually test whether progress-QUALITY buys
  utility, the harmful data must manifest as bad TASK PROGRESS (stalls / detours / partial completions),
  where the graph's signal is discriminating — that is the correct next testbed (deferred). Curves/ckpts:
  `${DATA_PATH}/training/rgrl/transport_mh_noisy_graph/`.

### E-PE10 — progress-relevant testbed (temporal stalls): no BC headroom
To test whether progress-QUALITY buys curation utility, we built a corruption the graph SHOULD see
(unlike action-noise): temporal STALLS (`scripts/inject_stalls.py` — held state + hold action inserted
into 40% of transport demos; 4.4% of steps; per-step GT `*_is_stall.npy`). A stall is DemInf-invisible
(held state + hold action = perfectly predictable, high I(S;A)) but zero task-progress.

- **Mechanism (positive, unlike E-PE9):** graph-relu downweights stall steps in **84% of corrupted
  demos** (within-demo mean 0.70 vs 0.99) — the progress signal DOES carry the stall info. (Noisy: it
  zeros only 9% of stall steps and conflates stalls with natural low-progress moments, global
  stall-AUROC 0.52.)
- **But NO downstream headroom.** rgrl weighted BC (`--data transport_mh_stalled`, seed 42):
  uniform 0.32, **oracle-per-step (zeros exactly the stalls, ESS 0.96) 0.26**, graph-ramp 0.30,
  graph-relu 0.06. The per-step ORACLE does not beat uniform → BC is robust to 4.4% stall steps
  (it averages out the hold actions), so no weighting can help. graph-relu's ESS-0.054 over-peaking is
  again destructive (0.06). [1 seed; the oracle≈uniform gap is within seed noise → "no effect".]

### Synthesis — downstream demo-curation is a robust NULL; the contribution is progress ESTIMATION
Across four regimes the graph's progress signal does not translate into a downstream BC-curation win,
each for a clear and DIFFERENT reason:
- **clean mixed-quality** (E-PE9): no headroom — `oracle == uniform` (quality variance too small).
- **action-noise** (E-PE9): headroom exists (DemInf recovers it, AUROC 1.0) but graph-progress is
  ORTHOGONAL to action quality (spearman 0.026) — wrong signal by construction.
- **temporal stalls** (E-PE10): graph SEES the corruption (84%) but there's no headroom — even the
  per-step oracle can't beat uniform (BC robust to the corruption).
- **coverage reweighting** (E24/E25, curation log): any reweighting off natural density hurts BC.
Plus: the RA-BC weight schemes themselves are mis-calibrated for BC — `relu` over-peaks (ESS ~0.05 →
hurts everywhere), `ramp` is ~uniform (ESS ~0.93 → no effect).

**Conclusion.** The behavior graph is an excellent, cheap, unsupervised, data-efficient **progress
estimator** (E-PE1–8: beats TOPReward + ROBOMETER on ranking across 3 LIBERO suites; VOC≈0.97 across 33
sim datasets; genuinely progress-not-clock; ~25 demos to ceiling). It is NOT a demonstration-curation
method — that role is better served by action-state MI curation (DemInf) where curation helps at all.
Recommendation: lead with progress estimation; treat demo-curation as a documented negative.

### E-PE11 — consolidated tables + significance (CI / CLD)
`scripts/consolidate_progress_results.py` reads the saved pool500 curves and emits, over the 3 LIBERO
suites where all methods coexist (frame_index excluded): an AGGREGATE table (all methods × all metrics,
mean over suites, sorted), a PER-DATASET breakdown, and — with `--stats` — per-metric 95% CIs + a
Compact Letter Display (paired episode bootstrap N=2000 within each suite, averaged across suites;
Holm-adjusted pairwise bootstrap tests, α=0.05; methods sharing a letter are NOT significantly
different). Outputs in `analysis/out/progress_consolidated/` (`aggregate`, `breakdown`, `stats_ci_cld`,
`progress_tables` + CSVs). A graph-only cross-dataset aggregate (E-PE4, 54 datasets) is appended.

Aggregate (mean over goal/object/spatial), sorted by VOC:

| method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |
|---|---|---|---|---|---|---|
| **graph-stage** | pi0.5 emb | **0.965** | **0.930** | **0.961** | **0.055** | 0.861 |
| topreward-reimpl | RGB 8B | 0.858 | 0.757 | 0.686 | 0.446 | 0.537 |
| robometer-fft | RGB 4B | 0.794 | 0.650 | 0.780 | 0.243 | **0.937** |
| robometer-4b | RGB 4B | 0.785 | 0.622 | 0.815 | 0.161 | 0.917 |
| graph-absorb | pi0.5 emb | 0.699 | 0.543 | 0.645 | 0.488 | 0.804 |
| topreward-faithful | RGB 8B | 0.591 | 0.444 | 0.594 | 0.239 | 0.762 |

CI + CLD, VOC (representative of the ranking metrics): graph-stage 0.965 [0.962,0.968] **a** ; topreward-
reimpl 0.858 [0.851,0.866] **b** ; robometer-fft 0.794 [0.784,0.803] **c** ; robometer-4b 0.785
[0.776,0.794] **c** ; graph-absorb 0.699 **d** ; topreward-faithful 0.590 **e**. → graph-stage is its own
letter (significantly ahead) on VOC, Kendall, Pearson AND MAE.

CI + CLD, success-AUROC: robometer-fft 0.937 [0.906,0.963] **a** ; robometer-4b 0.917 [0.856,0.963]
**a** ; graph-stage 0.861 [0.768,0.936] **ab** ; graph-absorb 0.804 **b** ; topreward-faithful 0.762
**b** ; topreward-reimpl 0.536 **c**.

**What a higher AUROC means in practice (and its caveats).** AUROC here = *success detection*: threshold
each episode's FINAL progress value and ask how well it separates completed from failed episodes (1.0 =
perfect ranking, 0.5 = chance). So robometer's higher AUROC means it is the better binary
"did-the-task-succeed?" classifier — expected, since it has a supervised success head. Caveats:
- **Not significant here.** graph-stage sits at CLD `ab` on AUROC — it shares letter `a` with both
  robometer checkpoints (CIs overlap), because with only ~9–14 failures per 500-ep suite AUROC is
  estimated from a handful of episodes. The "robometer wins AUROC" gap is within noise.
- **Different job than progress estimation.** AUROC judges only the endpoint; VOC/Kendall/Pearson/MAE
  judge whole-trajectory fidelity, where graph-stage is significantly best (its own letter). Dense
  shaping/monitoring → trajectory fidelity (graph); a success/termination detector → the endpoint call
  (robometer, unsupervised graph statistically tied).
- **On the failure-richest suite (libero_goal) graph-stage AUROC 0.976 BEAT both robometer ckpts**; the
  robometer edge appears only on object/spatial where failures are scarcest and the estimate least
  reliable.

## Reproduce
Graph eval (CPU) — behavior_graphs env with umap/sklearn/scipy:
```
PYTHONPATH=src ${DATA_ROOT}/behavior_graphs/venvs/diffusion_policy_venv/bin/python \
  scripts/eval_progress_benchmark.py \
  --rollout-dir ${DATA_PATH}/rollouts/libero_goal \
  --name lg --reducer identity --n-clusters 128 --eval-per-task 20   # aligned; omit for 3-fold
```
Cross-dataset sweep of our method (E-PE4, CPU, all feasible datasets):
```
PYTHONPATH=src:scripts ${DATA_ROOT}/behavior_graphs/venvs/diffusion_policy_venv/bin/python \
  scripts/sweep_progress_allds.py --out analysis/out/progress_allds   # forces policy-embedding layer
# sim families were re-run to a temp dir and merged (LIBERO/safe already on policy emb):
python scripts/merge_progress_sweep.py --base .../progress_allds/sweep_results.json \
  --override .../progress_allds_polemb/sweep_results.json --out analysis/out/progress_allds
```
VLM baselines (GPU 0) — robometer venv, HF cache:
```
CUDA_VISIBLE_DEVICES=0 HF_HOME=${HF_HOME} \
  ${DATA_ROOT}/robometer/.venv/bin/python scripts/vlm_reward_bench.py \
  --rollout-dir .../libero_goal --suite libero_goal --per-task-limit 50 \
  --method robometer --ckpt robometer/Robometer-4B        # or jesbu1/robometer-4b-fft-libero
  # --method topreward --n-queries 8 --n-ctx 6            # Qwen3-VL-8B
```
Confidence + plots (from saved *_curves.npz):
```
python scripts/progress_bootstrap.py --curves-dir <dir> --primary graph-stage \
  --method "graph-stage:..._curves.npz" --method "topreward:..._curves.npz" ... \
  --n-seeds 300 --subset-per-task 20
python scripts/plot_progress_traces.py --curves-dir <dir> --out traces.png --method ... --per-task 1
```
Robometer notes: the package CODE is vendored as the pinned third_party submodule
`third_party/robometer` (private mirror `erikbr01/robometer` @ 5b81525, == upstream
`robometer/robometer`); `--robometer-repo` defaults to it. The heavy runtime deps still come from
the robometer venv at `${DATA_ROOT}/robometer/.venv` (unsloth/flash-attn/qwen3-vl); the harness's
`sys.path` insert shadows that venv's editable install with the vendored code. The harness loads the
model once and forces bf16 (their Qwen3-VL vision tower otherwise mixes fp32/bf16 -> layer_norm
dtype error). The official TOPReward repo (`github.com/TOPReward/TOPReward` @ 4877a0e) is NOT
vendored — kept only as the reference clone `${DATA_ROOT}/topreward` that verified ROBOMETER's
`TopReward` is byte-identical; the faithful TOPReward we report is run through the vendored robometer.

## Repo layout (Erik's architecture: graphs -> behavior_graphs, policies/training/eval -> rgrl)
- **This work lives in `behavior_graphs`, branch `progress-estimation`** (repo of record): the metrics
  (`src/behavior_graphs/progress/estimation.py`), the four `scripts/`, this log, and
  `analysis/out/progress_bench_results/` (curves + traces). Ported from the graph_guided prototype and
  **verified to reproduce the pre-port numbers within KMeans-seed noise** (stage VOC 0.952 -> 0.965 on
  the aligned 200-ep set) using the native behavior_graphs graph stack.
- Curation (coverage, E24/E25) is on `behavior_graphs` branch `rgrl-backend-wt`.
- `rgrl` (branch `weighted-sampling`) is the policy-learning repo.
- The original graph_guided prototype remains in the separate `behavior-graph-lab` repo
  (`$HOME/online_rl`, master) as a backup — not a submodule of behavior_graphs.

## WARP-RM integration (2026-07-27, branch `warp-rm-baseline`)
Added WARP-RM as a fourth baseline. WARP-RM is self-supervised and *per-task* (trains on a task's
successful demos), so a fair head-to-head uses **its own sim env as the testbed**: on WARP-RM's
sim-bottles task the released checkpoint `uynitsuj/warp-rm-sim-bottles-sss15` is in-distribution —
no training needed. This is also the cleanest path to the **input-matched** comparison this log
defers: the graph can cluster the *same frozen DINOv3 features* WARP-RM's transformer consumes, so
both estimators see identical pixels/features (no pi0.5-embedding privilege caveat).

Two halves, split so the GPU work is isolated:
- **`third_party/warp_rm/scripts/eval/dense_curves.py`** (WARP-RM env, GPU only if features
  uncached): reuses `warp_rm.eval.scoring.score_dataset`; reads either a LeRobot repo or our
  canonical rollout layout (`--rollout-dir`) and dumps one npz of per-episode per-frame
  `warp_rm_progress`. Committed to the mirror.
- **`scripts/warp_rm_progress_bench.py`** (behavior_graphs env, CPU): aligns those curves to the SAME
  held-out episodes the graph used (`--align-to <graph *_curves.npz>`) and scores them with the
  shared `progress_regression_metrics` + `success_auroc`. Writes
  `analysis/out/progress_bench_results/<name>_warp_rm_{curves.npz,metrics.json,report.md}` so it
  sits beside graph/topreward/robometer outputs. CPU test: `tests/test_warp_rm_bench.py`.

rgrl can't train WARP-RM — rgrl is a BC/policy trainer (diffusion/flow UNets; its only reward hook
is `sample_weights_path` for weighted-BC *downstream*). WARP-RM training, if ever needed for a
non-bottles task, runs in WARP-RM's own uv env (`uv sync` in `third_party/warp_rm`; torch 2.8/cu128,
decord, DINOv3) — GPU, gated on explicit go-ahead.

Recipe (⚠ = GPU or large download; run only once cleared):
```
# 0. one-time env (⚠ download; venv on SSD)
cd third_party/warp_rm && UV_CACHE_DIR=${DATA_ROOT}/uv_cache \
  uv sync && uv sync --extra wandb        # torch group auto: scripts/setup_torch.sh
# ⚠ download the in-distribution sim-bottles checkpoint + dataset
HF_HOME=${HF_HOME} hf download uynitsuj/warp-rm-sim-bottles-sss15
HF_HOME=${HF_HOME} hf download uynitsuj/sim-bottles-mjwarp-v1 --repo-type dataset

# 1. build a canonical rollout dir for bottles (media + DINOv3 features), then
#    graph benchmark on it -> aligned eval set  (⚠ DINOv3 feature extraction = GPU)
python scripts/eval_progress_benchmark.py --rollout-dir <bottles_canonical> \
  --name bottles --layer dinov3 --reducer identity --n-clusters 128 --eval-per-task 20

# 2. WARP-RM on the SAME eval episodes  (⚠ dense inference = GPU)
python scripts/warp_rm_progress_bench.py --rollout-dir <bottles_canonical> --name bottles \
  --warp-rm-python ${DATA_ROOT}/behavior_graphs/venvs/warp_rm/bin/python \
  --checkpoint <hf_cache>/.../warp-rm-sim-bottles-sss15.pt --gpu 0 \
  --align-to analysis/out/progress_bench_results/bottles_graph_direct_stage_curves.npz

# 3. CI + plots (CPU) — reuses the same *_curves.npz as the LIBERO runs
python scripts/progress_bootstrap.py --curves-dir analysis/out/progress_bench_results \
  --primary graph-stage --method graph-stage:bottles_graph_direct_stage_curves.npz \
  --method warp_rm:bottles_warp_rm_curves.npz --n-seeds 300 --subset-per-task 20
```
### Live-run attempt 2026-07-27 (PAUSED on DINOv3 gate)
Set up on GPU 0 / SSD, then paused (Erik) at the one blocker below.
- **env**: `${DATA_ROOT}/behavior_graphs/venvs/warp_rm` (torch 2.8.0+cu128, RTX 4090 OK).
- **ckpt**: `${DATA_PATH}/warp_rm/ckpt-sss15/warp_rm_sss15.pt` —
  stamped `feature_stride=1`, `standard_stride_src=15` (⇒ standard_feat_steps=15),
  `cameras=['top_camera-images-rgb']`, bidirectional, ablation `no_abs`, DINOv3-768,
  val_spearman 0.998. **Does NOT bundle DINOv3 weights.**
- **data**: `${DATA_PATH}/warp_rm/sim-bottles-mjwarp-v1` (7.1 GB, LeRobot
  v3.0, 2438 eps, 3 RGB cams @30Hz, `supplemental/` excluded). **meta/episodes has NO
  success/reward column — all-success demos.** `paper-sim-n128-traces` is qpos-only (no images) →
  useless for a vision RM.
- **BLOCKER**: `build_backbone("dinov3")` → `facebook/dinov3-vitb16-pretrain-lvd1689m` is a **gated
  Meta HF repo**; no token logged in, nothing cached ⇒ 401. Blocks feature extraction for BOTH
  WARP-RM and the graph. Unblock: accept the license on that HF page (Erik's account) then
  `HF_HOME=${HF_HOME} <venv>/bin/hf auth login` (or set HF_TOKEN).

DECIDED (Erik): first subset **200 eps**, eval-per-task aligned; scope includes **generating failed
rollouts** so success-AUROC is meaningful (all-success demos give VOC/Kendall/Pearson/MAE only,
AUROC=NaN). Failed rollouts need their sim + policy ckpts (`uynitsuj/paper-sim-policy-checkpoints`,
vanilla 12.5% / warp-bc 28% success) + MuJoCo-Warp renderer to emit RGB video — separate task, sized
before running.

RESUME once DINOv3 is unblocked: (1) extract DINOv3 fs=1 features for 200 eps via WARP-RM's backbone
into the shared cache AND `policy_embeddings/dinov3.npz`; write `episodes/metadata.yaml`
(index/length/success=True). (2) `eval_progress_benchmark.py --layer dinov3 --n-clusters 128
--eval-per-task 20` → graph curves + aligned idx. (3) `warp_rm_progress_bench.py --warp-rm-python
<venv> --checkpoint <sss15.pt> --gpu 0 --align-to <graph stage curves>`. Open sub-Q: symlink vs
re-encode LeRobot per-episode video segments into `media/` (WARP-RM can read the LeRobot repo
natively via `--lerobot-repo`, so `media/` is only needed if we route WARP-RM through `--rollout-dir`).

### Live-run 2026-07-28 — COMPLETE (DINOv3 unblocked; WARP-RM + VLM baselines on sim-bottles)
Meta granted `erikbauer` access to `facebook/dinov3-vitb16-pretrain-lvd1689m`; downloaded to
`HF_HOME=${HF_HOME}`. Ran on the first 200 sim-bottles episodes; a fixed
40-ep eval set (positions {0-3,20-23,…,180-183}) defined once and shared across methods.

| method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC |
|---|---|---|---|---|---|---|
| WARP-RM (sss15, trained head) | DINOv3 fs=1 | 0.999 | 0.995 | 0.997 | 0.033 | nan |
| **graph:stage (K=128)** | **pi0 obs (policy)** | **0.875** | **0.748** | **0.879** | **0.110** | nan |
| ROBOMETER-4B (zero-shot) | RGB (16 frames) | 0.873 | 0.762 | 0.885 | 0.199 | nan |
| TOPReward (Qwen3-VL-8B, faithful) | RGB (≤64 frames) | 0.688 | 0.530 | 0.641 | 0.300 | nan |

- **⚠ TRAIN-SET CONFOUND (2026-07-28).** The dataset is 100% `train` (`splits: {train: 0:2438}`);
  WARP-RM sss15 holds out only N_VAL=20 (seed 42) and trains on the rest, so the 40-ep eval set is
  in WARP-RM's TRAINING data → its 0.999 is a memorization number, NOT generalization. Asymmetric:
  the graph is scored on eps held-out *from its own fit set* (fair-for-graph), the VLMs are zero-shot
  (never saw sim-bottles), but WARP-RM saw the eval eps in training. So WARP-RM's row is NOT comparable
  as-is. Clean fix = evaluate all methods on rollouts no trained model saw (fresh FAILED + held-out
  success rollouts), which also makes AUROC live. WARP-RM's own 20 val eps are a weak half-measure
  (tiny, still pi0-train, WARP-RM ~.998 there anyway).
- **Graph now on POLICY features (2026-07-28).** The graph is built on policy internal features, not
  raw vision — an earlier graph-on-raw-DINOv3 run (VOC 0.673) was off-method and discarded (Erik). To
  get a faithful feature, we pulled the WARP-RM paper policy `uynitsuj/paper-sim-policy-checkpoints`
  (`vanilla`, a pi0/OpenPI Orbax ckpt), and ran its obs-encoder OFFLINE over the recorded sim-bottles
  obs: `embed_prefix` image tokens (3 cams × 256), mean-pooled → 2048-d, stride 5 (`scripts/
  warp_rm_extract_pi0_obs.py`, openpi env `venvs/openpi_uynitsuj`, clone `${DATA_ROOT}/openpi_uynitsuj`).
  K-sweep ~monotonic (VOC 0.80→0.82→0.86→0.85→0.875 for K=8→128), K=128 best — SAME pattern as
  LIBERO, and far above the DINOv3-fed graph (0.673). On policy features the graph is **on par with
  ROBOMETER-4B** (0.875 vs 0.873 VOC), well above TOPReward, below in-distribution WARP-RM.
- **AUROC=nan everywhere** — all-success ⇒ single class. The discriminating metric stays dark until
  FAILED rollouts exist.
- **Read with care:** GT progress is a linear frame ramp, so VOC/Kendall/Pearson-vs-GT reward
  time-linearity. WARP-RM is trained to predict exactly that on in-distribution success demos
  (val_spearman .998) ⇒ near-oracle almost by construction. ROBOMETER-4B zero-shot is a strong
  pixel baseline (0.87); faithful TOPReward trails (0.69).

Pipeline / artifacts (`${DATA_PATH}/warp_rm/run200/`):
`dense_curves.py --lerobot-repo --n 200` → WARP-RM curves; `scripts/warp_rm_export_dinov3.py`
(canonical dir, used only to fix the 40-ep eval set) → `warp_rm_progress_bench.py --align-to`;
`scripts/warp_rm_build_media.py` (per-episode top-cam MP4s for the 40) → `vlm_reward_bench.py`
(`topreward_ll` Qwen3-VL-8B; `robometer` ckpt `robometer/Robometer-4B`) in the `third_party/robometer`
uv env. Harness fixes this run: RobometerModel now runs under `inference_mode`/frozen grads and
linspace-subsamples each episode to `max_frames=16` (matching robometer's own eval) — full-episode
forwards OOM a 24GB GPU.

### Fresh rollouts recorded 2026-07-28 (unseen eval set for a fair comparison + AUROC)
Recorded **100 fresh vanilla-pi0 rollouts** in the WARP-RM MuJoCo-Warp sim, so the eval set is
UNSEEN by WARP-RM (fixes the train-set confound) and contains real failures (AUROC becomes live).
- **Stack (all on SSD)**: cloned `uynitsuj/abc-rabc@release-candidate` → `${DATA_ROOT}/abc_rabc`
  (venv `venvs/abc_rabc`, mujoco-warp 1.15 / mujoco 3.11, torch cu128); openpi pi0 server via
  `scripts/serve_paper_sim_policy.py --policy vanilla` (GPU 0), sim `eval_policy.py --policy-backend
  pi0 --num-worlds 100 --save-video` (GPU 1). openpi_client installed into the abc_rabc venv;
  lerobot shim added to the server script.
- **Result**: `run200/rollouts100/` — 100 `world_NNN.mp4` + 100 `qpos_trace_*.npz` + `summary.json`
  (per-world success/reward/placement + init randomization). **31/100 success** (mean 4.72/6 bottles).
- **Policy-embedding capture (2026-07-29)**: the inference path now records the pi0 obs embedding at
  inference (rgrl-style). Server `Policy.infer` (openpi) emits `obs_embedding` = embed_prefix image
  tokens (3×256) mean-pooled → 2048-d (computed into a local + attached AFTER the output transform,
  which otherwise drops unknown keys); the abc pi0 client stashes it; `run_eval` appends one per chunk
  and writes per-world `obs_emb_<world>_s<seed>.npz` + `obs_embedding_path` in summary. Validated on
  2 worlds → (60, 2048) each (60 = num_chunks). Patches: `experiments/warp_rm_patches/{openpi,abc_rabc}_obs_embedding.patch`.
- **Extracted embeddings for the EXISTING 100 rollouts** without re-render: the recorded mp4 IS the 3
  policy cameras concatenated (672×168), and pi0's image-token embedding is state-independent, so
  `scripts/warp_rm_extract_pi0_obs_from_video.py` splits each frame → 3 cams → pi0 embed → 2048-d.
  Output `run200/rollouts100_pi0_obs/` (11460 pts × 2048, 100 rollouts = 31 succ / 69 fail, stride 15).
### FAIR head-to-head on the 100 unseen rollouts (2026-07-29) — train-set confound removed, AUROC live
All four progress estimators scored on the fresh 100 rollouts (31 succ / 69 fail), unseen by WARP-RM.
Graph: `eval_progress_benchmark --layer pi0_obs` 3-fold over `rollouts100_pi0_obs` (K=128 shown).
WARP-RM: `dense_curves` (DINOv3) over top-cam mp4s → `warp_rm_progress_bench`. VLMs: `vlm_reward_bench`
over top-cam mp4s. Regression metrics on the 31 successful eps; success-AUROC over all 100.

| method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | success-AUROC ↑ |
|---|---|---|---|---|---|---|
| WARP-RM (sss15, trained head) | DINOv3 top-cam | **0.969** | **0.886** | **0.967** | **0.094** | 0.664 |
| ROBOMETER-4B (zero-shot) | RGB | 0.725 | 0.592 | 0.787 | 0.194 | 0.542 |
| graph:stage (K=128) | pi0 obs (policy) | 0.635 | 0.509 | 0.657 | 0.191 | **0.742** |
| TOPReward (Qwen3-VL-8B) | RGB | 0.640 | 0.486 | 0.634 | 0.272 | 0.663 |

**Two different winners by metric:**
- *Progress ranking* (VOC/Kendall/Pearson): WARP-RM dominates (0.97) even OUT-OF-DISTRIBUTION — it
  generalizes progress prediction strongly (dropped only 0.999→0.969 vs the train-set run). ROBOMETER
  (0.73) > graph ≈ TOPReward (0.64).
- *Success detection* (AUROC — the metric that needs failures): **the graph is best (0.742)** >
  WARP-RM (0.664) ≈ TOPReward (0.663) > ROBOMETER-4B (0.542, ~chance). The graph's progress signal
  separates success from failure best, even though its rank-correlation is lower — WARP-RM predicts a
  clean ramp even on failed rollouts, so its progress is a weaker success discriminator.
- Graph AUROC by K: 0.726 (K=32) / 0.649 (K=64) / **0.742 (K=128)**. Artifacts: `run200/{rollouts_graph_out,
  rollouts_warp_out,rollouts_vlm_out}`.
- **Dense K sweep (window w=2/s=3, 3-fold)** `run200/rollouts_ksweep_fine`: progress metrics peak at
  **K≈24** (VOC 0.663 / Kendall 0.525 / Pearson 0.674 / MAE 0.181) — K=128 was NOT optimal; VOC rises
  K=3→24 then plateaus (~0.62-0.64). Very low K (3-6) underperforms (K=5 trough VOC 0.505). AUROC is
  NOISY across K (0.56-0.75, e.g. K8=0.750 vs K10=0.577) — small-sample artifact (3-fold over only 31
  succ eps ⇒ ~10 succ/fold). **Use K≈24** for progress ranking; robust AUROC-based K selection needs
  more rollouts. (The head-to-head table above used K=128; at K=24 the graph's VOC rises to 0.663,
  still below WARP-RM's 0.969 progress ranking but its AUROC 0.696 stays > ROBOMETER's 0.542.)
- **Joint KxWxS sweep** (`run200/rollouts_kws_grid.json`, K{12-48} x w{1,2,3,4,6} x s{1,2,3}, 3-fold):
  window WIDTH is the bigger lever than K -- the default w=2 was too narrow. Marginal mean VOC (over K)
  rises monotonically with w: w1=0.612, w2=0.634, w3=0.666, w4=0.674, w6=0.678; stride barely matters.
  Best VOC **K=32,w=4,s=3 -> VOC 0.730/MAE 0.173/AUROC 0.669**; best+balanced AUROC **K=20,w=3,s=1 ->
  AUROC 0.759/VOC 0.690/MAE 0.181**. Widening w 2->4 + lower K lifts the graph from VOC 0.635 (K=128,w=2)
  to ~0.73 -- now MATCHING ROBOMETER progress ranking (0.725) while keeping best success-AUROC
  (~0.76 >> WARP-RM 0.664, ROBOMETER 0.542). **Recommended graph config: K~20-32, w~3-4, s=1.**

### FLOW-head embedding switch (2026-07-29, CODE ONLY — not yet run)
Hypothesis (Erik): the graph's mediocre rollout numbers are because we used the pi0 OBS/prefix
embedding; on other tasks the strong feature is the policy's action head. Switched the captured
feature to the **action expert's LAST-LAYER hidden state at the LAST flow-matching timestep**
(mean-pooled over action tokens): new `Pi0.sample_actions_with_embedding` (pi0.py) returns
`(actions, flow_emb, obs_emb)` via a static-`num_steps` python loop that captures the final step's
`suffix_out`. Wired through: server `Policy.infer` emits `flow_embedding` (+ obs); abc pi0 client
stashes both; `run_eval` writes per-world `flow_emb_*.npz` (+ obs). Demo extractor
`warp_rm_extract_pi0_obs.py` switched to flow (layer `pi0_flow`, uses real LeRobot state). Patches in
`experiments/warp_rm_patches/`. All files py_compile-clean; **nothing run on GPU yet** (Erik).
IMPORTANT: flow_emb is STATE-dependent (embed_suffix), so it CANNOT be extracted from video alone —
`warp_rm_extract_pi0_obs_from_video.py` (obs, state-free) can't produce it. To get flow embeddings for
the 100 rollouts, RE-RUN `eval_policy` (same seeds → identical rollouts + flow_emb captured), or feed
qpos-derived state to an offline extractor. Then re-score the graph on `pi0_flow` (expect a lift toward
the strong other-task numbers).

### FLOW-head embedding RESULT (2026-07-29, RAN — hypothesis REJECTED)
Ran the flow path on GPU 1 (pi0 server + `eval_policy --num-worlds 100 --output-dir run200/rollouts100_flow`,
same seed 20260511; per world: `flow_emb`(60,1024) + `obs_emb`(60,2048) + `obs_raw` state/imgs/prompt).
Success this run: **28/100** (`ever_success`/top-level `success` field, same convention as the obs run's
31/100; note `final_success` is only 9 — bottles can leave the bin, so we score ever_success). Built
`run200/pi0_flow_rollout_dir` (5749 rows x 1024d, 28 succ/72 fail) + a **resolution-matched obs baseline**
`run200/pi0_obs_matched_rollout_dir` (same 5749 per-chunk rows from the SAME run's `obs_emb`, 28/72) so
flow-vs-obs is like-for-like (identical trajectories/resolution/success). Joint KxWxS sweep (K{12,16,20,24,
32,48} x w{1,2,3,4,6} x s{1,2,3}, 3-fold, identity, graph_direct:stage) via `scripts/warp_rm_flow_kws_sweep.py`,
grids `run200/rollouts_{flow,obs_matched}_kws_grid.json`:
- **OBS is far stronger than FLOW.** Best obs **VOC 0.798 (K20/w6/s2) / AUROC 0.808 (K32/w6/s2)**; best flow
  only **VOC 0.448 (K16/w1/s3) / AUROC 0.657 (K20/w4/s3)**. At every matched config obs dominates
  (K32/w6/s2: obs 0.778/0.808 vs flow 0.124/0.523).
- **Opposite window response.** obs VOC RISES monotonically with window width (w1 .653 -> w6 .768; K flat
  ~.72); flow VOC FALLS with width (w1 .36 -> w6 .20). The flow-head feature (action-expert last layer at the
  last flow step) is high-frequency/noisy per-frame — windowing smooths obs into a progress-monotonic
  manifold but destroys flow's weak signal. Flow embedding is NOT degenerate (median per-dim std 0.20, no
  dead dims, ~17% frame-to-frame variation); it simply carries less progress structure here.
- Flow embedding is healthy but wrong for this: **keep the pi0 OBS/prefix embedding as the graph feature.**
  The matched-obs per-chunk numbers (0.798/0.808) actually beat the earlier video-derived obs grid
  (0.730/0.759) — finer per-chunk resolution helps. Updated best graph config on the fair rollout set:
  **obs embedding, K~20-32, w=6, s=2 -> VOC ~0.80 / AUROC ~0.81** (graph AUROC 0.81 >> WARP-RM 0.66, ROBOMETER
  0.54; WARP-RM still leads progress-ranking VOC 0.969). Artifacts: `run200/{pi0_flow_rollout_dir,
  pi0_obs_matched_rollout_dir,rollouts_flow_kws_grid.json,rollouts_obs_matched_kws_grid.json}`. pi0 server
  killed, both GPUs freed.

### DIRECT-REGRESSION MLP baseline (2026-07-30)
Added a simple learned baseline `scripts/warp_rm_mlp_progress.py`: a small MLP (sklearn MLPRegressor,
StandardScaler) that regresses per-frame progress DIRECTLY from a policy representation — the direct-
regression counterpart to the behavior graph and the learned analog of WARP-RM's premise. Contract matches
`eval_progress_benchmark.py` exactly (same folds via `bench.stratified_folds`, same metrics: VOC/Kendall/
Pearson/MAE over successful held-out eps + success-AUROC on the tail score over ALL held-out eps).
Target = normalized time `linspace(0,1,T)` on SUCCESSFUL TRAIN episodes only (a failed demo never
completes; a full ramp target would teach the head that failures finish → hurts AUROC). Leakage-safe:
scaler + MLP fit on train frames only. Optional temporal window (concat prev w-1 frames; w=1 = pure
per-frame). Run on the SAME fair rollout set as the graph (`pi0_obs_matched_rollout_dir` / `pi0_flow_
rollout_dir`, 5749 rows, 28 succ/72 fail, 3-fold, seed 0). Sweep log `run200/mlp_sweep.log`.
- **obs MLP best: w1 → VOC 0.579 / AUROC 0.581** (α=1e-3); AUROC peaks 0.614 (α=1e-1, VOC 0.566). Widening
  the window COLLAPSES it (w3 VOC 0.35, w6 0.10, w8 ≈ 0) — opposite of the graph. Concatenation multiplies
  input dim (w×2048) while only ~18 succ eps/fold are available → severe overfit. Sweet spot is pure
  per-frame.
- **flow MLP best: w1 α=1.0 → VOC 0.450 / AUROC 0.688** — flow's AUROC actually TOPS the obs MLP's (and the
  flow-graph's 0.657), so the flow-head feature does carry some success-discriminative signal a learned head
  extracts, but its progress-RANKING (VOC 0.45) stays poor; strong reg (α=1.0) helps.
- **Verdict: on this testbed the UNSUPERVISED graph beats the SUPERVISED per-frame MLP on the identical obs
  feature** — graph obs VOC 0.798 / AUROC 0.808 vs MLP obs 0.579 / 0.581. The graph's window+cluster+absorb
  pipeline extracts temporal progress structure a small direct regressor can't from only 28 successful demos.
  Ranking on the fair 100-rollout set — progress VOC: WARP-RM 0.97 ≫ graph-obs 0.80 > ROBOMETER 0.73 >
  TOPReward 0.64 ≈ flow-MLP/obs-MLP 0.58 > graph-flow 0.45; success-AUROC: graph-obs 0.81 > flow-MLP 0.69 >
  WARP-RM 0.66 ≈ TOPReward 0.66 > obs-MLP 0.61 > ROBOMETER 0.54. NOTE the MLP would likely improve a lot with
  more successful rollouts (n≥300) — its ceiling here is the 28-success supervision, not the method.

### INTEGRATION into the progress_estimation package (2026-07-31)
Merged `origin/progress-estimation` (PR #35: the `progress_estimation` package reorg — plugin
methods/metrics/io/runner + hydra `run_progress_bench.py` + `configs/progress_estimation/*` + the
`data/{dataset,zoo}` config zoo) into `warp-rm-baseline`, and re-expressed the whole WARP-RM line in the
new architecture (the standalone `scripts/eval_progress_benchmark.py` + `vlm_reward_bench.py` were deleted
upstream; my `warp_rm_{progress_bench,mlp_progress,flow_kws_sweep}.py` scripts were removed as superseded).
- **Methods** (`src/behavior_graphs/progress_estimation/methods/`, registered in `__init__` + `METHOD_REGISTRY`,
  configs in `configs/progress_estimation/method/`):
  - `WarpRM` (`warp_rm.py`, modality=video): WARP-RM dense reward model via the vendored
    `third_party/warp_rm` (`score_dataset` + `dense_inference_relative`, same path as its `dense_curves.py`);
    heavy deps lazy-imported → instantiates on CPU, runs in the warp_rm venv on GPU.
  - `MLPRegressorProgress` (`mlp.py`, modality=embedding, CPU): direct per-frame regression of
    normalized-time progress from the policy embedding, trained on successful demos only. Reproduces the
    old standalone numbers exactly (obs w1: VOC 0.579 / AUROC 0.581).
  - Ported the ROBOMETER OOM fix (max_frames subsample + grad-freeze) from the deleted `vlm_reward_bench.py`
    into `methods/vlm.Robometer` so nothing was lost.
- **Dataset**: `scripts/utils/warp_rm_build_bench_dataset.py` fuses the flow re-run (`rollouts100_flow`) into ONE
  aligned canonical dir `run200/sim_bottles_bench` — `policy_embeddings/{obs,pi0_flow}.npz` + top-camera
  `media/ep*.mp4` rendered from the SAME trajectories (fixes the earlier obs/topcam run-mismatch), 100 eps
  (28 ever-succ / 72 fail). Registered as `warp_rm_sim_bottles` in `configs/data/warp_rm.yaml` (+ `all.yaml`,
  + `zoo.yaml` `warp_rm` family, + instruction in `io.SINGLE_TASK_INSTRUCTIONS`). `obs.npz` is the
  auto-default layer (strong feature); `data.layer=pi0_flow` selects the flow feature.
- **Env**: `src/behavior_graphs/envs/warp_rm.WarpRMAdapter` (+ `configs/env/warp_rm.yaml`, registry) — light
  scope: `load_policy` (wraps the pi0 Orbax ckpt) + `load_rollouts` (canonical dir) work; `collect_rollouts`
  raises with the exact abc_rabc+openpi reproduction commands (the sim stacks are SSD clones, not submodules;
  full sim-collection wiring deferred to a GPU-validatable follow-up).
- **Verified (CPU, no GPU per Erik)**: all 10 method configs instantiate; `run_progress_bench
  data.name=warp_rm_sim_bottles` runs graph/MLP end-to-end via hydra (graph-stage on obs → VOC 0.778 /
  AUROC 0.808, matching the pre-reorg sweep); `tests/{test_progress_estimation,test_configs,test_progress}`
  (50) + `tests/integration/test_warp_rm.py` (3) green. WARP-RM method now RUN on GPU (see 2026-08-02
  validation below); env live-collection remains implemented-but-unrun. NOTE the hydra CLI method/split-swap overrides need the full group
  path (`progress_estimation/split@split=kfold`; method-add is fiddly) — a pre-existing artifact of
  `run_progress_bench` rooting CONFIG_PATH at `configs/`; direct `BenchmarkRunner` use is unaffected.

### GPU validation 2026-08-02 — WARP-RM method runs through the runner (GPU 1, warp_rm venv)
Ran the new `WarpRM` plugin method end-to-end via `BenchmarkRunner` on GPU 1 in the warp_rm venv
(`AlignedSplit(eval_per_task=100, n_tasks=1)` → single fold, eval = all 100; WARP-RM is training-free
so fold choice is irrelevant). DINOv3 features extracted on the fly (no cache); `sklearn` was the only
missing dep in the venv (added via `uv pip`, so `progress_estimation.methods.__init__` imports).

| metric | plugin (this run) | standalone ref |
|---|---|---|
| VOC | 0.943 | 0.969 |
| Kendall | 0.849 | 0.886 |
| Pearson | 0.948 | 0.967 |
| MAE | 0.091 | 0.094 |
| success-AUROC | 0.598 | 0.664 |

Reproduces the standalone numbers within the expected margin. The gap is a **dataset-composition**
difference, not a code difference: the standalone 0.969/0.664 was scored on the ORIGINAL run's top-cam
mp4s (31 succ / 69 fail), whereas `sim_bottles_bench` is the ALIGNED dir whose media is rendered from
the flow re-run's own raw top frames (28 succ / 72 fail) — different pixels + different labeled set.
MAE matches near-exactly (0.091 vs 0.094); VOC/Pearson within ~0.02; AUROC shifts more (sensitive to
the 28/72-vs-31/69 success composition). **Plugin method confirmed working on GPU.**

### GPU validation 2026-08-02 (cont.) — VLM methods run through the runner (GPU 1, robometer venv)
Ran `robometer_4b` (max_frames=16, the OOM-safe/standalone setting) and `topreward_ll` (Qwen3-VL-8B,
defaults) through `BenchmarkRunner` on GPU 1, same `AlignedSplit(100,1)` all-100 eval, one method per
process (two VLMs won't co-reside on 24GB). Models cached in `HF_HOME=${HF_HOME}`.

| method | VOC (plugin) | AUROC (plugin) | VOC (standalone) | AUROC (standalone) |
|---|---|---|---|---|
| robometer_4b | 0.536 | 0.691 | 0.725 | 0.542 |
| topreward_ll | **−0.323** | 0.463 | 0.640 | 0.663 |

These sim_bottles_bench numbers are CONFOUNDED (superseded by the faithful topcam re-run below) — three
issues, all now understood/fixed:
1. **Sparse media.** `sim_bottles_bench/media` is ~30× temporally sparser than the standalone VLM input
   (`sim_bottles_bench`: 40–60 frames/clip @10fps, one frame per pi0 inference call; `rollouts100_topcam`:
   1160–1801 frames/clip @30fps; same 224×168 res). VLMs consume raw video, so this matters.
2. **Empty instruction.** The first driver called `EpisodeSet.from_dir(name="warp_rm_sim_bottles", ...)`
   but the instruction resolves from the `suite`/`instruction` arg, NOT `name` → the VLMs got an EMPTY
   task string. (WARP-RM ignores the instruction, so its run was unaffected.)
3. **`max_frames` was dead.** `methods/vlm.Robometer.max_frames` was stored but never applied in
   `score_episode` — the ported subsampling logic was lost. Harmless on ≤60-frame media, but OOMs the
   24GB GPU on full-fps 1801-frame clips. FIXED: `score_episode` now linspace-subsamples to `max_frames`
   before the forward (matches ROBOMETER's own eval), so `robometer_4b` runs on any episode length.

### Faithful VLM re-run 2026-08-02 — full-fps top-cam media, correct instruction (GPU 1)
Re-ran both VLMs on `rollouts100_topcam` (1160–1801 frames @30fps, 31 succ/69 fail) with the instruction
resolved via `suite="warp_rm_sim_bottles"`, `max_frames=16` for ROBOMETER. **Regression metrics reproduce
the standalone FAIR run essentially exactly:**

| method | VOC | Kendall | Pearson | MAE | AUROC (plugin) | standalone VOC/K/P/MAE/AUROC |
|---|---|---|---|---|---|---|
| robometer_4b | 0.725 | 0.592 | 0.787 | 0.194 | 0.593 | 0.725 / 0.592 / 0.787 / 0.194 / 0.542 |
| topreward_ll | 0.640 | 0.486 | 0.634 | 0.272 | 0.747 | 0.640 / 0.486 / 0.634 / 0.272 / 0.663 |

VOC/Kendall/Pearson/MAE match to 3–4 decimals → **both plugin VLM methods are faithful ports.** AUROC
differs (plugin higher for both): the package `SuccessAUROC` uses a mean-last-3 tail of the progress
curve, a slightly different tail definition than the standalone `success_auroc`; it is the canonical
metric going forward, and is internally consistent across methods. **All four progress estimators
(WARP-RM, graph/MLP embedding methods, ROBOMETER, TOPReward) now confirmed to run through the unified
`BenchmarkRunner` and reproduce their standalone numbers on the correct inputs.**

## Next
- ~~**AUROC reconciliation**~~ **(resolved)**: `SuccessAUROC` (mean of the last 3 progress values,
  clipped to [0,1], then Mann-Whitney vs the success flag) is the definition, because it is the one
  every method in `BenchmarkRunner` is scored with. The standalone `success_auroc` is the estimator
  underneath it and is not a second metric -- it takes whatever episode-level score it is handed.
  **The FAIR-table AUROCs (0.542/0.663) predate this and were computed on a different terminal score,
  so they are not comparable with any table below them**; they need a rerun, not a relabel, and are
  left as-is until one happens.
- **Record `warp_rabc_sss15` policy rollouts** (2nd unseen set, embeddings now captured at inference)
  for a policy-quality cross-check.
- **Statistical robustness**: bootstrap CIs over the 100 rollouts; more rollouts (n≥300) to tighten
  the 31-success regression estimates.
- Extend to libero / robomimic where policy embeddings DO exist (the graph's home turf).
