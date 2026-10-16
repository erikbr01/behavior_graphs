# Public-release audit

Working document for the public release. **Delete before publishing.**

Built from `policy_graphs` branch `abc-bottles-boundary-study` @ `9eaf473` (a
strict superset of `main` @ `1f674dc`, and the only place the paper's ablation
code lives) plus `corl_2026_behavior_graphs` @ `38af031` under `paper/`.

The rule applied: **keep what the paper reports or what the website demo the
paper links to needs; remove everything else.** 1091 tracked files -> 534.

Verification after every removal: `pytest` (1794 passed, 20 skipped),
`ruff check` / `ruff format` clean, every module imports, and all three app
entrypoints start under `AppTest`.

---

## 1. What the paper covers (the keep set)

| Paper element | Code |
|---|---|
| SIII-A construction (embed -> window -> reduce -> cluster -> collapse -> graph -> VLM label) | `stages/{features,windowing,reduction,clustering,graph,labeling}`, `pipeline/`, `core/` |
| Representations: policy embeddings, proprioception, DINOv3 | `stages/features/{policy_embedding,state_action,state,vision_backbone,concat}.py` |
| Reductions: UMAP (+ cuML UMAP), PCA, identity | `stages/reduction/{umap,cuml_umap,pca,identity,normalize}.py` |
| Clustering: K-means | `stages/clustering/kmeans.py` |
| Graph: first-order transitions, sticky smoothing, prefix tree, branch folding | `stages/graph/{transition,smoothing,simplify}.py`, `viz/plotly/trajectory_tree.py`, `viz/simplification.py` |
| VLM annotation via contact sheets, two-pass | `stages/labeling/vlm_labeler.py`, `vlm/adapters.py`, `scripts/label_nodes.py` |
| Eq. 6-7 success and path likelihood | `stages/evaluation/values.py`, `user_study/path_explorer.py` |
| Interactive graph, node/edge video playback | `viz/streamlit/`, `apps/{demo,study,reviewer}` |
| Fig. 3 retrained-policy success rates, Beta posteriors, CLD, FWER | `stats/policy_comparison.py`, `scripts/export_paper_results.py` |
| SIV-A user study as run | `user_studies/variants/three_stage/`, `user_study/`, `apps/study` |
| Fig. 4 + Tab. I mixed-effects, EMMs, regret | `user_study/{mixed_effects,regret,figure_export,table_export}.py` |
| SIV-B progress estimation + Tab. II (43 datasets) | `progress_estimation/`, `configs/progress_estimation/`, `scripts/run_progress_bench.py` |
| SIV-C ablation + Tab. III | `ablation_study/`, `stages/evaluation/{_boundary_metrics,subtask_boundaries}.py`, `annotations/gripper.py`, `scripts/ablation_study/paper_ablation.py` |
| Six policy classes on the website | `envs/{robomimic,mimicgen,droid,libero,libero_pro,vla_foundry,rgrl_env}.py`, `envs/safe_import.py` |
| The reported numbers, as data | `results/paper_results.json`, `scripts/export_paper_results.py` |

`configs/progress_estimation/data/zoo.yaml` now sums to exactly the paper's 43:
LIBERO 6, Robomimic 10, MimicGen 8, real-world 19.

---

## 2. What was removed

### A. Rollout outcome classification / failure detection
Five graph predictors, two literature baselines, SAFE cross-benchmark eval,
calibrated/early-detection views. The result was negative (a graph-free
episode-mean control beat the graph's contribution) and nothing from it is in
the paper. `outcome_classifier/` (30 files), `configs/outcome_classifier/`,
`scripts/outcome_classifier/`, the demo's Failure Detection page, two docs, four
analysis scripts, eight tests.

Kept: `envs/safe_import.py` + `configs/data/safe.yaml` -- the SAFE datasets are
where the paper's OpenVLA and pi0-FAST graphs come from. Only their
outcome-classifier use went.

### B. Demonstration curation / weighted BC / rgrl campaigns
The MimicGen curation layer (E1-E25), demo scoring, weighted behaviour cloning,
the sim-bottles RABC arm. `curation/`, `scripts/{curation,rabc}/`, the
noisy-pool and salvage dataset builders, `inject_stalls.py`, three experiment
logs.

Kept, after first removing it by mistake: `envs/rgrl_env.py`. rgrl is the
flow-matching policy stack the paper's real-robot results run on, and its
`eval_policy` is where Fig. 3's success rates come from -- it is an env adapter,
not part of the curation campaign that shares its name.

### C. WARP-RM progress baseline
A fourth baseline, staged and then paused on a gated DINOv3 download. The paper
reports three (MLP, Robometer, TOPReward). Method, env adapter, five extraction
scripts, the patch set, two tests, and the `warp_rm` zoo family.

### D. Markov-order and trie-vs-graph analyses
The investigation behind the paper's one paragraph about prefix trees, which
reports no experiment. Six analysis scripts, the `markov_test` evaluator and its
`markov_violation` metric, the sweep page's per-cluster drill-down.

### E. VLM chat agent and MCP server
`chat/`, `mcp/`, `apps/mcp_server/`, the demo's Chat page, `configs/chat/`.
The shared VLM model list moved to `configs/vlm/models.yaml`, since the node
labeler reads it too.

### F. Pipeline stages the paper never uses
Clusterers `{adaptive_kmeans, gmm, hdbscan, hsmm, _edhmm, knn, spectral_knn}`;
reducers `{sae, vae, _temporal, _torch}`; the state-merging graph builder
(ALERGIA + outcome-support guard) and the demo's "State merging" mode;
data-support retrieval, end to end (stage, `core/retrieval.py`, the artifact
read/write path, and the tree views' "Data support" colouring); the `outcome_mi`
evaluator. Exploratory grids: the SAE/HSMM/smoothing/temporal/DINO-scan set and
`standard.yaml`, which *was* the SAE/HSMM exploration grid.

Retiring SAE/VAE/HSMM emptied the `-m fit` pytest tier, so the marker is gone
and `docs/testing.md` is rewritten around what is left.

Kept, on inspection: `stages/graph/smoothing.py` (the paper's ablation grid
sweeps `smoothing: [none, sticky]`), `reduction/normalize.py` (shared by every
surviving reducer), `features/concat.py` (the DINOv3 arm), and
`stages/subclustering/` -- the demo tree's leaf expansion depends on it, and
removing it would damage a figure the paper shows.

### G. Hand annotation tooling
The paper's ground truth is "gripper opening and closing as a heuristic", not
hand annotation. The span/pill timeline editor, the failure-mode annotation
page, their demo pages, the export script, the doc and six tests.

Kept: `annotations/{gripper,subtasks,store}.py` -- `build_gripper_gt.py` writes
the gripper ground truth *through* the subtask store -- and
`annotations/failure_modes.py`, which the study's failure-mode question reads.

### H. Superseded user-study designs
Three earlier designs preceded `three_stage_v1`: the `order_effect`,
`contrastive_same_page` and `production` renderers, `app_group_{a,b}.py`, their
variant configs, and the sim-era pilot session configs. `apps/study/Home.py` now
serves `three_stage_v1` by default instead of falling back to a random A/B
assignment.

Kept: `variants/contrastive_same_page/questions.py` -- the published variant
imports its question bank.

Also removed: the gift-card raffle, including the email field on the exit page.
Recruitment machinery, and the one place the study collected a contact detail.

### I. Environments outside the 43 datasets / six policy classes
`envs/{molmoact2,robocasa,robocasa365}.py` and their configs. MolmoAct, GR00T
and RoboCasa are none of the six policy classes the paper names, and the zoo's
`other_sim` family is outside the 43. `HFCheckpointPolicy` moved from
`molmoact2.py` to `envs/_hf_policy.py`, which `vla_foundry` needs.

### J. Progress-estimation extras outside Tab. II
The `graph_absorb`, `robometer_fft`, `topreward_ll` and `dummy` method configs;
the efficiency sweep, the wall-clock control and the bootstrap script.

Kept: `frame_index`, the duration-is-a-label-proxy control. Tab. II omits it,
but it is the sanity baseline that catches a dataset where episode length alone
predicts the outcome.

### K. Legacy, fleet and one-off analyses
The policy_doctor importers, `fleet.sh` + `docs/model_fleet.md`, the
graph-quality-tier and why-length probes.

### L. Submission-video pipeline
`scripts/video/` (18 files). The manim pipeline for the supplementary video.

### M. Production deploy stack and ops
GCP VM creation, the Cloudflare tunnel, nginx templates, staging/workstation
compose, `push_deploy.sh`, the 60 KB ops runbook, `scripts/ops/`,
`docs/app_performance.md`, `.claude/`.

`deploy/` is now a Dockerfile plus one compose file that runs the demo on :8501
and the study on :8502 against a bind-mounted data root, with a matching
`.env.example`. GCS response mirroring stays as an opt-in `gcs` extra.

### N. The ablation branch's exploratory layer
~70 probe scripts, the `wgo_bench` package and its 23 scripts, seven ABC tasks
the paper never mentions (bottles, chess, earbuds, key, legos, surgical, towels,
whiteboard), 24 exploratory grids, and the per-task experiment logs. The
`FEATURE_NORMS` / `normalize_episode` helpers that
`stages/features/episode_norm.py` imported from `wgo_bench` are inlined there.

Kept: the whole `ablation_study/` package, the three DROID task configs, and the
two grids they name (`ablation`, `ablation_droid`).

### O. Paper directory
`corl_archive/` (an unused CoRL template) and the 18 superseded draft figures,
leaving the five `ieee.tex` actually includes.

---

## 3. Also done

- **Machine paths parameterized.** 57 files carried `/mnt/ssdB/erik/...` or
  `/home/erbauer/...`; those are now `${DATA_PATH}` / `${DATA_ROOT}` /
  `${OUTPUT_PATH}` / `$HOME`. The cuML sweep interpreter defaults to
  `sys.executable` instead of a hard-coded venv.
- **Package renamed** `policy_graphs` -> `behavior_graphs`, imports, hydra
  `_target_`s and `POLICY_GRAPHS_*` -> `BEHAVIOR_GRAPHS_*` env vars included.
- **LICENSE (MIT), CITATION.cff**, and a README rewritten around the paper with
  a reproduce-the-paper table.
- **`tests/test_app_entrypoints.py` added.** `apps/demo/Home.py` was still
  registering four deleted pages, which `st.navigation` turns into a crash on
  every page -- and nothing caught it, because the existing app tests open
  `_pages/graph_demo.py` directly. The new test checks the registry both ways,
  the config files the UI reads by path, and the study variant renderers.

---

## 4. Still open

Current as of the last working session. Items 1-3 are decided; what remains is
listed under "Before publishing" and "Known defects".

**Decided**
- rgrl lineage: `port`. `main` now carries port (#27), source's four unique
  commits (#30) and the sim-success fix (#29); the release pins `main`.
- RoboCasa: kept. Behavior graphs and progress estimation are wanted for it,
  so the adapters, configs and zoo entry were restored after an earlier cut.
- Submodules stay private until everything releases together; data is being
  handled separately.

**Before publishing**
1. **Delete this file.** It names internal dead ends and is a working document.
2. **Private submodules** still make `clone --recurse-submodules` fail for
   anyone outside. This is the hard blocker, deferred by decision, not solved.
3. **`paper/ieee.tex` loads `lipsum`** behind a "remove once placeholders are
   written" comment, and its commented-out `\twocolumn` block references
   `figures/fig1_title.pdf`, which is not in the repo.
4. **`experiments/PROGRESS_ESTIMATION_LOG.md`** (925 lines) reads as an internal
   working log. Keep, trim, or drop.

**Known defects, found but not fixed**
5. **Eval and collection disagree on the same policy.** `eval.py` reads
   `configs/env/robomimic_transport.yaml` (max_steps 700, parallel_envs 25,
   seed 10000); the collector reads `configs/collect_rollouts.yaml`
   (max_timesteps 1000, num_envs 1, seed 0). Measured on one transport_mh
   policy: eval 0.46, collected rollouts 44/50 = 0.88. Both are honest under
   their own settings, but a reader who sees `success_rate 0.46` and then a
   graph with 44 successes will reasonably think something is broken. Aligning
   the collector's rollout settings with the eval env's is a behavioural choice
   about which harness is canonical, so it was left open.
6. **Weighted sampling is now inconsistent across repos.** diffusion_policy
   merged `weighted-bc` (#12); rgrl's equivalent (`weighted-sampling`, #1) was
   closed as conflicting. Decide deliberately rather than by which PR happened
   to be mergeable.
7. **`robometer` and `warp_rm` have never been checked out** here, so their
   pins are recorded but unverified. robometer is a Tab. II baseline.
8. **REPRODUCE.md section 5 (the Figures page) is written from reading the
   code, not from a verified run.** Every other section is backed by an
   actual run; that one is not.

**Environment notes (not shipped, useful to whoever picks this up)**
- The analysis extra lives in a separate venv at
  `/mnt/ssdB/erik/venvs/bg_analysis` rather than the shared one, because pymc
  pins numpy/scipy tightly and other sessions use the shared venv.
- `third_party/rgrl/.venv` here is a symlink to
  `/mnt/ssdB/erik/venvs/bg_rgrl_pinned`; the boot drive is ~93% full.
- `STUDY_RESULTS_DIR` must be `$DATA_PATH/study_results` (84 sessions, 18 on
  the reported layout). A dev checkout's own `study_results/` holds lab
  previews with `origin: lab` and yields zero participants.
- This box has a soft real-time tenant (X-Plane rollouts). Convention agreed
  with the other sessions: announce heavy jobs, `nice -n 19`, yield GPU 0 to
  real-time runs. `ionice` is a no-op here -- both NVMe are `[none]` scheduler
  and the cgroup io controller is not delegated to user slices, so there is no
  user-space I/O throttle. Deferring a write is the only real lever.
