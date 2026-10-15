# scripts/

Top level holds the **canonical entry points** — the ones docs, configs and habit
point at. Everything else is grouped by what it is for.

| entry point | purpose |
| --- | --- |
| `collect_rollouts.py` | Collect policy rollouts with embeddings in the canonical layout. |
| `train_policy.py` | Train a policy for one environment, delegating to its submodule. |
| `run_sweep.py` | Pipeline sweep (hydra entry point). |
| `label_nodes.py` | Standalone VLM node labeler for an existing run directory. |
| `generate_report.py` | Self-contained HTML report for a labeled behavior-graph run. |
| `export_paper_results.py` | Write `results/paper_results.json` from the artifacts the paper's numbers come off. |
| `setup_env.sh` | Write `.env` with local paths. Re-run after moving the data directory. |

## Progress estimation (Tab. II)

| entry point | purpose |
| --- | --- |
| `run_progress_bench.py` | One dataset, every method and metric. |
| `sweep_progress_allds.py` | The whole zoo (`configs/progress_estimation/data/zoo.yaml`). |
| `bench_allds_with_vlms.py` | The same, including the VLM baselines (Robometer, TOPReward). |
| `merge_progress_sweep.py`, `consolidate_progress_results.py` | Combine per-dataset results into one table. |
| `plot_progress_traces.py`, `render_progress_videos.py` | Per-episode progress curves, as plots and as overlaid video. |
| `demo_fit_progress.py` | Fit the stage schedule on demonstrations and score rollouts through it. |

## Subdirectories

- **`ablation_study/`** — the paper's ablation (Tab. III).
  `paper_ablation.py` prints the table and verifies it against the published
  values; `study.py` is the general report CLI over a finished sweep;
  `build_gripper_gt.py` cuts the gripper-derived subtask ground truth;
  `build_dataset.py`, `build_droid_state.py`, `extract_*.py` prepare the
  representations each arm sweeps over; `launch_sweep.sh` and the `chain_*.sh`
  drivers run the grid.
- **`analysis/`** — the paper's statistics. `user_study_mixed_effects.py` fits
  the mixed-effects model behind Tab. I. Fig. 3's compact letter display comes
  from `behavior_graphs.stats.policy_comparison` via `export_paper_results.py`,
  not from here.
- **`utils/`** — data plumbing: the DROID conversion/merge/trim pipeline, the
  SAFE dataset importers, trajectory and video featurization, and sweep-artifact
  housekeeping.
