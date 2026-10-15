#!/usr/bin/env bash
# Table II -- progress estimation across 43 rollout collections.
#
# This is the expensive one, and the only result whose intermediate artifacts
# are not carried in this repo. Three ways to engage with it, cheapest first.
#
#   ./02_table2_progress.sh verify   # check the published table is self-consistent
#   ./02_table2_progress.sh one DS   # re-run the benchmark for one dataset
#   ./02_table2_progress.sh all      # re-run the whole zoo (hours; see below)
#
# `verify` needs nothing but the checked-in results. `one` and `all` need
# $DATA_PATH/rollouts plus, for the VLM baselines, the robometer submodule and
# a TOPReward checkout -- see REPRODUCE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
mode=${1:-verify}

case "$mode" in
verify)
  # The export reads $PROGRESS_RESULTS_ROOT/{dataset}_metrics.json. Those are
  # not in this repo, so a recompute-and-diff is only possible where the
  # benchmark has been run. What is always checkable is that the published
  # table is internally consistent: the per-domain means are the means of the
  # per-dataset rows behind them.
  PYTHONPATH="$PWD/src" exec "$PY" reproduce/check_progress_consistency.py
  ;;
one)
  ds=${2:?usage: 02_table2_progress.sh one <dataset-name>}
  : "${DATA_PATH:?export DATA_PATH to your data root}"
  PYTHONPATH="$PWD/src" exec "$PY" scripts/run_progress_bench.py \
    "data.name=$ds" "out=${PROGRESS_RESULTS_ROOT:-progress_results}/$ds"
  ;;
all)
  : "${DATA_PATH:?export DATA_PATH to your data root}"
  echo "The full zoo is 43 collections x 4 methods. The two VLM baselines"
  echo "dominate the cost and need their own checkouts. Expect hours."
  PYTHONPATH="$PWD/src" exec "$PY" scripts/sweep_progress_allds.py "${@:2}"
  ;;
*) echo "usage: $0 [verify|one <dataset>|all]" >&2; exit 2 ;;
esac
