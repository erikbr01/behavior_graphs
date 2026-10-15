#!/usr/bin/env bash
# Every reproduction step, cheapest first, continuing past failures so one
# missing input does not hide the rest. Prints a summary table at the end.
#
#   DATA_PATH=... STUDY_RESULTS_DIR=... ./reproduce/run_all.sh
#
# PYTHON may point at an interpreter with the extras installed (see REPRODUCE.md).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
declare -a NAME STATUS

step() {
  local label=$1; shift
  printf '\n\033[1m== %s ==\033[0m\n' "$label"
  if "$@"; then NAME+=("$label"); STATUS+=("reproduces"); else
    local rc=$?
    NAME+=("$label"); STATUS+=("FAILED (exit $rc)")
  fi
}

step "preflight"                 bash reproduce/00_preflight.sh
step "Table III -- ablation"     bash reproduce/01_table3_ablation.sh
step "Table II  -- progress"     bash reproduce/02_table2_progress.sh verify
step "Figure 3  -- policies"     "$PY" reproduce/verify_against_published.py policies
step "Figure 4  -- study answers" "$PY" reproduce/verify_against_published.py study
step "Table I   -- study effects" "$PY" reproduce/check_study_claims.py

printf '\n\033[1m== summary ==\033[0m\n'
for i in "${!NAME[@]}"; do printf '  %-26s %s\n' "${NAME[$i]}" "${STATUS[$i]}"; done
for s in "${STATUS[@]}"; do [[ $s == FAILED* ]] && exit 1; done
exit 0
