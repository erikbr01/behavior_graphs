#!/usr/bin/env bash
# Checks everything the other scripts need, and says what is missing rather
# than failing halfway through a long run. Read-only; safe to run any time.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
ok=0; miss=0
say()  { printf "  %-46s %s\n" "$1" "$2"; }
PY=${PYTHON:-python3}
have() { if eval "$2" >/dev/null 2>&1; then say "$1" "OK"; ok=$((ok+1)); else say "$1" "MISSING"; miss=$((miss+1)); fi; }

echo "== interpreter and package =="
have "python >= 3.10"                 "$PY -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)'"
have "behavior_graphs importable"     "PYTHONPATH=$REPO/src $PY -c 'import behavior_graphs'"
have "numpy / pandas / yaml"          "PYTHONPATH=$REPO/src $PY -c 'import numpy,pandas,yaml'"

echo "== optional extras (only some steps need these) =="
have "analysis extra (bambi, arviz)"  "$PY -c 'import bambi,arviz'"
have "viz extra (streamlit, plotly)"  "$PY -c 'import streamlit,plotly'"

echo "== data roots =="
: "${DATA_PATH:=}"
if [ -z "$DATA_PATH" ]; then
  say "DATA_PATH" "UNSET -- export it to your data root"; miss=$((miss+1))
else
  say "DATA_PATH=$DATA_PATH" "$([ -d "$DATA_PATH" ] && echo OK || echo 'NOT A DIRECTORY')"
  have "  rollouts/"                  "[ -d '$DATA_PATH/rollouts' ]"
  have "  sweeps/"                    "[ -d '$DATA_PATH/sweeps' ]"
fi
: "${STUDY_RESULTS_DIR:=}"
if [ -z "$STUDY_RESULTS_DIR" ]; then
  say "STUDY_RESULTS_DIR" "UNSET (only step 04 needs it)"
else
  have "STUDY_RESULTS_DIR"            "[ -d '$STUDY_RESULTS_DIR' ]"
fi

echo "== per-step inputs =="
have "Tab III: ablation rescored.json" \
  "ls '$DATA_PATH'/sweeps/*settable*/rescored.json '$DATA_PATH'/sweeps/*preparetray*/rescored.json '$DATA_PATH'/sweeps/*cooking*/rescored.json"
have "Fig 3: scored rollout collections" "ls '$DATA_PATH'/rollouts/set_table_* "
have "Checked-in reported values"     "[ -f src/behavior_graphs/results/paper_results.json ]"

echo
echo "  $ok present, $miss missing."
[ "$miss" -eq 0 ] && echo "  Everything needed is here." \
  || echo "  See REPRODUCE.md for what each missing item gates."
exit 0
