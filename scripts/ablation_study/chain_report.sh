#!/usr/bin/env bash
# Wait for every sweep arm of one dataset, then re-score and write the reports.
#
#   chain_report.sh bottles                 # wait on the pidfiles under sweep_pids/
#   chain_report.sh legos 234526 387056     # plus these already-running pids
#
# Re-scoring has to come after the last arm, not after each one: `rescored.json`
# is written wholesale, so a run that starts while an arm is still going records
# a partial grid and the comparison silently comes out over whatever had
# finished. The extra pid arguments exist for arms launched before pidfiles were
# being written.
#
# The reports land under experiments/<dataset>/ as well as stdout, so the
# numbers in the paper have a file behind them rather than a scrollback.
set -uo pipefail

DATASET="${1:?usage: chain_report.sh <dataset> [extra_pids...]}"
shift
EXTRA=("$@")

W=<repo>/.claude/worktrees/wgo-bench-droid
DATA=${DATA_PATH}
PIDDIR=$DATA/sweep_pids
PY=python
LOG=$DATA/${DATASET}_report_chain.log
OUT=$W/experiments/$DATASET

cd "$W"
export PYTHONPATH=src

say() { echo "[report:$DATASET $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

live() {
  local n=0 p
  for f in "$PIDDIR/${DATASET}_"*.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null) || continue
    kill -0 "$p" 2>/dev/null && n=$((n + 1))
  done
  for p in ${EXTRA[@]+"${EXTRA[@]}"}; do
    kill -0 "$p" 2>/dev/null && n=$((n + 1))
  done
  echo "$n"
}

say "waiting for sweep arms ($(live) live, plus ${#EXTRA[@]} passed in)"
while [ "$(live)" -gt 0 ]; do sleep 120; done
say "all arms idle"

# The rgrl chain may still be between its extraction and its sweep launch, in
# which case an arm is about to appear that `live` cannot see yet. Wait it out.
if [ -f "$DATA/${DATASET}_rgrl_chain.out" ] &&
  ! grep -q "chain complete\|stopping" "$DATA/${DATASET}_rgrl_chain.out"; then
  say "the rgrl chain has not finished handing off; waiting for it"
  while ! grep -q "chain complete\|stopping" "$DATA/${DATASET}_rgrl_chain.out"; do sleep 120; done
  say "  handed off; re-checking arms"
  while [ "$(live)" -gt 0 ]; do sleep 120; done
fi

say "rescore"
"$PY" scripts/ablation_study/study.py rescore --dataset "$DATASET" >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"
if [ "$rc" -ne 0 ]; then
  say "rescore failed; not writing reports off a stale artefact"
  exit 1
fi

say "reports -> $OUT"
mkdir -p "$OUT"
"$PY" scripts/ablation_study/study.py report --dataset "$DATASET" --all --out "$OUT" >> "$LOG" 2>&1
say "  exit=$?"
say "chain complete"
