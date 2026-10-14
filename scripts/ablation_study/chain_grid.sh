#!/usr/bin/env bash
# Run one grid over a list of tasks on one GPU, then read each task.
#
#   chain_grid.sh <gpu> <wait_pid|0> <grid> <layers,csv> <task,csv> [report_probe]
#
# Generic replacement for chain_nn_sweep.sh / chain_nn_dinov3.sh /
# chain_episode_norm.sh, which were three copies of this differing only in the
# grid name and the report probe. A fourth copy was the alternative.
#
# Sequential within a GPU because RAM is the constraint, not the card: a dinov3
# feature block is 17-27 GB resident and deliberately not cached, and a
# whiteboard core rescore peaked at 38.8 GB. Two of these per GPU would risk
# the OOM killer on a 124 GB box.
set -uo pipefail

GPU="${1:?usage: chain_grid.sh <gpu> <wait_pid|0> <grid> <layers> <tasks> [probe]}"
WAIT_PID="${2:?}"
GRID="${3:?}"
LAYERS="${4:?}"
TASKS_CSV="${5:?}"
PROBE="${6:-}"
# Representations run concurrently. 1 is the safe default; RAM is the limit,
# not the card -- a dinov3 block is 17-27 GB resident and is deliberately not
# cached, so 2 needs ~55 GB free before it is worth it.
WORKERS="${7:-1}"

W=<repo>/.claude/worktrees/wgo-bench-droid
DATA=${DATA_PATH}
PY=python
LOG=$DATA/${GRID}_gpu${GPU}.log

cd "$W"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="$GPU"

say() { echo "[$GRID gpu$GPU $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Wait on an explicit pid, never a `pgrep -f` pattern: that pattern matches the
# very command used to check it, which has hung this three times.
if [ "$WAIT_PID" != "0" ]; then
  say "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "  it exited"
fi

IFS=, read -r -a TASKS <<< "$TASKS_CSV"
say "=== $GRID on ${TASKS[*]}, layers $LAYERS ==="

failed=()
for ds in "${TASKS[@]}"; do
  say "sweep $ds"
  if "$PY" scripts/ablation_study/study.py sweep \
      --dataset "$ds" --grid "$GRID" --layers "$LAYERS" --workers "$WORKERS" >> "$LOG" 2>&1; then
    say "  sweep $ds OK"
  else
    say "  sweep $ds FAILED (rc $?)"
    failed+=("$ds")
    continue
  fi
  if [ -n "$PROBE" ]; then
    say "report $ds"
    "$PY" "scripts/ablation_study/probes/$PROBE" "$ds" > "$DATA/${GRID}_${ds}.txt" 2>> "$LOG" \
      && say "  report -> $DATA/${GRID}_${ds}.txt" || say "  report $ds FAILED"
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  say "=== done, FAILED: ${failed[*]} ==="
else
  say "=== done, all ${#TASKS[@]} tasks OK ==="
fi

# These cells vary dials the core pins, so without this check they would look
# like core cells to anything reading the sweep dir. `core.PINNED_DIALS` now
# rejects them at re-score time; this reports the count so it is visible.
say "contamination check"
"$PY" scripts/ablation_study/probes/check_contamination.py >> "$LOG" 2>&1 \
  && say "  ran" || say "  FAILED"
