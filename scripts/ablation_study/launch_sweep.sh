#!/usr/bin/env bash
# Run the ablation grid for one dataset, detached.
#
#   launch_sweep.sh bottles dinov3        # one representation
#   launch_sweep.sh legos                 # every representation in its config
#   launch_sweep.sh bottles "" 3          # all reps, three layers at a time
#   launch_sweep.sh legos "" 3 ablation_droid  # fill the complete balanced design
#
# study.py itself only builds the command and shells out to the cuml-sweep
# interpreter, so it can run under the project venv.
set -euo pipefail

DATASET="${1:?usage: launch_sweep.sh <dataset> [layers,csv] [workers] [grid]}"
LAYERS="${2:-}"
# Representations to run at once. Default 1 keeps the old behaviour; the useful
# setting is one worker per layer, bounded by RAM rather than cores because each
# holds its own features and windows (legos ~22 GB a worker, bottles ~7 GB).
WORKERS="${3:-1}"
# Override the per-representation grid mapping for every arm. `ablation_droid` fills
# the complete balanced design, which every arm has to run on the same dials --
# including `state`, whose own grid uses n_components the others never see.
GRID="${4:-}"
VENV=python
# The layers go in the filename, and the redirect appends. Several arms of one
# dataset run concurrently, and a single `${DATASET}_sweep_launch.log` opened
# with `>` meant the second launch truncated the first arm's log out from under
# a process that was still writing to it -- which is how the bottles dinov3
# sweep lost its output to the rgrl re-sweep.
TAG="${GRID:+${GRID}_}${LAYERS:+${LAYERS//,/-}_}"
LOG="${DATA_PATH}/${DATASET}_sweep_${TAG}launch.log"
# One pidfile per arm, in a directory `chain_report.sh` scans. A pattern like
# `pgrep -f "study.py sweep"` cannot be used to tell whether a dataset's arms
# have finished: it matches any shell whose command line mentions the phrase,
# including the commands used to check.
PIDDIR=${DATA_PATH}/sweep_pids
mkdir -p "$PIDDIR"

export PYTHONPATH=src

ARGS=(scripts/ablation_study/study.py sweep --dataset "$DATASET" --workers "$WORKERS")
[ -n "$LAYERS" ] && ARGS+=(--layers "$LAYERS")
[ -n "$GRID" ] && ARGS+=(--grid "$GRID")

setsid nohup "$VENV" "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
SWEEP_PID=$!
echo "$SWEEP_PID" > "$PIDDIR/${DATASET}_${TAG:-all}.pid"

echo "launched $DATASET sweep${LAYERS:+ (layers $LAYERS)}, pid $SWEEP_PID, log $LOG"
