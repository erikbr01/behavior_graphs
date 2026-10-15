#!/usr/bin/env bash
# Table III -- the ablation over representation x reduction.
#
# Needs: $DATA_PATH/sweeps/{abc_settable,abc_preparetray,abc_cooking_v2}/rescored.json
# Time:  seconds (reads the rescored sweep, recomputes nothing)
#
# This does not merely print the table: it checks every value against the one
# published and exits non-zero if any has drifted, so a green run is a
# reproduction rather than a rendering. --latex emits the table body.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_PATH:?export DATA_PATH to your data root (see REPRODUCE.md)}"
PYTHONPATH="$PWD/src" exec "${PYTHON:-python3}" scripts/ablation_study/paper_ablation.py "$@"
