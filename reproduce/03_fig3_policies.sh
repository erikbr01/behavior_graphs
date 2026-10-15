#!/usr/bin/env bash
# Figure 3 -- retrained-policy success rates, Beta posteriors, letters, p-values.
#
# Needs: $DATA_PATH/rollouts/<collection>/ for the baseline and retrained arms
# Time:  a minute or two (pairwise exact tests over pooled counts)
#
# Writes the `policies` section of results/paper_results.json, which is what the
# website serves. Pairwise Barnard tests with family-wise error control run here
# so the page does no statistics of its own.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_PATH:?export DATA_PATH to your data root}"
PYTHONPATH="$PWD/src" exec "${PYTHON:-python3}" scripts/export_paper_results.py --sections policies "$@"
