#!/usr/bin/env bash
# Table I + Figure 4 -- the user study.
#
# Needs: $STUDY_RESULTS_DIR (the response store) and the `analysis` extra
#        (bambi, arviz) for the mixed-effects fit:  uv pip install -e ".[analysis]"
# Time:  minutes (four MCMC chains)
#
# Table I is the adjusted effect per outcome (estimated marginal means per
# display and their difference); Figure 4 is the raw ratings, choices and regret.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STUDY_RESULTS_DIR:?export STUDY_RESULTS_DIR to the response store}"
"${PYTHON:-python3}" -c 'import bambi, arviz' 2>/dev/null || {
  echo "the analysis extra is not installed: uv pip install -e \".[analysis]\"" >&2; exit 2; }
PYTHONPATH="$PWD/src" exec "${PYTHON:-python3}" scripts/export_paper_results.py --sections study,adjusted "$@"
