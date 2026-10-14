"""Merge the policy-embedding sim re-run into the full cross-dataset sweep so the
whole table uses each dataset's POLICY EMBEDDING (never raw simulator state).
LIBERO + safe_* already used policy embeddings in the base run; the sim families
are overridden from the re-run. Regenerates the combined table.

    python scripts/merge_progress_sweep.py \
        --base analysis/out/progress_allds/sweep_results.json \
        --override analysis/out/progress_allds_polemb/sweep_results.json \
        --out analysis/out/progress_allds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_graphs.progress_estimation.report import sweep_table_md  # noqa: E402

_TITLE = (
    "Cross-dataset progress estimation (graph on POLICY EMBEDDINGS, identity, "
    "K chosen per fold on inner validation, 3-fold)"
)
_SUBTITLE = (
    "Every dataset uses its policy-embedding layer (same modality the VLM comparison "
    "contrasts against pixels), NOT raw simulator state. graph:stage = SARM stage "
    "progress; graph:absorb = P(absorb@SUCCESS); frame_index = trivial linear baseline."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--override", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged = dict(json.loads(args.base.read_text()))
    merged.update(json.loads(args.override.read_text()))  # policy-embedding re-run wins

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep_results.json").write_text(json.dumps(merged, indent=2))
    table = sweep_table_md(merged, title=_TITLE, subtitle=_SUBTITLE, show_emb=True)
    (args.out / "sweep_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
