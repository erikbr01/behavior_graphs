"""Score the selected configurations on val and on the reserved test episodes.

The gap between the two measures selection optimism directly: the val figure is
the maximum of many scores measured on those episodes, so it carries the
winner's curse, while the test figure does not.

Every job's labels already cover all episodes, so this is a re-score rather than
a re-run. What it spends is the test set: once those episodes have reported a
number they are no longer clean, and any later change to the pipeline puts you
back where you started.
"""

from __future__ import annotations

import json

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.report.ablation import _default_from_plateau
from behavior_graphs.ablation_study.scoring import Scorer


def run(cfg: DatasetConfig, **_) -> str:
    rows = C.load(cfg)
    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    tol = cfg.evaluation.tolerance
    sel = cfg.evaluation.criterion

    targets: list[tuple[str, C.Cell]] = [
        ("grid argmax", max(rows, key=lambda c: c.score)),
        ("recommended default", _default_from_plateau(rows, cfg.evaluation.noise_floor)),
    ]
    for layer in cfg.representations:
        b = C.best(rows, layer=layer)
        if b is not None:
            targets.append((f"{layer}: own best", b))

    out = [
        f"{cfg.name}: val {len(named[splits.VAL])} episodes, "
        f"test {len(named[splits.TEST])} episodes",
        "",
        f"{'configuration':>24} {'clu':>8} {'nc':>4} {'ms':>4} "
        f"{'val':>8} {'test':>8} {'drop':>8} {'unif':>7} {'head':>7} {'dens':>6}",
    ]
    seen: set[tuple[str, str]] = set()
    payload = []
    for name, cell in targets:
        key = (cell.dir, cell.min_segment)
        if key in seen:
            continue
        seen.add(key)
        v = scorer.score_cell(cell, named[splits.VAL], tol)
        t = scorer.score_cell(cell, named[splits.TEST], tol)
        unif = t[C.companion_key(sel, "uniform")]
        head = (t[sel] - unif) / max(1e-9, 1 - unif)
        out.append(
            f"{name:>24} {cell.clusterer:>8} {str(cell.n_components):>4} "
            f"{cell.min_segment:>4} {v[sel]:>8.4f} {t[sel]:>8.4f} "
            f"{t[sel] - v[sel]:>+8.4f} {unif:>7.4f} {head:>7.3f} {t['density']:>6.2f}"
        )
        payload.append(
            {
                "name": name,
                "dir": cell.dir,
                **{a: cell.axis(a) for a in C.AXES},
                "val": v[sel],
                "test": t[sel],
                "test_uniform": unif,
                "test_headroom": head,
                "test_density": t["density"],
            }
        )

    dest = cfg.sweep_dir / "heldout_scores.json"
    dest.write_text(json.dumps(payload, indent=2, default=float))
    out += ["", f"wrote {dest}"]
    return "\n".join(out)
