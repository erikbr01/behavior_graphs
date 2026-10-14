"""Held-out ablation: change one axis of the default and leave the rest fixed.

Selection stays on the val episodes; the test episodes only report. Choosing the
best level per axis on the test set would be selecting on it.

Two views. Varying one axis around a fixed default isolates that axis, which is
what an ablation is for. Taking the best configuration containing each level is
the more common table format but conflates axes, since every level's argmax is a
different configuration -- that is how n_components=100 can appear to be worth
0.665 when the number really belongs to nc=100 plus a different clusterer.
"""

from __future__ import annotations

import json

import numpy as np

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.scoring import Scorer

_AXES = C.AXES


def _default_from_plateau(rows: list[C.Cell], se: float) -> C.Cell:
    """The recommended default: simplest member of the within-one-SE plateau."""
    rank = {"kmeans": 0, "gmm": 1, "hsmm": 2, "hdbscan": 3}
    return min(
        C.plateau(rows, se),
        key=lambda c: (
            rank.get(c.clusterer, 9),
            0 if c.smoothing == "none" else 1,
            c.n_components or 0,
            float(c.min_segment),
        ),
    )


def run(cfg: DatasetConfig, best_achievable: bool = False, **_) -> str:
    rows = C.load(cfg)
    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    tol = cfg.evaluation.tolerance
    star = _default_from_plateau(rows, cfg.evaluation.noise_floor)

    out = [
        f"{cfg.name}: held-out ablation on {len(named[splits.TEST])} test episodes "
        f"(selection used {len(named[splits.VAL])})",
        "  default: " + ", ".join(f"{a}={star.axis(a)}" for a in _AXES),
        "",
    ]

    base = scorer.score_cell(star, named[splits.TEST], tol)
    out += [
        f"  default scores {base[cfg.evaluation.criterion]:.4f} against a uniform "
        f"reference of {base[C.companion_key(cfg.evaluation.criterion, 'uniform')]:.4f} "
        f"at density {base['density']:.2f}",
        "",
        f"{'axis':>14} {'level':>14} {'F1':>8} {'rho':>6} {'vs default':>11}",
    ]
    payload: dict[str, list[dict]] = {}
    for axis in _AXES:
        payload[axis] = []
        for lvl in C.levels(rows, axis):
            match = [
                c
                for c in rows
                if str(c.axis(axis)) == lvl
                and all(str(c.axis(a)) == str(star.axis(a)) for a in _AXES if a != axis)
            ]
            if not match:
                continue
            sc = scorer.score_cell(match[0], named[splits.TEST], tol)
            f1 = sc[cfg.evaluation.criterion]
            mark = " *" if lvl == str(star.axis(axis)) else ""
            out.append(
                f"{axis:>14} {lvl:>14} {f1:>8.4f} {sc['density']:>6.2f} "
                f"{f1 - base[cfg.evaluation.criterion]:>+11.4f}{mark}"
            )
            payload[axis].append(
                {
                    "level": lvl,
                    "f1": f1,
                    "density": sc["density"],
                    "delta": f1 - base[cfg.evaluation.criterion],
                }
            )

    if best_achievable:
        out += [
            "",
            "Best achievable per level (each level's own argmax; conflates axes)",
            f"{'axis':>14} {'level':>14} {'val':>8} {'test':>8} {'drop':>8}",
        ]
        for axis in _AXES:
            for lvl in C.levels(rows, axis):
                grp = [c for c in rows if str(c.axis(axis)) == lvl]
                if not grp:
                    continue
                pick = max(grp, key=lambda c: c.score)
                sc = scorer.score_cell(pick, named[splits.TEST], tol)
                f1 = sc[cfg.evaluation.criterion]
                out.append(
                    f"{axis:>14} {lvl:>14} {pick.score:>8.4f} {f1:>8.4f} {f1 - pick.score:>+8.4f}"
                )

    dest = cfg.sweep_dir / "ablation_heldout.json"
    dest.write_text(json.dumps(payload, indent=2, default=float))
    out += ["", f"wrote {dest}   (* marks the default level)"]
    _ = np  # numpy kept available for callers extending this table
    return "\n".join(out)
