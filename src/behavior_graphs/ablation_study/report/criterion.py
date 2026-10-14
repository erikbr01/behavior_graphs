"""Is raw segmental F1 safe to maximise, or does it need the gain correction?

Raw *boundary* F1 was gameable: point matching within a tolerance rewards extra
predictions whenever base recall is low, which drove every arm to density
2.2-3.0. That is why gain plus a density band existed.

Raw *segmental* F1 is not gameable, because duplicate suppression caps true
positives at the ground-truth count while every surplus prediction still enters
the precision denominator, and a long prediction has low IoU with each segment it
spans. It should therefore be single-peaked in density, and this report checks
that rather than asserting it.

Gain has the opposite pathology, worth seeing side by side: its count-matched
reference is recomputed at each configuration's own density, so it is not a
fixed target and its mean rises at absurd densities where the reference degrades
faster than the method does.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study.config import DatasetConfig

_BINS = (0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, float("inf"))


def run(cfg: DatasetConfig, **_) -> str:
    rows = C.load(cfg)
    out = [
        f"{cfg.name}: {len(rows)} cells, criterion {cfg.evaluation.criterion}",
        "",
        "Score as a function of segment density. Single-peaked means the "
        "criterion cannot be\ngamed in either direction and needs no band.",
        f"{'density':>14} {'n':>6} {'mean score':>11} {'max':>8} {'mean gain':>10} {'max gain':>9}",
    ]
    for lo, hi in zip(_BINS[:-1], _BINS[1:]):
        sub = [c for c in rows if lo <= c.density < hi]
        if not sub:
            continue
        s = np.asarray([c.score for c in sub])
        g = np.asarray([c.gain for c in sub])
        label = f"[{lo:.1f},{hi:.1f})" if np.isfinite(hi) else f"[{lo:.1f},inf)"
        out.append(
            f"{label:>14} {len(sub):>6} {s.mean():>11.4f} {s.max():>8.4f} "
            f"{np.nanmean(g):>+10.4f} {np.nanmax(g):>+9.4f}"
        )

    peak = max(
        (
            (np.mean([c.score for c in rows if lo <= c.density < hi]), f"[{lo:.1f},{hi:.1f})")
            for lo, hi in zip(_BINS[:-1], _BINS[1:])
            if any(lo <= c.density < hi for c in rows)
        )
    )
    out += ["", f"  peak mean score in density bin {peak[1]} at {peak[0]:.4f}"]

    out += [
        "",
        "Unconstrained argmax per criterion, no band applied. A criterion that "
        "needs a band\nputs its argmax outside one.",
        f"{'criterion':>14} {'repr':>10} {'density':>8} {'score':>8} {'gain':>8}",
    ]
    for label, key in (("raw", "score"), ("gain", "gain")):
        for layer in cfg.representations:
            sub = [c for c in rows if c.layer == layer]
            if not sub:
                continue
            c = max(sub, key=lambda c: getattr(c, key))
            out.append(f"{label:>14} {layer:>10} {c.density:>8.2f} {c.score:>8.4f} {c.gain:>+8.4f}")

    inband = [c for c in rows if c.metrics.get("in_band")]
    if inband:
        rho_all = spearmanr([c.score for c in rows], [c.gain for c in rows]).statistic
        rho_in = spearmanr([c.score for c in inband], [c.gain for c in inband]).statistic
        out += [
            "",
            f"Spearman(raw, gain): all cells {rho_all:+.3f}, in-band only {rho_in:+.3f} "
            f"(n={len(inband)})",
            "  Low overall and high in-band is the signature of gain being "
            "degenerate outside the band.",
        ]
    return "\n".join(out)
