"""Which design choices win, and how much of the variance each axis explains.

Two views that answer different questions and are meant to be read together.
The dial marginal is the best achievable at each level, which says "if I must
use this option, how well can I do". Eta-squared is that axis's share of
variance across the whole grid, which says "does this axis decide whether a
configuration works at all". On this data they invert: window geometry matters
most at the optimum and explains almost nothing globally, while representation,
K and clusterer do the reverse.
"""

from __future__ import annotations

import collections

import numpy as np

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study.config import DatasetConfig

_AXES_NO_MS = tuple(a for a in C.AXES if a != "min_segment")


def _floors(rows: list[C.Cell], thresholds) -> list[str]:
    out = [
        "What the count-matched uniform reference scores",
        f"{'IoU':>6} {'uniform':>9} {'best real':>10} {'gain':>8} {'headroom':>9}",
    ]
    for t in thresholds:
        uk, rk = f"segmental_uniform_f1@{t}", f"segmental_f1@{t}"
        vals = [r.metrics[uk] for r in rows if uk in r.metrics]
        reals = [r.metrics[rk] for r in rows if rk in r.metrics]
        if not vals or not reals:
            continue
        u, best = float(np.median(vals)), max(reals)
        out.append(
            f"{t:>6} {u:>9.4f} {best:>10.4f} {best - u:>+8.4f} "
            f"{(best - u) / max(1e-9, 1 - u):>9.3f}"
        )
    return out


def _per_representation(rows: list[C.Cell], layers) -> list[str]:
    out = [
        "",
        "Best configuration per representation",
        f"{'repr':>10} {'red':>10} {'nc':>4} {'clu':>8} {'K':>5} {'w':>3} {'s':>2} "
        f"{'sm':>7} {'ms':>4} {'dens':>5} {'score':>8}",
    ]
    for layer in layers:
        best = C.best(rows, layer=layer)
        if best is None:
            continue
        out.append(
            f"{layer:>10} {best.reduction:>10} {str(best.n_components):>4} "
            f"{best.clusterer:>8} {best.k_label():>5} {best.width:>3} {best.stride:>2} "
            f"{best.smoothing:>7} {best.min_segment:>4} {best.density:>5.2f} "
            f"{best.score:>8.4f}"
        )
    return out


def _marginals(rows: list[C.Cell], jobs: list[C.Cell]) -> list[str]:
    out = ["", "Dial marginals. dens@best guards against buying score with segment count."]
    for axis in C.AXES:
        key = "k_label" if axis == "k" else axis
        by: dict[str, list[C.Cell]] = collections.defaultdict(list)
        for r in rows:
            by[r.k_label() if axis == "k" else str(r.axis(axis))].append(r)
        out += [
            f"\n  {key}",
            f"    {'level':>14} {'n':>6} {'best':>8} {'dens@best':>10} {'median':>8}",
        ]
        for lvl in sorted(by, key=lambda v: (0, float(v)) if _num(v) else (1, v)):
            grp = by[lvl]
            top = max(grp, key=lambda c: c.score)
            med = float(np.median([c.score for c in grp]))
            out.append(
                f"    {lvl:>14} {len(grp):>6} {top.score:>8.4f} {top.density:>10.2f} {med:>8.4f}"
            )
    out += ["", "Variance explained per axis (one row per job, min_segment collapsed)"]
    eta = sorted(((C.eta_squared(jobs, a), a) for a in _AXES_NO_MS), reverse=True)
    for e, a in eta:
        out.append(f"  {a:>14} eta2 = {e:6.3f}")
    return out


def _num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def run(cfg: DatasetConfig, top: int = 15, **_) -> str:
    rows = C.load(cfg)
    jobs = C.per_job(rows)
    out = [
        f"{cfg.name}: {len(rows)} cells over {len(jobs)} jobs, "
        f"criterion {cfg.evaluation.criterion}, gt_set {cfg.evaluation.gt_set}",
        "",
    ]
    out += _floors(rows, cfg.evaluation.iou_thresholds)
    out += _per_representation(rows, cfg.representations)
    out += _marginals(rows, jobs)
    out += [
        "",
        f"Top {top} cells",
        f"{'rk':>3} {'repr':>10} {'red':>10} {'nc':>4} "
        f"{'clu':>8} {'K':>5} {'w':>3} {'s':>2} {'sm':>7} {'ms':>4} {'dens':>5} {'score':>8}",
    ]
    for i, r in enumerate(sorted(rows, key=lambda c: -c.score)[:top], 1):
        out.append(
            f"{i:>3} {r.layer:>10} {r.reduction:>10} {str(r.n_components):>4} "
            f"{r.clusterer:>8} {r.k_label():>5} {r.width:>3} {r.stride:>2} "
            f"{r.smoothing:>7} {r.min_segment:>4} {r.density:>5.2f} {r.score:>8.4f}"
        )
    return "\n".join(out)


def plateau_report(cfg: DatasetConfig, **_) -> str:
    """The configuration to recommend, read off the plateau rather than the argmax.

    The argmax is the maximum of tens of thousands of scores on one split, so
    part of its margin is selection noise and several axes at the top are
    separated by less than the across-fold standard error. Where an axis does
    not resolve, the default takes the simplest option instead of the winner.
    """
    rows = C.load(cfg)
    se = cfg.evaluation.noise_floor
    star = max(rows, key=lambda c: c.score)
    near = C.plateau(rows, se)
    out = [
        f"{cfg.name}: argmax {star.score:.4f}; {len(near)} cells within one SE ({se})",
        "",
        f"{'axis':>14} {'verdict':>11}  levels among the plateau",
    ]
    resolved: dict[str, str] = {}
    for axis in C.AXES:
        ct = collections.Counter(c.k_label() if axis == "k" else str(c.axis(axis)) for c in near)
        lvl, n = ct.most_common(1)[0]
        frac = n / len(near)
        verdict = "FIXED" if frac == 1.0 else ("dominant" if frac >= 0.8 else "unresolved")
        if frac >= 0.8:
            resolved[axis] = lvl
        out.append(
            f"{axis:>14} {verdict:>11}  " + "  ".join(f"{k}:{v}" for k, v in ct.most_common())
        )

    # Simplest member of the tie: cheapest clusterer, no smoothing, fewest
    # components, least post-hoc merging. Preference order, not a score.
    rank = {"kmeans": 0, "gmm": 1, "hsmm": 2, "hdbscan": 3}
    pick = min(
        near,
        key=lambda c: (
            rank.get(c.clusterer, 9),
            0 if c.smoothing == "none" else 1,
            c.n_components or 0,
            float(c.min_segment),
        ),
    )
    out += [
        "",
        "Simplest configuration statistically indistinguishable from the best:",
        "  " + ", ".join(f"{a}={pick.axis(a)}" for a in C.AXES),
        f"  score {pick.score:.4f} ({pick.score - star.score:+.4f} against the argmax), "
        f"density {pick.density:.2f}",
        "",
        f"  The argmax is the maximum of {len(rows)} scores on one split. The 2nd "
        f"through 10th best span "
        f"{sorted(c.score for c in rows)[-2]:.4f} to {sorted(c.score for c in rows)[-10]:.4f}.",
    ]
    return "\n".join(out)
