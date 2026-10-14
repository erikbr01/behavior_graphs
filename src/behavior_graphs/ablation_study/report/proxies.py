"""Can any annotation-free quantity select the pipeline?

The question is not whether silhouette is a good clustering metric but whether
this method needs ground-truth annotations to tune. If some unsupervised
quantity ranked configurations well, hyperparameters could be set on a new task
for free. None does.

Scored by regret and by the percentile a proxy's argmax lands in, not by
correlation, because correlation misleads badly here: transition entropy
correlates with the metric at rho = +0.66 and still selects below a random draw.
A quantity can track the metric on average while its maximum sits in a poor
region, and selection uses only the maximum.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import percentileofscore, spearmanr

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study.config import DatasetConfig


def _assess(rows: list[C.Cell], name: str) -> tuple[float, float, float, float]:
    """(rho, argmax score, regret, percentile) for one proxy on one subset."""
    sign = C.PROXIES[name][2]
    ok = [r for r in rows if np.isfinite(r.proxies.get(name, np.nan))]
    if len(ok) < 8:
        return (float("nan"),) * 4
    x = np.asarray([r.proxies[name] for r in ok]) * sign
    y = np.asarray([r.score for r in ok])
    pick = float(y[int(x.argmax())])
    return (
        float(spearmanr(x, y).statistic),
        pick,
        pick - float(y.max()),
        float(percentileofscore(y, pick)),
    )


def run(cfg: DatasetConfig, **_) -> str:
    rows = C.per_job(C.load(cfg, with_proxies=True))
    y = np.asarray([r.score for r in rows])
    out = [
        f"{cfg.name}: {len(rows)} jobs, best {y.max():.4f}, "
        f"a random draw scores {y.mean():.4f} "
        f"({percentileofscore(y, y.mean()):.1f}th percentile)",
        "",
        "Whole grid. Sign oriented so larger is meant to be better.",
        f"{'proxy':>22} {'rho':>7} {'argmax':>8} {'regret':>8} {'pctile':>7}",
    ]
    scored = [(name, *_assess(rows, name)) for name in C.PROXIES]
    for name, rho, pick, regret, pct in sorted(scored, key=lambda r: -(r[4] or -9)):
        out.append(f"{name:>22} {rho:>+7.3f} {pick:>8.4f} {regret:>+8.4f} {pct:>6.1f}%")

    out += [
        "",
        "Per representation: the realistic case, selecting dials at a fixed representation.",
        f"{'proxy':>22} " + " ".join(f"{L:>12}" for L in cfg.representations),
        f"{'':>22} " + " ".join(f"{'pctile':>12}" for _ in cfg.representations),
    ]
    for name, *_rest in sorted(scored, key=lambda r: -(r[4] or -9)):
        cellstr = []
        for layer in cfg.representations:
            sub = [r for r in rows if r.layer == layer]
            _, _, _, pct = _assess(sub, name)
            cellstr.append(f"{pct:>11.1f}%" if np.isfinite(pct) else f"{'--':>12}")
        out.append(f"{name:>22} " + " ".join(cellstr))

    out += [
        "",
        "A proxy is only usable if its percentile beats a random draw's. Anything "
        "at or below\nthat is not a selection criterion, whatever its correlation.",
    ]
    return "\n".join(out)
