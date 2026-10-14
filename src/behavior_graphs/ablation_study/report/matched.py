"""Compare representations over the subgrid all of them actually searched.

The arms did not get equal search budgets, and the gap is large enough to be
mistaken for a result. On bottles: ``state`` has 1,728 cells, the three policy
taps 1,152 each, and ``dinov3`` 196, because a 3,072-d feature makes the full
width-by-stride cross I/O-bound (see ``configs/grid/ablation_dinov3.yaml``). ``state``
is the widest of all, having accumulated an extra pass at ``n_components`` values
the other arms never saw.

So "each arm's own best" partly measures how hard each arm was searched. This
report gives the other number: the best each representation reaches when every
arm is restricted to the dial combinations *all* of them cover. The winner of an
unmatched comparison can differ from the winner of a matched one, and where it
does, that is worth knowing rather than smoothing over.

Both columns are reported side by side rather than one replacing the other. The
unmatched best is what the method would actually deliver if you searched that
representation properly; the matched best is what isolates the representation.
"""

from __future__ import annotations

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.scoring import Scorer

#: The axes a cell must agree on to count as the same configuration across
#: representations. `layer` is excluded, being the thing under comparison.
MATCH_AXES = tuple(a for a in C.AXES if a != "layer")


def common_configs(rows: list[C.Cell], layers: list[str]) -> set[tuple]:
    """Dial combinations present in every one of ``layers``."""
    per_layer = []
    for layer in layers:
        per_layer.append(
            {tuple(str(c.axis(a)) for a in MATCH_AXES) for c in rows if c.layer == layer}
        )
    return set.intersection(*per_layer) if per_layer else set()


def run(cfg: DatasetConfig, **_) -> str:
    rows = C.load(cfg)
    present = [layer for layer in cfg.representations if any(c.layer == layer for c in rows)]
    missing = [layer for layer in cfg.representations if layer not in present]
    if len(present) < 2:
        return f"{cfg.name}: only {present} present; nothing to match"

    shared = common_configs(rows, present)
    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    sel = cfg.evaluation.criterion

    out = [
        f"{cfg.name}: representation comparison, {sel}",
        f"  arms present: {', '.join(present)}"
        + (f"   MISSING: {', '.join(missing)}" if missing else ""),
        f"  shared configurations: {len(shared)} of "
        + " / ".join(f"{layer} {sum(1 for c in rows if c.layer == layer)}" for layer in present),
        "",
        f"{'representation':>14} {'cells':>6} "
        f"{'own best':>9} {'matched':>9} {'test':>9} {'unif':>7} {'head':>7} "
        f"{'dens':>6}  configuration",
    ]

    for layer in present:
        mine = [c for c in rows if c.layer == layer]
        own = max(mine, key=lambda c: c.score)
        inter = [c for c in mine if tuple(str(c.axis(a)) for a in MATCH_AXES) in shared]
        m = max(inter, key=lambda c: c.score) if inter else None

        # The reported figure is the *matched* winner's held-out score where a
        # matched winner exists, so the test column and the matched column
        # describe one configuration rather than two.
        pick = m or own
        t = scorer.score_cell(pick, named[splits.TEST])
        unif = t[f"segmental_uniform_f1@{sel.split('@')[1]}"]
        head = t[sel] - unif
        out.append(
            f"{layer:>14} {len(mine):>6} "
            f"{own.score:>9.4f} {(m.score if m else float('nan')):>9.4f} "
            f"{t[sel]:>9.4f} {unif:>7.4f} {head:>+7.3f} {t['density']:>6.2f}  "
            f"{pick.reduction}-{pick.n_components}/{pick.clusterer}/"
            f"K={pick.k_label()}/w{pick.width}s{pick.stride}/"
            f"{pick.smoothing}/ms{pick.min_segment}"
        )

    out += [
        "",
        f"  own best = highest cell of that arm, over the {len(rows)} cells it searched.",
        "  matched  = highest cell restricted to the shared configurations, so the",
        "             columns differ only in representation.",
        f"  test     = the matched winner re-scored on the {len(named[splits.TEST])} reserved",
        "             episodes; head is that minus the count-matched uniform baseline.",
        f"  Differences below {cfg.evaluation.noise_floor:.3f} are inside the across-fold",
        "  standard error and should not be read as an ordering.",
    ]
    return "\n".join(out)
