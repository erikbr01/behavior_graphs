"""The complete balanced core design, read from its own grid config.

The wide sweep is a conditional space -- ``n_components`` is nested inside
``reduction``, ``k`` inside ``clusterer``, and ``k`` is not even the same
quantity across clusterers -- so no factorial exists over it and a
best-per-level table there compares levels searched over unequal, confounded
sets. ``configs/grid/ablation_core.yaml`` declares the largest rectangle that does
exist; this selects it out of a loaded sweep.

The level sets are parsed from that YAML rather than restated here. Two copies
of a design definition is exactly how the table came to claim a balance the
sweep did not have.
"""

from __future__ import annotations

import collections
import functools
import pathlib
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import yaml

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study.config import DatasetConfig

GRID = "ablation_core"
GRID_PATH = pathlib.Path(__file__).resolve().parents[3] / "configs" / "grid" / f"{GRID}.yaml"

#: Reduction and clustering targets name their own dial, so the grid's shape has
#: to be read per stage rather than assumed uniform.
_K_KEYS = ("n_clusters", "n_components", "min_cluster_size")


@functools.cache
def design() -> dict[str, tuple[str, ...]]:
    """``axis -> levels`` of the core, as strings matching ``Cell.axis`` output."""
    stages = (yaml.safe_load(GRID_PATH.read_text()) or {})["stages"]
    sliding = stages["windowing"]["sliding"]
    reductions = stages["reduction"]
    clustering = stages["clustering"]

    ncs: set[str] = set()
    for spec in reductions.values():
        n = spec.get("n_components")
        ncs.update(str(v) for v in (n if isinstance(n, list) else [n]))
    ks: set[str] = set()
    for spec in clustering.values():
        for key in _K_KEYS:
            if key in spec:
                v = spec[key]
                ks.update(str(x) for x in (v if isinstance(v, list) else [v]))
                break

    return {
        "reduction": tuple(sorted(reductions)),
        "n_components": tuple(sorted(ncs, key=float)),
        "clusterer": tuple(sorted(clustering)),
        "k": tuple(sorted(ks, key=float)),
        "width": tuple(str(v) for v in sliding["width"]),
        "stride": tuple(str(v) for v in sliding["stride"]),
        "smoothing": tuple(str(v) for v in stages["graph"]["transition"]["smoothing"]),
    }


#: Axes the core crosses. ``layer`` is excluded (it is what gets compared) and
#: ``min_segment`` is excluded because it is a post-hoc merge, applied after
#: clustering, so it multiplies every job rather than varying it.
def axes() -> tuple[str, ...]:
    return tuple(design())


#: Dials the core design *pins* instead of sweeping, and the values that count
#: as pinned. A cell varying one of these has the same swept axes as a core
#: cell, so before 2026-09-10 it passed both `is_core` and `is_core_dials` --
#: which is how 204 `n_neighbors` cells landed in `rescored_core.json` and then
#: won `select`'s newest-wins dedup for being the newest directories on disk.
#:
#: ``"None"`` is allowed everywhere because a dial that does not apply is
#: genuinely absent: PCA has no `n_neighbors`, k-means no `min_duration`. It
#: also keeps records written before these keys existed loadable, which is
#: correct -- those artefacts predate the exploratory grids.
PINNED_DIALS: dict[str, tuple[str, ...]] = {
    # cuml_umap.yaml / umap.yaml default.
    "n_neighbors": ("None", "15"),
    # hsmm.yaml default.
    "min_duration": ("None", "1"),
    # configs/config.yaml default.
    "episode_norm": ("None", "none"),
}


def _pinned_ok(get: Callable[[str], Any]) -> bool:
    """Whether every pinned dial sits at its core value."""
    return all(str(get(name)) in allowed for name, allowed in PINNED_DIALS.items())


def is_core(cell: C.Cell) -> bool:
    d = design()
    if not all(str(cell.axis(a)) in d[a] for a in d):
        return False
    return _pinned_ok(lambda n: getattr(cell, n, None))


def _mtime(d: str) -> float:
    try:
        return pathlib.Path(d).stat().st_mtime
    except OSError:
        return 0.0


def select(rows: Iterable[C.Cell], *, quiet: bool = False) -> list[C.Cell]:
    """Just the cells inside the core design, one job dir per design point.

    The dedup is not defensive tidying; without it the core silently stops
    being a factorial. Changing a job's key scheme leaves the old directory
    on disk at the same design point as the new one, and both carry a
    `config.yaml` naming the same arm, so both load. That is what made the
    `state_dinov3` arm report 78 of 72 cells on earbuds after `55a5369`:
    6 pre-fix directories plus a complete fresh 72. An argmax barely notices a
    duplicate, but a mean, a standard deviation and an eta-squared all weight
    those 6 design points twice.

    Deduping here rather than in `cells.load` is deliberate. The core is a full
    factorial over `design()`, so within it identical dials can only mean the
    same design point ran twice. The wide grid is not: it varies axes that
    `cells.AXES` does not carry, `graph.sticky_lambda` among them, and two jobs
    there can legitimately share every axis this module knows about.

    Ties break toward the newest directory, since the reason a duplicate exists
    is that the newer run is the one the current keying intends.

    The unit chosen is the *directory*, not the cell. `min_segment` is excluded
    from the design point because it is a post-hoc merge that multiplies a job,
    so every cell of one directory shares one design point -- and callers pass
    `per_job` output, where each directory has already collapsed to its best
    `min_segment`. Two duplicate directories can peak at different levels, so a
    key that included `min_segment` would let both survive.
    """
    core = [c for c in rows if is_core(c)]
    winner: dict[tuple[str, ...], str] = {}
    seen: dict[tuple[str, ...], set[str]] = {}
    layer_of: dict[str, str] = {}
    for c in core:
        key = (str(c.layer), *(str(c.axis(a)) for a in design()))
        layer_of[c.dir] = str(c.layer)
        seen.setdefault(key, set()).add(c.dir)
        cur = winner.get(key)
        if cur is None or _mtime(c.dir) > _mtime(cur):
            winner[key] = c.dir
    keep = set(winner.values())
    dropped: collections.Counter[str] = collections.Counter()
    for key, dirs in seen.items():
        for d in dirs - keep:
            dropped[layer_of[d]] += 1
    if dropped and not quiet:
        detail = ", ".join(f"{layer} {n}" for layer, n in sorted(dropped.items()))
        print(
            f"WARNING core.select: {sum(dropped.values())} stale job dirs at "
            f"duplicate design points ignored in favour of the newest "
            f"({detail}). Expected after a job-key change; the stale dirs can "
            f"be removed once the fresh arm is complete.",
            file=sys.stderr,
        )
    return [c for c in core if c.dir in keep]


def is_core_dials(dials: Mapping[str, Any]) -> bool:
    """``is_core`` for a re-score's raw dials dict rather than a loaded Cell.

    The re-score filters before it loads a job's labels, so it only has the
    dials it read out of `config.yaml`; `is_core` wants a `Cell`, which does not
    exist until after scoring. Both compare the same axis names against the same
    grid, because `_dials` derives `reduction`/`clusterer` from `_target_` the
    same way the rescored records do.

    This is what makes a core re-score affordable: 180 jobs in minutes against
    thousands for the whole grid.

    Also rejects cells that vary a dial the core pins -- see `PINNED_DIALS`.
    Without that check this returned True for every `n_neighbors`,
    `min_duration` and `episode_norm` cell, because those dials are not axes of
    `design()` and so were never compared against anything.
    """
    d = design()
    if not all(str(dials.get(a)) in d[a] for a in d):
        return False
    return _pinned_ok(dials.get)


def ablation_axes(rows: Sequence[C.Cell]) -> tuple[str, ...]:
    """Axes worth a row in a core table: those with more than one level present.

    ``n_components``, ``width`` and ``stride`` are single-valued in the core by
    construction, so they carry no comparison and are dropped rather than
    printed as a block with nothing to contrast.
    """
    keep = ["layer", *design(), "min_segment"]
    return tuple(a for a in C.AXES if a in keep and len(C.levels(rows, a)) > 1)


def report(cfg: DatasetConfig, rows: Sequence[C.Cell]) -> str:
    """Whether the core is complete and balanced here, as text."""
    import itertools

    core = select(rows)
    per_job = C.per_job(core)
    d = design()
    need = set(itertools.product(*[d[a] for a in axes()]))
    out = [f"{cfg.name}: {len(per_job)} core jobs ({len(core)} cells)"]
    complete = True
    for layer in cfg.representations:
        mine = {tuple(str(c.axis(a)) for a in axes()) for c in per_job if c.layer == layer}
        missing = need - mine
        complete &= not missing
        out.append(
            f"  {layer:10s} {len(mine):3d}/{len(need)}"
            + ("  complete" if not missing else f"  MISSING {len(missing)}")
        )
    even = True
    for a in ("layer", *axes()):
        counts = collections.Counter(str(c.axis(a)) for c in per_job)
        even &= len(set(counts.values())) == 1
    out.append(f"  => {'BALANCED' if complete and even else 'NOT balanced'}")
    return "\n".join(out)
