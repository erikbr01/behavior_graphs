"""The sweep as a flat table of cells, loaded once.

Eleven analysis scripts each re-implemented the same four steps: read
``rescored.json``, join each job's ``metrics.json`` for the unsupervised
statistics, flatten the dial settings into a record, and expand ``min_segment``
into its own axis. That is what this module is.

A *cell* is one (job, min_segment) pair: the finest unit that has a score, since
``min_segment`` is a post-hoc merge applied after clustering and so multiplies
each job rather than varying it. Reports that want one row per job should
aggregate with :func:`per_job`, and must say which way -- taking the best over
``min_segment`` and treating the result as a single observation is right for a
dial marginal and wrong for an eta-squared over ``min_segment`` itself.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

import numpy as np

from behavior_graphs.ablation_study.config import DatasetConfig

#: rescored.json's key for each min_segment level, and the label we use.
MS_KEYS = {"0.0": "primary", "0.5": "ms0.5", "1": "ms1", "1.5": "ms1.5", "2": "ms2"}


def companion_key(criterion: str, kind: str) -> str:
    """The ``uniform`` or ``gain`` metric key at the criterion's own threshold.

    Four reports hardcoded ``segmental_uniform_f1@0.5`` while their score
    column followed ``EvalSpec.criterion``. That was consistent only while the
    criterion was also at 0.50; after the move to ``segmental_f1@0.75`` the
    held-out table printed IoU-0.75 scores against IoU-0.50 floors, which makes
    every row look like it loses to a uniform predictor (0.30 against 0.52)
    when at its own threshold it beats one comfortably.

    A criterion that is not a ``segmental_f1@<t>`` key has no companion, so the
    0.50 keys are returned unchanged rather than a guess -- the caller's
    numbers stay wrong in that case, but wrong in the way they already were.
    """
    prefix = "segmental_f1@"
    target = {"uniform": "segmental_uniform_f1@", "gain": "segmental_gain@"}[kind]
    if not criterion.startswith(prefix):
        return target + "0.5"
    return target + criterion[len(prefix) :]


#: Every design axis, in the order reports display them.
AXES = (
    "layer",
    "reduction",
    "n_components",
    "clusterer",
    "k",
    "width",
    "stride",
    "smoothing",
    "min_segment",
)

#: Annotation-free statistics, as (metrics.json block, key, +1 if larger is
#: meant to be better). Used by the proxy report; none of them selects well.
PROXIES = {
    "silhouette": ("cluster_quality", "silhouette", +1),
    "davies_bouldin": ("cluster_quality", "davies_bouldin", -1),
    "calinski_harabasz": ("cluster_quality", "calinski_harabasz", +1),
    "noise_fraction": ("cluster_quality", "noise_fraction", -1),
    "n_clusters": ("cluster_quality", "n_clusters", +1),
    "num_edges": ("graph_stats", "num_edges", +1),
    "mean_out_degree": ("graph_stats", "mean_out_degree", +1),
    "transition_entropy": ("graph_stats", "mean_transition_entropy", +1),
    "swap_rate_per_window": ("label_stats", "swap_rate_per_window", +1),
    "distinct_per_episode": ("label_stats", "distinct_per_episode", +1),
    "mean_segment_length": ("label_stats", "mean_segment_length", +1),
}


@dataclasses.dataclass
class Cell:
    """One configuration at one ``min_segment``, with its scores."""

    dir: str
    layer: str
    reduction: str
    n_components: Any
    clusterer: str
    k: Any
    width: int
    stride: int
    smoothing: str
    min_segment: str
    score: float
    density: float
    gain: float
    #: Dials the core pins rather than sweeps, so they are deliberately absent
    #: from `AXES` and never displayed as a comparison. They are carried anyway
    #: because `core.is_core` has to reject a cell that varies one: without
    #: them an `n_neighbors=400` cell is indistinguishable from a core cell.
    #: `None` both where the dial does not apply (PCA has no `n_neighbors`) and
    #: for records written before the re-score recorded them.
    n_neighbors: Any = None
    min_duration: Any = None
    episode_norm: Any = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    proxies: dict[str, float] = dataclasses.field(default_factory=dict, repr=False)

    def axis(self, name: str) -> Any:
        return getattr(self, name)

    def k_label(self) -> str:
        """HDBSCAN's dial is a minimum cluster size, not a cluster count.

        The re-score stored ``n_clusters or n_components or min_cluster_size``
        in one field, so pooling them puts 200 next to 32 as though it were a
        larger K when it means the opposite -- a higher floor yields fewer
        clusters. Tagged so the two never average together.
        """
        return f"mcs{self.k}" if self.clusterer == "hdbscan" else str(self.k)


def load(
    cfg: DatasetConfig,
    *,
    with_proxies: bool = False,
    gt_set: str | None = None,
    criterion: str | None = None,
    path: pathlib.Path | None = None,
) -> list[Cell]:
    """Every cell of the sweep. ``with_proxies`` joins each job's metrics.json.

    The join is the slow part (one small file read per job), so it is opt-in;
    only the proxy and leakage reports need it.
    """
    gt = gt_set or cfg.evaluation.gt_set
    sel = criterion or cfg.evaluation.criterion
    # `path` reads an alternative artefact -- the core design's `rescored_core.json`,
    # which is 180 records against the wide grid's thousands and so can be
    # refreshed in minutes rather than an hour.
    src = path or cfg.rescored
    if not src.exists():
        raise SystemExit(f"{cfg.name}: no {src}; run `study.py rescore --dataset {cfg.name}`")

    records = json.loads(src.read_text())
    # Say so up front when the artefact has outlived its jobs. Reports that only
    # read cached scores still work, and should; the ones that reopen labels.npy
    # will fail, and this is the reason why.
    gone = sum(1 for rec in records if not (pathlib.Path(rec["dir"]) / "labels.npy").exists())
    if gone:
        print(
            f"WARNING {cfg.name}: {gone} of {len(records)} jobs in {src.name} "
            f"no longer have labels.npy. Any report that re-scores will fail; "
            f"run `study.py rescore --dataset {cfg.name}`.",
            file=sys.stderr,
        )

    out: list[Cell] = []
    for rec in records:
        block = rec["metrics"][gt]
        proxies: dict[str, float] = {}
        if with_proxies:
            mp = pathlib.Path(rec["dir"]) / "metrics.json"
            m = json.loads(mp.read_text()) if mp.exists() else {}
            for name, (blk, key, _) in PROXIES.items():
                v = (m.get(blk) or {}).get(key)
                proxies[name] = float(v) if v is not None and np.isfinite(v) else float("nan")
        for ms, key in MS_KEYS.items():
            b = block.get(key)
            if not b or sel not in b:
                continue
            out.append(
                Cell(
                    dir=rec["dir"],
                    layer=rec["layer"],
                    reduction=rec["reduction"],
                    n_components=rec["n_components"],
                    clusterer=rec["clusterer"],
                    k=rec["k"],
                    width=int(rec["width"]),
                    stride=int(rec["stride"]),
                    smoothing=rec["smoothing"] or "none",
                    min_segment=ms,
                    score=float(b[sel]),
                    density=float(b["density"]),
                    gain=float(b.get(companion_key(sel, "gain"), float("nan"))),
                    n_neighbors=rec.get("n_neighbors"),
                    min_duration=rec.get("min_duration"),
                    episode_norm=rec.get("episode_norm"),
                    metrics=b,
                    proxies=proxies,
                )
            )
    if not out:
        raise SystemExit(f"{cfg.name}: no cells with criterion {sel!r} on gt_set {gt!r}")
    return out


def per_job(cells: Iterable[Cell], how: str = "best") -> list[Cell]:
    """Collapse the ``min_segment`` axis to one cell per job.

    ``how='best'`` keeps the highest-scoring level, which is what a dial
    marginal wants -- otherwise every job appears five times and the
    ``min_segment`` axis inflates its own share of variance while diluting
    every other axis.
    """
    if how != "best":
        raise ValueError(f"unknown aggregation {how!r}")
    best: dict[str, Cell] = {}
    for c in cells:
        cur = best.get(c.dir)
        if cur is None or c.score > cur.score:
            best[c.dir] = c
    return sorted(best.values(), key=lambda c: c.dir)


def levels(cells: Iterable[Cell], axis: str) -> list[str]:
    """The distinct levels of one axis, numerically ordered where possible."""
    seen = {str(c.axis(axis)) for c in cells}

    def key(v: str) -> tuple[int, float | str]:
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    return sorted(seen, key=key)


def eta_squared(cells: list[Cell], axis: str) -> float:
    """Share of variance in the score explained by one axis."""
    y = np.asarray([c.score for c in cells])
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0:
        return 0.0
    by: dict[str, list[float]] = collections.defaultdict(list)
    for c in cells:
        by[str(c.axis(axis))].append(c.score)
    ssb = sum(len(v) * (float(np.mean(v)) - y.mean()) ** 2 for v in by.values())
    return ssb / sst


def best(cells: Iterable[Cell], **where: Any) -> Cell | None:
    """Highest-scoring cell matching every given axis value."""
    hits = [c for c in cells if all(str(c.axis(a)) == str(v) for a, v in where.items())]
    return max(hits, key=lambda c: c.score) if hits else None


def plateau(cells: list[Cell], within: float) -> list[Cell]:
    """Cells within ``within`` of the best -- the set that dials must agree on."""
    top = max(c.score for c in cells)
    return [c for c in cells if c.score >= top - within]
