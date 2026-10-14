"""One seeded partition that both the study and the policy read.

The bottles study kept two lists: ``splits.json`` for the graph pipeline, and a
hand-built ``episode_ids`` for the policy. The second went stale against the
first, and the policy trained on 76 of the 87 episodes reported as held out. The
fix is structural rather than procedural: draw the partition once, write the
policy's training ids from that same draw, and pin both with a digest so a
mismatch is loud.

Splits balance on episodes, boundaries and duration rather than on episode count
alone. A 20% test set holding 20% of the episodes but a tenth of the boundaries
is not a 20% test set, and with segment counts varying 3x across episodes a
plain random draw does that often enough to matter.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from behavior_graphs.ablation_study.config import DatasetConfig

#: Names used everywhere downstream, whatever a legacy file happens to call them.
TRAIN, VAL, TEST = "train", "val", "test"


def _episode_stats(ann: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Per annotated episode: how much of each thing it contributes."""
    out = {}
    for rec in ann:
        segs = rec.get("segments") or []
        out[int(rec["index"])] = {
            "episodes": 1.0,
            "segments": float(len(segs)),
            "duration": float(rec.get("duration_sec") or 0.0),
        }
    return out


def balanced_three_way(
    stats: dict[int, dict[str, float]],
    fracs: tuple[float, float, float],
    seed: int = 0,
    n_trials: int = 20000,
) -> dict[str, list[int]]:
    """Partition episode indices into three parts balanced on every statistic.

    Draws ``n_trials`` seeded permutations and keeps the one minimising the worst
    relative imbalance over episodes, segments and duration across all three
    parts. Deterministic in ``seed``, and the achieved imbalance is reported by
    :func:`report` rather than assumed to be zero.
    """
    ids = sorted(stats)
    keys = ("episodes", "segments", "duration")
    totals = {k: sum(stats[i][k] for i in ids) or 1.0 for k in keys}
    n = len(ids)
    cuts = (
        int(round(n * fracs[0])),
        int(round(n * (fracs[0] + fracs[1]))),
    )

    rng = np.random.RandomState(seed)
    best, best_cost = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(n)
        parts = (perm[: cuts[0]], perm[cuts[0] : cuts[1]], perm[cuts[1] :])
        cost = 0.0
        for want, part in zip(fracs, parts):
            for k in keys:
                got = sum(stats[ids[j]][k] for j in part) / totals[k]
                cost = max(cost, abs(got - want) / want)
        if cost < best_cost:
            best_cost, best = cost, parts
            if cost < 1e-12:
                break

    return {name: sorted(int(ids[j]) for j in part) for name, part in zip((TRAIN, VAL, TEST), best)}


def digest(named: dict[str, list[int]]) -> str:
    """Stable fingerprint of a partition, for pinning against silent re-draws."""
    canonical = json.dumps({k: sorted(v) for k, v in sorted(named.items())})
    return hashlib.sha256(canonical.encode()).hexdigest()


def generate(cfg: DatasetConfig) -> dict[str, Any]:
    """Draw the partition described by ``cfg.split`` over annotated episodes."""
    spec = cfg.split
    if spec.file:
        raise ValueError(f"{cfg.name}: split is read from a file, nothing to generate")
    ann = json.loads(cfg.annotations.read_text())
    stats = _episode_stats(ann)
    if not stats:
        raise SystemExit(f"{cfg.name}: no annotated episodes in {cfg.annotations}")
    named = balanced_three_way(stats, (spec.train, spec.val, spec.test), seed=spec.seed)
    return {
        "dataset": str(cfg.root),
        "task": cfg.task,
        "seed": spec.seed,
        "fractions": {TRAIN: spec.train, VAL: spec.val, TEST: spec.test},
        "annotated_only": spec.annotated_only,
        "n_annotated": len(stats),
        "sha256": digest(named),
        **named,
    }


def resolve(cfg: DatasetConfig) -> dict[str, list[int]]:
    """The partition as ``{train, val, test}``, whether generated or legacy.

    Legacy files are renamed through ``cfg.split.keys`` here, so downstream code
    only ever sees the three canonical names.
    """
    path = cfg.splits_path
    if not path.exists():
        raise SystemExit(f"{cfg.name}: no split at {path}; run `study splits --dataset {cfg.name}`")
    raw = json.loads(path.read_text())
    keys = cfg.split.keys if cfg.split.file else {TRAIN: TRAIN, VAL: VAL, TEST: TEST}
    out = {}
    for canonical, source in keys.items():
        if source not in raw:
            raise SystemExit(f"{cfg.name}: split file {path} has no key {source!r}")
        out[canonical] = sorted(int(i) for i in raw[source])
    overlap = [
        (a, b, sorted(set(out[a]) & set(out[b])))
        for a, b in ((TRAIN, VAL), (TRAIN, TEST), (VAL, TEST))
    ]
    bad = [(a, b, o) for a, b, o in overlap if o]
    if bad:
        raise SystemExit(
            f"{cfg.name}: split parts overlap, which would leak: "
            + "; ".join(f"{a} n {b} = {o[:5]}{'...' if len(o) > 5 else ''}" for a, b, o in bad)
        )
    return out


def report(cfg: DatasetConfig, named: dict[str, list[int]]) -> str:
    """Human-readable check that each part carries its share of the statistics."""
    ann = json.loads(cfg.annotations.read_text())
    stats = _episode_stats(ann)
    keys = ("episodes", "segments", "duration")
    totals = {k: sum(stats[i][k] for i in stats if i in stats) or 1.0 for k in keys}
    lines = [f"{'part':>6} {'episodes':>9} {'segments':>9} {'duration':>10}   shares"]
    for name in (TRAIN, VAL, TEST):
        idx = [i for i in named.get(name, []) if i in stats]
        got = {k: sum(stats[i][k] for i in idx) for k in keys}
        shares = "  ".join(f"{k[:3]} {got[k] / totals[k]:.3f}" for k in keys)
        lines.append(
            f"{name:>6} {int(got['episodes']):>9} {int(got['segments']):>9} "
            f"{got['duration']:>9.0f}s   {shares}"
        )
    return "\n".join(lines)


def write(cfg: DatasetConfig, payload: dict[str, Any]) -> pathlib.Path:
    out = cfg.splits_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def rgrl_train_ids(cfg: DatasetConfig, cache_order: pathlib.Path) -> dict[str, Any]:
    """The policy's ``episode_ids``: everything in the cache except the held-out.

    ``episode_ids`` indexes the *cache* build order, a different ordering from
    the study's annotated-only indices, so the two are joined through the episode
    directory name.

    The rule is exclusion, not inclusion, and the distinction matters. Only
    annotated episodes are ever scored, so an unannotated one can never leak;
    withholding it buys nothing and costs training data. Taking the intersection
    with the ``train`` split instead would have thrown away all 1,120 of bottles'
    unannotated episodes and trained its policy on 52 -- a policy so much weaker
    than the legos one that any cross-task comparison would be measuring
    training-set size.

    For an annotated-only cache the two rules coincide, which is why legos gets
    exactly its train split either way.
    """
    import yaml

    named = resolve(cfg)
    order = json.loads(cache_order.read_text())["episode_dirs"]

    meta = yaml.safe_load((cfg.root / "episodes" / "metadata.yaml").read_text())
    eps = meta["episodes"] if isinstance(meta, dict) and "episodes" in meta else meta
    id_of = {i: e.get("wgo_id") for i, e in enumerate(eps)}

    held_dirs = {id_of[i] for part in (VAL, TEST) for i in named[part] if i in id_of}
    ids = [i for i, name in enumerate(order) if name not in held_dirs]

    train_dirs = {id_of[i] for i in named[TRAIN] if i in id_of}
    in_cache = {name for name in order}
    missing = sorted(train_dirs - in_cache)
    unannotated = len(ids) - len(train_dirs & in_cache)

    # Belt and braces: the exclusion above is the mechanism, this is the check.
    leaked = sorted(i for i in ids if order[i] in held_dirs)
    if leaked:
        raise SystemExit(f"{cfg.name}: {len(leaked)} held-out episodes in the train ids")

    return {
        "episode_ids": ids,
        "count": len(ids),
        "expected_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "split_sha256": digest(named),
        "n_train_episodes": len(named[TRAIN]),
        "n_annotated_train_in_cache": len(train_dirs & in_cache),
        "n_unannotated": unannotated,
        "n_missing_from_cache": len(missing),
        "n_held_out_excluded": len(held_dirs),
    }
