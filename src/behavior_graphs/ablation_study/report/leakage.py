"""Did the policy that produced the features train on the episodes we score on?

For bottles it did. The exclusion list was built for an earlier 40-annotated-episode
study that held out 20, and never updated when the study grew to 174 with 87 held
out, so the policy trained on 76 of those 87. The gain over the count-matched
floor turned out to be the same on seen and unseen episodes (+0.101 against
+0.099), so the result survived, but by luck rather than design.

Datasets configured with generated splits derive the policy's training ids from
the same draw, so this report should find no overlap at all. It is kept as a
standing check rather than a one-off: a report that only ever prints zero is
exactly the report worth keeping.

`state` features are never exposed, having no policy in their path, which is
what makes them the calibration for how much of any seen/unseen difference is
just sampling.
"""

from __future__ import annotations

import json
import pathlib

import yaml

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.report.ablation import _default_from_plateau
from behavior_graphs.ablation_study.scoring import Scorer


def _policy_trained_dirs(cfg: DatasetConfig) -> set[str] | None:
    """Episode directory names the policy trained on, or None if unknowable."""
    if not cfg.policy_run or not cfg.policy_zarr:
        return None
    run = cfg.policy_run / "config.json"
    orders = list(cfg.policy_zarr.glob("*/abc_episode_order.json"))
    if not run.exists() or not orders:
        return None
    ids = json.loads(run.read_text())["data"]["episode_ids"]
    order = json.loads(orders[0].read_text())["episode_dirs"]
    return {order[i] for i in ids if i < len(order)}


def run(cfg: DatasetConfig, **_) -> str:
    trained = _policy_trained_dirs(cfg)
    if trained is None:
        return (
            f"{cfg.name}: no policy run or cache order on disk "
            f"(policy_run={cfg.policy_run}, policy_zarr={cfg.policy_zarr}); "
            "nothing to audit yet."
        )

    named = splits.resolve(cfg)
    meta = yaml.safe_load((cfg.root / "episodes" / "metadata.yaml").read_text())
    eps = meta["episodes"] if isinstance(meta, dict) and "episodes" in meta else meta
    id_of = {i: e.get("wgo_id") for i, e in enumerate(eps)}

    out = [
        f"{cfg.name}: policy trained on {len(trained)} cache episodes",
        "",
        f"{'split':>8} {'n':>5} {'in policy training':>20}",
    ]
    for part in (splits.TRAIN, splits.VAL, splits.TEST):
        idx = named[part]
        inside = [i for i in idx if id_of.get(i) in trained]
        out.append(f"{part:>8} {len(idx):>5} {len(inside):>20}")

    test = named[splits.TEST]
    seen = [i for i in test if id_of.get(i) in trained]
    unseen = [i for i in test if id_of.get(i) not in trained]
    if not seen or not unseen:
        out += [
            "",
            "No usable seen/unseen contrast on the test split"
            + (
                " -- nothing leaked, which is the intended state."
                if not seen
                else " -- every test episode was in policy training."
            ),
        ]
        return "\n".join(out)

    rows = C.load(cfg)
    scorer = Scorer(cfg)
    tol = cfg.evaluation.tolerance
    sel = cfg.evaluation.criterion
    star = _default_from_plateau(rows, cfg.evaluation.noise_floor)

    out += [
        "",
        f"Recommended configuration on test episodes the policy saw ({len(seen)}) "
        f"against those it did not ({len(unseen)}).",
        "The two sets differ in difficulty, so the gain over each set's own "
        "count-matched\nfloor is the comparable quantity, not raw F1.",
        "",
        f"{'layer':>10} {'seen F1':>8} {'gain':>8} {'unseen F1':>10} {'gain':>8}",
    ]
    gains = {}
    for layer in cfg.representations:
        cell = C.best(
            rows,
            layer=layer,
            **{
                a: star.axis(a)
                for a in (
                    "reduction",
                    "n_components",
                    "clusterer",
                    "k",
                    "width",
                    "stride",
                    "smoothing",
                )
            },
        )
        if cell is None:
            continue
        s = scorer.score_cell(cell, seen, tol)
        u = scorer.score_cell(cell, unseen, tol)
        uk = C.companion_key(sel, "uniform")
        gs = s[sel] - s[uk]
        gu = u[sel] - u[uk]
        gains[layer] = (gs, gu)
        out.append(f"{layer:>10} {s[sel]:>8.4f} {gs:>+8.4f} {u[sel]:>10.4f} {gu:>+8.4f}")

    if "state" in gains and len(gains) > 1:
        out += [
            "",
            "  `state` has no policy in its path, so its seen/unseen difference of "
            f"{gains['state'][1] - gains['state'][0]:+.4f}\n  is pure sampling and "
            "calibrates the others.",
        ]
    _ = pathlib
    return "\n".join(out)
