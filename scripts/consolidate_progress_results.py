"""Consolidate the progress-estimation head-to-head into two tables:

  (1) AGGREGATE  — every method x every metric, averaged over datasets, methods
      sorted by VOC (TOPReward's primary metric). One row per method.
  (2) BREAKDOWN  — the same, split out per dataset (long format).

Both are emitted as markdown + CSV. Metrics (TOPReward/ROBOMETER convention, via
behavior_graphs.progress_estimation): VOC, Kendall tau-b, Pearson, MAE (over
SUCCESSFUL episodes vs linear GT) and success-AUROC (terminal progress vs flag).

Reads the saved per-episode curve npzs in --curves-dir (no model re-runs) via the
shared io loader + method registry. The all-methods comparison exists on the 3
LIBERO suites (where the VLM baselines were run); a graph-only cross-dataset
aggregate (from the E-PE4 sweep_results.json) is appended when --sweep-json is given.

    PYTHONPATH=src python scripts/consolidate_progress_results.py \
        --curves-dir analysis/out/progress_bench_results \
        --sweep-json analysis/out/progress_allds/sweep_results.json \
        --out-dir analysis/out/progress_consolidated
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from behavior_graphs.progress_estimation.estimation import success_auroc  # noqa: E402
from behavior_graphs.progress_estimation.io import (  # noqa: E402
    METHOD_REGISTRY,
    load_curves,
    resolve_curve_path,
)
from behavior_graphs.progress_estimation.metrics import (  # noqa: E402
    MAE,
    VOC,
    KendallTau,
    Pearson,
    SuccessAUROC,
)
from behavior_graphs.progress_estimation.report import chose_k_off_eval  # noqa: E402

TAIL = 3
N_BOOT = 2000
ALPHA = 0.05
METRICS = ["voc", "kendall", "pearson", "mae", "auroc"]
HIGHER_BETTER = {"voc": True, "kendall": True, "pearson": True, "mae": False, "auroc": True}
_RANK_OBJS = {"voc": VOC(), "kendall": KendallTau(), "pearson": Pearson(), "mae": MAE()}
_AUROC = SuccessAUROC(tail=TAIL)

# LIBERO suites where every method was scored (pool500 = 50 eps/task).
DATASETS = {
    "libero_goal": "lg_pool500",
    "libero_object": "libero_object_pool500",
    "libero_spatial": "libero_spatial_pool500",
}

# Methods shown in the head-to-head (registry keys). The trivial frame_index
# baseline is excluded; display name / input label / npz infixes come from the registry.
METHOD_KEYS = [
    "graph_stage",
    "graph_absorb",
    "topreward_ll",
    "topreward",
    "robometer_4b",
    "robometer_fft",
]


def _display(m):
    return METHOD_REGISTRY[m].display


def _input(m):
    return METHOD_REGISTRY[m].input_label


def metrics_for(path: Path) -> dict | None:
    try:
        c = load_curves(path)
    except Exception as e:  # e.g. numpy-2 pickled in a numpy-1 env
        print(f"  !! could not load {path.name}: {type(e).__name__}")
        return None
    flags = list(c.flags)
    reg = {k: _RANK_OBJS[k].compute(c.curves, flags) for k in ("voc", "kendall", "pearson", "mae")}
    reg["auroc"] = _AUROC.compute(c.curves, flags)
    return reg


def fmt(v, m):
    return "  —  " if v is None or not np.isfinite(v) else f"{v:.3f}"


# ------------------------------------------------------------- CI + CLD (--stats)
def _per_episode(path):
    """Per-episode ranking metrics (nan for failed eps) + terminal + flag, by idx."""
    c = load_curves(path)
    curves, flags, idx = c.curves, c.flags, c.idx
    per = {k: np.full(len(curves), np.nan) for k in ("voc", "kendall", "pearson", "mae")}
    term = np.full(len(curves), np.nan)
    for i, cv in enumerate(curves):
        term[i] = _AUROC.per_episode(cv, bool(flags[i]))
        for k in per:
            per[k][i] = _RANK_OBJS[k].per_episode(cv, bool(flags[i]))
    return {"idx": idx, "flags": flags, "term": term, **per}


def _holm(pairs_p: dict) -> dict:
    """Holm-Bonferroni; returns {pair: adjusted_p}."""
    items = sorted(pairs_p.items(), key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for rank, (pair, p) in enumerate(items):
        adj = max(running, min(1.0, (n - rank) * p))
        running = adj
        out[pair] = adj
    return out


def cld_letters(order, different):
    """Compact-letter-display via Piepho (2004) insert-and-absorb.
    `order`: methods sorted best->worst. `different`: set of frozenset pairs that
    ARE significantly different. Returns {method: 'a'/'ab'/...}; shared letter =>
    NOT significantly different."""
    groups = [set(order)]
    for a in order:
        for b in order:
            if a >= b or frozenset((a, b)) not in different:
                continue
            new = []
            for g in groups:
                if a in g and b in g:
                    new += [g - {a}, g - {b}]
                else:
                    new.append(g)
            uniq = []
            for g in new:
                if g and g not in uniq:
                    uniq.append(g)
            groups = [g for g in uniq if not any(g < h for h in uniq)]
    groups.sort(key=lambda g: min(order.index(m) for m in g))
    letters = {chr(ord("a") + i): g for i, g in enumerate(groups)}
    return {m: "".join(L for L, g in letters.items() if m in g) for m in order}


def compute_stats(curves_dir, rng):
    """Paired episode bootstrap within each suite, averaged across suites.
    Returns per-metric: {method: (mean, lo, hi)}, {method: cld_letter}, order."""
    data = {ds: {} for ds in DATASETS}
    for ds, prefix in DATASETS.items():
        for method in METHOD_KEYS:
            path = resolve_curve_path(curves_dir, prefix, method)
            if path is not None:
                try:
                    data[ds][method] = _per_episode(path)
                except Exception:
                    pass
    methods = [m for m in METHOD_KEYS if all(m in data[ds] for ds in DATASETS)]

    if not methods:
        raise SystemExit(
            f"no method has curves for every dataset in {list(DATASETS)}; "
            f"found { ({ds: sorted(data[ds]) for ds in DATASETS}) }"
        )

    boot = {k: {m: None for m in methods} for k in METRICS}
    per_suite = {k: {m: [] for m in methods} for k in METRICS}
    for ds in DATASETS:
        # Resample episode *ids*, not row positions. `save_curves` writes rows in
        # fold-accumulation order, so a method scored with KFoldSplit(3) carries a
        # shuffled `idx` while one scored with a single AlignedSplit fold carries
        # 0..n-1. Drawing positions therefore pairs episode k of one method with a
        # different episode of another -- which is not a paired test, and it is
        # the Holm-adjusted comparisons and the CLD letters that pay for it.
        # `_per_episode` returns `idx` precisely so this is possible.
        ids = {m: [int(e) for e in data[ds][m]["idx"]] for m in methods}
        shared = sorted(set.intersection(*(set(v) for v in ids.values())))
        if not shared:
            raise SystemExit(f"{ds}: methods share no episode ids, so nothing can be paired")
        # Never silently truncate to the shortest method, as indexing by position
        # did: say what pairing costs.
        if any(len(v) != len(shared) for v in ids.values()):
            print(
                f"  {ds}: pairing on {len(shared)} shared episodes "
                + ", ".join(f"{m} scored {len(ids[m])}" for m in methods)
            )
        row = {m: {e: i for i, e in enumerate(ids[m])} for m in methods}
        picks = rng.integers(0, len(shared), size=(N_BOOT, len(shared)))
        for m in methods:
            d = data[ds][m]
            # One draw of episode ids, translated into each method's own rows, so
            # every method is scored on the same episodes in the same order.
            take = np.array([row[m][e] for e in shared])
            arr = {k: np.empty(N_BOOT) for k in METRICS}
            for b in range(N_BOOT):
                s = take[picks[b]]
                fl = d["flags"][s]
                for k in ("voc", "kendall", "pearson", "mae"):
                    v = d[k][s]
                    arr[k][b] = np.nanmean(v) if np.isfinite(v).any() else np.nan
                arr["auroc"][b] = (
                    success_auroc(d["term"][s], fl) if fl.any() and (~fl).any() else np.nan
                )
            for k in METRICS:
                per_suite[k][m].append(arr[k])
    for k in METRICS:
        for m in methods:
            boot[k][m] = np.nanmean(np.stack(per_suite[k][m], 0), axis=0)

    results = {}
    for k in METRICS:
        means = {m: float(np.nanmean(boot[k][m])) for m in methods}
        ci = {
            m: (float(np.nanpercentile(boot[k][m], 2.5)), float(np.nanpercentile(boot[k][m], 97.5)))
            for m in methods
        }
        order = sorted(methods, key=lambda m: means[m], reverse=HIGHER_BETTER[k])
        praw = {}
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                diff = boot[k][a] - boot[k][b]
                diff = diff[np.isfinite(diff)]
                p = 1.0 if len(diff) == 0 else 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())
                praw[frozenset((a, b))] = min(1.0, p)
        padj = _holm(praw)
        different = {pair for pair, p in padj.items() if p < ALPHA}
        letters = cld_letters(order, different)
        results[k] = {"means": means, "ci": ci, "order": order, "cld": letters}
    return results, methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", type=Path, required=True)
    ap.add_argument("--sweep-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/out/progress_consolidated"))
    ap.add_argument("--sort-by", default="voc", choices=METRICS)
    ap.add_argument(
        "--stats",
        action="store_true",
        help="also emit per-metric CI + compact-letter-display (paired episode bootstrap)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # cell[method][dataset] = metric dict
    cell: dict[str, dict[str, dict]] = {m: {} for m in METHOD_KEYS}
    for ds, prefix in DATASETS.items():
        for method in METHOD_KEYS:
            path = resolve_curve_path(args.curves_dir, prefix, method)
            if path is None:
                continue
            r = metrics_for(path)
            if r is not None:
                cell[method][ds] = r

    # ---- aggregate (mean over datasets present) ----
    agg = {}
    for method in METHOD_KEYS:
        rows = list(cell[method].values())
        if not rows:
            continue
        agg[method] = {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS}
        agg[method]["n"] = len(rows)
    order = sorted(agg, key=lambda m: agg[m][args.sort_by], reverse=HIGHER_BETTER[args.sort_by])

    lines = [
        "# Progress estimation — AGGREGATE (mean over LIBERO goal/object/spatial)",
        "",
        f"All methods, all metrics; sorted by {args.sort_by.upper()} "
        f"({'higher' if HIGHER_BETTER[args.sort_by] else 'lower'}=better). "
        "VOC/Kendall/Pearson/MAE over successful episodes vs linear GT; AUROC = success detection.",
        "NOTE: graph consumes pi0.5 policy embeddings; VLMs consume RGB frames (not input-matched). "
        "The trivial frame_index (GT-order) baseline is excluded.",
        "",
        "| rank | method | input | n_ds | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(order, 1):
        a = agg[m]
        lines.append(
            f"| {i} | {_display(m)} | {_input(m)} | {a['n']} | "
            + " | ".join(fmt(a[k], k) for k in METRICS)
            + " |"
        )
    agg_md = "\n".join(lines) + "\n"

    # ---- per-dataset breakdown (long) ----
    b = [
        "# Progress estimation — PER-DATASET BREAKDOWN",
        "",
        "| dataset | method | input | VOC ↑ | Kendall ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        present = [(m, cell[m][ds]) for m in METHOD_KEYS if ds in cell[m]]
        present.sort(key=lambda x: x[1][args.sort_by], reverse=HIGHER_BETTER[args.sort_by])
        for m, r in present:
            b.append(
                f"| {ds} | {_display(m)} | {_input(m)} | "
                + " | ".join(fmt(r[k], k) for k in METRICS)
                + " |"
            )
    brk_md = "\n".join(b) + "\n"

    # ---- optional: graph-only cross-dataset aggregate (E-PE4) ----
    sweep_md = ""
    if args.sweep_json and args.sweep_json.exists():
        import json

        d = json.loads(args.sweep_json.read_text())
        scheme_key = {"graph_stage": "graph_direct:stage", "graph_absorb": "graph_direct:absorb"}
        rows = {m: [] for m in scheme_key}
        skipped_on_eval = 0
        for _, r in d.items():
            if "per_k" not in r:
                continue
            # A K chosen by scoring every candidate on the folds being reported
            # is selection on the published metric. Not aggregated, so a re-run
            # of this script cannot regenerate the table that was retracted for
            # exactly that.
            if not chose_k_off_eval(r):
                skipped_on_eval += 1
                continue
            # Same trap as report.py: a dataset where no K scored finite carries
            # `per_k` with `best_k: None`, and `per_k["None"]` is a KeyError. It
            # has no best-K row to aggregate, so it is not one of the datasets
            # this mean is over.
            if r.get("best_k") is None:
                continue
            best = r["per_k"][str(r["best_k"])]
            for m, sk in scheme_key.items():
                rows[m].append(best[sk])
        s = [
            "",
            f"# Graph-only cross-dataset AGGREGATE (E-PE4, {len(rows['graph_stage'])} datasets, nested K)",
            "",
            "Graph schemes only (VLMs were not run on these). mean over datasets.",
            "K is chosen inside each outer fold, on a validation split of that fold's train half,",
            "so every number here is scored on a test fold that had no say in the choice.",
            *(
                [
                    f"SKIPPED {skipped_on_eval} dataset(s) whose K was chosen on the eval fold: "
                    "those numbers are a maximum over the data they call unseen.",
                ]
                if skipped_on_eval
                else []
            ),
            "",
            "| method | n_ds | VOC ↑ | Pearson ↑ | MAE ↓ | AUROC ↑ |",
            "|---|---|---|---|---|---|",
        ]
        for m, sk in scheme_key.items():
            vals = {
                k: np.nanmean([x[k] for x in rows[m]]) for k in ["voc", "pearson", "mae", "auroc"]
            }
            s.append(
                f"| {_display(m)} | {len(rows[m])} | {vals['voc']:.3f} | {vals['pearson']:.3f} | "
                f"{vals['mae']:.3f} | {vals['auroc']:.3f} |"
            )
        sweep_md = "\n".join(s) + "\n"

    # ---- optional: CI + CLD (--stats) ----
    stats_md = ""
    if args.stats:
        res, methods = compute_stats(args.curves_dir, np.random.default_rng(args.seed))
        s = [
            "# Progress estimation — CI + CLD "
            f"(paired episode bootstrap, N={N_BOOT}, mean over LIBERO goal/object/spatial)",
            "",
            "Per metric: mean [95% CI], sorted best->worst, with a Compact Letter Display. "
            f"Methods that SHARE a letter are NOT significantly different (paired bootstrap, "
            f"Holm-adjusted, alpha={ALPHA}); methods sharing NO letter differ significantly.",
            "",
        ]
        label = {
            "voc": "VOC ↑",
            "kendall": "Kendall ↑",
            "pearson": "Pearson ↑",
            "mae": "MAE ↓",
            "auroc": "success-AUROC ↑",
        }
        for k in METRICS:
            r = res[k]
            s.append(f"## {label[k]}")
            s.append("| method | input | mean [95% CI] | CLD |")
            s.append("|---|---|---|---|")
            for m in r["order"]:
                lo, hi = r["ci"][m]
                s.append(
                    f"| {_display(m)} | {_input(m)} | {r['means'][m]:.3f} [{lo:.3f}, {hi:.3f}] "
                    f"| {r['cld'][m]} |"
                )
            s.append("")
        stats_md = "\n".join(s) + "\n"
        (args.out_dir / "stats_ci_cld.md").write_text(stats_md)

    # ---- write ----
    (args.out_dir / "aggregate.md").write_text(agg_md)
    (args.out_dir / "breakdown.md").write_text(brk_md)
    combined = agg_md + "\n" + brk_md + ("\n" + sweep_md if sweep_md else "")
    combined += "\n" + stats_md if stats_md else ""
    (args.out_dir / "progress_tables.md").write_text(combined)

    with open(args.out_dir / "aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "input", "n_datasets", *METRICS])
        for m in order:
            w.writerow([_display(m), _input(m), agg[m]["n"], *[agg[m][k] for k in METRICS]])
    with open(args.out_dir / "breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "input", *METRICS])
        for ds in DATASETS:
            for m in METHOD_KEYS:
                if ds in cell[m]:
                    w.writerow([ds, _display(m), _input(m), *[cell[m][ds][k] for k in METRICS]])

    print(combined)
    print(f"wrote {args.out_dir}/{{aggregate,breakdown,progress_tables}}.md + .csv")


if __name__ == "__main__":
    main()
