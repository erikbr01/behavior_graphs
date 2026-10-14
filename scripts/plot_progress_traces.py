"""Plot per-episode progress estimates from all methods on one axis per episode.

Loads the *_curves.npz saved by scripts/run_progress_bench.py. Each method's curve
has a different length (graph windows vs VLM query points vs ROBOMETER frames), so
every curve is drawn against NORMALIZED time in [0,1]; the dashed diagonal is the
linear-GT progress reference. One subplot per episode, laid out as a grid;
successes and failures are labeled.

Usage:
  plot_progress_traces.py --curves-dir analysis/out/... --out traces.png \
    --method "graph:lg_pool500_graph_direct_stage_curves.npz" \
    --method "robometer-libero:lg_pool500_rbmlibero_robometer_curves.npz" \
    --method "topreward:lg_pool500_top_topreward_curves.npz" \
    --per-task 1 --n-tasks 10 --task-block 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from behavior_graphs.progress_estimation.io import load_curves  # noqa: E402


def load(path):
    c = load_curves(path)
    return {int(i): (cv, bool(f)) for cv, f, i in zip(c.curves, c.flags, c.idx)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", type=Path, required=True)
    ap.add_argument("--method", action="append", required=True, help="label:file.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=1, help="episodes per task to plot")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--task-block", type=int, default=100)
    ap.add_argument("--instr", action="append", default=[], help="optional task titles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {}
    for spec in args.method:
        label, fn = spec.split(":", 1)
        methods[label] = load(args.curves_dir / fn)
    labels = list(methods)
    common = sorted(set.intersection(*[set(m) for m in methods.values()]))

    by_task = {}
    for i in common:
        by_task.setdefault(i // args.task_block, []).append(i)
    tasks = sorted(by_task)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for t in tasks:
        pool = by_task[t]
        chosen += list(rng.choice(pool, min(args.per_task, len(pool)), replace=False))

    ncol = args.n_tasks if args.per_task == 1 else args.per_task
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False, sharey=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for ax, ep in zip(axes.ravel(), chosen):
        x = np.linspace(0, 1, 50)
        ax.plot(x, x, "--", color="gray", lw=1, alpha=0.7, label="GT (linear)")
        ok = None
        for lab, col in zip(labels, colors):
            c, f = methods[lab][ep]
            ok = f
            t = np.linspace(0, 1, len(c))
            ax.plot(t, c, color=col, lw=1.6, alpha=0.9, label=lab)
        ax.set_title(
            f"ep{ep} (task {ep // args.task_block}, {'succ' if ok else 'FAIL'})", fontsize=8
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("normalized episode time")
    fig.supylabel("predicted progress")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({len(chosen)} episodes, methods={labels})")


if __name__ == "__main__":
    main()
