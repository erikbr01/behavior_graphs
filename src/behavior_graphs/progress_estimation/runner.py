"""Benchmark runner: score a collection of methods with a collection of metrics
over one dataset, and emit the same report/metrics/curve artifacts as the original
scripts.

A :class:`SplitSpec` produces the (train, eval) folds; for each fold every method
is fit on the train view and scored on the eval view, metrics are computed per fold
and aggregated (mean ± std over folds), and per-episode curves are accumulated
(idx-keyed) and saved in the canonical ``*_curves.npz`` schema so downstream
analysis scripts keep working.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from behavior_graphs.progress_estimation.io import EpisodeSet, save_curves


# ------------------------------------------------------------------------- splits
class SplitSpec(ABC):
    @abstractmethod
    def folds(self, flags: list[bool]) -> list[tuple[list[int], list[int]]]:
        """Return a list of (train_indices, eval_indices)."""


class KFoldSplit(SplitSpec):
    """Stratified k-fold over success flags (n_folds=1 -> single stratified split)."""

    def __init__(self, n_folds: int = 3, test_frac: float = 0.3, seed: int = 0):
        self.n_folds, self.test_frac, self.seed = n_folds, test_frac, seed

    def folds(self, flags):
        rng = np.random.default_rng(self.seed)
        idx_s = rng.permutation([i for i, ok in enumerate(flags) if ok])
        idx_f = rng.permutation([i for i, ok in enumerate(flags) if not ok])
        out = []
        if self.n_folds == 1:
            te = set(
                list(idx_s[: int(len(idx_s) * self.test_frac)])
                + list(idx_f[: int(len(idx_f) * self.test_frac)])
            )
            return [([i for i in range(len(flags)) if i not in te], sorted(te))]
        for k in range(self.n_folds):
            te = set(list(idx_s[k :: self.n_folds]) + list(idx_f[k :: self.n_folds]))
            out.append(([i for i in range(len(flags)) if i not in te], sorted(te)))
        return out


class AlignedSplit(SplitSpec):
    """Single split: eval on the first ``eval_per_task`` episodes of each task block,
    fit on the complement (task blocks are contiguous, block size = n // n_tasks).
    Matches the VLM ``--per-task-limit`` eval set so both can score the same episodes."""

    def __init__(self, eval_per_task: int = 50, n_tasks: int = 10):
        self.eval_per_task, self.n_tasks = eval_per_task, n_tasks

    def folds(self, flags):
        n = len(flags)
        per = max(1, n // self.n_tasks)
        # `eval_per_task >= per` takes every index of every block, leaving the
        # fit an empty list -- which surfaces much later as a bare
        # `np.concatenate([])` inside a backbone. The defaults (50, 10) do this
        # to any dataset under 1000 episodes, so say so here instead.
        if self.eval_per_task >= per:
            raise ValueError(
                f"AlignedSplit would leave nothing to train on: eval_per_task="
                f"{self.eval_per_task} covers the whole of each {per}-episode task block "
                f"({n} episodes / n_tasks={self.n_tasks}). Set eval_per_task below {per}, "
                f"or lower n_tasks if the blocks really are bigger than that."
            )
        te = [i for i in range(n) if (i % per) < self.eval_per_task]
        tr = [i for i in range(n) if (i % per) >= self.eval_per_task]
        return [(tr, te)]


# ------------------------------------------------------------------------- runner
_ARROW = lambda hb: "↑" if hb else "↓"  # noqa: E731


def _fmt(value) -> str:
    """A metric for the progress line, or ``n/a`` when it is missing or NaN."""
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    return f"{value:.3f}"


class BenchmarkRunner:
    def __init__(self, metrics, split: SplitSpec, out_dir: Path | None = None):
        self.metrics = list(metrics)
        self.split = split
        self.out_dir = Path(out_dir) if out_dir else None

    def run(self, data: EpisodeSet, methods) -> dict:
        methods = list(methods)
        folds = self.split.folds(data.flags)
        # One metric names the per-fold progress line, chosen once so the
        # columns stay comparable: `voc` when it was selected, otherwise
        # whatever came first. A metric set without `voc` -- which
        # `metric/auroc.yaml` alone produces -- used to format None and raise
        # here, after the fold's models had already been fit.
        names = [metric.name for metric in self.metrics]
        headline = "voc" if "voc" in names else (names[0] if names else None)
        per_fold: dict[str, list[dict]] = {m.name: [] for m in methods}
        acc: dict[str, dict] = {
            m.name: {"curves": [], "flags": [], "idx": [], "instr": []} for m in methods
        }

        for fi, (tr, te) in enumerate(folds):
            train_view, eval_view = data.view(tr), data.view(te)
            for m in methods:
                m.reset()
            for m in methods:
                m.fit(train_view)
                curves = m.predict(eval_view)
                flags = eval_view.flags
                per_fold[m.name].append(
                    {metric.name: metric.compute(curves, flags) for metric in self.metrics}
                )
                a = acc[m.name]
                a["curves"].extend(curves)
                a["flags"].extend(flags)
                a["idx"].extend(te)
                if m.modality == "video":
                    a["instr"].extend(eval_view.instruction(i) for i in te)
            if headline:
                print(
                    f"  fold {fi}: "
                    + " | ".join(
                        f"{m.name} {headline}={_fmt(per_fold[m.name][-1].get(headline))}"
                        for m in methods
                    )
                )

        summary = {}
        for m in methods:
            summary[m.name] = {
                metric.name: self._agg(per_fold[m.name], metric.name) for metric in self.metrics
            }
            if self.out_dir is not None:
                a = acc[m.name]
                save_curves(
                    self.out_dir,
                    data.name,
                    m.output_tag,
                    a["curves"],
                    a["flags"],
                    a["idx"],
                    instr=(a["instr"] or None) if m.modality == "video" else None,
                )

        if self.out_dir is not None:
            self._write_reports(data, methods, summary, folds)
        return summary

    @staticmethod
    def _agg(rows, key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    def _write_reports(self, data, methods, summary, folds):
        # MERGE, do not overwrite. A cross-dataset run is sharded by method --
        # one process per VLM, because the checkpoints do not share a GPU -- so
        # several runs write this same file for one dataset. Overwriting left
        # each dataset's metrics.json holding only whichever pass finished last
        # (libero_goal ended up with `robometer_fft` alone, out of seven
        # methods whose curves were all sitting next to it), and everything
        # reading metrics.json then saw one method.
        path = self.out_dir / f"{data.name}_metrics.json"
        merged: dict = {}
        if path.is_file():
            try:
                merged = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                merged = {}
        merged.update(summary)  # this run's methods win on a rerun
        path.write_text(json.dumps(merged, indent=2))
        header = " | ".join(f"{mt.name.upper()} {_ARROW(mt.higher_better)}" for mt in self.metrics)
        sep = "|".join(["---"] * (len(self.metrics) + 1))
        n_folds = len(folds)
        # Counts and the fold count in the shape the app's _SPLIT_KFOLD regex
        # reads: the old "(N eps, K fold(s), eval=M)" carried no succ/fail, so
        # neither regex matched and the metrics-table caption silently lost its
        # split note on every run this writes.
        n_succ = int(sum(bool(f) for f in data.flags))
        desc = (
            f"{data.n} eps, {n_succ}succ/{data.n - n_succ}fail, "
            f"{n_folds}-fold, eval={len(folds[0][1])}"
            if n_folds
            else f"{data.n} eps, {n_succ}succ/{data.n - n_succ}fail"
        )
        lines = [
            f"# Progress-estimation benchmark: {data.name} ({desc})",
            "",
            "Metrics match TOPReward/ROBOMETER (higher=better unless ↓).",
            "",
            f"| method | {header} |",
            f"|{sep}|",
        ]
        for m in methods:
            cells = " | ".join(
                f"{summary[m.name][mt.name][0]:.3f}±{summary[m.name][mt.name][1]:.3f}"
                for mt in self.metrics
            )
            lines.append(f"| {m.name} | {cells} |")
        report = "\n".join(lines) + "\n"
        (self.out_dir / f"{data.name}_report.md").write_text(report)
        print("\n" + report)
