"""Build and run the ablation grid for one dataset.

Replaces the per-dataset shell wrappers. Two things made those hard to
generalise and both are handled here rather than duplicated per task.

The representation axis is not one axis. ``rgrl_*`` and ``dinov3`` are *layers*
of the same cheap array reader, which ``run_sweep`` sweeps natively with
``layers=[...]``, while ``state`` is a different extractor and needs its own
pass with a features override. They accumulate into one sweep dir because
``run_sweep`` skips a job whose ``metrics.json`` exists and regenerates its
reports over everything completed.

And the sweep needs a RAPIDS environment. cuML and torch do not coexist here, so
this shells out to the cuml-sweep interpreter explicitly instead of assuming the
caller is already in the right venv.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import time

from behavior_graphs.ablation_study.config import DatasetConfig

#: The interpreter that has cuML. The project venv does not, and the vision venv
#: does not either; a sweep launched from the wrong one fails at the reduction
#: stage after doing all the feature work.
#:
#: Overridable because the requirement is per *grid*, not per repo: cuML does
#: not coexist with every other dependency in one environment. Point
#: `$BEHAVIOR_GRAPHS_SWEEP_PYTHON` at the interpreter the grid's reducers need;
#: it defaults to the one running this process.
CUML_PYTHON = pathlib.Path(os.environ.get("BEHAVIOR_GRAPHS_SWEEP_PYTHON", sys.executable))

REPO = pathlib.Path(__file__).resolve().parents[3]


#: Representations selected by a features *config* rather than by an embedding
#: layer. `layers=[...]` names an npz under `policy_embeddings/`, which is how
#: the policy taps and dinov3 are addressed; these instead swap the extractor,
#: so they cannot be batched into one `layers=` pass and get a command each.
FEATURE_CONFIGS = {
    "state": "state",
    "state_dinov3": "state_dinov3",
}


def commands(
    cfg: DatasetConfig,
    layers: tuple[str, ...] | None = None,
    per_layer: bool = False,
    grid: str | None = None,
) -> list[tuple[str, list[str]]]:
    """The (label, argv) passes needed to cover every representation.

    Grouped by grid, not just by extractor. A representation whose features are
    much wider than the others gets its own subgrid (see `grids` in the config),
    and passes accumulate into one sweep dir because run_sweep skips a job whose
    metrics.json exists.

    ``per_layer`` emits one pass per representation instead of one per grid,
    which is what :func:`run` needs to parallelise. The partition matters: two
    runners given disjoint layers can never pick the same job, so they need no
    coordination beyond the ``jobs.json`` lock that already exists. Handing both
    the same layer would be safe but wasteful, since each would recompute what
    the other is midway through.
    """
    reps = tuple(layers) if layers else cfg.representations
    split = [f"split.file={cfg.splits_path}", f"split.fold={cfg.sweep_fold}"]
    # An explicit grid overrides the per-representation mapping for every arm,
    # which is what running the complete core design needs: the core is a
    # property of the *design*, not of one representation, so `state` has to
    # run it on the same dials as the rest rather than on its own grid.
    grid_for = (lambda _layer: grid) if grid else cfg.grid_for

    def base(grid: str) -> list[str]:
        return [
            str(CUML_PYTHON),
            str(REPO / "scripts" / "run_sweep.py"),
            f"grid={grid}",
            f"sweep_dir={cfg.sweep_dir}",
            f"datasets={{{cfg.key}: {cfg.root}}}",
        ]

    by_grid: dict[str, list[str]] = {}
    for r in reps:
        if r in FEATURE_CONFIGS:
            continue
        by_grid.setdefault(grid_for(r), []).append(r)

    out: list[tuple[str, list[str]]] = []
    for grid, group in sorted(by_grid.items()):
        groups = [[r] for r in group] if per_layer else [group]
        for g in groups:
            out.append(
                (
                    f"{grid}: " + ",".join(g),
                    base(grid) + [f"layers=[{','.join(g)}]", f"extra_overrides={split}"],
                )
            )
    for r in reps:
        if r not in FEATURE_CONFIGS:
            continue
        grid_name = grid_for(r)
        override = f"pipeline/features@features={FEATURE_CONFIGS[r]}"
        out.append(
            (
                f"{grid_name}: {r}",
                base(grid_name) + [f"extra_overrides={split + [override]}"],
            )
        )
    return out


def plan(cfg: DatasetConfig, layers: tuple[str, ...] | None = None, grid: str | None = None) -> str:
    lines = [
        f"{cfg.name}: sweep_dir={cfg.sweep_dir}",
        f"  fold={cfg.sweep_fold}  split={cfg.splits_path}",
        f"  representations: {', '.join(layers or cfg.representations)}",
        "",
    ]
    for label, argv in commands(cfg, layers, grid=grid):
        lines += [f"  pass: {label}", "    " + " ".join(shlex.quote(a) for a in argv), ""]
    return "\n".join(lines)


#: GPUs to spread concurrent runners over, round-robin. cuML picks device 0
#: unless told otherwise, and the runner's environment is built from scratch
#: (see `_env`), so without this every worker would pile onto one GPU.
GPUS = (0, 1)

#: Threads each runner's numeric libraries may use. Unset, every one of them
#: sizes its pool to all 32 cores independently, so the runners do not share the
#: machine -- they each try to own it. Measured on the ablation grid: eight
#: runners at 82 threads apiece drove load average to 87 on 32 cores, which is
#: 2.7x oversubscribed and spends the difference on context switches.
#:
#: The right value is cores / concurrent runners, so it cannot be a constant
#: here; `run` computes it from ``workers`` and this is only the fallback for a
#: single runner. Override with ``$BEHAVIOR_GRAPHS_SWEEP_THREADS``.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _threads(workers: int) -> int:
    """Per-runner thread budget: the machine's cores split over the runners."""
    override = os.environ.get("BEHAVIOR_GRAPHS_SWEEP_THREADS")
    if override:
        return max(1, int(override))
    cores = os.cpu_count() or 8
    return max(1, cores // max(1, workers))


def _env(gpu: int | None = None, threads: int | None = None) -> dict[str, str]:
    """A minimal environment for a runner.

    Built rather than inherited so a sweep does not pick up whatever happened to
    be exported in the launching shell -- in particular a stray
    ``CUDA_VISIBLE_DEVICES`` or a ``PYTHONPATH`` pointing at another checkout.
    Which also means thread limits have to be set here explicitly: nothing the
    caller exported survives (see :data:`THREAD_VARS`).
    """
    env = {
        "PYTHONPATH": str(REPO / "src"),
        "DATA_PATH": "${DATA_PATH}",
        "PATH": "/usr/bin:/bin",
        "HOME": str(pathlib.Path.home()),
    }
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for name in THREAD_VARS:
        env[name] = str(threads if threads is not None else _threads(1))
    return env


def run(
    cfg: DatasetConfig,
    layers: tuple[str, ...] | None = None,
    log: pathlib.Path | None = None,
    workers: int = 1,
    grid: str | None = None,
) -> str:
    """Run the passes, appending to ``log``.

    ``workers=1`` runs them in sequence, one grid-grouped pass at a time.
    Anything higher splits per representation and runs that many at once.

    Parallelising is worth it because a single runner leaves the machine idle:
    measured mid-sweep on this 32-core box, one runner held ~6 cores at load
    average 5.9 and both GPUs sat at 0%. Jobs are independent and a sweep
    directory is already safe to fill from several processes -- the ``jobs.json``
    lock exists for exactly that -- so the only thing needed is a partition that
    stops two runners racing for the same job, which is what per-layer passes
    give.

    Each worker holds its own features and windows in memory, so ``workers`` is
    bounded by RAM rather than cores: a legos runner peaked at 22 GB against a
    bottles runner's ~7 GB.
    """
    if not CUML_PYTHON.exists():
        raise SystemExit(
            f"no sweep interpreter at {CUML_PYTHON} (set $BEHAVIOR_GRAPHS_SWEEP_PYTHON to override)"
        )
    dest = log or (cfg.sweep_dir.parent / f"{cfg.name}_sweep.log")
    dest.parent.mkdir(parents=True, exist_ok=True)
    env_note = f"PYTHONPATH={REPO / 'src'}"
    passes = commands(cfg, layers, per_layer=workers > 1, grid=grid)
    out = [f"{cfg.name}: {len(passes)} passes, {workers} at a time, log {dest}"]

    if workers <= 1:
        with dest.open("a") as fh:
            for label, argv in passes:
                fh.write(f"\n=== pass: {label} ({env_note}) ===\n")
                fh.flush()
                rc = subprocess.call(
                    argv,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=_env(threads=_threads(1)),
                    cwd=REPO,
                )
                done = len(list(cfg.sweep_dir.rglob("metrics.json")))
                out.append(f"  {label}: exit {rc}, {done} jobs complete")
        return "\n".join(out)

    # One log per pass when running concurrently. Interleaving several runners
    # into one file makes it unreadable, and the per-stage timings in it are the
    # only way to tell a slow sweep from a stuck one.
    running: list[tuple[str, subprocess.Popen, object]] = []
    pending = list(passes)
    while pending or running:
        while pending and len(running) < workers:
            label, argv = pending.pop(0)
            tag = label.split(": ")[-1].replace(",", "-")
            path = dest.with_name(f"{dest.stem}_{tag}{dest.suffix}")
            fh = path.open("a")
            fh.write(f"\n=== pass: {label} ({env_note}) ===\n")
            fh.flush()
            gpu = GPUS[len(running) % len(GPUS)] if GPUS else None
            proc = subprocess.Popen(
                argv,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=_env(gpu, threads=_threads(workers)),
                cwd=REPO,
            )
            out.append(f"  started {label} -> {path.name} (pid {proc.pid}, gpu {gpu})")
            running.append((label, proc, fh))
        for entry in list(running):
            label, proc, fh = entry
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            running.remove(entry)
            done = len(list(cfg.sweep_dir.rglob("metrics.json")))
            out.append(f"  {label}: exit {rc}, {done} jobs complete")
        if running:
            time.sleep(10)
    return "\n".join(out)
