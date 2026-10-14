"""Sweep runner: grouped hyperparameter grids over the pipeline.

Hydra's ``-m`` multirun takes a flat cross-product, which breaks for stage
sweeps: ``clustering.n_clusters`` only exists for k-means, GMM has
``n_components``, HDBSCAN has neither. A sweep *spec* instead nests each
stage option with its own parameter grid, and the runner takes the
cross-product of (option × its grid) across stage groups::

    stages:
      windowing:
        sliding:
          width: [1, 3, 5, 8]            # explicit list
          stride: 2                       # scalar = fixed
      reduction:
        pca: {n_components: [8, 16, 32]}
        umap:
          n_neighbors: {_target_: behavior_graphs.pipeline.grid.irange, _args_: [10, 40, 10]}
      clustering:
        kmeans: {n_clusters: {_target_: behavior_graphs.pipeline.grid.irange, _args_: [4, 20, 2]}}
        hdbscan: {min_cluster_size: [10, 15, 30]}

This module is the library behind ``scripts/run_sweep.py`` (the hydra
entry point). Specs come from the ``grid`` config group
(``configs/grid/*.yaml``); :func:`run_sweep` takes the resolved spec dict.

Specs are validated up front against the config groups (unknown options or
misspelled parameters fail before any job runs, with the valid choices in
the error). Jobs are ordered upstream-first (downstream params vary
fastest), so the stage cache (:mod:`behavior_graphs.pipeline.cache`) turns
the sweep into "compute each upstream artifact once, then iterate the
cheap stages".

Outputs, under the sweep dir:

- ``[<dataset>/]<hash12>/`` — a self-contained run dir per job, where
  ``hash12`` is a 12-char SHA-256 prefix of (sorted overrides, resolved
  dataset path).  The hash is stable: re-running identical jobs lands in
  the same dir (skip if ``metrics.json`` exists) and different grids
  (PCA+UMAP then SAE) never collide.
- ``spec.yaml`` — the resolved sweep config from the most recent run;
- ``datasets.yaml`` — accumulated ``{name: path}`` mapping across all
  runs in this dir (so reports always cover the full scope);
- ``jobs.json`` — accumulated ``hash → overrides`` mapping; written
  atomically before any jobs launch; new grids are merged in, existing
  entries are never overwritten;
- ``failures.json`` — jobs that raised an exception this run; entries
  for jobs that recovered (``metrics.json`` now exists) are removed;
- ``sweep_index.csv`` — one row per completed job: hyperparameters +
  all metrics + the composite selection score (normalized per dataset);
- ``best_per_k.csv`` — recommended hyperparameters per K (and dataset);
- ``sensitivity_<metric>.csv`` — per-hyperparameter sensitivity tables.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from behavior_graphs.core.io import atomic_write_json
from behavior_graphs.core.stages import STAGE_GROUPS
from behavior_graphs.pipeline.cache import canonical_json
from behavior_graphs.pipeline.run_discovery import discover_all_job_dirs

log = logging.getLogger(__name__)

#: Metrics for which a sensitivity table is written.
SENSITIVITY_METRICS = (
    "silhouette",
    "davies_bouldin",
    "swap_rate",
    # Only populated for rollout dirs carrying subtask annotations; rows
    # without the metric simply come out empty rather than failing.
    "boundary_gain",
    "boundary_f1",
)


# ──────────────────────────────────────────────────────────── spec expansion


def format_override_value(value: Any) -> str:
    """Render a python value as a hydra CLI override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def expand_values(values: Any) -> list[Any]:
    """One grid entry -> list of values.

    Accepts a scalar (fixed value), an explicit list, or a ``_target_``
    generator config (see :mod:`behavior_graphs.pipeline.grid`) that returns a
    list when instantiated.
    """
    if isinstance(values, Mapping):
        if "_target_" not in values:
            raise ValueError(
                "a grid-value mapping must be a _target_ generator "
                f"(see behavior_graphs.pipeline.grid); got keys {sorted(values)}"
            )
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        produced = instantiate(OmegaConf.create(dict(values)))
        return list(produced)
    if isinstance(values, list | tuple):
        return list(values)
    return [values]


def expand_spec(spec: dict[str, Any]) -> list[list[str]]:
    """Expand a sweep spec into one hydra-override list per job.

    ``stages`` sweeps config groups; ``params`` sweeps plain root-level keys of
    the pipeline config (``episode_norm``, say), crossed with the stage
    combinations. Root keys have to go here rather than in ``extra_overrides``,
    which are not part of a job key -- two values of one would collide into a
    single job directory, which is the failure 55a5369 fixed.
    """
    stages: dict[str, dict[str, Any]] = spec.get("stages") or {}
    unknown = set(stages) - set(STAGE_GROUPS)
    if unknown:
        raise ValueError(
            f"unknown stage group(s) in spec: {sorted(unknown)} (valid: {list(STAGE_GROUPS)})"
        )

    per_group: list[list[list[str]]] = []  # group -> option-combos -> overrides
    for group in STAGE_GROUPS:
        options = stages.get(group)
        if not options:
            continue
        combos: list[list[str]] = []
        for option, grid in options.items():
            grid = grid or {}
            keys = sorted(grid)
            value_lists = [expand_values(grid[key]) for key in keys]
            # Stage configs live under configs/pipeline/<group>/ in the Hydra
            # config root; the group name itself is the Python config key.
            group_path = f"pipeline/{group}"
            for values in itertools.product(*value_lists):
                # Hydra requires the @<package> qualifier when the defaults entry
                # uses a package override (pipeline/reduction@reduction).
                combo = [f"{group_path}@{group}={option}"]
                combo += [f"{group}.{k}={format_override_value(v)}" for k, v in zip(keys, values)]
                combos.append(combo)
        per_group.append(combos)

    params: dict[str, Any] = spec.get("params") or {}
    if params:
        keys = sorted(params)
        value_lists = [expand_values(params[k]) for k in keys]
        per_group.append(
            [
                [f"{k}={format_override_value(v)}" for k, v in zip(keys, values)]
                for values in itertools.product(*value_lists)
            ]
        )

    base: list[str] = list(spec.get("base_overrides") or [])
    jobs = []
    for parts in itertools.product(*per_group):
        jobs.append(base + [o for combo in parts for o in combo])
    return jobs


# ──────────────────────────────────────────────────────────── spec validation


def validate_spec(spec: dict[str, Any], config_path: Path | None = None) -> None:
    """Fail fast, before any job runs, with the valid choices in the error.

    Checks stage options against ``configs/<group>/*.yaml``, grid parameter
    names against each option's config keys, and ``selection`` metric keys
    against the metric registry.
    """
    if config_path is None:
        from behavior_graphs.pipeline.run import CONFIG_PATH

        config_path = Path(CONFIG_PATH)

    errors: list[str] = []
    stages: dict[str, dict[str, Any]] = spec.get("stages") or {}
    unknown = set(stages) - set(STAGE_GROUPS)
    if unknown:
        errors.append(f"unknown stage group(s): {sorted(unknown)} (valid: {list(STAGE_GROUPS)})")
    for group, options in stages.items():
        if group in unknown:
            continue
        group_path = f"pipeline/{group}"
        available = sorted(p.stem for p in (config_path / group_path).glob("*.yaml"))
        for option, grid in (options or {}).items():
            if option not in available:
                errors.append(f"stages.{group}: no option '{option}' (available: {available})")
                continue
            cfg = yaml.safe_load((config_path / group_path / f"{option}.yaml").read_text()) or {}
            allowed = sorted(k for k in cfg if k != "_target_")
            bad = sorted(set(grid or {}) - set(allowed))
            if bad:
                errors.append(
                    f"stages.{group}.{option}: unknown parameter(s) {bad} (valid: {allowed})"
                )

    params = spec.get("params") or {}
    if params:
        root = yaml.safe_load((config_path / "config.yaml").read_text()) or {}
        known = {k for k in root if k != "defaults"}
        bad = sorted(set(params) - known)
        if bad:
            errors.append(f"params: unknown root config key(s) {bad} (valid: {sorted(known)})")
        clash = sorted(set(params) & set(STAGE_GROUPS))
        if clash:
            errors.append(f"params: {clash} are stage groups; sweep them under `stages`")

    selection = spec.get("selection") or {}
    if selection:
        from behavior_graphs.model_selection.scoring import METRICS

        for field in ("weights", "constraints"):
            bad = sorted(set(selection.get(field) or {}) - set(METRICS))
            if bad:
                errors.append(
                    f"selection.{field}: unknown metric(s) {bad} (valid: {sorted(METRICS)})"
                )

    datasets = spec.get("datasets")
    if datasets is not None and not isinstance(datasets, Mapping):
        errors.append("datasets must be a mapping {name: path}")

    if errors:
        raise ValueError("invalid sweep spec:\n  - " + "\n  - ".join(errors))


def resolve_datasets(
    spec: dict[str, Any], datasets: dict[str, Path] | None = None
) -> dict[str, Path]:
    """{name: rollout dir path}. An explicit ``datasets`` kwarg takes
    precedence over ``spec["datasets"]``."""
    if datasets is not None:
        return {k: Path(v) for k, v in datasets.items()}
    spec_datasets = spec.get("datasets")
    if spec_datasets:
        return {str(name): Path(p) for name, p in dict(spec_datasets).items()}
    raise ValueError("no data given: set datasets in the spec or config")


# ──────────────────────────────────────────────── job identity + accumulation


def _job_hash(ds_path: Path, overrides: list[str]) -> str:
    """12-char hex fingerprint of (resolved dataset path, sorted overrides).

    Stable across runs: identical (path, overrides) → same hash → same dir →
    metrics.json skip.  Different grids (PCA vs SAE) produce different hashes
    so multiple grids can accumulate in one sweep dir without collision.
    12 chars = 6 bytes; birthday collision probability is ~1/16M at 864 jobs.
    """
    canonical = "\n".join(sorted(overrides) + [str(ds_path.resolve())])
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def data_digest(rollouts: Any) -> str:
    """Fingerprint of a rollout dir's episode content, for staleness checks.

    Over the (length, outcome) pairs, not the metadata file's mtime, so a
    re-assembly reproducing identical episodes doesn't read as a change.
    """
    eps = [(int(e.num_samples), e.success) for e in rollouts.episodes()]
    return hashlib.sha256(canonical_json(eps).encode()).hexdigest()[:16]


def _load_digests(sweep_dir: Path) -> dict[str, str]:
    """Recorded per-dataset digests, or {} when none have been written yet.

    Unreadable is fatal, not empty: this is the only record of what the
    completed jobs were computed from.
    """
    p = sweep_dir / "data_digests.json"
    if not p.exists():
        return {}
    try:
        loaded = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{p} is unreadable ({exc}), so no dataset can be checked against the "
            "content its completed jobs used. Restore or delete the file (deleting "
            "re-records every dataset as it is now, without re-running the jobs)."
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{p} should hold a dataset->digest mapping, got {type(loaded).__name__}")
    return loaded


def check_data_digests(
    sweep_dir: Path,
    rollouts_by_name: dict[str, Any],
    allow_stale: bool = False,
) -> dict[str, str]:
    """Compare each dataset against the content it had when last swept.

    Nothing in a job's identity (``_job_hash``) or its skip condition (a
    ``metrics.json``) looks at the episodes, so re-assembling a dataset in place
    would silently keep the old results. Refuse instead, naming the dir to
    delete; ``allow_stale=True`` proceeds. Only datasets with completed jobs can
    be stale, which is what makes deleting ``sweep_dir/<dataset>`` the fix.

    Returns the digest map to persist. *allow_stale* keeps the **recorded**
    digest, so one opt-in can't disable the check for good.
    """
    recorded = _load_digests(sweep_dir)
    current = {name: data_digest(r) for name, r in rollouts_by_name.items()}
    stale = {
        n: (recorded[n], d)
        for n, d in current.items()
        if n in recorded and recorded[n] != d and _has_completed_jobs(sweep_dir, n)
    }
    if stale and not allow_stale:
        detail = "\n".join(
            f"  {n}: swept as {was}, now {now} — delete {sweep_dir / n} to redo it"
            for n, (was, now) in sorted(stale.items())
        )
        raise ValueError(
            f"{len(stale)} dataset(s) changed since they were swept, so completed "
            f"jobs no longer describe the data on disk:\n{detail}\n"
            "Pass allow_stale_data=true to sweep anyway, keeping those results."
        )
    for name, (was, now) in sorted(stale.items()):
        log.warning(
            "dataset %s changed (%s -> %s); keeping existing results and the recorded "
            "digest, so this is reported again next run",
            name,
            was,
            now,
        )
    return {**recorded, **{n: d for n, d in current.items() if n not in stale}}


def _has_completed_jobs(sweep_dir: Path, ds_name: str) -> bool:
    """Whether *ds_name* has any job dir with a metrics.json under *sweep_dir*."""
    root = sweep_dir / ds_name if ds_name else sweep_dir
    if not root.is_dir():
        return False
    return any(p.is_dir() and (p / "metrics.json").exists() for p in root.iterdir())


def _load_persisted_datasets(sweep_dir: Path) -> dict[str, Path]:
    p = sweep_dir / "datasets.yaml"
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return {str(k): Path(v) for k, v in data.items()}


def _save_persisted_datasets(sweep_dir: Path, datasets: dict[str, Path]) -> None:
    (sweep_dir / "datasets.yaml").write_text(
        yaml.safe_dump({k: str(v) for k, v in sorted(datasets.items())}, sort_keys=False)
    )


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an advisory lock on a sibling of *path* for the whole read-write.

    A lock on the file itself would not survive the atomic write, which replaces
    it by rename, so the lock lives beside it.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _merge_jobs(jobs_path: Path, new_jobs: dict[str, list[str]]) -> None:
    """Merge new_jobs into jobs.json under a lock, with an atomic write.

    Raises RuntimeError on hash collision (two different override lists
    mapping to the same key), which indicates a normalization bug in
    _job_hash rather than a real SHA-256 collision.
    Written in full before any job launches so a crash mid-sweep never
    leaves a partially-written jobs.json.

    The lock is what makes one sweep dir safe to fill from several sweep
    processes at once, which is how a multi-dataset sweep is actually run. The
    atomic write alone prevented a torn file but not a lost update: six
    concurrent processes each read, merged and replaced this file, and the
    later writers silently dropped 144 job keys per dataset for four of the
    twelve datasets in sweeps/sweep_all. Reports read jobs.json in preference
    to walking the tree, so those runs vanished from every CSV while staying on
    disk and visible to the apps.
    """
    with _exclusive(jobs_path):
        existing: dict[str, list[str]] = {}
        if jobs_path.exists():
            existing = json.loads(jobs_path.read_text())
        for key, overrides in new_jobs.items():
            if key in existing and existing[key] != overrides:
                raise RuntimeError(
                    f"Hash collision on job key {key!r} — normalization bug in _job_hash.\n"
                    f"  stored: {existing[key]}\n"
                    f"  new:    {overrides}"
                )
        merged = {**existing, **new_jobs}
        atomic_write_json(jobs_path, merged)


# ───────────────────────────────────────────────────────────────── execution


def run_sweep(
    spec: dict[str, Any],
    datasets: dict[str, Path] | None = None,
    sweep_dir: Path | None = None,
    extra_overrides: Sequence[str] = (),
    limit: int | None = None,
    dry_run: bool = False,
    write_reports: bool = True,
    allow_stale_data: bool = False,
) -> Path:
    """Execute every job of a resolved sweep spec; returns the sweep dir.

    Set ``write_reports=False`` when calling repeatedly into the same sweep
    dir (e.g. once per embedding layer) and finish with a single
    :func:`regenerate_reports` — reports always cover every completed job in
    the dir, so per-call rebuilding is redundant work.
    """
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory
    from behavior_graphs.pipeline.run import compose_pipeline_config, run_pipeline

    validate_spec(spec)
    jobs = expand_spec(spec)
    if limit is not None:
        jobs = jobs[:limit]
    try:
        datasets = resolve_datasets(spec, datasets)
    except ValueError:
        if not dry_run:
            raise
        datasets = {"": Path("<rollout-dir>")}  # spec inspection without data
    sweep_dir = Path(
        sweep_dir or spec.get("sweep_dir") or f"multirun/sweep_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    #: Extra overrides that *select a different representation* and so must be
    #: part of a job's identity. Everything else in `extra_overrides` is
    #: constant for a run -- the split file and fold -- and including it would
    #: rekey every job dir ever written for no gain.
    #:
    #: Without this, two arms that differ only in `extra_overrides` collide.
    #: `state` and `state_dinov3` are both selected by
    #: `pipeline/features@features=...` while the grid varies neither, so for a
    #: given cell they hashed to the same key: whichever ran first wrote the
    #: dir, and the second was skipped as "already complete". Measured on
    #: legos, that left zero of 72 `state_dinov3` cells despite the runner
    #: reporting all 72 complete, and on earbuds, key and whiteboard it left 2
    #: to 10 -- whichever cells the concat arm happened to reach first.
    #:
    #: Results were never misattributed, since the rescore names an arm from
    #: the `config.yaml` the winning run actually wrote. The two arms were
    #: mutually exclusive per cell, which is why the concat looked broken at
    #: 0.111 rather than absent.
    KEYED_EXTRAS = ("pipeline/features@features=",)

    def key_overrides(overrides: list[str]) -> list[str]:
        return list(overrides) + [o for o in extra_overrides if str(o).startswith(KEYED_EXTRAS)]

    def job_key(ds_name: str, ds_path: Path, overrides: list[str]) -> str:
        h = _job_hash(ds_path, key_overrides(overrides))
        return f"{ds_name}/{h}" if ds_name else h

    # Before the dry_run return: printing the job list is when you want to know
    # a resume would be refused, not after committing to the real run.
    rollouts_by_name = {
        name: RolloutDirectory(d) for name, d in datasets.items() if (d / "episodes").is_dir()
    }
    digests = check_data_digests(sweep_dir, rollouts_by_name, allow_stale=allow_stale_data)

    if dry_run:
        for ds_name, ds_dir in datasets.items():
            for overrides in jobs:
                print(f"[{job_key(ds_name, ds_dir, overrides)}] {' '.join(overrides)}")
        print(f"{len(jobs)} jobs x {len(datasets)} dataset(s) -> {sweep_dir}")
        return sweep_dir

    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate rollouts across runs so reports always cover the full scope.
    all_datasets = {**_load_persisted_datasets(sweep_dir), **datasets}
    _save_persisted_datasets(sweep_dir, all_datasets)

    # spec.yaml records the most recent grid; overwrite on each run.
    (sweep_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # Write jobs.json atomically before launching anything. A crash mid-sweep
    # never leaves a partially-written file, and a resume finds all keys intact.
    new_jobs = {
        job_key(ds_name, ds_dir, overrides): overrides
        for ds_name, ds_dir in datasets.items()
        for overrides in jobs
    }
    _merge_jobs(sweep_dir / "jobs.json", new_jobs)

    objective = spec.get("objective")
    # Before the loop: a sweep killed part-way still leaves skippable job dirs,
    # which a digest written only on clean exit would leave unguarded.
    atomic_write_json(sweep_dir / "data_digests.json", digests)

    new_failures: dict[str, str] = {}
    for ds_name, ds_dir in datasets.items():
        rdir = rollouts_by_name.get(ds_name) or RolloutDirectory(ds_dir)
        log.info(
            "dataset %s: %d jobs on %d episodes",
            ds_name or ds_dir,
            len(jobs),
            len(rdir.episodes()),
        )
        for overrides in jobs:
            key = job_key(ds_name, ds_dir, overrides)
            job_dir = sweep_dir / key
            if (job_dir / "metrics.json").exists():
                log.info("[%s] already complete, skipping", key)
                continue
            cfg = compose_pipeline_config(
                [f"rollout_dir={ds_dir}"] + overrides + list(extra_overrides)
            )
            if objective is not None and cfg.get("objective") is None:
                from omegaconf import OmegaConf

                cfg.objective = OmegaConf.create(dict(objective))
            log.info("[%s] %s", key, " ".join(overrides))
            try:
                run_pipeline(cfg, rdir, job_dir, shared_root=sweep_dir)
            except Exception as e:  # one bad combo must not kill the sweep
                log.exception("[%s] failed", key)
                new_failures[key] = f"{type(e).__name__}: {e}"

    # Merge failures from all runs; remove entries that have since recovered
    # (metrics.json now exists). Delete failures.json when everything is clean.
    failures_path = sweep_dir / "failures.json"
    prior_failures: dict[str, str] = {}
    if failures_path.exists():
        prior_failures = json.loads(failures_path.read_text())
    all_failures = {
        k: v
        for k, v in {**prior_failures, **new_failures}.items()
        if not (sweep_dir / k / "metrics.json").exists()
    }
    if all_failures:
        failures_path.write_text(json.dumps(all_failures, indent=2))
        log.warning(
            "%d/%d current jobs failed (see failures.json)",
            len(new_failures),
            len(jobs) * len(datasets),
        )
    elif failures_path.exists():
        failures_path.unlink()  # all previously failed jobs have recovered

    # Re-generate reports from ALL completed jobs in the dir so that adding
    # a new grid (e.g. SAE after PCA+UMAP) produces a unified index.
    if write_reports:
        regenerate_reports(sweep_dir, spec)
    return sweep_dir


# ─────────────────────────────────────────────────────────────────── reports


def regenerate_reports(sweep_dir: Path, spec: dict[str, Any]) -> None:
    """Rebuild all reports from every completed job under *sweep_dir*."""
    sweep_dir = Path(sweep_dir)
    write_sweep_reports(
        sweep_dir,
        discover_all_job_dirs(sweep_dir),
        spec,
        _load_persisted_datasets(sweep_dir),
    )


def _build_selection(spec: dict[str, Any]):
    """Instantiate the spec's selection rule (a _target_ CompositeSelection),
    falling back to a default rule when the spec omits one."""
    from behavior_graphs.model_selection.scoring import CompositeSelection

    sel_cfg = spec.get("selection")
    if sel_cfg and "_target_" in sel_cfg:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        return instantiate(OmegaConf.create(dict(sel_cfg)))
    if sel_cfg:  # plain {weights, constraints} without _target_
        return CompositeSelection(
            weights=dict(sel_cfg.get("weights") or {}),
            constraints={k: dict(v) for k, v in (sel_cfg.get("constraints") or {}).items()},
        )
    return CompositeSelection()  # registry defaults (silhouette floor etc.)


def write_sweep_reports(
    sweep_dir: Path,
    job_dirs: Sequence[Path],
    spec: dict[str, Any],
    datasets: dict[str, Path] | None = None,
) -> None:
    """Index + model-selection + sensitivity reports for a finished sweep."""
    import pandas as pd

    from behavior_graphs.model_selection.index import load_runs_index, varying_columns
    from behavior_graphs.model_selection.scoring import best_per_k
    from behavior_graphs.model_selection.sensitivity import sensitivity_table

    df = load_runs_index([d for d in job_dirs if (d / "metrics.json").exists()])
    if df.empty:
        log.warning("no completed jobs; skipping reports")
        return

    # Dataset column: map each run's rollout dir back to its spec name.
    by_path = {str(p.resolve()): (name or p.name) for name, p in (datasets or {}).items()}
    if "rollout_dir" in df.columns and by_path:
        df["dataset"] = df["rollout_dir"].map(
            lambda r: (
                by_path.get(str(Path(r).resolve()), Path(str(r)).name)
                if isinstance(r, str)
                else None
            )
        )
    multi_dataset = "dataset" in df.columns and df["dataset"].nunique() > 1

    selection = _build_selection(spec)

    # Scores are population-normalized — normalize within each dataset so a
    # hard task's runs aren't all dominated by an easy task's.
    def _score(group: pd.DataFrame) -> pd.DataFrame:
        scored = selection.score(group)
        return group.assign(score=scored["score"], feasible=scored["feasible"])

    if multi_dataset:
        df = pd.concat(
            [_score(g) for _, g in df.groupby("dataset", sort=False, dropna=False)]
        ).sort_index()
    else:
        df = _score(df)
    df.to_csv(sweep_dir / "sweep_index.csv", index=False)

    if multi_dataset:
        best = pd.concat(
            [best_per_k(g) for _, g in df.groupby("dataset", sort=False, dropna=False)],
            ignore_index=True,
        )
    else:
        best = best_per_k(df)
    if not best.empty:
        best.to_csv(sweep_dir / "best_per_k.csv", index=False)

    params = varying_columns(df)
    if multi_dataset:
        params = ["dataset"] + params
    for metric in SENSITIVITY_METRICS:
        table = sensitivity_table(df, params, metric)
        if not table.empty:
            table.to_csv(sweep_dir / f"sensitivity_{metric}.csv", index=False)

    n_ok = int(df["score"].notna().sum())
    log.info(
        "sweep_index.csv: %d runs (%d scored, weights=%s, constraints=%s)",
        len(df),
        n_ok,
        selection.weights,
        selection.constraints,
    )
