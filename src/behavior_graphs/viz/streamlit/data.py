"""Run-directory selection + artifact loading for the Streamlit apps.

This is the integration layer that replaced policy_doctor's
VisualizerConfig / task_configs / clustering_results loaders. Everything is
read through :class:`behavior_graphs.pipeline.artifacts.RunArtifacts`; videos
come from the run's ``metrics.json["rollout_dir"]`` via
:class:`behavior_graphs.envs.trajectory_dir.RolloutDirectory`.

Environment variables:
    BEHAVIOR_GRAPHS_RUN_DIR        Pin one run directory (skips the selector).
    BEHAVIOR_GRAPHS_RUNS_ROOT      Colon-separated roots to scan for run dirs
                                 (default: ``./runs:./multirun``).
    BEHAVIOR_GRAPHS_DISPLAY_DATA   A ``configs/data/`` group name; the selector
                                 then lists only that group's datasets rather
                                 than every dataset in the sweep dir. Unset =
                                 no filtering. See :func:`display_dataset_names`.
    DATA_PATH                    Repo-wide data root (same variable configs
                                 use via ``${oc.env:DATA_PATH}``). A run's
                                 recorded rollout_dir is relative to it.
                                 Read at the app entrypoints and passed to
                                 :func:`load_run`. Docker sets it to
                                 ``/data``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.io import (  # noqa: F401  (resolve_data_path re-exported)
    DATA_GROUP_NON_DATASET_KEYS,
    resolve_data_path,
)
from behavior_graphs.core.subclusters import SubClusterSet
from behavior_graphs.core.types import WindowSet
from behavior_graphs.core.yaml_io import load_yaml
from behavior_graphs.envs.trajectory_dir import RolloutDirectory
from behavior_graphs.model_selection.index import (
    index_schema_version,
    load_runs_index,
    representation_label,
)
from behavior_graphs.pipeline.artifacts import RunArtifacts, resolve_rollout_dir
from behavior_graphs.pipeline.run_discovery import (  # noqa: F401  (re-exported for existing importers)
    DEFAULT_ROOTS,
    ENV_RUN_DIR,
    ENV_RUNS_ROOT,
    discover_run_dirs,
    is_run_dir,
    runs_roots,
)
from behavior_graphs.viz.compat import meta_from_windows

_log = logging.getLogger(__name__)

ENV_DISPLAY_DATA = "BEHAVIOR_GRAPHS_DISPLAY_DATA"


#: Params with their own top-level sidebar selector rather than a slot in the
#: "Advanced hyperparameters" expander, because they are choices a viewer makes
#: deliberately: what the graph is *of*, and how complex it is.
_PROMOTED_PARAMS = ("representation", "k")

#: Keys in a best-per-K entry that are not hyperparameters: the score the run
#: won on, and the run itself. A caller that writes every key of an entry into
#: ``f"{key}_{param}"`` selector state has to skip these, or it puts a path and
#: a float into the state widgets read.
BEST_PER_K_NON_PARAM_KEYS = frozenset({"score", "path"})


def _task_name(dirname: str) -> str:
    """Task label: the part of a run dir name before the first '__'."""
    return dirname.split("__")[0]


def run_task_name(d: Path) -> str:
    """Human-readable task name for a run dir (public wrapper for callers outside data.py)."""
    return _effective_task(d, set(runs_roots()))


def _effective_task(d: Path, roots: set[Path]) -> str:
    """Return a human-readable task name for run dir *d*.

    For sweep layouts (sweep_name/task_name/<uuid>) the run dir's parent
    is the task folder, so use that name.  For flat layouts where the run
    dir sits directly inside a root (root/<run_name>__params) fall back to
    splitting on '__'.
    """
    parent = d.parent
    if parent in roots:
        return _task_name(d.name)
    return parent.name


#: A camera's entry under ``cameras:``. ``rotate`` is degrees clockwise
#: (multiples of 90) and ``mirror`` flips left-right, applied before the
#: rotation; both are render-time only — no clip or proxy is rewritten.
#: ``default: true`` marks the view shown first (see :func:`primary_camera`) and
#: is not geometry, so it is kept apart from the transform the media layer sees.
_GEOMETRY_KEYS = ("rotate", "mirror")
_CAMERA_KEYS = (*_GEOMETRY_KEYS, "default")
_ROTATIONS = (0, 90, 180, 270)


@st.cache_data(ttl=60, show_spinner=False)
def _data_group_cached(group: str, root_str: str) -> dict[str, dict] | None:
    """One read of a group's ``defaults:`` tree per TTL, or None if unreadable.

    Cached because the labels feed ``format_func``, which Streamlit runs once
    per option: uncached that re-parsed the whole tree (9 files for the deployed
    ``visualize``) per option, ~120 ms a rerun. It keeps its TTL — 9 small YAML
    files, so a ``configs/`` edit lands without a restart and the cost of
    getting it wrong is bounded — where the run-discovery caches gave theirs up
    and now re-walk only when the user asks.

    A group that can't be read at all (missing file, YAML error mid-edit —
    ``configs/`` is bind-mounted, so a half-saved file is reachable) warns and
    caches None, rather than re-raising once per option for the whole TTL. The
    callers decide what that means: no labels is survivable, no allowlist isn't.
    """
    try:
        return _data_group_entries(group, Path(root_str))
    except (OSError, yaml.YAMLError) as e:
        _log.warning("configs/data/%s.yaml is unreadable (%s)", group, e)
        return None


def _display_group_entries() -> dict[str, dict]:
    """The selected group's entries, or ``{}`` when there is nothing to read.

    Shared by the label and camera-transform lookups, both of which run inside
    rendering and must not raise: an unreadable or absent group degrades to no
    labels and no transforms (logged in :func:`_data_group_cached`).
    """
    root = configs_dir()
    if not (root / "data").is_dir():  # a bare install with no configs/ to read
        return {}
    group = os.environ.get(ENV_DISPLAY_DATA, "").strip() or "all"
    return _data_group_cached(group, str(root)) or {}


def camera_transforms(dataset: str) -> dict[str, dict]:
    """``{camera: {rotate, mirror}}`` for *dataset*, from ``configs/data/``.

    Display-only geometry, for cameras whose mounting does not match how the
    frames are stored — DROID's wrist camera hangs upside down, so its clips
    read 180° rotated. Baked into the web proxy and poster
    (:func:`behavior_graphs.viz.media.ensure_proxy`), which keys the cache entry:
    the recorded clips are never rewritten, so changing this costs a re-transcode
    of the derived cache and nothing more.

    Never raises, for the same reason :func:`task_display_names` doesn't: this
    is called while rendering a panel, and no orientation is much better than no
    panel.
    """
    return _display_group_entries().get(dataset, {}).get("cameras", {})


def primary_camera(dataset: str) -> str | None:
    """The view *dataset* shows first — its ``cameras:`` entry with ``default: true``.

    The default view in every sense that matters: the single clip a panel or
    thumbnail strip shows when it shows one, the leftmost of the side-by-side
    players, and the first option in the sidebar picker. For DROID that is the
    left exterior camera — the one that frames the whole task, where the wrist
    view is close-up and only legible next to it.

    Declared rather than inferred because the fallback is alphabetical
    (``media/<camera>/`` scanned in sorted order), which gets the right answer for
    ``exterior_image_1_left`` before ``wrist_image_left`` purely by luck: a camera
    named ``agentview`` or ``back_left`` would take over the default with nothing
    said. None when nothing is flagged, which keeps that alphabetical order.

    Never raises, like :func:`camera_transforms`: a missing default view is a
    cosmetic ordering, not worth a page.
    """
    return _display_group_entries().get(dataset, {}).get("primary_camera")


def task_display_names() -> dict[str, str]:
    """``{task name: label}`` for the datasets that declare a ``display_name``.

    Read from the same ``configs/data/`` group as the allowlist, so a label
    lives next to the dataset it renames. Unset ``BEHAVIOR_GRAPHS_DISPLAY_DATA``
    (a local checkout, no filtering) falls back to ``all``, so labels read the
    same locally as in the deployment. Why a field rather than a better config
    key: docs/droid_workflow.md, "Relabelling a dataset in the app".

    Never raises: this is called from ``format_func``, so an exception here
    would replace a whole page with a traceback over a cosmetic name. An
    unreadable group logs (in :func:`_data_group_cached`) and shows raw task
    names — degraded, but not silent and not broken.
    """
    return {n: e["display_name"] for n, e in _display_group_entries().items() if e["display_name"]}


def task_display_name(task: str) -> str:
    """The label for *task*; datasets without a ``display_name`` pass through.

    Display only — the task name is also a key (see :func:`select_run_dir`).
    """
    return task_display_names().get(task, task)


def _resolve_task(value: str, tasks: Sequence[str]) -> str:
    """*value* as a task key: itself, the key it labels, else the first task.

    A shared URL is as likely to carry the label a user copied off the page as
    the key, so accept both instead of leaving the selector on a value that
    matches no run.
    """
    if value in tasks:
        return value
    by_label = {label: task for task, label in task_display_names().items() if task in tasks}
    return by_label.get(value, tasks[0])


def dataset_display_name(path: str) -> str:
    """:func:`task_display_name` applied to the last segment of a dataset path.

    The dataset selectors label options with a run's recorded ``rollout_dir``,
    so only the trailing dataset name is relabelled.
    """
    head, sep, name = path.rpartition("/")
    return f"{head}{sep}{task_display_name(name)}"


def param_sort_key(v: object) -> tuple:
    if isinstance(v, int | float):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def _data_group_labels(
    group: str, root: Path, _seen: set[str] | None = None
) -> dict[str, str | None]:
    """``{dataset name: display_name or None}`` for ``configs/data/<group>.yaml``."""
    return {n: e["display_name"] for n, e in _data_group_entries(group, root, _seen).items()}


def _data_group_entries(group: str, root: Path, _seen: set[str] | None = None) -> dict[str, dict]:
    """``{dataset name: {display_name, cameras}}`` for ``configs/data/<group>.yaml``.

    Camera transforms are resolved here, most specific first: a dataset's own
    ``cameras:`` entry, then the ``cameras:`` block of the nearest group that
    includes it, out to the group that was asked for. A block therefore reaches
    every dataset its group pulls in through ``defaults:`` — so the selected
    group can correct what it inherits — but not the siblings of that group:
    ``droid.yaml`` saying the wrist camera is upside down does not follow
    ``all.yaml`` onto the LIBERO datasets.

    Its ``defaults:`` parents included. A dataset's value is either the rollout
    dir on its own or a mapping carrying that dir plus display metadata
    (``display_name``, ``cameras``); the path is the sweep's business, so only
    the name and that metadata come back here.

    A group may declare a top-level ``cameras:`` block, which is where a camera
    transform usually belongs: "the wrist camera is mounted upside down" is a
    fact about the rig, not about one rollout collection, and repeating it per
    dataset is how the two drift apart.

    A malformed entry keeps its name (dropping it would hide the dataset from
    the app, which is worse than showing it unlabelled) and is logged. Same for
    a mapping key we don't know: a misspelled ``display-name:`` would otherwise
    look exactly like the label cache being stale.

    Plain YAML rather than Hydra: this reruns on a live server, and
    hydra.initialize's process-global state makes that awkward.
    """
    return {
        name: {
            "display_name": e["display_name"],
            # Most specific wins, at the level of the decision rather than per
            # camera: a dataset that overrides one camera's *geometry* is not
            # thereby saying anything about which view comes first.
            "primary_camera": _flagged_default(e["own_cameras"], path_hint=name)
            or _flagged_default(e["group_cameras"], path_hint=name),
            "cameras": {
                cam: geometry
                for cam, spec in (e["group_cameras"] | e["own_cameras"]).items()
                # `default: true` is not a transform, and an override can turn one
                # back off (`rotate: 0`), which _camera_specs keeps as an empty
                # spec; both drop out here rather than shipping the media layer a
                # no-op transform (and so a second cache key for one orientation).
                if (geometry := {k: v for k, v in spec.items() if k in _GEOMETRY_KEYS})
            },
        }
        for name, e in _read_data_group(group, root, _seen if _seen is not None else set()).items()
    }


def _flagged_default(specs: dict[str, dict], *, path_hint: str) -> str | None:
    """The camera marked ``default: true``, or None; first declared if several."""
    flagged = [camera for camera, spec in specs.items() if spec.get("default")]
    if len(flagged) > 1:
        _log.warning(
            "dataset %r marks %d cameras as `default: true` (%s) — using %r; "
            "only one view can be shown first",
            path_hint,
            len(flagged),
            ", ".join(flagged),
            flagged[0],
        )
    return flagged[0] if flagged else None


def _read_data_group(group: str, root: Path, seen: set[str]) -> dict[str, dict]:
    """Entries with ``own_cameras``/``group_cameras`` kept apart until composed.

    Separate because the precedence differs per level: a dataset's own block
    always wins, while group blocks are applied on the way back up, each level
    over the ones it inherits.
    """
    if group in seen:  # a defaults cycle would otherwise recurse forever
        return {}
    seen.add(group)
    path = root / "data" / f"{group}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no such data config group: {path}")  # noqa: TRY003
    cfg = load_yaml(path.read_text()) or {}
    entries: dict[str, dict] = {}
    # A group's own entries win over an inherited namesake, so recurse first.
    for parent in cfg.get("defaults") or []:
        if isinstance(parent, str) and parent != "_self_":
            entries |= _read_data_group(parent, root, seen)
    for name, val in cfg.items():
        if name in DATA_GROUP_NON_DATASET_KEYS:
            continue
        label = None
        own_cameras: dict[str, dict] = {}
        if isinstance(val, dict) and isinstance(val.get("path"), str):
            label = val.get("display_name")
            own_cameras = _camera_specs(val.get("cameras"), path, f"{name}.cameras:")
            if unknown := set(val) - {"path", "display_name", "cameras"}:
                _log.warning(
                    "%s: dataset %r has unknown key(s) %s — ignored (typo for `display_name:`?)",
                    path,
                    name,
                    ", ".join(sorted(map(str, unknown))),
                )
        elif not isinstance(val, str):
            _log.warning(
                "%s: dataset %r should be a rollout dir, or a mapping with a `path:` "
                "(and optionally `display_name:`/`cameras:`); got %r — listed unlabelled",
                path,
                name,
                val,
            )
        entries[name] = {
            # str() so a YAML scalar (`display_name: 200`, or a bare date) is a
            # usable label instead of a protobuf error out of format_func.
            "display_name": str(label) if label else None,
            "own_cameras": own_cameras,
            # Inherited group blocks survive a re-declaration. This entry
            # replaces its inherited namesake — that is how a group overrides a
            # label — but `group_cameras` was never part of the entry: it is the
            # geometry of the rig, applied by whichever *group* pulled the
            # dataset in, and that group still does. `visualize.yaml` exists to
            # re-list datasets `droid.yaml` also declares, so without this,
            # uncommenting one there silently stopped rotating its wrist camera.
            "group_cameras": (entries.get(name) or {}).get("group_cameras") or {},
        }
    # This level's block over everything the level includes, its inherited
    # datasets included, but only theirs: a sibling group's datasets are not in
    # `entries` here, which is what keeps droid's wrist rotation off LIBERO.
    if this_level := _camera_specs(cfg.get("cameras"), path, "cameras:"):
        for e in entries.values():
            e["group_cameras"] = e["group_cameras"] | this_level
    return entries


def _camera_specs(raw: object, path: Path, where: str) -> dict[str, dict]:
    """Validate a ``cameras:`` block into ``{camera: {rotate, mirror, default}}``.

    Every rejection is logged and dropped rather than raising: this parses on a
    live server, and a typo in one camera's spec should cost that camera its
    transform, not the page.

    A camera listed only to carry ``default: true`` is normal — that is how the
    default view is declared alongside the rig's geometry rather than in a second
    key that could name a camera this block never mentions.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _log.warning(
            "%s: %s should be a mapping of camera name to transform; got %r", path, where, raw
        )
        return {}
    specs: dict[str, dict] = {}
    for camera, spec in raw.items():
        if not isinstance(spec, dict):
            _log.warning(
                "%s: %s %r should be a mapping, e.g. {rotate: 180}; got %r",
                path,
                where,
                camera,
                spec,
            )
            continue
        if unknown := set(spec) - set(_CAMERA_KEYS):
            _log.warning(
                "%s: %s %r has unknown key(s) %s — ignored",
                path,
                where,
                camera,
                ", ".join(sorted(map(str, unknown))),
            )
        out: dict[str, object] = {}
        usable = True
        rotate = spec.get("rotate", 0)
        if (
            isinstance(rotate, bool)
            or not isinstance(rotate, int)
            or rotate % 360 not in _ROTATIONS
        ):
            _log.warning(
                "%s: %s %r rotate must be one of %s; got %r — not rotated",
                path,
                where,
                camera,
                list(_ROTATIONS),
                rotate,
            )
            usable = False
        elif rotate % 360:
            out["rotate"] = rotate % 360
        mirror = spec.get("mirror", False)
        if not isinstance(mirror, bool):
            _log.warning(
                "%s: %s %r mirror must be true or false; got %r — not mirrored",
                path,
                where,
                camera,
                mirror,
            )
            usable = False
        elif mirror:
            out["mirror"] = True
        is_default = spec.get("default", False)
        if not isinstance(is_default, bool):
            _log.warning(
                "%s: %s %r default must be true or false; got %r — not the default view",
                path,
                where,
                camera,
                is_default,
            )
            usable = False
        elif is_default:
            out["default"] = True
        if out or usable:
            # An empty spec is recorded only when it was *asked* for: `rotate: 0`
            # on a dataset is how you cancel a group default for one camera, and
            # a dropped key could not express that. A spec whose every key was
            # rejected is dropped instead, so it falls back to the group default
            # rather than silently cancelling it — a typo'd `rotate: 45` used to
            # leave the camera upside down while the log only said "not rotated".
            specs[str(camera)] = out
    return specs


def _data_group_names(group: str, root: Path, _seen: set[str] | None = None) -> set[str]:
    """Dataset names declared by ``configs/data/<group>.yaml``, defaults included."""
    return set(_data_group_entries(group, root, _seen))


def _group_families(group: str, root: Path, seen: set[str]) -> dict[str, list[str]]:
    """``{group: [dataset name]}`` for *group*, descending through aggregators.

    A group that declares no datasets of its own is a way of naming a set of
    other groups, not a family — ``all.yaml`` is ten benchmarks and ``visualize``
    is ``all`` with some entries commented out, and either one offered as a
    single choice is the whole corpus in one dropdown. So a group with
    ``defaults:`` contributes its children's families instead of itself, and
    only what it declares directly (nothing, for those two) stays under its own
    name.

    Order follows ``defaults:``, which is the order the benchmarks are listed in
    rather than the alphabetical one a set would give back.
    """
    if group in seen:  # a defaults cycle would otherwise recurse forever
        return {}
    seen.add(group)
    path = root / "data" / f"{group}.yaml"
    if not path.is_file():  # a `defaults:` naming a group this checkout lacks
        _log.warning("configs/data/%s.yaml does not exist — no family for it", group)
        return {}
    cfg = load_yaml(path.read_text()) or {}
    children = [p for p in (cfg.get("defaults") or []) if isinstance(p, str) and p != "_self_"]
    families: dict[str, list[str]] = {}
    for child in children:
        families |= _group_families(child, root, seen)
    own = [n for n in cfg if n not in DATA_GROUP_NON_DATASET_KEYS]
    if own or not children:
        # `_data_group_names` rather than `own` for a leaf: identical today (no
        # family file has a `defaults:`), and right if one ever gains one.
        families[group] = sorted(_data_group_names(group, root) if not children else own)
    return families


@st.cache_data(ttl=60, show_spinner=False)
def _dataset_families_cached(group: str, root_str: str) -> dict[str, list[str]]:
    return _group_families(group, Path(root_str), set())


def dataset_families() -> dict[str, list[str]]:
    """``{family group: [dataset name]}`` for the app's display group.

    The families a reviewer picks between — the benchmark groups under
    ``configs/data/``, keyed by their file name. Same source as the allowlist
    and the display names, so a dataset added to a group shows up in all three
    without a second edit.

    Never raises, for the same reason :func:`task_display_names` doesn't: this
    feeds a selector, and an unreadable group should cost the grouping rather
    than the page. Empty means "no families" — the caller decides whether that
    is fatal (it is not: the dataset list is what matters, and it comes from the
    run index).
    """
    root = configs_dir()
    if not (root / "data").is_dir():  # a bare install with no configs/ to read
        return {}
    group = os.environ.get(ENV_DISPLAY_DATA, "").strip() or "all"
    try:
        return _dataset_families_cached(group, str(root))
    except (OSError, yaml.YAMLError) as e:
        _log.warning("cannot read the family tree under configs/data/%s.yaml (%s)", group, e)
        return {}


def configs_dir() -> Path:
    """The repo's ``configs/``: alongside the installed package, else the CWD.

    Not CWD-only: Streamlit is launched from outside the checkout root.
    """
    packaged = Path(__file__).resolve().parents[4] / "configs"
    return packaged if packaged.is_dir() else Path.cwd() / "configs"


def display_dataset_names() -> set[str] | None:
    """Dataset allowlist for the app, or None when unfiltered.

    ``BEHAVIOR_GRAPHS_DISPLAY_DATA`` names a ``configs/data/`` group to list only
    the datasets of; unset (a local checkout) shows everything under the runs
    roots. An unresolvable group raises rather than falling back to unfiltered,
    which would look identical to a working filter — the opposite call from
    :func:`task_display_names`, which shares the same cached read: a missing
    label is cosmetic, a missing filter shows runs that shouldn't be public.
    The re-raise is on the uncached loader so the error names the actual cause.
    """
    group = os.environ.get(ENV_DISPLAY_DATA, "").strip()
    if not group:
        return None
    entries = _data_group_cached(group, str(configs_dir()))
    if entries is None:  # unreadable: re-raise the real error instead of guessing
        entries = _data_group_entries(group, configs_dir())
    names = set(entries)
    if not names:
        raise ValueError(
            f"{ENV_DISPLAY_DATA}={group!r} resolves to no datasets "
            f"({configs_dir() / 'data' / f'{group}.yaml'} is unreadable, names none, or they "
            "are all commented out), which would hide every run. Unset the variable to show all."
        )
    return names


@st.cache_resource(show_spinner="Scanning run roots…", max_entries=2)
def _discover_run_dirs_cached(roots_str: str, max_depth: int) -> list[str]:
    """Cached filesystem walk (paths as strings so the cache key is hashable).

    No TTL: "Refresh runs" (:func:`refresh_run_index`) is what re-walks now.

    The walk has to read every run dir to see whether it holds a ``labels.npy``,
    so it costs a directory read per run — ~0.2 s when this cache was written
    against a 5 000-run root, 6-8 s at 43 000 runs with a sweep writing
    underneath it. On a 60 s TTL that landed on whichever widget interaction
    happened to fall after the expiry, so one hyperparameter change a minute
    stalled for seconds and the rest felt instant. A sweep in progress
    guarantees the expiry fires every single minute, which is exactly when
    someone is watching runs land.
    """
    roots = [Path(p) for p in roots_str.split(":") if p]
    return [str(d) for d in discover_run_dirs(roots, max_depth)]


def _discovered_run_dirs(roots: list[Path], max_depth: int = 4) -> list[Path]:
    token = ":".join(str(r) for r in roots)
    return [Path(s) for s in _discover_run_dirs_cached(token, max_depth)]


@st.cache_resource(show_spinner=False, max_entries=2)
def _display_run_dirs_cached(roots_str: str, display_key: str) -> list[Path]:  # noqa: ARG001
    """The allowlist filter, memoized. *display_key* is the allowlist's identity.

    ``cache_resource``, like :func:`_run_index` and for the same reason: this is
    a read-only list of 40 000 entries, and ``cache_data`` would unpickle a
    fresh copy on every hit. It hands back ``Path`` objects rather than strings
    so callers share them too — rebuilding 36 000 ``Path`` objects per call was
    itself 230 ms. Every caller only reads the list (checked); nothing may sort
    or append to it in place.

    ``_effective_task`` is a ``Path`` construction and a set lookup per run dir:
    450 ms across 43 000 of them, and several callers ask for the list on every
    rerun, so the filter alone cost more than a second of every widget
    interaction. Same lifetime as the walk it filters — cleared by
    :func:`refresh_run_index` and by nothing else.
    """
    roots = [Path(p) for p in roots_str.split(":") if p]
    dirs = _discovered_run_dirs(roots)
    allowed = display_dataset_names()
    if allowed is None:
        return dirs
    roots_set = set(roots)
    return [d for d in dirs if _effective_task(d, roots_set) in allowed]


def display_run_dirs(roots: list[Path] | None = None) -> list[Path]:
    """Run dirs the app should show: discovered, then cut to the allowlist.

    The single choke point — selectors, the model-selection table and the
    best-per-K defaults all go through here, so an excluded dataset can't
    reappear in one of them or land in session state.
    """
    roots = roots if roots is not None else runs_roots()
    return _display_run_dirs_cached(":".join(str(r) for r in roots), display_filter_key())


def paths_token(paths: Sequence[str]) -> str:
    """Short content hash of a path list, for use as a cache key.

    Passing the list itself to a cached function makes Streamlit hash all of it
    on *every* lookup, hit or miss — ~11 ms for a 5 000-run sweep root, on every
    rerun. Hashing it once here into 16 chars costs ~0.4 ms; the list then rides
    along as an underscore-prefixed argument, which Streamlit skips.
    """
    h = hashlib.blake2b(digest_size=8)
    for s in paths:
        h.update(str(s).encode())
        h.update(b"\0")
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _metrics_fingerprint_cached(token: str, _run_dir_strs: tuple[str, ...]) -> tuple[int, float]:
    """(count, max mtime) over every metrics.json — a cheap "did anything change".

    One ``stat`` per run dir: ~1.3 s at 43 000 runs. No TTL, for the same reason
    as the walk above — and a TTL here is worse than useless, because expiring
    it mid-sweep mints a new fingerprint, which is a fresh cache key for the
    runs index and every memo built on it. The index is the expensive one
    (config.yaml + metrics.json per run), so the periodic re-stat did not just
    cost its own 1.3 s, it threw away the frame as well.
    """
    paths = [Path(s) / "metrics.json" for s in _run_dir_strs]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


# Memory only, deliberately not persist="disk". Streamlit's disk layer is
# append-only in practice — TTL/LRU eviction drops the in-memory entry but
# leaves the pickled frame in ~/.streamlit/cache, and only an explicit
# delete()/clear() removes it. The key here includes a 60 s-TTL mtime
# fingerprint, so a sweep writing while a tab is open mints a fresh key every
# minute and each one strands another multi-MB DataFrame on the boot drive,
# which is the scarce disk on this box. What that bought was a few seconds off
# the first page view after a restart; not worth an unbounded pile nothing
# prunes.
@st.cache_data(show_spinner="Indexing runs…", max_entries=8)
def _runs_index_cached(
    token: str,  # noqa: ARG001  (cache key: which run dirs)
    fingerprint: tuple[int, float],  # noqa: ARG001  (cache key: their metrics.json state)
    schema: str,  # noqa: ARG001  (cache key: the code that builds the frame)
    _run_dir_strs: tuple[str, ...],
) -> pd.DataFrame:
    return load_runs_index([Path(s) for s in _run_dir_strs])


def runs_index(run_dirs: Sequence[Path] | None = None) -> pd.DataFrame:
    """The runs index for *run_dirs*, cached across pages and process restarts.

    A single entry point, so the demo's two index consumers — the
    model-selection panel and Sweep Analysis — share one cached frame instead of
    each building and disk-persisting their own copy of the same thing.

    It owns its entire cache key, which is the point: a caller cannot forget the
    mtime fingerprint (making the index stale) or the schema version (making the
    frame outlive the code that built it, so the page dies on a KeyError for a
    column that frame never had — the live-mount deployment hot-reloads edited
    source without dropping caches, so that is reachable without a restart).
    """
    dirs = list(run_dirs) if run_dirs is not None else display_run_dirs()
    strs = tuple(str(d) for d in dirs)
    token = paths_token(strs)
    return _runs_index_cached(
        token, _metrics_fingerprint_cached(token, strs), index_schema_version(), strs
    )


def metrics_fingerprint(run_dirs: Sequence[Path]) -> tuple[int, float]:
    """Freshness signal for anything derived from these runs' metrics.json.

    Public because every memo built on top of the index needs the same signal:
    the run-dir list is unchanged when a run is re-executed in place, so a cache
    keyed on paths alone silently serves results for metrics that no longer
    exist.
    """
    strs = tuple(str(d) for d in run_dirs)
    return _metrics_fingerprint_cached(paths_token(strs), strs)


def display_filter_key() -> str:
    """Cache key for the active allowlist; part of every cached index's key.

    The allowlist is read inside the cache-miss body, so without this a cached
    index would survive a change of group or an edit to the group file.
    """
    allowed = display_dataset_names()
    return "" if allowed is None else ",".join(sorted(allowed))


@st.cache_resource(show_spinner="Loading runs…", max_entries=2)
def _run_index(roots_str: str, display_key: str = "", schema: str = "") -> list[dict]:
    """Flat list of {path, task, k, w, s, representation, ...} for all run dirs.

    *display_key* and *schema* are unused in the body — they are here so the
    allowlist (:func:`display_filter_key`) and the row schema
    (:func:`index_schema_version`) are part of the cache key. Without the
    latter, the deployed stack — which bind-mounts ``src/`` and hot-reloads —
    keeps serving rows built before a column was added, and the sidebar then
    either loses a selector or raises a KeyError for it.

    This is a read-only process-local resource rather than cache data: the demo
    reruns this selector when a task changes, and serializing/deserializing the
    ~10k-row list on every hit needlessly makes that interaction feel like a
    second run discovery. It stays cached until the user explicitly refreshes
    the runs, so hyperparameter changes do not periodically rebuild the index.
    """
    roots = [Path(p) for p in roots_str.split(":") if p]
    roots_set: set[Path] = set(roots)
    dirs = display_run_dirs(roots)
    rows: list[dict] = []
    for d in dirs:
        cfg: dict = {}
        try:
            cfg = load_yaml((d / "config.yaml").read_text()) or {}
        except Exception as e:
            _log.warning(
                "failed to read %s (%s); run listed with unknown hyperparameters",
                d / "config.yaml",
                e,
            )
        clu = cfg.get("clustering") or {}
        win = cfg.get("windowing") or {}
        red = cfg.get("reduction") or {}
        feat = cfg.get("features") or {}
        gph = cfg.get("graph") or {}
        smoothing = str(gph.get("smoothing") or "none")

        red_tgt = str(red.get("_target_", ""))
        reduction = red_tgt.rsplit(".", 1)[-1].replace("Reduction", "").lower() or None
        clu_tgt = str(clu.get("_target_", ""))
        clustering = clu_tgt.rsplit(".", 1)[-1].replace("Clustering", "").lower() or None
        layer = str(feat.get("layer") or "") or None

        rows.append(
            {
                "path": str(d),
                "task": _effective_task(d, roots_set),
                "k": clu.get("n_clusters") or clu.get("n_components"),
                "w": win.get("width"),
                "s": win.get("stride"),
                "reduction": reduction,
                # Two runs can share K and differ in how the labels were made.
                "clustering": clustering,
                "n_components": red.get("n_components"),
                # Two autoencoder runs can differ in nothing else, so without
                # these the picker offers the plain and the temporally-smoothed
                # fit as the same option and shows whichever it finds first.
                "temporal_coef": red.get("temporal_coef"),
                "contrastive_coef": red.get("contrastive_coef"),
                "beta": red.get("beta"),
                # The graph stage can relabel windows, so two runs sharing every
                # upstream knob are still different graphs. Runs predating the
                # swept smoothing axis have no `graph.smoothing` key at all;
                # those are the unsmoothed arm, so read the default rather than
                # None (None would drop them out of the selector's value set and
                # make the axis look like it never varies).
                "smoothing": smoothing,
                # Only meaningful for the sticky arm; recording it for the
                # unsmoothed runs too would add a selector that partitions runs
                # by a number none of them used.
                "sticky_lambda": gph.get("sticky_lambda") if smoothing == "sticky" else None,
                "layer": layer,
                # What the graph is a graph *of*: the policy layer for policy
                # embeddings, backbone+pooling for video features.
                "representation": representation_label(feat),
                "has_metrics": (d / "metrics.json").exists(),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def _varying_params(roots_str: str, display_key: str, schema: str = "") -> set[str]:
    """Params that take more than one value anywhere in the index.

    One set per param over every row — 560 ms across 36 000 runs, and it decided
    which selectors to draw, so it ran on every rerun before any of them were
    drawn. It only changes when the index does, so it is cached with it.
    """
    rows = _run_index(roots_str, display_key, index_schema_version())
    return {
        param
        for param, _ in PARAM_LABELS
        if len({r[param] for r in rows if r[param] is not None}) > 1
    }


def run_index_rows() -> list[dict]:
    """The picker's index — one row of hyperparameters per discoverable run.

    The public way to ask "which runs exist, and what were they built with":
    :func:`select_run_dir` filters these rows down to one, and Policy
    Comparison filters the same rows once per policy. Empty when no run root is
    configured. Cheap on a warm cache (it is the same memo the sidebar built),
    tens of seconds cold — a caller that is not already showing runs should
    render behind a lazy section.
    """
    roots = runs_roots()
    if not roots:
        return []
    return _run_index(":".join(str(r) for r in roots), display_filter_key(), index_schema_version())


def refresh_run_index() -> None:
    """Clear cached run discovery/index data after the user requests a refresh."""
    for cached in (
        _discover_run_dirs_cached,
        _display_run_dirs_cached,
        _metrics_fingerprint_cached,
        _runs_index_cached,
        _varying_params,
    ):
        cached.clear()
    _run_index.clear()
    # These caches live in the model-selection module, which imports this one;
    # import lazily here to avoid a module-level cycle.
    from behavior_graphs.viz.streamlit.components.selection_panel import (
        clear_best_per_k_cache,
    )

    clear_best_per_k_cache()


# Ordered param list for the cascading selectors. Public because the run
# picker is not the only thing that has to name a run by its hyperparameters:
# Policy Comparison draws the same axes to hold several policies at one
# pipeline configuration (see
# :mod:`behavior_graphs.viz.streamlit.comparison_graphs`), and a second copy of
# this list would drift the moment a sweep gained an axis.
PARAM_LABELS: list[tuple[str, str]] = [
    ("representation", "Representation"),
    ("k", "K (clusters)"),
    ("w", "Window width"),
    ("s", "Stride"),
    ("reduction", "Reduction"),
    ("clustering", "Clusterer"),
    ("n_components", "N components"),
    ("temporal_coef", "Temporal coef"),
    ("contrastive_coef", "Contrastive coef"),
    ("beta", "Beta (KL weight)"),
    ("smoothing", "Label smoothing"),
    ("sticky_lambda", "Sticky λ"),
    # No ("layer", "Layer") entry: `representation` is the same axis, one
    # level up -- it returns the layer verbatim for a policy run and names the
    # backbone for a run whose features never came from a policy (which has no
    # layer to show). Keeping both drew a second selector that, after a
    # representation was picked, had exactly one option. The `layer` row key
    # stays: pipeline_builder and the best-per-K applier still use it.
]

# Query params that used to select a run and no longer do. Cleared, never
# written: a bookmarked `?layer=cond_proj` would otherwise outlive the selector
# it fed (`representation` replaced it) and survive every reset.
_LEGACY_QUERY_PARAM_NAMES = ("layer",)

# Canonical URL query-param names for select_run_dir's run-selection widgets,
# shared verbatim across every page that calls select_run_dir (demo_run,
# chat_run, ...) so the same URL query string works regardless of which page
# it's opened on.
QUERY_PARAM_NAMES = (
    "task",
    "representation",
    "k",
    "w",
    "s",
    "reduction",
    "clustering",
    "n_components",
    "temporal_coef",
    "contrastive_coef",
    "beta",
    "smoothing",
    "sticky_lambda",
)

# Session-wide (unprefixed) fallback carrying the last-resolved run selection
# across pages. st.navigation's sidebar links clear st.query_params before the
# destination page's script runs (confirmed empirically: a query string
# present on Graph Demo is simply gone by the time Chat's script starts), so a
# fresh URL open relies on st.query_params but an in-app page switch has to
# fall back to this session_state copy instead -- session_state (unlike
# query_params) survives Streamlit's page-switch navigation.
_SHARED_RUN_PARAMS_KEY = "_shared_run_params"


def seed_selectors_from_query_params(key: str) -> None:
    """Seed ``f"{key}_*"`` run selectors from ``st.query_params``, once per session.

    Call this after :func:`init_best_defaults` and before :func:`select_run_dir`
    (same key) so an explicit URL param always wins over the "best per K"
    auto-pick -- ``init_best_defaults`` unconditionally overwrites these same
    keys the first time it fires, so seeding has to happen afterward rather
    than merely being skipped if already set.

    Falls back to :data:`_SHARED_RUN_PARAMS_KEY` for any name missing from
    ``st.query_params`` (see its docstring) so switching from e.g. Graph Demo
    to Chat via the sidebar keeps showing the same run even though the URL's
    query string was cleared by the page switch itself.
    """
    sentinel = f"{key}_qp_applied"
    if st.session_state.get(sentinel):
        return
    st.session_state[sentinel] = True
    shared: dict[str, str] = st.session_state.get(_SHARED_RUN_PARAMS_KEY) or {}
    for name in QUERY_PARAM_NAMES:
        raw = st.query_params.get(name, shared.get(name))
        if raw is None:
            continue
        if name in ("k", "w", "s", "n_components"):
            try:
                raw = int(raw)
            except ValueError:
                continue
        elif name in ("temporal_coef", "contrastive_coef", "beta", "sticky_lambda"):
            try:
                # Written as an int by the sweep (0, 1) but a float is legal.
                raw = int(raw) if float(raw).is_integer() else float(raw)
            except ValueError:
                continue
        st.session_state[f"{key}_{name}"] = raw
    manual = st.query_params.get("run_dir", shared.get("run_dir"))
    if manual is not None:
        st.session_state[f"{key}_manual"] = manual


def sync_query_params_from_selectors(key: str) -> None:
    """Mirror current ``f"{key}_*"`` selector values into ``st.query_params``.

    Keeps the address bar (and thus anything copied from it) in sync with the
    run actually being shown, so a user can share the current view just by
    copying the URL. Only touches the known param names above -- never
    clears/replaces the whole query string, so ``pwd`` (or anything else)
    already present is left untouched. Also mirrors the same values into
    :data:`_SHARED_RUN_PARAMS_KEY` for :func:`seed_selectors_from_query_params`
    on other pages to fall back to.
    """
    shared: dict[str, str] = {}
    for name in QUERY_PARAM_NAMES:
        val = st.session_state.get(f"{key}_{name}")
        val_str = "" if val is None else str(val)
        if not val_str:
            continue
        shared[name] = val_str
        if st.query_params.get(name) != val_str:
            st.query_params[name] = val_str
    manual = st.session_state.get(f"{key}_manual")
    if manual:
        shared["run_dir"] = manual
        if st.query_params.get("run_dir") != manual:
            st.query_params["run_dir"] = manual
    st.session_state[_SHARED_RUN_PARAMS_KEY] = shared


def clear_run_query_params() -> None:
    """Remove every run-selection query param (``QUERY_PARAM_NAMES`` + ``run_dir``).

    Without this, a "reset" action that only clears session_state (e.g. a
    sidebar "clear cache & reset session" button) silently no-ops: the URL's
    query params are untouched, so the next rerun's
    :func:`seed_selectors_from_query_params` immediately re-seeds the exact
    same task/K/etc. right back from the still-present ``st.query_params``.
    Leaves any other query param (e.g. ``pwd``, ``page``) untouched.
    """
    for name in (*QUERY_PARAM_NAMES, *_LEGACY_QUERY_PARAM_NAMES, "run_dir"):
        if name in st.query_params:
            del st.query_params[name]


def _apply_best_for_task_k(key: str, best_per_k: dict) -> None:
    """on_change callback for K changes: apply best params for the current (task, K) pair."""
    task = st.session_state.get(f"{key}_task")
    k_val = st.session_state.get(f"{key}_k")
    row = (best_per_k.get(task) or {}).get(k_val)
    if row is None:
        st.toast(
            f"No best-hyperparameter data for {task_display_name(task)} at K={k_val} — "
            "other selectors left unchanged.",
            icon="⚠️",
        )
        return
    # Keys produced by selection_panel._row_to_param_dict ("score" excluded).
    for param in ("w", "s", "reduction", "n_components", "layer", "representation"):
        val = row.get(param)
        if val is not None:
            st.session_state[f"{key}_{param}"] = val


def _apply_task_best(key: str, best_per_k: dict) -> None:
    """on_change callback for Task changes: reset K to the best K for the new task."""
    task = st.session_state.get(f"{key}_task")
    task_params = best_per_k.get(task) or {}
    if not task_params:
        st.toast(
            f"No best-hyperparameter data for {task_display_name(task)} — K and other "
            "selectors left at their previous values.",
            icon="⚠️",
        )
        return
    best_k = max(task_params, key=lambda k: task_params[k].get("score", -1))
    st.session_state[f"{key}_k"] = best_k
    for param, val in task_params[best_k].items():
        if param not in BEST_PER_K_NON_PARAM_KEYS:
            st.session_state[f"{key}_{param}"] = val


def _warn_if_no_metrics(p: Path) -> None:
    """Flag a run dir with no metrics.json instead of silently handing it out.

    ``is_run_dir`` only requires window metadata + labels.npy, so a run that
    crashed (or was interrupted) before its evaluation stage finished is
    still a valid, selectable run dir -- it just has no success rate, no
    scores, and no rollout_dir (so no videos). Surfacing this in the
    sidebar means a user (or the "best per K" selector, see
    ``select_run_dir``) doesn't silently end up on a broken run.
    """
    if not (p / "metrics.json").exists():
        st.sidebar.warning(
            f"`{p.name}` has no metrics.json — this run may have crashed or been "
            "interrupted before its evaluation stage finished. Success rate, "
            "scores, and rollout videos will be unavailable."
        )


def _show_run_id(p: Path, slot: object | None = None) -> None:
    """Sidebar: the id everything outside the app knows the selected run by.

    The selectors name a run by the hyperparameters they filter on; the sweep
    index, the run dirs on disk and the stage caches all name it by its job
    hash. Without that hash on screen there is no way to get from "the run I am
    looking at" to a row in ``sweep_index.csv``, a path to hand to a script, or
    a bug report someone else can reproduce.

    *slot* is a container reserved earlier in the sidebar (the run is only known
    after the selectors have run, but belongs on screen next to them).
    """
    dg = slot if slot is not None else st.sidebar
    dg.caption("Run ID", help=str(p))
    dg.code(p.name, language=None)


def select_run_dir(
    key: str = "run_dir_select",
    best_per_k: dict[str, dict[int, dict]] | None = None,
) -> Path | None:
    """Sidebar: task selector → K selector → (expander) other params → run path.

    When *best_per_k* is supplied (from ``compute_best_per_k``), changing K
    automatically applies the best-scoring hyperparameters for that K.
    """
    pinned = os.environ.get(ENV_RUN_DIR, "").strip()
    if pinned:
        p = Path(pinned)
        if is_run_dir(p):
            st.sidebar.caption(f"Run dir (from ${ENV_RUN_DIR}): `{p}`")
            _warn_if_no_metrics(p)
            return p
        st.sidebar.error(f"${ENV_RUN_DIR}={p} is not a run dir.")
        return None

    st.sidebar.header("Run")
    if st.sidebar.button(
        "Refresh runs",
        key=f"{key}_refresh_runs",
        help=(
            "Rescan the configured run roots for newly added or removed runs. "
            "The run list is a snapshot taken when the page first loaded — a "
            "sweep writing right now will not appear until you press this."
        ),
    ):
        refresh_run_index()
        st.rerun()
    roots = runs_roots()
    rows = (
        _run_index(":".join(str(r) for r in roots), display_filter_key(), index_schema_version())
        if roots
        else []
    )

    if not rows:
        st.sidebar.warning(
            "No run directories found under "
            + ", ".join(str(r) for r in (roots or [Path(r) for r in DEFAULT_ROOTS]))
            + f". Set ${ENV_RUNS_ROOT} or ${ENV_RUN_DIR}, or type a path below."
        )
        manual = st.sidebar.text_input("Run dir path", key=f"{key}_manual").strip()
        if manual:
            p = Path(manual)
            if is_run_dir(p):
                _show_run_id(p)
                _warn_if_no_metrics(p)
                return p
            st.sidebar.error(f"`{p}` is not a run dir.")
        return None

    from collections import defaultdict

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    # Decide which params vary anywhere across the sweep so the selector set
    # is the same for every task (stable across task switches).
    global_varying_params = _varying_params(
        ":".join(str(r) for r in roots), display_filter_key(), index_schema_version()
    )

    tasks = sorted(by_task)
    task_kw: dict = {}
    if best_per_k:
        task_kw = {"on_change": _apply_task_best, "args": (key, best_per_k)}
    # A seeded `?task=` is whatever was in the URL, and Streamlit's own check
    # compares *labels* once format_func is set, so it would accept a label as
    # a value and then select nothing.
    if (seeded := st.session_state.get(f"{key}_task")) is not None:
        st.session_state[f"{key}_task"] = _resolve_task(seeded, tasks)
    # format_func, not renamed options: the selected value is the task *key*
    # (best_per_k lookups, the `{key}_task` query param).
    task = st.sidebar.selectbox(
        "Task", tasks, key=f"{key}_task", format_func=task_display_name, **task_kw
    )
    candidates = list(by_task[task])

    # ── Representation selector (prominent) ──────────────────────────────────
    # The coarsest choice there is: two runs on different representations are
    # graphs of different things, not two settings of one thing. Above K,
    # which is a complexity choice *within* a representation.
    if "representation" in global_varying_params:
        reps = sorted(
            {r["representation"] for r in candidates if r["representation"] is not None},
            key=param_sort_key,
        )
        if reps:
            picked_rep = st.sidebar.selectbox(
                "Representation",
                reps,
                key=f"{key}_representation",
                help=(
                    "What the graph is built from: a policy layer (needs a trained, "
                    "instrumented policy) or a frozen vision backbone over the episode "
                    "video (needs no policy at all)."
                ),
            )
            candidates = [r for r in candidates if r["representation"] == picked_rep]

    # ── K selector (prominent) ───────────────────────────────────────────────
    # K is the user's complexity choice, so it gets its own top-level selector.
    # When best_per_k is provided, changing K auto-applies the best-scoring
    # hyperparameters for that K so the user never has to tune them manually.
    if "k" in global_varying_params:
        k_vals = sorted({r["k"] for r in candidates if r["k"] is not None}, key=param_sort_key)
        if k_vals:
            on_change_kw: dict = {}
            if best_per_k:
                on_change_kw = {"on_change": _apply_best_for_task_k, "args": (key, best_per_k)}
            picked_k = st.sidebar.selectbox("K (clusters)", k_vals, key=f"{key}_k", **on_change_kw)
            candidates = [r for r in candidates if r["k"] == picked_k]

    # ── Advanced hyperparameters (collapsed by default) ──────────────────────
    # Remaining params are scoped to candidates that already match the chosen K
    # and task, so cross-task / cross-K combos never produce "No run matches".
    _other_varying = [
        p for p, _ in PARAM_LABELS if p not in _PROMOTED_PARAMS and p in global_varying_params
    ]
    if _other_varying:
        with st.sidebar.expander("Advanced hyperparameters", expanded=False):
            for param, label in PARAM_LABELS:
                if param in _PROMOTED_PARAMS or param not in global_varying_params:
                    continue
                vals = sorted(
                    {r[param] for r in candidates if r[param] is not None},
                    key=param_sort_key,
                )
                if not vals:
                    continue
                picked = st.selectbox(label, vals, key=f"{key}_{param}")
                candidates = [r for r in candidates if r[param] == picked]
                if not candidates or len(candidates) == 1:
                    break

    run_id_slot = st.sidebar.container()
    manual = st.sidebar.text_input(
        "…or run dir path",
        key=f"{key}_manual",
        help="Paste an absolute path to bypass the selectors above.",
    ).strip()
    if manual:
        p = Path(manual)
        if is_run_dir(p):
            _show_run_id(p, run_id_slot)
            _warn_if_no_metrics(p)
            return p
        st.sidebar.error(f"`{p}` is not a run dir.")
        return None

    if not candidates:
        st.sidebar.error("No run matches the selected parameters.")
        return None

    # Prefer a metrics-complete candidate over a broken/interrupted one that
    # happens to share the same (task, k, w, s, reduction, layer) tuple --
    # e.g. a crashed job retried under a new hash. Falls back to whatever
    # matched if every candidate lacks metrics.json (nothing to prefer).
    with_metrics = [r for r in candidates if r.get("has_metrics")]
    if with_metrics:
        candidates = with_metrics

    p = Path(candidates[0]["path"])
    if not is_run_dir(p):
        st.sidebar.error(f"`{p}` is not a valid run dir.")
        return None
    _show_run_id(p, run_id_slot)
    _warn_if_no_metrics(p)
    return p


def _resolve_rollout_dir(raw: str | None, data_path: str | None) -> Path | None:
    """Existing rollout dir for a run, or None (with a warning) when missing.

    Wraps :func:`behavior_graphs.pipeline.artifacts.resolve_rollout_dir` with
    the existence check the viz apps need: a missing dir means no videos, and
    the usual cause is $DATA_PATH not being set, so say so instead of
    disabling videos silently.
    """
    p = resolve_rollout_dir(raw, data_path)
    if p is None:
        return None
    if not p.is_dir():
        _log.warning(
            "rollout_dir %r resolves to %s which does not exist — videos disabled. "
            "Recorded paths are relative to the data root; is $DATA_PATH set?",
            raw,
            p,
        )
        return None
    return p


@dataclass
class LoadedRun:
    """Everything the ported pages need, in their legacy shapes."""

    run_dir: Path
    labels: np.ndarray
    metadata: list[dict]  # legacy per-window dict list (see viz.compat)
    graph: BehaviorGraph
    windows: WindowSet
    coords: np.ndarray | None
    node_values: dict[int, float] | None
    #: The reduced features the clustering was fitted on. Anything measuring
    #: distance to a cluster wants these, not ``windows.features`` — see
    #: ``in_clustering_space``. None when the run has no ``reduced.npy``.
    reduced: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    rollout_dir: Path | None = None
    #: The run's persisted sub-cluster splits, or None when the optional
    #: subclustering stage never ran for it (the app then computes any
    #: expansion on demand).
    subclusters: SubClusterSet | None = None


@st.cache_resource(show_spinner="Loading run artifacts…")
def load_run(run_dir_str: str, data_path: str | None = None) -> LoadedRun:
    """Load a run's artifacts.

    ``data_path`` resolves a ``metrics.json["rollout_dir"]`` recorded relative
    to the ``$DATA_PATH`` root; pass :func:`resolve_data_path` from the caller
    (an app entrypoint) rather than reading the env var here.
    """
    run_dir = Path(run_dir_str)
    artifacts = RunArtifacts(run_dir)
    windows = artifacts.load_windows()
    labels = artifacts.load_array("labels").astype(np.int64)
    metadata = meta_from_windows(windows)
    if (run_dir / "graph.json").exists():
        graph = artifacts.load_graph()
    else:
        graph = BehaviorGraph.from_window_labels(labels, windows)
    coords = None
    if (run_dir / "coords_2d.npy").exists():
        coords = artifacts.load_array("coords_2d")
    reduced = artifacts.load_array("reduced") if (run_dir / "reduced.npy").exists() else None
    metrics = artifacts.load_metrics()
    rollout_dir = _resolve_rollout_dir(metrics.get("rollout_dir"), data_path)
    return LoadedRun(
        run_dir=run_dir,
        labels=labels,
        metadata=metadata,
        graph=graph,
        windows=windows,
        coords=coords,
        reduced=reduced,
        node_values=artifacts.load_node_values(),
        metrics=metrics,
        rollout_dir=rollout_dir,
        subclusters=artifacts.load_subclusters(),
    )


def empty_mp4_index() -> dict:
    """The shape :func:`legacy_mp4_index` returns when there are no videos.

    A constructor rather than a literal so ``by_index`` cannot be forgotten —
    :func:`find_mp4_episode` indexes it directly.
    """
    return {"episodes": [], "by_index": {}, "cameras": []}


def _build_mp4_index(rollout_dir_str: str, primary: str | None = None) -> dict:
    rollout_dir = Path(rollout_dir_str)
    try:
        rollouts = RolloutDirectory(rollout_dir, primary_camera=primary)
    except (FileNotFoundError, ValueError) as e:
        _log.warning("no rollout videos: %s is not a rollout dir (%s)", rollout_dir, e)
        return empty_mp4_index()
    episodes = []
    for ep in rollouts.episodes():
        # Every camera view of the episode. "path" stays the first one so that
        # the panels and thumbnail strips that show a single view are unchanged.
        clips = rollouts.video_paths(ep.index)
        if not clips:
            continue
        paths = {camera: str(clip.relative_to(rollout_dir)) for camera, clip in clips.items()}
        episodes.append(
            {
                "index": int(ep.index),
                "success": ep.success,
                "path": next(iter(paths.values())),
                "paths": paths,
                "frame_count": int(ep.num_samples),
            }
        )
    # "by_index" gives O(1) episode lookup — the video panels resolve an entry
    # per rendered clip and per thumbnail, which was a linear scan each time.
    return {
        "episodes": episodes,
        "by_index": {e["index"]: e for e in episodes},
        # Views no surviving episode has a clip for are dropped, so the camera
        # picker never offers one that would render an empty panel.
        "cameras": [c for c in rollouts.camera_names() if any(c in e["paths"] for e in episodes)],
    }


# ttl, like the sibling caches above: a rollout dir is live data on the
# deployed stack (bind-mounted while collection and curation run), so a
# process-lifetime cache would hide newly written episodes indefinitely and
# keep serving deleted ones.
_legacy_mp4_index_cached = st.cache_data(ttl=60, show_spinner=False)(_build_mp4_index)


def legacy_mp4_index(rollout_dir: Path | None) -> tuple[Path | None, dict]:
    """Build the legacy ``(mp4_dir, index.json)`` pair from a rollout dir.

    The ported pages join ``mp4_dir / entry["path"]`` for each episode entry,
    so we emit paths relative to the rollout dir root. The dir is None whenever
    there are no videos to join it with — no rollout dir, not a rollout dir, or
    a valid one that happens to contain no clips — which is the absence every
    consumer already checks for (``clips.clip_paths``, ``if mp4_dir``), rather
    than a fake path they have to know to recognise. Cached for 60 s:
    scanning a rollout directory costs a stat per episode, which is not worth
    repeating on every widget interaction, but the directory does change under
    a running server.
    """
    if rollout_dir is None:
        return None, empty_mp4_index()
    # The configured default view is part of the key, not read inside the cached
    # build: an edit to `primary_camera:` then lands on the next rerun like the
    # rest of configs/data, instead of waiting out this cache's own TTL.
    index = _legacy_mp4_index_cached(str(rollout_dir), primary_camera(rollout_dir.name))
    if not index["episodes"]:
        return None, index
    return rollout_dir, index


def mp4_index_cameras(mp4_index: dict) -> list[str]:
    """The camera views present in *mp4_index*, in the layout's own order.

    ``[]`` when there are no videos, one name for a single-view dataset. Anything
    longer is what the sidebar camera picker offers (see
    :mod:`behavior_graphs.viz.streamlit.components.camera_views`). ``.get``, not
    indexing: an index pickled by an older build has no "cameras" key.
    """
    return list(mp4_index.get("cameras") or [])


def find_mp4_episode(ep_idx: int, mp4_index: dict) -> dict | None:
    """The :func:`legacy_mp4_index` episode entry for *ep_idx*, if present.

    Indexes ``by_index`` directly: every index comes from this module (see
    :func:`empty_mp4_index`), so a KeyError here means someone hand-built one
    and should be told, not silently given an O(n) scan.
    """
    return mp4_index["by_index"].get(ep_idx)
