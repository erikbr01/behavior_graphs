"""Rollout-outcome tallies for the Policy Comparison page.

Discovers every rollout dataset the app can see and reads each one's
success/failure counts. Deliberately does *not* go through
:func:`behavior_graphs.viz.streamlit.data.load_run`: comparing policies needs
nothing but ``episodes/metadata.yaml``, whereas loading a run pulls in windows,
labels and the graph — tens of MB per dataset for numbers that live in a few kB
of YAML.

A "policy" here is one rollout dataset: one policy rolled out on one task, with
per-episode outcomes. Pipeline runs are *not* the unit — a sweep holds hundreds
of runs over the same rollouts, and their success rates are all identical
because the graph pipeline never touches the outcomes.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from behavior_graphs.core.io import resolve_data_path
from behavior_graphs.envs.trajectory_dir import EPISODES_DIR, METADATA_FILE, RolloutDirectory
from behavior_graphs.pipeline.artifacts import resolve_rollout_dir
from behavior_graphs.stats.policy_comparison import PolicyOutcomes
from behavior_graphs.viz.streamlit.data import (
    dataset_display_name,
    display_dataset_names,
    display_run_dirs,
    runs_index,
)

_log = logging.getLogger(__name__)

#: Where :func:`behavior_graphs.core.io.resolve_data_path` keeps rollout datasets
#: (``$DATA_PATH/rollouts``, mirroring ``$ROLLOUTS_DIR`` in deploy/.env).
ROLLOUTS_SUBDIR = "rollouts"


def _metadata_path(dataset_dir: Path) -> Path:
    return dataset_dir / EPISODES_DIR / METADATA_FILE


def is_rollout_dataset(path: Path) -> bool:
    """Whether *path* is a rollout dataset in the canonical layout."""
    return _metadata_path(path).is_file()


@st.cache_data(ttl=60, show_spinner=False)
def _scan_rollouts_root(data_path: str | None) -> dict[str, str]:
    """Rollout datasets directly under ``$DATA_PATH/rollouts``. One readdir."""
    if not data_path:
        return {}
    root = Path(data_path) / ROLLOUTS_SUBDIR
    if not root.is_dir():
        return {}
    return {
        child.name: str(child)
        for child in sorted(root.iterdir())
        if child.is_dir() and is_rollout_dataset(child)
    }


@st.cache_data(ttl=60, show_spinner="Looking for rollout datasets…")
def _scan_indexed_runs(data_path: str | None, raw_dirs: tuple[str, ...]) -> dict[str, str]:
    """Datasets named by the ``rollout_dir`` of an indexed run.

    The fallback route, because it is the expensive one: building the runs
    index walks and parses every run dir under the sweep roots — thousands of
    them, tens of seconds cold — for a page that otherwise needs no runs at
    all. Worth paying only when the standard location came up empty.

    Unlike ``$DATA_PATH/rollouts``, these directories come from arbitrary
    parents, so their basenames can collide (``sweep_a/lift``, ``sweep_b/lift``)
    — keying on the basename alone would drop one of the two from the selector
    entirely. Colliding names are qualified with their parent; unique ones stay
    bare, so the common case reads the same by either route.
    """
    datasets: dict[str, Path] = {}
    for raw in raw_dirs:
        candidate = resolve_rollout_dir(raw, data_path)
        if candidate is not None and is_rollout_dataset(candidate):
            datasets.setdefault(str(candidate), candidate)
    repeats = Counter(p.name for p in datasets.values())
    return {
        (p.name if repeats[p.name] == 1 else f"{p.parent.name}/{p.name}"): str(p)
        for p in datasets.values()
    }


def discover_datasets() -> dict[str, Path]:
    """``{dataset name: dir}`` for every rollout dataset the app may show.

    Cut to the same ``BEHAVIOR_GRAPHS_DISPLAY_DATA`` allowlist as the run
    selectors — a dataset excluded from the deployment must not reappear here
    just because this page found it by a different route.
    """
    data_path = resolve_data_path()
    found = dict(_scan_rollouts_root(data_path))
    if not found:
        # Non-standard layout, or $DATA_PATH unset: ask the runs index where
        # the rollouts each indexed run was built from actually live.
        index = runs_index(display_run_dirs())
        raw_dirs: tuple[str, ...] = ()
        if "rollout_dir" in index.columns:
            raw_dirs = tuple(sorted({str(v) for v in index["rollout_dir"].dropna().unique()}))
        found = _scan_indexed_runs(data_path, raw_dirs)
    allowed = display_dataset_names()
    # Filtered on the directory name, not the key: a key qualified to break a
    # basename collision would match no allowlist entry and silently hide a
    # dataset the deployment does list.
    return {
        name: Path(p)
        for name, p in sorted(found.items())
        if allowed is None or Path(p).name in allowed
    }


@st.cache_data(show_spinner=False)
def _outcomes_cached(dataset_dir: str, name: str, _mtime: float) -> tuple[int, int, int]:
    """``(successes, failures, unscored)`` for one dataset.

    ``_mtime`` is a cache key, not an argument: re-collecting a dataset in
    place leaves its path unchanged, and a tally cached on the path alone would
    then describe episodes that no longer exist. That key is also why there is
    no TTL — an expiry could only re-parse 55 kB of YAML to reach the answer it
    already had. Noticing a *new* dataset is ``_scan_rollouts_root``'s job.
    """
    episodes = RolloutDirectory(Path(dataset_dir)).episodes()
    successes = sum(1 for e in episodes if e.success is True)
    failures = sum(1 for e in episodes if e.success is False)
    return successes, failures, len(episodes) - successes - failures


@dataclass(frozen=True)
class DatasetOutcomes:
    """One dataset's tally, with the unscored episodes kept separate.

    ``unscored`` is not folded into either column: an episode whose ``success``
    is null was never labelled, and counting it as a failure would quietly
    depress the rate of exactly the datasets whose labelling is incomplete.
    The page surfaces the count instead.
    """

    name: str
    successes: int
    failures: int
    unscored: int

    @property
    def scored(self) -> int:
        return self.successes + self.failures

    def as_arm(self, name: str | None = None) -> PolicyOutcomes:
        """The scored episodes, as the stats layer's input.

        *name* is what to call the policy, for a page that has been given a
        label for it — the arm's name is what every reading of the comparison
        ends up showing, on the chart's axis, in the tables and in the exported
        figure, so renaming it here renames it everywhere at once. Defaults to
        the dataset's own display name.
        """
        return PolicyOutcomes(
            name=name or dataset_display_name(self.name),
            successes=self.successes,
            trials=self.scored,
        )


def load_outcomes(name: str, dataset_dir: Path) -> DatasetOutcomes:
    """Tally one dataset; an unreadable one comes back empty rather than raising.

    A half-written ``metadata.yaml`` is reachable on a live server (rollout
    collection appends to the data root the app is reading), and it must not
    take down a page comparing five other datasets that are fine.

    Deliberately ``except Exception``: a truncated file is not one failure mode
    but a family of them, and which one surfaces depends on where the write was
    cut. ``write_trajectory_dir`` emits the ``source:`` block before
    ``episodes:``, so the likeliest interception raises ``KeyError`` — not an
    ``OSError`` — and a narrow clause would let exactly the documented case
    through.
    """
    meta = _metadata_path(dataset_dir)
    try:
        mtime = meta.stat().st_mtime
        successes, failures, unscored = _outcomes_cached(str(dataset_dir), name, mtime)
    except Exception as exc:
        _log.warning("cannot read outcomes for %s (%s)", dataset_dir, exc)
        successes = failures = unscored = 0
    return DatasetOutcomes(
        name=name,
        successes=successes,
        failures=failures,
        unscored=unscored,
    )
