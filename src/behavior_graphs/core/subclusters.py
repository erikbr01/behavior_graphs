"""Sub-clustering of a terminal (or any) node's episodes.

Behavior graphs and the trajectory tree both collapse every episode ending
at the same outcome into one node/leaf. This module re-clusters the points
belonging to one such node to reveal finer-grained structure (e.g. distinct
failure modes within one FAILURE leaf).

The "points" clustered are every window in each episode's *last segment* —
the maximal trailing run of consecutive same-label windows before the node's
terminal transition, mirroring the segment concept :meth:`BehaviorGraph.
from_window_labels` collapses into one visit. Clustering runs at the window
level (more points, finer local structure than one point per episode); an
episode is then assigned to a sub-cluster by majority vote across its own
segment's window-level sub-labels.

:func:`subcluster_episodes` is the whole gather -> reduce -> cluster ->
vote sequence for one node's episode set, and is the single entry point for
both consumers:

* the optional ``subclustering`` pipeline stage (:mod:`behavior_graphs.stages.
  subclustering.terminal`), which runs it over every terminal leaf of the
  trajectory tree and persists the result as ``subclusters.json``; and
* the demo app, which runs it on demand for one leaf the user expanded, at
  whatever algorithm/k the sub-cluster settings are currently set to.

The app therefore stays interactive whether or not the stage ran — a run
with the artifact just starts from the parameters it recorded.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from behavior_graphs.core.stages import Clustering, DimReduction
from behavior_graphs.core.types import WindowSet

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
#: Bare clusterer configs the sub-clustering algorithm selector offers. A
#: sub-directory of the ``subclustering`` stage group rather than the group
#: itself: the group's own entries (e.g. ``terminal.yaml``) are whole stages,
#: these are the clusterers such a stage — or the app's on-demand path —
#: picks from by name.
SUBCLUSTERING_CONFIG_DIR = CONFIG_ROOT / "pipeline" / "subclustering" / "algo"
#: Reused as-is (not a subclustering-specific copy) so there's one source of
#: truth for reduction algorithms/params; n_components is always overridden
#: per call anyway (see reduce_for_subclustering).
REDUCTION_CONFIG_DIR = CONFIG_ROOT / "pipeline" / "reduction"

#: Noise / unclustered bucket key, matching the -1 noise-label convention
#: used everywhere else in the pipeline (e.g. HDBSCAN).
UNCLUSTERED = -1

#: Sub-clustering's default reducer and target dimensionality, applied to a
#: leaf's pooled last-segment window features before clustering (see
#: reduce_for_subclustering). Chosen empirically: a leaf's raw per-window
#: features (e.g. 1024-d) mix real between-episode structure with at least
#: as much within-episode temporal drift (a "last segment" can span dozens
#: of windows), so clustering on them directly needs a large k just to carve
#: out real structure from the noise. A local UMAP refit -- fit fresh on
#: just this leaf's points, not reusing the run's global reduction, which
#: isn't tuned for this leaf's local structure -- separates the two far
#: better than the raw space or a linear reduction (PCA) does, letting a
#: small k recover real modes.
DEFAULT_SUBCLUSTER_ALGORITHM = "kmeans"
DEFAULT_SUBCLUSTER_K = 4
DEFAULT_SUBCLUSTER_REDUCTION = "umap"
DEFAULT_SUBCLUSTER_COMPONENTS = 10
#: Default floor on leaf size for a sub-cluster split to be offered, shared
#: by the ``subclustering`` pipeline stage (:class:`TerminalSubClustering`'s
#: own ``min_episodes`` default) and the demo app's "+" badge threshold, so
#: the two can't silently drift apart.
DEFAULT_MIN_SUBCLUSTER_EPISODES = 4
#: Below this many pooled points, refitting UMAP is unreliable (too few
#: neighbors) and pointless (there's nothing to separate) -- cluster on the
#: raw features instead.
_MIN_POINTS_FOR_REDUCTION = 4

#: Sub-clustering algorithm name -> the constructor kwarg its config uses for
#: cluster count. Lets one ``k`` setting drive any entry of the algo config
#: dir without the caller having to know which clusterer it named.
SUBCLUSTER_K_PARAM = {"kmeans": "n_clusters", "gmm": "n_components"}
_DEFAULT_K_PARAM = "n_clusters"
_SYNTHETIC_ID_NAMESPACE = uuid.UUID("3f6f9a2e-3e0e-4c66-9b0d-7a3b1a2f5c11")
_SYNTHETIC_ID_BASE = 100_000
_SYNTHETIC_ID_SPACE = 90_000_000


def available_subcluster_configs() -> list[str]:
    """Names of the configured sub-clustering algorithms (yaml stems)."""
    if not SUBCLUSTERING_CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in SUBCLUSTERING_CONFIG_DIR.glob("*.yaml"))


def available_subcluster_reductions() -> list[str]:
    """Names of reducers available for a local per-leaf fit."""
    if not REDUCTION_CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in REDUCTION_CONFIG_DIR.glob("*.yaml"))


def _instantiate_config(config_dir: Path, name: str, **overrides):
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in config_dir.glob("*.yaml")) if config_dir.is_dir() else []
        raise ValueError(f"no config {name!r} in {config_dir} (available: {available})")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return instantiate(cfg)


def load_subcluster_clustering(name: str, **overrides) -> Clustering:
    """Instantiate a :class:`Clustering` from ``configs/pipeline/subclustering/algo/{name}.yaml``.

    Reuses the same Hydra ``_target_`` instantiation pattern (and the same
    concrete clusterer classes) as the main pipeline's ``clustering`` config
    group — this *is* the codebase's "configurable clustering mechanism",
    just a separate config group so defaults can be tuned for small
    per-node N and it's never picked up by a pipeline sweep.
    """
    if not (SUBCLUSTERING_CONFIG_DIR / f"{name}.yaml").exists():
        raise ValueError(f"no subclustering config {name!r} in {available_subcluster_configs()}")
    return _instantiate_config(SUBCLUSTERING_CONFIG_DIR, name, **overrides)


@cache
def subcluster_algo_config(name: str) -> dict:
    """Resolved clusterer config dict — the yaml ``split()`` loads by name.

    The top-level pipeline config only carries the *name* ``kmeans``; the
    actual ``n_init``/``random_state`` live in the file. The stage cache key
    must include these or editing the file yields a stale hit.

    Cached: the app's "does the artifact still match current settings" check
    (:func:`behavior_graphs.viz.streamlit.components.trajectory_tree_view.
    _subcluster_groups_for_leaf`) calls this once per expanded leaf on every
    Streamlit rerun, and the config files don't change within a process's
    lifetime.
    """
    return OmegaConf.to_container(OmegaConf.load(SUBCLUSTERING_CONFIG_DIR / f"{name}.yaml"))


@cache
def subcluster_reduction_config(name: str) -> dict:
    """Resolved reducer config dict, same rationale (and caching) as
    :func:`subcluster_algo_config`."""
    return OmegaConf.to_container(OmegaConf.load(REDUCTION_CONFIG_DIR / f"{name}.yaml"))


def load_subcluster_reduction(name: str, **overrides) -> DimReduction:
    """Instantiate a :class:`DimReduction` from ``configs/pipeline/reduction/{name}.yaml``.

    Reuses the main pipeline's own reduction config group and ``DimReduction``
    ABC, same rationale as :func:`load_subcluster_clustering` reusing the
    clustering group.
    """
    path = REDUCTION_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in REDUCTION_CONFIG_DIR.glob("*.yaml"))
        raise ValueError(f"no reduction config {name!r} (available: {available})")
    cfg = OmegaConf.load(path)
    supported = {key: value for key, value in overrides.items() if key in cfg}
    if supported:
        cfg = OmegaConf.merge(cfg, supported)
    return instantiate(cfg)


def validate_subcluster_params(
    *,
    k: int,
    n_components: int,
    min_episodes: int | None = None,
    max_leaves: int | None = None,
) -> None:
    """Validate shared pipeline/UI sub-clustering parameters."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if min_episodes is not None and min_episodes < 1:
        raise ValueError(f"min_episodes must be >= 1, got {min_episodes}")
    if max_leaves is not None and max_leaves < 1:
        raise ValueError(f"max_leaves must be null or >= 1, got {max_leaves}")


def reduce_for_subclustering(
    feats: np.ndarray,
    n_components: int = DEFAULT_SUBCLUSTER_COMPONENTS,
    reduction: str | None = DEFAULT_SUBCLUSTER_REDUCTION,
) -> np.ndarray:
    """Locally refit a reducer on just these pooled window features.

    Fit fresh on only the points passed in (one leaf's last-segment windows),
    not the run's global reduction -- a reduction tuned for the whole run's
    variance isn't tuned for one leaf's local structure. No-op below
    :data:`_MIN_POINTS_FOR_REDUCTION` points, once ``n_components`` would
    not shrink the data, or when *reduction* is None (cluster the raw
    features).
    """
    validate_subcluster_params(k=2, n_components=n_components)
    if reduction is None:
        return feats
    n_points, n_dims = feats.shape
    if n_points < _MIN_POINTS_FOR_REDUCTION or n_components >= n_dims:
        return feats
    n = min(n_components, n_points - 1)
    n_neighbors = min(15, n_points - 1)
    reducer = load_subcluster_reduction(reduction, n_components=n, n_neighbors=n_neighbors)
    return reducer.fit_transform(feats)


def subcluster_episodes(
    episode_indices: list[int],
    labels: np.ndarray,
    windows: WindowSet,
    algorithm: str = DEFAULT_SUBCLUSTER_ALGORITHM,
    k: int = DEFAULT_SUBCLUSTER_K,
    n_components: int = DEFAULT_SUBCLUSTER_COMPONENTS,
    reduction: str | None = DEFAULT_SUBCLUSTER_REDUCTION,
) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, int, int]]]]:
    """Gather -> reduce -> cluster -> majority-vote, for one node's episodes.

    The single implementation behind both the ``subclustering`` pipeline
    stage and the app's on-demand expansion, so a persisted split and one
    recomputed live at the same settings are the same split.

    ``k`` is clamped to the number of points available, so a leaf with fewer
    windows than ``k`` yields fewer sub-clusters instead of failing.

    Returns ``(groups, window_timeline)``:

    * ``groups[sub_label]`` — the episodes whose own windows voted for that
      sub-cluster (see :func:`assign_episode_subclusters`); the union over
      groups is always ``episode_indices`` exactly.
    * ``window_timeline[ep]`` — that episode's own last-segment windows as
      ``(start, end, sub_label)``, sorted by start. This is the window-level
      detail majority voting collapses away, and is what lets a consumer
      render a per-window sub-cluster timeline (e.g. over video playback)
      rather than just the episode's single winning label.

    Both are empty when the node has no non-noise windows at all.
    """
    validate_subcluster_params(k=k, n_components=n_components)
    ep_ids, rows = _terminal_last_segment_rows(episode_indices, labels, windows)
    if not rows:
        return {}, {}
    feats = windows.features[rows]
    reduced = reduce_for_subclustering(feats, n_components=n_components, reduction=reduction)
    kwarg = SUBCLUSTER_K_PARAM.get(algorithm, _DEFAULT_K_PARAM)
    n = max(1, min(k, len(reduced)))
    clustering = load_subcluster_clustering(algorithm, **{kwarg: n})
    sub_labels = clustering.fit_predict(reduced)
    groups = assign_episode_subclusters(episode_indices, ep_ids, sub_labels)

    spans = [(windows.windows[r].start, windows.windows[r].end) for r in rows]
    window_timeline: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for ep_idx, (start, end), sub_label in zip(ep_ids, spans, sub_labels):
        window_timeline[int(ep_idx)].append((int(start), int(end), int(sub_label)))
    for entries in window_timeline.values():
        entries.sort(key=lambda t: t[0])
    return groups, dict(window_timeline)


def _terminal_last_segment_rows(
    episode_indices: list[int],
    labels: np.ndarray,
    windows: WindowSet,
) -> tuple[list[int], list[int]]:
    """Shared gather behind :func:`terminal_last_segment_windows` and
    :func:`terminal_last_segment_window_spans`: window row indices (into
    ``windows.windows``/``windows.features``) for each episode's trailing
    same-label run, not yet materialized into features or spans.

    For each requested episode: sort its windows by ``start`` (skipping
    noise windows, ``label == -1`` — window existence isn't tied to
    clustering, but a noise window carries no meaningful segment identity),
    then walk backward from the end collecting every window that shares the
    last window's label. An episode with no non-noise windows contributes
    nothing (skipped, not zero-padded).

    Returns ``(window_episode_ids, rows)`` where ``window_episode_ids[i]``
    is the episode owning ``windows.windows[rows[i]]``; an episode may
    contribute more than one row. Deterministic given the same inputs, so
    two separate calls (e.g. one for features, one for spans) always agree
    on ``window_episode_ids`` and row order.
    """
    labels = np.asarray(labels)
    wanted = set(episode_indices)
    per_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(
        list
    )  # ep -> [(start, label, row)]
    for i, w in enumerate(windows.windows):
        if w.episode_idx not in wanted:
            continue
        label = int(labels[i])
        if label == -1:
            continue
        per_episode[w.episode_idx].append((w.start, label, i))

    window_episode_ids: list[int] = []
    rows: list[int] = []
    for ep_idx in episode_indices:
        entries = per_episode.get(ep_idx)
        if not entries:
            continue
        entries.sort(key=lambda e: e[0])
        last_label = entries[-1][1]
        for start, label, row in reversed(entries):
            if label != last_label:
                break
            window_episode_ids.append(ep_idx)
            rows.append(row)
        # collected back-to-front; order within the segment doesn't matter
        # for clustering, so no need to reverse back.
    return window_episode_ids, rows


def terminal_last_segment_windows(
    episode_indices: list[int],
    labels: np.ndarray,
    windows: WindowSet,
) -> tuple[list[int], np.ndarray]:
    """Feature vectors of every window in each episode's trailing same-label run.

    Returns ``(window_episode_ids, features)`` where ``window_episode_ids[i]``
    is the episode owning ``features[i]``; an episode may contribute more
    than one row. See :func:`_terminal_last_segment_rows` for the gather.
    """
    window_episode_ids, rows = _terminal_last_segment_rows(episode_indices, labels, windows)
    if not rows:
        return [], np.empty((0, windows.features.shape[1]), dtype=windows.features.dtype)
    return window_episode_ids, windows.features[rows]


def terminal_last_segment_window_spans(
    episode_indices: list[int],
    labels: np.ndarray,
    windows: WindowSet,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Same gather as :func:`terminal_last_segment_windows`, but returns each
    window's own ``(start, end)`` frame span instead of its feature vector.

    Pair index-for-index with the features/sub-labels from
    :func:`terminal_last_segment_windows` (same ``episode_indices``/
    ``labels``/``windows`` -> identical ``window_episode_ids`` and row
    order) to render a per-window sub-cluster timeline for one episode,
    e.g. during video playback.
    """
    window_episode_ids, rows = _terminal_last_segment_rows(episode_indices, labels, windows)
    spans = [(windows.windows[r].start, windows.windows[r].end) for r in rows]
    return window_episode_ids, spans


def assign_episode_subclusters(
    episode_indices: list[int],
    window_episode_ids: list[int],
    sub_labels: np.ndarray,
) -> dict[int, list[int]]:
    """Roll window-level sub-cluster labels up to one label per episode.

    Each episode is assigned the sub-label with the most votes among its own
    windows (ties broken by the lower sub-label id, for determinism); an
    episode whose winning vote is noise (``-1``) is grouped under
    :data:`UNCLUSTERED` rather than dropped. Episodes absent from
    ``window_episode_ids`` (e.g. they had no non-noise windows) are grouped
    under :data:`UNCLUSTERED` too — the union of every returned group's
    episodes always equals ``episode_indices`` exactly.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for ep_idx, label in zip(window_episode_ids, sub_labels):
        votes[int(ep_idx)][int(label)] += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for ep_idx in episode_indices:
        counts = votes.get(ep_idx)
        if not counts:
            groups[UNCLUSTERED].append(ep_idx)
            continue
        best_count = max(counts.values())
        winner = min(label for label, c in counts.items() if c == best_count)
        groups[UNCLUSTERED if winner == -1 else winner].append(ep_idx)
    return dict(groups)


def stable_subcluster_id(parent_key: str, sub_label: int, used_ids: set[int] | None = None) -> int:
    """Deterministic UI-only node id for a leaf/sub-label pair."""
    used_ids = used_ids if used_ids is not None else set()
    key = f"{parent_key}:{sub_label}"
    nid = _SYNTHETIC_ID_BASE + (uuid.uuid5(_SYNTHETIC_ID_NAMESPACE, key).int % _SYNTHETIC_ID_SPACE)
    while nid in used_ids:
        nid += 1
    used_ids.add(nid)
    return nid


# --------------------------------------------------------------- artifacts
# One pipeline run's persisted sub-clusters. Kept here rather than in
# core.result so the compute and the shape it produces stay together; the
# PipelineResult field and the subclusters.json (de)serialization both point
# at these types.


@dataclass
class SubClusterGroup:
    """One trajectory-tree terminal leaf's sub-cluster overlay."""

    #: The node's trajectory-tree path (run-length-collapsed cluster
    #: sequence ending in a terminal token) — the same tuple the app keys
    #: its expanded-leaf state on, so a split can be looked up by the leaf
    #: the user clicked.
    path: tuple[int, ...]
    #: Episode index -> its sub-cluster label; :data:`UNCLUSTERED` for an
    #: episode whose windows were all noise or voted noise. Covers every
    #: episode reaching this leaf.
    episode_subclusters: dict[int, int]
    #: Episode index -> its own last-segment windows as
    #: ``(start, end, sub_label)``, sorted by start. The window-level detail
    #: the per-episode vote collapses away; empty when not retained.
    window_subclusters: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def cluster_id(self) -> int:
        """The terminal token this node ends at (SUCCESS / FAILURE / END)."""
        return self.path[-1]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_subclusters)

    def episode_groups(self) -> dict[int, list[int]]:
        """``{sub_label: sorted episode indices}`` — the inverse mapping."""
        groups: dict[int, list[int]] = defaultdict(list)
        for ep, sub_label in self.episode_subclusters.items():
            groups[sub_label].append(ep)
        return {sub_label: sorted(eps) for sub_label, eps in sorted(groups.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "episode_subclusters": {str(k): int(v) for k, v in self.episode_subclusters.items()},
            "window_subclusters": {
                str(ep): [list(w) for w in spans] for ep, spans in self.window_subclusters.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubClusterGroup:
        return cls(
            path=tuple(int(c) for c in d["path"]),
            episode_subclusters={int(k): int(v) for k, v in d["episode_subclusters"].items()},
            window_subclusters={
                int(ep): [(int(s), int(e), int(sl)) for s, e, sl in spans]
                for ep, spans in d.get("window_subclusters", {}).items()
            },
        )


@dataclass
class SubClusterSet:
    """Every sub-cluster split one run produced, plus the settings used.

    ``params`` records what the stage was configured with (algorithm, k,
    reduction, ...) so a consumer can tell whether a live recompute at its
    own current settings would reproduce this — the demo app uses it to
    decide between serving the artifact and recomputing.
    """

    groups: list[SubClusterGroup] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def by_path(self) -> dict[tuple[int, ...], SubClusterGroup]:
        return {g.path: g for g in self.groups}

    def to_dict(self) -> dict[str, Any]:
        return {"params": self.params, "groups": [g.to_dict() for g in self.groups]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubClusterSet:
        return cls(
            groups=[SubClusterGroup.from_dict(g) for g in d.get("groups", [])],
            params=d.get("params", {}),
        )
