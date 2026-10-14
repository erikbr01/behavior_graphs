"""Shared trajectory-tree rendering — used by both the comparison app and the
main Policy Doctor visualizer's Behavior Graph tab.

Renders one of {native SVG node-edge tree, Sunburst, Icicle, Treemap} given
a (labels, metadata) pair, plus a "Hide branches reaching fewer than N
episodes" slider and an effectively-uncapped max-depth control. The native
SVG variant reuses the existing clickable-graph component (with curved edges,
drag, reset, and the click-to-explore video panel).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as _components

from behavior_graphs.core import subclusters
from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
    BehaviorNode,
)
from behavior_graphs.core.types import WindowSet
from behavior_graphs.viz import media
from behavior_graphs.viz import simplification as gs
from behavior_graphs.viz.labels import short_node_label, vlm_detail
from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS
from behavior_graphs.viz.simplification import RARE_COLLAPSE_CLUSTER_ID
from behavior_graphs.viz.streamlit.clips import (
    dataset_transforms,
    episode_camera_clips,
    prepare_entry_clips,
)
from behavior_graphs.viz.streamlit.components.graph_selection import (
    labels_identity,
    sync_graph_identity,
)
from behavior_graphs.viz.streamlit.components.panel_layout import DEFAULT_LAYOUT, graph_panel_slots
from behavior_graphs.viz.streamlit.data import empty_mp4_index

# A leaf needs at least this many episodes for a sub-cluster split to be
# meaningful; below this the badge is simply not offered. Shared with the
# ``subclustering`` pipeline stage's own default so the UI never offers a
# split the stage's persisted artifact wouldn't also attempt.
_MIN_SUBCLUSTER_EPISODES = subclusters.DEFAULT_MIN_SUBCLUSTER_EPISODES

#: Minimum horizontal room reserved around a parent → success rail. This is
#: measured in the tree's pre-SVG [0, 1] x coordinates; the component later
#: scales the whole tree, so the important property is the relative gap.
_SUCCESS_RAIL_CLEARANCE = 0.14

#: Canvas pixels of vertical room to give each level of the tree. The layout is
#: normalised to fill whatever canvas it gets, so depth and height together decide
#: how far apart two consecutive levels land — and a label sits 13px below its
#: node's edge on a node that can be 25px in radius, so much under 60px per level
#: draws every label inside the node one level down. See
#: :func:`_canvas_height`.
_PX_PER_LEVEL = 62
#: The component's own top and bottom padding (`padding` in index.html's
#: ``render``), which the levels do not get to use.
_CANVAS_PADDING_PX = 60
#: However deep the tree, one screenful is as far as growing the canvas helps:
#: past this the reader is scrolling the picture rather than reading it, and the
#: labels the extra room would have separated are off-screen anyway.
_CANVAS_MAX_PX = 900


def _canvas_height(nodes: list[dict], minimum: int) -> int:
    """Canvas height in pixels for a tree of this depth, at least *minimum*.

    A fixed height (this was hardcoded at 650) means a 4-deep tree wastes most of
    the canvas while a 14-deep one gets ~40px a level — less than the offset its
    own labels are drawn at, so every one of them lands on the node below.
    """
    depth = max((int(nd["depth"]) for nd in nodes), default=1)
    fitted = 2 * _CANVAS_PADDING_PX + depth * _PX_PER_LEVEL
    return max(minimum, min(_CANVAS_MAX_PX, fitted))


@st.cache_data(show_spinner="Computing sub-clusters...")
def _cached_subcluster_groups(
    _labels: np.ndarray,
    _windows: WindowSet,
    cache_namespace: str,
    leaf_path: tuple[int, ...],
    episode_indices: tuple[int, ...],
    algo_name: str,
    k: int,
    reduction: str | None,
    n_components: int,
) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, int, int]]]]:
    """Cached :func:`subclusters.subcluster_episodes` for one leaf.

    ``_labels``/``_windows`` are excluded from Streamlit's expensive default
    hashing. ``cache_namespace`` identifies the immutable run artifacts, and
    the remaining arguments identify the leaf and every compute parameter.

    Same function the ``subclustering`` pipeline stage runs, so an on-demand
    split at settings matching a run's ``subclusters.json`` reproduces it.
    """
    return subclusters.subcluster_episodes(
        list(episode_indices),
        _labels,
        _windows,
        algorithm=algo_name,
        k=k,
        reduction=reduction,
        n_components=n_components,
    )


def _subcluster_defaults(
    precomputed: subclusters.SubClusterSet | None,
) -> tuple[str, int, str | None, int]:
    """Sub-cluster settings to open the app with.

    A run whose ``subclustering`` stage ran already chose these; starting
    from them means the first expansion is served from the artifact instead
    of recomputing something slightly different. Falls back to the app's own
    defaults for runs without the stage.
    """
    params = (precomputed.params if precomputed else None) or {}
    algo = params.get("algorithm", subclusters.DEFAULT_SUBCLUSTER_ALGORITHM)
    k = int(params.get("k", subclusters.DEFAULT_SUBCLUSTER_K))
    reduction = params.get("reduction", subclusters.DEFAULT_SUBCLUSTER_REDUCTION)
    n_components = int(params.get("n_components", subclusters.DEFAULT_SUBCLUSTER_COMPONENTS))
    return algo, k, reduction, n_components


def _subcluster_groups_for_leaf(
    leaf_path: tuple,
    episode_indices: list[int],
    labels: np.ndarray,
    windows: WindowSet,
    cache_namespace: str,
    algo_name: str,
    k: int,
    reduction: str | None,
    n_components: int,
    precomputed: subclusters.SubClusterSet | None,
) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, int, int]]]]:
    """One leaf's split, from the run's artifact when it matches, else live.

    The artifact is only usable when the user hasn't moved the sub-cluster
    settings away from what produced it — otherwise the on-screen controls
    would silently not apply. Same for a leaf the stage skipped (below its
    ``min_episodes``, or past its ``max_leaves`` cap): fall through and
    compute it.
    """
    if precomputed is not None:
        want = _subcluster_defaults(precomputed)
        current = (algo_name, k, reduction, n_components)
        keeps_timeline = bool(precomputed.params.get("keep_window_detail", True))
        configs_match = precomputed.params.get(
            "algorithm_config"
        ) == subclusters.subcluster_algo_config(algo_name) and precomputed.params.get(
            "reduction_config"
        ) == (subclusters.subcluster_reduction_config(reduction) if reduction is not None else None)
        if current == want and keeps_timeline and configs_match:
            group = precomputed.by_path().get(tuple(leaf_path))
            if group is not None:
                return group.episode_groups(), group.window_subclusters
    return _cached_subcluster_groups(
        labels,
        windows,
        cache_namespace,
        tuple(int(c) for c in leaf_path),
        tuple(episode_indices),
        algo_name,
        k,
        reduction,
        n_components,
    )


def _terminal_subcluster_nodes(
    leaf: dict,
    groups: dict[int, list[int]],
    root_n_episodes: int,
    used_ids: set[int],
) -> list[dict]:
    """Format one leaf's sub-cluster groups as tree-node dicts.

    Matches :func:`build_trajectory_tree`'s node schema so injected nodes
    are indistinguishable from real trie nodes to every downstream
    consumer (layout, coloring, the video panel) -- plus one extra
    ``is_subcluster`` marker so the label-rendering code below knows to use
    the precomputed ``label`` instead of treating ``cluster_id`` as a real
    behavior/cluster id (it's a synthetic id from
    :func:`subclusters.stable_subcluster_id`, out of that range). A terminal
    leaf's outcome is fixed, so ``n_success``/``n_failure`` come straight
    from the leaf's own cluster_id rather than being recomputed.
    """
    leaf_path = tuple(leaf["path"])
    cid = leaf["cluster_id"]
    out = []
    for sub_label, members in sorted(groups.items()):
        nid = subclusters.stable_subcluster_id(str(leaf_path), sub_label, used_ids=used_ids)
        n = len(members)
        name = (
            f"{leaf['label']} (unclustered)"
            if sub_label == subclusters.UNCLUSTERED
            else f"{leaf['label']} #{sub_label + 1}"
        )
        out.append(
            {
                "path": leaf_path + (nid,),
                "label": name,
                "cluster_id": nid,
                "depth": leaf["depth"] + 1,
                "n_episodes": n,
                "n_success": n if cid == SUCCESS_NODE_ID else 0,
                "n_failure": n if cid == FAILURE_NODE_ID else 0,
                "parent_path": leaf_path,
                "episode_indices": sorted(members),
                "visit_fraction": n / root_n_episodes if root_n_episodes else 0.0,
                "is_subcluster": True,
            }
        )
    return out


_THUMB_STRIP = _components.declare_component(
    "thumb_strip",
    path=str(Path(__file__).parent / "_thumb_strip"),
)

TREE_VIEW_OPTIONS = ["native_svg", "sunburst", "icicle"]
TREE_VIEW_LABELS = {
    "native_svg": "Trajectory tree",
    "sunburst": "Sunburst",
    "icicle": "Icicle",
}

# Fixed namespace for deriving each tree node's synthetic id from its path
# via uuid5 — arbitrary but constant, so the same path always hashes to the
# same id (see _stable_id in _render_native_svg). Not a secret; just needs
# to not change between releases.
_NODE_ID_NAMESPACE = uuid.UUID("6f52f9c4-7b8b-4b7a-9b7d-7e9c9e6d4b1a")


def _format_branch_path(path: tuple, cluster_names: dict[int, str] | None) -> str:
    """ "START → Behavior 3 → Behavior 9"-style label for a branch-point path."""
    cluster_names = cluster_names or {}
    disp = ["START"]
    for cid in path:
        if cid == SUCCESS_NODE_ID:
            disp.append("✓")
        elif cid == FAILURE_NODE_ID:
            disp.append("✗")
        elif cid == END_NODE_ID:
            disp.append("END")
        else:
            disp.append(cluster_names.get(int(cid), f"Behavior {cid}"))
    return " → ".join(disp)


def _build_tree_nodes(
    labels: np.ndarray,
    metadata: list[dict],
    level: str,
    min_branch: int,
    max_depth_cap: int,
) -> tuple[list[dict], list[dict], int]:
    """Trie for the trajectory tree plus the filtered/pruned node list.

    The prune is a fixed point over "reachable from root" and "reaches a
    terminal", which is quadratic in the node count — worth doing once per
    (run, min_branch) instead of on every rerun.
    """
    nodes = gs.build_trajectory_tree(labels, metadata, level=level)
    nodes_f = [
        nd
        for nd in nodes
        if nd["n_episodes"] >= int(min_branch) and nd["depth"] <= int(max_depth_cap)
    ]

    # Auto-prune unreachable + false-terminal nodes to a fixed point.
    n_pruned = 0
    while True:
        keep = {tuple(nd["path"]) for nd in nodes_f}
        children: dict[tuple, list[tuple]] = defaultdict(list)
        for nd in nodes_f:
            if nd["parent_path"] is not None and tuple(nd["parent_path"]) in keep:
                children[tuple(nd["parent_path"])].append(tuple(nd["path"]))
        new_f = []
        by_p = {tuple(nd["path"]): nd for nd in nodes_f}
        for nd in nodes_f:
            p = tuple(nd["path"])
            # Reachable from root
            anc = nd["parent_path"]
            ok = True
            while anc is not None:
                if tuple(anc) not in keep:
                    ok = False
                    break
                anc = tuple(anc)[:-1] if len(anc) > 0 else None
                if anc == ():
                    break
            if not ok:
                continue
            # Reaches a terminal
            if nd["cluster_id"] in TERMINAL_NODE_IDS:
                new_f.append(nd)
                continue
            stack = [p]
            seen = set()
            found = False
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for cp in children.get(cur, []):
                    if cp not in by_p:
                        continue
                    if by_p[cp]["cluster_id"] in TERMINAL_NODE_IDS:
                        found = True
                        break
                    stack.append(cp)
                if found:
                    break
            if found:
                new_f.append(nd)
        if len(new_f) == len(nodes_f):
            break
        n_pruned += len(nodes_f) - len(new_f)
        nodes_f = new_f
    return nodes, nodes_f, n_pruned


@st.cache_resource(show_spinner=False, max_entries=64)
def _tree_nodes_cached(
    sig: str,
    level: str,
    min_branch: int,
    max_depth_cap: int,
    _labels: np.ndarray,
    _metadata: list[dict],
) -> tuple[list[dict], list[dict], int]:
    """Cached :func:`_build_tree_nodes`, keyed by the run's artifact signature.

    ``cache_resource`` (not ``cache_data``): the node dicts are read-only for
    every caller — see the copy-on-relabel in :func:`render_trajectory_tree` —
    so handing out the same objects beats deep-copying thousands of dicts on
    each rerun. ``_labels``/``_metadata`` are excluded from the cache key
    (leading underscore); *sig* identifies them.
    """
    return _build_tree_nodes(_labels, _metadata, level, min_branch, max_depth_cap)


def outcome_color_bins() -> list[str]:
    """The five colors a node is drawn in, worst outcome first.

    Module level because the study draws a legend for this scale: a second copy
    of the palette would keep the graph and its key in sync only by luck.
    """
    if st.session_state.get("colorblind_mode", False):
        return ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#0072B2"]  # Okabe-Ito
    return ["#d62728", "#ff7f0e", "#e8c32a", "#9dc95d", "#2ca02c"]


def outcome_color(rate: float) -> str:
    """The bin a node with this success rate is drawn in.

    Module level for the same reason as :func:`outcome_color_bins`: the study's
    "how to read" figure colors its example nodes, and a second copy of the
    thresholds would agree with the graph only by luck.
    """
    return outcome_color_bins()[min(4, int(max(0.0, min(1.0, rate)) * 5))]


def _tree_nodes(
    labels: np.ndarray,
    metadata: list[dict],
    *,
    level: str,
    min_branch: int,
    max_depth_cap: int,
) -> tuple[list[dict], list[dict], int]:
    from behavior_graphs.viz.streamlit.episode_index import signature

    return _tree_nodes_cached(
        signature(labels, metadata), level, min_branch, max_depth_cap, labels, metadata
    )


def render_trajectory_tree(
    labels: np.ndarray,
    metadata: list[dict],
    *,
    view_mode: str = "native_svg",
    min_branch: int = 2,
    max_depth_cap: int = 500,
    color_mode: str = "outcome",
    node_values: dict[int, float] | None = None,
    cluster_names: dict[int, str] | None = None,
    node_vlm_labels: dict[int, dict] | None = None,
    mp4_dir: Path | None = None,
    mp4_index: dict | None = None,
    height: int = 600,
    level: str = "rollout",
    key_prefix: str = "tree",
    show_stats: bool = True,
    show_paths: bool = True,
    show_fold_toggles: bool = True,
    show_annotations: bool = True,
    show_camera_labels: bool = True,
    show_vlm_descriptions: bool = True,
    height_extra: int = 0,
    theme: str | None = None,
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 16.0,
    min_visit_fraction: float | None = None,
    #: Show the dataset's human subtask segmentation under each clip. Off by
    #: default: these panels also render inside the user study, where the
    #: ground truth is the answer key to the judgement being measured.
    show_ground_truth: bool = False,
    windows: WindowSet | None = None,
    precomputed_subclusters: subclusters.SubClusterSet | None = None,
    subcluster_cache_namespace: str | None = None,
    panel_layout: str = DEFAULT_LAYOUT,
    static_figure: bool = False,
    episode_numbers: dict[int, int] | None = None,
    paper_export: bool = False,
    paper_export_filename: str = "behavior_graph",
    paper_export_preset: str = "tuned",
    paper_export_expanded: bool = False,
    clip_export: bool = False,
    clip_export_filename: str = "clip",
) -> None:
    """Render the trajectory tree. Self-contained: produces controls (none —
    the caller is expected to provide the min_branch / view selectors)
    and the chart + optional stats below it.

    ``height_extra`` adds vertical room on top of whatever the tree's depth
    works out to (:func:`_canvas_height`), for a page that wants the graph to
    breathe rather than to fit.

    ``min_visit_fraction``, if set, bounds branching *without hiding any
    outcome*: at each branch point, children accounting for less than this
    fraction of the whole run's episodes fold into one "+N rare variants"
    node carrying their aggregate stats and full episode list, while
    children at or above the fraction keep their own subtree regardless of
    how many there are (so a wide, roughly-uniform branch point — e.g. the
    root splitting into K similarly-sized initial behaviors — doesn't get a
    majority of its data mislabeled "rare" just for not being in some fixed
    top-N). Unlike ``min_branch`` (which can disconnect a whole rare-outcome
    tail from the tree entirely), a rare branch stays one click away.

    ``windows``, if given, enables a "+" badge on terminal leaves
    (SUCCESS/FAILURE/END) with enough episodes: expanding one runs a
    second, user-configurable clustering pass over the points belonging
    only to that leaf's episodes and reveals the result as synthetic child
    nodes (see :mod:`behavior_graphs.core.subclusters`). Without ``windows``
    terminal leaves render exactly as before, with no badge.

    ``precomputed_subclusters`` is a run's persisted ``subclusters.json``
    (present when the optional ``subclustering`` pipeline stage ran). It
    seeds the sub-cluster settings and serves expansions directly while
    those settings are unchanged; move them and expansions recompute live.
    Passing None just means every expansion computes on demand.

    ``subcluster_cache_namespace`` must identify the immutable run artifacts
    whenever ``windows`` is supplied. It prevents two runs with the same
    episode indices and settings from sharing cached feature-space results.

    ``panel_layout`` puts the selection videos to the right of the tree (the
    default) or below it (see
    :mod:`behavior_graphs.viz.streamlit.components.panel_layout`); it only applies
    to the ``native_svg`` view, the only one with a selection panel.

    ``height`` is a floor on the canvas height, not the canvas height: the
    ``native_svg`` view grows it with the tree's depth so every level gets room
    for its labels (:func:`_canvas_height`). The plotly views take it as given.

    ``static_figure`` draws the tree as a picture: no dragging, no clicking, no
    selection panel, no fold badges. For a page holding several trees at once
    (Policy Comparison draws one per policy), where the way into any single tree
    is a link to the page built for exploring it.

    ``clip_export`` adds a panel under the video player offering the clip on
    screen as an MP4 with its behavior bar burned in (see
    :mod:`behavior_graphs.viz.streamlit.components.clip_export`). Off by default
    and opted into by the two pages where a figure is being made: a study app
    must not offer a participant a download, and the reviewer page is anonymous,
    where a file named after a run directory is exactly the thing it withholds.
    ``clip_export_filename`` is the stem those files get, with the episode
    number appended.
    """
    # The component is self-themed: its stylesheet has a dark default and a
    # `theme-light` override, and nothing was choosing between them, so every
    # caller got the dark one. Resolved from Streamlit's own theme here rather
    # than at each call site: the one deployment that is light (the reviewer
    # site) would otherwise draw light labels on a white canvas.
    if theme is None:
        theme = str(st.get_option("theme.base") or "dark")

    cluster_names = cluster_names or {}
    if windows is not None and not subcluster_cache_namespace:
        raise ValueError("window-backed sub-clustering requires subcluster_cache_namespace")

    # Before anything reads a selection: if these labels are a different
    # clustering than the one the stored selection was made on, drop it. Named
    # by labels alone — folding a branch changes the tree's node ids without
    # making it a different graph. Done here rather than inside
    # render_graph_full_width below because _render_stats renders the path
    # buttons first and marks the active one from `highlighted_path`.
    tree_identity = labels_identity(labels, metadata)
    sync_graph_identity(key_prefix, tree_identity)

    # Build the trie (cached — see _tree_nodes)
    nodes, nodes_f, n_pruned = _tree_nodes(
        labels,
        metadata,
        level=level,
        min_branch=int(min_branch),
        max_depth_cap=int(max_depth_cap),
    )
    if cluster_names:
        # The trie is cached and shared between callers, so relabel copies
        # rather than mutating the cached dicts in place.
        def _named(nd: dict) -> dict:
            cid = nd["cluster_id"]
            name = cluster_names.get(int(cid)) if cid >= 0 else None
            return {**nd, "label": name} if name else nd

        nodes = [_named(nd) for nd in nodes]
        nodes_f = [_named(nd) for nd in nodes_f]

    if not nodes_f:
        st.warning("No nodes match the filter; lower 'min branch'.")
        return

    # Slots for the tree and its path summary, reserved before the fold/collapse
    # bookkeeping below so everything that block renders — how many branches are
    # folded, which ones the user opened by hand, the sub-cluster settings —
    # lands *under* the picture it is describing. That bookkeeping has to run
    # first (it decides which nodes the tree draws), and it used to render where
    # it ran: five rows of small print above the tree, so the page opened on
    # captions about a graph that was below the fold.
    graph_area = st.container()
    stats_area = st.container()

    # Part of the stats, not of the picture: "17 nodes pruned." is a note about
    # how the drawing was made, for a reader who chose the min_branch that
    # pruned them. A study participant chose nothing and has no way to act on
    # it — it just reads as a warning that the graph in front of them is
    # incomplete. Off wherever the rest of the stats are off.
    if show_stats and n_pruned > 0:
        st.caption(f"{n_pruned} nodes pruned.")

    # None (rather than a number) tells _render_native_svg to fall back to
    # its own dynamic max-depth computation — fine when nothing is ever
    # collapsed, since the tree shape never changes between clicks then.
    max_depth_reachable: int | None = None
    expanded_for_render: frozenset[tuple] = frozenset()
    manually_collapsed_for_render: frozenset[tuple] = frozenset()

    # Every non-terminal node gets a +/- badge (see fold_state in
    # _render_native_svg), independent of the rare-branch-folding toggle
    # below. Two session-state sets track why a branch point is collapsed:
    #   `expanded`           — a "+N rare variants" bucket the user opened
    #                          (only meaningful while min_visit_fraction
    #                          is on; re-collapsing folds it back into the
    #                          bucket instead of hiding it standalone).
    #   `manually_collapsed` — any other branch point the user hid by hand;
    #                          works regardless of rare-branch folding.
    # Both are populated by the *same* click message (see fold_toggle below)
    # so there is one badge protocol for every node, not two.
    _expanded_key = f"{key_prefix}_expanded_buckets"
    _expand_processed_key = f"{key_prefix}_expand_last_processed"
    _mc_key = f"{key_prefix}_manually_collapsed"
    _toggle_processed_key = f"{key_prefix}_toggle_last_processed"
    expanded: set[tuple] = set(st.session_state.get(_expanded_key, set()))
    manually_collapsed: set[tuple] = set(st.session_state.get(_mc_key, set()))
    _id_to_path = st.session_state.get(f"{key_prefix}_id_to_path", {})

    # A third, independent session-state set: terminal leaves the user has
    # expanded into feature-space sub-clusters. Kept separate from
    # `expanded` (rare-branch-bucket reveals) — the two badges mean
    # different things and conflating them would make the unified
    # fold-toggle handler below ambiguous about what a given path means.
    _sc_key = f"{key_prefix}_expanded_subclusters"
    expanded_subclusters: set[tuple] = set(st.session_state.get(_sc_key, set()))

    if min_visit_fraction:
        # A summary node the user has clicked open stays open: track which
        # branch points ("+N rare variants" parent paths) have been expanded
        # in session state, keyed off the click id the graph component
        # already writes to f"{key_prefix}_graph_selected" (same mechanism
        # the video panel below reads). Detected here, before collapsing,
        # using the *previous* render's id_to_path — the mapping that was
        # actually in effect when the user clicked. Kept as a second way to
        # open a bucket (in addition to its "+" badge) since the caption
        # below still advertises "click a summary node to expand it".
        _clicked_id = st.session_state.get(f"{key_prefix}_graph_selected")
        if _clicked_id is not None and _clicked_id != st.session_state.get(_expand_processed_key):
            st.session_state[_expand_processed_key] = _clicked_id
            _clicked_path = _id_to_path.get(_clicked_id)
            if _clicked_path and _clicked_path[-1] == gs.RARE_COLLAPSE_CLUSTER_ID:
                _bucket_parent = _clicked_path[:-1]
                # The whole bucket, not its first layer: everything inside is
                # rarer than the child that got folded, so revealing one level
                # would just hand the user another summary node (see
                # gs.bucket_reveal_paths).
                _reveal = gs.bucket_reveal_paths(nodes_f, min_visit_fraction, _bucket_parent)
                if not _reveal <= expanded:
                    expanded |= _reveal
                    st.session_state[_expanded_key] = expanded
                    # This click expanded a bucket rather than selecting a
                    # real node — don't also try to show it in the video
                    # panel below (that node no longer exists after this).
                    st.session_state.pop(f"{key_prefix}_graph_selected", None)

    # Unified +/- badge click, reported separately from normal node
    # selection so it never also opens the video/node panel. One message
    # covers every case a badge click can mean:
    #   bucket "+"                -> reveal its rare children (-> expanded)
    #   a revealed rare node "-"  -> fold back into its sibling bucket
    #   a manually-collapsed "+"  -> reveal its subtree again
    #   any other node "-"        -> hide its subtree by hand
    _ft_key = f"{key_prefix}_graph_fold_toggle"
    _ft_value = st.session_state.get(_ft_key)
    if _ft_value is not None and _ft_value != st.session_state.get(_toggle_processed_key):
        st.session_state[_toggle_processed_key] = _ft_value
        _ft_id = _ft_value[0] if isinstance(_ft_value, tuple | list) else _ft_value
        _ft_path = _id_to_path.get(_ft_id)
        if _ft_path is not None:
            if _ft_path and _ft_path[-1] == gs.RARE_COLLAPSE_CLUSTER_ID:
                # Same gesture as clicking the summary node itself, so the same
                # reveal: all the way down, not one layer.
                expanded |= gs.bucket_reveal_paths(nodes_f, min_visit_fraction, _ft_path[:-1])
                st.session_state[_expanded_key] = expanded
            elif _ft_path in expanded:
                expanded = {q for q in expanded if q[: len(_ft_path)] != _ft_path}
                st.session_state[_expanded_key] = expanded
            elif _ft_path in expanded_subclusters:
                expanded_subclusters = expanded_subclusters - {_ft_path}
                st.session_state[_sc_key] = expanded_subclusters
            elif _ft_path and _ft_path[-1] in TERMINAL_NODE_IDS:
                expanded_subclusters = expanded_subclusters | {_ft_path}
                st.session_state[_sc_key] = expanded_subclusters
            elif _ft_path in manually_collapsed:
                manually_collapsed = {q for q in manually_collapsed if q != _ft_path}
                st.session_state[_mc_key] = manually_collapsed
            else:
                manually_collapsed = manually_collapsed | {_ft_path}
                st.session_state[_mc_key] = manually_collapsed
            # No st.rerun() here: render_graph_component (called later this
            # same script run, inside _render_native_svg) is what first
            # writes _ft_key's fresh value and forces the rerun that gets us
            # here with fresh data — see the comment there. By the time this
            # code sees a fresh _ft_value, we're already on that follow-up
            # run, early enough that just continuing this run picks up the
            # updated collapsed/expanded sets below.

    # The deepest depth reachable through *any* combination of expansion —
    # captured before collapsing — anchors the vertical scale (see max_d in
    # _render_native_svg) so revealing a deeper level never rescales
    # already-visible nodes.
    max_depth_reachable = max((nd["depth"] for nd in nodes_f), default=0)

    # The full tree, kept for the questions that are about what *could* be shown
    # rather than what is: which paths a bucket stands for (bucket_reveal_paths
    # below) cannot be answered from a list those paths have been folded out of.
    nodes_uncollapsed = nodes_f

    manually_collapsed_for_render = frozenset(manually_collapsed)
    nodes_f = gs.collapse_manual(nodes_f, manually_collapsed_for_render)

    if min_visit_fraction:
        expanded_for_render = frozenset(expanded)
        nodes_f = gs.collapse_rare_children(nodes_f, min_visit_fraction, expanded_for_render)
        n_folded = sum(
            nd["n_folded_branches"]
            for nd in nodes_f
            if nd["cluster_id"] == gs.RARE_COLLAPSE_CLUSTER_ID
        )
        if n_folded > 0 or expanded:
            # Terse on purpose: the long version explaining that nothing is
            # hidden and that a summary node expands lives in the "+N" node's
            # own tooltip and in the reading guide beside the graph. Repeating
            # it above every tree cost three lines to say what one says.
            st.caption(
                f"{n_folded} rare branches (< {min_visit_fraction:.1%} of episodes) "
                + (
                    'folded into the "+N" summary nodes.'
                    if static_figure
                    # A figure has no badge to press and sends no clicks, so the
                    # instruction would be an instruction to do nothing.
                    else "folded into ➕ summary nodes — click one to expand it."
                )
            )

        # Expanding a bucket puts its whole revealed subtree in `expanded`, so
        # only the top of each such subtree is worth offering: the entries below
        # it are showing because it is. A path is one of those tops iff none of
        # its own *proper, non-empty* prefixes was expanded — depth-many set
        # lookups each, rather than comparing every entry against every other.
        #
        # The empty tuple has to be excluded from that test even though it is a
        # real branch point (the root, whose rare children are the tree's rare
        # initial behaviors): as a prefix of every path it would make one root
        # expansion hide every other, unrelated one.
        expanded_roots = {
            p for p in expanded if not any(p[:i] in expanded for i in range(1, len(p)))
        }
        if expanded_roots:
            # What each listed entry would take back with it, worked out while the
            # full tree is in hand: the reveal cannot be recomputed from a node
            # list those paths have been folded out of. The callbacks below capture
            # these sets rather than the tree itself.
            _reveals = {
                _root: gs.bucket_reveal_paths(nodes_uncollapsed, min_visit_fraction, _root)
                for _root in expanded_roots
            }

            def _collapse(root: tuple) -> None:
                # Drop exactly what expanding this branch point revealed, rather
                # than every expanded path beneath it: an expansion somewhere else
                # under the same ancestor is a separate choice, and collapsing this
                # one should not undo it.
                st.session_state[_expanded_key] = (
                    set(st.session_state.get(_expanded_key, set())) - _reveals[root]
                )

            with st.expander(f"Expanded branches ({len(expanded_roots)})", expanded=False):
                for _p in sorted(expanded_roots, key=len):
                    _c1, _c2 = st.columns([6, 1])
                    _c1.caption(_format_branch_path(_p, cluster_names))
                    # on_click, not a post-hoc st.rerun: the callback lands before
                    # the rerun it triggers, so the tree below is already drawn
                    # from the new state (and AppTest can drive it, which a
                    # fragment-scoped st.rerun cannot be).
                    _c2.button(
                        "Collapse",
                        key=f"{key_prefix}_collapse_{_p}",
                        on_click=_collapse,
                        args=(_p,),
                    )
                st.button(
                    "Collapse all",
                    key=f"{key_prefix}_collapse_all",
                    on_click=lambda: st.session_state.update({_expanded_key: set()}),
                )

    if manually_collapsed:
        st.caption(
            f"{len(manually_collapsed)} branch"
            f"{'es' if len(manually_collapsed) != 1 else ''} manually collapsed "
            '(click a node\'s "+" badge to reveal it again).'
        )
        with st.expander(f"Collapsed branches ({len(manually_collapsed)})", expanded=False):

            def _reveal_manual(path: tuple) -> None:
                st.session_state[_mc_key] = {
                    q for q in st.session_state.get(_mc_key, set()) if q != path
                }

            for _p in sorted(manually_collapsed, key=len):
                _c1, _c2 = st.columns([6, 1])
                _c1.caption(_format_branch_path(_p, cluster_names))
                _c2.button(
                    "Expand",
                    key=f"{key_prefix}_expand_manual_{_p}",
                    on_click=_reveal_manual,
                    args=(_p,),
                )
            st.button(
                "Expand all",
                key=f"{key_prefix}_expand_all_manual",
                on_click=lambda: st.session_state.update({_mc_key: set()}),
            )

    # ── Terminal-leaf sub-clustering: inject synthetic children ──────────────
    # Runs after collapse_manual/collapse_rare_children so it operates on the
    # final visible node set, and before rendering so the injected nodes are
    # indistinguishable from real trie nodes to the layout/coloring/panel code.
    # path -> {episode_idx: [(start, end, sub_cluster_label, color), ...]},
    # so the video panel can splice each episode's own per-window
    # sub-cluster timeline into its last segment when that leaf (not one of
    # its synthetic children) is the one selected -- see
    # _render_selection_panel. Declared here (not just inside the `if`
    # below) and always written to session_state so collapsing every leaf
    # actually clears stale segments instead of leaving the last render's
    # mapping behind.
    window_segments: dict[tuple, dict[int, list[tuple[int, int, str, str]]]] = {}
    if windows is not None and expanded_subclusters:
        root_n_episodes = nodes[0]["n_episodes"] if nodes else 0
        _algo_key = f"{key_prefix}_subcluster_algo"
        _k_key = f"{key_prefix}_subcluster_k"
        _reduction_key = f"{key_prefix}_subcluster_reduction"
        _components_key = f"{key_prefix}_subcluster_components"
        _algo_options = subclusters.available_subcluster_configs() or [
            subclusters.DEFAULT_SUBCLUSTER_ALGORITHM
        ]
        # Seeded from the run's persisted split when it has one, so the first
        # expansion is served from the artifact rather than recomputing a
        # slightly different split at the app's own defaults.
        _seed_algo, _seed_k, _seed_reduction, _seed_components = _subcluster_defaults(
            precomputed_subclusters
        )
        _default_algo = _seed_algo if _seed_algo in _algo_options else _algo_options[0]
        _reduction_options: list[str | None] = [
            None,
            *subclusters.available_subcluster_reductions(),
        ]
        _default_reduction = (
            _seed_reduction
            if _seed_reduction in _reduction_options
            else subclusters.DEFAULT_SUBCLUSTER_REDUCTION
        )
        with st.expander(
            f"Sub-cluster settings ({len(expanded_subclusters)} leaf expanded)"
            if len(expanded_subclusters) == 1
            else f"Sub-cluster settings ({len(expanded_subclusters)} leaves expanded)",
            expanded=False,
        ):
            # Read (never write) session_state before creating these keyed
            # widgets -- Streamlit owns the key once the widget exists, and
            # writing to it first triggers a "value set via the Session
            # State API" warning on every rerun.
            _algo_default = st.session_state.get(_algo_key, _default_algo)
            _algo_idx = _algo_options.index(_algo_default) if _algo_default in _algo_options else 0
            st.selectbox("Algorithm", _algo_options, index=_algo_idx, key=_algo_key)
            _reduction_default = st.session_state.get(_reduction_key, _default_reduction)
            _reduction_idx = (
                _reduction_options.index(_reduction_default)
                if _reduction_default in _reduction_options
                else 0
            )
            st.selectbox(
                "Reduction",
                _reduction_options,
                index=_reduction_idx,
                format_func=lambda value: "none" if value is None else value,
                key=_reduction_key,
            )
            _live_k = int(st.session_state.get(_k_key, _seed_k))
            st.slider(
                "Sub-clusters (k)",
                min_value=2,
                max_value=max(8, _seed_k, _live_k),
                value=_live_k,
                key=_k_key,
            )
            st.number_input(
                "Reduction dimensions",
                min_value=1,
                value=int(st.session_state.get(_components_key, _seed_components)),
                step=1,
                key=_components_key,
                disabled=st.session_state[_reduction_key] is None,
            )
            st.caption("Applies to every currently expanded terminal leaf below.")
            if precomputed_subclusters is not None:
                _live = (
                    st.session_state[_algo_key],
                    int(st.session_state[_k_key]),
                    st.session_state[_reduction_key],
                    int(st.session_state[_components_key]),
                ) != (
                    _seed_algo,
                    _seed_k,
                    _seed_reduction,
                    _seed_components,
                )
                st.caption(
                    "Recomputing live — this run's saved sub-clusters used "
                    f"{_seed_algo}, k={_seed_k}, reduction={_seed_reduction}."
                    if _live
                    else f"Serving this run's saved sub-clusters "
                    f"({len(precomputed_subclusters.groups)} leaves precomputed)."
                )
        _algo_name = st.session_state[_algo_key]
        _algo_k = int(st.session_state[_k_key])
        _reduction_name = st.session_state[_reduction_key]
        _reduction_components = int(st.session_state[_components_key])

        by_path_full = {tuple(nd["path"]): nd for nd in nodes}
        present_paths = {tuple(nd["path"]) for nd in nodes_f}
        used_ids: set[int] = set()
        injected: list[dict] = []
        _sc = _subcluster_colors()
        for _p in sorted(expanded_subclusters):
            leaf = by_path_full.get(_p)
            # A leaf can vanish from nodes_f (min_branch, an ancestor
            # getting manually collapsed, etc.) without its own expand
            # state being cleared -- skip silently rather than injecting
            # an orphaned subtree.
            if leaf is None or _p not in present_paths:
                continue
            groups, window_timeline = _subcluster_groups_for_leaf(
                _p,
                leaf["episode_indices"],
                labels,
                windows,
                subcluster_cache_namespace,
                _algo_name,
                _algo_k,
                _reduction_name,
                _reduction_components,
                precomputed_subclusters,
            )
            leaf_children = _terminal_subcluster_nodes(leaf, groups, root_n_episodes, used_ids)
            injected.extend(leaf_children)
            # sub_label -> (color, name) for EVERY distinct sub_label seen at
            # the window level, not just the ones that won some episode's
            # majority vote (== have their own injected tree node) -- a
            # label can be a real, contiguous run within one episode's
            # segment while never being that episode's own majority, and
            # the whole point of this timeline is to surface exactly that.
            # Naming matches _terminal_subcluster_nodes's formula so a
            # window's label always agrees with its tree node when the two
            # do correspond.
            _all_sub_labels = sorted(
                {sl for entries in window_timeline.values() for _, _, sl in entries}
            )
            sub_label_style = {
                sub_label: (
                    _sc[i % len(_sc)],
                    f"{leaf['label']} (unclustered)"
                    if sub_label == subclusters.UNCLUSTERED
                    else f"{leaf['label']} #{sub_label + 1}",
                )
                for i, sub_label in enumerate(_all_sub_labels)
            }
            leaf_segments: dict[int, list[tuple[int, int, str, str]]] = {}
            for ep, entries in window_timeline.items():
                # Run-length-encode: merge consecutive (already start-sorted)
                # same-sub_label windows into one block. Windows overlap
                # (stride < width), so consecutive entries almost never have
                # end == next start exactly -- merge on name alone, not
                # frame contiguity, and extend to the later window's end.
                blocks: list[tuple[int, int, str, str]] = []
                for start, end, sub_label in entries:
                    color, name = sub_label_style.get(sub_label, (None, str(sub_label)))
                    if blocks and blocks[-1][2] == name:
                        blocks[-1] = (blocks[-1][0], end, name, color)
                    else:
                        blocks.append((start, end, name, color))
                leaf_segments[ep] = blocks
            window_segments[_p] = leaf_segments
        if injected:
            nodes_f = nodes_f + injected
            max_depth_reachable = max(max_depth_reachable, max(nd["depth"] for nd in injected))
    st.session_state[f"{key_prefix}_subcluster_window_segments"] = window_segments

    # Into the slots reserved above, so the tree is the first thing on the page
    # and the path buttons follow it. Both render *after* every notice in code
    # order but *above* them on screen — the graph has to be drawn last because
    # `_render_stats` looks its paths up in the `{key}_path_to_id` map that
    # drawing fills.
    if view_mode == "native_svg":
        with graph_area:
            _render_native_svg(
                nodes_f,
                labels,
                metadata,
                cluster_names=cluster_names,
                node_vlm_labels=node_vlm_labels,
                mp4_dir=mp4_dir,
                mp4_index=mp4_index,
                height=height,
                level=level,
                key_prefix=key_prefix,
                color_mode=color_mode,
                node_values=node_values or {},
                theme=theme,
                edge_style=edge_style,
                edge_width_slope=edge_width_slope,
                node_size_slope=node_size_slope,
                max_depth_hint=max_depth_reachable,
                expanded_branches=expanded_for_render,
                manually_collapsed_branches=manually_collapsed_for_render,
                expanded_subclusters=frozenset(expanded_subclusters),
                subclustering_enabled=windows is not None,
                panel_layout=panel_layout,
                identity=tree_identity,
                show_annotations=show_annotations,
                show_camera_labels=show_camera_labels,
                show_vlm_descriptions=show_vlm_descriptions,
                show_fold_toggles=show_fold_toggles,
                height_extra=height_extra,
                static_figure=static_figure,
                episode_numbers=episode_numbers,
                show_ground_truth=show_ground_truth,
                paper_export=paper_export,
                paper_export_filename=paper_export_filename,
                paper_export_preset=paper_export_preset,
                paper_export_expanded=paper_export_expanded,
                clip_export=clip_export,
                clip_export_filename=clip_export_filename,
            )
        if show_stats:
            with stats_area:
                _render_stats(
                    nodes,
                    nodes_f,
                    cluster_names,
                    key_prefix=key_prefix,
                    interactive=True,
                    show_paths=show_paths,
                )
    else:
        with graph_area:
            _render_plotly(
                nodes_f,
                view_mode,
                color_mode,
                node_values or {},
                height,
                key=f"{key_prefix}_plotly",
            )
        if show_stats:
            with stats_area:
                _render_stats(
                    nodes,
                    nodes_f,
                    cluster_names,
                    key_prefix=key_prefix,
                    interactive=False,
                    show_paths=show_paths,
                )


# ── Native SVG branch ────────────────────────────────────────────────────────


#: Rough advance width of the label font (10px sans-serif) per character, plus
#: the gap two neighbouring labels need between them. Estimated rather than
#: measured because the layout is computed here and the text is rendered in the
#: browser; a few pixels either way only shifts how generously width is shared.
_LABEL_CHAR_PX = 5.4
_LABEL_GAP_PX = 12.0

#: A label wider than this wraps onto another line instead of demanding the room
#: for one long one. Two lines of a name cost 11 vertical pixels, which the gaps
#: between depths have to spare; the horizontal room a one-line name would need
#: is what the tree runs out of. Mirrored by LABEL_MAX_PX in the component —
#: change both together or the layout budgets for a shape it isn't drawing.
_LABEL_MAX_PX = 108.0
_LABEL_MAX_LINES = 2

#: Terminal labels ("FAILURE 12%") stack at any width — outcome above share. See
#: LABEL_WRAP_ALWAYS in the component, which does the same to the drawn text; a
#: budget that assumed one line here would hand every leaf twice the room it uses.
_LABEL_WRAP_ALWAYS = 0.0


def _wrap_label(text: str, max_px: float = _LABEL_MAX_PX) -> list[str]:
    """``text`` split at spaces into at most :data:`_LABEL_MAX_LINES` lines.

    Greedy fill, and the last line takes whatever is left rather than dropping
    it: a name is only useful whole, so an unwrappable one is allowed to be
    wider than ``max_px`` instead of being truncated. ``max_px`` of 0 wraps at
    every space it is allowed to.
    """
    words = text.split()
    if len(words) < 2:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(lines) == _LABEL_MAX_LINES - 1 or len(candidate) * _LABEL_CHAR_PX <= max_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _label_demand(
    nodes: list[dict],
    children_of: dict[tuple, list[tuple]],
) -> dict[tuple, float]:
    """Horizontal room each subtree needs for its labels, in px estimates.

    A leaf needs its own label. A parent needs at least the sum of what its
    children need — so crowding deep in a subtree propagates up to the level
    where horizontal space is actually divided, which is the only place it can
    be fixed.

    A label is measured as wrapped (:func:`_wrap_label`), not as one long line:
    the component wraps it too, and budgeting for the unwrapped width is what
    made a tree of long names need more room than any screen has. Terminal labels
    carry a reach percentage appended in the browser, so their estimate allows
    for the widest one.

    Measured in its drawn, shortened form (:func:`short_node_label`) — budgeting
    for "Behavior 12" and drawing "B12" would leave two thirds of every slot
    empty and spread the tree over width it does not need.
    """
    own: dict[tuple, float] = {}
    for nd in nodes:
        text = short_node_label(str(nd.get("label") or f"B{nd['cluster_id']}"))
        is_terminal = nd["cluster_id"] in TERMINAL_NODE_IDS
        if is_terminal:
            text = f"{text} 100%".strip()  # the widest percentage the player can append
        max_px = _LABEL_WRAP_ALWAYS if is_terminal else _LABEL_MAX_PX
        widest = max(len(line) for line in _wrap_label(text, max_px))
        own[tuple(nd["path"])] = widest * _LABEL_CHAR_PX + _LABEL_GAP_PX

    demand: dict[tuple, float] = {}

    def _resolve(path: tuple) -> float:
        if path in demand:
            return demand[path]
        kids = children_of.get(path, [])
        # Set before recursing: a malformed tree with a cycle would otherwise
        # recurse forever, and a wrong number is better than a hung app.
        demand[path] = own.get(path, _LABEL_GAP_PX)
        if kids:
            demand[path] = max(demand[path], sum(_resolve(k) for k in kids))
        return demand[path]

    for nd in nodes:
        _resolve(tuple(nd["path"]))
    return demand


def _render_native_svg(
    nodes_f: list[dict],
    labels: np.ndarray,
    metadata: list[dict],
    *,
    cluster_names: dict[int, str],
    node_vlm_labels: dict[int, dict] | None = None,
    mp4_dir: Path | None,
    mp4_index: dict | None,
    height: int,
    level: str,
    key_prefix: str,
    color_mode: str = "outcome",
    node_values: dict[int, float] | None = None,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 16.0,
    max_depth_hint: int | None = None,
    expanded_branches: frozenset[tuple] = frozenset(),
    manually_collapsed_branches: frozenset[tuple] = frozenset(),
    expanded_subclusters: frozenset[tuple] = frozenset(),
    subclustering_enabled: bool = False,
    panel_layout: str = DEFAULT_LAYOUT,
    identity: str = "",
    show_annotations: bool = True,
    show_camera_labels: bool = True,
    show_vlm_descriptions: bool = True,
    show_fold_toggles: bool = True,
    height_extra: int = 0,
    static_figure: bool = False,
    episode_numbers: dict[int, int] | None = None,
    show_ground_truth: bool = False,
    paper_export: bool = False,
    paper_export_filename: str = "behavior_graph",
    paper_export_preset: str = "tuned",
    paper_export_expanded: bool = False,
    clip_export: bool = False,
    clip_export_filename: str = "clip",
) -> None:
    node_values = node_values or {}
    from behavior_graphs.viz.streamlit.user_study.graph_explorer import render_graph_full_width

    by_path_all: dict[tuple, dict] = {tuple(nd["path"]): nd for nd in nodes_f}

    # Every branch point currently showing at least one child gets a "-"
    # badge (see fold_state below), independent of the rare-fold/manual-
    # collapse mechanics — this is just "does this path have children right
    # now" over the already-collapsed nodes_f.
    _parent_paths_with_children = {
        tuple(nd["parent_path"]) for nd in nodes_f if nd["parent_path"] is not None
    }

    # Assign synthetic IDs. Each leaf — including each terminal — gets a
    # unique id, deliberately kept independent of its *display* label
    # (`node.label`/`cluster_names`, e.g. "Behavior 0") — the same behavior
    # cluster legitimately reappears at unrelated tree positions (different
    # paths, same label), so the label can never double as an identity key.
    # Color/symbol overrides use this id too, keeping terminals visually
    # distinct and same-cluster nodes the same color across the tree.
    #
    # The id itself is a uuid5 of each node's own path (not a sequential
    # counter), so the *same* logical node always gets the *same* id across
    # renders regardless of what else is currently collapsed or expanded.
    # A sequential counter reassigns ids whenever the visible node set
    # changes shape (any expand/collapse), and the JS component keeps a
    # global, id-keyed drag-position cache (`userOverrides`) that survives
    # across renders — reusing an id for a different logical node made
    # that node "jump" to wherever the id's *previous* owner had last been
    # dragged. uuid5's 128-bit hash makes a collision between two distinct
    # paths astronomically unlikely, but linear probing resolves one
    # deterministically if it ever happens rather than leaving it to chance.
    path_to_id: dict[tuple, int] = {}
    symbol_override: dict[int, str] = {}
    color_override: dict[int, str] = {}
    fold_state: dict[int, str] = {}
    _used_ids: set[int] = set()
    _ID_SPACE = 90_000_000

    def _stable_id(path: tuple) -> int:
        nid = 100_000 + (uuid.uuid5(_NODE_ID_NAMESPACE, str(path)).int % _ID_SPACE)
        while nid in _used_ids:
            nid += 1
        _used_ids.add(nid)
        return nid

    # Helpers for value / outcome color modes.
    def _v_for_cluster(cid: int, n_success: int, n_episodes: int) -> float:
        if cid == SUCCESS_NODE_ID:
            return 1.0
        if cid == FAILURE_NODE_ID:
            return -1.0
        if cid in (END_NODE_ID, START_NODE_ID):
            return 0.0
        return float(node_values.get(int(cid), 0.0))

    _cb = st.session_state.get("colorblind_mode", False)
    _SUCCESS_COL = "#0072B2" if _cb else "#2ca02c"
    _FAILURE_COL = "#D55E00" if _cb else "#d62728"

    def _diverging(t: float) -> str:
        return outcome_color(t)

    # Pre-compute V range for value mode. Exclude SUCCESS / FAILURE /
    # END / START terminals — their hardcoded ±1 values dominate the
    # range and squash every behavioral cluster into the grey middle.
    v_range = 1.0
    if color_mode == "value":
        _terminal_ids = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}
        v_range = (
            max(
                (
                    abs(_v_for_cluster(nd["cluster_id"], nd["n_success"], nd["n_episodes"]))
                    for nd in nodes_f
                    if nd["cluster_id"] not in _terminal_ids
                ),
                default=1.0,
            )
            or 1.0
        )
    for nd in nodes_f:
        p = tuple(nd["path"])
        cid = nd["cluster_id"]
        if cid == START_NODE_ID:
            path_to_id[p] = START_NODE_ID
        else:
            nid = _stable_id(p)
            path_to_id[p] = nid
            # Symbol stays the same regardless of color mode.
            if cid == SUCCESS_NODE_ID:
                symbol_override[nid] = "star"
            elif cid == FAILURE_NODE_ID:
                symbol_override[nid] = "x"
            elif cid == END_NODE_ID:
                symbol_override[nid] = "square"
            # A folded "+N rare variants" bucket: always color by its own
            # outcome mix, regardless of the page's color_mode — an id/value
            # palette color would be meaningless for a synthetic node that
            # doesn't correspond to one real cluster.
            if cid == RARE_COLLAPSE_CLUSTER_ID:
                rate = nd["n_success"] / max(1, nd["n_episodes"])
                color_override[nid] = _diverging(rate)
                fold_state[nid] = "collapsed"
                continue
            if p in manually_collapsed_branches:
                fold_state[nid] = "collapsed"
            elif p in expanded_branches or (
                p in _parent_paths_with_children and cid not in TERMINAL_NODE_IDS
            ):
                fold_state[nid] = "expanded"
            elif (
                subclustering_enabled
                and cid in TERMINAL_NODE_IDS
                and nd["n_episodes"] >= _MIN_SUBCLUSTER_EPISODES
            ):
                # A terminal leaf with enough episodes to sub-cluster
                # meaningfully: badge reveals/hides synthetic children (see
                # expanded_subclusters handling in render_trajectory_tree),
                # independent of the rare-branch/manual-collapse mechanics
                # above. Gated on subclustering_enabled (windows was passed)
                # since without it the expand handler has nothing to reveal.
                fold_state[nid] = "expanded" if p in expanded_subclusters else "collapsed"
            # Color depends on color_mode.
            is_special = cid in (SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID)
            if color_mode == "outcome" and not is_special:
                rate = nd["n_success"] / max(1, nd["n_episodes"])
                color_override[nid] = _diverging(rate)
            elif color_mode == "value" and not is_special:
                v = _v_for_cluster(cid, nd["n_success"], nd["n_episodes"])
                color_override[nid] = _diverging(0.5 + v / (2 * v_range))
            elif cid == SUCCESS_NODE_ID:
                color_override[nid] = _SUCCESS_COL
            elif cid == FAILURE_NODE_ID:
                color_override[nid] = _FAILURE_COL
            elif cid == END_NODE_ID:
                color_override[nid] = "#888888"
            else:
                # id mode
                color_override[nid] = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

    # Aggregate stats per synth id (with unique terminals per leaf, agg is
    # essentially a one-to-one map).
    agg: dict[int, dict] = {}
    for nd in nodes_f:
        p = tuple(nd["path"])
        nid = path_to_id[p]
        if nid not in agg:
            cid = nd["cluster_id"]
            if cid == SUCCESS_NODE_ID:
                name = "SUCCESS"
            elif cid == FAILURE_NODE_ID:
                name = "FAILURE"
            elif cid == END_NODE_ID:
                name = "END"
            elif cid == START_NODE_ID:
                name = "START"
            elif cid == RARE_COLLAPSE_CLUSTER_ID:
                name = nd["label"]
            elif nd.get("is_subcluster"):
                name = nd["label"]
            else:
                name = cluster_names.get(int(cid), f"Behavior {cid}")
            agg[nid] = {
                "name": name,
                "n_episodes": 0,
                "n_success": 0,
                "n_failure": 0,
                "episode_indices": set(),
            }
        agg[nid]["n_episodes"] += nd["n_episodes"]
        agg[nid]["n_success"] += nd["n_success"]
        agg[nid]["n_failure"] += nd["n_failure"]
        agg[nid]["episode_indices"].update(nd.get("episode_indices", []))

    synth_nodes: dict[int, BehaviorNode] = {}
    for nid, info in agg.items():
        synth_nodes[nid] = BehaviorNode(
            cluster_id=nid,
            name=info["name"],
            num_windows=info["n_episodes"],
            num_episodes=len(info["episode_indices"]),
            episode_indices=sorted(info["episode_indices"]),
        )

    # Build transitions
    synth_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for nd in nodes_f:
        p = tuple(nd["path"])
        if nd["parent_path"] is None:
            continue
        parent = tuple(nd["parent_path"])
        if parent not in path_to_id or p not in path_to_id:
            continue
        src = path_to_id[parent]
        tgt = path_to_id[p]
        if src == tgt:
            continue
        synth_counts[src][tgt] += nd["n_episodes"]
    synth_probs: dict[int, dict[int, float]] = {}
    for src, tgts in synth_counts.items():
        total = sum(tgts.values()) or 1
        synth_probs[src] = {t: c / total for t, c in tgts.items()}

    synth_graph = BehaviorGraph(
        nodes=synth_nodes,
        transition_counts={k: dict(v) for k, v in synth_counts.items()},
        transition_probs=synth_probs,
        num_episodes=agg.get(START_NODE_ID, {}).get("n_episodes", 0),
    )

    # Tree layout: split [0,1] x-range among children weighted by subtree
    # size; y = -depth (root at top).
    children_of: dict[tuple, list[tuple]] = defaultdict(list)
    for nd in nodes_f:
        if nd["parent_path"] is not None and tuple(nd["parent_path"]) in by_path_all:
            children_of[tuple(nd["parent_path"])].append(tuple(nd["path"]))
    for k in children_of:
        children_of[k].sort(key=lambda pp: by_path_all[pp]["cluster_id"])

    # How much horizontal room each subtree needs for its labels to sit side by
    # side. A node needs its own label's width; a parent needs whatever its
    # children need together. That bottom-up max is the whole trick: it carries a
    # deep subtree's crowding up to the level where the width is actually divided.
    label_demand = _label_demand(nodes_f, children_of)

    pos_paths: dict[tuple, tuple[float, float]] = {}

    def _assign(path: tuple, x_lo: float, x_hi: float, depth: int) -> None:
        pos_paths[path] = ((x_lo + x_hi) / 2, -depth)
        ch = children_of.get(path, [])
        if not ch:
            return
        # Width is split by what each subtree's labels need. The graph never
        # draws wider than its frame, so this is a fixed pot — whatever one
        # branch gets, a sibling goes without — and splitting it by demand is
        # what keeps each label inside its own slot. Episode count is carried by
        # node radius and edge width instead.
        weights = [label_demand[c] for c in ch]
        wtotal = sum(weights) or 1.0
        cur = x_lo
        for c, w in zip(ch, weights):
            span = (x_hi - x_lo) * w / wtotal
            _assign(c, cur, cur + span, depth + 1)
            cur += span

    _assign((), 0.0, 1.0, 0)

    # Success is the prominent outcome marker in this view, so keep it on its
    # parent's vertical rail rather than letting weighted child slots pull it
    # sideways into a neighboring branch. Failure leaves intentionally retain
    # their normal slots: centering every terminal would push the continuing
    # behavior path sideways at every failure and turn the tree into a stair.
    _successes_by_parent: dict[tuple, list[tuple]] = defaultdict(list)
    for path in pos_paths:
        if path and path[-1] == SUCCESS_NODE_ID:
            _successes_by_parent[path[:-1]].append(path)
    for parent, sibling_paths in _successes_by_parent.items():
        parent_x = pos_paths[parent][0]
        ordered = sorted(sibling_paths)
        step = min(0.12, 0.24 / max(1, len(ordered) - 1))
        for rank, terminal_path in enumerate(ordered):
            offset = (rank - (len(ordered) - 1) / 2) * step
            _x, y = pos_paths[terminal_path]
            pos_paths[terminal_path] = (parent_x + offset, y)

        # A single regular behavior child can otherwise remain in the middle
        # of the parent → success rail after the outcome is centred above.
        # Only adjust this simple, unfurled shape. Fold-summary nodes reserve
        # room for several hidden branches; translating them (or several
        # visible siblings) after layout can make their edges cross.
        behavior_siblings = [
            sibling
            for sibling in children_of.get(parent, [])
            if sibling not in sibling_paths and sibling[-1] not in TERMINAL_NODE_IDS
        ]
        if len(behavior_siblings) != 1 or behavior_siblings[0][-1] == RARE_COLLAPSE_CLUSTER_ID:
            continue

        for sibling in behavior_siblings:
            sibling_x = pos_paths[sibling][0]
            direction = 1.0 if sibling_x >= parent_x else -1.0
            target_x = parent_x + direction * _SUCCESS_RAIL_CLEARANCE
            if direction > 0:
                target_x = max(target_x, sibling_x)
            else:
                target_x = min(target_x, sibling_x)
            delta = target_x - sibling_x
            if delta == 0:
                continue
            for descendant in pos_paths:
                if descendant[: len(sibling)] == sibling:
                    x, y = pos_paths[descendant]
                    pos_paths[descendant] = (x + delta, y)

    # Collapse positions by synth_id (each path is now unique → 1:1 map)
    pos_synth: dict[int, tuple[float, float]] = {}
    for path, (x, y) in pos_paths.items():
        pos_synth[path_to_id[path]] = (x, y)

    # Map [0,1] x → [-2.5, 2.5]; depth → y in SVG coords (smaller = top).
    # Anchored to max_depth_hint (the deepest depth reachable through any
    # combination of fold-expansion, not just what's currently visible) when
    # available, so expanding a bucket into a deeper subtree never rescales
    # the vertical position of every other, already-visible node — only
    # max_d changing would do that, and this keeps it constant across clicks.
    max_d = max_depth_hint if max_depth_hint else max(abs(y) for _, y in pos_synth.values())
    max_d = max_d or 1
    pos_final: dict[int, tuple[float, float]] = {}
    for nid, (x, y) in pos_synth.items():
        pos_final[nid] = (-2.5 + 5.0 * x, -2.5 + 5.0 * ((-y) / max_d))

    # Synthetic per-window labels: each window's assignment is the synth id
    # of the tree node at the corresponding depth of its episode's
    # run-length-collapsed sequence. So clicking a tree node shows videos
    # of episodes that took that specific prefix.
    synth_labels = np.full(len(labels), -1, dtype=np.int64)
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    ep_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        ep = m.get(ep_key, 0)
        ts = m.get("window_start", m.get("timestep", 0))
        ep_groups[ep].append((ts, c, i))
    for ep, lst in ep_groups.items():
        lst.sort()
        phases: list[tuple[int, list[int]]] = []
        for _, c, i in lst:
            if phases and phases[-1][0] == c:
                phases[-1][1].append(i)
            else:
                phases.append((c, [i]))
        running_path: list[int] = []
        for c, idxs in phases:
            running_path.append(c)
            p = tuple(running_path)
            if p in path_to_id:
                nid = path_to_id[p]
                for i in idxs:
                    synth_labels[i] = nid

    # Save mapping so `_render_stats` can convert a clicked path's
    # cluster-id sequence back into synthetic node ids to highlight.
    st.session_state[f"{key_prefix}_path_to_id"] = {tuple(p): nid for p, nid in path_to_id.items()}
    # Reverse map: synth_id → path tuple (cluster IDs), used by terminal-node handler
    st.session_state[f"{key_prefix}_id_to_path"] = {nid: tuple(p) for p, nid in path_to_id.items()}
    st.session_state[f"{key_prefix}_cluster_names"] = dict(cluster_names)
    _highlighted_path = st.session_state.get(f"{key_prefix}_highlighted_path")
    _fps = int((mp4_index or {}).get("fps", 10))

    # Beside the tree by default, so a clicked branch and its clips are on
    # screen together. The trade is horizontal space, which a hierarchy wants
    # more than a flat graph does: a deep, wide tree can need the full content
    # width before its sibling nodes, labels and +/- badges are legible — that
    # is what panel_layout="below" is for.
    # A figure gets no panel and therefore no split: the tree takes the whole
    # width it was given rather than two thirds of it with an empty third.
    graph_slot, panel_slot = (
        (st.container(), None) if static_figure else graph_panel_slots(panel_layout)
    )
    with graph_slot:
        render_graph_full_width(
            graph=synth_graph,
            labels=synth_labels,
            metadata=metadata,
            mp4_dir=mp4_dir if mp4_dir is not None else Path("/tmp/_nonexistent"),
            mp4_index=mp4_index if mp4_index is not None else empty_mp4_index(),
            key_prefix=key_prefix,
            min_edge_prob=0.0,
            # Every tree node has exactly one parent edge, so a nonzero
            # prob_floor here would orphan a node whose only incoming edge
            # falls below it (e.g. a sub-1% branch rendered as a
            # disconnected subtree). No probability floor for the tree.
            prob_floor=0.0,
            pos=pos_final,
            symbol_override=symbol_override,
            color_override=color_override,
            highlighted_path=_highlighted_path,
            theme=theme,
            edge_style=edge_style,
            edge_width_slope=edge_width_slope,
            node_size_slope=node_size_slope,
            # Was hardcoded at 650 inside render_graph_full_width, which left this
            # function's own `height` argument doing nothing at all for the SVG
            # view. It is now the floor, and a deep tree gets more.
            # Added after the fit and its cap, so it is room a caller asked for
            # rather than room the depth happened to need.
            height=_canvas_height(nodes_f, height) + height_extra,
            suppress_video_panel=True,
            # Empty when the badges are off: the component draws one per entry,
            # so withholding the map is the whole switch — no branch is folded
            # differently, there is just nothing to click. The fold *state* is
            # still computed above, because it is what decides which nodes are
            # drawn at all.
            fold_state=fold_state if show_fold_toggles else {},
            show_legend=False,
            # The tree's identity, not one derived from the synthetic graph it
            # is handed: those node ids change every time a branch folds.
            identity=identity,
            static_figure=static_figure,
            # The tree is a graph like any other by the time it gets here, so
            # the figure export needs nothing tree-specific: it is handed the
            # synthetic graph and the tree layout that were just drawn.
            paper_export=paper_export,
            paper_export_filename=paper_export_filename,
            paper_export_preset=paper_export_preset,
            paper_export_expanded=paper_export_expanded,
        )
    if panel_slot is None:
        return
    with panel_slot:
        _render_selection_panel(
            labels=synth_labels,
            metadata=metadata,
            mp4_dir=mp4_dir,
            mp4_index=mp4_index if mp4_index is not None else empty_mp4_index(),
            key_prefix=key_prefix,
            fps=_fps,
            graph=synth_graph,
            node_vlm_labels=node_vlm_labels,
            show_annotations=show_annotations,
            show_camera_labels=show_camera_labels,
            show_vlm_descriptions=show_vlm_descriptions,
            episode_numbers=episode_numbers,
            show_ground_truth=show_ground_truth,
            clip_export=clip_export,
            clip_export_filename=clip_export_filename,
        )


# ── Plotly branches (sunburst / icicle / treemap) ────────────────────────────
# All Plotly figure construction is delegated to behavior_graphs.viz.plotly
# per the repo's Streamlit/plotting separation convention.


def _render_plotly(
    nodes_f: list[dict],
    view_mode: str,
    color_mode: str,
    node_values: dict[int, float],
    height: int,
    key: str,
) -> None:
    from behavior_graphs.viz.plotly.trajectory_tree import (
        create_trajectory_icicle,
        create_trajectory_sunburst,
        create_trajectory_treemap,
    )

    kwargs = dict(color_mode=color_mode, node_values=node_values, height=height)
    if view_mode == "sunburst":
        fig = create_trajectory_sunburst(nodes_f, **kwargs)
    elif view_mode == "icicle":
        fig = create_trajectory_icicle(nodes_f, **kwargs)
    else:  # treemap
        fig = create_trajectory_treemap(nodes_f, **kwargs)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Stats below the chart ────────────────────────────────────────────────────


def _render_stats(
    nodes_all: list[dict],
    nodes_filtered: list[dict],
    cluster_names: dict[int, str],
    *,
    key_prefix: str = "ttree",
    interactive: bool = False,
    show_paths: bool = True,
) -> None:
    """Top success / failure paths summary below the chart.

    With *show_paths* off only the node-count line is drawn: the path list is a
    second way to read the same graph, and a caller may want the graph read on
    its own terms.
    """
    root = nodes_all[0]
    if not show_paths:
        return
    # Source leaves from the FILTERED tree so the list reflects what's
    # actually visible above. Paths pruned by min_branch don't show up.
    leaves = [
        nd
        for nd in nodes_filtered
        if nd["cluster_id"] in (SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID)
    ]
    succ_paths = sorted(
        [nd for nd in leaves if nd["cluster_id"] == SUCCESS_NODE_ID],
        key=lambda nd: -nd["n_episodes"],
    )[:5]
    fail_paths = sorted(
        [nd for nd in leaves if nd["cluster_id"] == FAILURE_NODE_ID],
        key=lambda nd: -nd["n_episodes"],
    )[:5]

    def _format_path(nd):
        # nd["path"] omits the root START node (it lives in a separate
        # entry at path=()), but visually a path from START makes more
        # sense, so prepend it.
        #
        # Names in their drawn short form, so a chain reads as a chain and not as
        # a paragraph: "START → B4 → B11 → ✗" is the same information as
        # "START → Behavior 4 → Behavior 11 → ✗" in a third of the width, and
        # matches what the reader is looking at in the tree.
        disp = ["START"]
        for cid in nd["path"]:
            if cid == SUCCESS_NODE_ID:
                disp.append("✓")
            elif cid == FAILURE_NODE_ID:
                disp.append("✗")
            elif cid == END_NODE_ID:
                disp.append("END")
            else:
                nm = cluster_names.get(int(cid))
                disp.append(short_node_label(nm) if nm else f"B{cid}")
        return " → ".join(disp)

    def _path_steps(nd) -> list[str]:
        """The same steps as :func:`_format_path`, one per element, no START.

        With VLM annotations on, a step's name is a whole sentence, so five of
        them joined by arrows is a paragraph the reader has to parse before
        they can compare two paths. START is dropped rather than given a line
        of its own: every path begins there, so the line carries nothing.
        """
        return [d for d in _format_path(nd).split(" → ") if d != "START"]

    def _button_label(nd) -> str:
        """One step per line, arrow-led after the first, count on top.

        Two trailing spaces before each newline: in a Streamlit button label a
        bare "\n" collapses to a space and list syntax renders literally, so a
        markdown hard break is the only thing that actually breaks the line
        (checked in a browser, not just AppTest).
        """
        steps = _path_steps(nd)
        pct = nd["n_episodes"] / total_eps * 100
        n_eps = nd["n_episodes"]
        lines = [f"**{pct:.0f}%** ({n_eps} episode{'' if n_eps == 1 else 's'})"]
        lines += [steps[0]] if steps else []
        lines += [f"→ {step}" for step in steps[1:]]
        return "  \n".join(lines)

    total_eps = max(1, root["n_episodes"])

    _current = st.session_state.get(f"{key_prefix}_highlighted_path")

    def _maybe_button(label: str, key: str, leaf_path: tuple, nd: dict) -> None:
        """Render a button when interactive, otherwise a markdown bullet.

        Clicking a path highlights it in the graph and reveals its videos in
        the right panel; clicking the active path again deselects it.
        """
        if not interactive:
            # Continuation lines indented so they stay inside the bullet.
            st.markdown("- " + label.replace("  \n", "  \n  "))
            return
        path_to_id = st.session_state.get(f"{key_prefix}_path_to_id", {})
        synth_path: list[int] = []
        start_nid = path_to_id.get(())
        if start_nid is not None:
            synth_path.append(start_nid)
        for i in range(1, len(leaf_path) + 1):
            nid = path_to_id.get(tuple(leaf_path[:i]))
            if nid is not None:
                synth_path.append(nid)
        is_active = _current is not None and list(_current) == synth_path
        if st.button(
            label,
            key=key,
            use_container_width=True,
            type=("primary" if is_active else "secondary"),
        ):
            if is_active:
                st.session_state.pop(f"{key_prefix}_highlighted_path", None)
                st.session_state.pop(f"{key_prefix}_path_ep_list", None)
                st.session_state.pop(f"{key_prefix}_path_label", None)
                st.session_state.pop(f"{key_prefix}_path_leaf_path", None)
            elif synth_path:
                st.session_state[f"{key_prefix}_highlighted_path"] = synth_path
                st.session_state[f"{key_prefix}_path_ep_list"] = sorted(
                    nd.get("episode_indices", [])
                )
                st.session_state[f"{key_prefix}_path_label"] = _format_path(nd)
                # So the video panel can look up this leaf's per-window
                # sub-cluster timeline (see _subcluster_window_segments) the
                # same way a direct node click does -- "Top success/failure
                # paths" buttons select a leaf too, just via a different
                # session-state slot (path_eps) than a graph click
                # (graph_selected).
                st.session_state[f"{key_prefix}_path_leaf_path"] = tuple(nd["path"])
                # Store (synth_node_id, label) pairs for multi-segment annotation.
                # We use synth node IDs (not raw cluster IDs) because the right
                # panel receives synth_labels, not the original labels array.
                _term = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}
                _path_to_id = st.session_state.get(f"{key_prefix}_path_to_id", {})
                _path_synth_ids = []
                for _depth in range(1, len(nd["path"]) + 1):
                    _prefix = nd["path"][:_depth]
                    _cid = _prefix[-1]
                    if _cid in _term:
                        continue
                    _nid = _path_to_id.get(tuple(_prefix))
                    if _nid is not None:
                        _lbl = cluster_names.get(int(_cid), f"B{_cid}")
                        _path_synth_ids.append((_nid, _lbl))
                st.session_state[f"{key_prefix}_path_synth_ids"] = _path_synth_ids
                # Clear any node selection so the right panel shows path videos
                st.session_state.pop(f"{key_prefix}_graph_selected", None)
            st.rerun(scope="fragment")

    # No "(click to highlight)" on either heading: these render as buttons, which
    # already say so, and the hint appeared twice on one row.
    # A button centres its label, which for a step-per-line path means every
    # line starts at a different x — unreadable as a list. Scoped to this
    # block's container so no other button on the page is restyled.
    scope = f"{key_prefix}_paths"
    if interactive:
        st.markdown(
            f"<style>.st-key-{scope} button {{ justify-content: flex-start; }}"
            f".st-key-{scope} button p {{ text-align: left; }}</style>",
            unsafe_allow_html=True,
        )
    with st.container(key=scope):
        c_s, c_f = st.columns(2)
        with c_s:
            st.markdown("**Top success paths**")
            for i, nd in enumerate(succ_paths):
                _maybe_button(
                    _button_label(nd),
                    key=f"{key_prefix}_succ_path_{i}",
                    leaf_path=tuple(nd["path"]),
                    nd=nd,
                )
        with c_f:
            st.markdown("**Top failure paths**")
            for i, nd in enumerate(fail_paths):
                _maybe_button(
                    _button_label(nd),
                    key=f"{key_prefix}_fail_path_{i}",
                    leaf_path=tuple(nd["path"]),
                    nd=nd,
                )


# ── Right-column single-video panel ──────────────────────────────────────────

_SEG_COLORS_DEFAULT = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]
# Okabe-Ito categorical — safe for deuteranopia/protanopia
_SEG_COLORS_CB = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _seg_colors() -> list[str]:
    return _SEG_COLORS_CB if st.session_state.get("colorblind_mode", False) else _SEG_COLORS_DEFAULT


# Sub-cluster segments splice into the same timeline bar as coarse behavior
# segments (see _splice_last_segment), so they need their own hue family --
# reusing _seg_colors by index (as the coarse segments do) means a
# sub-cluster's color frequently lands on the exact same hex as one of the
# coarse blocks right next to it, making a real split look like a single
# color run at a glance even though the legend text is correct.
_SUBCLUSTER_COLORS_DEFAULT = [
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#393b79",
    "#843c39",
    "#7b4173",
]
# Distinct from the Okabe-Ito coarse palette (_SEG_COLORS_CB) while staying
# in the blue/purple/magenta family that's least likely to be confused with
# Okabe-Ito's orange/green/yellow-forward hues under deuteranopia/protanopia.
_SUBCLUSTER_COLORS_CB = [
    "#785EF0",
    "#DC267F",
    "#648FFF",
    "#FE6100",
    "#FFB000",
    "#3DDBD9",
    "#920000",
    "#490092",
]


def _subcluster_colors() -> list[str]:
    return (
        _SUBCLUSTER_COLORS_CB
        if st.session_state.get("colorblind_mode", False)
        else _SUBCLUSTER_COLORS_DEFAULT
    )


def _splice_last_segment(
    coarse: dict[int, list] | None,
    fine: dict[int, list[tuple[int, int, str, str]]],
) -> dict[int, list]:
    """Replace each episode's last (coarse) path segment with its own
    per-window sub-cluster timeline, leaving earlier segments untouched.

    The coarse segments (built from real behavior-cluster labels, e.g.
    "reaching for object" then "Behavior 11") already end with exactly the
    span sub-clustering re-analyzed -- its "last segment" is defined the
    same way (the trailing same-label run before the terminal transition).
    An episode with no fine-grained data (e.g. it had only noise windows)
    keeps its coarse segments as-is.
    """
    out: dict[int, list] = {ep: list(segs) for ep, segs in (coarse or {}).items()}
    for ep, fine_segs in fine.items():
        if not fine_segs:
            continue
        existing = out.get(ep, [])
        out[ep] = existing[:-1] + list(fine_segs) if existing else list(fine_segs)
    return out


def _lookup_window_segments(
    path: tuple, window_segments: dict[tuple, dict[int, list]]
) -> dict[int, list] | None:
    """``window_segments`` is keyed by an expanded leaf's own path -- but a
    leaf's synthetic sub-cluster children are just as likely (more likely,
    even; they're what a click naturally lands on) to be the thing
    selected, and their path is the leaf's path plus their own synthetic
    id. Fall back to the immediate parent path so viewing a single
    sub-cluster's own (already-filtered) episode list still gets each
    episode's per-window timeline, not just the leaf's aggregate view.
    """
    if path in window_segments:
        return window_segments[path]
    if path:
        return window_segments.get(path[:-1])
    return None


def _render_edge_panel_narrow(
    src_id: int,
    tgt_id: int,
    graph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str,
    fps: int,
    show_annotations: bool = True,
    show_camera_labels: bool = True,
    episode_numbers: dict[int, int] | None = None,
    show_ground_truth: bool = False,
    clip_export: bool = False,
    clip_export_filename: str = "clip",
) -> None:
    from behavior_graphs.viz.streamlit.user_study.graph_explorer import (
        _episodes_for_edge,
        _episodes_for_node,
    )

    src_node = graph.nodes.get(src_id)
    tgt_node = graph.nodes.get(tgt_id)
    src_name = src_node.name if src_node else str(src_id)
    tgt_name = tgt_node.name if tgt_node else str(tgt_id)
    prob = graph.transition_probs.get(src_id, {}).get(tgt_id, 0.0)
    st.caption(f"**{src_name} → {tgt_name}** · {prob:.0%}")

    triples = _episodes_for_edge(src_id, tgt_id, labels, metadata, graph=graph)
    if not triples:
        st.info("No episodes found for this transition.")
        return

    _sc = _seg_colors()
    ep_list = [t[0] for t in triples]
    ep_set = set(ep_list)
    _term_cids = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}

    def _is_terminal(nid: int) -> bool:
        gn = graph.nodes.get(nid)
        return gn is not None and gn.cluster_id in _term_cids

    # Build full behavior ranges using _episodes_for_node (same as path/terminal)
    ep_segs: dict[int, list] = {}
    for i, (nid, name, col) in enumerate(
        [
            (src_id, src_name, _sc[0]),
            (tgt_id, tgt_name, _sc[1 % len(_sc)]),
        ]
    ):
        if _is_terminal(nid):
            continue
        for ep_idx, ts_s, ts_e in _episodes_for_node(nid, labels, metadata):
            if ep_idx in ep_set:
                ep_segs.setdefault(ep_idx, []).append((ts_s, ts_e, name, col))

    # Close gaps between adjacent segments
    for ep_idx, segs in ep_segs.items():
        if len(segs) > 1:
            segs.sort(key=lambda s: s[0])
            closed = [
                (segs[j][0], segs[j + 1][0], segs[j][2], segs[j][3]) for j in range(len(segs) - 1)
            ]
            closed.append(segs[-1])
            ep_segs[ep_idx] = closed

    _show_one_video_panel(
        ep_list,
        {},
        mp4_dir,
        mp4_index,
        f"{key_prefix}_edge_{src_id}_{tgt_id}",
        fps,
        ep_segs_by_idx=ep_segs if ep_segs else None,
        show_annotations=show_annotations,
        show_camera_labels=show_camera_labels,
        episode_numbers=episode_numbers,
        show_ground_truth=show_ground_truth,
        clip_export=clip_export,
        clip_export_filename=clip_export_filename,
    )


def _node_caption(cluster_id: int | None, cluster_names: dict[int, str]) -> str:
    """What to call a node with no annotation of its own.

    Terminals never have one — they are outcomes, not behaviors — so they get a
    phrase naming the outcome rather than falling through to a cluster name.
    """
    if cluster_id == SUCCESS_NODE_ID:
        return "Rollouts that succeeded here"
    if cluster_id == FAILURE_NODE_ID:
        return "Rollouts that failed here"
    if cluster_id == END_NODE_ID:
        return "Rollouts that ended without an outcome"
    if cluster_id is None:
        return "Selected node"
    return cluster_names.get(int(cluster_id), f"Behavior {cluster_id}")


def _visit_frequency(n_rollouts: int, metadata: list[dict]) -> str:
    """``"12 of 60 rollouts"`` — the share of the run a selection accounts for."""
    total = len({m.get("rollout_idx", m.get("episode_idx")) for m in metadata})
    return f"{n_rollouts} of {total} rollouts" if total else f"{n_rollouts} rollouts"


@st.fragment
def _render_selection_panel(
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path | None,
    mp4_index: dict,
    key_prefix: str,
    fps: int = 10,
    graph=None,
    node_vlm_labels: dict[int, dict] | None = None,
    show_annotations: bool = True,
    show_camera_labels: bool = True,
    show_vlm_descriptions: bool = True,
    episode_numbers: dict[int, int] | None = None,
    show_ground_truth: bool = False,
    clip_export: bool = False,
    clip_export_filename: str = "clip",
) -> None:
    """One-video-at-a-time selection panel below the trajectory tree."""
    from behavior_graphs.viz.streamlit.user_study.graph_explorer import (
        _episodes_for_node,
    )

    selected_node = st.session_state.get(f"{key_prefix}_graph_selected")
    selected_edge = st.session_state.get(f"{key_prefix}_graph_selected_edge")
    path_eps: list[int] | None = st.session_state.get(f"{key_prefix}_path_ep_list")

    if selected_edge is not None and graph is not None and mp4_dir is not None:
        src_id, tgt_id = selected_edge
        _render_edge_panel_narrow(
            src_id,
            tgt_id,
            graph,
            labels,
            metadata,
            mp4_dir,
            mp4_index,
            key_prefix,
            fps,
            show_annotations=show_annotations,
            show_camera_labels=show_camera_labels,
            episode_numbers=episode_numbers,
            show_ground_truth=show_ground_truth,
            clip_export=clip_export,
            clip_export_filename=clip_export_filename,
        )
    elif selected_node is not None:
        ep_slices = _episodes_for_node(int(selected_node), labels, metadata)
        ep_slices_by_idx: dict[int, tuple[int, int]] = {e[0]: (e[1], e[2]) for e in ep_slices}
        all_eps = sorted(ep_slices_by_idx.keys())
        # Terminal nodes (SUCCESS/FAILURE) are never assigned synth_labels so
        # ep_slices will be empty — fall back to the graph node's episode list.
        ep_segs_by_idx_node: dict[int, list] | None = None
        # Where each episode's player opens when a terminal leaf is selected —
        # see the comment below the segment build.
        seek_by_idx_node: dict[int, int] | None = None
        if not all_eps and graph is not None:
            gnode = graph.nodes.get(int(selected_node))
            if gnode is not None:
                all_eps = sorted(gnode.episode_indices)
                # Build full-path segment annotations (same as path button)
                _id_to_path = st.session_state.get(f"{key_prefix}_id_to_path", {})
                _cnames = st.session_state.get(f"{key_prefix}_cluster_names", {})
                _p2id = st.session_state.get(f"{key_prefix}_path_to_id", {})
                _term_path = _id_to_path.get(int(selected_node), ())
                _term = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}
                _path_synth_ids = []
                for _d in range(1, len(_term_path) + 1):
                    _cid = _term_path[_d - 1]
                    if _cid in _term:
                        continue
                    _nid = _p2id.get(_term_path[:_d])
                    if _nid is not None:
                        _lbl = _cnames.get(int(_cid), f"B{_cid}")
                        _path_synth_ids.append((_nid, _lbl))
                if _path_synth_ids:
                    ep_segs_by_idx_node = {}
                    _ep_set = set(all_eps)
                    _sc = _seg_colors()
                    for _i, (_nid, _lbl) in enumerate(_path_synth_ids):
                        _col = _sc[_i % len(_sc)]
                        for _ep, _ts_s, _ts_e in _episodes_for_node(_nid, labels, metadata):
                            if _ep in _ep_set:
                                ep_segs_by_idx_node.setdefault(_ep, []).append(
                                    (_ts_s, _ts_e, _lbl, _col)
                                )
                    for _ep, _segs in ep_segs_by_idx_node.items():
                        if len(_segs) > 1:
                            _segs.sort(key=lambda s: s[0])
                            _closed = [
                                (_segs[j][0], _segs[j + 1][0], _segs[j][2], _segs[j][3])
                                for j in range(len(_segs) - 1)
                            ]
                            _closed.append(_segs[-1])
                            ep_segs_by_idx_node[_ep] = _closed
                    # A terminal leaf's segments cover the whole path that
                    # reached the outcome, so opening at the earliest of them
                    # (mp4_player's default) starts the clip at the top of the
                    # episode: click "✗ FAILURE" and you watch a minute of a run
                    # going fine before the part you asked about. Open on the
                    # last behavior instead — the one that led into the outcome.
                    # Taken from the coarse segments, before _splice_last_segment
                    # below can replace that block with a leaf's finer
                    # sub-cluster breakdown of the same span.
                    seek_by_idx_node = {
                        _ep: int(_segs[-1][0])
                        for _ep, _segs in ep_segs_by_idx_node.items()
                        if _segs
                    }
        _id_to_path = st.session_state.get(f"{key_prefix}_id_to_path", {})
        _node_path = _id_to_path.get(int(selected_node), ())
        _cluster_id = int(_node_path[-1]) if _node_path else None
        _vlm = (node_vlm_labels or {}).get(_cluster_id) if _cluster_id is not None else None
        if _vlm:
            _annotation = f"{_vlm['label']} (Behavior {_cluster_id})"
            _annotation_detail = vlm_detail(_vlm, show_descriptions=show_vlm_descriptions)
        else:
            # The node's own name, not the branch that leads to it. Spelling the
            # whole prefix out put a paragraph of behavior sentences under the
            # video -- and on a terminal, where there is no annotation to fall
            # back to, that was every step of the branch joined by arrows.
            _annotation = _node_caption(
                _cluster_id, st.session_state.get(f"{key_prefix}_cluster_names", {})
            )
            _annotation_detail = None
        _wseg_map = st.session_state.get(f"{key_prefix}_subcluster_window_segments", {})
        _window_segs = _lookup_window_segments(_node_path, _wseg_map)
        if _window_segs:
            ep_segs_by_idx_node = _splice_last_segment(ep_segs_by_idx_node, _window_segs)
        # The behavior alone, before the frequency is appended for the caption:
        # it is what the export's single band is called, and "…— 12 of 60
        # rollouts" is a fact about the run rather than about the span.
        _band_label = _annotation
        # How much of the run reached this node, on the panel that shows it.
        _annotation = f"{_annotation} — {_visit_frequency(len(all_eps), metadata)}"
        _show_one_video_panel(
            all_eps,
            ep_slices_by_idx,
            mp4_dir,
            mp4_index,
            f"{key_prefix}_node_{selected_node}",
            fps,
            ep_segs_by_idx=ep_segs_by_idx_node,
            seek_by_idx=seek_by_idx_node,
            context_label=_annotation,
            context_detail=_annotation_detail,
            slice_label=_band_label,
            show_annotations=show_annotations,
            show_camera_labels=show_camera_labels,
            episode_numbers=episode_numbers,
            show_ground_truth=show_ground_truth,
            clip_export=clip_export,
            clip_export_filename=clip_export_filename,
        )
    elif path_eps:
        # Build per-episode multi-segment data using synth node IDs (which
        # match the synth_labels array passed as `labels` to this panel).
        path_synth_ids: list = st.session_state.get(f"{key_prefix}_path_synth_ids", [])
        ep_segs_by_idx: dict[int, list] = {}
        path_eps_set = set(path_eps)
        for i, (nid, lbl) in enumerate(path_synth_ids):
            _sc = _seg_colors()
            col = _sc[i % len(_sc)]
            for ep_idx, ts_s, ts_e in _episodes_for_node(nid, labels, metadata):
                if ep_idx in path_eps_set:
                    ep_segs_by_idx.setdefault(ep_idx, []).append((ts_s, ts_e, lbl, col))
        # Close gaps between adjacent segments: extend each segment's end to
        # the start of the next so transition frames are always covered.
        for ep_idx, segs in ep_segs_by_idx.items():
            if len(segs) > 1:
                segs.sort(key=lambda s: s[0])
                closed = []
                for j in range(len(segs) - 1):
                    closed.append((segs[j][0], segs[j + 1][0], segs[j][2], segs[j][3]))
                closed.append(segs[-1])
                ep_segs_by_idx[ep_idx] = closed
        _leaf_path = st.session_state.get(f"{key_prefix}_path_leaf_path")
        # Same reasoning as the terminal-node branch above, and for the same
        # nodes: a "Top failure paths" button selects the leaf its label ends
        # at, so the two gestures land on one outcome and have to open on the
        # same frame. Computed before the sub-cluster splice, as there.
        _seek_by_idx: dict[int, int] | None = None
        if _leaf_path and _leaf_path[-1] in (SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID):
            _seek_by_idx = {
                ep_idx: int(segs[-1][0]) for ep_idx, segs in ep_segs_by_idx.items() if segs
            }
        _wseg_map = st.session_state.get(f"{key_prefix}_subcluster_window_segments", {})
        _window_segs = _lookup_window_segments(_leaf_path or (), _wseg_map)
        if _window_segs:
            ep_segs_by_idx = _splice_last_segment(ep_segs_by_idx, _window_segs)
        _show_one_video_panel(
            path_eps,
            {},
            mp4_dir,
            mp4_index,
            f"{key_prefix}_path",
            fps,
            ep_segs_by_idx=ep_segs_by_idx,
            seek_by_idx=_seek_by_idx,
            # Not the path spelled out: with annotations on, `path_label` is a
            # chain of sentences that has already been read on the button that
            # got here. What the panel can add is how common the branch is.
            context_label=_visit_frequency(len(path_eps), metadata),
            show_annotations=show_annotations,
            show_camera_labels=show_camera_labels,
            episode_numbers=episode_numbers,
            show_ground_truth=show_ground_truth,
            clip_export=clip_export,
            clip_export_filename=clip_export_filename,
        )
    else:
        # Nothing selected is how the page opens, and it needs no notice: the
        # graph above says "click any node to inspect" inside itself, and a line
        # on screen longest should not be an instruction the drawing repeats.
        return


def _strip_tiles(
    entries: Sequence[tuple[int, dict | None]], mp4_dir: Path
) -> list[tuple[Path, dict | None]]:
    """``[(clip, transform)]`` for the strip's tiles — each its own camera's.

    A tile is its own episode's *first* view, which for a DROID dataset is the
    exterior camera and so usually has no transform — but a dataset whose first
    view is the rotated one would get an upside-down strip under an upright
    player. Resolved per tile rather than once for the strip: the views come from
    ``episode_camera_clips``, the same resolution the player above uses (so the
    two can't disagree, a pre-multi-camera entry included), and a collection that
    lost a camera partway through has episodes whose first view is a *different*
    camera — one transform for the lot turned those tiles the wrong way up.
    """
    transforms = dataset_transforms(mp4_dir)
    tiles: list[tuple[Path, dict | None]] = []
    for _, entry in entries:
        if not entry:
            continue
        views = episode_camera_clips(entry, mp4_dir)
        camera = views[0][0] if views else None
        tiles.append((mp4_dir / entry["path"], transforms.get(camera) if camera else None))
    return tiles


def _strip_thumb_urls(
    tiles: Sequence[tuple[Path, dict | None]], key_prefix: str
) -> list[str | None]:
    """Thumbnail URLs for the strip, generating any missing tiles in parallel."""
    # Tiles only, deliberately not prepare_clips: a 180-episode strip needs 180
    # first frames, and building 180 proxies to get them would cost far more
    # than decoding frame 0 straight from each source.
    #
    # The spinner is opened from inside thumb_srcs (which already knows how many
    # tiles are missing) via ExitStack, so the strip's paths are resolved once
    # per rerun rather than once to size the message and again to build.
    with ExitStack() as stack:

        def _pending(n: int) -> None:
            stack.enter_context(
                st.spinner(f"Preparing {n} episode thumbnail{'s' if n > 1 else ''}…")
            )

        return media.thumb_srcs(
            tiles,
            coordinates_prefix=f"strip:{key_prefix}",
            on_pending=_pending,
        )


def _shown(ep_idx: int, episode_numbers: dict[int, int] | None) -> int:
    """The number to print for *ep_idx*: its display number, or itself.

    Falls back to the true index rather than raising on a missing key: an
    episode absent from the map is a clip with an odd caption, while an
    exception here is a blank panel where the evidence should be.
    """
    if not episode_numbers:
        return ep_idx
    return episode_numbers.get(ep_idx, ep_idx)


def _show_one_video_panel(
    ep_list: list[int],
    ep_slices_by_idx: dict[int, tuple[int, int]],
    mp4_dir: Path | None,
    mp4_index: dict,
    key_prefix: str,
    fps: int,
    ep_segs_by_idx: dict[int, list] | None = None,
    seek_by_idx: dict[int, int] | None = None,
    context_label: str | None = None,
    context_detail: str | None = None,
    slice_label: str | None = None,
    show_annotations: bool = True,
    show_camera_labels: bool = True,
    episode_numbers: dict[int, int] | None = None,
    show_ground_truth: bool = False,
    clip_export: bool = False,
    clip_export_filename: str = "clip",
) -> None:
    """Render one episode video with prev / next controls below.

    ``episode_numbers`` maps a true episode index to the number to *show* for
    it. A study that hides the real indices numbers its clips from a
    permutation (:func:`~behavior_graphs.viz.streamlit.user_study.video_browser
    .shuffled_numbers`), and this panel has to use the same map or the two
    displays it is comparing disagree about what a clip is called -- and the
    graph hands back the true, sequential index the video grid was careful not
    to show.

    ``ep_segs_by_idx``, if given, maps an episode index to its own list of
    ``(start, end, label, color)`` blocks rendered as a colored timeline bar
    under the video -- for a terminal leaf with expanded sub-clusters, its
    last block is each episode's own per-window sub-cluster breakdown (see
    ``_splice_last_segment``) rather than one solid block for the whole
    last real behavior segment.

    ``seek_by_idx`` overrides which frame each episode's player opens on, for a
    selection whose highlight starts earlier than the moment it is about — see
    :func:`~behavior_graphs.viz.streamlit.components.mp4_player.start_frame`. It is
    per episode because the segments are.

    ``show_ground_truth`` adds the dataset's human subtask segmentation as a
    strip under the clip, when the dataset has one. **Off by default**, because
    these panels also render inside the user study
    (:mod:`~behavior_graphs.viz.streamlit.user_study.app_group_b`,
    :mod:`~behavior_graphs.viz.streamlit.user_studies.evidence`) and the ground
    truth is the answer key to the segmentation judgement being measured there.
    Only the demo's own pages opt in.

    ``clip_export`` adds the MP4 export panel under the episode controls, and
    ``slice_label`` names the highlighted span in the file it writes — a
    single-behavior selection draws one band, and the band has to say which
    behavior it is. The panel is last on purpose: the strip and the arrows are
    how an episode is *picked*, and a download between the player and its picker
    would read as belonging to neither.
    """
    from behavior_graphs.viz.streamlit.components.camera_views import player_views
    from behavior_graphs.viz.streamlit.components.mp4_player import mp4_player
    from behavior_graphs.viz.streamlit.data import find_mp4_episode as _find_mp4_episode
    from behavior_graphs.viz.streamlit.subtask_gt import episode_ground_truth

    n = len(ep_list)
    if n == 0:
        st.info("No episodes for this selection.")
        return

    vp_key = f"{key_prefix}_vid_page"
    # Reset to 0 when the episode list changes (different path/node selected)
    list_sig = (ep_list[0] if ep_list else -1, n)
    sig_key = f"{key_prefix}_vid_list_sig"
    if st.session_state.get(sig_key) != list_sig:
        st.session_state[sig_key] = list_sig
        st.session_state[vp_key] = 0
    vp = max(0, min(st.session_state.get(vp_key, 0), n - 1))
    ep_idx = ep_list[vp]
    ep_entry = _find_mp4_episode(ep_idx, mp4_index)

    #: What the export panel at the bottom needs, filled in while the player is
    #: rendered so the two cannot disagree about which clip is on screen.
    _export: dict | None = None

    if ep_entry and mp4_dir:
        _views = player_views(ep_entry, mp4_dir, mp4_index)
        prepare_entry_clips([ep_entry], mp4_dir)
        ts_range = ep_slices_by_idx.get(ep_idx)
        segs = (ep_segs_by_idx or {}).get(ep_idx)
        # Extend the last segment to the end of the video so there's no gap.
        if segs:
            _fc = ep_entry.get("frame_count")
            if _fc:
                segs = [*segs[:-1], (segs[-1][0], _fc, segs[-1][2], segs[-1][3])]
        success = ep_entry.get("success")
        status = "✓ Success" if success is True else "✗ Failure" if success is False else ""
        # Extend single-behavior slice_end by a small buffer to cover the
        # last window (max_ts is window_start, actual coverage is +W-1 frames).
        _fc = ep_entry.get("frame_count")
        _slice_end = ts_range[1] if ts_range else None
        if _slice_end is not None and _fc:
            _slice_end = min(_slice_end + fps, _fc)
        # `mp4_dir` is the rollout dir itself (`legacy_mp4_index` returns it),
        # so the annotations travel with the dataset the clip came from and
        # there is nothing extra to plumb through.
        _gt = episode_ground_truth(mp4_dir, ep_idx) if show_ground_truth else None
        mp4_player(
            **_views,
            key=f"{key_prefix}_vid_{ep_idx}",
            max_height_px=220,
            slice_start=ts_range[0] if ts_range else None,
            slice_end=_slice_end,
            total_frames=ep_entry.get("frame_count"),
            fps=fps,
            segments=segs,
            seek_frame=(seek_by_idx or {}).get(ep_idx),
            show_segments=show_annotations,
            show_camera_labels=show_camera_labels,
            gt_segments=(
                [(g.start_sec, g.end_sec, g.label, g.is_task) for g in _gt.segments]
                if _gt
                else None
            ),
            gt_duration_sec=_gt.duration_sec if _gt else None,
        )
        if context_label:
            st.caption(f"**{context_label}**")
        if context_detail:
            st.caption(context_detail)
        st.caption(f"Episode {_shown(ep_idx, episode_numbers)} — {status}")
        if clip_export:
            from behavior_graphs.viz.streamlit.components import clip_export as _clip_export

            # The bands the player just drew, in the mode it drew them in: a
            # chain for a path or an outcome, one span for a single behavior.
            # Nothing at all with the annotations off, where the strip is not on
            # screen to be exported.
            _bands = []
            if show_annotations:
                _bands = _clip_export.bands_from_segments(segs) or _clip_export.slice_band(
                    ts_range[0] if ts_range else None, _slice_end, slice_label
                )
            _export = {
                "views": _views["videos"],
                "bands": _bands,
                "total_frames": ep_entry.get("frame_count"),
                "fps": fps,
                "view_transforms": _views["view_transforms"],
                "visible_cameras": _views["visible_cameras"],
                "key": f"{key_prefix}_clip_{ep_idx}",
                "filename": f"{clip_export_filename}_ep{_shown(ep_idx, episode_numbers)}",
            }
    else:
        st.warning(f"No video found for episode {_shown(ep_idx, episode_numbers)}.")

    # Horizontal thumbnail strip. Tiles are served from Streamlit's /media
    # endpoint and lazy-loaded by the browser: a 180-episode path used to
    # decode 180 full-resolution first frames with imageio and ship ~700 KB
    # of base64 through the component args on every rerun.
    entries = [(ep, _find_mp4_episode(ep, mp4_index)) for ep in ep_list]
    urls: list[str | None] = [None] * len(entries)
    if mp4_dir is not None:
        # Each tile carries its own camera's transform, so the orientation is
        # right even where episodes disagree about which view comes first.
        tiles = _strip_tiles(entries, mp4_dir)
        if tiles:
            it = iter(_strip_thumb_urls(tiles, key_prefix))
            urls = [next(it) if e else None for _, e in entries]
    thumbs = []
    for (ep, ep_e), url in zip(entries, urls):
        icon = (
            "✓"
            if ep_e and ep_e.get("success") is True
            else "✗"
            if ep_e and ep_e.get("success") is False
            else "•"
        )
        thumbs.append({"url": url or "", "label": f"{icon} {_shown(ep, episode_numbers)}"})

    clicked = _THUMB_STRIP(
        thumbs=thumbs,
        selected=vp,
        key=f"{key_prefix}_thumb_strip",
    )
    _seq_key = f"{key_prefix}_thumb_seq"
    if isinstance(clicked, list) and len(clicked) == 2:
        new_vp, seq = int(clicked[0]), int(clicked[1])
        if seq != st.session_state.get(_seq_key):
            st.session_state[_seq_key] = seq
            if new_vp != vp:
                st.session_state[vp_key] = new_vp
                st.rerun(scope="fragment")

    # Both arrows fill their column, so the pair sits at the two ends of the row
    # with the counter centred between them. Left to size themselves, each hugged
    # the left edge of its own column: "→" ended up a third of the way in from
    # the right, reading as misaligned against the "←" it mirrors. The counter is
    # centred by the row rather than by a hand-tuned padding-top, which was a
    # guess at the height of a button.
    c1, c2, c3 = st.columns([1, 4, 1], vertical_alignment="center")
    with c1:
        if st.button("←", key=f"{key_prefix}_prev", disabled=(vp == 0), use_container_width=True):
            st.session_state[vp_key] = vp - 1
            st.rerun(scope="fragment")
    c2.markdown(
        f"<div style='text-align:center;color:#888;font-size:0.82em;'>{vp + 1} / {n}</div>",
        unsafe_allow_html=True,
    )
    with c3:
        if st.button(
            "→", key=f"{key_prefix}_next", disabled=(vp >= n - 1), use_container_width=True
        ):
            st.session_state[vp_key] = vp + 1
            st.rerun(scope="fragment")

    if _export is not None:
        from behavior_graphs.viz.streamlit.components.clip_export import clip_export_panel

        clip_export_panel(**_export)
