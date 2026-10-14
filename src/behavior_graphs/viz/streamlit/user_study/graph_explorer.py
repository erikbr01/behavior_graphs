from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
    BehaviorNode,
)
from behavior_graphs.viz.labels import vlm_detail
from behavior_graphs.viz.streamlit.clips import prepare_episode_clips
from behavior_graphs.viz.streamlit.components.camera_views import player_views
from behavior_graphs.viz.streamlit.components.graph_figure_export import (
    graph_figure_export_panel,
)
from behavior_graphs.viz.streamlit.components.graph_selection import (
    bump_render_token,
    graph_identity,
    sync_graph_identity,
)
from behavior_graphs.viz.streamlit.components.mp4_player import mp4_player
from behavior_graphs.viz.streamlit.components.pager import page_slice
from behavior_graphs.viz.streamlit.components.panel_layout import (
    DEFAULT_LAYOUT,
    empty_state_direction,
    graph_panel_slots,
    is_narrow,
    videos_per_page,
)
from behavior_graphs.viz.streamlit.data import find_mp4_episode
from behavior_graphs.viz.streamlit.episode_index import get_episode_index
from behavior_graphs.viz.streamlit.user_study.graph_plot import (
    component_layout,
    render_graph_component,
)

#: Terminals an episode can *end* at, as opposed to START. These have clips
#: behind them (the episodes that ended there); START does not.
_OUTCOME_TERMINALS = frozenset({SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID})


def _episodes_for_node(
    node_id: int,
    labels: np.ndarray,
    metadata: list[dict],
) -> list[tuple[int, int, int]]:
    return get_episode_index(labels, metadata).episodes_for_node(node_id)


def _episodes_for_terminal(
    node: BehaviorNode,
    labels: np.ndarray,
    metadata: list[dict],
) -> dict[int, int]:
    """``{episode: first frame of the behavior that led into this outcome}``.

    A terminal has no windows of its own, so it used to render no clips at all —
    which left the most interesting nodes on the graph, "these 7 episodes
    failed", with nothing behind them. The episodes are on the node
    (:meth:`BehaviorGraph.from_window_labels` fills SUCCESS/FAILURE from the
    recorded outcomes); what each clip should open on is that episode's last
    behavior run, since the outcome is the frames right after it. Starting at
    frame 0 instead means watching a whole run to see how it ended.
    """
    rle = get_episode_index(labels, metadata).rle
    return {int(ep): int(rle[ep][-1][0]) for ep in node.episode_indices if rle.get(ep)}


# Tableau-style palette matching graph_plot.py's node colors. Indexed by
# (cluster_id % len) on the player side, so 20 colors covers K up to 15.
_CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def _clip_page(items: list, per_page: int, page_key: str) -> list:
    """One page of ``items``, with ←/→ nav above it when there is more than one."""
    shown, page, total_pages = page_slice(items, per_page, page_key)
    start = page * max(1, per_page)

    if total_pages > 1:
        prev_col, label_col, next_col = st.columns([1, 18, 1], vertical_alignment="center")
        with prev_col:
            st.button(
                "←",
                disabled=page == 0,
                key=f"{page_key}_prev",
                use_container_width=True,
                on_click=lambda: st.session_state.update({page_key: page - 1}),
            )
        label_col.markdown(
            f"<div style='text-align:center;color:#888;font-size:0.82em;'>"
            f"Episodes {start + 1}–{start + len(shown)} of {len(items)}</div>",
            unsafe_allow_html=True,
        )
        with next_col:
            st.button(
                "→",
                disabled=page >= total_pages - 1,
                key=f"{page_key}_next",
                use_container_width=True,
                on_click=lambda: st.session_state.update({page_key: page + 1}),
            )
    return shown


def _gt_player_kwargs(mp4_dir: Path, ep_idx: int, show_ground_truth: bool) -> dict:
    """``mp4_player`` kwargs for one episode's ground-truth strip, or ``{}``.

    Empty when the caller did not opt in or the dataset has no annotation, so
    the call site stays a single splat either way. ``mp4_dir`` is the rollout
    dir itself, which is where the annotations live.
    """
    if not show_ground_truth:
        return {}
    from behavior_graphs.viz.streamlit.subtask_gt import episode_ground_truth

    gt = episode_ground_truth(mp4_dir, ep_idx)
    if gt is None:
        return {}
    return {
        "gt_segments": [(g.start_sec, g.end_sec, g.label, g.is_task) for g in gt.segments],
        "gt_duration_sec": gt.duration_sec,
    }


def _per_frame_labels(
    ep_idx: int,
    labels: np.ndarray,
    metadata: list[dict],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct a per-frame cluster label array (-1 = unlabeled).

    Each cluster window covers [window_start, window_end). Overlapping
    windows are resolved by last-write-wins on sorted window_start.

    The orange "slice" bar in mp4_player used to draw [min_ws, max_ws],
    which spans across non-active stretches when a cluster recurs.
    Passing this per-frame mask lets the player draw the true segments.
    """
    return get_episode_index(labels, metadata).per_frame_labels(ep_idx, total_frames)


def _episodes_for_edge(
    src_id: int,
    tgt_id: int,
    labels: np.ndarray,
    metadata: list[dict],
    graph: BehaviorGraph | None = None,
) -> list[tuple[int, int, int | None, int | None]]:
    """Find episodes containing the src -> tgt transition.

    Returns [(ep_idx, ts_src, ts_tgt, ts_tgt_end), ...] sorted by ep_idx.
      • ts_src: window_start of the src behavior
      • ts_tgt: window_start of the tgt behavior (None for X→FAILURE/SUCCESS,
        which extend the source bar to episode end and omit the target bar)
      • ts_tgt_end: window_start of the behavior AFTER tgt (None if tgt is last)
    """
    index = get_episode_index(labels, metadata)
    ep_success = index.success

    result: list[tuple[int, int, int | None, int | None]] = []
    for ep_idx, rle in index.rle.items():
        if not rle:
            continue

        first_ts, first_lab = rle[0]
        last_ts, last_lab = rle[-1]
        success = ep_success.get(ep_idx)

        if src_id == START_NODE_ID:
            if tgt_id == first_lab:
                tgt_end = rle[1][0] if len(rle) > 1 else None
                result.append((ep_idx, first_ts, first_ts, tgt_end))
            continue
        if tgt_id == FAILURE_NODE_ID:
            if last_lab == src_id and success is False:
                result.append((ep_idx, last_ts, None, None))
            continue
        if tgt_id == SUCCESS_NODE_ID:
            if last_lab == src_id and success is True:
                result.append((ep_idx, last_ts, None, None))
            continue

        for i in range(len(rle) - 1):
            if rle[i][1] == src_id and rle[i + 1][1] == tgt_id:
                ts_tgt_end = rle[i + 2][0] if i + 2 < len(rle) else None
                result.append((ep_idx, rle[i][0], rle[i + 1][0], ts_tgt_end))
                break

    # Fallback for synthetic graphs (trajectory tree, etc.) where the
    # special START/SUCCESS/FAILURE checks don't match because terminals
    # have synthetic ids per branch.
    if not result and graph is not None:
        tgt_node = graph.nodes.get(tgt_id) if hasattr(graph, "nodes") else None
        if tgt_node is not None and getattr(tgt_node, "episode_indices", None):
            for ep_idx in tgt_node.episode_indices:
                wins = index.windows.get(ep_idx, ())
                last_ts_local = wins[-1][0] if wins else 0
                result.append((ep_idx, last_ts_local, None, None))

    return sorted(result)


@st.fragment
def _render_edge_panel(
    src_id: int,
    tgt_id: int,
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str,
    panel_layout: str = DEFAULT_LAYOUT,
    show_ground_truth: bool = False,
) -> None:
    src_node = graph.nodes.get(src_id)
    tgt_node = graph.nodes.get(tgt_id)
    src_name = src_node.name if src_node else str(src_id)
    tgt_name = tgt_node.name if tgt_node else str(tgt_id)
    prob = graph.transition_probs.get(src_id, {}).get(tgt_id, 0.0)

    with st.container(border=True):
        header_col, close_col = st.columns([20, 1])
        with header_col:
            st.subheader(f"{src_name}  →  {tgt_name}")
            st.caption(f"Transition probability: {prob:.1%}")
        with close_col:
            if st.button(
                "✕",
                key=f"{key_prefix}_edge_panel_close",
                help="Dismiss",
                use_container_width=True,
            ):
                st.session_state.pop(f"{key_prefix}_graph_selected_edge", None)
                bump_render_token(key_prefix)
                # Dismissing has to repaint the graph above us, so this one
                # escapes the fragment.
                st.rerun(scope="app")

        all_ep_triples = _episodes_for_edge(src_id, tgt_id, labels, metadata, graph=graph)

        if not all_ep_triples:
            st.info("No episodes found for this transition.")
            return

        show_triples = _clip_page(
            all_ep_triples,
            videos_per_page(panel_layout),
            f"{key_prefix}_edge_vid_page_{src_id}_{tgt_id}",
        )

        prepare_episode_clips([t[0] for t in show_triples], mp4_index, mp4_dir)
        vid_cols = st.columns(min(videos_per_page(panel_layout), len(show_triples)))
        for col, (ep_idx, ts_src, ts_tgt, ts_tgt_end) in zip(vid_cols, show_triples):
            ep_entry = find_mp4_episode(ep_idx, mp4_index)
            if ep_entry is None:
                continue
            success = ep_entry.get("success")
            status = "✓" if success is True else "✗" if success is False else ""
            total_frames = ep_entry.get("frame_count")
            # Terminal transitions (X→FAILURE/SUCCESS) have ts_tgt=None.
            # Extend the source bar to episode end and skip the target bar.
            effective_tgt = ts_tgt if ts_tgt is not None else total_frames
            # If tgt is the LAST behavior in this episode, no follow-on
            # behavior gives us a ts_tgt_end — extend bar2 to episode end
            # so the target's full duration is still visible.
            if ts_tgt is not None:
                effective_tgt_end = ts_tgt_end if ts_tgt_end is not None else total_frames
            else:
                effective_tgt_end = None
            with col:
                st.caption(f"Ep {ep_idx} {status}")
                ep_fps = int(round(ep_entry.get("fps") or 10))
                pfl = _per_frame_labels(ep_idx, labels, metadata, total_frames or 0)
                mp4_player(
                    **player_views(ep_entry, mp4_dir, mp4_index),
                    key=f"{key_prefix}_edge_vid_{src_id}_{tgt_id}_{ep_idx}",
                    max_height_px=220,
                    fps=ep_fps,
                    slice_start=ts_src,
                    slice_end=effective_tgt,
                    total_frames=total_frames,
                    slice2_start=ts_tgt if effective_tgt_end is not None else None,
                    slice2_end=effective_tgt_end,
                    bar1_label=src_name,
                    bar2_label=tgt_name if effective_tgt_end is not None else "",
                    per_frame_labels=pfl if len(pfl) else None,
                    cluster_colors=_CLUSTER_COLORS,
                    **_gt_player_kwargs(mp4_dir, ep_idx, show_ground_truth),
                )


def render_graph_full_width(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str = "gex",
    height: int = 650,
    highlighted_path: list[int] | None = None,
    excluded_node_ids: frozenset[int] = frozenset(),
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    pos: dict[int, tuple[float, float]] | None = None,
    symbol_override: dict[int, str] | None = None,
    color_override: dict[int, str] | None = None,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 16.0,
    suppress_video_panel: bool = False,
    #: Show the dataset's human subtask segmentation under each clip. Off by
    #: default: this module's panels render inside the user study, where the
    #: ground truth is the answer key to the judgement being measured.
    show_ground_truth: bool = False,
    prob_floor: float = 0.01,
    rare_edge_max_count: int = 0,
    fold_state: dict[int, str] | None = None,
    node_vlm_labels: dict[int, dict] | None = None,
    show_legend: bool = True,
    panel_layout: str = DEFAULT_LAYOUT,
    identity: str | None = None,
    static_figure: bool = False,
    paper_export: bool = False,
    paper_export_filename: str = "behavior_graph",
    paper_export_preset: str = "tuned",
    paper_export_expanded: bool = False,
) -> None:
    """Clickable behavior graph. Clicking a node or edge opens a details panel.

    ``static_figure`` draws it as a picture instead: no dragging, no clicking, no
    selection panel and no "click any node to inspect" line. For a page showing
    several graphs side by side, where a details panel per graph is not a
    comparison and the way in to one graph is a link to the page built for it.

    ``height`` is the canvas height in pixels. The layout is normalised to fill
    it, so this is really "how much vertical room does one graph level get" — the
    default suits the handful of BFS layers a Markov graph has, while a caller
    drawing something deeper (the trajectory tree) computes its own from the
    depth so labels don't land on the nodes a level below.

    ``panel_layout`` places that panel to the right of the graph (the default:
    graph and clips stay on screen together) or below it (the graph gets the full
    content width) — see
    :mod:`behavior_graphs.viz.streamlit.components.panel_layout`.

    ``paper_export`` adds the panel that offers this graph as a paper-ready PDF
    (:mod:`behavior_graphs.viz.streamlit.components.graph_figure_export`). Off by
    default and switched on per page rather than always: the participant-facing
    study draws its graphs through here too, and a download button under one of
    them is an invitation to do something the study is not asking for.

    ``identity`` names the graph being drawn so a selection made on a *previous*
    graph is dropped rather than greying out the new one (see
    :mod:`behavior_graphs.viz.streamlit.components.graph_selection`). Defaults to
    the labels plus this graph's node ids; a caller whose node set changes
    without the graph changing — the trajectory tree, where folding a branch
    hides nodes — passes its own.
    """

    sync_graph_identity(
        key_prefix,
        identity if identity is not None else graph_identity(labels, metadata, graph.nodes),
    )

    # A caller that suppresses the panel renders its own (the trajectory tree
    # does, into slots it made itself), so splitting again here would hand the
    # graph two thirds of a slot that is already two thirds wide and leave the
    # rest of it empty. Take whatever container we were called in.
    if suppress_video_panel:
        graph_slot = panel_slot = st.container()
    else:
        graph_slot, panel_slot = graph_panel_slots(panel_layout)

    encoding = "Line width and grey level" if edge_style == "lines" else "Arrow thickness"
    with graph_slot:
        clicked_node_id = render_graph_component(
            graph,
            height=height,
            key=f"{key_prefix}_graph",
            highlighted_path=highlighted_path,
            mp4_dir=mp4_dir,
            excluded_node_ids=excluded_node_ids,
            min_edge_prob=min_edge_prob,
            min_edge_count=min_edge_count,
            pos=pos,
            symbol_override=symbol_override,
            color_override=color_override,
            theme=theme,
            edge_style=edge_style,
            edge_width_slope=edge_width_slope,
            node_size_slope=node_size_slope,
            prob_floor=prob_floor,
            rare_edge_max_count=rare_edge_max_count,
            # No fold badges on a figure: they are the one control drawn inside
            # the canvas, and a static component sends no clicks, so leaving
            # them on would draw a "+" that does nothing when pressed.
            fold_state=None if static_figure else fold_state,
            static_figure=static_figure,
        )

        # Under the canvas, where a figure's legend goes — and where it costs the
        # graph nothing. Above it, this row sat between the page header and the
        # picture for the sake of a control most readers open once. The "click a
        # node to explore it" caption it used to share the row with is drawn
        # inside the graph itself (see #instructions in
        # _graph_component/index.html), so as a caption it was the same sentence
        # twice. Content unchanged: marks a reader hasn't seen yet, explained as a
        # paragraph of prose, are read once and in the way ever after — so the
        # legend stays folded until it is asked for.
        # No use_container_width: a button that stretches to a fifth of the page
        # reads as a section of its own rather than as the footnote it is.
        if paper_export:
            # Under the graph and under the legend: it is the last thing anyone
            # does with a graph, and it is the only control here that opens into
            # a panel of its own.
            graph_figure_export_panel(
                graph,
                key=key_prefix,
                pos=component_layout(graph, pos, excluded_node_ids),
                excluded=excluded_node_ids,
                colors=color_override,
                # Without the glyphs the tree's outcome leaves are indistinguishable
                # from behaviors: they are synthetic nodes, and this is what says
                # which of them ended well.
                symbols=symbol_override,
                min_edge_prob=min_edge_prob,
                min_edge_count=min_edge_count,
                prob_floor=prob_floor,
                rare_edge_max_count=rare_edge_max_count,
                arrows=edge_style == "arrows",
                filename=paper_export_filename,
                preset=paper_export_preset,
                expanded=paper_export_expanded,
            )

        if show_legend:
            _, _legend_col = st.columns([6, 1], vertical_alignment="center")
            with _legend_col.popover("Legend"):
                st.markdown(
                    f"""
- **Circle size** — episodes through that behavior
- **{encoding}** — transition probability
- **✓** reached success · **✕** reached failure
- **%** beside a terminal — share of all episodes ending there
- **➕ / ➖** on a node — expand or collapse that branch
"""
                )

    selected_edge = st.session_state.get(f"{key_prefix}_graph_selected_edge")

    if suppress_video_panel or static_figure:
        return

    with panel_slot:
        if clicked_node_id is not None and clicked_node_id in graph.nodes:
            _render_node_panel(
                node_id=clicked_node_id,
                graph=graph,
                labels=labels,
                metadata=metadata,
                mp4_dir=mp4_dir,
                mp4_index=mp4_index,
                key_prefix=key_prefix,
                node_vlm_labels=node_vlm_labels,
                panel_layout=panel_layout,
                show_ground_truth=show_ground_truth,
            )
        elif selected_edge is not None and (
            selected_edge[0] in graph.nodes and selected_edge[1] in graph.nodes
        ):
            _render_edge_panel(
                src_id=selected_edge[0],
                tgt_id=selected_edge[1],
                graph=graph,
                labels=labels,
                metadata=metadata,
                mp4_dir=mp4_dir,
                mp4_index=mp4_index,
                key_prefix=key_prefix,
                panel_layout=panel_layout,
                show_ground_truth=show_ground_truth,
            )
        else:
            # A caption, not st.info: the empty state is the page's resting
            # state, and a full-width blue alert saying "nothing selected"
            # directly under the graph is the loudest thing on screen for the
            # longest time.
            st.caption(
                "Click a node or edge in the graph "
                f"{empty_state_direction(panel_layout)} to explore it."
            )


@st.fragment
def _render_node_panel(
    node_id: int,
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str,
    node_vlm_labels: dict[int, dict] | None = None,
    panel_layout: str = DEFAULT_LAYOUT,
    show_ground_truth: bool = False,
) -> None:
    """Inline bordered panel showing node details + video clips for one node."""
    node = graph.nodes[node_id]

    _vlm = (node_vlm_labels or {}).get(node_id)

    with st.container(border=True):
        header_col, close_col = st.columns([20, 1])
        with header_col:
            if _vlm:
                st.subheader(f"{_vlm['label']} ({node.name})")
            else:
                st.subheader(node.name)
        with close_col:
            if st.button(
                "✕",
                key=f"{key_prefix}_panel_close",
                help="Dismiss",
                use_container_width=True,
            ):
                st.session_state.pop(f"{key_prefix}_graph_selected", None)
                bump_render_token(key_prefix)
                # Dismissing has to repaint the graph above us, so this one
                # escapes the fragment.
                st.rerun(scope="app")

        # ── VLM description above videos ──────────────────────────────────────
        # Not gated the way the trajectory tree's copy is: this panel belongs to
        # the demo, where the model's own account of a node is the point.
        _detail = vlm_detail(_vlm)
        if _detail:
            st.caption(_detail)

        # ── Videos first — they are the primary content ──────────────────────
        # An outcome terminal (✓ / ✗ / END) has no windows of its own but does
        # have the episodes that ended there, each opening on the behavior that
        # led into the outcome — see _episodes_for_terminal. Only START is
        # genuinely nothing to watch.
        is_outcome_terminal = node.cluster_id in _OUTCOME_TERMINALS
        if node.is_special and not is_outcome_terminal:
            st.info("No video clips for the START node.")
        else:
            if is_outcome_terminal:
                _terminal_starts = _episodes_for_terminal(node, labels, metadata)
                # No end frame: the behavior runs to the end of the episode, and
                # only the loop below knows how many frames that is.
                ep_slices_by_idx = {ep: (ts, None) for ep, ts in _terminal_starts.items()}
                all_eps = sorted(_terminal_starts)
            else:
                ep_slices = _episodes_for_node(node_id, labels, metadata)
                ep_slices_by_idx = {e[0]: (e[1], e[2]) for e in ep_slices}
                all_eps = node.episode_indices

            if not all_eps:
                st.info("No videos available for this node.")
            else:
                show_eps = _clip_page(
                    list(all_eps), videos_per_page(panel_layout), f"{key_prefix}_vid_page_{node_id}"
                )

                prepare_episode_clips(show_eps, mp4_index, mp4_dir)
                vid_cols = st.columns(min(videos_per_page(panel_layout), len(show_eps)))
                available = 0
                for col, ep_idx in zip(vid_cols, show_eps):
                    ep_entry = find_mp4_episode(ep_idx, mp4_index)
                    if ep_entry is None:
                        continue
                    available += 1
                    success = ep_entry.get("success")
                    status = (
                        "✓ Success" if success is True else "✗ Failure" if success is False else ""
                    )
                    ts_range = ep_slices_by_idx.get(ep_idx)
                    total_frames = ep_entry.get("frame_count")
                    # A terminal's slice has no end of its own — the behavior
                    # that led into the outcome runs to the end of the episode.
                    effective_end = ts_range[1] if ts_range else None
                    if ts_range and effective_end is None:
                        effective_end = total_frames
                    ep_fps = int(round(ep_entry.get("fps") or 10))
                    # The (min_ws, max_ws) range was painting a single bar
                    # across non-active stretches for clusters that recur
                    # within an episode (kendama-class problem — robomimic
                    # episodes are short enough that recurrence is rare).
                    # Pass per-frame labels so the player draws the true
                    # multi-segment colored strip for THIS node's behavior.
                    pfl = _per_frame_labels(ep_idx, labels, metadata, total_frames or 0)
                    with col:
                        st.caption(f"Episode {ep_idx} — {status}")
                        mp4_player(
                            **player_views(ep_entry, mp4_dir, mp4_index),
                            key=f"{key_prefix}_panel_vid_{node_id}_{ep_idx}",
                            max_height_px=220,
                            fps=ep_fps,
                            slice_start=ts_range[0] if ts_range else None,
                            slice_end=effective_end,
                            total_frames=total_frames,
                            per_frame_labels=pfl if len(pfl) else None,
                            cluster_colors=_CLUSTER_COLORS,
                            **_gt_player_kwargs(mp4_dir, ep_idx, show_ground_truth),
                        )
                if available == 0:
                    st.info("No videos found in index for this node.")

        # ── Stats + outgoing transitions below ────────────────────────────────
        ep_success = get_episode_index(labels, metadata).success

        success_count = sum(1 for eidx in node.episode_indices if ep_success.get(eidx) is True)

        # Beside the graph there is no room to put the metrics and the
        # transition chart side by side — stack them instead.
        if not is_narrow(panel_layout):
            stats_col, trans_col = st.columns([1, 2])
        else:
            stats_col, trans_col = st.container(), st.container()
        with stats_col:
            if is_outcome_terminal:
                # "Timesteps: 0" and "Success rate: 0%" are both true of a
                # FAILURE node and both tell the reader nothing — a terminal has
                # no windows, and its success rate is its own definition.
                total_eps = graph.num_episodes or 0
                m1, m2 = st.columns(2)
                m1.metric("Episodes ending here", node.num_episodes)
                m2.metric(
                    "Share of all episodes",
                    f"{node.num_episodes / total_eps:.0%}" if total_eps else "—",
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Timesteps", node.num_windows)
                m2.metric("Episodes", node.num_episodes)
                m3.metric(
                    "Success rate",
                    f"{success_count / node.num_episodes:.0%}" if node.num_episodes else "—",
                )
                st.caption(
                    "Success rate = fraction of episodes that visit this node "
                    "and ultimately succeed."
                )

        outgoing = graph.transition_probs.get(node_id, {})
        with trans_col:
            if outgoing and not node.is_special:
                tgt_labels = [graph.nodes[t].name if t in graph.nodes else str(t) for t in outgoing]
                tgt_probs = list(outgoing.values())
                bar_colors = [
                    "#2ca02c"
                    if t == SUCCESS_NODE_ID
                    else "#d62728"
                    if t == FAILURE_NODE_ID
                    else "#1f77b4"
                    for t in outgoing
                ]
                fig_trans = go.Figure(
                    go.Bar(x=tgt_probs, y=tgt_labels, orientation="h", marker_color=bar_colors)
                )
                fig_trans.update_layout(
                    title="Where does this node lead?",
                    height=max(100, 32 * len(tgt_labels) + 50),
                    margin=dict(l=0, r=0, t=36, b=0),
                    xaxis=dict(range=[0, 1], title="Probability"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(
                    fig_trans, use_container_width=True, key=f"{key_prefix}_panel_trans"
                )
