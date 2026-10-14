"""Interactive graph-testing playground.

Pick a pipeline run directory, choose a visualization (native-SVG tree
default, plus sunburst / icicle / BFS Markov / temporal Markov), and
interactively explore nodes and edges with the click-to-explore panel.

The graph is the page. Everything that *configures* it lives in the sidebar and
everything that *elaborates* on it (model selection, the inspector, the analysis
report) sits below it, so the picture is the first thing on screen rather than the
seventh — see the layout note above the header strip.

Data access goes through RunArtifacts (labels / windows / graph /
node_values); videos come from the run's rollout dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from behavior_graphs.core.graph_analysis import load_graph_analysis
from behavior_graphs.core.node_labels import load_node_labels
from behavior_graphs.stages.graph import TransitionGraphBuilder
from behavior_graphs.stages.graph.smoothing import in_clustering_space
from behavior_graphs.viz import simplification as gs
from behavior_graphs.viz.labels import figure_filename, number_repeats
from behavior_graphs.viz.streamlit.components.camera_views import render_camera_selector
from behavior_graphs.viz.streamlit.components.graph_analysis_panel import (
    render_graph_analysis_panel,
)
from behavior_graphs.viz.streamlit.components.panel_layout import (
    panel_layout_selector,
    remembered_panel_layout,
)
from behavior_graphs.viz.streamlit.components.selection_panel import (
    compute_best_per_k,
    init_best_defaults,
    render_model_selection,
)
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (
    render_trajectory_tree,
)
from behavior_graphs.viz.streamlit.data import (
    legacy_mp4_index,
    load_run,
    mp4_index_cameras,
    resolve_data_path,
    run_task_name,
    seed_selectors_from_query_params,
    select_run_dir,
    sync_query_params_from_selectors,
    task_display_name,
)
from behavior_graphs.viz.streamlit.episode_index import get_episode_index
from behavior_graphs.viz.streamlit.user_study.graph_explorer import (
    render_graph_full_width,
)
from behavior_graphs.viz.streamlit.user_study.graph_plot import (
    compute_pruned_graph_nodes,
)

_log = logging.getLogger(__name__)

_VIZ_CONFIG_PATH = Path.cwd() / "configs" / "viz" / "graph_demo.yaml"


@st.cache_data(show_spinner=False)
def _load_viz_defaults() -> dict:
    """Interactive-control defaults from configs/viz/graph_demo.yaml — distinct
    from a run's own config.yaml (Hydra pipeline hyperparameters)."""
    return yaml.safe_load(_VIZ_CONFIG_PATH.read_text()) or {}


VIZ = _load_viz_defaults()

st.set_page_config(page_title="Graph Demo — Policy Graphs", layout="wide")

# Page chrome. Colours are all mixes of `currentColor` rather than fixed hexes so
# one rule serves both themes: the light-mode block further down repaints the
# canvas and the text, and these follow whatever the text ended up as.
st.markdown(
    """
    <style>
      /* Streamlit opens with ~6rem of empty canvas above the first element,
         which on this page is 6rem of nothing above the graph. */
      [data-testid="stMainBlockContainer"], .stMainBlockContainer {
        padding-top: 2.1rem;
      }

      /* Header strip: the run on the left, the three numbers that describe it on
         the right, one line. Replaces a page title, a caption and three
         st.metric tiles — a stack that said what the sidebar's nav label and run
         picker already say, and spent ~180px doing it. */
      .pg-head {
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem 1.2rem;
        padding: 0.55rem 0 0.75rem; margin-bottom: 0.9rem;
        border-bottom: 1px solid color-mix(in srgb, currentColor 13%, transparent);
        /* Streamlit clips its markdown containers, and a 1.4rem line in a block
           sized for body text loses its ascenders to the top edge. */
        overflow: visible; line-height: 1.4;
      }
      .pg-head-title {
        font-size: 1.4rem; font-weight: 650; letter-spacing: -0.012em; line-height: 1.4;
      }
      .pg-head-title em {
        font-style: normal; font-size: 0.82rem; font-weight: 400;
        opacity: 0.5; margin-left: 0.5rem;
      }
      .pg-stats { display: flex; gap: 0.4rem; flex-wrap: wrap; }
      .pg-stat {
        display: inline-flex; align-items: baseline; gap: 0.32rem;
        padding: 0.16rem 0.62rem; border-radius: 999px;
        background: color-mix(in srgb, currentColor 6%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
        font-size: 0.76rem; white-space: nowrap; line-height: 1.25;
      }
      .pg-stat b { font-weight: 620; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
      .pg-stat span { opacity: 0.58; }

      /* The graph canvas, as a card. The component draws on a transparent
         background, so without a frame a tree with a sparse top edge just
         floats. Keyed off the component's own `key`, which Streamlit turns
         into an `st-key-…` class on its container.
         A border and nothing else, deliberately: the component sizes an opaque
         backing chip behind every node label to hide the edge running under it,
         and that chip is the canvas colour. Tinting the card even a few percent
         turns each of them into a visible dark box. */
      .st-key-demo_tree_graph, .st-key-demo_markov_graph {
        border: 1px solid color-mix(in srgb, currentColor 11%, transparent);
        border-radius: 14px;
        padding: 0.2rem 0.2rem 0;
        overflow: hidden;
      }

      /* Secondary panels read as one quiet stack under the graph rather than
         three boxes competing with it. */
      [data-testid="stExpander"] details { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar: run picker ─────────────────────────────────────────────────────
_data_path = resolve_data_path()

_best_per_k = compute_best_per_k()
init_best_defaults("demo_run", _best_per_k)
seed_selectors_from_query_params("demo_run")
run_dir = select_run_dir(key="demo_run", best_per_k=_best_per_k)
if run_dir is None:
    st.stop()
sync_query_params_from_selectors("demo_run")
run = load_run(str(run_dir), data_path=_data_path)
labels, meta, bg, emb = run.labels, run.metadata, run.graph, run.coords
_node_vlm_labels = load_node_labels(run_dir)
_graph_analysis = load_graph_analysis(run_dir)
# Videos come from the run's rollout dir (metrics.json["rollout_dir"]).
_MP4_DIR, _MP4_INDEX = legacy_mp4_index(run.rollout_dir)

# Every panel that needs per-episode views of (labels, meta) goes through this
# cached index instead of walking the ~20k-window metadata list itself.
_ep_index = get_episode_index(labels, meta)


@st.cache_data(show_spinner=False)
def _run_config(run_dir_str: str) -> tuple[dict, str]:
    """``(parsed config.yaml, error)`` for a run — ``({}, reason)`` on any failure.

    Never raises, on purpose: the sidebar caption below is a nicety and a run
    whose config can't be read is still perfectly viewable, so this must not be
    able to take the page down. ``load_yaml_file`` catches only OSError and
    YAMLError, which leaves e.g. UnicodeDecodeError, and a config whose top level
    parses to a list would break the ``.get()`` calls below. Returning the reason
    rather than logging it keeps the message the sidebar used to show, without
    the caller having to re-stat the file to find out whether to complain.
    """
    from behavior_graphs.core.yaml_io import load_yaml_file

    path = Path(run_dir_str) / "config.yaml"
    try:
        cfg = load_yaml_file(path, default=None)
    except Exception as e:
        return {}, str(e)
    if cfg is None:
        return {}, "" if not path.exists() else "unreadable or empty"
    return (cfg, "") if isinstance(cfg, dict) else ({}, "top level is not a mapping")


# Show the run's key pipeline parameters in the sidebar.
_cfg, _cfg_err = _run_config(str(run_dir))
if _cfg_err:
    st.sidebar.caption(f"(could not read config.yaml: {_cfg_err})")
_clu_cfg = (_cfg.get("clustering", {}) or {}) if _cfg else {}
_win_cfg = (_cfg.get("windowing", {}) or {}) if _cfg else {}
if _cfg:
    st.sidebar.caption(
        f"clustering: `{str(_clu_cfg.get('_target_', '?')).split('.')[-1]}` "
        f"(k={_clu_cfg.get('n_clusters', '—')}) · "
        f"window: w={_win_cfg.get('width', '—')}, s={_win_cfg.get('stride', '—')}"
    )

# Whether this run's own graph stage already smoothed. If it did, its saved labels
# are smoothed and re-running the decoder here would smooth twice, so the control
# is withdrawn rather than offered on top. The ablation grids sweep this,
# so the picker's "Label smoothing" selector is the way to compare the two.
_graph_cfg = (_cfg.get("graph") or {}) if _cfg else {}
_run_smoothing = str(_graph_cfg.get("smoothing", "none") or "none")
_run_sticky_lambda = _graph_cfg.get("sticky_lambda")


# ── Sidebar: view ───────────────────────────────────────────────────────────
# Which view, and what the colours mean: the two choices that change what is on
# screen, and the only ones a visit normally makes. They used to sit in the
# content column, where — with the folding row, the advanced panel and the
# annotate button under them — they pushed the graph itself off the first screen.

VIZ_OPTIONS = [
    "tree_native_svg",
    "tree_sunburst",
    "tree_icicle",
    "markov_svg_bfs",
    "markov_svg_temporal",
]
VIZ_LABELS = {
    "tree_native_svg": "Trajectory tree",
    "tree_sunburst": "Sunburst",
    "tree_icicle": "Icicle",
    "markov_svg_bfs": "Markov graph — BFS-layered",
    "markov_svg_temporal": "Markov graph — temporal mean",
}

st.sidebar.header("View")
viz_type = st.sidebar.selectbox(
    "Visualization",
    options=VIZ_OPTIONS,
    format_func=lambda v: VIZ_LABELS[v],
    index=VIZ_OPTIONS.index(VIZ["viz"]["visualization"]),
    key="demo_viz_type",
)
is_tree = viz_type.startswith("tree_")

if is_tree:
    color_opts = ["outcome", "id", "value"]
    color_labels = {
        "outcome": "Outcome (success rate)",
        "id": "Cluster ID (palette)",
        "value": "Value V(s) (Bellman)",
    }
else:
    color_opts = ["id", "value", "timesteps"]
    color_labels = {
        "id": "Cluster ID (palette)",
        "value": "Value V(s) (Bellman)",
        "timesteps": "Timestep count (viridis)",
    }
# For the Markov views, the value-based divergent red→green colouring
# is the most informative default (and what the paper figures use).
# For trees we keep the cluster-id palette as the entry point.
_default_color_idx = color_opts.index(
    VIZ["viz"]["color_by_tree"] if is_tree else VIZ["viz"]["color_by_graph"]
)
color_by = st.sidebar.selectbox(
    "Color nodes by",
    options=color_opts,
    format_func=lambda v: color_labels[v],
    index=_default_color_idx,
)

# ── Graph construction / simplification mode ─────────────────────────────────
# Branch folding (tree views only) is the default simplification: it folds
# rare branches into a summary node without ever hiding an observed outcome.
# The lower-level graph-construction modes — "count pruning" (classic
# hide-below-threshold) and "state merging" (ALERGIA-style prefix-tree state
# merging with an exact outcome guard) — are power-user knobs and live in
# Advanced viz settings below.
_RARE_EDGE_MAX_COUNT = VIZ["simplification"]["rare_edge_max_count"]

min_visit_fraction: float | None = None
if is_tree:
    _cap_branching = st.sidebar.checkbox(
        "Fold rare branches",
        value=VIZ["folding"]["enabled"],
        help=(
            "The trajectory tree draws one node per distinct root-to-node path, "
            "so it can stay large even after merging/smoothing simplifies the "
            "underlying states. This folds only branches that are actually rare "
            "in absolute terms — every child accounting for at least this share "
            "of all episodes still gets its own subtree, however many of them "
            "there are, so a branch point with many similarly-sized children "
            "(e.g. the root) doesn't get most of the data mislabeled 'rare' "
            "just for not being in some fixed top few. Nothing is hidden: the "
            "summary node's own click-to-explore panel still lists every "
            "folded episode, unlike count pruning which can disconnect a "
            "whole rare-outcome branch."
        ),
    )
    if _cap_branching:
        min_visit_fraction = (
            st.sidebar.slider(
                "Fold below this share of episodes",
                0.5,
                20.0,
                VIZ["folding"]["min_visit_fraction_pct"],
                step=0.5,
                format="%.1f%%",
            )
            / 100.0
        )


# ── Sidebar: appearance ─────────────────────────────────────────────────────
# Below the run picker and the view, deliberately: which run to look at and how
# to draw it are the decisions a visitor comes here to make, and these are
# toggles most sessions never touch.

st.sidebar.header("Appearance")
light_mode = st.sidebar.toggle(
    "Light mode",
    value=st.session_state.get("light_mode", VIZ["appearance"]["light_mode"]),
    key="light_mode",
)
colorblind_mode = st.sidebar.toggle(
    "Colorblind",
    value=st.session_state.get("colorblind_mode", VIZ["appearance"]["colorblind_mode"]),
    key="colorblind_mode",
)
# Only offered for the views that have a video panel to place: the plotly tree
# views (sunburst, icicle) have none, and a control that visibly does nothing
# is worse than no control.
_VIEWS_WITH_PANEL = ("tree_native_svg", "markov_svg_bfs", "markov_svg_temporal")
# `below` by default here (configs/viz/graph_demo.yaml), where a study session's
# default stays `right`: this page's job is reading the graph, and the trees it
# draws want the whole content width more than they want an empty video panel
# holding a third of it. Set on the demo's own default rather than on
# DEFAULT_LAYOUT so a study whose config omits `video_panel` is unaffected.
_panel_default = VIZ["appearance"]["video_panel"]
# Which camera views the video panels show, for a dataset that has more than
# one. The panels read the selection themselves (it has to be read live inside
# their fragments), so this is called for the widget, not its return value.
with st.sidebar:
    render_camera_selector(mp4_index_cameras(_MP4_INDEX))

    # Keep every video-panel control in the sidebar. This widget used to be
    # created inside the graph fragment with a page-level container, which put
    # the selector above the graph even though it configures the sidebar's
    # selection-video panel.
    panel_layout = (
        panel_layout_selector(
            key="demo_panel_layout",
            default=_panel_default,
        )
        if viz_type in _VIEWS_WITH_PANEL
        # The choice still applies on a view that has no panel to place; the
        # selector remembers it across the runs where it isn't rendered.
        else remembered_panel_layout(key="demo_panel_layout", default=_panel_default)
    )

if light_mode:
    # Streamlit's runtime theme is dark (see [theme] in .streamlit/config.toml);
    # we override the surfaces that need to look right on white. The palette is
    # deliberately muted (warm grays) so input chrome doesn't punch out.
    #
    # Variables (kept here so tuning is in one place):
    #   --text:    body / heading copy
    #   --muted:   captions, helper text
    #   --surface: input/button background
    #   --border:  input/button border
    st.markdown(
        """
        <style>
          :root {
            --pd-text:    #2b2b2b;
            --pd-muted:   #6b6b6b;
            --pd-surface: #f4f4f5;
            --pd-border:  #d4d4d8;
          }
          /* Main canvas + sidebar */
          [data-testid="stAppViewContainer"], [data-testid="stHeader"],
          [data-testid="stSidebar"], section.main, .stApp, .block-container,
          body { background: #ffffff !important; color: var(--pd-text) !important; }
          /* Text + headers in the content area */
          .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
          .stApp p, .stApp label, .stApp .stMarkdown,
          [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--pd-text) !important;
          }
          /* Captions sit slightly dimmer */
          .stApp [data-testid="stCaptionContainer"], .stApp small,
          .stApp [class*="caption"] { color: var(--pd-muted) !important; }
          /* Dividers + chip-style buttons need light borders, not dark */
          hr, [data-testid="stDivider"] { border-color: var(--pd-border) !important; }
          /* Sidebar nav: page links are very faint on white by default. */
          [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span,
          [data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {
            color: var(--pd-text) !important;
          }

          /* ── Inputs ───────────────────────────────────────────────────
             By default Streamlit ships its dark theme through to widgets
             even after we flip the page bg, so dropdowns/buttons end up
             nearly black on the new white canvas. Force them onto the
             muted surface palette. */
          /* Selectbox (BaseWeb Select) */
          .stApp [data-baseweb="select"] > div {
            background-color: var(--pd-surface) !important;
            border-color: var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          .stApp [data-baseweb="select"] svg { fill: var(--pd-muted) !important; }
          /* Native popover for selectbox options */
          [data-baseweb="popover"] [role="listbox"],
          [data-baseweb="popover"] [role="option"] {
            background-color: #ffffff !important;
            color: var(--pd-text) !important;
          }
          [data-baseweb="popover"] [role="option"]:hover {
            background-color: var(--pd-surface) !important;
          }
          /* st.popover's own panel (the Legend beside the graph) is a portal
             outside .stApp and keeps the runtime theme's dark surface. */
          [data-testid="stPopoverBody"], [data-testid="stPopoverBody"] > div {
            background-color: #ffffff !important;
            color: var(--pd-text) !important;
            border-color: var(--pd-border) !important;
          }
          [data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] li,
          [data-testid="stPopoverBody"] strong {
            color: var(--pd-text) !important;
          }
          /* Text + number inputs */
          .stApp input[type="text"], .stApp input[type="number"],
          .stApp textarea {
            background-color: var(--pd-surface) !important;
            border-color: var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          /* Buttons — secondary uses the muted surface, primary keeps
             Streamlit's red accent so the active "highlight path"
             button is visibly differentiated from the inactive ones.
             Matched on the button's own testid rather than on an ancestor
             class: Streamlit wraps any button carrying `help=` in
             stTooltipHoverTarget instead of .stButton, so an
             `.stButton > button` rule skipped precisely the buttons that have
             help text — the annotate button, the SVG export, the graph's
             Legend — and left them dark-on-white. stPopoverButton is the
             Legend's trigger, which is a button of its own kind. */
          .stApp button[data-testid="stBaseButton-secondary"],
          .stApp button[data-testid="stBaseButton-secondaryFormSubmit"],
          .stApp button[data-testid="stPopoverButton"] {
            background-color: var(--pd-surface) !important;
            border: 1px solid var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          .stApp button[data-testid="stBaseButton-secondary"]:hover,
          .stApp button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
          .stApp button[data-testid="stPopoverButton"]:hover {
            background-color: #ebebed !important;
            border-color: #b4b4b8 !important;
          }
          /* Disabled state: lighter surface, dimmer text. */
          .stApp button[data-testid^="stBaseButton-"]:disabled {
            background-color: #fafafa !important;
            color: var(--pd-muted) !important;
            border-color: #e4e4e7 !important;
          }
          .stApp button[data-testid="stBaseButton-primary"] {
            background-color: #ff4b4b !important;
            border: 1px solid #ff4b4b !important;
            color: #ffffff !important;
          }
          .stApp button[data-testid="stBaseButton-primary"]:hover {
            background-color: #ff2b2b !important;
            border-color: #ff2b2b !important;
          }
          /* Sliders: leave the red track, tone down the thumb halo and the
             min/max labels. */
          .stApp [data-baseweb="slider"] [role="slider"] {
            border-color: var(--pd-border) !important;
          }
          /* Expander frame + body. Streamlit's stExpanderDetails container
             keeps the default dark fill even after our top-level bg flip,
             so we override every surface it draws on. */
          .stApp [data-testid="stExpander"],
          .stApp [data-testid="stExpander"] details,
          .stApp [data-testid="stExpander"] details > summary,
          .stApp [data-testid="stExpanderDetails"],
          .stApp [data-testid="stExpanderContent"] {
            background-color: #ffffff !important;
            color: var(--pd-text) !important;
            border-color: var(--pd-border) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
_theme = "light" if light_mode else "dark"


# When the run changes, drop the sub-clustering *settings*: they describe a
# second pass over this run's own windows, so carrying them into another run
# means nothing.
#
# Selection and fold state used to be cleared here too, by a hand-maintained
# list that had drifted (it named `demo_tree_selected`, while the tree's key is
# `demo_tree_graph_selected`). The graph views now do it themselves when they
# notice the new run's labels, in one place — see
# behavior_graphs.viz.streamlit.components.graph_selection.
_prev_run = st.session_state.get("_demo_prev_run")
if _prev_run != str(run_dir):
    st.session_state["_demo_prev_run"] = str(run_dir)
    for suffix in (
        "_subcluster_algo",
        "_subcluster_k",
        "_subcluster_reduction",
        "_subcluster_components",
        "_subcluster_window_segments",
    ):
        st.session_state.pop("demo_tree" + suffix, None)
    # The dataset filter widget uses a task-tag-scoped key (ms_dataset_<tag>),
    # so it auto-resets when the task changes. No explicit pop needed; removing
    # the old unscoped key here was a no-op.


# ── Cached graph builds ──────────────────────────────────────────────────────


def _windows_for(_run, smoothing: str):
    """Windows for a graph build: the clustering space only when it is read.

    Graph construction ignores ``features``; only smoothing measures distance to
    a centroid. Converting when nothing will look at it would let a run whose
    ``reduced`` is stale against ``windows`` fail a path that never needed it.
    """
    return in_clustering_space(_run.windows, _run.reduced) if smoothing != "none" else _run.windows


# max_entries on both: the λ slider has 100 stops, so an unbounded memo
# mints a graph and a label array per stop the user drags through.
@st.cache_resource(show_spinner="Measuring what smoothing would do…", max_entries=16)
def _smoothing_effect(run_dir_str: str, sticky_lambda: float, data_path: str | None):
    """Cached :func:`gs.smoothing_effect` — two tree builds plus a decode."""
    _run = load_run(run_dir_str, data_path=data_path)
    return gs.smoothing_effect(
        _run.labels, _windows_for(_run, "sticky"), _run.metadata, sticky_lambda
    )


@st.cache_resource(show_spinner="Smoothing label flicker…", max_entries=16)
def _smoothed_graph(run_dir_str: str, method: str, sticky_lambda: float, data_path: str | None):
    """``(graph, relabeled labels)`` for smoothing without state merging.

    The pipeline's own graph builder with its smoothing turned on, not a second
    implementation. Smoothing relabels windows, so the graph is rebuilt from the
    new labels and both are returned — that keeps the Markov view,
    click-to-explore, the videos and the inspector on one sequence.
    """
    _run = load_run(run_dir_str, data_path=data_path)
    builder = TransitionGraphBuilder(smoothing=method, sticky_lambda=sticky_lambda)
    graph = builder.build(_run.labels, _windows_for(_run, method))
    return graph, builder.last_labels_


# ── Sidebar: advanced viz settings ───────────────────────────────────────────
# Collapsed, and last — count pruning and state merging are power-user overrides
# of the default branch folding above, and these knobs tune the encoding
# (width ↔ probability, radius ↔ visit count) and export the graph as
# paper-ready SVG.
# on_change="rerun" makes the panel stateful so `.open` is readable: the widgets
# below have to be created on every rerun regardless (their values are read
# further down), but pricing the smoothing control costs ~2s on a 1000-episode
# run, which is not something to spend on a panel nobody has opened.

_default_min_branch = VIZ["simplification"]["min_branch"]
min_branch = _default_min_branch
smoothing_method = "none"
sticky_lambda = VIZ["simplification"]["sticky_lambda"]
graph_mode = VIZ["simplification"]["mode"]
max_depth = VIZ["simplification"]["max_depth_cap"]

_adv_panel = st.sidebar.expander("Advanced viz settings", on_change="rerun")
with _adv_panel:
    _MODE_OPTIONS = [
        "None (folding only)",
        "Count pruning (hide rare edges)",
    ]
    _MODE_KEY_TO_LABEL = {
        "none": _MODE_OPTIONS[0],
        "count": _MODE_OPTIONS[1],
    }
    _mode_label = st.radio(
        "Simplification",
        options=_MODE_OPTIONS,
        index=_MODE_OPTIONS.index(_MODE_KEY_TO_LABEL[VIZ["simplification"]["mode"]]),
        help=(
            "None leaves the graph as-is beyond the branch folding above. Count "
            "pruning hides branches/edges seen fewer than N times — cheap, but rare "
            "success/failure paths disappear."
        ),
    )
    if _mode_label.startswith("Count pruning"):
        graph_mode = "count"
    else:
        graph_mode = "none"

    # Applies in both modes: it runs before any graph is built. Measured in
    # the clustering space, without which it relabels 13.7% of windows and grows
    # the tree instead. Off by default and priced per run — see docs/sweep_study.md.
    _already_smoothed = _run_smoothing != "none"
    _smooth_on = st.checkbox(
        "Smooth label flicker (sticky decode)",
        value=VIZ["simplification"]["smoothing_enabled"] and not _already_smoothed,
        disabled=_already_smoothed,
        help=(
            "Per-window clustering has no temporal awareness, so overlapping "
            "windows can flicker between adjacent clusters — an A→B→A oscillation "
            "that run-length collapse cannot remove. Each one costs the tree a "
            "level of depth and a node per distinct path through it.\n\n"
            "But flicker is not why most trees are big: its median here is 1% of "
            "path steps, while node count tracks how many distinct paths the "
            "episodes take. Folding and a smaller K address that; this does not.\n\n"
            "Outcomes are never affected — an episode's terminal comes from its "
            "recorded success, not its labels."
        ),
    )
    if _already_smoothed:
        _swept = f"`graph.smoothing: {_run_smoothing}`"
        if _run_smoothing == "sticky" and _run_sticky_lambda is not None:
            _swept += f" at λ={_run_sticky_lambda}"
        st.caption(f"This run was swept with {_swept} already.")
    if _smooth_on:
        smoothing_method = "sticky"
        sticky_lambda = st.slider(
            "Smoothing strength (sticky λ)",
            min_value=0.5,
            max_value=50.0,
            value=VIZ["simplification"]["sticky_lambda"],
            step=0.5,
            help=(
                "Transition penalty against each window's distance to a cluster "
                "centroid, in the space the clusters were fitted in. Higher merges "
                "more flicker into longer runs, and far enough merges genuinely "
                "distinct behaviors too (λ=5 takes the median episode from 3.4 "
                "behaviors to 2.1). 1–2 is the safe band. Its strength shifts with "
                "K and cluster spread, so watch the line below."
            ),
        )

    # What it actually does to *this* run, both ways round, so the choice is not a
    # guess. Only while the panel is open — see the on_change note above.
    if _adv_panel.open:
        _sm_raw, _sm_new = _smoothing_effect(str(run_dir), float(sticky_lambda), _data_path)
        _sm_verdict = (
            "smaller"
            if _sm_new[0] < _sm_raw[0]
            else "larger"
            if _sm_new[0] > _sm_raw[0]
            else "the same"
        )
        st.caption(
            f"On this run, λ={sticky_lambda:g} makes the trajectory tree **{_sm_verdict}**: "
            f"{_sm_raw[0]} nodes at depth {_sm_raw[1]} → {_sm_new[0]} at depth {_sm_new[1]}."
        )

    if graph_mode == "count":
        min_branch = st.slider(
            "Hide transitions where count(s, s′) < N",
            1,
            max(50, _default_min_branch + 10),
            _default_min_branch,
            help=(
                "count(s, s′) is the number of rollouts in which the transition s → s′ "
                "was observed. Edges with fewer than N observations are hidden, and "
                "any nodes that become unreachable from START are pruned as a "
                "consequence."
            ),
        )

    st.divider()
    _EDGE_STYLE_OPTIONS = ["arrows", "lines"]
    edge_style = st.radio(
        "Edges",
        options=_EDGE_STYLE_OPTIONS,
        index=_EDGE_STYLE_OPTIONS.index(VIZ["encoding"]["edge_style"]),
        horizontal=True,
        help=(
            "Default is ‘lines’: width + grey level already encode "
            "transition probability cleanly. Switch to ‘arrows’ if "
            "you need explicit direction-of-motion."
        ),
    )
    edge_width_slope = st.slider(
        "Edge width per probability (px)",
        min_value=0.0,
        max_value=12.0,
        value=VIZ["encoding"]["edge_width_slope"],
        step=0.5,
        help=(
            "Linear: width = 0.6 + slope × p. The slope is the extra "
            "pixels added to the stroke at p = 1. Grey opacity also "
            "tracks p linearly from 0.2 to 1.0."
        ),
    )
    node_size_slope = st.slider(
        "Node radius per visit fraction (px)",
        min_value=0.0,
        max_value=36.0,
        value=VIZ["encoding"]["node_size_slope"],
        step=1.0,
        help=(
            "Linear: radius = 6 + slope × (visits / max-visits). The "
            "slope is the extra pixels added to the radius at the "
            "most-visited node."
        ),
    )

    # SVG export. The iframe auto-publishes a fresh snapshot to Python on
    # every render (deduped by content hash), so this download_button is
    # always armed with the latest SVG — one click, native browser
    # download. (anchor.click() driven from inside the iframe isn't a real
    # user gesture, so Chrome drops it; the parent-window button works.)
    _captured = st.session_state.get("captured_svg", "")
    st.download_button(
        "⬇ Export SVG",
        data=(_captured or "").encode("utf-8"),
        file_name="behavior_graph.svg",
        mime="image/svg+xml",
        disabled=not _captured,
        help="Saves the current graph state (no background) as behavior_graph.svg.",
        use_container_width=True,
    )

# Swap in the smoothed graph (and its per-window relabeling, so
# click-to-explore, videos, and the inspector all resolve the same node ids)
# before anything downstream touches `bg` / `labels`.
_n_resmoothed = 0
if smoothing_method != "none":
    bg, _smoothed_labels = _smoothed_graph(
        str(run_dir), smoothing_method, float(sticky_lambda), _data_path
    )
    _n_resmoothed = int((_smoothed_labels != labels).sum())
    labels = _smoothed_labels

# Cache token identifying (bg, labels): the run dir plus the simplification
# settings, since smoothing relabels the windows.
# Deliberately NOT including min_branch — it doesn't change the graph,
# and _pruned_graph already takes it as its own cache-key argument, so folding
# it in here would only throw away the temporal layout on every slider move.
_graph_token = f"{run_dir}|{graph_mode}|{smoothing_method}|{sticky_lambda}"


# max_entries: min_branch is a slider, so an unbounded memo mints an entry per
# stop the user drags through and never evicts on this always-on deployment.
@st.cache_resource(show_spinner=False, max_entries=32)
def _pruned_graph(_bg, graph_token: str, min_branch: int) -> tuple[frozenset[int], int]:
    """(nodes to exclude, edges hidden) for the current edge-count threshold.

    Cached: compute_pruned_graph_nodes walks the transition table to a
    reachability fixed point, and both the filter-summary caption and the
    dispatch below need the same answer on every rerun. ``_bg`` is excluded
    from the cache key (leading underscore); *graph_token* identifies it.
    """
    excluded = compute_pruned_graph_nodes(
        _bg,
        min_visit_prob=0.0,
        n_total=_bg.num_episodes,
        min_edge_count=min_branch,
    )
    hidden = sum(
        1
        for src, targets in _bg.transition_counts.items()
        for tgt, cnt in targets.items()
        if src in excluded or tgt in excluded or cnt < min_branch
    )
    return excluded, hidden


@st.cache_resource(show_spinner=False, max_entries=8)
def _temporal_layout(_bg, _labels, _meta, graph_token: str):
    """Cached ``gs.temporal_layout`` — seconds on a large graph, and the layout
    only changes when the graph does."""
    return gs.temporal_layout(_bg, _labels, _meta, level="rollout")


# Node values for the value coloring: prefer the run's saved MRP values,
# fall back to solving the Bellman system on the loaded graph. Saved values
# are keyed by the original cluster ids.
node_values: dict[int, float] = {}
if color_by == "value":
    # Saved values are keyed by the graph as swept, so smoothing's relabel has
    # to solve its own Bellman system.
    if run.node_values and smoothing_method == "none":
        node_values = dict(run.node_values)
    else:
        try:
            node_values = bg.compute_values()
        except Exception as e:
            st.warning(f"compute_values() failed ({e}); falling back to ID coloring.")
            color_by = "id"


# ── Header strip ─────────────────────────────────────────────────────────────
# One row: which run, and the three numbers that describe it. "Configs swept"
# describes the sweep this run came out of, not the behavior on screen, and it is
# one click away in the model-selection panel below. Everything else that used to
# stand between the top of the page and the graph is now in the sidebar, so the
# picture starts a single row down.
_n_rollouts = len(_ep_index.success)
_n_success = sum(1 for v in _ep_index.success.values() if v is True)
_success_rate = (_n_success / _n_rollouts) if _n_rollouts else 0.0
_k_clusters = _clu_cfg.get("n_clusters")

_stats = [
    ("Rollouts", f"{_n_rollouts}"),
    ("Success", f"{_success_rate:.0%}" if _n_rollouts else "—"),
]
st.markdown(
    "<div class='pg-head'>"
    f"<div class='pg-head-title'>{task_display_name(run_task_name(run_dir))}</div>"
    "<div class='pg-stats'>"
    + "".join(f"<div class='pg-stat'><b>{v}</b><span>{k}</span></div>" for k, v in _stats)
    + "</div></div>",
    unsafe_allow_html=True,
)

# ── What the current simplification shows / drops ────────────────────────────
# One line, and only when there is something to say. Above the graph rather than
# below it because it qualifies what the reader is about to look at — smoothing
# changes which windows, and so which clips, belong to each node.
_notes: list[str] = []
if smoothing_method != "none":
    _share = f" ({_n_resmoothed / len(labels):.0%} of windows relabeled)" if _n_resmoothed else ""
    _notes.append(
        f"Flicker smoothing on (sticky λ={sticky_lambda:g}){_share}; "
        "every episode keeps its recorded outcome"
    )
if not is_tree:
    # Edge-count threshold is meaningful for the Markov views; the tree view
    # uses min_branch to filter nodes directly and renders its own pruning
    # summary inside render_trajectory_tree.
    _excluded, _n_edges_hidden = _pruned_graph(bg, _graph_token, int(min_branch))
    # Only when something actually went missing. At the default threshold nothing
    # does, so this line spent a row of the page saying "0 edges hidden, 0 nodes
    # pruned" — a caption whose only content is that it has nothing to report.
    if _n_edges_hidden or _excluded:
        _notes.append(f"{_n_edges_hidden} edges hidden, {len(_excluded)} nodes pruned")
if _notes:
    st.caption(" · ".join(_notes) + ".")


# ── Dispatch ─────────────────────────────────────────────────────────────────
# Wrapped in a fragment so clicking a node/edge/badge only re-executes this
# block, not the whole page script — the sidebar, header metrics, model
# selection panel, and every other widget above stay untouched. Every
# st.rerun() inside the graph-rendering call chain (render_trajectory_tree,
# render_graph_component, graph_explorer's video panels) uses
# scope="fragment" to match; without that they'd still force a full-page
# rerun despite living inside this fragment.


@st.fragment
def _render_graph_dispatch() -> None:
    # VLM labels are keyed by cluster id, which is what the node ids are.
    # no-op identity lookup.
    _dispatch_vlm_labels = (
        {nid: _node_vlm_labels[nid] for nid in bg.nodes if nid in _node_vlm_labels}
        if _node_vlm_labels
        else {}
    )
    if is_tree:
        # Prefer the VLM label, falling back to bg.nodes[nid].name for nodes
        # with no VLM label yet.
        # Numbered where a name repeats: the labeling model describes two
        # clusters in the same words often enough (set_table has two "reaching
        # for the red can"), and drawn that way the tree says one behavior
        # appears twice when in fact two distinct clusters do.
        _tree_cluster_names = number_repeats(
            {
                nid: (_node_vlm_labels[nid]["label"] if nid in _node_vlm_labels else node.name)
                for nid, node in bg.nodes.items()
            }
        )
        render_trajectory_tree(
            labels=labels,
            metadata=meta,
            view_mode=viz_type.replace("tree_", ""),
            # The demo is the one surface that may show the answer key; the
            # study renders these same panels and must not.
            show_ground_truth=True,
            min_branch=int(min_branch),
            max_depth_cap=int(max_depth),
            color_mode=color_by,
            node_values=node_values,
            cluster_names=_tree_cluster_names,
            node_vlm_labels=_dispatch_vlm_labels or None,
            mp4_dir=_MP4_DIR,
            mp4_index=_MP4_INDEX,
            height=600,
            level="rollout",
            key_prefix="demo_tree",
            paper_export_filename=figure_filename(run.rollout_dir, run_dir.name, "tree"),
            theme=_theme,
            edge_style=edge_style,
            edge_width_slope=float(edge_width_slope),
            node_size_slope=float(node_size_slope),
            min_visit_fraction=min_visit_fraction,
            windows=run.windows,
            precomputed_subclusters=run.subclusters,
            subcluster_cache_namespace=str(run.run_dir.resolve()),
            panel_layout=panel_layout,
            paper_export=True,
            # The clips are exportable here and on Figures, nowhere else: this
            # is the page for deciding what a figure should say, so the video
            # half of one belongs beside the graph half. See
            # `viz.streamlit.components.clip_export`.
            clip_export=True,
            clip_export_filename=figure_filename(run.rollout_dir, run_dir.name, "clip"),
        )
    else:
        # Markov SVG (BFS or temporal layout)
        excluded, _ = _pruned_graph(bg, _graph_token, int(min_branch))
        pos = None
        if viz_type == "markov_svg_temporal":
            try:
                pos = _temporal_layout(bg, labels, meta, _graph_token)
            except Exception as e:
                st.warning(f"Temporal layout failed ({e}); falling back to BFS-layered.")
        color_override: dict[int, str] | None = None
        _SPECIAL = {SUCCESS_NODE_ID, FAILURE_NODE_ID, START_NODE_ID, END_NODE_ID}
        _OUTCOME_BINS = (
            # Okabe-Ito diverging: vermilion → orange → yellow → teal → blue
            ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#0072B2"]
            if colorblind_mode
            else ["#d62728", "#ff7f0e", "#e8c32a", "#9dc95d", "#2ca02c"]
        )

        def _div(t: float) -> str:
            return _OUTCOME_BINS[min(4, int(max(0.0, min(1.0, t)) * 5))]

        _SUCCESS_COL = "#0072B2" if colorblind_mode else "#2ca02c"
        _FAILURE_COL = "#D55E00" if colorblind_mode else "#d62728"
        # Always override terminal node colors so colorblind mode applies everywhere
        color_override = {
            nid: (_SUCCESS_COL if node.cluster_id == SUCCESS_NODE_ID else _FAILURE_COL)
            for nid, node in bg.nodes.items()
            if node.cluster_id in (SUCCESS_NODE_ID, FAILURE_NODE_ID)
        }
        if color_by == "value" and node_values:
            non_term = [v for cid, v in node_values.items() if cid not in _SPECIAL]
            vr = max((abs(v) for v in non_term), default=1.0) or 1.0
            color_override = {
                nid: _div(0.5 + node_values.get(nid, 0.0) / (2 * vr))
                for nid in bg.nodes
                if nid not in _SPECIAL
            }
        elif color_by == "timesteps":
            try:
                import plotly.express as _px

                viridis = _px.colors.sequential.Viridis
            except Exception:
                viridis = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
            ts_max = (
                max(
                    (
                        np.log1p(bg.nodes[nid].num_windows)
                        for nid in bg.nodes
                        if nid not in _SPECIAL
                    ),
                    default=1.0,
                )
                or 1.0
            )
            color_override = {}
            for nid in bg.nodes:
                if nid in _SPECIAL:
                    continue
                t = np.log1p(bg.nodes[nid].num_windows) / ts_max
                idx = int(min(len(viridis) - 1, max(0, t * (len(viridis) - 1))))
                color_override[nid] = viridis[idx]

        render_graph_full_width(
            graph=bg,
            labels=labels,
            metadata=meta,
            # As in the tree call above: the demo may show the answer key, the
            # study may not.
            show_ground_truth=True,
            mp4_dir=_MP4_DIR if _MP4_DIR is not None else Path("/tmp/_nonexistent"),
            mp4_index=_MP4_INDEX,
            key_prefix="demo_markov",
            paper_export_filename=figure_filename(run.rollout_dir, run_dir.name, "markov"),
            min_edge_count=int(min_branch),
            pos=pos,
            excluded_node_ids=excluded,
            color_override=color_override,
            theme=_theme,
            edge_style=edge_style,
            edge_width_slope=float(edge_width_slope),
            node_size_slope=float(node_size_slope),
            node_vlm_labels=_dispatch_vlm_labels or None,
            prob_floor=0.01,
            rare_edge_max_count=0,
            panel_layout=panel_layout,
            paper_export=True,
        )


_render_graph_dispatch()

# ── Model selection ──────────────────────────────────────────────────────────
# Below the graph, not above it: which of 200+ swept configs to open is a
# decision made once, and it used to be the last thing between the page header
# and the picture.
#
# Indexing 200+ configs, scoring, and rendering the Pareto/sensitivity tables is
# expensive (~1-2s). on_change="rerun" makes the expander stateful so .open is
# readable here, and the body is skipped entirely while it is collapsed instead
# of running on every unrelated rerun (every graph/tree click on this page).
_ms_panel = st.expander(
    "📊 Model selection — hyperparameters × eval scores, Pareto, best per K",
    on_change="rerun",
)
if _ms_panel.open:
    with _ms_panel:
        render_model_selection(current_run_path=str(run_dir), sidebar_key="demo_run")

# ── Node / Transition Inspector ───────────────────────────────────────────────
import plotly.graph_objects as go  # noqa: E402

_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Per-episode stats come from the shared cached index. Smoothing relabels
# `labels` above, so this may be a second index keyed on the new labels --
# get_episode_index caches by content signature, so both stay warm.
_insp_index = get_episode_index(labels, meta)


# Non-special behavior nodes available for comparison (selected via the
# dropdown only — click-to-add was unreliable and has been removed).
# is_special, not a raw-id check against _SPECIAL_IDS: a split graph's
# per-leaf terminals and sub-cluster children carry synthetic ids, so an id
# check would list them as pseudo-behaviors (with empty stats — their ids
# never appear in the window labels).
_behavior_nodes = {nid: node.name for nid, node in sorted(bg.nodes.items()) if not node.is_special}
_node_opts = list(_behavior_nodes.keys())
# Split context copies (state-merging mode) inherit their origin cluster's
# VLM label; the node name already carries the "#k" copy suffix.
_node_labels = {
    nid: (f"{_node_vlm_labels[nid]['label']} ({name})" if nid in _node_vlm_labels else name)
    for nid, name in _behavior_nodes.items()
}
_edge_opts = [
    (src, tgt)
    for src in sorted(bg.transition_counts)
    for tgt in sorted(bg.transition_counts[src])
    if not bg.nodes[src].is_special or not bg.nodes[tgt].is_special
]
_edge_labels = {
    (
        src,
        tgt,
    ): f"{bg.nodes[src].name if src in bg.nodes else src} → {bg.nodes[tgt].name if tgt in bg.nodes else tgt}"
    for src, tgt in _edge_opts
}

# Inspector controls (radio + node multiselect) live inside the expander
# so the page header isn't cluttered when the panel is collapsed.
_active_nodes: list[int] = []
_is_edge_mode = False

with st.expander("🔍 Node / Transition Inspector", expanded=False):
    _INSPECTOR_OPTIONS = ["Nodes", "Transitions"]
    _insp_mode = st.radio(
        "Inspect",
        options=_INSPECTOR_OPTIONS,
        index=_INSPECTOR_OPTIONS.index(VIZ["inspector"]["mode"]),
        horizontal=True,
        key="dbg_insp_mode",
    )
    _is_edge_mode = _insp_mode == "Transitions"

    if not _is_edge_mode:
        _cmp_nodes = st.multiselect(
            "Compare nodes",
            options=_node_opts,
            format_func=lambda n: _node_labels.get(n, str(n)),
            default=[],
            key="dbg_cmp_nodes",
        )
        _active_nodes = list(dict.fromkeys(_cmp_nodes))

    if _is_edge_mode:
        # ── Edge mode ────────────────────────────────────────────────────────
        _cmp_edges = st.multiselect(
            "Compare transitions",
            options=_edge_opts,
            format_func=lambda e: _edge_labels.get(e, str(e)),
            default=[],
            key="dbg_cmp_edges",
        )
        _active_edges: list[tuple[int, int]] = list(dict.fromkeys(_cmp_edges))
        if not _active_edges:
            st.info("Pick one or more transitions from the dropdown above.")
            st.stop()

        _edge_data: dict[str, list[dict]] = {}
        for _e in _active_edges:
            _lbl = _edge_labels.get(_e, str(_e))
            _src_c, _tgt_c = _e
            if _src_c is None or _tgt_c is None:
                continue
            _edge_data[_lbl] = _insp_index.edge_stats(_src_c, _tgt_c)

        if not _edge_data:
            st.info("No data found for selected transitions.")
        else:
            _dc1, _dc2 = st.columns(2)
            with _dc1:
                _fig_pos = go.Figure()
                for _lbl, _rows in _edge_data.items():
                    _pos = [r["position"] for r in _rows]
                    if _pos:
                        _fig_pos.add_trace(
                            go.Violin(x=_pos, name=_lbl, meanline_visible=True, orientation="h")
                        )
                _fig_pos.update_layout(
                    title="Transition position in episode",
                    height=300,
                    xaxis_title="Fraction of episode (0=start, 1=end)",
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_pos, use_container_width=True, key="dbg_edge_pos")
            with _dc2:
                _fig_sr = go.Figure()
                _sr_names, _sr_vals, _sr_n = [], [], []
                for _lbl, _rows in _edge_data.items():
                    _succ = [r for r in _rows if r["success"] is True]
                    _tot = len(_rows)
                    _sr_names.append(_lbl)
                    _sr_vals.append(len(_succ) / _tot if _tot else 0)
                    _sr_n.append(_tot)
                _fig_sr.add_trace(
                    go.Bar(
                        x=_sr_names,
                        y=_sr_vals,
                        text=[f"{v:.0%} (n={n})" for v, n in zip(_sr_vals, _sr_n)],
                        textposition="outside",
                        marker_color="#4e79a7",
                    )
                )
                _fig_sr.update_layout(
                    title="Success rate after transition",
                    height=300,
                    yaxis=dict(range=[0, 1.15], title="Success rate"),
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_sr, use_container_width=True, key="dbg_edge_sr")

    else:
        # ── Node mode ────────────────────────────────────────────────────────
        if not _active_nodes:
            st.caption("Pick nodes from the dropdown above, or click a node in the graph.")
        else:
            # VLM labels summary
            if any(nid in _node_vlm_labels for nid in _active_nodes):
                for _nid in _active_nodes:
                    _vlm = _node_vlm_labels.get(_nid)
                    if _vlm:
                        _conf = _vlm.get("confidence")
                        _conf_str = f" · {_conf:.0%} confidence" if _conf is not None else ""
                        _src = _vlm.get("source", "VLM")
                        _desc = _vlm.get("description", "")
                        st.info(
                            f"**{_vlm['label']}** ({_behavior_nodes.get(_nid, f'Behavior {_nid}')}) — "
                            f"{_desc}  \n_{_src}{_conf_str}_"
                        )
                        _reas = _vlm.get("reasoning", "")
                        if _reas:
                            with st.expander("VLM reasoning", expanded=False):
                                st.caption(_reas)

            _node_data: dict[str, list[dict]] = {}
            for _nid in _active_nodes:
                _lbl = _node_labels.get(_nid, str(_nid))
                _node_data[_lbl] = _insp_index.node_stats(_nid)

            # Row 1: duration | windows assigned
            _dc1, _dc2 = st.columns(2)
            with _dc1:
                _fig_dur = go.Figure()
                for _ci, (_lbl, _rows) in enumerate(_node_data.items()):
                    _dur = [r["duration"] for r in _rows]
                    if _dur:
                        _fig_dur.add_trace(
                            go.Violin(
                                y=_dur,
                                name=_lbl,
                                meanline_visible=True,
                                box_visible=False,
                                marker_color=_PALETTE[_ci % len(_PALETTE)],
                            )
                        )
                _fig_dur.update_layout(
                    title="Visit duration (window span)",
                    height=320,
                    yaxis_title="Windows (span)",
                    xaxis_title="Behavior",
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_dur, use_container_width=True, key="dbg_dur")

            with _dc2:
                _fig_cnt = go.Figure()
                for _ci, (_lbl, _rows) in enumerate(_node_data.items()):
                    _cnt = [r["count"] for r in _rows]
                    if _cnt:
                        _fig_cnt.add_trace(
                            go.Violin(
                                y=_cnt,
                                name=_lbl,
                                meanline_visible=True,
                                box_visible=False,
                                marker_color=_PALETTE[_ci % len(_PALETTE)],
                            )
                        )
                _fig_cnt.update_layout(
                    title="Windows assigned to node",
                    height=320,
                    yaxis_title="Window count",
                    xaxis_title="Behavior",
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_cnt, use_container_width=True, key="dbg_cnt")

            # Row 2: relative position | absolute start/end (split violin)
            _dc3, _dc4 = st.columns(2)
            with _dc3:
                _fig_pos = go.Figure()
                for _ci, (_lbl, _rows) in enumerate(_node_data.items()):
                    _pos = [r["position"] for r in _rows]
                    if _pos:
                        _fig_pos.add_trace(
                            go.Violin(
                                y=_pos,
                                name=_lbl,
                                meanline_visible=True,
                                box_visible=False,
                                marker_color=_PALETTE[_ci % len(_PALETTE)],
                            )
                        )
                _fig_pos.update_layout(
                    title="Relative position in episode",
                    height=320,
                    yaxis=dict(range=[0, 1], title="Fraction (0=start, 1=end)"),
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_pos, use_container_width=True, key="dbg_pos")

            with _dc4:
                # Split violin: left half = start timestep, right half = end timestep
                _fig_abs = go.Figure()
                for _ci, (_lbl, _rows) in enumerate(_node_data.items()):
                    _col = _PALETTE[_ci % len(_PALETTE)]
                    _starts = [r["ts_start"] for r in _rows]
                    _ends = [r["ts_end"] for r in _rows]
                    if _starts:
                        _fig_abs.add_trace(
                            go.Violin(
                                x=[_lbl] * len(_starts),
                                y=_starts,
                                name="start",
                                legendgroup="start",
                                side="negative",
                                line_color=_col,
                                fillcolor=_hex_to_rgba(_col, 0.8),
                                meanline_visible=True,
                                box_visible=False,
                                showlegend=(_ci == 0),
                                scalegroup=_lbl,
                            )
                        )
                        _fig_abs.add_trace(
                            go.Violin(
                                x=[_lbl] * len(_ends),
                                y=_ends,
                                name="end",
                                legendgroup="end",
                                side="positive",
                                line_color=_col,
                                fillcolor=_hex_to_rgba(_col, 0.3),
                                meanline_visible=True,
                                box_visible=False,
                                showlegend=(_ci == 0),
                                scalegroup=_lbl,
                            )
                        )
                _fig_abs.update_layout(
                    title="Absolute start / end timestep (split)",
                    violinmode="overlay",
                    height=320,
                    yaxis_title="Window index",
                    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                    margin=dict(l=0, r=0, t=48, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_abs, use_container_width=True, key="dbg_abs")

            # Row 3: success rate | outgoing transitions (all nodes)
            _dc5, _dc6 = st.columns(2)
            with _dc5:
                _fig_sr = go.Figure()
                for _ci, (_lbl, _rows) in enumerate(_node_data.items()):
                    _tot = len(_rows)
                    _succ = sum(1 for r in _rows if r["success"] is True)
                    _val = _succ / _tot if _tot else 0
                    _fig_sr.add_trace(
                        go.Bar(
                            x=[_lbl],
                            y=[_val],
                            text=[f"{_val:.0%} (n={_tot})"],
                            textposition="outside",
                            marker_color=_PALETTE[_ci % len(_PALETTE)],
                            showlegend=False,
                        )
                    )
                _fig_sr.update_layout(
                    title="Success rate (episodes visiting node)",
                    height=300,
                    yaxis=dict(range=[0, 1.15], title="Success rate"),
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_sr, use_container_width=True, key="dbg_sr")

            with _dc6:
                _fig_tr = go.Figure()
                for _ci, _nid in enumerate(_active_nodes):
                    _lbl = _node_labels.get(_nid, str(_nid))
                    _out = bg.transition_probs.get(_nid, {})
                    if _out:
                        _sorted_items = sorted(_out.items(), key=lambda x: -x[1])
                        _tgt_names = [
                            bg.nodes[t].name if t in bg.nodes else str(t) for t, _ in _sorted_items
                        ]
                        _tgt_probs = [p for _, p in _sorted_items]
                        _fig_tr.add_trace(
                            go.Bar(
                                name=_lbl,
                                x=_tgt_names,
                                y=_tgt_probs,
                                text=[f"{p:.0%}" for p in _tgt_probs],
                                textposition="outside",
                                marker_color=_PALETTE[_ci % len(_PALETTE)],
                            )
                        )
                _fig_tr.update_layout(
                    title="Outgoing transitions",
                    height=320,
                    yaxis=dict(range=[0, 1.15], title="Transition prob."),
                    barmode="group",
                    margin=dict(l=0, r=0, t=36, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_tr, use_container_width=True, key="dbg_tr")

# ── Graph Analysis Report ─────────────────────────────────────────────────────
if _graph_analysis:
    render_graph_analysis_panel(
        _graph_analysis,
        node_vlm_labels=_node_vlm_labels if _node_vlm_labels else None,
    )
