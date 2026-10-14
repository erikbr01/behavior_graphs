"""The site the paper prints: one page, in the order of the argument.

Run with:
    streamlit run apps/reviewer/Home.py

Served at try.behavior-graphs.com (the ``reviewer`` service in
``deploy/docker-compose.yml``), which is the URL the paper's abstract carries.

**One page, not a set of tabs.** The four things this shows are one argument —
here is the representation, here is how it is built, here is what operators did
with it, here is what the same structure is worth for something else — and tabs
let a reader land in the middle of it with no way to tell what they had skipped.
The page is long on purpose and reads top to bottom.

**It carries no author, institution, repository or contact**, and nothing links
out to the demo or the study app. That began as review anonymity and is kept
because it is the right shape for the page: a reader arriving from the abstract
wants the argument, and the paper beside it already carries the attribution.

**Where the content comes from.** The behavior graphs, the embeddings and the
segmentations are read live from the pipeline runs this deployment mounts, so
they are the runs the study's participants read rather than a re-tuned
lookalike. The *results* are read from :mod:`behavior_graphs.results`, exported by
``scripts/export_paper_results.py``, because this container mounts the runs and
nothing else: no response store, no progress benchmark. The overview figure is
the paper's own, under ``assets/``.

**What is deliberately absent.** No control changes what a result *is*. The
selectors pick which task, which collection and which outcome group to look at;
none of them re-fits a model or re-runs a test. K is not among them on purpose:
it is the one setting that changes what a graph *says* rather than how it looks,
and a reader sliding it would be comparing graphs of different granularity with
nothing on the page about how K was chosen for the paper. Each graph here is the
one its config pins, or the best-scoring run at the K the model selection chose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the repo's src tree is importable when behavior_graphs isn't
# pip-installed (e.g. running straight from a checkout).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="Behavior Graphs",
    layout="wide",
    page_icon="🤖",
    # Belt to `hide_sidebar`'s braces: nothing renders into the sidebar, but if
    # something ever does, collapsed is a better failure than open.
    initial_sidebar_state="collapsed",
)

from behavior_graphs.core.node_labels import load_node_labels  # noqa: E402
from behavior_graphs.viz.labels import number_repeats  # noqa: E402
from behavior_graphs.viz.plotly import construction  # noqa: E402
from behavior_graphs.viz.plotly import results as charts  # noqa: E402
from behavior_graphs.viz.streamlit import reviewer  # noqa: E402
from behavior_graphs.viz.streamlit.chrome import hide_sidebar  # noqa: E402
from behavior_graphs.viz.streamlit.components import (  # noqa: E402
    paper_results,
    references,
)
from behavior_graphs.viz.streamlit.components.graph_config import (  # noqa: E402
    tree_kwargs_from_config,
)
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)
from behavior_graphs.viz.streamlit.components.video_curve_player import (  # noqa: E402
    video_curve_player,
)
from behavior_graphs.viz.streamlit.data import (  # noqa: E402
    legacy_mp4_index,
    load_run,
    resolve_data_path,
    run_index_rows,
)
from behavior_graphs.viz.streamlit.lazy import lazy_section  # noqa: E402
from behavior_graphs.viz.streamlit.user_study.graph_intro import render_graph_intro  # noqa: E402

#: The four stages of construction, in the order the method runs them, drawn
#: one after another down the page. Numbered because they are a sequence and a
#: reader arriving mid-page should be able to tell where in it they are.
CONSTRUCTION_STAGES = (
    "1. Windowing",
    "2. Behavior Discovery",
    "3. Segments",
    "4. Behavior Graph",
)


@st.cache_data(show_spinner=False)
def progress_clip(dataset: str, index: int, success: bool) -> str | None:
    """The clip for one rollout of one collection, or ``None`` if not mounted.

    ``_episode_mp4`` rather than a path built here: the rollout dirs use two
    layouts (``media/ep0000_fail.mp4`` in sim, ``media/<camera>/…`` on the real
    robot) and that function knows both, including which camera view to prefer.
    Private, and imported by the demo's progress page the same way.
    """
    from behavior_graphs.progress_estimation.io import _episode_mp4
    from behavior_graphs.viz.streamlit.rollout_outcomes import ROLLOUTS_SUBDIR

    root = resolve_data_path()
    if not root:
        return None
    found = _episode_mp4(Path(root) / ROLLOUTS_SUBDIR / dataset, index, success)
    return str(found) if found is not None else None


@st.cache_data(show_spinner=False)
def task_clip(mp4_dir: str | None, index: dict) -> str | None:
    """A URL for one successful rollout of the task, or ``None``.

    The same clips the study's video browser showed: the rollout collection's
    own mp4s, through :func:`~behavior_graphs.viz.media.ensure_proxy` so the page
    serves a web-sized transcode rather than a full-resolution capture.

    ``ensure_proxy`` and not ``video_src``: the latter returns a URL for an
    HTML ``<video src=…>`` and ``st.video`` treats a non-http string as a file
    path, so handing it one raised ``Error opening '/mock/media/….mp4'``.

    A success rather than the first episode, because the clip is here to show
    what the task *is* and a failed rollout shows the task not being done. By
    lowest index among the successes, so it does not change between reruns.
    """
    from behavior_graphs.viz.media import ensure_proxy

    if not mp4_dir:
        return None
    wins = sorted(
        (e for e in index.get("episodes", ()) if e.get("success")), key=lambda e: e["index"]
    )
    if not wins:
        return None
    clip = Path(mp4_dir) / wins[0]["path"]
    return str(ensure_proxy(clip)) if clip.is_file() else None


hide_sidebar()

st.markdown(
    """
    <style>
      /* Streamlit opens with ~6rem of empty canvas above the first element,
         which on a page whose first element is the point is 6rem of nothing. */
      [data-testid="stMainBlockContainer"], .stMainBlockContainer { padding-top: 2.2rem; }

      /* A measure for the whole page, centred. `layout="wide"` lets every
         figure span the window, which on a wide monitor stretches a
         three-panel chart to a couple of thousand pixels: the panels grow but
         the ink in them does not, so the reader gets more whitespace between
         violins rather than a bigger violin. Bounded here instead, and the
         graph component sizes itself to whatever it is given. */
      [data-testid="stMainBlockContainer"], .stMainBlockContainer {
        max-width: 72rem;
        margin-inline: auto;
      }

      /* A narrower measure again for prose, inset inside the page's. Figures
         want the width and a line of text does not: 72rem of 14px text is
         about 120 characters, which is roughly twice a comfortable line.
         Applied to the text block rather than to its paragraphs, so headings,
         lists, prose and captions all share one centred column -- a centred
         paragraph under a left-aligned heading is worse than either. The text
         inside stays left-aligned: this centres the measure, not the lines. */
      [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"],
      [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"],
      [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] {
        max-width: 60rem;
        margin-inline: auto;
      }

      /* The masthead only. Centred prose is harder to read, so this is the
         one block that gets it: a title and a subtitle, each one line. */
      .st-key-masthead [data-testid="stMarkdownContainer"] { text-align: center; }

      /* Air above each section heading, so the page reads as parts rather than
         as one column of text. */
      [data-testid="stMainBlockContainer"] h2 { margin-top: 2.6rem; }

      /* The graph canvas, as a card. The component draws on a transparent
         background, so without a frame a tree with a sparse top edge floats.
         A border and nothing else: the component sizes an opaque backing chip
         behind every node label in the canvas colour, and tinting the card
         turns each of them into a visible box. */
      .st-key-build_graph, .st-key-explore_graph {
        border: 1px solid color-mix(in srgb, currentColor 11%, transparent);
        border-radius: 14px;
        padding: 0.2rem 0.2rem 0;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── What a behavior graph is ─────────────────────────────────────────────────

with st.container(key="masthead"):
    st.title("Behavior Graphs")
    st.markdown("#### Unsupervised Graph Structures for Interpreting Robot Deployment Data")
st.image(str(Path(__file__).parent / "assets" / "overview.png"))
st.markdown(
    "Data flywheels promise continuous improvement via insights from "
    "deployment data. However, robot deployments create large volumes of "
    "high-dimensional rollout data that is difficult to interpret and thus "
    "extract actionable insights from. To facilitate data flywheels for "
    "robotic manipulation, we propose behavior graphs, which compress policy "
    "rollouts into a single interpretable, directed graph. We construct the "
    "graphs by identifying similar behavior modes across rollouts, connecting "
    "the behaviors temporally, and finally linking them to outcomes. In a user "
    "study on three real-world tasks, 16 of 18 participants preferred behavior "
    "graphs to browsing rollout videos, reaching comparable conclusions with "
    "40.3% lower perceived temporal demand and 37.6% lower effort. Beyond "
    "facilitating interpretability for deployment data, behavior graphs are "
    "capable progress predictors: across 43 rollout collections they raise "
    "value-order correlation from 0.769 to 0.945 and halve the mean absolute "
    "error of the strongest baseline. As the number of deployed robots grows, "
    "behavior graphs introduce the structure needed to interpret and reason "
    "about deployment data at scale."
)

# ── How a graph is built ─────────────────────────────────────────────────────

st.header("Graph Construction")
st.markdown(
    "On the three tasks of the user study. Each graph is the run its participants were shown."
)

_rows = run_index_rows()
if not {row["task"] for row in _rows if row["k"] is not None}:
    st.error("No runs are mounted here, so there is nothing to show.")
    st.stop()

_pins = reviewer.pinned_graphs(_rows)
_study_pins = reviewer.study_graphs(_rows)
if not _study_pins:
    st.error("None of the study's three graphs resolve to a mounted run.")
    st.stop()

_policies = paper_results.section("policies")
_paper_task_names = {
    str(arm["dataset"]): str(task["label"])
    for task in _policies["tasks"]
    for arm in task["arms"]
    if arm["baseline"]
}
_pin_labels = {_paper_task_names.get(pin.task, pin.label): pin for pin in _study_pins}
_task_name = st.selectbox("Task", list(_pin_labels), key="build_task")
_pin = _pin_labels[_task_name]
if _pin.description:
    st.markdown(_pin.description)
_build_run = load_run(str(_pin.run_dir), data_path=resolve_data_path())
_mp4_dir, _mp4_index = legacy_mp4_index(_build_run.rollout_dir)
_example = task_clip(_mp4_dir, _mp4_index)
if _example is not None:
    _left, _middle, _right = st.columns([1, 2, 1])
    _middle.video(_example)
_build_vlm = load_node_labels(Path(_pin.run_dir))
_build_names = number_repeats({nid: entry["label"] for nid, entry in _build_vlm.items()})
_build_k = len(set(_build_run.labels.tolist()))

# The four stages inline rather than behind a selector. A reader following a
# construction wants to see the step before and the step after at once, and a
# radio hides three quarters of a sequence behind a click. It costs one more
# scatter than the selector did, the graph component being the only expensive
# thing here and there being one of it either way.

st.subheader(CONSTRUCTION_STAGES[0])
st.markdown(
    "Each rollout is cut into overlapping windows of fixed width and stride, "
    "and every window is embedded with the policy's observation encoder."
)
if _build_run.coords is None:
    st.info(
        "This run stored no two-dimensional projection, so the embedding "
        "panels cannot be drawn. The segments and the graph are unaffected."
    )
else:
    st.plotly_chart(
        construction.embedding_figure(
            _build_run.coords,
            _build_run.labels,
            _build_run.metadata,
            _build_names,
            clustered=False,
        ),
        key="build_windows",
        config=charts.CHART_CONFIG,
    )
    st.caption(
        f"{len(_build_run.labels):,} windows from "
        f"{_build_run.graph.num_episodes} rollouts, projected to two "
        f"dimensions with UMAP {references.cite('umap')}."
    )

st.subheader(CONSTRUCTION_STAGES[1])
st.markdown(
    f"k-means {references.cite('kmeans')} partitions the windows into K "
    "behaviors. A vision-language model "
    "captions each cluster from a contact sheet of its frames."
)
if _build_run.coords is not None:
    st.plotly_chart(
        construction.embedding_figure(
            _build_run.coords,
            _build_run.labels,
            _build_run.metadata,
            _build_names,
            clustered=True,
        ),
        key="build_behaviors",
        config=charts.CHART_CONFIG,
    )
    st.caption(
        f"The same windows, each coloured by the behavior it was assigned to, of {_build_k}."
    )

st.subheader(CONSTRUCTION_STAGES[2])
st.markdown(
    "Consecutive windows assigned to the same behavior collapse into one "
    "segment, reducing each rollout from a few hundred windows to a short "
    "sequence of behaviors."
)
st.plotly_chart(
    construction.segment_figure(_build_run.labels, _build_run.metadata, _build_names),
    key="build_segments",
    config=charts.CHART_CONFIG,
)

st.subheader(CONSTRUCTION_STAGES[3])
st.markdown(
    "The sequences define the graph: one node per behavior, one edge per "
    "observed transition weighted by its empirical probability, and a terminal "
    "edge into each rollout's outcome. A node's success likelihood is the "
    "fraction of the rollouts visiting it that succeeded."
)
st.caption(f"**{_task_name}** · {_build_run.graph.num_episodes} rollouts · {_build_k} behaviors")


@st.fragment
def _render_build_graph() -> None:
    """A fragment because the component reruns itself when a panel closes."""
    render_trajectory_tree(
        labels=_build_run.labels,
        metadata=_build_run.metadata,
        **tree_kwargs_from_config(reviewer.display_params(_pin.task, _study_pins)),
        cluster_names=_build_names,
        node_vlm_labels=_build_vlm or None,
        mp4_dir=_mp4_dir,
        mp4_index=_mp4_index,
        level="rollout",
        key_prefix="build",
    )


_render_build_graph()
_min_branch = int(reviewer.display_params(_pin.task, _study_pins).get("min_branch", 1) or 1)
_pruned = (
    f" Branches reached by fewer than {_min_branch} of the "
    f"{_build_run.graph.num_episodes} rollouts are pruned."
    if _min_branch > 1
    else ""
)
st.caption(
    "Node area is visitation count; colour is success likelihood, red to "
    "green. Click a node or an edge to play the rollouts passing through it." + _pruned
)

if lazy_section("How to Read a Behavior Graph", key="build_intro"):
    render_graph_intro()

# ── The user study ───────────────────────────────────────────────────────────

st.header("Evaluating Behavior Graphs for Active Data Collection")

_study = paper_results.section("study")
_adjusted = paper_results.section("adjusted")["rows"]
_effects = {row["outcome"]: row for row in _adjusted}
_preference = _study["preference"]

st.markdown(
    f"{_study['participants']} participants analysed rollouts for the three "
    "tasks above. Two stages assigned the display method, a behavior graph for "
    "one task and a browser of rollout videos for the other; the third let "
    "the participant choose, which measures preference. After each task they "
    "ranked candidate failure modes, selected the demonstrations to collect "
    "next, and rated perceived temporal demand, effort and confidence. All "
    "panels below use the two assigned stages."
)

st.subheader("Study Responses")
_randomized = [row for row in _study["regret"] if row["randomized"]]
st.plotly_chart(
    charts.study_answers_figure(
        _preference["counts"], _study["ratings"], _study["outcomes"], _randomized
    ),
    key="study_answers",
    config=charts.CHART_CONFIG,
)
st.caption(
    "(a) Display chosen in the third stage. (b-d) Ratings, 0 to 100 in steps "
    "of 20, one dot per rating; arrows mark the improving direction. "
    "(e) Success rate regret, the gap between the best available strategy and "
    f"the one selected, over {len(_randomized)} responses."
)
st.markdown(
    f"{_preference['counts']['graph']} of {_preference['n_chose']} participants "
    "chose the graph. Perceived temporal demand falls by "
    f"{abs(_effects['temporal_demand']['relative']):.0f}% and perceived effort "
    f"by {abs(_effects['effort']['relative']):.0f}%. Confidence and regret are "
    "unchanged."
)

st.subheader("Retrained Policies")
st.markdown(
    "The regret in panel (e) is measured against what the collection "
    "strategies are actually worth, so every candidate strategy was collected "
    "and trained on: 30 additional demonstrations over the task's baseline "
    "data, one retrained policy per strategy."
)
st.plotly_chart(
    charts.success_rate_figure(
        _policies["tasks"],
        _study["choices"],
        prior=_policies["prior"],
        credible_level=_policies["credible_level"],
    ),
    key="policy_success",
    config=charts.CHART_CONFIG,
)
st.caption(
    "Beta posteriors over each policy's success rate under a uniform prior "
    f"{references.cite('lbm')}, with the median and a "
    f"{int(100 * _policies['credible_level'])}% credible interval. Letters are "
    f"a compact letter display {references.cite('letters')} over pairwise "
    f"{_policies['test'].title()} tests {references.cite('comparison')}, "
    f"{_policies['correction']}-corrected "
    f"at α = {_policies['alpha']:g}; policies sharing a letter are not "
    "separated. Bars give the share of each display's responses requesting "
    "that policy's data."
)
st.markdown(
    "In *set table*, only the held-out carton position improves performance, "
    "as the alternatives add data for conditions the policy already handles. "
    "The bottom-shelf bowl is likewise the better choice in *prepare tray*, by "
    "a smaller margin. In *cooking*, 30 demonstrations targeted at a single "
    "subtask produced no significant difference in success rate between the "
    "options."
)

st.subheader("Adjusted Estimates")
st.markdown(
    "After fitting a linear mixed-effects model with fixed effects for task, "
    "study stage and self-reported experience, and a per-participant random "
    "intercept, we report the resulting estimated marginal means and 95% "
    "intervals for the difference between the graph and videos."
)
st.dataframe(
    charts.effect_table(_adjusted),
    hide_index=True,
    column_config={
        "Graph": st.column_config.NumberColumn(format="%.1f"),
        "Videos": st.column_config.NumberColumn(format="%.1f"),
        "Difference": st.column_config.NumberColumn(format="%+.1f"),
    },
)

# ── Progress estimation ──────────────────────────────────────────────────────

st.header("Progress Estimation")

_progress = paper_results.section("progress")
_curves = paper_results.curves()
_methods = _progress["methods"]
_metrics = _progress["metrics"]
_averages = {row["method"]: row for row in _progress["means"] if row["domain"] == "Average"}
_ours_key = next(method["key"] for method in _methods if method.get("ours"))

st.markdown(
    "How far a rollout got through its task is a number a lot of downstream "
    "work needs: analysing a deployment retroactively, ranking rollouts, "
    "scoring them as a reward model. Behavior graphs supply it without a "
    "second model, which also tests whether the structure is worth something "
    "beyond being read."
)
st.markdown(
    "Each behavior is treated as a subtask, the stage-conditioned "
    f"formulation of SARM {references.cite('sarm')} with the stages read off "
    "the graph rather than annotated. Progress is linear within a behavior, so "
    "a rollout's estimate is piecewise linear with two parameters per "
    "behavior: the fraction of a rollout it occupies, and the progress "
    "completed when it begins. Both are fit on successful trajectories. No "
    "network is trained and no subtask is annotated."
)
st.markdown(
    f"Evaluated on {_progress['datasets']} rollout collections from "
    f"{len(_progress['domains'])} sources, LIBERO {references.cite('libero')}, "
    f"real-world DROID {references.cite('droid')}, Robomimic "
    f"{references.cite('robomimic')} and MimicGen "
    f"{references.cite('mimicgen')}, against two vision-language estimators, "
    f"ROBOMETER {references.cite('robometer')} and TOPReward "
    f"{references.cite('topreward')}, and an MLP. The MLP is trained on the "
    "same window embeddings the graph is built from, and thereby ablates "
    "whether the graph topology contributes to progress estimation "
    "performance."
)
st.markdown(
    "Averaged over the collections, against the best baseline on each metric: "
    f"value-order correlation {_averages[_ours_key]['voc']:.3f} against "
    f"{max(r['voc'] for k, r in _averages.items() if k != _ours_key):.3f}, "
    f"Kendall's τ {_averages[_ours_key]['kendall']:.3f} against "
    f"{max(r['kendall'] for k, r in _averages.items() if k != _ours_key):.3f}, "
    f"mean absolute error {_averages[_ours_key]['mae']:.3f} against "
    f"{min(r['mae'] for k, r in _averages.items() if k != _ours_key):.3f}. "
    "ROBOMETER retains a narrow lead on AUROC."
)

_entries = charts.curve_datasets(_curves, _progress["domains"])
_entry_labels = {f"{e['dataset']}  ·  {e['domain']}": e for e in _entries}
# Opens on a collection the page has already covered, rather than on whichever
# one sorts first.
_study_datasets = {pin.task for pin in _study_pins}
_default_entry = next(
    (
        index
        for index, name in enumerate(_entry_labels)
        if _entry_labels[name]["dataset"] in _study_datasets
    ),
    0,
)
_entry_name = st.selectbox(
    "Rollout collection", list(_entry_labels), index=_default_entry, key="curve_dataset"
)
_entry = _entry_labels[_entry_name]
_rollouts = {charts.rollout_label(r): r for r in _entry.get("rollouts", ())}
_rollout_name = st.selectbox("Rollout", list(_rollouts), key="curve_rollout")
_rollout = _rollouts[_rollout_name]

_figure = charts.progress_curve_figure(_rollout, _curves["grid"], _methods)
_clip = progress_clip(_entry["dataset"], int(_rollout["idx"]), bool(_rollout["success"]))
_drawn = False
if _clip is not None:
    _drawn = video_curve_player(
        _clip, _figure, key=f"vcp_{_entry['dataset']}_{_rollout['idx']}", chart_height_px=360
    )
if not _drawn:
    st.plotly_chart(_figure, key="progress_curves", config=charts.CHART_CONFIG)

if not _clip:
    st.caption("No clip is mounted for this rollout.")

_frame = charts.progress_table(_progress["means"], _metrics, _methods)
_best = charts.best_cells(_frame, _metrics)
# Hand-built HTML rather than `st.dataframe`, for the one thing a dataframe
# cannot do: the paper's \multirow. A dataframe can blank the repeated Source
# values but not merge the cells, so the block stays four rows with one label
# floating at the top of them, and its explicit height either clips the last
# row or leaves empty ones under it.
_cells = []
for _position, (_index, _row) in enumerate(_frame.iterrows()):
    _block = int((_frame["Source"] == _row["Source"]).sum())
    _first = _position == 0 or _frame["Source"].iloc[_position - 1] != _row["Source"]
    _tds = [f'<th rowspan="{_block}" class="src">{_row["Source"]}</th>'] if _first else []
    _tds.append(f'<th class="mth">{_row["Method"]}</th>')
    for _column in _frame.columns[2:]:
        _value = _row[_column]
        _shown = "n/a" if _value != _value else f"{float(_value):.3f}"
        _weight = ' class="best"' if (_index, _column) in _best else ""
        _tds.append(f"<td{_weight}>{_shown}</td>")
    # A plain concatenation rather than an f-string: the deployment image
    # runs a Python that rejects a backslash inside an f-string expression.
    _rule = ' class="rule"' if _first and _position else ""
    _cells.append("<tr" + _rule + ">" + "".join(_tds) + "</tr>")
_head = "".join(f"<th>{_column}</th>" for _column in _frame.columns)
st.markdown(
    f"""
    <style>
      table.progress {{
        border-collapse: collapse;
        margin: 0 auto 0.4rem;
        font-variant-numeric: tabular-nums;
        font-size: 0.86rem;
      }}
      /* Streamlit's base stylesheet puts a border on every cell, which turns
         a four-column table into a grid. Booktabs instead: horizontal rules
         only, one under the header and one between blocks. */
      /* Streamlit's base stylesheet rules every cell, which turns a four
         column table into a grid. Booktabs instead: a rule under the header,
         one between blocks and one under the last row, and nothing else. */
      /* On the row as well as the cells: Streamlit draws the row separator on
         `tr`, which a cell-level reset leaves standing. */
      table.progress tr {{ border: none !important; }}
      table.progress th, table.progress td {{
        border: none !important;
        padding: 0.32rem 0.95rem;
        text-align: right;
      }}
      table.progress thead th {{
        font-weight: 600;
        white-space: nowrap;
        border-bottom: 1px solid currentColor !important;
      }}
      table.progress thead th:nth-child(-n+2), table.progress th.mth {{ text-align: left; }}
      /* The source named once per block and sitting against the middle of it,
         as the paper does it. */
      table.progress th.src {{
        text-align: center;
        vertical-align: middle;
        font-weight: 600;
        padding-right: 1.4rem;
      }}
      table.progress th.mth {{ font-weight: 400; }}
      table.progress tr.rule > * {{
        border-top: 1px solid color-mix(in srgb, currentColor 25%, transparent) !important;
      }}
      /* On the tbody rather than its last row: the last row of a block has no
         source cell of its own, so a rule on the row stops short of the first
         column. */
      table.progress tbody {{ border-bottom: 1px solid currentColor !important; }}
      table.progress td.best {{ font-weight: 700; }}
    </style>
    <table class="progress"><thead><tr>{_head}</tr></thead><tbody>{"".join(_cells)}</tbody></table>
    """,
    unsafe_allow_html=True,
)


# ── References ───────────────────────────────────────────────────────────────

st.header("References")
# One markdown block rather than a loop: Streamlit's ordered list renders from
# the "1." prefixes, and twelve separate blocks would each restart at one.
st.markdown(references.as_markdown())
