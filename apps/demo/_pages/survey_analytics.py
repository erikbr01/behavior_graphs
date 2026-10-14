"""Survey analytics — visualize collected user-study responses.

Reads from GCS (if SURVEY_GCS_BUCKET is set) or from the local results dir
(``$STUDY_RESULTS_DIR``, default ``./study_results``).

Every layout the study has ever had writes into the same store, and they
disagree about where an answer goes — see
:mod:`behavior_graphs.viz.streamlit.user_study.analytics`, which is where the
reading of a response dict happens. This page picks a version, asks that module
for tidy rows, and draws whatever came back: a panel with no rows for the chosen
version is not drawn at all, so the same page serves a run of the production A/B
study and a run of a contrastive one without pretending either has the other's
answers.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from behavior_graphs.viz.streamlit.lazy import lazy_section
from behavior_graphs.viz.streamlit.user_study import (
    analytics,
    failure_scoring,
    figure_export,
    mixed_effects,
    regret,
)
from behavior_graphs.viz.streamlit.user_study.response_store import (
    GCSResponseStore,
    LocalResponseStore,
    results_dir,
)

st.set_page_config(page_title="Survey Analytics — Policy Graphs", layout="wide")
st.title("Survey Analytics")
st.caption("Live view of collected user-study responses.")

# Phases are the comparison every two-phase layout is built around, so they get
# one colour each and keep it across every chart on the page.
_PHASE_COLORS = {"videos": "#1f77b4", "graph": "#ff7f0e", "": "#7f7f7f"}
_CHART_LAYOUT = dict(
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


def _displays_in(phases) -> list[str]:
    """The displays present in *phases*, graph first.

    ``regret.PHASE_ORDER`` rather than a tuple of this page's own: it is already
    the order the regret panels and the paper figures draw, for the reason the
    constant gives — the graph is what the study is about, so it belongs on the
    left of a panel. A value that order does not name (a layout that recorded no
    display writes ``""``) keeps its rows and goes after them, since leaving it
    out of the order is not the same as leaving it out of the chart.
    """
    present = set(phases)
    named = [phase for phase in regret.PHASE_ORDER if phase in present]
    return named + sorted(present - set(regret.PHASE_ORDER))


# ── Load responses ────────────────────────────────────────────────────────────


@st.cache_data(ttl=60, show_spinner="Loading responses…")
def _better_ticks(fig, order) -> None:
    """Put the way each rated dimension improves after its tick label.

    On the ticks rather than in the y-axis title, because the dimensions on one
    of these axes do not agree on a direction: every load item runs
    low-is-better -- ``performance`` included, its scale being Perfect to
    Failure -- and confidence runs the other way. One note in the axis title
    would be wrong for one of the boxes under it.

    Horizontal, too, which the y-axis title is not: Plotly rotates that title,
    and an arrow rotates with it, so a down arrow set there comes out pointing
    sideways. The same reason the paper figures put theirs beside the panel name
    (:func:`behavior_graphs.viz.streamlit.user_study.figure_export._label`).

    The arrow and nothing else, as in those figures -- what it means is said
    once, in the caption, rather than once per tick.

    *order* is the tick order already given to ``category_orders``; a dimension
    the instrument does not name keeps its tick and gains no arrow.
    """
    arrows = {"down": "↓", "up": "↑"}
    direction = {
        analytics.TLX_LABELS[key]: arrows[value]
        for key, value in analytics.BETTER_DIRECTION.items()
        if key in analytics.TLX_LABELS
    }
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(order),
        ticktext=[
            f"{label} {direction[label]}" if label in direction else label for label in order
        ],
    )


def _load_responses() -> list[dict]:
    bucket = os.environ.get("SURVEY_GCS_BUCKET", "").strip()
    if bucket:
        try:
            return GCSResponseStore(bucket).list_all()
        except Exception as e:
            st.error(f"GCS error: {e}")
            return []

    # Local results dir ($STUDY_RESULTS_DIR, default ./study_results)
    return LocalResponseStore(results_dir()).list_all()


col_refresh, col_info = st.columns([1, 8])
with col_refresh:
    if st.button("Refresh"):
        # Scoped clear — st.cache_data.clear() would also drop the runs
        # index, the rollout index and every episode index in the process.
        _load_responses.clear()

all_responses = _load_responses()

if not all_responses:
    st.info(
        "No responses found yet.  "
        "Set `SURVEY_GCS_BUCKET` or place responses under "
        "`$STUDY_RESULTS_DIR` (default ./study_results)."
    )
    st.stop()

# ── Which study ───────────────────────────────────────────────────────────────

counts = analytics.version_counts(all_responses)
_ALL = "All versions"
options = [*counts, _ALL] if len(counts) > 1 else list(counts)

version = st.selectbox(
    "Study version",
    options=options,
    format_func=lambda v: (
        f"{v} — {len(all_responses)} responses"
        if v == _ALL
        else f"{v} — {counts[v]} response{'s' if counts[v] != 1 else ''}"
    ),
    help=(
        "Which layout of the study to analyse. Layouts record their answers "
        "differently, so mixing them in one chart is usually a mistake — "
        "though the task-load and timing panels are comparable across all of "
        "them."
    ),
)
in_version = analytics.for_version(all_responses, None if version == _ALL else version)
# Before the filters below narrow it, so the "nothing to show" message can say
# how many there really are rather than how many survived.
_for_version_total = len(in_version)

# ── Participants, or this page's own previews too ─────────────────────────────
# Layouts that checkpoint (the three-stage one) write a record from the first
# page load, so rendering one on the User Study page — to check a wording, or to
# pilot by hand — leaves a record that reads as a participant who quit on the
# introduction. It would otherwise land in the funnel below and in the
# completed-of-started metric, which is the number this page exists to report
# honestly while a study is live.

preview_count = sum(1 for r in in_version if analytics.origin(r) == analytics.LAB_ORIGIN)
if preview_count:
    if not st.toggle(
        f"Include {preview_count} lab {'preview' if preview_count == 1 else 'previews'}",
        value=False,
        help=(
            "Records written by the lab app's own User Study page rather than by "
            "a participant on a recruitment link. Hand-run pilots are in here "
            "too, so they are hidden rather than dropped."
        ),
    ):
        in_version = analytics.from_participants(in_version)

# ── Finished, or everything ───────────────────────────────────────────────────
# The three-stage layout writes a record as the answers come in, so sessions that
# were abandoned sit in the store beside the completed ones. Every number below
# would otherwise quietly include someone who stopped at stage one, so the page
# shows only finished attempts unless asked — and says how many it is leaving
# out, because a silent filter is its own way of being wrong.

status_counts = analytics.status_counts(in_version)
unfinished_count = sum(n for name, n in status_counts.items() if name != "submitted")
responses = in_version
if unfinished_count:
    if not st.toggle(
        f"Include {unfinished_count} unfinished "
        f"{'session' if unfinished_count == 1 else 'sessions'}",
        value=False,
        help=(
            "Sessions that were abandoned or are still in progress. Their "
            "answers are real as far as they go, but they are not participants "
            "who completed the study, and pooling them reweights every average."
        ),
    ):
        responses = analytics.completed(in_version)

if not responses:
    # The filters above are on by default and can empty a store that is not
    # empty: a live study before anybody has submitted is *all* unfinished, and
    # that is exactly when the drop-off funnel below is worth reading. Without
    # this the page raises KeyError on a frame with no columns, which reads as
    # the analytics being broken rather than as a filter being on.
    st.info(
        "Nothing to show with the current filters. "
        f"The store holds {_for_version_total} "
        f"{'session' if _for_version_total == 1 else 'sessions'} for this version; "
        "turn on the unfinished-sessions or lab-preview toggles above to include them."
    )
    st.stop()

people = pd.DataFrame(analytics.participant_rows(responses))

# Which responses are in view, and how recently each was written — the cache key
# every figure on this page is memoized on. The three-stage layout rewrites a
# session's file as the answers come in, so keying on the ids alone would keep
# serving a figure drawn from those sessions half-finished.
_responses_key = tuple(
    sorted(
        (
            str(r.get("_response_id", "")),
            str(r.get("updated_at") or r.get("timestamp", "")),
        )
        for r in responses
    )
)

# ── Overview ──────────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
m1.metric("Responses", len(responses))
m2.metric("Participants", people["participant"].nunique())
median_minutes = people["minutes"].median(skipna=True)
m3.metric(
    "Median duration",
    "—" if pd.isna(median_minutes) else f"{median_minutes:g} min",
)
# How many of the sessions that were started got finished — the number that says
# whether the study is too long, and the one a layout that only saved on submit
# could not report at all.
started = sum(status_counts.values())
m4.metric(
    "Completed",
    f"{status_counts.get('submitted', 0)}/{started}" if started else "—",
    help="Finished sessions out of every session that was started.",
)

progress = pd.DataFrame(analytics.progress_rows(in_version))
if not progress.empty:
    st.caption(
        "Where the unfinished sessions stopped — each one is a participant who "
        "gave us this much and no more. Ordered by the page they reached."
    )
    st.dataframe(
        progress.sort_values("step")
        .groupby(["stopped_on", "status"], sort=False)
        .size()
        .reset_index(name="sessions")
        .rename(columns={"stopped_on": "Stopped on", "status": "Status", "sessions": "Sessions"}),
        hide_index=True,
    )

# Counted, not hard-coded to A and B: "A" is the no-graph condition in the
# production layout and the videos-first order in every layout since, and a
# group named group_a exists too.
by_group = people.groupby("group_label").size().reset_index(name="responses")
st.dataframe(
    by_group.rename(columns={"group_label": "Group", "responses": "Responses"}),
    hide_index=True,
)

st.divider()

# ── Submission timeline ───────────────────────────────────────────────────────

st.subheader("Submission timeline")


def _parsed(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(str(stamp)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


timeline = people.assign(time=people["timestamp"].map(_parsed)).dropna(subset=["time"])
if timeline.empty:
    st.info("No parseable timestamps yet.")
else:
    timeline = timeline.sort_values("time")
    timeline["cumulative"] = timeline.groupby("group_label").cumcount() + 1
    fig = px.line(
        timeline,
        x="time",
        y="cumulative",
        color="group_label",
        markers=True,
        labels={"cumulative": "Cumulative submissions", "time": "Time", "group_label": "Group"},
        title="Cumulative submissions over time",
    )
    fig.update_layout(height=320, **_CHART_LAYOUT)
    st.plotly_chart(fig, key="sa_timeline")

st.divider()

# ── Stages: which task, which display, and who picked it ──────────────────────

stages = pd.DataFrame(analytics.stage_rows(responses))
if not stages.empty:
    st.subheader("Stages")
    st.caption(
        "The three-stage layout shows one task per stage and one display per "
        "task, in an order drawn per participant — so which display a task was "
        "read in is a property of the response, not of the group."
    )

    # The layout's own result, and the reason the stage exists: with nothing
    # assigned, which display does someone reach for? Counted separately from
    # the assigned stages, which would otherwise drown it 2:1.
    chosen = stages[stages["assignment"].eq("chosen") & stages["phase"].ne("")]
    if chosen.empty:
        st.caption("No free-choice stage in these responses.")
    else:
        picked = chosen.groupby("phase").size().reset_index(name="participants")
        fig = px.bar(
            picked,
            x="phase",
            y="participants",
            color="phase",
            color_discrete_map=_PHASE_COLORS,
            labels={"phase": "Display", "participants": "Participants"},
            title="Which display participants chose when it was up to them",
        )
        fig.update_layout(height=320, showlegend=False, **_CHART_LAYOUT)
        st.plotly_chart(fig, key="sa_chosen_modality")

    # Every stage of every participant: the check that the draw is actually
    # spreading tasks and displays across stages rather than favouring one.
    coverage = (
        stages.groupby(["stage", "task_label", "phase"]).size().reset_index(name="participants")
    )
    with st.expander("Task and display per stage"):
        st.dataframe(
            coverage.rename(
                columns={
                    "stage": "Stage",
                    "task_label": "Task",
                    "phase": "Display",
                    "participants": "Participants",
                }
            ),
            hide_index=True,
        )
    st.divider()

# ── Task load ─────────────────────────────────────────────────────────────────

st.subheader("NASA Task Load Index")

tlx = pd.DataFrame(analytics.tlx_rows(responses))
if tlx.empty:
    st.info("No task-load ratings in these responses.")
else:
    rated_per_phase = tlx["phase"].ne("").any()
    # Per task means one display each: a box is then people rating *different*
    # tasks off one display, not the same task off two. Saying "the same task"
    # over the three-stage layout would describe a comparison it never made —
    # and both captions are said where both kinds of response are in view, since
    # one three-stage response among thirty contrastive ones would otherwise
    # relabel all thirty.
    per_task = tlx["task"].ne("").any()
    per_modality_only = (tlx["task"].eq("") & tlx["phase"].ne("")).any()
    if per_modality_only:
        st.caption(
            "Rated once per modality, so the two boxes over a dimension are the "
            "same people rating the same task off different evidence."
        )
    if per_task:
        st.caption(
            "Where a response rated one display per task (the three-stage "
            "layout), a box is that display's ratings across tasks rather than "
            "two readings of one task."
        )
    order = [dim for dim in analytics.TLX_LABELS.values() if dim in set(tlx["dimension"])]
    # Graph on the left, as everywhere else the two displays are drawn side by
    # side — px otherwise orders them by whichever it met first in the frame,
    # so the same two boxes swap sides when a different response loads first.
    phase_order = _displays_in(tlx["phase"])
    fig = px.box(
        tlx,
        x="dimension",
        y="score",
        color="phase" if rated_per_phase else "group_label",
        color_discrete_map=_PHASE_COLORS if rated_per_phase else None,
        category_orders={"dimension": order, "phase": phase_order},
        labels={
            "score": "TLX score (0–100)",
            "dimension": "Dimension",
            "phase": "Evidence",
            "group_label": "Group",
        },
        title="Task load by dimension",
        points="all",
    )
    _better_ticks(fig, order)
    fig.update_layout(height=420, yaxis=dict(range=[0, 100]), **_CHART_LAYOUT)
    st.plotly_chart(fig, key="sa_tlx")

    if rated_per_phase:
        per_phase = (
            tlx.groupby(["phase", "dimension"])["score"].mean().round(1).unstack(0).reindex(order)
        )
        st.dataframe(per_phase)

    # ── The same ratings, by task ────────────────────────────────────────────
    # Only the per-task layouts have this to say: a session rated once has one
    # task and nothing to compare it with.
    if per_task:
        st.subheader("Task load per task")
        st.caption(
            "One row per task, longest first, and inside a row the display the "
            "task was read in — which the panel above pools across tasks and "
            "this one does not. Two readings at once: whether a longer or more "
            "complex task cost the participant more, and whether the display "
            "mattered more on the task that did. Each participant rated a task "
            "off one display, so the two boxes in a row are different people, "
            "and the tasks came in a drawn order — at these numbers this is "
            "something to look at, not a test."
        )
        by_task = tlx[tlx["task"].ne("")]
        rated = set(by_task["dimension"])
        # The instrument's order first, then anything a layout asked that the
        # instrument does not name — which is a column of real answers, so it
        # goes last rather than nowhere.
        dims = [dim for dim in order if dim in rated] + sorted(rated - set(order))

        # Longest first, by the time participants actually spent on the task.
        # "Longer" is what the ratings are being read against, and the clock is
        # the only thing here that is measured rather than rated — including
        # against the temporal-demand item, which asks the same thing of the
        # participant and can disagree with it.
        spent = pd.DataFrame(analytics.task_duration_rows(responses))
        median_seconds = (
            spent.groupby("task_label")["seconds"].median()
            if not spent.empty
            else pd.Series(dtype="float64")
        )
        tasks = (
            by_task[["task_label"]]
            .drop_duplicates()
            .assign(minutes=lambda df: df["task_label"].map(median_seconds).div(60).round(1))
            .sort_values(["minutes", "task_label"], ascending=[False, True], na_position="last")
            .set_index("task_label")
        )
        longest_first = list(tasks.index)

        # One full-width row per task rather than three narrow columns: the x
        # axis is the six-or-seven-category dimension axis, which does not
        # survive being a third as wide, and it is drawn once at the bottom for
        # all of them. Colour is the display, from the page's own two hues — a
        # task is named by its row, not by a colour of its own.
        shown_phases = _displays_in(by_task["phase"])
        fig = px.box(
            by_task,
            x="dimension",
            y="score",
            color="phase",
            color_discrete_map=_PHASE_COLORS,
            facet_row="task_label",
            category_orders={
                "dimension": dims,
                "task_label": longest_first,
                "phase": shown_phases,
            },
            labels={
                "score": "TLX score (0–100)",
                "dimension": "Dimension",
                "phase": "Evidence",
                "task_label": "Task",
            },
            title="Task load by dimension, task and display",
            points="all",
        )
        # px labels a facet "task_label=Cook the meal"; the column name is not
        # news to anyone reading the chart.
        fig.for_each_annotation(lambda note: note.update(text=note.text.split("=")[-1]))
        # The dimension axis is drawn once, under the bottom facet, so this
        # reaches the one that is labelled.
        _better_ticks(fig, dims)
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(height=240 * len(longest_first) + 120, **_CHART_LAYOUT)
        st.plotly_chart(fig, key="sa_tlx_by_task")

        # Every (task, display) that somebody actually rated, tasks longest
        # first and the displays always in the same order within a task.
        rated_cells = set(map(tuple, by_task[["task_label", "phase"]].drop_duplicates().values))
        cells = pd.MultiIndex.from_tuples(
            [
                (task, phase)
                for task in longest_first
                for phase in shown_phases
                if (task, phase) in rated_cells
            ],
            names=["Task", "Display"],
        )

        # ``groupby`` and ``unstack`` rather than ``pivot_table``: that drops a
        # column whose every cell is NaN, which is exactly what the spread
        # table below is at one rating per cell — and a missing column is a
        # KeyError rather than a blank cell.
        def _per_cell(how: str) -> pd.DataFrame:
            return (
                by_task.groupby(["task_label", "phase", "dimension"])["score"]
                .agg(how)
                .unstack("dimension")
                .reindex(index=cells, columns=dims)
                .round(1)
            )

        means = _per_cell("mean")
        overview = pd.DataFrame(
            {
                # Attempts, not distinct ``participant``: the recruitment link
                # carries no participant id, so every live record is "anon" and
                # a nunique over it reports 1 for a cell of any size.
                "Sessions": by_task.groupby(["task_label", "phase"])["response_id"].nunique(),
                "Median min on task": (
                    spent.groupby(["task_label", "phase"])["seconds"].median().div(60).round(1)
                    if not spent.empty
                    else pd.Series(dtype="float64")
                ),
            }
        ).reindex(cells)
        st.dataframe(overview.join(means))

        with st.expander("How much the answers moved"):
            st.caption(
                "Three kinds of movement, and the question needs them apart: "
                "how much the people who rated one task off one display "
                "disagreed with each other, how much the displays disagreed "
                "within a task, and how much the tasks disagreed at all. A cell "
                "with one rating has no spread of its own and is blank in the "
                "first table."
            )
            within = _per_cell("std")
            st.dataframe(within)

            by_phase = (
                by_task.groupby(["task_label", "dimension", "phase"])["score"]
                .mean()
                .unstack("phase")
            )
            if {"graph", "videos"} <= set(by_phase.columns):
                st.caption("Graph minus videos, per task — positive means the graph rated higher.")
                st.dataframe(
                    (by_phase["graph"] - by_phase["videos"])
                    .unstack("dimension")
                    .reindex(index=longest_first, columns=dims)
                    .round(1)
                    .rename_axis("Task")
                )

            st.caption("And the tasks against each other, pooling the displays as the panel above.")
            pooled = (
                by_task.groupby(["task_label", "dimension"])["score"]
                .mean()
                .unstack("dimension")
                .reindex(index=longest_first, columns=dims)
                .round(1)
            )
            st.dataframe(
                (pooled.max() - pooled.min()).to_frame().T.set_axis(["Max − min of the task means"])
            )

st.divider()

# ── Did the evidence change the answer? ───────────────────────────────────────

changes = pd.DataFrame(analytics.phase_disagreements(responses))
if not changes.empty:
    st.subheader("Answers that changed between the two modalities")
    st.caption(
        "One row per participant per question: what they answered after the "
        "first modality, what they answered after the second, and whether it "
        "moved. This is the comparison the two-phase layouts exist to make."
    )
    summary = (
        changes.groupby("block")["changed"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "changed", "count": "answered"})
    )
    summary["changed"] = summary["changed"].astype(int)
    st.dataframe(summary)
    with st.expander("Every answer pair"):
        st.dataframe(
            changes[
                [
                    "participant",
                    "group_label",
                    "block",
                    "first_phase",
                    "first_answer",
                    "second_answer",
                    "changed",
                ]
            ],
            hide_index=True,
        )
    st.divider()

# ── Ordered answers ───────────────────────────────────────────────────────────

rankings = pd.DataFrame(analytics.ranking_rows(responses))
if not rankings.empty:
    st.subheader("Ranked answers")
    _BLOCK_TITLES = {
        "failure_mode": "Failure modes, by how highly they were ranked",
        "data_collection": "Data-collection priorities",
        "strategy": "Strategies picked into a top-N",
    }
    # Grouped by task as well as by block, and empty-tasked layouts fall into one
    # group per block exactly as before. Each task asks for a ranking of its own
    # depth — set_table ranks two failure modes, cooking three — and the weight a
    # rank is worth is computed from that depth below, so pooling the tasks put
    # one task's rank 2 (its last) on another's three-deep scale and made the
    # bars incomparable under an axis that claims otherwise.
    for (block, task), rows in rankings.groupby(["block", "task"]):
        # Rank 1 is worth more than rank 3, so weight by it rather than
        # counting mentions: an option every participant put last would
        # otherwise look as popular as one they all put first.
        depth = int(rows["rank"].max())
        rows = rows.assign(weight=(depth + 1 - rows["rank"]) / depth)
        # Split by phase where the layout has phases and by group where it does
        # not: the production study ranks once per participant, and colouring
        # that by phase summed its two conditions into one bar — losing the A/B
        # comparison the study exists to make.
        split = "phase" if rows["phase"].ne("").any() else "group_label"
        agg = (
            rows.groupby(["option", split])["weight"]
            .sum()
            .reset_index(name="score")
            .sort_values("score", ascending=False)
        )
        fig = px.bar(
            agg,
            x="score",
            y="option",
            color=split,
            color_discrete_map=_PHASE_COLORS if split == "phase" else None,
            orientation="h",
            barmode="group",
            labels={
                "score": "Rank-weighted picks",
                "option": "",
                "phase": "Evidence",
                "group_label": "Group",
            },
            title=_BLOCK_TITLES.get(block, block) + (f" — {task}" if task else ""),
        )
        fig.update_layout(
            height=120 + 46 * agg["option"].nunique(),
            yaxis=dict(autorange="reversed"),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig, key=f"sa_rank_{block}_{task}")
    st.divider()

# ── Were the rankings right? ─────────────────────────────────────────────────
# The section above shows how the rankings compare to each other. This one
# compares them to what actually failed: every failed episode of the study's
# rollouts carries a hand-written failure mode, in the study's own wordings, so
# the two join on the string (see user_study.failure_scoring).


@st.cache_data(show_spinner=False)
def _ground_truth() -> tuple[dict, dict]:
    """The annotated failures and the option banks, per task.

    Cached on nothing: both are read from disk, and an annotation written while
    the page is open should show up on the next reload rather than never. The
    cost is a hundred small JSON reads, so "next reload" is the right grain.
    """
    return failure_scoring.load_ground_truth(), failure_scoring.question_banks()


_truth, _banks = _ground_truth()
_scores = pd.DataFrame(failure_scoring.score_rows(responses, _truth, _banks))
if not _scores.empty:
    st.subheader("How well the failures were read")
    st.caption(
        "The rankings above, against the failure mode annotated on every failed "
        "episode of the rollouts each task showed. **Top-1** is whether rank 1 "
        "named the most common real failure. **Coverage** is the share of the "
        "failed episodes the participant's picks account for — what the ranking "
        "is actually for — with **achievable** the best any pick of that many "
        "modes could do and **chance** what picking at random would give. "
        "**Order τ** is Kendall's tau between their order and the true "
        "frequencies, over the modes they picked."
    )

    _summary = pd.DataFrame(failure_scoring.summarise(_scores.to_dict("records")))
    _shown = _summary.assign(
        top1_hit_rate=100 * _summary["top1_hit_rate"],
        coverage=100 * _summary["coverage"],
        achievable=100 * _summary["achievable"],
        coverage_ratio=100 * _summary["coverage_ratio"],
        chance=100 * _summary["chance"],
    ).rename(
        columns={
            "task": "Task",
            "phase": "Evidence",
            "n": "Rankings",
            "top1_hit_rate": "Top-1",
            "coverage": "Coverage",
            "achievable": "Achievable",
            "coverage_ratio": "Of achievable",
            "chance": "Chance",
            "order_tau": "Order τ",
            "order_tau_n": "τ from",
        }
    )
    st.dataframe(
        _shown,
        hide_index=True,
        use_container_width=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.0f%%")
            for name in ("Top-1", "Coverage", "Achievable", "Of achievable", "Chance")
        }
        | {"Order τ": st.column_config.NumberColumn(format="%.2f")},
    )

    # Coverage against its own two reference lines, because the number alone is
    # not readable: 89% is the ceiling on cooking and a near-miss on set table.
    _long = _scores.melt(
        id_vars=["task", "phase"],
        value_vars=["coverage", "chance"],
        var_name="measure",
        value_name="share",
    )
    _fig = px.box(
        _long[_long["measure"] == "coverage"],
        x="task",
        y="share",
        color="phase",
        points="all",
        color_discrete_map=_PHASE_COLORS,
        labels={"share": "Coverage of the real failures", "task": "", "phase": "Evidence"},
        title="Coverage per task and evidence, with the achievable ceiling",
    )
    for _task, _counts in _truth.items():
        if _task not in set(_scores["task"]):
            continue
        _ceiling = _scores.loc[_scores["task"] == _task, "achievable"].max()
        _fig.add_hline(
            y=float(_ceiling),
            line_dash="dot",
            line_color="#6b6b6b",
            annotation_text=f"{_task} ceiling",
            annotation_position="top left",
        )
    _fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.05]), **_CHART_LAYOUT)
    st.plotly_chart(_fig, key="sa_failure_coverage")

    # A task where one mode is nearly every failure has almost nothing for a
    # ranking to discriminate, and a high score there is a giveaway rather than
    # a reading. Said here rather than left for a reader to infer.
    _lopsided = {
        task: failure_scoring.dominant_share(counts)
        for task, counts in _truth.items()
        if failure_scoring.dominant_share(counts) >= 0.9 and task in set(_scores["task"])
    }
    if _lopsided:
        st.warning(
            "Little to discriminate in "
            + ", ".join(f"**{task}** ({share:.0%} one mode)" for task, share in _lopsided.items())
            + ". Rank 1 is nearly forced there and the ranks below it carry no signal, "
            "so read those scores as a floor rather than as a result."
        )
    with st.expander("Per ranking"):
        st.dataframe(
            _scores.sort_values(["task", "phase", "coverage"], ascending=[True, True, False]),
            hide_index=True,
            use_container_width=True,
        )
    st.divider()
elif _truth:
    st.caption(
        "No failure-mode rankings in this selection to score against the "
        f"{sum(len(counts) for counts in _truth.values())} annotated failure modes."
    )

# ── Single-choice answers ─────────────────────────────────────────────────────

choices = pd.DataFrame(analytics.choice_rows(responses))
if not choices.empty:
    st.subheader("Data-collection choices")
    st.caption(
        "Which additional data a participant would collect, per modality. Each "
        "task offers its own options, so a bar only ever competes with the other "
        "options for its own task."
    )
    # Not grouped by task: the option text identifies the task already, and a
    # (choice, phase) pair split across two tasks would draw two bars at one
    # categorical position — plotly stacks them into a single offset group, so
    # the smaller one disappears behind the larger rather than appearing beside
    # it. Two layouts that asked the same task the same question pool here,
    # which is what this panel did before the three-stage layout existed.
    agg = choices.groupby(["choice", "phase"]).size().reset_index(name="times_chosen")
    fig = px.bar(
        agg,
        x="times_chosen",
        y="choice",
        color="phase",
        color_discrete_map=_PHASE_COLORS,
        orientation="h",
        barmode="group",
        labels={"times_chosen": "Times chosen", "choice": "", "phase": "Evidence"},
        title="Choices by modality",
    )
    fig.update_layout(
        height=120 + 46 * agg["choice"].nunique(),
        yaxis=dict(autorange="reversed"),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, key="sa_choices")
    st.divider()

# ── What those choices cost ───────────────────────────────────────────────────
# The same answers as the panel above, scored: every option of every task was
# collected, the policy was retrained on it, and the retrained policy was rolled
# out 60 times, so the question has a measured right answer. Regret is the
# distance to it in success-rate points. See
# :mod:`behavior_graphs.viz.streamlit.user_study.regret` for the option-to-policy
# mapping and for why an option's arm being unmeasured drops the whole task
# rather than being scored against the arms that are.

st.subheader("Data-collection outcome")


@st.cache_data(ttl=60, show_spinner="Reading retrained-policy outcomes…")
def _arm_rates() -> dict[str, regret.SuccessRate]:
    """Success rates of the retrained policies, read off the rollout store.

    Short TTL rather than none: these outcomes are hand-annotated real-robot
    rollouts, and a correction to a label should reach this panel without a
    restart. The per-dataset read underneath is itself cached on the metadata
    file's mtime, so an expiry that finds nothing changed costs a stat call.
    """
    return regret.measured_success_rates()


arm_rates = _arm_rates()
regrets = pd.DataFrame(regret.regret_rows(responses, arm_rates))
unscored = regret.unscored_datasets(arm_rates)

if regrets.empty:
    if unscored:
        # The likeliest cause on a deployment: the retrained collections are not
        # mounted. Named, because "no regret to show" over a store full of
        # answered choices otherwise reads as the scoring being broken.
        st.info(
            "No choice can be scored yet: these retrained policies have no "
            "readable rollouts — "
            + "; ".join(
                f"**{task}** ({', '.join(datasets)})" for task, datasets in unscored.items()
            )
            + ". Every option of a task needs a measured success rate before that "
            "task's best option is known."
        )
    else:
        st.info(
            "These responses recorded no data-collection choice this page can "
            "score against a retrained policy."
        )
else:
    st.caption(
        "Each option the study offers was collected, the policy was retrained on "
        "it, and the retrained policy was rolled out 60 times. **Regret** is the "
        "success rate of the task's best option minus that of the option the "
        "participant picked, in success-rate points — 0 for the best choice, and "
        "the performance left on the table otherwise. Lower is a better decision."
    )
    if unscored:
        st.caption(
            "Left out: "
            + "; ".join(f"{task} ({', '.join(datasets)})" for task, datasets in unscored.items())
            + " — a task with an unmeasured option has no known best option."
        )

    # Ordered by the study's stages rather than alphabetically, and cut to the
    # tasks these responses actually answered — a phantom category draws an
    # empty slot that reads as a task nobody scored.
    answered = set(regrets["task_label"])
    task_labels = [task.label for task in regret.TASK_ARMS if task.label in answered]
    # Regret, which is `regret.DECISION_METRIC`: the success rate the chosen
    # option left on the table against the best one available, so LOWER IS
    # BETTER and zero is having picked the best.
    #
    # Gain over the baseline was the headline here for a while and reads more
    # easily on its own -- higher is better, and it can say an option made the
    # policy WORSE, which regret can only score as far from best. It is a worse
    # measure of the decision, though, because it carries the task's headroom:
    # +28.3 points were available on prepare tray and +3.3 on cooking. Gain is
    # still on every row, in the hover, and in "beat the baseline" below.
    fig = px.box(
        regrets,
        x="task_label",
        y=regret.DECISION_METRIC,
        color="phase",
        color_discrete_map=_PHASE_COLORS,
        points="all",
        category_orders={"task_label": task_labels, "phase": list(regret.PHASE_ORDER)},
        labels={
            regret.DECISION_METRIC: "Regret (success-rate points, lower is better)",
            "task_label": "",
            "phase": "Evidence",
        },
        title="Regret of the data-collection choice, by task and display",
        hover_data=[
            "participant",
            "choice_label",
            "success_rate",
            "baseline_rate",
            "regret",
            "best_label",
            "best_rate",
        ],
    )
    # Zero is the optimum here rather than a neutral point -- no regret is
    # having picked the best option -- so it is a line rather than something a
    # reader infers from the axis.
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#7f7f7f")
    fig.update_layout(height=420, boxmode="group", **_CHART_LAYOUT)
    st.plotly_chart(fig, key="sa_decision_regret")

    summary = pd.DataFrame(regret.summary_rows(regrets.to_dict("records")))
    st.caption(
        "Raw points first, then the two normalizations, because points do not "
        "pool: one arm set spans 33 success-rate points and another 8, so a "
        "pooled mean of points is largely a mean over the widest task. "
        "**Normalized** is the distance to the best option as a share of the "
        "spread between best and worst — bounded, poolable, and to be read "
        "against **chance**, which is what picking an option at random would "
        "score and is *not* a half. **Of gain** is the same distance as a share "
        "of what choosing well was worth (best option minus collecting "
        "nothing): above 100% means the choice left the policy worse than "
        "collecting no data at all, which the bounded score cannot say. It is "
        "blank for a task whose best option is not separated from its baseline "
        "— there is no measured gain to take a share of, and dividing by a few "
        "points of noise produces a large number that means nothing."
    )
    _no_gain = [
        task.label
        for task in regret.scoreable(arm_rates)
        if regret.gain_over_baseline(task, arm_rates) is not None
        and regret.measurable_gain(task, arm_rates) is None
    ]
    if _no_gain:
        st.info(
            "No measured gain to score against in "
            + ", ".join(f"**{label}**" for label in _no_gain)
            + ": the best option is not separated from collecting nothing "
            f"(α={regret.GAIN_ALPHA:g}), so **of gain** is left blank there. The "
            "normalized column still scores those choices — being the best of "
            "several options that all did nothing is still a better read than "
            "picking the worst."
        )
    st.dataframe(
        summary.assign(
            mean_regret=summary["mean_regret"].round(1),
            median_regret=summary["median_regret"].round(1),
            mean_normalized=100 * summary["mean_normalized"],
            chance_normalized=100 * summary["chance_normalized"],
            mean_of_gain=100 * summary["mean_of_gain"],
            chose_best_rate=summary["chose_best_rate"].round(0),
        )[
            [
                "task_label",
                "phase",
                "answers",
                "mean_regret",
                "median_regret",
                "mean_normalized",
                "chance_normalized",
                "mean_of_gain",
                "worse_than_baseline_rate",
                "chose_best",
                "chose_best_rate",
            ]
        ].rename(
            columns={
                "task_label": "Task",
                "phase": "Display",
                "answers": "Choices",
                "mean_regret": "Mean regret (pts)",
                "median_regret": "Median regret (pts)",
                "mean_normalized": "Normalized",
                "chance_normalized": "Chance",
                "mean_of_gain": "Of gain",
                "worse_than_baseline_rate": "Worse than nothing",
                "chose_best": "Picked the best",
                "chose_best_rate": "Picked the best (%)",
            }
        ),
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.0f%%")
            for name in (
                "Normalized",
                "Chance",
                "Of gain",
                "Worse than nothing",
                "Picked the best (%)",
            )
        },
    )

    # Normalized rather than points, and against its own chance line: this is
    # the version of the chart above that can be read across tasks.
    _norm = regrets[regrets["regret_normalized"].notna()]
    if not _norm.empty:
        _nfig = px.box(
            _norm,
            x="task_label",
            y="regret_normalized",
            color="phase",
            color_discrete_map=_PHASE_COLORS,
            points="all",
            category_orders={"task_label": task_labels, "phase": list(regret.PHASE_ORDER)},
            labels={
                "regret_normalized": "Normalized regret (0 = best option, 1 = worst)",
                "task_label": "",
                "phase": "Evidence",
            },
            title="Normalized regret, with each task's chance line",
            hover_data=["participant", "choice_label", "regret", "regret_of_gain"],
        )
        for _task_label in task_labels:
            _chance = _norm.loc[_norm["task_label"] == _task_label, "chance_normalized"].mean()
            if pd.notna(_chance):
                _nfig.add_annotation(
                    x=_task_label,
                    y=float(_chance),
                    text=f"chance {_chance:.2f}",
                    showarrow=False,
                    yshift=8,
                    font=dict(size=10, color="#6b6b6b"),
                )
        _nfig.update_layout(height=420, boxmode="group", **_CHART_LAYOUT)
        st.plotly_chart(_nfig, key="sa_regret_normalized")

    with st.expander("The retrained policies these regrets are measured against"):
        st.caption(
            "One row per option, plus the baseline policy whose rollouts the "
            "participant actually read. The baseline is not an option — nobody "
            'was offered "collect nothing" — so it is never the best option '
            "here even where it beats every arm, which is the case for one of "
            "these tasks."
        )
        arms = pd.DataFrame(regret.arm_rows(arm_rates))
        st.dataframe(
            arms.assign(
                success_rate=(100 * arms["success_rate"]).round(1),
                is_best=arms["is_best"].map({True: "★", False: ""}),
            )[
                ["task_label", "label", "dataset", "successes", "trials", "success_rate", "is_best"]
            ].rename(
                columns={
                    "task_label": "Task",
                    "label": "Option",
                    "dataset": "Rollout dataset",
                    "successes": "Successes",
                    "trials": "Scored rollouts",
                    "success_rate": "Success rate (%)",
                    "is_best": "Best",
                }
            ),
            hide_index=True,
        )

    # ── The regret figure the paper carries ───────────────────────────────────

    st.markdown("**Paper figure — regret**")
    st.caption(
        "The panel above as a paper-ready PDF, in the same style as the study "
        "figure below: every decision made off each display as a box and a dot, "
        "pooled over the tasks, split by task, or both in one figure. Drawn "
        "from the responses and filters in view, and captionless on purpose — "
        "the caption belongs in the LaTeX."
    )
    # Restricted to the assigned stages by default, which the panel above is
    # NOT: that chart is the page exploring every scored answer, and this is the
    # file the paper takes. Every other figure the paper carries is drawn on the
    # randomized stages, and an export that quietly included the stage where the
    # participant picked their own display would be the one figure in the paper
    # covering different evidence from its neighbours.
    regret_randomized = st.checkbox(
        "Randomized stages only",
        value=True,
        key="sa_regret_figure_randomized",
        help=(
            "Count only the stages whose display was **assigned**. The third "
            "stage let the participant choose, and 16 of 18 chose the graph, so "
            "its answers describe the people who reached for the graph rather "
            "than what the graph led people to — pooling them in reverses which "
            "display led to the better decisions. Off draws every scored "
            "answer, which is what the chart above this shows."
        ),
    )
    regret_mode = st.radio(
        "What the figure shows",
        options=list(figure_export.REGRET_FIGURE_MODES),
        format_func=lambda mode: figure_export.REGRET_FIGURE_MODES[mode],
        horizontal=True,
        key="sa_regret_figure_mode",
    )
    pooled_mark = st.radio(
        "How the pooled panel draws it",
        options=list(figure_export.REGRET_MARKS),
        format_func=lambda mark: figure_export.REGRET_MARKS[mark],
        horizontal=True,
        key="sa_regret_pooled_mark",
        help=(
            "The box shows the answers: regret takes one of three values per "
            "task, so they pile up at 0 and at the cost of each wrong option, "
            "and an error bar around their average implies a spread between "
            "the piles that nobody could have answered. The mean reads faster "
            "and prints the number the paper's sentences quote. The per-task "
            "panels are distributions either way."
        ),
    )

    @st.cache_data(show_spinner="Rendering figure…")
    def _regret_figure(
        _rows: list[dict], fingerprint: tuple, mode: str, mark: str, randomized: bool
    ) -> dict[str, bytes] | None:
        """The regret figure's PDF and preview PNG, memoized on *fingerprint*.

        The rows are passed unhashed (the leading underscore) and keyed on the
        responses they were scored from plus the mode, the mark and the
        restriction, exactly as the study figure below is: hashing a few hundred
        row dicts on every rerun costs more than the figure does. The
        restriction is in the key as well as in the rows — it changes the figure
        without changing which responses it came from.
        """
        return figure_export.regret_figure_files(
            regret.randomized_rows(_rows) if randomized else _rows, mode, pooled_mark=mark
        )

    _scored = regrets.to_dict("records")
    regret_files = _regret_figure(
        _scored, _responses_key, regret_mode, pooled_mark, regret_randomized
    )
    _dropped = len(_scored) - len(regret.randomized_rows(_scored)) if regret_randomized else 0
    if regret_files is None:
        st.info(
            "Nothing to draw: no scored choice from a randomized stage names the "
            "display it was made off."
            if regret_randomized
            else "Nothing to draw: no scored choice names the display it was made off."
        )
    else:
        if _dropped:
            # Said out loud, because the chart above this one is drawn from the
            # rows this figure leaves out: a reader comparing a box here with
            # the box there needs to know why they differ.
            st.caption(
                f"Drawn from **{len(_scored) - _dropped}** of {len(_scored)} scored "
                f"answers: {_dropped} came from the stage whose display the "
                "participant chose and are left out. The chart above includes them."
            )
        st.download_button(
            "Download regret figure (PDF)",
            data=regret_files["pdf"],
            file_name=(
                f"study_regret_{regret_mode}_{pooled_mark}"
                f"{'_randomized' if regret_randomized else '_all_stages'}_"
                f"{'all_versions' if version == _ALL else version}.pdf"
            ),
            mime="application/pdf",
            help=(
                "Vector PDF with TrueType-embedded 8 pt type, at IEEE text width "
                "— one column where the figure is a single panel, both columns "
                "where it has several. Drop it in at width=\\linewidth, unscaled."
            ),
        )
        with st.expander("Preview the regret figure"):
            st.image(regret_files["png"], width="stretch")

    with st.expander("Every scored choice"):
        st.dataframe(
            regrets[
                [
                    "participant",
                    "task_label",
                    "phase",
                    "choice_label",
                    "success_rate",
                    "best_label",
                    "best_rate",
                    "regret",
                    "gain_over_baseline",
                ]
            ].rename(
                columns={
                    "participant": "Participant",
                    "task_label": "Task",
                    "phase": "Display",
                    "choice_label": "Chose",
                    "success_rate": "Chosen success rate (%)",
                    "best_label": "Best option",
                    "best_rate": "Best success rate (%)",
                    "regret": "Regret (pts)",
                    "gain_over_baseline": "Gain over baseline (pts)",
                }
            ),
            hide_index=True,
        )

st.divider()

# ── Numeric agree/disagree blocks ─────────────────────────────────────────────

ratings = pd.DataFrame(analytics.rating_rows(responses))
if not ratings.empty:
    st.subheader("Rating scales")
    for block, rows in ratings.groupby("block"):
        per_phase = rows["phase"].ne("").any()
        fig = px.box(
            rows,
            x="question",
            y="score",
            color="phase" if per_phase else "group_label",
            color_discrete_map=_PHASE_COLORS if per_phase else None,
            points="all",
            labels={
                "score": "Rating",
                "question": "",
                "phase": "Evidence",
                "group_label": "Group",
            },
            title=block,
        )
        fig.update_layout(height=400, **_CHART_LAYOUT)
        st.plotly_chart(fig, key=f"sa_rating_{block}")
    st.divider()

# ── Budget allocations (production layout) ────────────────────────────────────

allocations = pd.DataFrame(analytics.allocation_rows(responses))
if not allocations.empty:
    st.subheader("Data allocation choices")
    st.caption("How participants distributed their demo-collection budget across strategies.")
    fig = px.box(
        allocations,
        x="strategy",
        y="allocation",
        color="group_label",
        points="all",
        labels={"allocation": "Demos allocated", "strategy": "Strategy", "group_label": "Group"},
        title="Allocation by strategy and group",
    )
    fig.update_layout(height=400, **_CHART_LAYOUT)
    st.plotly_chart(fig, key="sa_alloc")
    st.divider()

# ── Time on each step ─────────────────────────────────────────────────────────

durations = pd.DataFrame(analytics.duration_rows(responses))
if not durations.empty:
    st.subheader("Time per step")
    st.caption(
        "Named from the layout the response came from, in the order the "
        "participant met them — so two versions' steps only line up where they "
        "happen to share a name."
    )
    durations["minutes"] = durations["seconds"] / 60
    # Ordered by index rather than by name: "Intermission" before "Rollout
    # videos + questions" is the sequence, and alphabetical is not.
    step_order = (
        durations[["step_index", "step"]]
        .drop_duplicates()
        .sort_values("step_index")["step"]
        .tolist()
    )
    fig = px.box(
        durations,
        x="step",
        y="minutes",
        color="group_label",
        points="all",
        category_orders={"step": step_order},
        labels={"minutes": "Minutes", "step": "Step", "group_label": "Group"},
        title="Minutes spent per step",
    )
    fig.update_layout(height=380, **_CHART_LAYOUT)
    st.plotly_chart(fig, key="sa_durations")
    st.divider()

# ── The figure the paper carries ──────────────────────────────────────────────
# Built from the same filtered responses every panel above is drawn from, so a
# lab preview or an abandoned session left out of the charts is left out of the
# export too — and rebuilt whenever those responses change, which is what makes
# the downloaded file the current data rather than the first data of the session.

st.subheader("Paper figure")
st.caption(
    "Chosen display, the adjusted graph--video rating effects, and the same "
    "contrast on data-collection regret, as one IEEE double-column PDF. The "
    "intervals come from the mixed-effects models used in the paper; raw "
    "response distributions remain above as descriptive diagnostics. "
    "Deliberately captionless: the caption belongs in the LaTeX. Use Refresh "
    "at the top of the page to re-read the store first."
)

# The raw responses, as their own export. No model, so no toggle: it costs a
# matplotlib render and it is the figure a reader believes before they believe
# any estimate -- which is exactly why it is restricted to the randomized rows
# the estimate is fit on. Including the self-chosen third stage here would put
# a third more data in the boxes than in the interval beside them, and a reader
# who checks the two against each other would find a discrepancy nobody can
# explain.
st.markdown("**Raw responses (randomized stages)**")
# The same choice the regret figure above offers, for the same panel: this
# figure's last one is that panel at a fifth of the width.
_raw_mark = st.radio(
    "How the regret panel draws it",
    options=list(figure_export.REGRET_MARKS),
    format_func=lambda mark: figure_export.REGRET_MARKS[mark],
    horizontal=True,
    key="sa_raw_regret_mark",
    help=(
        "The box is the answers themselves and matches the rating panels "
        "beside it; the mean bar is the faster read and the number the paper "
        "quotes. It changes nothing else in the figure."
    ),
)
_raw_files = figure_export.study_figure_files(
    responses,
    regrets.to_dict("records") if not regrets.empty else (),
    randomized_only=True,
    regret_mark=_raw_mark,
)
if _raw_files is None:
    st.info("No rated randomized stage in these responses, so there is nothing to draw.")
else:
    st.image(_raw_files["png"], width="stretch")
    st.download_button(
        "Download raw-responses figure (PDF)",
        data=_raw_files["pdf"],
        file_name=f"study_raw_{'all_versions' if version == _ALL else version}.pdf",
        mime="application/pdf",
        key="sa_raw_fig_pdf",
        help="Same figure as the preview, as the vector PDF the paper takes.",
    )
    st.caption(
        "Every rating from a stage whose display was **assigned**, which is the "
        "evidence the adjusted models are fit on — the third stage let the "
        "participant pick, so it answers a preference question rather than a "
        "randomized one and is not in here. `n` under each box is the count it "
        "is drawn from."
    )
st.divider()

# Behind a toggle, unlike every panel above: this is four Monte-Carlo chains per
# outcome and the page renders the whole study beneath it. Opening it is the
# consent to wait; a rerun of a filter above should not silently re-pay for it.
if lazy_section(
    "Fit the adjusted models and draw the figure",
    key="sa_paper_figure",
    help=(
        "Fits the Bayesian mixed models the figure's intervals come from — "
        "seconds to a minute on a full store, then cached until the responses "
        "or the retrained-policy outcomes change."
    ),
):
    # Keyed on the arms too, not just the responses: correcting a rollout's
    # outcome label on disk moves the regret the model is fit to, and the
    # figure should follow rather than serve the posterior of the old labels.
    _rates_key = tuple(
        sorted((name, rate.successes, rate.trials) for name, rate in arm_rates.items())
    )

    @st.cache_data(show_spinner="Fitting the adjusted rating models…")
    def _rating_effects(_responses: list[dict], fingerprint: tuple) -> list[dict]:
        """The three fitted rating effects, memoized on *fingerprint*.

        Not keyed on the figure's mode: both modes are read off this same
        posterior, so keying it there would refit three models to redraw a
        panel that is already computed.
        """
        return mixed_effects.continuous_effects(_responses)

    @st.cache_data(show_spinner="Fitting the adjusted decision model…")
    def _fit_regret_effect(_responses: list[dict], _rates: dict, fingerprint: tuple) -> dict | None:
        """The fitted graph--video decision-regret effect, memoized on *fingerprint*."""
        return mixed_effects.decision_regret_effect(_responses, _rates)

    paper_figure_mode = st.radio(
        "Rating display",
        options=list(figure_export.ADJUSTED_FIGURE_MODES),
        format_func=lambda mode: figure_export.ADJUSTED_FIGURE_MODES[mode],
        horizontal=True,
        key="sa_paper_figure_mode",
        help=(
            "Both views use the same mixed-effects model. The first shows the direct "
            "graph--video effect; the second shows adjusted population-average ratings."
        ),
    )

    try:
        effects = _rating_effects(responses, _responses_key)
    except RuntimeError as error:
        st.warning(str(error))
        effects = []

    # The decision panel is the one part of this figure that can be missing on
    # a working deployment — it needs every option's retrained rollouts on disk
    # — so its absence costs the panel and not the figure.
    regret_effect = None
    if effects and not regrets.empty:
        try:
            regret_effect = _fit_regret_effect(responses, arm_rates, (_responses_key, _rates_key))
        except RuntimeError as error:
            st.warning(f"{error} Drawing the figure without its decision panel.")

    @st.cache_data(show_spinner="Rendering figure…")
    def _paper_figure(
        _responses: list[dict],
        _effects: list[dict],
        _regret_effect: dict | None,
        fingerprint: tuple,
        mode: str,
    ) -> dict[str, bytes] | None:
        """The export's PDF and preview PNG, memoized on *fingerprint* and *mode*.

        The responses and the fitted effects are passed unhashed (the leading
        underscores) and keyed on the fingerprint below instead: hashing the
        responses and four posteriors' summaries on every rerun costs more
        than the figure does.
        """
        return figure_export.adjusted_study_figure_files(
            _responses, _effects, regret_effect=_regret_effect, mode=mode
        )

    figure_files = _paper_figure(
        responses,
        effects,
        regret_effect,
        (_responses_key, _rates_key, regret_effect is not None),
        paper_figure_mode,
    )
    if figure_files is None:
        st.info(
            "Nothing to draw for this selection: the adjusted figure needs "
            "completed randomized three-stage responses."
        )
    else:
        st.download_button(
            "Download paper figure (PDF)",
            data=figure_files["pdf"],
            file_name=f"study_figure_{'all_versions' if version == _ALL else version}.pdf",
            mime="application/pdf",
            type="primary",
            help=(
                "Vector PDF, 7.16 in wide with TrueType-embedded 8 pt type — drop "
                "it into a figure* environment at width=\textwidth, unscaled."
            ),
        )
        with st.expander("Preview the figure"):
            st.image(figure_files["png"], width="stretch")

st.divider()

# ── Remarks and the raw table ─────────────────────────────────────────────────

exits = pd.DataFrame(analytics.exit_answer_rows(responses))
if not exits.empty:
    st.subheader("Closing questions")

    answered = exits[exits["experience"].ne("")]
    if answered.empty:
        st.caption("This layout does not ask about prior experience.")
    else:
        st.caption(
            "How much experience the participant reported with collecting data "
            "for robot learning — the one answer that plausibly explains two "
            "participants reading the same graph differently."
        )
        counts_by_answer = (
            answered.groupby(["experience", "group_label"]).size().reset_index(name="participants")
        )
        fig = px.bar(
            counts_by_answer,
            x="experience",
            y="participants",
            color="group_label",
            barmode="stack",
            # Weakest first, not alphabetical: it is a scale.
            category_orders={"experience": list(analytics.EXPERIENCE_SCALE)},
            labels={
                "experience": "Prior experience",
                "participants": "Participants",
                "group_label": "Group",
            },
            title="Reported experience with robot data collection",
        )
        fig.update_layout(height=340, **_CHART_LAYOUT)
        st.plotly_chart(fig, key="sa_experience")

    wrote = exits[exits["remarks"].str.strip().ne("")]
    if wrote.empty:
        st.caption("No remarks written yet.")
    else:
        st.markdown("**Remarks**")
        for row in wrote.itertuples():
            st.markdown(f"- **{row.participant}** ({row.group_label}) — {row.remarks}")
    st.divider()

with st.expander("Every response (table)"):
    st.dataframe(
        people.drop(columns=["remarks"]),
        hide_index=True,
    )
