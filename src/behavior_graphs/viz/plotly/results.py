"""Interactive figures over the paper's exported results.

The matplotlib pipeline in :mod:`behavior_graphs.viz.paper` draws the figures a
paper carries: fixed width, embedded fonts, byte-stable. These are the same
results for a browser, where the reader can hover a violin to read its counts
and switch the metric a chart is drawn on. Neither is a re-styling of the other
— a figure sized for one column of IEEE and a figure sized for a screen want
different things — but they are drawn from the same numbers, which is what
:mod:`behavior_graphs.results` is for.

No Streamlit imports, as everywhere else in this package: every function takes
the exported rows and returns a ``plotly.graph_objects.Figure`` or a frame, so
the charts can be asserted on without a page around them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: The study's two displays and their colours, graph first. The same pair the
#: paper figures use (``user_study.figure_export.PHASE_COLORS``), repeated here
#: rather than imported because that module pulls in matplotlib and the whole
#: study-analysis stack for two hex strings. ``test_results_charts`` asserts the
#: two agree, so the website and the paper cannot come to colour the displays
#: differently.
DISPLAY_COLORS = {"graph": "#ff7f0e", "videos": "#1f77b4"}

#: What to call each display on an axis.
DISPLAY_LABELS = {"graph": "Graph", "videos": "Videos"}

#: Order every panel puts them in: the graph is what the study is about, so it
#: is on the left.
DISPLAY_ORDER = ("graph", "videos")

#: The comparison figure's two kinds of policy. The baseline is the one every
#: panel has and the one every arm is read against, so it is the colour that
#: repeats across panels; the retrained arms share the other, since what
#: separates them is their letter and not their hue.
BASELINE_COLOR = "#8c8c8c"
RETRAINED_COLOR = "#4c78a8"

#: One colour per progress-benchmark domain, so a point keeps its meaning
#: across the metric radio.
DOMAIN_COLORS = {
    "LIBERO": "#4c78a8",
    "Real-world": "#f58518",
    "Robomimic": "#54a24b",
    "MimicGen": "#b279a2",
}

#: The progress estimators: ours in one accent, the baselines in greys of
#: rising lightness.
#:
#: Not a categorical palette: the greys say "comparison" rather than inviting a
#: reader to look for meaning in which baseline got which colour, and the
#: accent is deliberately not one of :data:`DOMAIN_COLORS`.
#:
#: Picked for a **white** page, which is what this site serves. The cyan these
#: were before is low-contrast on white, and the lightest grey was invisible.
OURS_COLOR = "#0b7285"
BASELINE_GREYS = ("#495057", "#868e96", "#adb5bd")

#: And a dash per baseline, because three greys on one chart are three greys.
#: Lightness separates two lines at a glance and not three, so the curve plot
#: carries the dash as well; it also survives the chart being printed. None of
#: them is ``dot``, which belongs to the reference diagonal.
BASELINE_DASHES = ("solid", "dash", "dashdot")

#: Ink for the furniture a reader should not notice: reference diagonals, zero
#: lines, interval whiskers.
_RULE = "rgba(100,100,100,0.55)"

#: A violin's fill, as its line colour at this opacity. Plotly needs the two as
#: separate colours and an opaque fill buries the median marker.
_FILL_ALPHA = 0.35

#: Which way an outcome improves, drawn beside its name. The instrument's own
#: direction (``user_study.analytics.BETTER_DIRECTION``): a reader who takes the
#: load items' arrow for all of them reads the confidence panel backwards.
_ARROW_DOWN = "\u2193"
_ARROW_UP = "\u2191"

#: How much of a category slot is *gap* rather than box. Plotly fills half the
#: slot by default, and a slot is half a panel wide when a panel holds two
#: boxes: the result reads as two blocks of colour rather than as a
#: distribution. A layout key rather than ``Box.width``, which is in data
#: coordinates and is ignored for traces placed by category.
_BOX_GAP = 0.62

#: Plotly's own toolbar, off, for ``st.plotly_chart(..., config=CHART_CONFIG)``.
#:
#: It is drawn at the top right of the figure and over whatever is there, which
#: in a panelled figure is the last panel's title: measured at 1500 px, the
#: toolbar sat on "Cooking" and on "Confidence". Nothing here needs it —
#: hovering is the interaction, and Streamlit's own expand button is the one
#: affordance a reader wants.
CHART_CONFIG = {"displayModeBar": False}


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _layout(
    figure: go.Figure,
    *,
    height: int,
    left: int = 74,
    top: int = 30,
    bottom: int = 40,
    legend_y: float | None = None,
) -> go.Figure:
    """The chrome every figure here shares.

    Transparent paper, because the page's theme owns the background and a
    figure that paints its own sits on the page as a grey slab.

    Every legend goes **below** the axes, and *legend_y* says how far. Above
    them is where :func:`~plotly.subplots.make_subplots` puts its subplot
    titles, and a horizontal legend at ``y=1.02`` lands on top of them in every
    panelled figure here; below the axes there is nothing but tick labels, so
    the only thing to get right is the bottom margin. Consistent placement is
    also one less thing for a reader moving between four pages to relearn.
    """
    figure.update_layout(
        height=height,
        margin={"l": left, "r": 20, "t": top, "b": bottom},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    if legend_y is not None:
        figure.update_layout(legend={"orientation": "h", "yanchor": "top", "y": legend_y, "x": 0})
    return figure


# ── Data collection: the retrained policies, and what the study asked for ────


def success_rate_figure(
    tasks: Sequence[Mapping],
    choices: Sequence[Mapping] = (),
    *,
    prior: Sequence[float] = (1.0, 1.0),
    credible_level: float = 0.95,
    height: int = 620,
) -> go.Figure:
    """The comparison figure: posterior success rates, and the choices under them.

    One column per task. The top row is a violin per policy — the Beta
    posterior over its success rate, drawn from the arm's own counts — with the
    posterior median and credible interval marked and the compact letter
    display above. Two policies sharing a letter are not separated by the
    paper's pairwise test.

    The bottom row is what the study's participants asked to collect for that
    task, as a share of the answers each display drew. It is in the same figure
    and on the same x axis on purpose: the decision and its consequence are
    joined by the policy they are about, so a reader follows one column down
    rather than carrying a policy's name between two figures.

    *choices* may be empty, which drops the bottom row rather than drawing an
    empty one.
    """
    from behavior_graphs.stats.policy_comparison import (
        PolicyOutcomes,
        credible_interval,
        posterior_median,
        posterior_sample_grid,
    )

    prior_pair = (float(prior[0]), float(prior[1]))
    with_choices = bool(choices)
    rows = 2 if with_choices else 1
    figure = make_subplots(
        rows=rows,
        cols=len(tasks),
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=[str(task["label"]) for task in tasks],
        # Wide enough that a four-arm panel is not three arms' worth of ink in
        # a two-arm panel's width.
        column_widths=[len(task["arms"]) for task in tasks],
        row_heights=[0.68, 0.32] if with_choices else None,
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
    )

    # Every arm's posterior up front, because the compact letter display is
    # placed on one shared baseline: a letter sitting just above its own
    # interval lands at a different height in every panel, and on a tall
    # violin it lands inside the next one's ink.
    stats: dict[int, list[tuple]] = {}
    for column, task in enumerate(tasks, start=1):
        stats[column] = []
        for arm in task["arms"]:
            outcomes = PolicyOutcomes(
                name=str(arm["label"]),
                successes=int(arm["successes"]),
                trials=int(arm["trials"]),
            )
            interval = credible_interval(outcomes, credible_level, prior_pair)
            stats[column].append(
                (
                    arm,
                    outcomes,
                    100 * posterior_sample_grid(outcomes, prior_pair),
                    100 * interval[0],
                    100 * interval[1],
                    100 * posterior_median(outcomes, prior_pair),
                )
            )
    letter_y = min(
        104.0,
        max(high for panel in stats.values() for *_, high, _ in panel) + 4,
    )

    for column, task in enumerate(tasks, start=1):
        labels = [str(arm["label"]) for arm in task["arms"]]
        for arm, outcomes, samples, low, high, median in stats[column]:
            colour = BASELINE_COLOR if arm.get("baseline") else RETRAINED_COLOR
            figure.add_trace(
                go.Violin(
                    x=[outcomes.name] * len(samples),
                    y=samples,
                    name=outcomes.name,
                    legendgroup="baseline" if arm.get("baseline") else "retrained",
                    showlegend=False,
                    line_color=colour,
                    fillcolor=_rgba(colour, _FILL_ALPHA),
                    width=0.85,
                    points=False,
                    hoverinfo="skip",
                    spanmode="hard",
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=[outcomes.name],
                    y=[median],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [high - median],
                        "arrayminus": [median - low],
                        "color": _RULE,
                        "width": 0,
                    },
                    mode="markers",
                    marker={"size": 8, "color": colour, "line": {"width": 1, "color": _RULE}},
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{outcomes.name}</b><br>"
                        f"{outcomes.successes} of {outcomes.trials} rollouts succeeded<br>"
                        f"posterior median {median:.1f}%<br>"
                        f"{int(100 * credible_level)}% interval {low:.1f} to {high:.1f}%<br>"
                        f"letters {arm.get('letters') or '—'}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            if arm.get("letters"):
                figure.add_annotation(
                    x=outcomes.name,
                    y=letter_y,
                    text=f"<b>{arm['letters']}</b>",
                    showarrow=False,
                    yanchor="bottom",
                    font={"size": 13},
                    row=1,
                    col=column,
                )

        figure.update_yaxes(
            title_text="Success rate (%)" if column == 1 else None,
            # Headroom for the shared letter baseline, which on a task whose
            # best arm is near ceiling sits above 100.
            range=[0, max(108.0, letter_y + 8)],
            row=1,
            col=column,
        )
        # Angled rather than left to Plotly's auto-rotation, which decides per
        # render width: "plus milk middle" fits in a four-arm panel on a wide
        # screen and does not on a laptop, and the figure should not change
        # shape between the two. Set on every row, since the shared axis shows
        # its labels on the bottom one.
        for row_index in range(1, rows + 1):
            figure.update_xaxes(
                categoryorder="array",
                categoryarray=labels,
                tickangle=-25,
                row=row_index,
                col=column,
            )

        if not with_choices:
            continue

        for display in DISPLAY_ORDER:
            # Joined on the policy's dataset rather than on the task: the two
            # exports name a task differently ("Set table" against "Set the
            # table"), and a dataset is what both rows are actually about.
            picked = {str(row["dataset"]): row for row in choices if row["display"] == display}
            if not any(str(arm["dataset"]) in picked for arm in task["arms"]):
                continue
            bars = [picked.get(str(arm["dataset"])) for arm in task["arms"]]
            figure.add_trace(
                go.Bar(
                    x=labels,
                    # No bar where a policy was not an option: nobody was
                    # offered "collect nothing", and a policy nobody could pick
                    # is not a policy nobody picked.
                    y=[
                        None if row is None else 100 * row["picks"] / row["answers"] for row in bars
                    ],
                    name=DISPLAY_LABELS[display],
                    legendgroup=display,
                    showlegend=column == 1,
                    marker_color=DISPLAY_COLORS[display],
                    customdata=[
                        [0, 0] if row is None else [row["picks"], row["answers"]] for row in bars
                    ],
                    # The count over the bar, because an option nobody picked
                    # draws no bar at all and would otherwise be
                    # indistinguishable from the baseline, which was never an
                    # option. A "0" says which of the two it is.
                    text=["" if row is None else str(row["picks"]) for row in bars],
                    textposition="outside",
                    textfont={"size": 10},
                    cliponaxis=False,
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>{DISPLAY_LABELS[display]}: "
                        "%{customdata[0]} of %{customdata[1]} answers<extra></extra>"
                    ),
                ),
                row=2,
                col=column,
            )
        figure.update_yaxes(
            title_text="Asked for (%)" if column == 1 else None,
            range=[0, 100],
            row=2,
            col=column,
        )

    figure.update_layout(barmode="group", bargap=0.35, violinmode="overlay")
    # Room under the axes for the policy labels and the legend beneath them.
    return _layout(figure, height=height, top=40, bottom=130, legend_y=-0.20)


def policy_table(
    task: Mapping, *, credible_level: float = 0.95, prior: Sequence[float] = (1.0, 1.0)
):
    """One task's arms as a frame: counts, rate, posterior interval, letters."""
    import pandas as pd

    from behavior_graphs.stats.policy_comparison import (
        PolicyOutcomes,
        credible_interval,
        posterior_median,
    )

    prior_pair = (float(prior[0]), float(prior[1]))
    rows = []
    for arm in task["arms"]:
        outcomes = PolicyOutcomes(
            name=str(arm["label"]), successes=int(arm["successes"]), trials=int(arm["trials"])
        )
        low, high = (100 * v for v in credible_interval(outcomes, credible_level, prior_pair))
        rows.append(
            {
                "Policy": outcomes.name,
                "Successes": outcomes.successes,
                "Rollouts": outcomes.trials,
                "Success rate (%)": round(100 * outcomes.rate, 1),
                "Posterior median (%)": round(100 * posterior_median(outcomes, prior_pair), 1),
                f"{int(100 * credible_level)}% interval": f"{low:.1f} to {high:.1f}",
                "Letters": arm.get("letters") or "",
            }
        )
    return pd.DataFrame(rows)


# ── The user study ───────────────────────────────────────────────────────────


def study_answers_figure(
    preference: Mapping[str, int],
    ratings: Sequence[Mapping],
    outcomes: Sequence[Mapping],
    regret: Sequence[Mapping],
    *,
    height: int = 420,
) -> go.Figure:
    """The study's answers, as the panels the paper's figure carries.

    One row: which display the participants chose when nobody assigned them
    one, then a box and every dot per display for each rated outcome, then the
    regret their data-collection choices turned out to be worth. The panels,
    their order and their arrows are the paper's
    (``user_study.figure_export.study_panels``), so a reader who has the pdf
    open is looking at the same figure rather than at a rearrangement of it.

    Points as well as boxes because the answers are on a six-point scale: a box
    over 18 ratings that take six values hides how many sit on one of them.
    Each panel keeps its own y axis -- a count, a rating and a success rate are
    three units, and one axis across them would be three claims at once.
    """
    titles = ["(a) Chosen display"]
    titles += [f"({chr(98 + i)}) {_outcome_title(o)}" for i, o in enumerate(outcomes)]
    titles.append(f"(e) Success rate regret {_ARROW_DOWN}")
    figure = make_subplots(
        rows=1,
        cols=len(titles),
        # The preference panel holds two bars against two boxes in each of the
        # others, so it needs less room; the paper's figure sizes it the same
        # way (`study_panel_widths`).
        column_widths=[0.6] + [1.0] * (len(titles) - 1),
        horizontal_spacing=0.045,
    )

    total = sum(int(preference.get(display, 0)) for display in DISPLAY_ORDER)
    for display in DISPLAY_ORDER:
        value = int(preference.get(display, 0))
        figure.add_trace(
            go.Bar(
                x=[DISPLAY_LABELS[display]],
                y=[value],
                marker_color=DISPLAY_COLORS[display],
                marker_line={"width": 0},
                text=[str(value)],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(f"{DISPLAY_LABELS[display]}: {value} of {total}<extra></extra>"),
            ),
            row=1,
            col=1,
        )
    figure.update_yaxes(title_text="Participants", range=[0, (total or 1) * 1.18], row=1, col=1)

    for index, outcome in enumerate(outcomes):
        for display in DISPLAY_ORDER:
            values = [
                float(row["score"])
                for row in ratings
                if row["outcome"] == outcome["key"] and row["display"] == display
            ]
            figure.add_trace(
                _display_box(values, display, hover="%{y:.0f}"),
                row=1,
                col=index + 2,
            )
        figure.update_yaxes(
            title_text="Rating" if index == 0 else None,
            range=[-8, 108],
            dtick=20,
            row=1,
            col=index + 2,
        )

    last = len(titles)
    for display in DISPLAY_ORDER:
        subset = [row for row in regret if row["display"] == display]
        figure.add_trace(
            _display_box(
                [float(row["regret"]) for row in subset],
                display,
                text=[f"{row['task_label']}: {row['choice_label']}" for row in subset],
                hover="%{text}<br>%{y:.1f} points",
            ),
            row=1,
            col=last,
        )
    figure.update_yaxes(title_text="Success rate points", rangemode="tozero", row=1, col=last)

    # Below the panels rather than above them, where `make_subplots` puts
    # them. A panel label reads as a caption for what the reader has just
    # looked at; above the axes it reads as a heading to get through first, and
    # on the preference panel it sat on top of the bar's own value label.
    for column, title in enumerate(titles, start=1):
        figure.add_annotation(
            text=title,
            xref=f"x{'' if column == 1 else column} domain",
            x=0.5,
            yref="paper",
            y=-0.17,
            showarrow=False,
            yanchor="top",
            font={"size": 13},
        )

    figure.update_layout(boxgap=_BOX_GAP, bargap=0.45, showlegend=False)
    return _layout(figure, height=height, top=16, bottom=92)


def _display_box(
    values: Sequence[float],
    display: str,
    *,
    text: Sequence[str] | None = None,
    hover: str = "%{y}",
) -> go.Box:
    """One display's box with every dot on it, in that display's colour."""
    return go.Box(
        y=list(values),
        name=DISPLAY_LABELS[display],
        marker_color=DISPLAY_COLORS[display],
        fillcolor=_rgba(DISPLAY_COLORS[display], _FILL_ALPHA),
        boxpoints="all",
        jitter=0.5,
        pointpos=0,
        marker={"size": 5, "opacity": 0.8},
        text=list(text) if text is not None else None,
        hovertemplate=f"{hover}<extra></extra>",
        showlegend=False,
    )


def _outcome_title(outcome: Mapping) -> str:
    arrow = _ARROW_DOWN if outcome.get("better") == "down" else _ARROW_UP
    return f"{outcome['label']} {arrow}"


def effect_table(rows: Sequence[Mapping]):
    """The adjusted estimates as the paper's table.

    Three of these rows are on the 0-100 scale the participants answered on and
    the fourth is in points of policy success rate. That was a column; it is one
    sentence in the caption instead, a column of two repeated strings being a
    column of noise.

    The interval is of the *difference*, and the header says so: an interval
    sitting beside three other numbers has to name which of them it is around.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Outcome": f"{row['label']} {_ARROW_DOWN if row.get('better') == 'down' else _ARROW_UP}",
                "Graph": round(float(row["mean_graph"]), 1),
                "Videos": round(float(row["mean_videos"]), 1),
                "Difference": round(float(row["difference"]), 1),
                "95% interval of difference": (
                    f"{float(row['lower']):+.1f} to {float(row['upper']):+.1f}"
                ),
            }
            for row in rows
        ]
    )


# ── Progress estimation ──────────────────────────────────────────────────────


def progress_curve_figure(
    rollout: Mapping,
    grid: Sequence[float],
    methods: Sequence[Mapping],
    *,
    height: int = 360,
) -> go.Figure:
    """What each estimator predicted along **one** rollout.

    One rollout, not an aggregate. A progress estimate is a per-rollout
    quantity and the estimators are deterministic, so a median across rollouts
    and a percentile band around it -- both of which this chart carried at one
    point -- are quantities the method never produces.

    The x axis is the fraction of the rollout elapsed, which is also what the
    video cursor reads, so the chart can sit under the clip
    (:func:`~behavior_graphs.viz.streamlit.components.video_curve_player`). The
    dotted diagonal is the elapsed fraction itself: on a rollout that
    succeeded that is true progress; on one that failed the robot stopped
    advancing, so true progress plateaus below it.
    """
    curves = rollout.get("curves") or {}
    figure = go.Figure()
    dashes = iter(BASELINE_DASHES)
    greys = iter(BASELINE_GREYS)
    for method in methods:
        values = curves.get(str(method["key"]))
        if values is None:
            continue
        ours = bool(method.get("ours"))
        colour = OURS_COLOR if ours else next(greys, BASELINE_GREYS[-1])
        figure.add_trace(
            go.Scatter(
                x=list(grid),
                y=list(values),
                mode="lines",
                name=str(method["label"]),
                line={
                    "color": colour,
                    "width": 3 if ours else 2,
                    "dash": "solid" if ours else next(dashes, "solid"),
                },
                hovertemplate=(
                    f"<b>{method['label']}</b><br>"
                    "at %{x:.0%} of the rollout: %{y:.2f}<extra></extra>"
                ),
            )
        )
    # Last, so it sits on top of the lines it is read against.
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Elapsed fraction",
            line={"color": "rgba(60,60,60,0.8)", "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.update_xaxes(title_text="Fraction of the rollout", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title_text="Estimated progress", range=[-0.03, 1.03])
    # This figure is the one that does NOT go through `st.plotly_chart`: it is
    # rendered to standalone HTML inside the video player's iframe, where
    # Streamlit's `theme="streamlit"` never reaches it, so it carries its own
    # template rather than inheriting the page's.
    figure.update_layout(template="plotly_white")
    return _layout(figure, height=height, left=84, bottom=110, legend_y=-0.22)


def rollout_label(rollout: Mapping) -> str:
    """How a rollout is named in a picker: its index and its outcome."""
    outcome = "success" if rollout.get("success") else "failure"
    return f"Rollout {int(rollout['idx']):03d} · {outcome}"


def curve_entry(curves: Mapping, dataset: str) -> dict | None:
    """One collection's curve bands, or ``None`` where it has none."""
    for entry in curves.get("datasets", ()):
        if str(entry.get("dataset")) == dataset:
            return dict(entry)
    return None


def curve_datasets(curves: Mapping, domains: Sequence[Mapping] | None = None) -> list[dict]:
    """The collections that have curves, grouped in the domains' own order.

    Ordered rather than sorted alphabetically: a dropdown of 43 names is easier
    to find a thing in when it runs LIBERO, real-world, Robomimic, MimicGen, as
    the table above it does.
    """
    entries = [dict(entry) for entry in curves.get("datasets", ())]
    order = [str(domain["label"]) for domain in (domains or ())]
    rank = {label: index for index, label in enumerate(order)}
    return sorted(
        entries, key=lambda e: (rank.get(str(e.get("domain")), len(rank)), str(e["dataset"]))
    )


def progress_table(
    means: Sequence[Mapping],
    metrics: Sequence[Mapping],
    methods: Sequence[Mapping],
):
    """The benchmark table: one block of rows per domain, one row per method.

    The domain column carries its dataset count, because the parenthetical is
    otherwise a puzzle to a reader who looks at the table alone, and because
    the count is the weight each row's mean was taken with.
    """
    import pandas as pd

    labels = {str(m["key"]): str(m["label"]) for m in methods}
    counts = {}
    for row in means:
        counts[str(row["domain"])] = int(row["datasets"])
    out = []
    for row in means:
        domain = str(row["domain"])
        cells = {
            "Source": f"{domain} ({counts[domain]})",
            "Method": labels.get(str(row["method"]), str(row["method"])),
        }
        for metric in metrics:
            key = str(metric["key"])
            arrow = "↑" if metric.get("higher_better") else "↓"
            value = row.get(key)
            cells[f"{metric['label']} {arrow}"] = None if value is None else round(float(value), 3)
        out.append(cells)
    return pd.DataFrame(out)


def best_cells(frame, metrics: Sequence[Mapping]) -> dict[tuple[int, str], bool]:
    """``{(row, column): True}`` for the winning cell of each domain block.

    Returned rather than applied, so the caller styles it however its renderer
    styles things: bold in a browser, ``\\textbf`` in a table.
    """
    best: dict[tuple[int, str], bool] = {}
    for metric in metrics:
        column = f"{metric['label']} {'↑' if metric.get('higher_better') else '↓'}"
        if column not in frame.columns:
            continue
        for source in frame["Source"].unique():
            block = frame[frame["Source"] == source][column].dropna()
            if block.empty:
                continue
            index = block.idxmax() if metric.get("higher_better") else block.idxmin()
            best[(int(index), column)] = True
    return best
