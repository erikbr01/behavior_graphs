"""The study's headline answers as paper-ready figures.

Two of them, in one style — double-column, or a single column where the figure
is a single panel:

:func:`study_figure`
    Four panels — which display participants reached for when nobody assigned
    them one, and the three ratings the paper argues over (temporal demand,
    effort, confidence) as a box per display.
:func:`regret_figure`
    What the choices *cost*: the regret of the data-collection choices made off
    each display, as a box and a dot per answer — pooled over the tasks, and
    again one panel per task. Scored
    against the retrained policies by
    :mod:`behavior_graphs.viz.streamlit.user_study.regret`, which is where the
    definition of regret and the option-to-policy mapping live.

Both read the same tidy rows the analytics page draws
(:mod:`behavior_graphs.viz.streamlit.user_study.analytics`), so every layout in
the store is understood here too, and a version that recorded none of these
answers produces no figure rather than an empty one.

Deliberately no caption, no title, and no suptitle: the caption belongs in the
LaTeX, and a figure that carries its own would be saying it twice. Each panel of
a multi-panel figure is labelled ``(a) …`` *below* the axes, where a subcaption
goes and where the figure's own caption will be reading up from — so the labels
the LaTeX refers to sit next to it rather than three inches away at the top.

Two things are kept out of the PDF on purpose. There is no transparency —
tinted fills are mixed against white here rather than drawn with an alpha, since
that is what survives a print pipeline — and no timestamp, so the same responses
always export byte-identical bytes. Fonts are embedded as TrueType (``fonttype
42``) because a Type-3 figure is what an IEEE submission gets bounced for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from behavior_graphs.viz import comparison_figure, paper
from behavior_graphs.viz.streamlit.user_study import analytics, regret

#: One colour per display, and the same two the analytics page uses, so a
#: reviewer looking at the live page and at the paper sees one scheme. The
#: matplotlib default blue/orange pair, which is the one categorical pair that
#: stays separable under deuteranopia and in greyscale.
PHASE_COLORS = {"videos": "#1f77b4", "graph": "#ff7f0e"}

#: Graph first: it is the display the paper is about, and the one a reader
#: should meet on the left of every panel. Taken from :mod:`regret`, which
#: groups its rows in this order, so the figure and the rows it is drawn from
#: cannot come to disagree about which display is on the left.
PHASE_ORDER = regret.PHASE_ORDER

#: Short enough to sit under a box in a 1.5-inch panel. Every box is labelled
#: with the display it belongs to rather than colour-coded against a legend, so
#: the figure survives being printed in greyscale.
PHASE_TICKS = {"graph": "Graph", "videos": "Videos"}

#: How a regret panel draws its answers, and what the pages offer as the choice.
#:
#: The **box** is the default and the better default: regret takes one of three
#: values per task (an arm set offers three options), so the answers are a
#: stack at 0 and a stack at the cost of each wrong option — a shape a mean
#: with a standard error around it implies is a spread, when no participant
#: could have answered between the stacks. It also matches the rating panels
#: beside it in the study figure.
#:
#: The **mean** is kept because it is the mark an audience reads fastest, and
#: because the number it draws is the one the paper's sentences quote. A slide
#: is exactly where it earns its place.
BOX_MARK = "box"
MEAN_MARK = "mean"
REGRET_MARKS = {
    BOX_MARK: "Distribution (box and dots)",
    MEAN_MARK: "Mean ± standard error",
}

#: The three rated dimensions the figure shows, in the order they are drawn —
#: which are the three the study asks
#: (``user_studies.variants.contrastive_same_page.questions._TLX_ITEMS``) and
#: not a selection of the six.
#:
#: ``effort`` rather than ``mental_demand`` because the instrument dropped
#: mental demand deliberately: at this length the two were two wordings of one
#: question, and effort is the wording kept. A figure that asked for mental
#: demand got a panel with nothing in it.
#:
#: ``confidence`` is not a TLX dimension at all — see
#: :mod:`behavior_graphs.viz.streamlit.user_study.nasa_tlx` — but it is asked by
#: the same block and is the one this figure exists to compare.
PAPER_DIMENSIONS = ("temporal_demand", "effort", "confidence")

#: The IEEE column widths, re-exported: see :mod:`behavior_graphs.viz.paper` for
#: why a figure is sized in inches rather than scaled in the LaTeX. A
#: single-column figure is what one panel of two bars wants — stretched across
#: both columns it wastes a page and reads as though its axis had been zoomed.
DOUBLE_COLUMN_WIDTH_IN = paper.DOUBLE_COLUMN_WIDTH_IN
SINGLE_COLUMN_WIDTH_IN = paper.SINGLE_COLUMN_WIDTH_IN
#: The height every figure here is drawn at, so two of them can be laid over
#: each other.
#:
#: It is a budget, not a preference. A rotated y label's *lines* run along the
#: panel height, and what is left of this after the panel's furniture is
#: **68.5 pt** where the x ticks take two lines (the box and marginal figures,
#: whose ticks carry an ``n``) and **76.5 pt** where they take one. The tallest
#: y label here — "Regret (success-rate points)", over two lines — needs
#: 67 pt of that, which is 1.5 pt of headroom on the tighter of the two.
#: Dropping "Mean" from it bought nothing: the binding line is
#: "(success-rate points)", which both wordings carry (re-measured).
#:
#: So there is no room for a taller label and none for another line of x label
#: either: this was 1.6 in for as long as the direction note was a second line
#: under the panel name, and every two-line y label in the figure was drawn
#: clipped at 1.4. Inline after the name, the note costs width, which a panel
#: label has to spare.
#:
#: Measured with `ax.get_window_extent()` inside `paper_rc()`. If a label
#: grows, measure again rather than guessing.
FIGURE_HEIGHT_IN = 1.4

#: The points a rating can take. Ticked rather than left to matplotlib so the
#: y-axis shows the scale participants actually answered on.
_RATING_TICKS = (0, 20, 40, 60, 80, 100)
#: Empty axes either side of the outermost mark, in category widths.
#: Matplotlib's categorical default is 0.5 -- half a slot of blank at each
#: end -- which on a 7.16 in strip is most of an inch of nothing. Narrow
#: enough that the marks reach for the panel edges, wide enough that an
#: interval's cap and its tick label are not clipped by the spine.
_CATEGORY_MARGIN = 0.32

#: The y-axis of every box panel. Just the name: a rotated label's *length*
#: runs along the vertical axis, so every word in it costs the figure height —
#: which is why the scale's direction ("0 = low, 100 = high") is not here. The
#: ticks show the range, and which way it runs belongs in the caption, where it
#: costs nothing.
_RATING_AXIS_LABEL = "Rating"

#: The same axis, said of an estimate rather than of the answers, on the
#: marginal-means figure. Named so a reader cannot take the interval for a
#: spread of ratings: it is the uncertainty in a mean, which is narrower than
#: the answers it is a mean of.
_ADJUSTED_RATING_AXIS_LABEL = "Adjusted rating"
#: Its decision-panel counterpart, the same two lines tall so the two figures
#: keep identical geometry and can be laid over each other. Two lines and no
#: longer: the panel is 68.5 pt tall and a rotated label's lines run along that
#: axis, so anything wider than this is drawn clipped. See
#: :data:`FIGURE_HEIGHT_IN` for where that number comes from.
_ADJUSTED_REGRET_AXIS_LABEL = "Adjusted regret\n(success-rate points)"

#: What a panel says about the way its y axis improves, by direction: the
#: arrow, and no word beside it.
#:
#: The arrow alone because it goes *after the panel name, on one line with it*,
#: where a word costs the panel width and a second line cost the figure 10 pt of
#: height. What it means is the caption's to say once, which is cheaper than
#: four panels saying "better" each.
#:
#: Mathtext rather than U+2193/U+2191, because the arrow has to come out of the
#: embedded face like the rest of the label: this figure is submitted as a PDF
#: with TrueType outlines in it (see :mod:`behavior_graphs.viz.paper`), and the
#: serif faces it prefers do not all carry the Unicode arrows — a missing glyph
#: is drawn as a box. The mathtext font set is STIX, which has both.
_BETTER_NOTES = {"down": r"$\downarrow$", "up": r"$\uparrow$"}

#: Which way each quantity this module draws improves. The rated dimensions'
#: directions are the instrument's, so they come from
#: :data:`analytics.BETTER_DIRECTION`; regret is a distance to the best option
#: available, so less of it is a better decision.
#:
#: Preference has no entry on purpose: it counts participants, and neither
#: direction of a count is better than the other.
_BETTER_DIRECTIONS = {**analytics.BETTER_DIRECTION, "regret": "down"}

#: Aliases, so the panel drawing below reads the same as it did when this
#: module owned the style. The definitions live in
#: :mod:`behavior_graphs.viz.paper`, shared with the behavior-graph figure — the
#: requirements are the paper's, not this figure's.
_INK = paper.INK
_MUTED = paper.MUTED


def paper_rc():
    """This figure's style: the shared paper style plus axis furniture."""
    return paper.paper_rc(paper.CHART_RC)


def _tint(hex_color: str, weight: float) -> tuple[float, float, float]:
    """*hex_color* mixed *weight* of the way to white, as an RGB triple.

    A fill light enough to sit under a dark median line, without an alpha:
    transparency in a submitted PDF is the thing that renders differently in
    the reviewer's viewer than in ours.
    """
    raw = hex_color.lstrip("#")
    rgb = tuple(int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(channel + (1.0 - channel) * weight for channel in rgb)  # type: ignore[return-value]


def preference_counts(responses: Iterable[dict]) -> dict[str, int]:
    """``{display: participants}`` for the stage nobody assigned a display.

    The three-stage layout's own result. Counted off
    :func:`analytics.stage_rows`, so the display a task was read in is resolved
    the same way the page resolves it, and both displays appear even when one
    was never picked — a missing bar reads as missing data rather than as a
    zero.
    """
    counts = {phase: 0 for phase in PHASE_ORDER}
    for row in analytics.stage_rows(responses):
        if row["assignment"] != "chosen":
            continue
        phase = row["phase"]
        if phase in counts:
            counts[phase] += 1
    return counts


def choice_series(
    rows: Iterable[Mapping],
    datasets: Sequence[str],
    labels: Mapping[str, str] | None = None,
) -> tuple[comparison_figure.ChoiceSeries, ...]:
    """The data-collection answers for one group of policies, per display.

    The join between the study's answers and the paper's policy comparison:
    *rows* are :func:`regret.choice_share_rows`, *datasets* is one group of the
    comparison (:data:`policy_groups.DEFAULT_GROUPS`), and what comes back is a
    :class:`~behavior_graphs.viz.comparison_figure.ChoiceSeries` per display,
    keyed by the label the figure draws that policy under — so the bars land
    under the right violins whatever the group is called.

    Matched on the dataset, not on the task: the group is the source of truth
    for which policies are in the panel, and the option-to-policy mapping is
    :mod:`regret`'s. A task whose arms are split across two groups therefore
    contributes to both, with each group's own denominator — which is the right
    answer, since a share is only a share among the options on one screen.

    This is where the displays' colours and names come from, and the reason the
    figure takes them as arguments: :data:`PHASE_COLORS` and
    :data:`PHASE_TICKS` are the study's, and the same two the analytics page
    uses. A display with no answers about this group is left out, so the figure
    draws no row for it.
    """
    from behavior_graphs.viz.streamlit import policy_groups

    wanted = set(datasets)
    series = []
    for phase in PHASE_ORDER:
        in_phase = [row for row in rows if row["phase"] == phase and row["dataset"] in wanted]
        if not in_phase:
            continue
        series.append(
            comparison_figure.ChoiceSeries(
                label=PHASE_TICKS.get(phase, phase),
                color=PHASE_COLORS.get(phase, _MUTED),
                picks={
                    policy_groups.paper_label(str(row["dataset"]), labels)
                    or str(row["dataset"]): int(row["picks"])
                    for row in in_phase
                },
                # Read off the rows rather than summed from the picks: the
                # group may hold a subset of the options that were offered, and
                # the denominator is what that display was asked, not what this
                # panel can draw. Its own answers, so the largest of them is the
                # one every option in the panel was a share of.
                answers=max(int(row["answers"]) for row in in_phase),
            )
        )
    return tuple(series)


def rating_scores(
    responses: Iterable[dict],
    dimensions: Sequence[str] = PAPER_DIMENSIONS,
    *,
    randomized_only: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """``{dimension: {display: [scores]}}`` for the rated dimensions asked.

    Only ratings that name a display: a layout that rated a whole session once
    (the production one) records no display, and pooling those into a
    graph-versus-videos box would compare a condition with a modality.

    ``randomized_only`` restricts the boxes to exactly the rows
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.assigned_rating_rows`
    models: the stages whose display was **assigned**, from completed
    participant records. Off by default, because the analytics page's own
    reading of "what did people rate" is every rating there is; **on for the
    paper figure**, where the boxes sit beside an adjusted estimate and a reader
    who checks one against the other has to be looking at the same evidence.

    The difference is the last stage, whose display the participant *chose*. It
    is not randomized, so it cannot enter the estimate -- and a figure that
    includes it while the interval beside it does not is a discrepancy a careful
    reader finds and nobody can explain. On the live store it is a third of the
    ratings (54 of 81 modelled), and it is the third where the graph was picked
    10 times out of 12.
    """
    scores: dict[str, dict[str, list[float]]] = {
        dimension: {phase: [] for phase in PHASE_ORDER} for dimension in dimensions
    }
    if randomized_only:
        # Imported here rather than at module scope: `mixed_effects` is the
        # module that needs the optional analysis extra, and this function must
        # keep working without it. `assigned_rating_rows` itself is pure --
        # the extra is only imported inside the fitting functions.
        from behavior_graphs.viz.streamlit.user_study.mixed_effects import assigned_rating_rows

        for row in assigned_rating_rows(responses):
            by_phase = scores.get(row["outcome"])
            phase = "graph" if row["graph"] else "videos"
            if by_phase is not None and phase in by_phase:
                by_phase[phase].append(row["score"])
        return scores
    for row in analytics.tlx_rows(responses):
        by_phase = scores.get(row["dimension_key"])
        if by_phase is None or row["phase"] not in by_phase:
            continue
        by_phase[row["phase"]].append(row["score"])
    return scores


def _fanned(values: Sequence[float], spread: float) -> list[float]:
    """Horizontal offsets that spread tied values evenly around 0.

    The scale is six points wide (0–100 in steps of 20), so a strip of points
    over a box is mostly ties, and a point drawn on top of another point is a
    participant the figure does not show. Spread by value rather than by random
    jitter, both so an exported PDF is reproducible and so equal answers read as
    a row of equal answers.
    """
    positions = [0.0] * len(values)
    for value in sorted(set(values)):
        tied = [i for i, other in enumerate(values) if other == value]
        # `<= 1`, not `== 1`: a NaN equals nothing, including itself, so its own
        # group comes back empty and the step below divided by zero. A NaN
        # should not reach here — but a panel crashing the page is a worse way
        # to find out than a point drawn without a fan.
        if len(tied) <= 1:
            continue
        step = min(spread, spread * 3 / len(tied))
        start = -step * (len(tied) - 1) / 2
        for slot, index in enumerate(tied):
            positions[index] = start + slot * step
    return positions


def _label(ax, text: str, *, better: str | None = None) -> None:
    """Name the panel under its axes, where a subcaption goes.

    The x-axis label rather than a title, so it lands below the tick labels and
    the layout engine reserves room for it like any other axis text — a
    hand-placed text at a fixed offset would collide with the tick labels the
    moment a panel gains or loses a box.

    *better* is a key of :data:`_BETTER_NOTES` and puts an arrow after the name
    for which way this panel's y axis improves, for a reader who does not
    already know that less perceived effort is the better display and more
    confidence is. After the name rather than on the y axis, for three reasons
    that all point the same way:

    * The three rating panels **share one y axis**, and they do not agree on a
      direction. An arrow on the shared axis would be wrong for confidence.
    * A y label is drawn rotated, and so is any arrow in it: a down arrow set
      in one comes out pointing *right*.
    * A y label's length runs along the figure's height, which the regret
      panels' two-line label already spends to its limit.

    On the name's own line, which is where it is free: an x label's length is
    its width, and a panel label has width to spare where the figure has no
    height to.
    """
    if better:
        text = f"{text} {_BETTER_NOTES[better]}"
    ax.set_xlabel(text, color=_INK, labelpad=3.0)


def _outcome_ticks(labels: Sequence[str], outcomes: Sequence[str]) -> list[str]:
    """*labels*, each with the way its outcome improves under it.

    For the panels that hold all three dimensions side by side, where the
    direction cannot go under the panel name because it differs per tick.
    """
    return [
        f"{label} {_BETTER_NOTES[_BETTER_DIRECTIONS[outcome]]}"
        if outcome in _BETTER_DIRECTIONS
        else label
        for label, outcome in zip(labels, outcomes, strict=True)
    ]


def _draw_preference(ax, counts: dict[str, int], title: str) -> None:
    phases = [phase for phase in PHASE_ORDER if phase in counts]
    values = [counts[phase] for phase in phases]
    ax.bar(
        range(len(phases)),
        values,
        width=0.58,
        color=[_tint(PHASE_COLORS[phase], 0.45) for phase in phases],
        edgecolor=[PHASE_COLORS[phase] for phase in phases],
        linewidth=0.8,
        zorder=3,
    )
    # The count on the bar, because a reader of a two-bar chart wants the
    # numbers and the y-axis of a small panel cannot be read to one participant.
    for x, value in zip(range(len(phases)), values, strict=True):
        ax.annotate(
            f"{value}",
            (x, value),
            xytext=(0, 1.6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=_INK,
        )
    ax.set_xticks(range(len(phases)), [PHASE_TICKS[phase] for phase in phases])
    ax.set_xlim(-0.6, len(phases) - 0.4)
    ax.set_ylim(0, max([*values, 1]) * 1.22)
    ax.set_ylabel("Participants")
    # Integer participants only: 2.5 people never chose anything.
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    _label(ax, title)


def _draw_adjusted_effects(
    ax, effects: Sequence[dict], title: str, *, show_ylabel: bool = False
) -> None:
    """Draw graph-minus-video mixed-model effects with their 95% intervals."""
    labels = {
        "temporal_demand": "Temporal demand",
        "effort": "Effort",
        "confidence": "Confidence",
    }
    ordered = [
        effect for outcome in PAPER_DIMENSIONS for effect in effects if effect["outcome"] == outcome
    ]
    positions = list(range(len(ordered)))
    means = [effect["mean"] for effect in ordered]
    lowers = [effect["lower"] for effect in ordered]
    uppers = [effect["upper"] for effect in ordered]
    ax.axhline(0, color=_MUTED, linewidth=0.8, zorder=1)
    ax.errorbar(
        positions,
        means,
        yerr=[
            [mean - lower for mean, lower in zip(means, lowers, strict=True)],
            [upper - mean for mean, upper in zip(means, uppers, strict=True)],
        ],
        fmt="o",
        color=PHASE_COLORS["graph"],
        ecolor=_INK,
        markersize=4.2,
        elinewidth=0.9,
        capsize=2.2,
        zorder=3,
    )
    extent = [0.0, *lowers, *uppers]
    pad = max(4.0, (max(extent) - min(extent)) * 0.08)
    ax.set_xlim(-_CATEGORY_MARGIN, len(ordered) - 1 + _CATEGORY_MARGIN)
    ax.set_ylim(min(extent) - pad, max(extent) + pad)
    ax.set_xticks(
        positions,
        _outcome_ticks(
            [labels[effect["outcome"]] for effect in ordered],
            [effect["outcome"] for effect in ordered],
        ),
    )
    # Two lines, because one is 79 pt of rotated text and the panel is 76.5:
    # this label was drawn clipped in every export of the figure until it was
    # measured. See `FIGURE_HEIGHT_IN` for the budget.
    ax.set_ylabel("Graph $-$ Videos\n(points)" if show_ylabel else "", color=_INK)
    if title:
        _label(ax, title)


def _draw_adjusted_ratings(ax, effects: Sequence[dict], title: str) -> None:
    """Draw population-level adjusted graph and video ratings side by side."""
    labels = {
        "temporal_demand": "Temporal demand",
        "effort": "Effort",
        "confidence": "Confidence",
    }
    ordered = [
        effect for outcome in PAPER_DIMENSIONS for effect in effects if effect["outcome"] == outcome
    ]
    positions = list(range(len(ordered)))
    for modality, offset in (("graph", -0.14), ("videos", 0.14)):
        values = [effect["marginal_means"][modality] for effect in ordered]
        means = [value["mean"] for value in values]
        ax.errorbar(
            [position + offset for position in positions],
            means,
            yerr=[
                [mean - value["lower"] for mean, value in zip(means, values, strict=True)],
                [value["upper"] - mean for mean, value in zip(means, values, strict=True)],
            ],
            fmt="o",
            color=PHASE_COLORS[modality],
            ecolor=PHASE_COLORS[modality],
            markersize=4.0,
            elinewidth=0.9,
            capsize=2.2,
            label=PHASE_TICKS[modality],
            zorder=3,
        )
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    ax.set_xticks(
        positions,
        _outcome_ticks(
            [labels[effect["outcome"]] for effect in ordered],
            [effect["outcome"] for effect in ordered],
        ),
    )
    ax.set_ylim(-5, 105)
    ax.set_yticks(_RATING_TICKS)
    ax.set_ylabel("Adjusted rating", color=_INK)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.3)
    _label(ax, title)


def _draw_regret_effect(ax, effect: dict, title: str) -> None:
    """Draw the adjusted graph-minus-video data-collection regret effect.

    Same shape as the rating-effect panel beside it, and the same direction:
    **below** the zero line is the graph leading to less regret, which is the
    graph leading to better decisions. The panel beside it draws rating
    differences, where below zero is also the better outcome for two of the
    three -- so the figure reads one way throughout, and the caption says which.
    """
    mean, lower, upper = effect["mean"], effect["lower"], effect["upper"]
    ax.axhline(0, color=_MUTED, linewidth=0.8, zorder=1)
    ax.errorbar(
        [0],
        [mean],
        yerr=[[mean - lower], [upper - mean]],
        fmt="o",
        color=PHASE_COLORS["graph"],
        ecolor=_INK,
        markersize=4.2,
        elinewidth=0.9,
        capsize=2.2,
        zorder=3,
    )
    extent = [0.0, lower, upper]
    pad = max(2.0, (max(extent) - min(extent)) * 0.12)
    ax.set_xlim(-_CATEGORY_MARGIN, _CATEGORY_MARGIN)
    ax.set_ylim(min(extent) - pad, max(extent) + pad)
    ax.set_xticks([0], ["Data collection"])
    # Its own unit, spelled out: the panel beside it is in rating points on a
    # 0--100 scale, and this one is in points of policy success rate. Two
    # unlabelled point-valued axes side by side read as one scale.
    ax.set_ylabel("Success-rate points", color=_INK)
    _label(ax, title, better=_BETTER_DIRECTIONS["regret"])


#: What :func:`adjusted_study_figure` can show, and what the pages offer.
ADJUSTED_FIGURE_MODES = {
    "difference": "Graph minus Videos (paper default)",
    "ratings": "Adjusted Graph and Videos ratings",
}


def adjusted_study_figure(
    responses: Iterable[dict],
    effects: Sequence[dict],
    *,
    regret_effect: dict | None = None,
    mode: str = "difference",
):
    """Paper figure for preference and adjusted, randomized-stage effects.

    Raw rating distributions remain available in Survey Analytics. This figure
    deliberately presents the same estimand as the paper's mixed-model text.

    ``mode="difference"`` shows the graph-minus-video coefficient; ``"ratings"``
    shows adjusted population-level marginal ratings under each visualization.

    *regret_effect* is :func:`mixed_effects.decision_regret_effect`, and it adds the
    third panel: the same graph-minus-video contrast read on the decision the
    study is actually about rather than on how the display felt. It is optional
    for the reason the regret panels elsewhere are conditional — scoring a
    choice needs every option's retrained policy on disk, which a deployment
    without those rollouts mounted does not have. Passed with ``mode="ratings"``
    it is ignored: that mode's middle panel is already two series wide, and a
    contrast panel beside two absolute-rating series would be read as a third.
    """
    import matplotlib.pyplot as plt

    if mode not in ADJUSTED_FIGURE_MODES:
        raise ValueError(f"Unknown adjusted-study figure mode: {mode!r}")
    if not effects:
        return None
    counts = preference_counts(responses)
    # The regret panel is one estimate over one tick, so it is sized like a
    # tick and not like a third of the row: the width it does not need goes to
    # the effects panel, which is the one carrying three intervals.
    carries_regret = mode == "difference" and regret_effect is not None
    width_ratios = [0.8, 2.5, 0.62] if carries_regret else [0.72, 1.7]
    with paper_rc():
        fig, axes = plt.subplots(
            1,
            len(width_ratios),
            figsize=(DOUBLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
            gridspec_kw={"width_ratios": width_ratios},
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.05, hspace=0.0)
        _draw_preference(axes[0, 0], counts, "(a) Chosen display")
        if mode == "difference":
            _draw_adjusted_effects(axes[0, 1], effects, "(b) Graph $-$ Videos", show_ylabel=True)
            if carries_regret:
                _draw_regret_effect(
                    axes[0, 2], regret_effect, f"(c) {STUDY_PANEL_TITLES['regret']}"
                )
        else:
            _draw_adjusted_ratings(axes[0, 1], effects, "(b) Adjusted ratings")
        fig.align_xlabels(axes[0])
    return fig


def adjusted_study_figure_files(
    responses: Iterable[dict],
    effects: Sequence[dict],
    *,
    regret_effect: dict | None = None,
    mode: str = "difference",
) -> dict[str, bytes] | None:
    """Render :func:`adjusted_study_figure` to the paper's PDF and PNG formats."""
    import matplotlib.pyplot as plt

    fig = adjusted_study_figure(responses, effects, regret_effect=regret_effect, mode=mode)
    if fig is None:
        return None
    try:
        with paper_rc():
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 220)}
    finally:
        plt.close(fig)


def _draw_ratings(
    ax,
    by_phase: dict[str, list[float]],
    title: str,
    *,
    first: bool,
    better: str | None = None,
) -> None:
    phases = [phase for phase in PHASE_ORDER if by_phase.get(phase)]
    for x, phase in enumerate(phases):
        values = by_phase[phase]
        color = PHASE_COLORS[phase]
        parts = ax.boxplot(
            [values],
            positions=[x],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            # The points are all drawn below, so a flier would be one
            # participant marked twice.
            whis=(0, 100),
            zorder=3,
        )
        for box in parts["boxes"]:
            box.set(facecolor=_tint(color, 0.6), edgecolor=color, linewidth=0.8)
        for line in (*parts["whiskers"], *parts["caps"]):
            line.set(color=color, linewidth=0.8)
        for line in parts["medians"]:
            line.set(color=_INK, linewidth=1.1)
        ax.scatter(
            [x + offset for offset in _fanned(values, 0.055)],
            values,
            s=3.6,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )
    ax.set_xticks(
        range(len(phases)),
        [f"{PHASE_TICKS[phase]}\nn={len(by_phase[phase])}" for phase in phases],
    )
    ax.set_xlim(-0.6, len(phases) - 0.4)
    ax.set_ylim(-6, 106)
    ax.set_yticks(_RATING_TICKS)
    if first:
        ax.set_ylabel(_RATING_AXIS_LABEL)
    else:
        # One y axis for the three of them (they are linked in `study_figure`),
        # so it is drawn once on the left and the panels beside it carry neither
        # its numbers nor its spine — three repeats of one scale is ink that
        # says nothing, and it makes the panels read as separate scales. The
        # gridlines are still at every tick, so a box is still readable against
        # the scale two panels away.
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.spines["left"].set_visible(False)
    # The direction goes under every panel even though they share an axis,
    # because the three dimensions do not share a direction.
    _label(ax, title, better=better)


def study_figure(
    responses: Iterable[dict],
    regret_rows: Iterable[dict] = (),
    *,
    randomized_only: bool = False,
    regret_mark: str = BOX_MARK,
):
    """The figure, or ``None`` when these responses answered none of it.

    ``None`` rather than an empty figure: the version selected on the analytics
    page may be one that never asked for a display preference or a rating, and
    exporting four blank axes for it would look like a plotting bug.

    *regret_rows* are
    :func:`behavior_graphs.viz.streamlit.user_study.regret.regret_rows`. Passed in
    rather than computed here, because scoring a choice needs the measured
    success rates of the retrained policies off disk and this module draws. A
    caller with none — or a study version that asked no data-collection
    question — gets the figure without that panel.

    ``randomized_only`` restricts every panel that compares the two displays --
    the ratings *and* the decision regret -- to the stages whose display was
    assigned, which are the rows the adjusted model is fit on. **This is what
    the paper wants**: the raw boxes are the evidence a reader believes before
    they believe any model, so they have to be the same evidence the estimate
    beside them came from. See :func:`rating_scores`.

    It matters most in the last panel, where the third stage's display was the
    participant's own pick and 16 of 18 picked the graph: pooling it in reversed
    which display led to better decisions. The preference panel is deliberately
    outside the restriction -- that stage *is* its subject.

    The last panel reports **regret**, where lower is better, as the answers
    themselves — a box and a dot per decision, like the rating panels beside
    it. Pooling those points across tasks is the thing to caveat: the options
    span 33 success-rate points on set table and 8 on cooking, so the pile is
    weighted toward the widest, and its upper tail is set table's alone.
    Regret is still the quantity to compare displays on -- unlike
    gain, it does not carry the task's headroom, which on the live store put the
    graph's mean gain 3.6 points ahead purely because the draw gave it more
    prepare-tray stages while the two displays' regret differed by 0.6.

    Save it through :func:`study_figure_files`, or from inside
    :func:`paper_rc`; a bare ``fig.savefig`` gets the process's own style.
    """
    import matplotlib.pyplot as plt

    panels = study_panels(responses, regret_rows, randomized_only=randomized_only)
    if not panels:
        return None

    with paper_rc():
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(DOUBLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
            gridspec_kw={"width_ratios": study_panel_widths(panels)},
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.035, hspace=0.0)
        draw_study_panels(axes[0], panels, regret_mark=regret_mark)
        # One baseline for the panel labels: the box panels carry a second line
        # of tick label (the n) and the preference panel does not, so left to
        # itself the row reads as labels at two different heights.
        fig.align_xlabels(axes[0])
    return fig


def study_panels(
    responses: Iterable[dict],
    regret_rows: Iterable[dict] = (),
    *,
    randomized_only: bool = False,
) -> list[tuple[str, object]]:
    """``[(key, data)]`` — which panels these responses answer, in figure order.

    Separated from the drawing because the grid has to be sized before any axes
    exist: how many panels there are is a property of the answers, and how wide
    each one should be depends on which ones they are
    (:func:`study_panel_widths`). :func:`adjusted_marginals_figure` reads the
    same selection, which is what keeps it panel for panel the same figure.

    An empty list is a version that answered none of it, which is why the
    figures return ``None`` rather than drawing blank axes.
    """
    responses = list(responses)
    counts = preference_counts(responses)
    ratings = rating_scores(responses, randomized_only=randomized_only)
    # Regret in success-rate points: the success rate the chosen option left on
    # the table against the best one available. LOWER IS BETTER here, which is
    # the one thing this panel has to make a reader hold, and why the axis says
    # "regret" rather than "points".
    #
    # Regret rather than gain over the baseline, which this figure reported for
    # a while: gain reads more easily by itself, but it carries the task's
    # headroom — +28.3 points available on prepare tray against +3.3 on cooking
    # — so a mean of gains is partly a statement about which tasks a display was
    # shown with. Regret asks only how well they chose among what they saw.
    #
    # Points rather than a normalized score because it is the unit a reader can
    # hold ("eight points of success rate left on the table"), where a
    # normalized one needs its chance line explained first. Pooling points has
    # its own cost, which belongs in the caption — the tasks' options span 33
    # points and 8, so the pile is weighted toward the widest of them — and the
    # per-task panels that show where it comes from are in `regret_figure`.
    regret_rows = list(regret_rows)
    decision_rows = regret.randomized_rows(regret_rows) if randomized_only else regret_rows
    regret_by_phase = regret.by_display(decision_rows)

    panels: list[tuple[str, object]] = []
    if any(counts.values()):
        panels.append(("preference", counts))
    for dimension in PAPER_DIMENSIONS:
        by_phase = ratings.get(dimension, {})
        if any(by_phase.values()):
            panels.append((dimension, by_phase))
    # Last, because it is the outcome the ratings are the subjective half of:
    # the row reads preference, then what the displays cost to use, then what
    # the decisions they led to were worth.
    if any(regret_by_phase.values()):
        panels.append(("regret", regret_by_phase))
    return panels


#: What each panel of the study row is called, under its axes.
STUDY_PANEL_TITLES = {
    "preference": "Chosen display",
    **{key: analytics.TLX_LABELS.get(key, key) for key in PAPER_DIMENSIONS},
    # Short, because the panel label has to fit under a panel a fifth of the row
    # wide: "(e) Data-collection regret" is wider than its own axes and overhangs
    # the figure. "Success rate regret" names the unit the panel is in, which is
    # the thing a reader has to hold about it — that the bar is success-rate
    # points left on the table, not a count and not a rating. Unhyphenated as a
    # panel name; "success-rate points" keeps its hyphen on the axis, where it
    # is a compound modifier of the unit.
    "regret": "Success rate regret",
}


def study_panel_widths(panels: Sequence[tuple[str, object]]) -> list[float]:
    """Width share per panel, in the order :func:`study_panels` returns them.

    The preference panel holds two bars and no y tick labels worth room, so it
    is given less width than the box panels it sits beside. The regret panel is
    two bars too, but its y label runs to two lines and a rotated label's lines
    come out of the panel's width, so it keeps a full share.
    """
    return [0.78 if key == "preference" else 1.0 for key, _ in panels]


def draw_study_panels(
    axes: Sequence,
    panels: Sequence[tuple[str, object]],
    *,
    regret_mark: str = BOX_MARK,
) -> None:
    """Draw *panels* onto *axes*, one each, in order.

    Whatever laid the axes out owns the geometry; this only draws, which is what
    lets the selection, the widths and the drawing be read separately.

    *regret_mark* is how the last panel draws its answers — see
    :data:`REGRET_MARKS`. The box is the default; the mean bar is what the
    figure carried before and is still the faster read for a slide.
    """
    # The rating panels share one y axis, linked rather than merely set to the
    # same limits: the three are the same 0-100 scale and a reader compares a
    # box in one against a box in another. Linked, a later change to the scale
    # cannot reach one panel and miss its neighbours. The preference panel is
    # left out of it — its axis counts participants.
    rating_axes = [
        ax for ax, (key, _) in zip(axes, panels, strict=True) if key not in ("preference", "regret")
    ]
    for ax in rating_axes[1:]:
        ax.sharey(rating_axes[0])
    drawn_ratings = 0
    for index, (ax, (key, data)) in enumerate(zip(axes, panels, strict=True)):
        # (a), (b), … so the LaTeX caption has something to point at without the
        # figure carrying a caption of its own.
        title = f"({chr(ord('a') + index)}) {STUDY_PANEL_TITLES[key]}"
        if key == "preference":
            _draw_preference(ax, data, title)  # type: ignore[arg-type]
        elif key == "regret":
            # `first`: this panel carries its own axis whatever its position,
            # because its scale is success-rate points and its neighbours' is
            # the 0-100 rating scale.
            #
            # No printed mean even when the mark is the mean: the panel is a
            # fifth of a row, and the number would be a third thing to read in
            # it after the bar and the axis.
            _draw_regret_panel(  # type: ignore[arg-type]
                ax, data, title, mark=regret_mark, first=True, show_values=False
            )
        else:
            _draw_ratings(  # type: ignore[arg-type]
                ax, data, title, first=drawn_ratings == 0, better=_BETTER_DIRECTIONS.get(key)
            )
            drawn_ratings += 1


def _draw_marginal_ratings(
    ax, by_phase: dict[str, dict], title: str, *, first: bool, better: str | None = None
) -> None:
    """One adjusted mean per display, with its 95% HDI.

    The box panel's replacement (:func:`_draw_ratings`), on the same axes so the
    two figures can be laid over each other: same 0--100 range, same ticks, same
    order, same ``n`` under each tick. What changes is what the mark means — an
    estimate with its uncertainty, rather than the answers themselves.
    """
    phases = [phase for phase in PHASE_ORDER if by_phase.get(phase)]
    for x, phase in enumerate(phases):
        estimate = by_phase[phase]
        mean = estimate["mean"]
        ax.errorbar(
            [x],
            [mean],
            yerr=[[mean - estimate["lower"]], [estimate["upper"] - mean]],
            fmt="o",
            color=PHASE_COLORS[phase],
            ecolor=PHASE_COLORS[phase],
            markersize=4.0,
            elinewidth=0.9,
            capsize=2.2,
            zorder=3,
        )
    ax.set_xticks(
        range(len(phases)),
        [
            f"{PHASE_TICKS[phase]}\nn={by_phase[phase]['n']}"
            if by_phase[phase].get("n") is not None
            else PHASE_TICKS[phase]
            for phase in phases
        ],
    )
    ax.set_xlim(-0.6, len(phases) - 0.4)
    ax.set_ylim(-6, 106)
    ax.set_yticks(_RATING_TICKS)
    if first:
        ax.set_ylabel(_ADJUSTED_RATING_AXIS_LABEL)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.spines["left"].set_visible(False)
    _label(ax, title, better=better)


def _draw_marginal_regret(ax, by_phase: dict[str, dict], title: str) -> None:
    """The adjusted regret each display led to, with its 95% HDI.

    **Lower is better**, and the dotted line at zero is the optimum rather than
    a neutral point: no regret is having picked the best option available. Kept
    in view for that reason, so the marks are read against what perfect would
    have looked like.
    """
    phases = [phase for phase in PHASE_ORDER if by_phase.get(phase)]
    ax.axhline(0, color=_INK, linewidth=0.6, linestyle=(0, (1.4, 1.6)), zorder=2)
    for x, phase in enumerate(phases):
        estimate = by_phase[phase]
        mean = estimate["mean"]
        ax.errorbar(
            [x],
            [mean],
            yerr=[[mean - estimate["lower"]], [estimate["upper"] - mean]],
            fmt="o",
            color=PHASE_COLORS[phase],
            ecolor=PHASE_COLORS[phase],
            markersize=4.0,
            elinewidth=0.9,
            capsize=2.2,
            zorder=3,
        )
    ax.set_xticks(
        range(len(phases)),
        [
            f"{PHASE_TICKS[phase]}\nn={by_phase[phase]['n']}"
            if by_phase[phase].get("n") is not None
            else PHASE_TICKS[phase]
            for phase in phases
        ],
    )
    ax.set_xlim(-0.6, len(phases) - 0.4)
    lows = [by_phase[phase]["lower"] for phase in phases]
    highs = [by_phase[phase]["upper"] for phase in phases]
    span = max(max(highs) - min(min(lows), 0.0), 1.0)
    ax.set_ylim(min(min(lows), 0.0) - 0.08 * span, max(max(highs), 0.0) + 0.12 * span)
    ax.set_ylabel(_ADJUSTED_REGRET_AXIS_LABEL)
    _label(ax, title, better=_BETTER_DIRECTIONS["regret"])


def adjusted_marginals_figure(
    responses: Iterable[dict],
    effects: Sequence[dict],
    *,
    regret_effect: dict | None = None,
    randomized_only: bool = True,
):
    """:func:`study_figure`'s layout, with the model's estimates in it.

    Panel for panel the same figure — preference, the three rated dimensions,
    the decision regret -- except that each rating and regret panel carries the
    **adjusted marginal mean per display and its 95% HDI** instead of the
    answers. So the two can be swapped in a paper, or laid over each other, and
    a reader comparing them is comparing like with like.

    Why have both: the boxes are the evidence and this is the estimate. The
    estimate is the one that holds task, stage and experience fixed, which
    matters because every participant rated one task off the graph and a
    different one off the videos — so a raw difference carries whatever task
    imbalance the randomization happened to produce, and this one does not.

    The preference panel is identical and deliberately unmodelled: it is a
    count of what 18 people picked, and there is nothing to adjust it for.

    **What this figure cannot be read for is the difference.** Two EMM intervals
    overlapping does not mean the displays are indistinguishable, for two
    reasons that compound. Each interval carries the uncertainty about where the
    whole scale sits -- the intercept, the task and stage levels -- and both
    displays share it, so it cancels in the contrast: on the live store the two
    posteriors correlate +0.64 for temporal demand, which puts the difference's
    SD at 6.2 where independence would give 10.2. And even uncorrelated, two
    equal-SE intervals only stop touching at 3.92 SE apart while their
    difference clears zero at 2.77, so there is a band where both are true at
    once. Effort is in it: the marks overlap by 6.6 points and the difference is
    -22.9 [-37.7, -9.0].

    The difference is the ``graph`` coefficient itself -- identical mean, SD and
    HDI, since the model is linear with no display-by-task interaction -- so
    :func:`adjusted_study_figure` is the figure that carries it, and this one is
    for showing the levels the contrast is between.

    *effects* are
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.continuous_effects`
    and *regret_effect*
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.decision_regret_effect`
    — fitted by the caller, since this module draws and does not sample.
    ``None`` when nothing could be fit, exactly as :func:`study_figure` returns
    ``None`` when there is nothing to draw.
    """
    import matplotlib.pyplot as plt

    responses = list(responses)
    counts = preference_counts(responses)
    # The n under each tick, taken from the rows the models are fit on — the
    # same call the raw figure's boxes come from, so the two figures cannot
    # disagree about how many answers are behind a panel.
    observed = rating_scores(responses, randomized_only=randomized_only)
    by_outcome = {str(effect.get("outcome", "")): effect for effect in effects}

    panels: list[tuple[str, object]] = []
    if any(counts.values()):
        panels.append(("preference", counts))
    for dimension in PAPER_DIMENSIONS:
        effect = by_outcome.get(dimension)
        marginal = (effect or {}).get("marginal_means") or {}
        if not marginal:
            continue
        counted = (effect or {}).get("n") or {}
        panels.append(
            (
                dimension,
                {
                    phase: {
                        **marginal[phase],
                        "n": counted.get(phase, len(observed.get(dimension, {}).get(phase, []))),
                    }
                    for phase in PHASE_ORDER
                    if phase in marginal
                },
            )
        )
    regret_marginal = (regret_effect or {}).get("marginal_means") or {}
    if regret_marginal:
        counted = (regret_effect or {}).get("n_by_display") or {}
        panels.append(
            (
                "regret",
                {
                    phase: {**regret_marginal[phase], "n": counted.get(phase)}
                    for phase in PHASE_ORDER
                    if phase in regret_marginal
                },
            )
        )
    if not panels:
        return None

    # The raw figure's names and widths, because this figure is that one panel
    # for panel with an estimate in place of the answers — two lists of titles
    # is how the pair comes to disagree about what a panel is called.
    titles = STUDY_PANEL_TITLES
    widths = study_panel_widths(panels)

    with paper_rc():
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(DOUBLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
            gridspec_kw={"width_ratios": widths},
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.035, hspace=0.0)
        rating_axes = [
            ax
            for ax, (key, _) in zip(axes[0], panels, strict=True)
            if key not in ("preference", "regret")
        ]
        for ax in rating_axes[1:]:
            ax.sharey(rating_axes[0])
        drawn_ratings = 0
        for index, (ax, (key, data)) in enumerate(zip(axes[0], panels, strict=True)):
            title = f"({chr(ord('a') + index)}) {titles[key]}"
            if key == "preference":
                _draw_preference(ax, data, title)  # type: ignore[arg-type]
            elif key == "regret":
                _draw_marginal_regret(ax, data, title)  # type: ignore[arg-type]
            else:
                _draw_marginal_ratings(  # type: ignore[arg-type]
                    ax, data, title, first=drawn_ratings == 0, better=_BETTER_DIRECTIONS.get(key)
                )
                drawn_ratings += 1
        fig.align_xlabels(axes[0])
    return fig


def adjusted_marginals_figure_files(
    responses: Iterable[dict],
    effects: Sequence[dict],
    *,
    regret_effect: dict | None = None,
    randomized_only: bool = True,
) -> dict[str, bytes] | None:
    """Render :func:`adjusted_marginals_figure` to the paper's PDF and PNG."""
    import matplotlib.pyplot as plt

    fig = adjusted_marginals_figure(
        responses, effects, regret_effect=regret_effect, randomized_only=randomized_only
    )
    if fig is None:
        return None
    try:
        with paper_rc():
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 220)}
    finally:
        plt.close(fig)


#: The height of :func:`adjusted_differences_figure`, in inches.
#:
#: The same height as every other figure here, because it is the same kind of
#: panel: one estimate per tick against a points axis. What makes this one
#: compact is its *width* — one column instead of two, one panel instead of
#: five — not a shortened axis.
DIFFERENCE_FIGURE_HEIGHT_IN = FIGURE_HEIGHT_IN

#: The y axis of the difference figure, worded as the difference panel of
#: :func:`adjusted_study_figure` words it — and wrapped the same way, since one
#: line of it is 79 pt of rotated text and the panel is not that tall.
#:
#: Just "points", though the ticks are not all the same kind of point: three
#: are rating points on the 0--100 scale the participant answered on and the
#: fourth is points of policy success rate. Naming both took 250 pt of a 252 pt
#: figure and overhung it, so it is the caption's to say — along with the
#: reading that follows from it, which is that each mark is read against *zero*
#: and two marks' distances from it are not comparable. The rule before the
#: decision tick is what the figure itself says about it.
_DIFFERENCE_AXIS_LABEL = "Graph $-$ Videos\n(points)"

#: The tick label per outcome, wrapped where the instrument's name does not fit
#: a quarter of a single column: four ticks share about 200 pt of axis here, so
#: "Temporal demand" at 62 pt on one line would sit under its neighbour.
_DIFFERENCE_TICKS = {
    "temporal_demand": "Temporal\ndemand",
    "effort": "Effort",
    "confidence": "Confidence",
    "regret": "Decision\nregret",
}


def _draw_differences(ax, rows: Sequence[tuple[str, str, dict]]) -> None:
    """One tick per outcome: the adjusted difference and its 95% HDI.

    *rows* are ``(outcome, label, effect)``, left to right, where *effect*
    carries ``mean``/``lower``/``upper``.

    The same panel as :func:`_draw_adjusted_effects` — marks over a points
    axis, zero drawn, the direction arrow in the tick label — because a reader
    coming from the other figures reads the y axis as the score and should not
    have to learn a second convention for the same quantity. This one adds the
    decision and drops everything that is not the contrast.
    """
    positions = list(range(len(rows)))
    means = [effect["mean"] for _outcome, _label, effect in rows]
    lowers = [effect["lower"] for _outcome, _label, effect in rows]
    uppers = [effect["upper"] for _outcome, _label, effect in rows]
    # Zero is what every mark is read against, so it is drawn like a spine
    # rather than like a gridline.
    ax.axhline(0, color=_MUTED, linewidth=0.8, zorder=1)
    ax.errorbar(
        positions,
        means,
        yerr=[
            [mean - lower for mean, lower in zip(means, lowers, strict=True)],
            [upper - mean for mean, upper in zip(means, uppers, strict=True)],
        ],
        fmt="o",
        color=PHASE_COLORS["graph"],
        ecolor=_INK,
        markersize=4.0,
        elinewidth=0.9,
        capsize=2.2,
        zorder=3,
    )
    ax.set_xticks(
        positions,
        _outcome_ticks(
            [label for _outcome, label, _effect in rows],
            [outcome for outcome, _label, _effect in rows],
        ),
    )
    ax.set_xlim(-_CATEGORY_MARGIN, len(rows) - 1 + _CATEGORY_MARGIN)
    extent = [0.0, *lowers, *uppers]
    pad = max(2.0, (max(extent) - min(extent)) * 0.08)
    ax.set_ylim(min(extent) - pad, max(extent) + pad)
    ax.set_ylabel(_DIFFERENCE_AXIS_LABEL, color=_INK)


def adjusted_differences_figure(effects: Sequence[dict], *, regret_effect: dict | None = None):
    """The four adjusted graph-minus-video differences, as one narrow figure.

    The estimates the paper's sentences quote, and nothing else: one row each
    for perceived time, effort, confidence and the decision, each the ``graph``
    coefficient of its own model with its 95% HDI. A row whose interval clears
    zero is a difference the model separates from none.

    One column and one panel, to be read beside the text rather than across the
    page — which is the difference between this figure and
    :func:`adjusted_study_figure`'s middle panel. That panel is three of these
    marks, in a figure that also carries the preference counts and sizes every
    panel for the row it shares; this is the contrast on its own, with the
    decision added, at a quarter of the page area.

    **The ticks are on one axis but not on one scale.** Three are rating points
    on the 0--100 scale the participant answered on and the fourth is points of
    policy success rate — see :data:`_DIFFERENCE_AXIS_LABEL`. Each mark is read
    against zero; two marks' distances from it are not comparable.

    ``None`` when nothing could be fit, as the other figures here return
    ``None`` rather than empty axes. *effects* are
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.continuous_effects`
    and *regret_effect*
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.decision_regret_effect`,
    fitted by the caller — a deployment without the retrained rollouts mounted
    cannot score a choice, and gets the three rating rows.
    """
    import matplotlib.pyplot as plt

    by_outcome = {str(effect.get("outcome", "")): effect for effect in effects}
    rows: list[tuple[str, str, dict]] = []
    for dimension in PAPER_DIMENSIONS:
        effect = by_outcome.get(dimension)
        if effect and effect.get("mean") is not None:
            rows.append((dimension, _DIFFERENCE_TICKS[dimension], effect))
    if regret_effect and regret_effect.get("mean") is not None:
        rows.append(("regret", _DIFFERENCE_TICKS["regret"], regret_effect))
    if not rows:
        return None

    with paper_rc():
        fig, ax = plt.subplots(
            figsize=(SINGLE_COLUMN_WIDTH_IN, DIFFERENCE_FIGURE_HEIGHT_IN),
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
        _draw_differences(ax, rows)
        if len(rows) > len(PAPER_DIMENSIONS):
            # A hairline before the decision tick, because the unit changes
            # there and nothing else in the figure says so.
            ax.axvline(len(rows) - 1.5, color="#dcdcdc", linewidth=0.5, zorder=0)
    return fig


def adjusted_differences_figure_files(
    effects: Sequence[dict], *, regret_effect: dict | None = None
) -> dict[str, bytes] | None:
    """Render :func:`adjusted_differences_figure` to the paper's PDF and PNG."""
    import matplotlib.pyplot as plt

    fig = adjusted_differences_figure(effects, regret_effect=regret_effect)
    if fig is None:
        return None
    try:
        with paper_rc():
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 220)}
    finally:
        plt.close(fig)


def study_figure_files(
    responses: Iterable[dict],
    regret_rows: Iterable[dict] = (),
    *,
    randomized_only: bool = False,
    regret_mark: str = BOX_MARK,
) -> dict[str, bytes] | None:
    """``{"pdf": …, "png": …}`` for these responses, or ``None`` if there is
    nothing to draw.

    The PNG is the on-page preview; the PDF is the file that goes into the
    paper. Both come off one figure so what a reviewer of the page sees is what
    they download, and the figure is closed either way — a Streamlit page that
    leaks a figure per rerun runs the process out of them.
    """
    import matplotlib.pyplot as plt

    fig = study_figure(
        responses, regret_rows, randomized_only=randomized_only, regret_mark=regret_mark
    )
    if fig is None:
        return None
    try:
        with paper_rc():
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 220)}
    finally:
        plt.close(fig)


#: What the regret figure can show, and what the page offers as its export.
#: Two readings of one quantity: every decision made off each display in one
#: distribution, and the same decisions split by task. "both" puts them in one
#: figure, which is the one to reach for unless the paper needs the space.
REGRET_FIGURE_MODES = {
    "both": "Pooled distribution, then one per task",
    "pooled": "Pooled distribution per display",
    "per_task": "One distribution per task",
}

#: The unit :func:`regret.regret_rows` reports, spelled out: a success rate is a
#: percentage and a difference of two of them is points, so "points" rather than
#: "%" is what keeps a reader from reading 12 as a relative change. Two lines so
#: the rotated label fits a 2.1-inch panel.
_REGRET_AXIS_LABEL = "Regret\n(success-rate points)"
#: The same unit, said of a mean: a panel drawn as :data:`MEAN_MARK` is one bar
#: per display and a reader has to know it is not the answers. It matters most
#: in the regret figure's ``both`` mode, where that bar can sit beside boxes of
#: the very answers it averages.
_MEAN_REGRET_AXIS_LABEL = "Mean regret\n(success-rate points)"


def _regret_panels(rows: Sequence[dict], mode: str) -> list[tuple[str, object, str]]:
    """``[(key, data, title)]`` for the requested mode, dropping empty panels.

    A display with no answers is not a panel: the selected version may have
    shown one task, or one display, and an axis with nothing on it reads as a
    plotting fault rather than as a study that did not run that arm.
    """
    panels: list[tuple[str, object, str]] = []
    if mode in ("both", "pooled"):
        pooled = regret.regret_by_display(rows)
        if any(pooled.values()):
            # Named for what it holds rather than for a statistic: every
            # decision made off that display, over all three tasks.
            panels.append(("pooled", pooled, "All tasks"))
    if mode in ("both", "per_task"):
        for task, by_phase in regret.regret_by_task(rows).items():
            if any(by_phase.values()):
                label = next(row["task_label"] for row in rows if row["task"] == task)
                panels.append((f"task:{task}", by_phase, label))
    return panels


def _mean_and_sem(values: Sequence[float]) -> tuple[float, float]:
    """The mean, and the standard error of it — 0 for a single answer.

    The error bar is over answers, which is what the panel counts: it says how
    precisely this set of decisions pins down the mean regret of decisions made
    off this display, and it is not a claim about a task's arms (those are 60
    rollouts each, and their own uncertainty is the reference table's business).
    """
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, (variance / n) ** 0.5


def _draw_regret_panel(
    ax,
    by_phase: dict[str, list[float]],
    title: str,
    *,
    mark: str = BOX_MARK,
    first: bool = True,
    show_values: bool = False,
) -> None:
    """One regret panel, drawn as *mark* — see :data:`REGRET_MARKS`.

    The one place the choice is made, so a page that offers it has a string to
    pass and every figure that draws regret answers to it the same way.
    """
    if mark not in REGRET_MARKS:
        raise ValueError(f"unknown regret mark: {mark!r}")  # noqa: TRY003
    if mark == MEAN_MARK:
        _draw_regret_means(ax, by_phase, title, first=first, show_values=show_values)
    else:
        _draw_regret_boxes(ax, by_phase, title, first=first)


def _draw_regret_means(
    ax,
    pooled: dict[str, list[float]],
    title: str,
    *,
    first: bool = True,
    show_values: bool = True,
) -> None:
    """Mean regret per display, with its standard error.

    The alternative mark to :func:`_draw_regret_boxes`, and the one to reach
    for when the figure is read at a glance — a talk slide, or a paper that
    quotes the mean in its sentences. What it cannot show is the shape: see
    :data:`REGRET_MARKS`.

    *show_values* prints each mean above its bar. On where the panel is the
    figure, whose whole job is to carry that number; off where it is a fifth of
    a row and the number is a third thing to read in it after the bar and the
    axis.

    *first* is the panel that carries the axis, as in
    :func:`_draw_regret_boxes`: a mean bar beside distributions of the answers
    it averages is a different quantity and keeps its own.
    """
    phases = [phase for phase in PHASE_ORDER if pooled.get(phase)]
    stats = [_mean_and_sem(pooled[phase]) for phase in phases]
    means = [mean for mean, _sem in stats]
    ax.bar(
        range(len(phases)),
        means,
        width=0.58,
        color=[_tint(PHASE_COLORS[phase], 0.45) for phase in phases],
        edgecolor=[PHASE_COLORS[phase] for phase in phases],
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        range(len(phases)),
        means,
        yerr=[sem for _mean, sem in stats],
        fmt="none",
        ecolor=_INK,
        elinewidth=0.7,
        capsize=1.8,
        capthick=0.7,
        zorder=4,
    )
    if show_values:
        for x, (mean, sem) in enumerate(stats):
            ax.annotate(
                f"{mean:.1f}",
                (x, mean + sem),
                xytext=(0, 1.8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=_INK,
            )
    ax.set_xticks(
        range(len(phases)),
        [f"{PHASE_TICKS[phase]}\nn={len(pooled[phase])}" for phase in phases],
    )
    ax.set_xlim(-0.6, len(phases) - 0.4)
    # Headroom for the printed value where there is one, and a little either
    # way so the tallest error bar does not touch the top of the panel.
    top = max([mean + sem for mean, sem in stats], default=1.0)
    ax.set_ylim(0, max(top * (1.3 if show_values else 1.1), 1.0))
    if first:
        ax.set_ylabel(_MEAN_REGRET_AXIS_LABEL)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.spines["left"].set_visible(False)
    _label(ax, title, better=_BETTER_DIRECTIONS["regret"])


def _draw_regret_boxes(ax, by_phase: dict[str, list[float]], title: str, *, first: bool) -> None:
    """One box and one dot per answer, per display — the decisions themselves.

    Used for the pooled panel and for each task's, which is what lets the two
    be read against each other: same mark, same axis, and the pooled panel is
    the task panels' answers in one pile rather than a summary of them.

    A mean with a standard error was the pooled panel for a while. It reads
    more easily and it hides the shape that matters here: regret takes one of
    three values per task (an arm set offers three options), so the pile is a
    stack at 0 and a stack at the cost of each wrong option, and an error bar
    around the average of that implies a spread between them that no
    participant could have answered.
    """
    phases = [phase for phase in PHASE_ORDER if by_phase.get(phase)]
    for x, phase in enumerate(phases):
        values = by_phase[phase]
        color = PHASE_COLORS[phase]
        parts = ax.boxplot(
            [values],
            positions=[x],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            zorder=3,
        )
        for box in parts["boxes"]:
            box.set(facecolor=_tint(color, 0.6), edgecolor=color, linewidth=0.8)
        for line in (*parts["whiskers"], *parts["caps"]):
            line.set(color=color, linewidth=0.8)
        for line in parts["medians"]:
            line.set(color=_INK, linewidth=1.1)
        # A task offers three options, so its regrets take three values and the
        # strip is nearly all ties — fanned by value, as in the rating panels.
        ax.scatter(
            [x + offset for offset in _fanned(values, 0.055)],
            values,
            s=3.6,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )
    ax.set_xticks(
        range(len(phases)),
        [f"{PHASE_TICKS[phase]}\nn={len(by_phase[phase])}" for phase in phases],
    )
    ax.set_xlim(-0.6, len(phases) - 0.4)
    # Zero is the floor of the axis, because regret is a distance to the best
    # option and an axis starting below it would imply an answer better than
    # the best one available exists. A caller drawing a row of these overrides
    # the top afterwards, on the shared axis, so every panel is on one scale.
    top = max((value for values in by_phase.values() for value in values), default=1.0)
    ax.set_ylim(0, max(top * 1.18, 1.0))
    if first:
        ax.set_ylabel(_REGRET_AXIS_LABEL)
    else:
        # One axis for the task panels, as in the rating row: they are the same
        # scale, and three repeats of it is ink that says nothing.
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.spines["left"].set_visible(False)
    # Once, under the panel that carries the axis — unlike the rating panels,
    # these are the same quantity on a shared axis, so one arrow is true of all
    # of them and four of them is four repeats of one fact.
    _label(ax, title, better=_BETTER_DIRECTIONS["regret"] if first else None)


def regret_figure(rows: Sequence[dict], mode: str = "both", *, pooled_mark: str = BOX_MARK):
    """The regret figure for *rows*, or ``None`` when there is nothing to draw.

    *rows* are :func:`behavior_graphs.viz.streamlit.user_study.regret.regret_rows`
    — answers already scored against the retrained policies — so this module
    never has to know which option maps to which policy, only how to draw the
    numbers. ``mode`` is a key of :data:`REGRET_FIGURE_MODES`.

    *pooled_mark* is how the pooled panel draws those answers
    (:data:`REGRET_MARKS`). The per-task panels are distributions either way —
    a mean of four answers per task is not a summary of anything — so this
    only chooses what the "all tasks" panel is. Drawn as the mean it stops
    sharing the task panels' axis: it is a different quantity from the boxes
    beside it, and a bar as tall as a box would otherwise invite the comparison.

    Zero is the floor of every panel: regret is a distance to the best option,
    so an axis that started below zero would imply an answer better than the
    best one exists.
    """
    import matplotlib.pyplot as plt

    if mode not in REGRET_FIGURE_MODES:
        raise ValueError(f"unknown regret figure mode: {mode!r}")  # noqa: TRY003
    rows = list(rows)
    panels = _regret_panels(rows, mode)
    if not panels:
        return None

    widths = [0.78 if key == "pooled" else 1.0 for key, _data, _title in panels]
    # One panel goes in one column: the pooled mode is two boxes, and at
    # double-column width they come out a third of a page wide each.
    width = DOUBLE_COLUMN_WIDTH_IN if len(panels) > 1 else SINGLE_COLUMN_WIDTH_IN
    with paper_rc():
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(width, FIGURE_HEIGHT_IN),
            gridspec_kw={"width_ratios": widths},
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.035, hspace=0.0)
        # Every distribution panel on one axis, the pooled one included when it
        # is drawn as one: it holds the same answers the task panels do, so a
        # box in it that looked taller than a box beside it would be saying
        # something about the scales rather than about the decisions. A pooled
        # *mean* is a different quantity and keeps its own axis.
        shared = [
            ax
            for ax, (key, _data, _title) in zip(axes[0], panels, strict=True)
            if key != "pooled" or pooled_mark == BOX_MARK
        ]
        for ax in shared[1:]:
            ax.sharey(shared[0])
        for index, (ax, (key, data, title)) in enumerate(zip(axes[0], panels, strict=True)):
            # (a), (b), … so the LaTeX caption has something to point at — but
            # not on a one-panel figure, where there is no (b) to tell it from.
            panel_title = f"({chr(ord('a') + index)}) {title}" if len(panels) > 1 else title
            _draw_regret_panel(  # type: ignore[arg-type]
                ax,
                data,
                panel_title,
                # Only the pooled panel takes the mark; a task's four answers
                # are not a mean of anything.
                mark=pooled_mark if key == "pooled" else BOX_MARK,
                # Which panels carry a labelled axis: the first of the shared
                # group, and a pooled mean, which is off on its own scale.
                first=(key == "pooled" and pooled_mark == MEAN_MARK)
                or bool(shared)
                and ax is shared[0],
                # The number is this panel's whole job where it is drawn as a
                # mean, unlike the study figure's fifth of a row.
                show_values=True,
            )
        # Set once, on the shared axis, and after every box is drawn: the tasks
        # have different regret ceilings (one arm set spans 33 points, another
        # 8), and a per-panel limit would let a reader compare box heights
        # across panels that turned out not to be on one scale.
        ceiling = max(
            (
                value
                for key, data, _title in panels
                if key != "pooled" or pooled_mark == BOX_MARK
                for values in data.values()  # type: ignore[union-attr]
                for value in values
            ),
            default=1.0,
        )
        if shared:
            shared[0].set_ylim(0, max(ceiling * 1.18, 1.0))
        fig.align_xlabels(axes[0])
    return fig


def regret_figure_files(
    rows: Sequence[dict], mode: str = "both", *, pooled_mark: str = BOX_MARK
) -> dict[str, bytes] | None:
    """``{"pdf": …, "png": …}`` for the regret figure, or ``None`` if empty."""
    import matplotlib.pyplot as plt

    fig = regret_figure(rows, mode, pooled_mark=pooled_mark)
    if fig is None:
        return None
    try:
        with paper_rc():
            return {
                "pdf": paper.rendered(fig, "pdf", 300),
                "png": paper.rendered(fig, "png", 220),
            }
    finally:
        plt.close(fig)
