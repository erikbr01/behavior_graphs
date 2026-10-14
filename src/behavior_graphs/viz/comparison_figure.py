"""The policy comparison as a figure that can go into the paper.

The page draws this in Plotly, for a reader who wants to hover a violin and be
told what is under the cursor. That is the right figure for a screen and the
wrong one for print: it comes out in the browser's sans-serif, at whatever size
the container happened to be, with tinted fills that are transparency rather
than colour. This draws the same comparison in matplotlib at the width it will
be printed at — see :mod:`behavior_graphs.viz.paper` for why each of the file's
properties is not cosmetic.

**Same statistics, same shapes.** The violin is the posterior over a policy's
success rate, drawn from the same evenly-spaced quantiles the Plotly figure uses
(:func:`~behavior_graphs.stats.policy_comparison.posterior_sample_grid`) and
smoothed the same way. Not the analytic Beta density, which would be the obvious
thing to reach for here and is wrong for a reason worth repeating: a Jeffreys
posterior at 0 or n successes has an integrable pole at the boundary, so
normalising the density by its maximum collapses every other violin in the
figure to a hairline. As quantiles the same posterior is perfectly ordinary.

matplotlib evaluates a violin's KDE over the data's own range, which is what
Plotly needs ``spanmode="hard"`` to do: without it the smoothing pads the span
and invents posterior mass at impossible success rates.

**The letters are the figure's point.** The compact letter display is drawn
above each violin as a direct label, so "these two are separated" is read off
the letters instead of guessed from whether two intervals overlap — the fallacy
the bar chart invites. The letters come from the tests, not from this drawing.

What the screen figure says and this one does not: the note that policies
sharing a letter are not separated. That sentence belongs in the LaTeX caption,
where the rest of the figure's prose is, and a figure carrying its own caption
says it twice — the same call
:mod:`behavior_graphs.viz.streamlit.user_study.figure_export` makes next door.

No transparency anywhere: the violin's pale fill is the accent mixed against
white here, not the accent drawn at 30% alpha, because a mixed colour is what
survives a print pipeline predictably.

**One panel per group, sharing the rate axis.** A paper comparing policies on
three tasks wants the three side by side at one scale, so a reader sees that
cooking is the hard one without doing arithmetic between two figures — that is
:func:`violin_panels_figure`. What it must not do is invite reading the letters
*across* panels: each group is its own family of tests, corrected over its own
pairs, and an ``a`` in one panel has nothing to do with an ``a`` in the next.
The panels are titled and lettered ``(a)``, ``(b)`` as separate figures for
exactly that reason, and the caption is where to say it in words.

Panels get width in proportion to how many policies they hold, so a violin is
the same width in every panel and the eye is not told that a two-policy task was
measured more coarsely than a four-policy one.

**The study's choices, under the violins.** Every policy but the baseline in
these panels is a policy retrained on data a study participant asked for, so
there is a second thing to say about each violin: how many people picked that
data, off which display. That is :class:`ChoiceSeries` — one strip per display
under the rate axis, a bar per policy, drawn at the violin's own x position so a
reader's eye goes straight down from "this arm was the best one" to "and this
many chose it". The alternative, a second figure keyed by policy name, makes the
reader do that join themselves.

Four properties of the strip are the reasoning, not the styling:

- **One row per display, labelled at the left**, rather than two colours and a
  legend. The figure has to survive being printed in greyscale, and the two
  hues that separate best on screen (:mod:`...user_study.figure_export`'s blue
  and orange) have nearly the same luminance.
- **Every bar carries its own count.** "60%" of six answers and of sixty are
  the same bar, and the paper's denominators are the small ones — so the
  annotation is ``60% (6)``, and a reader cannot take the share for a sample.
- **An option nobody picked is not an absence.** A zero share is drawn as a
  stub on the axis line, and a policy that was *never offered* — the baseline,
  which nobody was asked to choose — gets a muted dash instead. Left both
  blank, the figure would say the baseline went unchosen, which is not a thing
  anybody could have done.
- **One share scale across the whole figure**, so a bar in the cooking panel and
  a bar in the set-table panel mean the same height. The scale is the full
  0–100%: it needs no ticks, because each bar is labelled, and the rows have no
  frame at all for the same reason.

Pure: a result in, a ``matplotlib.figure.Figure`` out, no Streamlit — the
displays' colours and names come in with the series rather than being imported
from the study code, which is what keeps this module the paper's drawing of a
policy comparison rather than the study's. The panel that offers it lives in
:mod:`behavior_graphs.viz.streamlit.components.comparison_figure_export`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from behavior_graphs.stats.policy_comparison import ComparisonResult, posterior_sample_grid
from behavior_graphs.viz import paper

#: The violin outline. One hue for every policy: the x axis already says which
#: policy is which, so colouring by policy would be redundant, and colouring by
#: letter group breaks down the moment a policy carries two letters. The same
#: blue the page uses, so a reviewer looking at both sees one scheme.
ACCENT = "#2a78d6"

#: How far the fill is mixed toward white. The page draws the accent at 30%
#: alpha; this is that colour, computed rather than composited — see the module
#: docstring on why there is no alpha in the file.
FILL_WEIGHT = 0.30

#: Quantiles per violin. A fidelity budget rather than a payload one here (
#: nothing is serialised into a page), but there is no visible difference above
#: this once the KDE has smoothed it, and the numbers on the figure — the
#: median, the interval — are computed exactly rather than read off the grid.
RESOLUTION = 250

#: Violin width as a fraction of the slot between two policies.
VIOLIN_WIDTH = 0.8

#: The credible-interval whisker and the median dot, in points. Ink rather than
#: hue: they annotate the violin they sit on, and are not a second series.
INTERVAL_LW_PT = 1.0
MEDIAN_SIZE_PT = 3.4
#: A white ring around the median dot, so it reads against both the interval
#: line it sits on and the fill behind it.
MEDIAN_RING_PT = 0.7

#: Clear air between the top of the rate axis and the row of letters, in
#: points. In points, not as a fraction of the axes, because the row of panels
#: is drawn at whatever height a paper has room for: the same fraction is a
#: comfortable gap at 2.6 inches and almost none at 1.6, which is how a letter
#: ended up sitting on top of a violin that reached 97%.
LETTER_GAP_PT = 3.0

#: Air above the letters, in points, so the tallest of them does not sit on the
#: frame. The room *below* them is not a constant at all — see
#: :func:`_fit_letters`, which computes exactly how much the axis has to grow.
LETTER_PAD_PT = 1.0

#: Lines a policy name may wrap to under its violin. Three, because a dataset
#: name routinely has four or five underscore-separated parts and this figure is
#: a single column wide; the last line allowed takes the overflow rather than
#: the name being cut.
NAME_MAX_LINES = 3

#: Where a name may break: after an underscore, or after a space.
#:
#: The page's own tick labels break on neither — see ``_tick_label`` in
#: :mod:`behavior_graphs.viz.plotly.policy_comparison`, whose reasoning is that
#: splitting ``libero_object`` mid-word costs more legibility than the width
#: does. That is right on a screen, where the figure is as wide as the window.
#: In a 3.5-inch column it is not: ``diffusion_policy_set_table_baseline``
#: unbroken is three slots wide and lands on top of its neighbours. An
#: underscore is not mid-word anyway — it is the break the name itself carries,
#: and left at the end of a line it reads as the continuation it is.
_BREAK_AFTER = re.compile(r"(?<=[_ ])")

#: How much of a policy's slot its name may use, leaving a gutter so two full
#: labels do not touch.
_NAME_SLOT_SHARE = 0.9

#: Width in inches this is drawn at unless a caller says otherwise, and the
#: height that suits it: a single column, since a comparison of a handful of
#: policies is a figure to put beside text rather than across the page.
DEFAULT_WIDTH_IN = paper.SINGLE_COLUMN_WIDTH_IN
DEFAULT_HEIGHT_IN = 2.4

#: A row of panels, on the other hand, is a double-column figure: it exists to
#: be read across, so it gets the width of the page.
DEFAULT_PANELS_WIDTH_IN = paper.DOUBLE_COLUMN_WIDTH_IN
DEFAULT_PANELS_HEIGHT_IN = 1.2

#: A choice row's height, in inches — about two lines of body type, which is
#: what a row of bars annotated with ``60% (6)`` needs and no more. Taken out
#: of the figure height the caller asked for rather than added to it: a figure
#: is sized by where it goes on the page, and a strip that grew the figure would
#: be rescaled by the LaTeX along with the type.
CHOICE_ROW_HEIGHT_IN = 0.30

#: The violins keep at least this share of the axes area however many choice
#: rows are asked for. At two rows it binds below about 1.1 inches, where the
#: alternative is a rate axis shorter than the strip under it — a figure whose
#: subject has become the bars.
MIN_VIOLIN_SHARE = 0.45

#: Gap between the rows, in font-size units — ``tight_layout``'s own unit, so
#: it is given to the layout rather than to the grid: a ``GridSpec`` carrying an
#: explicit ``hspace`` is one ``tight_layout`` declares itself incompatible with
#: and then lays out anyway, warning once per export.
#:
#: Enough air that two rows of bars do not read as one stacked bar, which is the
#: misreading to avoid: the rows have two different denominators, so their
#: heights do not add.
CHOICE_H_PAD = 0.7

#: A choice bar's width, in slot fractions — narrower than the violin above it
#: (:data:`VIOLIN_WIDTH`), so the violin stays the widest mark in its slot and
#: the strip reads as an annotation of the panel rather than a second chart.
CHOICE_BAR_WIDTH = 0.62

#: How far a bar's fill is mixed toward white, with the display's own colour as
#: its outline — the violins' treatment (:data:`FILL_WEIGHT`), for the same
#: reason plus one: a bar pale enough to read black type on top of it can carry
#: its own ``60% (6)`` inside itself when it is too tall to carry it above.
CHOICE_FILL_WEIGHT = 0.45

#: The ``60% (6)`` on a bar, in points. Below the 7 pt of the axes, because it
#: is the most repeated text in the figure and the one a reader consults rather
#: than reads — but not below 6 pt, which is IEEE's floor for anything in a
#: figure.
CHOICE_LABEL_PT = 6.0

#: A bar at least this tall carries its label inside itself — in ink, which the
#: pale fill (:data:`CHOICE_FILL_WEIGHT`) is chosen to be readable under. Above
#: it the label would run past the top of the row and into the panel over it,
#: and clipping it is how a 100% bar ends up looking unlabelled.
CHOICE_INSIDE_SHARE = 0.55

#: The stub that marks an option nobody picked: a line on the axis at the bar's
#: own width, in the display's colour, in points.
CHOICE_ZERO_LW = 1.2

#: The modal bar's outline, in points — which option that display's answers went
#: to most, said in ink rather than in a second hue. Ties are all outlined:
#: "the most-picked option" of two arms at three answers each is both of them.
CHOICE_MODE_LW = 0.7

#: What a policy that was never an option gets instead of a bar: a muted dash,
#: low in its row, and the reason the strip can be read at all. The baseline is
#: in every one of these comparisons and was in nobody's question — see the
#: module docstring.
NOT_OFFERED_MARK = "—"

#: Where that dash sits, as a fraction of the row's height. Off the axis line,
#: so it is not read as a zero-height bar, and low enough that it is not read as
#: a bar of its own.
NOT_OFFERED_Y = 0.16


@dataclass(frozen=True)
class ChoiceSeries:
    """One display's data-collection answers, as picks per policy.

    Args:
        label: What the display is called at the left of its row — "Graph",
            "Videos". Direct labelling rather than a legend; see the module
            docstring.
        color: The bars' colour. Passed in rather than chosen here, so the
            figure and the live page use one scheme without this module
            importing the study's palette.
        picks: How many answers went to each policy, keyed by the label the
            figure draws it under — the same strings as ``result.labels``. It
            **must carry a zero for every option that was offered** and not
            picked: a policy missing from this mapping is drawn as "not
            offered", which is a different claim (see :data:`NOT_OFFERED_MARK`).
        answers: The denominator. Defaults to the picks' sum, which is right
            whenever every answer named one of the policies in the panel; give
            it explicitly for a question whose options are not all drawn here,
            and the shares will not add to 100%.
    """

    label: str
    color: str
    picks: Mapping[str, int]
    answers: int | None = None

    @property
    def total(self) -> int:
        """Answers the shares are of."""
        return int(self.answers if self.answers is not None else sum(self.picks.values()))

    def share(self, label: str) -> float | None:
        """*label*'s share of this display's answers, or ``None`` if not offered.

        ``None`` and ``0.0`` are the figure's two different marks, so they are
        two different return values here as well.
        """
        if label not in self.picks or not self.total:
            return None
        return int(self.picks[label]) / self.total


#: How a panel is named under its axes: a subcaption letter, then the group.
#: Below the axes rather than above, where a subcaption goes and where the
#: figure's own caption will be reading up from.
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

#: Clear air between the lowest tick label in the row and the subcaptions, as a
#: fraction of a panel's height. See :func:`_align_panel_labels`.
PANEL_LABEL_PAD = 0.045


def _mix(hex_color: str, weight: float) -> tuple[float, float, float]:
    """*hex_color* at *weight* over white, as an opaque colour."""
    from matplotlib.colors import to_rgb

    return tuple(1.0 - weight * (1.0 - channel) for channel in to_rgb(hex_color))


def _wrap_name(name: str, max_chars: int) -> str:
    """A policy name over its own violin, wrapped to *max_chars*.

    Greedy, breaking only where :data:`_BREAK_AFTER` allows, and never in the
    middle of a word: an underscore-separated name comes apart at its
    underscores and a phrase at its spaces. The last line allowed takes whatever
    is left over, however wide — a reader can read a wide label and cannot
    recover a truncated one.
    """
    text = " ".join(str(name).split())
    chunks = [chunk for chunk in _BREAK_AFTER.split(text) if chunk]
    if not chunks:
        return ""
    limit = max(6, max_chars)
    lines = [chunks[0]]
    for chunk in chunks[1:]:
        candidate = lines[-1] + chunk
        if len(candidate.rstrip()) <= limit or len(lines) >= NAME_MAX_LINES:
            lines[-1] = candidate
        else:
            lines.append(chunk)
    return "\n".join(line.rstrip() for line in lines)


def _uniform_trials(results: Sequence[ComparisonResult]) -> bool:
    """Does every policy in every one of these panels have the same rollout count?

    Then ``n=60`` under each name is the same four words repeated, and it belongs
    in the caption. Where the counts *differ* it is the denominator a reader
    needs at the violin and cannot get anywhere else, so the test is over the
    whole figure rather than per panel: cooking at 30 rollouts beside set table
    at 60 has to say so, even though neither panel varies on its own.
    """
    trials = {result.arms[index].trials for result in results for index in result.order}
    return len(trials) <= 1


def _tick_chars(width_in: float, policies: int) -> int:
    """Characters a name has room for under one violin.

    From the slot each policy gets and the average width of a character at the
    body size — about half its point size in a serif. An estimate, deliberately:
    the exact width needs a renderer, and being a character or two out here
    costs a wrapped line rather than a broken figure.
    """
    slot_pt = width_in * 72.0 / max(1, policies)
    return max(6, int(slot_pt * _NAME_SLOT_SHARE / (0.5 * float(paper.BASE_RC["font.size"]))))


def _slot_limits(policies: int) -> tuple[float, float]:
    """The x range of a panel holding *policies* slots.

    One function, because the violins and the choice rows under them have to
    agree on it to the pixel: a row scaled to its own bars puts them under the
    wrong violins, and by the time the figure is in a paper nobody is checking.
    """
    return -0.6, policies - 0.4


def _height_ratios(rows: int, height_in: float) -> list[float]:
    """Axes heights for a panel with *rows* choice rows under it.

    Each row wants :data:`CHOICE_ROW_HEIGHT_IN` of the figure, and gets less
    where that would leave the violins under :data:`MIN_VIOLIN_SHARE` of it.
    Ratios, so the gaps between the rows come out of them — a row is a little
    shorter than the constant asks for, which is what a fraction of a figure can
    promise without measuring the margins.
    """
    if rows <= 0:
        return [1.0]
    row = min(CHOICE_ROW_HEIGHT_IN / max(height_in, 0.1), (1.0 - MIN_VIOLIN_SHARE) / rows)
    return [1.0 - rows * row, *[row] * rows]


def _panel_grid(fig, widths: Sequence[float], rows: int, *, height_in: float, share_y: bool):
    """``(violin_axes, choice_axes)`` for one figure: a column per panel.

    The violin axes come first in ``fig.axes`` and the strips after, so a caller
    that reads ``fig.axes`` positionally finds the panels where they were before
    the strips existed.

    A shared rate axis is labelled once, on the left: ``sharey`` alone would
    leave every panel's tick labels drawn, which is what makes three panels read
    as three scales that happen to agree.
    """
    grid = fig.add_gridspec(
        1 + max(0, rows),
        len(widths),
        width_ratios=list(widths),
        height_ratios=_height_ratios(rows, height_in),
    )
    tops = []
    for column in range(len(widths)):
        ax = fig.add_subplot(grid[0, column], sharey=tops[0] if (share_y and tops) else None)
        if share_y and tops:
            ax.tick_params(labelleft=False)
        tops.append(ax)
    strips = [
        [fig.add_subplot(grid[row + 1, column]) for row in range(max(0, rows))]
        for column in range(len(widths))
    ]
    return tops, strips


def _choice_rows(choices: Sequence[Sequence[ChoiceSeries]]) -> list[str]:
    """The strip's rows, in the order the panels first mention them.

    One list for the whole figure rather than one per panel: the rows have to
    line up across panels, or "Graph" is the top row in one and the bottom row
    in the next. A display that answered nothing anywhere gets no row at all;
    one that answered nothing *in this panel* keeps its row, empty.
    """
    rows: list[str] = []
    for panel in choices:
        for series in panel:
            if series.total and series.label not in rows:
                rows.append(series.label)
    return rows


def _aligned(panel: Sequence[ChoiceSeries], rows: Sequence[str]) -> list[ChoiceSeries | None]:
    """*panel*'s series in *rows* order, with ``None`` where it has none."""
    by_label = {series.label: series for series in panel if series.total}
    return [by_label.get(row) for row in rows]


def _draw_choice_row(
    ax,
    series: ChoiceSeries | None,
    row_label: str,
    labels: Sequence[str],
    *,
    show_name: bool,
    show_policies: bool,
) -> None:
    """One display's bars under one panel's violins.

    *labels* are the panel's policies in the order its violins are drawn, so bar
    and violin share an x position. A ``None`` *series* is a display that
    answered nothing about this panel's task: the row is still drawn, empty,
    because the rows line up across panels.

    *show_policies* is for the lowest row, which carries the policy names for
    the whole column — the others draw the same slots with no labels under
    them. The row sets its own limits from :func:`_slot_limits` rather than
    letting matplotlib scale them to the bars: a row whose widest bar is the
    second policy would otherwise centre that bar under the first violin, which
    is the one failure of this figure a reader cannot see.
    """
    ax.set_xlim(*_slot_limits(len(labels)))
    ax.set_xticks(range(len(labels)))
    if not show_policies:
        ax.set_xticklabels([])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    # The strip's scale is the figure's, not this row's: see the module
    # docstring. So there is nothing for a grid line to say here, and the rc
    # turns one on for every axes.
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(paper.MUTED)
    ax.tick_params(axis="x", length=0)
    if show_name:
        # Horizontal, where the rate axis's label is rotated: a row a third of
        # an inch tall has no room for a rotated word, and the name reads as
        # this row's only if it sits on the row.
        ax.set_ylabel(
            row_label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=CHOICE_LABEL_PT + 1,
            labelpad=3.0,
        )
    if series is None:
        return

    shares = {label: series.share(label) for label in labels}
    modal = max((share for share in shares.values() if share), default=0.0)
    fill = _mix(series.color, CHOICE_FILL_WEIGHT)
    for position, label in enumerate(labels):
        share = shares[label]
        if share is None:
            ax.annotate(
                NOT_OFFERED_MARK,
                (position, NOT_OFFERED_Y),
                ha="center",
                va="center",
                fontsize=CHOICE_LABEL_PT,
                color=paper.MUTED,
            )
            continue
        picks = int(series.picks[label])
        if share > 0:
            ax.bar(
                position,
                share,
                width=CHOICE_BAR_WIDTH,
                facecolor=fill,
                # The display's hue outlines every bar; ink, heavier, outlines
                # the one its answers went to most.
                edgecolor=paper.INK if share >= modal else series.color,
                linewidth=CHOICE_MODE_LW if share >= modal else 0.7,
                zorder=2,
            )
        else:
            # Offered and picked by nobody: a stub on the axis, so the reader is
            # told the option was there rather than left to wonder.
            ax.plot(
                [position - CHOICE_BAR_WIDTH / 2, position + CHOICE_BAR_WIDTH / 2],
                [0.0, 0.0],
                color=series.color,
                linewidth=CHOICE_ZERO_LW,
                solid_capstyle="butt",
                zorder=2,
            )
        inside = share >= CHOICE_INSIDE_SHARE
        ax.annotate(
            f"{100 * share:.0f}% ({picks})" if picks else "0%",
            (position, share),
            xytext=(0.0, -1.5 if inside else 1.5),
            textcoords="offset points",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=CHOICE_LABEL_PT,
            color=paper.INK,
            zorder=3,
        )


def violin_figure(
    result: ComparisonResult,
    *,
    choices: Sequence[ChoiceSeries] = (),
    width_in: float = DEFAULT_WIDTH_IN,
    height_in: float = DEFAULT_HEIGHT_IN,
    resolution: int = RESOLUTION,
    show_letters: bool = True,
    show_counts: bool = True,
    y_label: str = "Success rate",
):
    """The comparison as posterior violins, best-first, letters above.

    Args:
        result: What the page computed. Nothing is recomputed here except the
            violins' own quantiles, so the figure and the screen cannot disagree
            about a p-value, a letter or an interval.
        choices: The study's data-collection answers as a row of bars under the
            violins, one :class:`ChoiceSeries` per display. Empty draws the
            figure exactly as it was before the strip existed. A series with no
            answers in it is left out rather than drawn as an empty row.
        resolution: Quantiles per violin — see :data:`RESOLUTION`.
        show_letters: Draw the compact letter display above each violin. On,
            because it is the one thing the figure is for; off for a talk slide
            that explains the grouping some other way.
        show_counts: Write ``n=`` under each policy name — the denominator every
            reading of this figure depends on. Only where it varies: if every
            policy in the figure was rolled out the same number of times, the
            number is the same words repeated under each violin and belongs in
            the caption instead. Off suppresses it either way.
        y_label: Renamed for a comparison that is not about success — the same
            drawing serves any rate.

    Save it through :func:`violin_figure_files`, or from inside
    :func:`behavior_graphs.viz.paper.paper_rc` — a bare ``savefig`` gets the
    process's own style and Type 3 fonts with it.
    """
    import matplotlib.pyplot as plt

    if not result.order:
        return None

    rows = _choice_rows([choices])
    fig = plt.figure(figsize=(width_in, height_in))
    (ax,), (strips,) = _panel_grid(fig, [1.0], len(rows), height_in=height_in, share_y=False)
    labels = [result.labels[index] for index in result.order]
    # The rows before the violins: a row sets the ticks of its own slots, and
    # setting ticks clears the labels on them — so the policy names have to be
    # written after the row that carries them exists and is styled.
    for position, (strip, row, series) in enumerate(
        zip(strips, rows, _aligned(choices, rows), strict=True)
    ):
        _draw_choice_row(
            strip,
            series,
            row,
            labels,
            show_name=True,
            show_policies=position == len(strips) - 1,
        )
    _draw_violins(
        ax,
        result,
        resolution=resolution,
        show_letters=show_letters,
        show_counts=show_counts and not _uniform_trials([result]),
        max_chars=_tick_chars(width_in, len(result.order)),
        # The policy names go under the lowest row when there is a strip: the
        # names label the whole column of marks, not just the violin.
        label_ax=strips[-1] if strips else None,
    )
    _style_rate_axis(ax, y_label=y_label, show_letters=show_letters)
    fig.tight_layout(pad=0.2, h_pad=CHOICE_H_PAD)
    if show_letters:
        _reserve_letter_room(fig, [ax])
    return fig


def _draw_violins(
    ax,
    result: ComparisonResult,
    *,
    resolution: int,
    show_letters: bool,
    show_counts: bool,
    max_chars: int,
    label_ax=None,
) -> None:
    """One group's violins, marks, letters and tick labels, onto *ax*.

    Shared by the single figure and each panel of a row, so a panel cannot drift
    from the figure a reader approved on the page.

    *label_ax* takes the policy names, for a panel that has choice rows under it:
    a name written under the violin and again under the bars would be the same
    word twice, and written under the violin only it labels the middle of the
    column rather than the bottom of it.
    """
    order = list(result.order)
    positions = list(range(len(order)))
    fill = _mix(ACCENT, FILL_WEIGHT)

    for position, index in zip(positions, order, strict=False):
        grid = posterior_sample_grid(result.arms[index], result.prior, resolution)
        parts = ax.violinplot(
            [grid],
            positions=[position],
            widths=VIOLIN_WIDTH,
            showextrema=False,
            showmedians=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(fill)
            body.set_edgecolor(ACCENT)
            body.set_linewidth(0.7)
            # violinplot fills at 0.3 alpha by default, which is the one thing
            # this figure does not allow itself (see the module docstring).
            body.set_alpha(1.0)

        low, high = result.intervals[index]
        ax.plot(
            [position, position],
            [low, high],
            color=paper.INK,
            linewidth=INTERVAL_LW_PT,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            [position],
            [result.medians[index]],
            marker="o",
            markersize=MEDIAN_SIZE_PT,
            markerfacecolor=paper.INK,
            markeredgecolor="white",
            markeredgewidth=MEDIAN_RING_PT,
            zorder=4,
        )

    if show_letters:
        for position, index in zip(positions, order, strict=False):
            ax.annotate(
                result.letters[index] or "—",
                # Just above the frame, not inside the plot: a letter drawn
                # inside costs data range, and at 1.2 inches that was a fifth
                # of the panel spent on a row of single characters. The room
                # they need comes out of the margin instead — see
                # `_reserve_letter_room`.
                (position, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0.0, LETTER_GAP_PT),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                color=paper.INK,
                annotation_clip=False,
            )

    names = [
        _wrap_name(result.labels[index], max_chars)
        + (f"\n$n$={result.arms[index].trials}" if show_counts else "")
        for index in order
    ]
    ax.set_xticks(positions)
    ax.set_xlim(*_slot_limits(len(order)))
    named = ax if label_ax is None else label_ax
    if named is not ax:
        ax.tick_params(axis="x", length=0, labelbottom=False)
        named.set_xticks(positions)
        named.set_xlim(ax.get_xlim())
    named.set_xticklabels(names)


def _style_rate_axis(ax, *, y_label: str | None, show_letters: bool) -> None:
    """The rate axis: percentages of the whole, and room for the letter row."""
    from matplotlib.ticker import PercentFormatter

    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # Headroom for the letters, and a hair below zero so a violin pressed
    # against 0% is not drawn on the axis line itself.
    # 100% at the top and nothing above it: the letters live in the margin
    # (see `_reserve_letter_room`), so the plot area is all data.
    ax.set_ylim(-0.03, 1.03)


def _reserve_letter_room(fig, axes: Sequence) -> None:
    """Shrink the axes until the row of letters fits above them in the figure.

    The letters sit just above each panel's frame, so they cost *margin* rather
    than data range. Two earlier versions took it out of the range instead —
    first as a fixed fraction of the axis, then as a computed one — and both
    spent a fifth of a 1.2-inch panel on a row of single characters, which is
    the space this reclaims.

    One pass is exact: the letters are anchored to the axes' top edge, so
    lowering that edge by *d* lowers them by *d* too, and *d* is however far
    they currently overshoot the figure.
    """
    renderer = fig.canvas.get_renderer()
    overshoot = 0.0
    for ax in axes:
        for text in ax.texts:
            top = text.get_window_extent(renderer).y1
            overshoot = max(overshoot, top + LETTER_PAD_PT * fig.dpi / 72.0 - fig.bbox.y1)
    if overshoot <= 0:
        return
    shrink = overshoot / fig.bbox.height
    for ax in axes:
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width, max(box.height - shrink, 0.1)))


def violin_panels_figure(
    results: Sequence[ComparisonResult],
    *,
    titles: Sequence[str] = (),
    choices: Sequence[Sequence[ChoiceSeries]] = (),
    width_in: float = DEFAULT_PANELS_WIDTH_IN,
    height_in: float = DEFAULT_PANELS_HEIGHT_IN,
    resolution: int = RESOLUTION,
    show_letters: bool = True,
    show_counts: bool = True,
    y_label: str = "Success rate",
    share_y: bool = True,
):
    """Several comparisons as a row of panels, or ``None`` if none has arms.

    Args:
        results: One per group, drawn left to right in the order given.
        titles: What to call each panel. Drawn under its axes as ``(a) Cooking``
            — where a subcaption goes, and where the figure's caption reads up
            from. Missing titles leave just the letter.
        choices: One sequence of :class:`ChoiceSeries` per panel, in the same
            order as *results*, adding the study's data-collection answers as
            bars under each panel's violins. The rows are the displays named
            across all the panels, in the order the panels first mention them,
            so a display is the same row in every panel; a panel that has no
            answers for a display keeps the row and draws it empty. Empty draws
            the row of panels exactly as it was before the strip existed.
        share_y: One rate axis for the row, with tick labels only on the left.
            The point of the figure: three tasks at one scale can be read
            against each other. Off draws each panel with its own labelled axis,
            for a row whose panels are not comparable.

    The letters in one panel are that group's own — corrected over that group's
    pairs — and mean nothing next to another panel's. See the module docstring.
    """
    import matplotlib.pyplot as plt

    drawable = [result for result in results if result.order]
    if not drawable:
        return None
    kept_titles = [
        title for title, result in zip(titles, results, strict=False) if result.order
    ] + [""] * max(0, len(drawable) - len(list(titles)))
    panel_choices = [list(entry) for entry in choices] + [[]] * max(
        0, len(results) - len(list(choices))
    )
    kept_choices = [
        panel for panel, result in zip(panel_choices, results, strict=False) if result.order
    ]
    rows = _choice_rows(kept_choices)

    # Width by policy count, so a violin is the same width in every panel — the
    # alternative tells the eye that a two-policy task was measured coarsely.
    widths = [max(1, len(result.order)) for result in drawable]
    fig = plt.figure(figsize=(width_in, height_in))
    axes, strips = _panel_grid(fig, widths, len(rows), height_in=height_in, share_y=share_y)
    counted = show_counts and not _uniform_trials(drawable)
    label_axes = []
    for position, (ax, result) in enumerate(zip(axes, drawable, strict=False)):
        panel_width_in = width_in * widths[position] / sum(widths)
        panel_strips = strips[position]
        label_ax = panel_strips[-1] if panel_strips else ax
        label_axes.append(label_ax)
        labels = [result.labels[index] for index in result.order]
        # Before the violins, which write the policy names onto the lowest row:
        # a row sets the ticks of its own slots, and setting ticks clears the
        # labels on them.
        for row_position, (strip, row, series) in enumerate(
            zip(panel_strips, rows, _aligned(kept_choices[position], rows), strict=True)
        ):
            # The row names go on the left of the row of panels, beside the rate
            # axis's own label: the rows are the same displays in every panel, so
            # naming them in each is the same two words three times.
            _draw_choice_row(
                strip,
                series,
                row,
                labels,
                show_name=position == 0,
                show_policies=row_position == len(panel_strips) - 1,
            )
        _draw_violins(
            ax,
            result,
            resolution=resolution,
            show_letters=show_letters,
            show_counts=counted,
            max_chars=_tick_chars(panel_width_in, len(result.order)),
            label_ax=None if label_ax is ax else label_ax,
        )
        # A shared axis is labelled once, on the left, which is what makes it
        # read as one scale rather than three that happen to agree.
        first = position == 0
        _style_rate_axis(
            ax,
            y_label=y_label if (first or not share_y) else None,
            show_letters=show_letters,
        )
        letter = PANEL_LETTERS[position] if position < len(PANEL_LETTERS) else "?"
        title = kept_titles[position] if position < len(kept_titles) else ""
        label_ax.set_xlabel(f"({letter}) {title}".rstrip(), color=paper.INK, labelpad=3.0)
    fig.tight_layout(pad=0.2, w_pad=0.6, h_pad=CHOICE_H_PAD)
    if show_letters:
        # Before the labels are aligned: moving the axes moves their tick
        # labels, and the alignment is measured off where those end up.
        _reserve_letter_room(fig, list(axes))
    _align_panel_labels(
        fig,
        label_axes,
        # The gap under the names is the same air whether the label hangs off a
        # 1.2-inch panel or a 0.3-inch choice row, so with rows it is measured
        # off the panel and passed in pixels. See `_align_panel_labels`.
        pad_px=(
            PANEL_LABEL_PAD * axes[0].get_window_extent(fig.canvas.get_renderer()).height
            if rows
            else None
        ),
    )
    return fig


def _align_panel_labels(fig, axes: Sequence, *, pad_px: float | None = None) -> None:
    """Put every panel's ``(a) …`` on one line, whatever its tick labels do.

    Each axes places its own x label under its own tick labels, so a panel whose
    policy names wrap onto a third line carries its subcaption a line lower than
    its neighbours — and a row of subcaptions at three different heights is the
    first thing a reader notices about the figure.

    Aligned to the *lowest* of them, which is the one ``tight_layout`` already
    reserved room for; the others move down to meet it, so nothing lands outside
    the figure. Measured after the layout, since where a label sits is not known
    until then.

    *pad_px* is the clear air under the tick labels in pixels, for a figure whose
    labelling axes is not the panel: :data:`PANEL_LABEL_PAD` is a fraction of the
    axes that carries the label, and a third-of-an-inch choice row is short
    enough that the same fraction of it is a point of gap. The caller measures
    the pad off the violins and passes it; ``None`` keeps the fraction, which is
    what a figure with no choice rows has always used.
    """
    if len(axes) < 2:
        return
    renderer = fig.canvas.get_renderer()
    bottoms = []
    for ax in axes:
        labels = [label for label in ax.get_xticklabels() if label.get_text()]
        if labels:
            bottoms.append(min(label.get_window_extent(renderer).y0 for label in labels))
    if not bottoms:
        return
    lowest = min(bottoms)
    for ax in axes:
        # Its own axes fraction: the panels are the same height, but reading the
        # transform per axes keeps this correct if they ever are not.
        _, y = ax.transAxes.inverted().transform((0.0, lowest))
        height = ax.get_window_extent(renderer).height or 1.0
        pad = PANEL_LABEL_PAD if pad_px is None else pad_px / height
        ax.xaxis.set_label_coords(0.5, y - pad)


def violin_panels_figure_files(**kwargs) -> dict[str, bytes] | None:
    """``{"pdf": …, "png": …}`` for a row of panels, or ``None`` with nothing in
    any of them.

    Takes the same arguments as :func:`violin_panels_figure`.
    """
    return _rendered(violin_panels_figure, kwargs)


def violin_figure_files(**kwargs) -> dict[str, bytes] | None:
    """``{"pdf": …, "png": …}`` for one comparison, or ``None`` with no arms.

    The PNG is the on-page preview and the PDF is what goes into the paper; both
    come off one figure, so what a reader of the page approves is what they
    download. The figure is closed either way — a Streamlit page that leaks one
    per rerun runs the process out of them.

    Takes the same arguments as :func:`violin_figure`.
    """
    return _rendered(violin_figure, kwargs)


def _rendered(draw, kwargs: dict) -> dict[str, bytes] | None:
    """*draw* called inside the paper style, saved to both formats, then closed.

    The style has to be in force for the *save* as well as the drawing: a font
    family and ``pdf.fonttype`` are resolved when the file is written, so a
    figure built inside the context and saved outside it comes out in the
    default sans-serif with Type 3 fonts.
    """
    import matplotlib.pyplot as plt

    with paper.paper_rc(paper.CHART_RC):
        fig = draw(**kwargs)
        if fig is None:
            return None
        try:
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 300)}
        finally:
            plt.close(fig)
