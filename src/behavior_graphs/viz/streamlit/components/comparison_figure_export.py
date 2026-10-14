"""The panel that offers the policy comparisons on screen as paper figures.

Sits under the group tabs on the Policy Comparison page. It offers the *same*
comparisons, not a second opinion about them: the results it is handed are the
results the tests produced, so every letter, interval and median on a figure is
the one in the table above it. What the controls here choose is what only a
figure has to answer — which column it goes in, how tall it is, and whether a
row of groups shares one rate axis. The drawing is
:mod:`behavior_graphs.viz.comparison_figure`.

Two layouts, because a paper wants both at different moments: **a row of panels
sharing the rate axis**, which is how three tasks are read against each other in
one figure, and **one figure per group**, for a paper that discusses them a page
apart. With a single group there is nothing to choose and the panel does not ask.

Only under the violins. The bar chart is offered on the page because it is the
convention an audience already reads, not because it is the better summary —
two standard errors overlapping is not a test — and a paper-ready export of it
would be this page endorsing the reading it exists to talk people out of.

A caller with the study's data-collection answers passes them as *choices* and
gets a third control: the strip of bars under the violins that says how many
participants asked for each policy's data, off which display
(:class:`~behavior_graphs.viz.comparison_figure.ChoiceSeries`). A checkbox rather
than a separate panel, because it is the same comparison either way — but it is
a *different file*, so the download's name says so and a paper repo ends up with
both rather than with one overwritten by the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import streamlit as st

from behavior_graphs.stats.policy_comparison import ComparisonResult
from behavior_graphs.viz import comparison_figure, paper
from behavior_graphs.viz.labels import slug
from behavior_graphs.viz.streamlit.components.paper_export import paper_figure_downloads

#: The width slider, in inches. Both column widths of the template are a whole
#: number of steps from the minimum, so the two sizes that have to be exact —
#: the ones a figure placed at ``\\textwidth`` has to hit or be rescaled by the
#: LaTeX — are reachable rather than approximable. See
#: :mod:`behavior_graphs.viz.paper`.
WIDTH_MIN_IN = 2.0
WIDTH_MAX_IN = 8.0
WIDTH_STEP_IN = 0.02

#: Where the size sliders start, per layout. A row exists to be read across, so
#: it gets the width of the page; a single group is a figure to put beside the
#: paragraph that argues about it.
ROW_LAYOUT = "row"
EACH_LAYOUT = "each"
LAYOUT_DEFAULTS = {
    ROW_LAYOUT: (
        comparison_figure.DEFAULT_PANELS_WIDTH_IN,
        comparison_figure.DEFAULT_PANELS_HEIGHT_IN,
    ),
    EACH_LAYOUT: (comparison_figure.DEFAULT_WIDTH_IN, comparison_figure.DEFAULT_HEIGHT_IN),
}
LAYOUT_NAMES = {
    ROW_LAYOUT: "One wide figure",
    EACH_LAYOUT: "One figure per group",
}

#: Kept for callers that want the single-group defaults by name.
DEFAULT_WIDTH_IN = comparison_figure.DEFAULT_WIDTH_IN
DEFAULT_HEIGHT_IN = comparison_figure.DEFAULT_HEIGHT_IN


def _result_state(result: ComparisonResult) -> list:
    """The numbers one panel draws, as something hashable.

    Every arm's counts, its letter, its median and its interval — the letters
    included, because a different correction changes them without changing a
    count and the figure would otherwise be served from the cache unchanged.
    """
    return [
        [
            result.labels[i],
            result.arms[i].successes,
            result.arms[i].trials,
            result.letters[i],
            round(result.medians[i], 9),
            [round(bound, 9) for bound in result.intervals[i]],
        ]
        for i in result.order
    ]


def _choice_state(panels: Sequence[Sequence[comparison_figure.ChoiceSeries]]) -> list:
    """The bars one strip draws, as something hashable.

    Every series' name, colour, denominator and picks: a response arriving while
    the page is open changes a count without changing a comparison, and the
    figure would otherwise be served from the cache with yesterday's bars on it.
    """
    return [
        [
            [series.label, series.color, series.total, sorted(series.picks.items())]
            for series in panel
        ]
        for panel in panels
    ]


def _fingerprint(
    results: Sequence[ComparisonResult],
    titles: Sequence[str],
    options: dict,
    choices: Sequence[Sequence[comparison_figure.ChoiceSeries]] = (),
) -> str:
    """A cache key for one drawing."""
    state = {
        "panels": [_result_state(result) for result in results],
        "titles": list(titles),
        "priors": [list(result.prior) for result in results],
        "credible": [result.credible_level for result in results],
        "options": options,
        "choices": _choice_state(choices),
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]


@st.cache_data(show_spinner=False, max_entries=8)
def _row_files(token: str, _payload: dict) -> dict[str, bytes] | None:
    """A row of panels, keyed on *token*.

    ``_payload`` is underscore-prefixed so Streamlit leaves it out of the hash:
    it holds result objects, which are not hashable, and *token* stands for
    them.
    """
    return comparison_figure.violin_panels_figure_files(**_payload)


@st.cache_data(show_spinner=False, max_entries=16)
def _single_files(token: str, _payload: dict) -> dict[str, bytes] | None:
    """One group's figure, keyed on *token*."""
    return comparison_figure.violin_figure_files(**_payload)


def comparison_figure_export_panel(
    results: Sequence[ComparisonResult],
    titles: Sequence[str] = (),
    *,
    choices: Sequence[Sequence[comparison_figure.ChoiceSeries]] | None = None,
    key: str = "policy_comparison",
    filename: str = "policy_comparison",
    expanded: bool = False,
) -> None:
    """Draw the export panel for the comparisons the page just drew.

    *results* and *titles* are one per group, in the order the page shows them.

    *choices* is one sequence of
    :class:`~behavior_graphs.viz.comparison_figure.ChoiceSeries` per group, also in
    that order — the study's data-collection answers, drawn as a row of bars per
    display under each panel. ``None`` (or a group with nothing in it) offers the
    figure without them.
    """
    given = list(choices or [])
    given += [()] * max(0, len(results) - len(given))
    # Filtered together, so a group with no policies in it takes its bars out of
    # the figure with it rather than shifting every later panel's bars one panel
    # to the left.
    kept = [(result, panel) for result, panel in zip(results, given, strict=False) if result.order]
    results = [result for result, _ in kept]
    panels = [list(panel) for _, panel in kept]
    titles = list(titles)[: len(results)] + [""] * max(0, len(results) - len(list(titles)))
    # Collapsed by default: on a page whose subject is the chart it sits under, the
    # export is secondary and a panel of controls in the way of it is worse than
    # a click. A page whose subject *is* the export opens it (`expanded=True`) —
    # there, a collapsed expander and no chart beside it reads as a section that
    # failed to draw.
    with st.expander("⬇ Export as a paper figure (PDF)", expanded=expanded):
        if not results:
            st.info("Nothing to draw: no group here has a policy in it.")
            return
        st.caption(
            "The violins above, redrawn for print: the paper's serif at the "
            "size it will be printed, embedded TrueType outlines, no "
            "transparency, and no timestamp — so re-exporting an unchanged "
            "comparison gives an unchanged file. The note about shared letters "
            "is left off on purpose; it belongs in the LaTeX caption."
        )
        if any(series.total for panel in panels for series in panel):
            st.caption(
                "The study's answers can ride under the violins as a row of "
                "bars per display — every bar with the count its share is of, "
                "since these denominators are single figures. **Say the "
                "denominator in the caption too**: a reader who reads the "
                "shares alone is reading percentages of four."
            )

        layout = ROW_LAYOUT if len(results) > 1 else EACH_LAYOUT
        if len(results) > 1:
            layout = st.radio(
                "Layout",
                (ROW_LAYOUT, EACH_LAYOUT),
                format_func=LAYOUT_NAMES.__getitem__,
                horizontal=True,
                key=f"{key}_fig_layout",
                help=(
                    "A row puts the groups at one scale, which is how a reader "
                    "sees that one task is the hard one without doing "
                    "arithmetic between two figures. Separate figures are for a "
                    "paper that discusses the groups a page apart."
                ),
            )

        default_width, default_height = LAYOUT_DEFAULTS[layout]
        # A strip of bars is height the violins would otherwise have had, so a
        # figure that has one starts taller by what the rows take: at the row
        # layout's own 1.2 in, two rows leave the rate axis half an inch and
        # `MIN_VIOLIN_SHARE` has to step in. The slider is keyed on whether
        # there are bars at all, so the two versions of the figure remember
        # their own sizes rather than inheriting each other's.
        rows = len({series.label for panel in panels for series in panel if series.total})
        if rows:
            default_height += rows * (comparison_figure.CHOICE_ROW_HEIGHT_IN + 0.06)
        left, right = st.columns(2)
        # Keyed on the layout, because a row and a single panel want genuinely
        # different sizes — switching should offer that layout's shape rather
        # than the one left over from the other.
        width_in = left.slider(
            "Width (in)",
            min_value=WIDTH_MIN_IN,
            max_value=WIDTH_MAX_IN,
            value=default_width,
            step=WIDTH_STEP_IN,
            key=f"{key}_fig_width::{layout}",
            help=(
                f"The size the figure is *placed* at, so its 8 pt type is 8 pt "
                f"on the page. {paper.SINGLE_COLUMN_WIDTH_IN} is this "
                f"template's single column and {paper.DOUBLE_COLUMN_WIDTH_IN} "
                "its double column — hit either exactly and the figure needs "
                "no `width=` in the LaTeX."
            ),
        )
        height_in = right.slider(
            "Height (in)",
            min_value=1.2,
            max_value=6.0,
            value=default_height,
            step=0.1,
            key=f"{key}_fig_height::{layout}::{bool(rows)}",
            help="Taller gives the violins more of the rate axis to spread over.",
        )
        show_letters = left.checkbox(
            "Letters above the violins",
            value=True,
            key=f"{key}_fig_letters",
            help=(
                "The compact letter display — the one thing this figure is for. "
                "Each group's letters are its own: they are corrected over that "
                "group's pairs and mean nothing next to another panel's."
            ),
        )
        show_counts = right.checkbox(
            "Rollout counts",
            value=True,
            key=f"{key}_fig_counts",
            help=(
                "`n=` under each policy: the denominator every reading depends "
                "on. Drawn only where it varies — if every policy in the figure "
                "was rolled out the same number of times, the number is the "
                "same words under every violin and belongs in the caption."
            ),
        )
        share_y = True
        if layout == ROW_LAYOUT:
            share_y = left.checkbox(
                "Shared rate axis",
                value=True,
                key=f"{key}_fig_share_y",
                help=(
                    "One axis for the row, labelled once on the left — what "
                    "makes the panels comparable at a glance. Off gives each "
                    "panel its own labelled axis."
                ),
            )
        show_choices = bool(rows)
        if show_choices:
            show_choices = right.checkbox(
                "Study choices under the violins",
                value=True,
                key=f"{key}_fig_choices",
                help=(
                    "A row of bars per display: how many participants asked for "
                    "each policy's data after reading the baseline through that "
                    "display, with the count beside every share. The baseline "
                    "carries a dash rather than a bar — nobody was offered "
                    "“collect nothing”. It is a different file from the "
                    "violins alone, so a paper can carry both."
                ),
            )
        drawn_choices = [panel if show_choices else [] for panel in panels]

        options = {
            "width_in": width_in,
            "height_in": height_in,
            "show_letters": show_letters,
            "show_counts": show_counts,
            "layout": layout,
            "share_y": share_y,
        }
        suffix = "_violins_choices" if show_choices else "_violins"
        if layout == ROW_LAYOUT:
            files = _row_files(
                _fingerprint(results, titles, options, drawn_choices),
                {
                    "results": list(results),
                    "titles": titles,
                    "choices": drawn_choices,
                    "width_in": width_in,
                    "height_in": height_in,
                    "show_letters": show_letters,
                    "show_counts": show_counts,
                    "share_y": share_y,
                },
            )
            if files is None:
                st.info("Nothing to draw: no group here has a policy in it.")
                return
            paper_figure_downloads(
                files,
                key=f"{key}_fig_row",
                filename=f"{filename}{suffix}",
                width_in=width_in,
                height_in=height_in,
            )
            return

        for position, (result, title) in enumerate(zip(results, titles, strict=False)):
            panel = drawn_choices[position] if position < len(drawn_choices) else []
            files = _single_files(
                _fingerprint([result], [title], options, [panel]),
                {
                    "result": result,
                    "choices": panel,
                    "width_in": width_in,
                    "height_in": height_in,
                    "show_letters": show_letters,
                    "show_counts": show_counts,
                },
            )
            if files is None:
                continue
            if title:
                st.caption(f"**{title}**")
            stem = f"{filename}_{slug(title)}{suffix}" if title else f"{filename}{suffix}"
            paper_figure_downloads(
                files,
                key=f"{key}_fig_{position}",
                filename=stem,
                width_in=width_in,
                height_in=height_in,
            )
