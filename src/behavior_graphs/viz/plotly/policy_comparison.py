"""Policy-comparison figures, each with a compact letter display.

Two ways to draw the same result:

  - :func:`violin_cld_figure` — the default, and what TRI's article recommends.
    One violin per policy: the *posterior* over its success rate, not a
    confidence interval.
  - :func:`bar_se_cld_figure` — the conventional bar chart with ±1 standard
    error, for talks and papers that expect it.

Both carry the CLD letters as direct labels above, so "these two are separated"
is read off the letters rather than guessed from whether two intervals overlap
— which is the fallacy the bar chart invites and the reason it is not the
default.

Statistics come in already computed, as a
:class:`behavior_graphs.stats.policy_comparison.ComparisonResult`; this module
only draws.
"""

from __future__ import annotations

import textwrap

import plotly.graph_objects as go

from behavior_graphs.stats.policy_comparison import ComparisonResult, posterior_sample_grid

#: Categorical slot 1 of the validated palette. One hue for every violin: the
#: x axis already encodes which policy is which, so colouring by policy would
#: be redundant, and colouring by letter group breaks down the moment a policy
#: carries two letters.
ACCENT = "#2a78d6"
ACCENT_FILL = "rgba(42, 120, 214, 0.30)"
#: Ink, not hue, for the summary marks drawn on top of the fill — they are an
#: annotation of the violin, not a second series.
INK = "#3d3d3d"
SURFACE = "#ffffff"
GRID = "rgba(0, 0, 0, 0.08)"
MUTED = "#6b6b6b"


def _tick_label(name: str, trials: int, max_chars: int = 22) -> str:
    """Policy name over its rollout count; long names wrap rather than collide.

    Dataset names are one long token as often as they are words, so neither
    long words nor hyphens are break points — splitting ``libero_object`` mid
    word would cost more legibility than the extra width does.
    """
    words = " ".join(name.split())  # stray double spaces would survive the wrap
    lines = textwrap.wrap(words, max_chars, break_long_words=False, break_on_hyphens=False) or [""]
    return "<br>".join(lines) + f"<br><i>n</i>={trials}"


def violin_cld_figure(
    result: ComparisonResult,
    *,
    height: int = 520,
    resolution: int = 250,
) -> go.Figure:
    """Posterior violins ordered best-first, with CLD letters above each.

    Policies with no scored rollouts are still drawn (their posterior is just
    the prior), because dropping them silently would hide that a selected
    dataset carries no outcome labels.

    *resolution* is quantiles per violin, and it is a payload budget rather than
    a fidelity one: every point is serialised into the page and re-sent on each
    rerun (8 policies is 122 kB at 1000, 42 kB at 250), while the rendered
    shapes are indistinguishable — the KDE smooths over the difference. The
    credible interval and median are computed exactly, not read off the grid,
    so no number on the figure depends on this.
    """
    fig = go.Figure()
    # The uniquified labels, not the raw names: they are the x-axis categories,
    # and two datasets sharing a display_name would otherwise stack into one
    # violin.
    labels = [result.labels[i] for i in result.order]

    for label, i in zip(labels, result.order):
        arm = result.arms[i]
        lo, hi = result.intervals[i]
        fig.add_trace(
            go.Violin(
                x0=label,
                y=posterior_sample_grid(arm, result.prior, resolution),
                name=label,
                # Clip the KDE to the quantile grid's own extent. The default
                # pads the span by two bandwidths, which both invents posterior
                # mass at impossible success rates (below 0, above 1) and, at
                # span [0, 1], draws every violin as a full-height hairline
                # through the region where the posterior has no mass at all.
                spanmode="hard",
                width=0.8,
                points=False,
                box_visible=False,
                meanline_visible=False,
                showlegend=False,
                fillcolor=ACCENT_FILL,
                line={"color": ACCENT, "width": 1.5},
                hoveron="violins",
                hoverinfo="text",
                text=(
                    f"<b>{arm.name}</b><br>"
                    f"{arm.successes}/{arm.trials} successful rollouts"
                    f" ({arm.rate:.1%})<br>"
                    f"posterior median {result.medians[i]:.1%}<br>"
                    f"{result.credible_level:.0%} credible {lo:.1%}–{hi:.1%}<br>"
                    f"group <b>{result.letters[i] or '—'}</b>"
                ),
            )
        )

    _add_summary_marks(fig, result, labels)
    _add_cld_labels(fig, result, labels)
    _apply_layout(fig, result, labels, height=height, y_range=(-0.03, 1.03))
    fig.update_layout(violinmode="overlay")
    return fig


def bar_se_cld_figure(
    result: ComparisonResult,
    *,
    height: int = 520,
    n_se: float = 1.0,
) -> go.Figure:
    """The same comparison as bars of the observed rate, ±\\ *n_se* standard errors.

    Offered because it is the convention every audience already reads, not
    because it is the better summary: the bar is the point estimate the tests
    were run on, and ``sqrt(p(1-p)/n)`` says nothing about whether two policies
    are separated — a pair whose bars overlap can be separated and a pair whose
    bars do not can fail to be. The letters remain the statistics; the bars are
    decoration around them.

    A policy at 0/n or n/n gets no whisker at all, which is the honest drawing
    of an SE that is exactly zero there and a good argument for the violins.
    """
    fig = go.Figure()
    labels = [result.labels[i] for i in result.order]
    rates = [result.arms[i].rate for i in result.order]
    errors = [n_se * result.arms[i].standard_error for i in result.order]

    fig.add_trace(
        go.Bar(
            x=labels,
            # A policy with no scored rollouts has a NaN rate: Plotly draws no
            # bar, which is the right picture — there is nothing to draw — and
            # the tick label still carries its n=0.
            y=rates,
            width=0.62,
            marker={"color": ACCENT_FILL, "line": {"color": ACCENT, "width": 1.5}},
            error_y={
                "type": "data",
                "array": errors,
                "color": INK,
                "thickness": 2,
                "width": 8,
            },
            showlegend=False,
            hoverinfo="text",
            text=None,
            hovertext=[
                (
                    f"<b>{result.arms[i].name}</b><br>"
                    f"{result.arms[i].successes}/{result.arms[i].trials} successful rollouts"
                    f" ({result.arms[i].rate:.1%})<br>"
                    f"± {n_se:g} SE = {result.arms[i].standard_error:.1%}<br>"
                    f"group <b>{result.letters[i] or '—'}</b>"
                )
                for i in result.order
            ],
        )
    )

    _add_cld_labels(fig, result, labels)
    # Bars are read against zero, so the axis has to start there; the top grows
    # only if a whisker would otherwise be clipped.
    top = max([1.03, *(r + e for r, e in zip(rates, errors) if r == r)])
    _apply_layout(fig, result, labels, height=height, y_range=(0.0, top))
    fig.update_yaxes(title=f"Success rate (±{n_se:g} SE)")
    return fig


def _apply_layout(
    fig: go.Figure,
    result: ComparisonResult,
    labels: list[str],
    *,
    height: int,
    y_range: tuple[float, float],
) -> None:
    """Shared chrome: category axis of policy names, percent y axis, margins."""
    fig.update_layout(
        height=height,
        # Top margin holds the CLD letter row and its caption, both drawn in
        # paper coordinates above the axes.
        margin={"l": 60, "r": 20, "t": 84, "b": 90},
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=labels,
        tickvals=labels,
        ticktext=[_tick_label(result.arms[i].name, result.arms[i].trials) for i in result.order],
        showgrid=False,
        linecolor=GRID,
        ticks="",
    )
    fig.update_yaxes(
        title="Success rate",
        range=list(y_range),
        tickformat=".0%",
        dtick=0.2,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        ticks="",
    )


def _add_summary_marks(fig: go.Figure, result: ComparisonResult, labels: list[str]) -> None:
    """Credible-interval whisker plus posterior-median dot, over each violin."""
    for label, i in zip(labels, result.order):
        lo, hi = result.intervals[i]
        fig.add_trace(
            go.Scatter(
                x=[label, label],
                y=[lo, hi],
                mode="lines",
                line={"color": INK, "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[result.medians[i] for i in result.order],
            mode="markers",
            marker={
                "color": INK,
                "size": 9,
                # Surface ring: the dot sits on the interval line and the fill.
                "line": {"color": SURFACE, "width": 2},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_cld_labels(fig: go.Figure, result: ComparisonResult, labels: list[str]) -> None:
    """The compact letter display, one label per policy, in a row above the axes."""
    for label, i in zip(labels, result.order):
        fig.add_annotation(
            x=label,
            xref="x",
            y=1.0,
            yref="paper",
            yanchor="bottom",
            text=f"<b>{result.letters[i] or '—'}</b>",
            showarrow=False,
            font={"size": 15, "color": INK},
        )
    # Above the letter row, not beside it: at x=0 paper the caption would sit
    # exactly where the leftmost letter lands once there are enough policies.
    fig.add_annotation(
        x=0,
        xref="paper",
        xanchor="left",
        y=1.11,
        yref="paper",
        yanchor="bottom",
        text="Policies sharing a letter are <i>not</i> statistically separated",
        showarrow=False,
        font={"size": 11, "color": MUTED},
    )
