"""The policy comparison as a paper figure.

What is pinned here is not how it looks — that is a judgement, made by looking
at it — but the promises it makes. It is exported at the width it will be
printed at, so its 8 pt type is 8 pt on the page. It embeds TrueType outlines,
because a Type 3 figure is what a submission gets bounced over. It carries no
timestamp, so an unchanged comparison re-exports unchanged. It draws the letters
the tests produced rather than any grouping of its own. And it has no
transparency in it, because that is the one property of a figure whose fate in a
print pipeline nobody can predict.
"""

from __future__ import annotations

import pytest

from behavior_graphs.stats.policy_comparison import PolicyOutcomes, compare_policies
from behavior_graphs.viz import comparison_figure, paper


def _result(*counts: tuple[str, int, int]):
    """A comparison of the given ``(name, successes, trials)`` policies."""
    return compare_policies([PolicyOutcomes(*count) for count in counts])


def _three():
    return _result(("alpha", 41, 60), ("beta", 32, 60), ("gamma", 12, 60))


def _letters(fig) -> list[str]:
    """The bold single-character labels — the compact letter display."""
    return [text.get_text() for text in fig.axes[0].texts]


# ── What the figure promises the paper ───────────────────────────────────────


def test_it_is_exported_at_the_width_it_is_asked_for():
    """Not scaled to it: type at 8 pt in a figure the LaTeX then shrinks is not
    8 pt on the page, which is the whole reason for sizing in inches."""
    fig = comparison_figure.violin_figure(_three())

    assert fig.get_figwidth() == pytest.approx(paper.SINGLE_COLUMN_WIDTH_IN)

    wide = comparison_figure.violin_figure(_three(), width_in=paper.DOUBLE_COLUMN_WIDTH_IN)
    assert wide.get_figwidth() == pytest.approx(paper.DOUBLE_COLUMN_WIDTH_IN)


def test_the_same_comparison_exports_the_same_bytes():
    """No timestamp in it, so a figure that has not changed does not turn up as
    a diff in the paper repo."""
    first = comparison_figure.violin_figure_files(result=_three())
    second = comparison_figure.violin_figure_files(result=_three())

    assert first is not None and second is not None
    assert first["pdf"] == second["pdf"]
    assert b"CreationDate" not in first["pdf"]


def test_the_text_is_text_in_an_embedded_truetype_font():
    """``/FontFile2`` is the TrueType font program ``pdf.fonttype: 42``
    produces and IEEE PDF eXpress wants; ``/Type3`` is the fallback it rejects,
    and matplotlib takes it silently."""
    files = comparison_figure.violin_figure_files(result=_three())

    assert files is not None
    assert b"/FontFile2" in files["pdf"]
    assert b"/Type3" not in files["pdf"]


def test_there_is_no_transparency_in_it():
    """A mixed colour is what survives a print pipeline predictably; an alpha is
    what the pipeline gets to decide about. ``violinplot`` fills at 0.3 alpha by
    default, so this is a live risk rather than a hypothetical one."""
    fig = comparison_figure.violin_figure(_three())

    for collection in fig.axes[0].collections:
        for face in collection.get_facecolor():
            assert face[3] == pytest.approx(1.0)
    # And the pale fill is the accent mixed toward white, not the accent faded.
    expected = comparison_figure._mix(comparison_figure.ACCENT, comparison_figure.FILL_WEIGHT)
    faces = {tuple(round(v, 4) for v in c.get_facecolor()[0][:3]) for c in fig.axes[0].collections}
    assert faces == {tuple(round(v, 4) for v in expected)}


# ── It draws the comparison the page computed ────────────────────────────────


def test_the_letters_are_the_ones_the_tests_produced():
    """Read off the letters, not guessed from whether two intervals overlap —
    which is the fallacy the bar chart invites and the reason the violins are
    the default. Nothing about the grouping is decided here."""
    result = _three()

    fig = comparison_figure.violin_figure(result)

    assert _letters(fig) == [result.letters[i] for i in result.order]


def test_a_policy_with_no_letter_is_marked_rather_than_left_blank():
    """An empty letter is a real state — a policy the correction separated from
    nothing — and a gap in the row would read as a drawing bug."""
    result = _three()
    result.letters[result.order[0]] = ""

    assert _letters(comparison_figure.violin_figure(result))[0] == "—"


def test_the_letters_can_be_left_off():
    fig = comparison_figure.violin_figure(_three(), show_letters=False)

    assert _letters(fig) == []


def test_every_policy_gets_a_violin_in_the_order_the_result_gives():
    """Best first, and the order is the result's: the figure does not re-sort a
    comparison behind the page's back."""
    result = _result(("low", 5, 60), ("high", 50, 60), ("mid", 30, 60))

    fig = comparison_figure.violin_figure(result)

    assert len(fig.axes[0].collections) == 3
    centres = [c.get_paths()[0].vertices[:, 0].mean() for c in fig.axes[0].collections]
    assert centres == sorted(centres)  # drawn left to right, in result.order
    assert [result.labels[i] for i in result.order] == ["high", "mid", "low"]


def test_a_policy_with_no_successes_is_still_drawn():
    """Its posterior is the lopsided bump against 0%, which is exactly what the
    quantile grid is for — the analytic density has a pole there and would
    collapse every other violin in the figure to a hairline. Dropping it would
    hide that the dataset carries no scored rollouts."""
    result = _result(("scored", 30, 60), ("unscored", 0, 60))

    fig = comparison_figure.violin_figure(result)

    assert len(fig.axes[0].collections) == 2
    lowest = fig.axes[0].collections[-1].get_paths()[0].vertices[:, 1]
    assert lowest.min() >= 0.0, "no posterior mass at an impossible success rate"
    assert lowest.max() < 0.3


def test_the_median_and_the_interval_are_drawn_over_each_violin():
    result = _three()

    fig = comparison_figure.violin_figure(result)

    # One interval line and one median marker per policy.
    lines = fig.axes[0].lines
    assert len(lines) == 2 * len(result.order)
    interval, median = lines[0], lines[1]
    low, high = result.intervals[result.order[0]]
    assert list(interval.get_ydata()) == pytest.approx([low, high])
    assert list(median.get_ydata()) == pytest.approx([result.medians[result.order[0]]])


def test_an_empty_comparison_draws_nothing():
    """``None``, not an empty axes: a blank page in a paper reads as a plotting
    bug rather than as an empty result."""
    result = _three()
    result.order.clear()

    assert comparison_figure.violin_figure(result) is None
    assert comparison_figure.violin_figure_files(result=result) is None


# ── A row of groups ──────────────────────────────────────────────────────────


def test_a_row_of_groups_shares_one_rate_axis():
    """The point of the figure: three tasks at one scale, so a reader sees which
    is the hard one without doing arithmetic between two figures."""
    groups = [_three(), _result(("x", 5, 60), ("y", 9, 60))]

    fig = comparison_figure.violin_panels_figure(groups, titles=["Set table", "Cooking"])

    first, second = fig.axes
    assert first.get_ylim() == second.get_ylim()
    # Labelled once, on the left — what makes it read as one scale rather than
    # two that happen to agree.
    assert first.get_ylabel() == "Success rate"
    assert second.get_ylabel() == ""
    assert not any(label.get_visible() for label in second.get_yticklabels())


def test_each_panel_is_lettered_and_named_below_its_axes():
    """Where a subcaption goes, and where the figure's own caption reads up
    from — so what the LaTeX refers to sits next to it."""
    fig = comparison_figure.violin_panels_figure(
        [_three(), _three(), _three()], titles=["Set table", "Cooking", "Prepare tray"]
    )

    assert [ax.get_xlabel() for ax in fig.axes] == [
        "(a) Set table",
        "(b) Cooking",
        "(c) Prepare tray",
    ]


def test_a_letter_keeps_clear_of_its_violin_however_short_the_figure_is():
    """The gap is in points, not a share of the axes: the row is drawn at
    whatever height a paper has room for, and the same fraction is a comfortable
    gap at 2.6 inches and almost none at 1.6 — which is how a letter ended up
    sitting on a violin that reached 97%."""
    # An arm at 60/60 puts its violin's mass hard against the top of the axis,
    # which is the case that has no room to spare.
    result = _result(("ceiling", 60, 60), ("middling", 30, 60))

    clearances = []
    for height in (2.6, 1.6, 1.2):
        with paper.paper_rc(paper.CHART_RC):
            fig = comparison_figure.violin_panels_figure([result], titles=["T"], height_in=height)
            try:
                ax = fig.axes[0]
                renderer = fig.canvas.get_renderer()
                top = max(
                    path.vertices[:, 1].max()
                    for collection in ax.collections
                    for path in collection.get_paths()
                )
                letter_bottom = ax.texts[0].get_window_extent(renderer).y0
                data_top = ax.transData.transform((0, top))[1]
                clearances.append((letter_bottom - data_top) * 72.0 / fig.dpi)
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)

    assert all(gap > 0 for gap in clearances), "a letter is sitting on its violin"
    # Bounded below by the gap the constant asks for, at every height — and not
    # collapsing as the figure gets shorter, which a fractional headroom did.
    assert min(clearances) >= comparison_figure.LETTER_GAP_PT * 0.9
    assert max(clearances) - min(clearances) < comparison_figure.LETTER_GAP_PT


def test_the_letters_cost_margin_and_not_plot_at_any_height():
    """They sit above the frame, and the axes shrink to make room. Two earlier
    versions took the room out of the *data range* instead — first as a fixed
    fraction of the axis, then as a computed one — and both spent a fifth of a
    1.2-inch panel on a row of single characters."""
    result = _result(("ceiling", 60, 60), ("middling", 30, 60))

    for height in (2.6, 1.4, 1.2, 1.0):
        with paper.paper_rc(paper.CHART_RC):
            fig = comparison_figure.violin_panels_figure([result], titles=["T"], height_in=height)
            try:
                ax = fig.axes[0]
                renderer = fig.canvas.get_renderer()
                # The plot area is all data: 100% is the top of it, whatever the
                # letters need.
                assert ax.get_ylim()[1] == pytest.approx(1.03)
                axes_top = ax.get_window_extent(renderer).y1
                for text in ax.texts:
                    box = text.get_window_extent(renderer)
                    assert box.y0 >= axes_top - 0.5, f"letter over the plot at {height} in"
                    assert box.y1 <= fig.bbox.y1, f"letter off the figure at {height} in"
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)


def test_a_panel_with_no_title_keeps_its_letter():
    fig = comparison_figure.violin_panels_figure([_three(), _three()])

    assert [ax.get_xlabel() for ax in fig.axes] == ["(a)", "(b)"]


def test_a_panel_gets_width_in_proportion_to_the_policies_in_it():
    """Otherwise a violin is wider in the two-policy panel than the four-policy
    one, which tells the eye that one task was measured more coarsely."""
    wide, narrow = _three(), _result(("x", 5, 60), ("y", 9, 60))

    fig = comparison_figure.violin_panels_figure([wide, narrow], width_in=7.16)

    first, second = (ax.get_position().width for ax in fig.axes)
    assert first > second
    assert first / second == pytest.approx(3 / 2, rel=0.15)


def test_the_shared_axis_can_be_turned_off():
    """For a row whose panels are not comparable — a rate and a rate of
    something else."""
    fig = comparison_figure.violin_panels_figure([_three(), _three()], share_y=False)

    assert all(ax.get_ylabel() == "Success rate" for ax in fig.axes)


def test_a_group_with_nothing_in_it_is_left_out_rather_than_drawn_empty():
    """A blank panel in a row reads as a plotting bug; a row of blank panels is
    a figure that should not have been exported at all."""
    empty = _three()
    empty.order.clear()

    fig = comparison_figure.violin_panels_figure([_three(), empty], titles=["Kept", "Gone"])

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "(a) Kept"
    assert comparison_figure.violin_panels_figure([empty]) is None
    assert comparison_figure.violin_panels_figure_files(results=[empty]) is None


def test_the_subcaptions_sit_on_one_line_however_the_names_wrap():
    """Each axes puts its x label under its own tick labels, so a panel whose
    policy names take a third line carries its subcaption a line lower than its
    neighbours — and a row of subcaptions at three heights is the first thing a
    reader notices about the figure."""
    # Four long names share one panel, so each gets a narrow slot and wraps;
    # two short ones share the other and do not.
    wrapping = _result(
        ("plus milk carton left", 46, 60),
        ("plus milk carton middle", 28, 60),
        ("plus milk carton right", 26, 60),
        ("baseline", 32, 60),
    )
    short = _result(("baseline", 37, 60), ("plus lid", 16, 60))

    fig = comparison_figure.violin_panels_figure(
        [wrapping, short], titles=["Set table", "Cooking"], width_in=paper.DOUBLE_COLUMN_WIDTH_IN
    )

    renderer = fig.canvas.get_renderer()
    # The panel with the wrapping names really does need more room, or this
    # would be asserting nothing.
    depths = [
        min(label.get_window_extent(renderer).y0 for label in ax.get_xticklabels())
        for ax in fig.axes
    ]
    assert depths[0] < depths[1]
    baselines = [ax.xaxis.label.get_window_extent(renderer).y0 for ax in fig.axes]
    assert baselines[0] == pytest.approx(baselines[1], abs=1.0)


def test_a_row_of_panels_re_exports_the_same_bytes():
    """The alignment above measures a rendered figure, so it is the one thing
    here that could have made the export depend on the order artists happened
    to be drawn in."""
    groups = [_three(), _result(("x", 5, 60), ("y", 9, 60))]

    first = comparison_figure.violin_panels_figure_files(results=groups, titles=["A", "B"])
    second = comparison_figure.violin_panels_figure_files(results=groups, titles=["A", "B"])

    assert first is not None and second is not None
    assert first["pdf"] == second["pdf"]


def test_a_row_is_a_double_column_figure_by_default():
    """It exists to be read across, so it gets the width of the page — unlike a
    single group, which goes beside the paragraph that argues about it."""
    assert comparison_figure.DEFAULT_PANELS_WIDTH_IN == paper.DOUBLE_COLUMN_WIDTH_IN
    assert comparison_figure.DEFAULT_WIDTH_IN == paper.SINGLE_COLUMN_WIDTH_IN

    files = comparison_figure.violin_panels_figure_files(results=[_three(), _three()])

    assert files is not None
    assert b"/FontFile2" in files["pdf"]
    assert b"/Type3" not in files["pdf"]
    assert b"CreationDate" not in files["pdf"]


# ── The labels under the violins ─────────────────────────────────────────────


def test_a_name_breaks_at_its_underscores():
    """A dataset name is one long token as often as it is words, and unbroken it
    is several slots wide and lands on its neighbours. An underscore is the
    break the name itself carries."""
    wrapped = comparison_figure._wrap_name("diffusion_policy_set_table_baseline", 19)

    assert wrapped == "diffusion_policy_\nset_table_baseline"
    # Every character survives, so the reader can still tell which run it is.
    assert wrapped.replace("\n", "") == "diffusion_policy_set_table_baseline"


def test_a_name_with_no_break_in_it_is_left_wide():
    """Splitting "liberoobject" into "libero-object" invents a boundary that is
    not there, and costs more legibility than the width does. A name with
    nowhere to break runs long instead — the figure gets wider, not wronger."""
    assert comparison_figure._wrap_name("liberoobject", 4) == "liberoobject"


def test_the_last_line_allowed_takes_the_overflow():
    """A reader can read a wide label and cannot recover a truncated one."""
    wrapped = comparison_figure._wrap_name("a_b_c_d_e_f_g_h", 3)

    assert len(wrapped.split("\n")) == comparison_figure.NAME_MAX_LINES
    assert wrapped.replace("\n", "") == "a_b_c_d_e_f_g_h"


def test_a_narrower_figure_gives_a_name_fewer_characters():
    """The wrap width is the slot each policy actually gets, so the same
    comparison at a single column breaks its names earlier than at a double."""
    single = comparison_figure._tick_chars(paper.SINGLE_COLUMN_WIDTH_IN, 4)
    double = comparison_figure._tick_chars(paper.DOUBLE_COLUMN_WIDTH_IN, 4)

    assert single < double
    assert comparison_figure._tick_chars(paper.SINGLE_COLUMN_WIDTH_IN, 12) >= 6


def test_the_rollout_count_rides_under_the_name():
    """The denominator every reading of this figure depends on."""
    fig = comparison_figure.violin_figure(_result(("alpha", 3, 7), ("beta", 5, 9)))

    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert any("$n$=7" in label for label in labels)
    assert any("$n$=9" in label for label in labels)

    bare = comparison_figure.violin_figure(
        _result(("alpha", 3, 7), ("beta", 5, 9)), show_counts=False
    )
    assert all("$n$=" not in t.get_text() for t in bare.axes[0].get_xticklabels())


def test_the_count_is_left_off_when_it_is_the_same_for_everything():
    """ "n=60" under each of four names is the same four words repeated, and it
    belongs in the caption. The test is over the whole figure, not per panel: a
    task rolled out 30 times beside one rolled out 60 has to say so even though
    neither panel varies on its own."""
    same = _result(("a", 30, 60), ("b", 20, 60))
    fewer = _result(("c", 10, 30), ("d", 5, 30))

    def labels(fig):
        return [text.get_text() for ax in fig.axes for text in ax.get_xticklabels()]

    # One panel, every arm at 60: no counts.
    assert all("$n$=" not in label for label in labels(comparison_figure.violin_figure(same)))
    # A row whose panels agree with each other: still none.
    row = comparison_figure.violin_panels_figure([same, same])
    assert all("$n$=" not in label for label in labels(row))
    # A row whose panels disagree: every panel says its own n, including the
    # one that does not vary internally.
    mixed = comparison_figure.violin_panels_figure([same, fewer])
    assert sum("$n$=60" in label for label in labels(mixed)) == 2
    assert sum("$n$=30" in label for label in labels(mixed)) == 2


def test_the_rate_axis_is_a_percentage_of_the_whole():
    """A success rate, not a count — and the axis spans the whole range, so two
    figures of different comparisons can be read against each other."""
    fig = comparison_figure.violin_figure(_three())

    ax = fig.axes[0]
    assert ax.get_ylim()[0] < 0.0
    assert ax.get_ylim()[1] >= 1.0
    assert ax.get_ylabel() == "Success rate"
    assert ax.yaxis.get_major_formatter().format_data(0.25) == "25%"


# ── The study's choices, under the violins ───────────────────────────────────


def _series(label: str, color: str, **picks: int) -> comparison_figure.ChoiceSeries:
    """One display's answers over policies named ``alpha``, ``beta``, ``gamma``."""
    return comparison_figure.ChoiceSeries(label=label, color=color, picks=picks)


def _graph(**picks: int):
    return _series("Graph", "#ff7f0e", **picks)


def _videos(**picks: int):
    return _series("Videos", "#1f77b4", **picks)


def _strips(fig, panels: int = 1) -> list:
    """The choice rows: everything after the panels in ``fig.axes``."""
    return list(fig.axes[panels:])


def _bars(ax) -> dict[float, float]:
    """``{x centre: height}`` of the row's bars."""
    return {round(p.get_x() + p.get_width() / 2, 6): round(p.get_height(), 6) for p in ax.patches}


def _annotations(ax) -> list[str]:
    return [text.get_text() for text in ax.texts]


def test_a_bar_sits_under_the_violin_it_belongs_to():
    """The whole reason the bars are in this figure rather than in a second one:
    the join between "this arm was picked" and "this arm's success rate" is the
    x position, and a reader does it by looking down."""
    result = _three()
    fig = comparison_figure.violin_figure(result, choices=[_graph(alpha=3, beta=1, gamma=0)])

    violins = [
        collection.get_paths()[0].vertices[:, 0].mean() for collection in fig.axes[0].collections
    ]
    (row,) = _strips(fig)
    assert violins == pytest.approx([0.0, 1.0, 2.0], abs=0.01)
    assert _bars(row) == {0.0: 0.75, 1.0: 0.25}
    # And the row is scaled to the panel's slots, not to its own bars.
    assert row.get_xlim() == fig.axes[0].get_xlim()


def test_a_bar_carries_the_count_its_share_is_of():
    """ "60%" of six answers and of sixty are the same bar, and the paper's
    denominators are the small ones."""
    fig = comparison_figure.violin_figure(_three(), choices=[_graph(alpha=3, beta=1, gamma=0)])
    (row,) = _strips(fig)

    assert "75% (3)" in _annotations(row)
    assert "25% (1)" in _annotations(row)


def test_an_option_nobody_picked_is_a_zero_and_not_an_absence():
    """A stub on the axis and a "0%": the option was there, and nobody took
    it — which is a result, not missing data."""
    fig = comparison_figure.violin_figure(_three(), choices=[_graph(alpha=3, beta=1, gamma=0)])
    (row,) = _strips(fig)

    assert "0%" in _annotations(row)
    assert 2.0 not in _bars(row)
    # Drawn at the third slot's own width, on the axis line.
    stub = next(line for line in row.lines if round(line.get_ydata()[0], 6) == 0.0)
    assert [round(x, 6) for x in stub.get_xdata()] == [
        2.0 - comparison_figure.CHOICE_BAR_WIDTH / 2,
        2.0 + comparison_figure.CHOICE_BAR_WIDTH / 2,
    ]


def test_a_policy_that_was_never_an_option_gets_a_dash_and_no_bar():
    """The baseline is in every one of these comparisons and was in nobody's
    question. A zero there would say it went unchosen, which is not a thing
    anybody could have done."""
    result = _result(("baseline", 30, 60), ("plus left", 45, 60))
    fig = comparison_figure.violin_figure(
        result,
        # No entry for the baseline at all — see `ChoiceSeries.picks`.
        choices=[comparison_figure.ChoiceSeries("Graph", "#ff7f0e", {"plus left": 4})],
    )
    (row,) = _strips(fig)

    assert comparison_figure.NOT_OFFERED_MARK in _annotations(row)
    assert "0%" not in _annotations(row)
    # One bar, in the offered policy's own slot — which is the leftmost here,
    # since the violins are drawn best-first and it is the better policy.
    offered = [result.labels[index] for index in result.order].index("plus left")
    assert list(_bars(row)) == [float(offered)]


def test_the_most_picked_option_is_outlined_and_the_others_are_not():
    """Where a display's answers went, said in ink rather than in a second hue,
    so the figure survives greyscale."""
    from matplotlib.colors import to_hex

    fig = comparison_figure.violin_figure(_three(), choices=[_graph(alpha=1, beta=3, gamma=0)])
    (row,) = _strips(fig)

    outlines = {
        round(p.get_x() + p.get_width() / 2, 6): to_hex(p.get_edgecolor()) for p in row.patches
    }
    assert outlines[1.0] == paper.INK
    assert outlines[0.0] == "#ff7f0e"


def test_tied_options_are_both_outlined():
    """ "The most-picked option" of two arms at two answers each is both of
    them, and picking one to mark would be the figure inventing a winner."""
    from matplotlib.colors import to_hex

    fig = comparison_figure.violin_figure(_three(), choices=[_graph(alpha=2, beta=2, gamma=0)])
    (row,) = _strips(fig)

    assert {to_hex(p.get_edgecolor()) for p in row.patches} == {paper.INK}


def test_each_display_gets_its_own_row_named_at_the_left():
    """Direct labelling rather than two colours and a legend: the hues that
    separate best on screen have nearly the same luminance in greyscale."""
    fig = comparison_figure.violin_figure(
        _three(), choices=[_graph(alpha=3, beta=1, gamma=0), _videos(alpha=1, beta=2, gamma=1)]
    )
    rows = _strips(fig)

    assert len(rows) == 2
    assert [row.get_ylabel() for row in rows] == ["Graph", "Videos"]


def test_a_display_with_no_answers_gets_no_row():
    fig = comparison_figure.violin_figure(
        _three(), choices=[_graph(alpha=3, beta=1, gamma=0), _videos()]
    )

    assert [row.get_ylabel() for row in _strips(fig)] == ["Graph"]


def test_the_policy_names_are_written_once_under_the_lowest_row():
    """A name under the violin and again under the bars is the same word twice;
    under the violin only it labels the middle of the column, not the bottom."""
    fig = comparison_figure.violin_figure(
        _three(), choices=[_graph(alpha=3, beta=1, gamma=0), _videos(alpha=1, beta=2, gamma=1)]
    )
    top, first_row, last_row = fig.axes

    def written(ax):
        return [t.get_text() for t in ax.get_xticklabels() if t.get_text()]

    assert written(top) == []
    assert written(first_row) == []
    assert written(last_row) == ["alpha", "beta", "gamma"]


def test_the_rows_are_the_same_displays_in_every_panel():
    """Aligned across panels, and a panel with nothing for a display keeps the
    row rather than promoting the next display into it."""
    fig = comparison_figure.violin_panels_figure(
        [_three(), _three()],
        titles=["Set table", "Cooking"],
        choices=[
            [_graph(alpha=2, beta=0, gamma=0), _videos(alpha=1, beta=1, gamma=0)],
            # This task drew no graph answers at all.
            [_videos(alpha=3, beta=0, gamma=0)],
        ],
        height_in=2.4,
    )
    rows = _strips(fig, panels=2)
    first, second = rows[:2], rows[2:]

    assert [row.get_ylabel() for row in first] == ["Graph", "Videos"]
    # Named on the left panel only: the rows are the same in every panel, so
    # naming them in each is the same two words three times.
    assert [row.get_ylabel() for row in second] == ["", ""]
    # The cooking panel's graph row is empty, and its videos row is the second.
    assert _bars(second[0]) == {}
    assert _bars(second[1]) == {0.0: 1.0}


def test_the_subcaption_hangs_off_the_lowest_row():
    """Under the policy names, where a subcaption goes — not under the violins
    with the bars drawn below it."""
    fig = comparison_figure.violin_panels_figure(
        [_three(), _three()],
        titles=["Set table", "Cooking"],
        choices=[[_graph(alpha=2, beta=0, gamma=0)], [_graph(alpha=1, beta=1, gamma=0)]],
        height_in=2.4,
    )
    panels, rows = list(fig.axes[:2]), _strips(fig, panels=2)

    assert [ax.get_xlabel() for ax in panels] == ["", ""]
    assert [ax.get_xlabel() for ax in rows] == ["(a) Set table", "(b) Cooking"]


def test_the_violins_keep_most_of_a_short_figure():
    """Two rows at their asked-for height leave a 1.2-inch figure half an inch
    of rate axis, so the rows give way instead."""
    fig = comparison_figure.violin_figure(
        _three(),
        choices=[_graph(alpha=3, beta=1, gamma=0), _videos(alpha=1, beta=2, gamma=1)],
        height_in=1.2,
    )
    panel, *rows = fig.axes

    heights = [ax.get_position().height for ax in (panel, *rows)]
    assert heights[0] / sum(heights) >= comparison_figure.MIN_VIOLIN_SHARE


def test_no_choices_draws_the_figure_it_always_drew():
    """The strip is another version of this figure, not a change to it: passing
    no answers has to give back the file the paper already has."""
    bare = comparison_figure.violin_figure_files(result=_three())
    empty = comparison_figure.violin_figure_files(result=_three(), choices=[])
    rows = comparison_figure.violin_panels_figure_files(
        results=[_three(), _three()], titles=["a", "b"]
    )
    rows_empty = comparison_figure.violin_panels_figure_files(
        results=[_three(), _three()], titles=["a", "b"], choices=[[], []]
    )

    assert bare is not None and bare["pdf"] == empty["pdf"]
    assert rows is not None and rows["pdf"] == rows_empty["pdf"]


def test_a_figure_with_choices_re_exports_the_same_bytes():
    def draw():
        return comparison_figure.violin_figure_files(
            result=_three(),
            choices=[_graph(alpha=3, beta=1, gamma=0), _videos(alpha=1, beta=2, gamma=1)],
        )

    first, second = draw(), draw()
    assert first is not None
    assert first["pdf"] == second["pdf"]
    assert first["pdf"] != comparison_figure.violin_figure_files(result=_three())["pdf"]


def test_there_is_no_transparency_in_the_bars_either():
    """The one property of a figure whose fate in a print pipeline nobody can
    predict — and a bar is the mark most likely to be given an alpha."""
    fig = comparison_figure.violin_figure(_three(), choices=[_graph(alpha=3, beta=1, gamma=0)])
    (row,) = _strips(fig)

    for patch in row.patches:
        assert patch.get_alpha() in (None, 1.0)
        assert patch.get_facecolor()[3] == 1.0
        assert patch.get_edgecolor()[3] == 1.0


def test_a_share_is_of_the_denominator_it_is_given():
    """A question whose options are not all drawn in the panel: the shares are
    of what that display was asked, so they do not add to 100%."""
    series = comparison_figure.ChoiceSeries("Graph", "#ff7f0e", {"alpha": 3}, answers=6)
    fig = comparison_figure.violin_figure(_three(), choices=[series])
    (row,) = _strips(fig)

    assert _bars(row) == {0.0: 0.5}
    assert "50% (3)" in _annotations(row)
