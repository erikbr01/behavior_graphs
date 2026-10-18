"""The public page, run end to end.

``tests/test_reviewer_page.py`` tests what the page *decides* — which run each
dataset gets, which graphs the study pinned. This tests that the page renders,
that the numbers on it are the paper's, that every control it offers still
works, and the one property the site exists to keep: nothing identifying
reaches a reader.

The graph component does not run under ``AppTest`` (custom components do not),
so the stages that draw one are exercised for the page logic around them rather
than for the drawing. ``tests/browser/`` is where a real component gets clicked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# pytest puts this file's directory on sys.path (tests/ is not a package).
import apptest_compat  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "reviewer" / "Home.py"

#: Names and addresses that must not reach a page served under review. Checked
#: against every rendered string rather than by reading the source: a label
#: assembled at render time is exactly the one a source grep misses.
IDENTIFYING = (
    "stanford",
    "erbauer",
    "erik",
    "bauer",
    "github.com",
    "demo.behavior-graphs",
    "study.behavior-graphs",
    "@",
)


def _mounted() -> bool:
    """Whether this host has pipeline runs for the page to draw.

    The repo carries none, and the page's first section is a live behavior
    graph, so on a plain checkout it renders its "nothing is mounted" error and
    stops. Skipping is honest about that; the alternative is a suite that is
    green only on a machine with the data root set.
    """
    from behavior_graphs.viz.streamlit.data import run_index_rows

    try:
        return any(row["k"] is not None for row in run_index_rows())
    except Exception:  # pragma: no cover - an unreadable index is "not mounted"
        return False


needs_runs = pytest.mark.skipif(
    not _mounted(), reason="no pipeline runs mounted; set DATA_PATH or BEHAVIOR_GRAPHS_RUNS_ROOT"
)


@pytest.fixture(scope="module")
def page() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=900).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert not app.error, [error.value for error in app.error]
    return app


def _text(app: AppTest) -> str:
    parts = [block.value for block in app.markdown if not block.value.startswith("<style>")]
    parts += [block.value for block in app.caption]
    parts += [block.label for block in app.get("metric")]
    parts += [str(block.value) for block in app.get("metric")]
    return "\n".join(parts)


# ── The shape of the page ────────────────────────────────────────────────────


def test_it_is_one_page_and_not_a_set_of_tabs():
    """The four parts are one argument, and tabs let a reader land in the
    middle of it with no way to tell what they skipped."""
    source = APP.read_text()
    assert "st.navigation" not in source
    assert not (APP.parent / "_pages").exists()


@needs_runs
def test_it_tells_the_story_in_the_order_the_paper_argues_it(page):
    assert [block.value for block in page.header] == [
        "Graph Construction",
        "Evaluating Behavior Graphs for Active Data Collection",
        "Progress Estimation",
        "References",
    ]


@needs_runs
def test_it_opens_with_the_papers_own_overview_figure(page):
    assert len(apptest_compat.images(page)) == 1
    assert (APP.parent / "assets" / "overview.png").is_file()


def test_it_is_served_without_a_gate():
    """A link in a paper has to work unaided, and a reviewer cannot be handed a
    password out of band without giving up the anonymity the link is for."""
    assert "password_gate" not in APP.read_text()


@needs_runs
def test_nothing_identifying_reaches_a_reader(page):
    rendered = _text(page).lower()
    # A page that rendered nothing would pass this trivially, which is the one
    # way an anonymity check can be worse than none.
    assert len(rendered) > 3000
    for token in IDENTIFYING:
        assert token not in rendered, token


# ── How a graph is built ─────────────────────────────────────────────────────


@needs_runs
def test_the_construction_walkthrough_offers_the_studys_three_tasks(page):
    task = next(box for box in page.selectbox if box.label == "Task")
    # The paper's names, not the study variant's stage labels ("Cook the
    # meal"): every other figure on the page that names a task takes its name
    # from the comparison, and two names for one task is one too many.
    assert list(task.options) == ["Set table", "Prepare tray", "Cooking"]


@needs_runs
def test_every_construction_stage_is_drawn_inline(page):
    """All four down the page rather than behind a selector: a reader
    following a construction wants the step before and the step after on
    screen at once."""
    assert [block.value for block in page.subheader][:4] == [
        "1. Windowing",
        "2. Behavior Discovery",
        "3. Segments",
        "4. Behavior Graph",
    ]
    assert not [radio for radio in page.radio if radio.label == "Stage"]


@needs_runs
def test_switching_task_redraws_the_whole_walkthrough(page):
    task = next(box for box in page.selectbox if box.label == "Task")
    for name in ("Prepare tray", "Cooking"):
        app = task.set_value(name).run()
        assert not app.exception, (name, [str(e.value) for e in app.exception])


# ── The user study ───────────────────────────────────────────────────────────


@needs_runs
def test_it_states_the_three_numbers_the_paper_reports(page):
    """In the prose rather than on metric cards: the page reads as a paper,
    and a card is a dashboard."""
    text = _text(page)
    assert "16 of 18 participants chose the graph" in text
    assert "40%" in text and "38%" in text
    # And the paper's own title leads the page.
    assert "Unsupervised Graph Structures for Interpreting Robot Deployment Data" in text


@needs_runs
def test_the_regret_panel_is_the_assigned_stages_only(page):
    """The third stage is the participant's own pick and 16 of 18 picked the
    graph, so pooling it would make the display a result of the participant."""
    assert any("over 36 responses" in block.value for block in page.caption)


@needs_runs
def test_the_letters_are_given_their_reading(page):
    """Otherwise they are decoration: `a` is the best policy, not the leftmost."""
    assert "sharing a letter are not separated" in _text(page)


@needs_runs
def test_the_study_results_are_the_papers_own_figure(page):
    """One figure with the paper's five panels, not a rearrangement of them
    across four charts and a row of cards."""
    assert not [radio for radio in page.radio if radio.label == "Rating view"]
    assert any("(a) Display chosen" in block.value for block in page.caption)
    assert any("(e) Success rate regret" in block.value for block in page.caption)


# ── Progress estimation ──────────────────────────────────────────────────────


@needs_runs
def test_the_curves_cover_every_collection_the_table_counts(page):
    collection = next(box for box in page.selectbox if box.label == "Rollout collection")
    from behavior_graphs.viz.streamlit.components import paper_results

    counted = paper_results.section("progress")["datasets"]
    assert len(collection.options) == counted


@needs_runs
def test_the_curves_are_one_rollout_at_a_time(page):
    """Not an aggregate. A progress estimate is a per-rollout quantity and the
    estimators are deterministic, so a median across rollouts and a percentile
    band around it -- both of which this page carried -- are quantities the
    method never produces."""
    picker = next(box for box in page.selectbox if box.label == "Rollout")
    assert picker.options
    assert all("success" in option or "failure" in option for option in picker.options), (
        picker.options
    )
    for option in list(picker.options)[:3]:
        app = picker.set_value(option).run()
        assert not app.exception, (option, [str(e.value) for e in app.exception])


@needs_runs
def test_the_progress_benchmark_is_the_papers_own_table(page):
    """The paper's benchmark under the curves: every method on every metric,
    per source, with the source spanning its block as the paper's \\multirow
    does it. Hand-built HTML, so it is checked as markup."""
    html = next(
        block.value
        for block in page.markdown
        if 'class="progress"' in block.value and "<tbody>" in block.value
    )
    assert html.count("<tr") >= 8  # at least two sources, four methods each
    assert 'rowspan="4"' in html  # one Source cell spanning its four methods
    assert "<th>Source</th>" in html and "<th>Method</th>" in html


# ── References ───────────────────────────────────────────────────────────────


def test_every_citation_is_numbered_in_the_order_it_appears():
    """What a bracketed number means: a reader who meets [4] before [3] is
    reading a list that has drifted from the text. Every reference is cited,
    too, a list with an entry nothing points at being a list with a mistake
    in it."""
    from behavior_graphs.viz.streamlit.components import references

    references.check_order(APP.read_text())


@needs_runs
def test_the_references_are_listed_at_the_bottom(page):
    """Last section on the page, and every entry in it."""
    from behavior_graphs.viz.streamlit.components import references

    assert [block.value for block in page.header][-1] == "References"
    rendered = _text(page)
    for _, entry in references.numbered():
        assert entry in rendered, entry[:50]


# ── Failure ──────────────────────────────────────────────────────────────────


@needs_runs
def test_a_page_whose_results_are_missing_says_which_command_writes_them(monkeypatch):
    """The failure a deployment hits when the export has not been written: a
    sentence naming the fix, rather than a traceback on a URL a paper prints."""
    from behavior_graphs import results as results_module
    from behavior_graphs.viz.streamlit.components import paper_results

    monkeypatch.setattr(results_module, "DEFAULT_PATH", Path("/nonexistent/paper_results.json"))
    # The loader memoises across sessions on purpose, so the page above has
    # already cached a good document into it.
    paper_results._loaded.clear()
    try:
        app = AppTest.from_file(str(APP), default_timeout=900).run()
        assert not app.exception, [str(error.value) for error in app.exception]
        assert app.error, "a page with no results rendered without saying so"
        assert "export_paper_results" in app.error[0].value
    finally:
        paper_results._loaded.clear()


# ── Tone ─────────────────────────────────────────────────────────────────────


@needs_runs
def test_the_prose_uses_no_em_dashes(page):
    """A standing preference for this project's writing, and the easiest thing
    in the world to regress. The table's "no value" glyph is a separate
    question and is not prose, so this reads the text blocks only."""
    assert "—" not in _text(page)


@needs_runs
def test_no_heading_is_a_clause_with_a_comma_in_it(page):
    """Headings here are noun phrases. A comma in one is the tell that it has
    turned into a sentence with the verb taken out."""
    headings = [block.value for block in page.header]
    headings += [block.value for block in page.subheader]
    headings += [block.value for block in page.title]
    assert headings
    for heading in headings:
        assert "," not in heading, heading
        assert ":" not in heading, heading
