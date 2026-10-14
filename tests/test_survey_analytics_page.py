"""The analytics page draws, for a store holding any mix of study layouts.

The module tests next door check the reading of a response; these check the page
survives what comes back. Worth having as a page test rather than by inspection:
every panel is guarded on "did this layout record that answer", and the failure
mode is a `KeyError` on a column that a different layout would have supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# pytest puts this file's directory on sys.path (tests/ is not a package).
from test_survey_analytics import (  # noqa: E402
    CONTRASTIVE,
    ORDER_EFFECT,
    PRODUCTION,
    PRODUCTION_ALLOCATING,
    THREE_STAGE,
)

PAGE = Path(__file__).resolve().parents[1] / "apps" / "demo" / "_pages" / "survey_analytics.py"

#: The raw-responses export's button, offered whether or not a model was fit.
RAW = "Download raw-responses figure (PDF)"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A results dir the page will read, seeded per test.

    The page memoizes the load for 60s and the key is the (argument-less) call,
    so without clearing it every test in this file would analyse the first one's
    responses. The real page has a Refresh button for the same reason.
    """
    import streamlit as st

    def _write(responses: list[dict]) -> Path:
        st.cache_data.clear()
        from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore

        store = LocalResponseStore(tmp_path)
        for response in responses:
            store.save(response)
        monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
        monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
        return tmp_path

    # Out of the data root, at setup rather than inside ``_write``: the regret
    # panel reads the retrained policies' rollouts from ``$DATA_PATH``, so a
    # developer who exports it would run these tests against whatever is on
    # their disk — an extra download button, and a panel whose numbers nobody
    # chose. Cleared here so a test that *wants* that panel can mount its own
    # arms from the test body (see ``arm_rollouts``) and have them stick.
    monkeypatch.delenv("DATA_PATH", raising=False)
    return _write


def _run(pick: str | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=60).run()
    if pick is not None:
        at = at.selectbox[0].set_value(pick).run()
    assert not at.exception, at.exception
    return at


def _sections(at: AppTest) -> set[str]:
    return {element.value for element in at.subheader}


@pytest.fixture()
def quick_fits(monkeypatch):
    """Replace the Monte-Carlo fits with the summaries they would have returned.

    A page test is about the page: that the section stays behind its toggle,
    that the export appears once the figure drew, that a fit which cannot
    happen costs its own section and not the study beneath it. Sampling four
    chains per outcome to learn that costs minutes a run and tells us nothing
    ``test_survey_mixed_effects`` does not. The stubs keep the two contracts
    the page leans on: no randomized ratings means no effects, and no scored
    choice means no regret effect.
    """
    from behavior_graphs.viz.streamlit.user_study import mixed_effects, regret

    def _rating_effects(responses, **kwargs):
        if not mixed_effects.assigned_rating_rows(responses):
            return []
        return [
            {
                "outcome": outcome,
                "mean": -6.0,
                "lower": -13.0,
                "upper": 1.0,
                "marginal_means": {
                    modality: {"mean": 50.0, "lower": 40.0, "upper": 60.0}
                    for modality in ("graph", "videos")
                },
            }
            for outcome in ("temporal_demand", "effort", "confidence")
        ]

    def _regret_effect(responses, rates, **kwargs):
        # NEGATIVE: less regret with the graph, which is the graph leading to
        # better decisions. The model reports regret, so a positive stub here
        # would assert the opposite of what the panel reads as -- and this
        # number has been flipped once already by a reframing.
        rows = regret.regret_rows(responses, rates)
        return {"mean": -8.0, "lower": -16.0, "upper": 1.0, "n": len(rows)} if rows else None

    monkeypatch.setattr(mixed_effects, "continuous_effects", _rating_effects)
    monkeypatch.setattr(mixed_effects, "decision_regret_effect", _regret_effect)


def _open_paper_figure(at: AppTest) -> AppTest:
    """Open the collapsed paper-figure section and return the rerun page.

    It is behind ``lazy_section`` because fitting its models is four chains per
    outcome, so nothing in it — the mode radio, the export — exists until a
    reader asks for it.
    """
    for button in at.button:
        if "adjusted models" in button.label:
            reopened = button.click().run()
            assert not reopened.exception, reopened.exception
            return reopened
    raise AssertionError(f"no paper-figure toggle among {[b.label for b in at.button]}")


def test_an_empty_store_says_so_rather_than_erroring(store):
    store([])

    at = _run()

    assert "No responses found yet." in " ".join(element.value for element in at.info)


def test_every_layout_in_one_store_is_selectable(store):
    store([PRODUCTION, PRODUCTION_ALLOCATING, ORDER_EFFECT, CONTRASTIVE])

    at = _run()

    # Commonest first, "All versions" last — and the counts are in the labels.
    assert at.selectbox[0].options[0].startswith("production_group_ab — 2 responses")
    assert at.selectbox[0].options[-1] == "All versions — 4 responses"


def test_the_production_layout_draws_its_own_panels(store):
    store([PRODUCTION, PRODUCTION_ALLOCATING])

    sections = _sections(_run("production_group_ab"))

    assert "Data allocation choices" in sections
    assert "Rating scales" in sections
    # One condition per participant, so there is no before/after to compare.
    assert "Answers that changed between the two modalities" not in sections


def test_the_contrastive_layout_gets_the_comparison_it_exists_for(store):
    store([PRODUCTION, CONTRASTIVE])

    at = _run("contrastive_set_table_v1")
    sections = _sections(at)

    assert "Answers that changed between the two modalities" in sections
    assert "Data-collection choices" in sections
    # Its task load is nested per phase; an empty panel here was the bug.
    assert "No task-load ratings in these responses." not in [e.value for e in at.info]
    assert [m.value for m in at.metric][:2] == ["1", "1"]
    # Nothing from the other layout leaks in.
    assert "Data allocation choices" not in sections


def test_the_order_effect_layout_draws_its_per_phase_answers(store):
    store([ORDER_EFFECT])

    at = _run()
    sections = _sections(at)

    assert "Ranked answers" in sections
    assert "Rating scales" in sections
    assert "No task-load ratings in these responses." not in [e.value for e in at.info]
    # A single version: no "All versions" entry to choose.
    assert at.selectbox[0].options == ["order_effect_v1 — 1 response"]


def test_mixing_every_layout_draws_the_union_of_their_panels(store):
    store([PRODUCTION, PRODUCTION_ALLOCATING, ORDER_EFFECT, CONTRASTIVE, THREE_STAGE])

    sections = _sections(_run("All versions"))

    assert {
        "NASA Task Load Index",
        "Answers that changed between the two modalities",
        "Ranked answers",
        "Data-collection choices",
        "Rating scales",
        "Data allocation choices",
        "Stages",
    } <= sections


def test_the_three_stage_layout_draws_its_stages_and_its_free_choice(store):
    """One task per stage and one display per task, so the panel that matters is
    which display a participant picked when nobody assigned one."""
    store([THREE_STAGE])

    at = _run()
    sections = _sections(at)

    assert {"Stages", "NASA Task Load Index", "Data-collection choices"} <= sections
    spec = next(
        el.proto.spec
        for el in at.get("plotly_chart")
        if "Which display participants chose" in el.proto.spec
    )
    assert "videos" in spec
    # Its task load is nested per task, not per modality, and is still read.
    assert "No task-load ratings in these responses." not in [e.value for e in at.info]
    # Three stages, so there is no two-modality before/after to compare.
    assert "Answers that changed between the two modalities" not in sections


def test_the_task_load_caption_says_what_a_box_actually_is(store):
    """Per task with one display each is not "the same task off two displays" —
    a caption asserting the comparison the layout did not make is worse than
    none, and a mixed store has both kinds of response to describe."""
    store([THREE_STAGE])
    captions = " ".join(element.value for element in _run().caption)
    assert "one display per task" in captions
    assert "rating the same task off different evidence" not in captions

    store([CONTRASTIVE])
    captions = " ".join(element.value for element in _run().caption)
    assert "Rated once per modality" in captions
    assert "one display per task" not in captions

    # Both, where both are in view: one three-stage response among the
    # contrastive ones must not relabel theirs.
    store([CONTRASTIVE, THREE_STAGE])
    captions = " ".join(element.value for element in _run("All versions").caption)
    assert "Rated once per modality" in captions
    assert "one display per task" in captions


def _figures(at: AppTest, title_fragment: str) -> list[dict]:
    """The charts whose title contains *title_fragment*, parsed.

    Parsed rather than string-matched: plotly serialises a numeric array as
    base64 in ``bdata``, so a substring check for a value silently never
    matches, and the title's em dash comes back escaped.
    """

    figures = []
    for element in at.get("plotly_chart"):
        figure = json.loads(element.proto.spec)
        if title_fragment in figure.get("layout", {}).get("title", {}).get("text", ""):
            figures.append(figure)
    return figures


def _numbers(trace: dict, axis: str) -> list[float]:
    """One axis of a trace, decoding plotly's binary array encoding."""
    import base64
    import struct

    values = trace[axis]
    if isinstance(values, dict):
        raw = base64.b64decode(values["bdata"])
        return list(struct.unpack(f"<{len(raw) // 8}d", raw))
    return list(values)


def test_a_task_s_ranking_is_weighted_on_its_own_scale(store):
    """The banks differ in depth — set_table ranks two failure modes, cooking
    three — so one chart per task, each weighted by its own depth. Pooled, a
    shallower ranking's last pick scored as if it were a deeper one's middle.
    """
    store([THREE_STAGE])

    figures = _figures(_run(), "how highly they were ranked")
    by_task = {figure["layout"]["title"]["text"].split("— ")[-1]: figure for figure in figures}

    assert set(by_task) == {"cooking", "set_table", "prepare_tray"}, "one chart per task"
    # Cooking's participant ranked two, so on its own two-deep scale rank 1 is
    # worth 1 and rank 2 is worth 0.5. Pooled with a three-deep task, that last
    # pick would have scored 2/3 — a middling pick rather than a final one.
    (cooking,) = by_task["cooking"]["data"]
    assert _numbers(cooking, "x") == [1.0, 0.5]
    # The one-pick tasks are a full-weight bar each rather than a fraction of
    # cooking's depth.
    for task in ("set_table", "prepare_tray"):
        (trace,) = by_task[task]["data"]
        assert _numbers(trace, "x") == [1.0], task


def test_two_layouts_that_asked_one_task_the_same_question_share_a_bar(store):
    """The choices panel is not split by task: two bars at one categorical
    position land in a single plotly offset group, so the smaller hides behind
    the larger instead of appearing beside it."""
    store([CONTRASTIVE, THREE_STAGE])

    figures = _figures(_run("All versions"), "Choices by modality")

    (figure,) = figures
    for trace in figure["data"]:
        assert len(set(trace["y"])) == len(trace["y"]), trace["y"]


def test_an_unfinished_session_is_kept_out_of_the_numbers_until_asked(store):
    """The three-stage layout writes as answers come in, so the store holds
    sessions nobody finished. Averaging them in with the completed ones
    reweights every result, and doing it silently is worse."""
    partial = dict(
        THREE_STAGE,
        participant_id="p6",
        status="in_progress",
        step=3,
        understanding_responses={"cooking": THREE_STAGE["understanding_responses"]["cooking"]},
        nasa_tlx={"cooking": {"effort": 90}},
    )
    store([THREE_STAGE, partial])

    at = _run()

    # Two sessions started, one finished — and only the finished one is counted.
    assert [m.value for m in at.metric][0] == "1"
    assert "1/2" in [m.value for m in at.metric]
    assert [t.label for t in at.toggle] == ["Include 1 unfinished session"]
    said = " ".join(element.value for element in at.caption)
    assert "Where the unfinished sessions stopped" in said
    # The task-load panel is computed over the finished session only, so the
    # unfinished one's effort of 90 is not in it.
    spec = next(el.proto.spec for el in at.get("plotly_chart") if "Task load" in el.proto.spec)
    assert "90" not in spec

    with_partials = at.toggle[0].set_value(True).run()
    assert [m.value for m in with_partials.metric][0] == "2"


def test_a_lab_preview_is_kept_out_of_the_numbers_until_asked(store):
    """A researcher opening the layout on the User Study page writes a record
    too. Left in, it is a participant who quit on the introduction: it lands in
    the funnel and in the completed-of-started metric this page reports while a
    study is live."""
    preview = dict(
        THREE_STAGE,
        participant_id="",
        origin="lab",
        status="in_progress",
        step=0,
    )
    store([THREE_STAGE, preview])

    at = _run()

    assert [t.label for t in at.toggle] == ["Include 1 lab preview"]
    # One participant, one session started, one finished — the preview counts
    # towards none of it.
    assert [m.value for m in at.metric][0] == "1"
    assert "1/1" in [m.value for m in at.metric]

    with_previews = at.toggle[0].set_value(True).run()
    assert not with_previews.exception
    assert "1/2" in [m.value for m in with_previews.metric]


def test_the_step_chart_is_labelled_in_words(store):
    """It used to plot 0 1 2 3 4 5 and leave the reader to remember the order."""
    store([CONTRASTIVE])

    at = _run()
    # Found by its title and read out of the serialised figure: AppTest does not
    # carry a chart's key, and `.value` goes looking for selection state that a
    # test run never wrote.
    spec = next(
        el.proto.spec for el in at.get("plotly_chart") if "Minutes spent per step" in el.proto.spec
    )

    assert "Introduction" in spec
    assert "Rollout videos + questions" in spec
    assert '"x":["0"' not in spec, "steps plotted by index"


def test_the_closing_questions_have_a_view(store):
    store([CONTRASTIVE])

    at = _run()

    assert "Closing questions" in _sections(at)
    written = " ".join(element.value for element in at.markdown)
    assert "the graph was easier to skim" in written


def test_the_ranked_panel_keeps_the_ab_split_where_there_are_no_phases(store):
    """The production study ranks once per participant, and its comparison is
    A against B — colouring that by phase summed the two into one bar."""
    a = dict(PRODUCTION, participant_id="pa", group="A", ranked_choices=["s1", "s2"])
    b = dict(PRODUCTION, participant_id="pb", group="B", ranked_choices=["s2", "s1"])
    store([a, b])

    at = _run()
    spec = next(
        el.proto.spec
        for el in at.get("plotly_chart")
        if "Times picked" in el.proto.spec or "Rank-weighted" in el.proto.spec
    )

    assert '"name":"A"' in spec and '"name":"B"' in spec, spec[:400]


def test_a_flat_understanding_block_does_not_take_the_page_down(store):
    """What the contrastive layout's first build wrote."""
    store(
        [
            dict(
                CONTRASTIVE,
                understanding_responses={
                    "failure_mode_ranking": ["fails to grasp the milk"],
                    "data_collection_choice": "Milk left, cola middle, cupcake right",
                },
            )
        ]
    )

    at = _run()

    assert [m.value for m in at.metric][:2] == ["1", "1"]
    assert "Data-collection choices" in _sections(at)


def test_the_experience_chart_is_ordered_by_the_scale(store):
    store([CONTRASTIVE])

    at = _run()
    spec = next(
        el.proto.spec for el in at.get("plotly_chart") if "Reported experience" in el.proto.spec
    )

    assert "A good deal" in spec
    # Weakest first, not alphabetical.
    assert '["None at all","A little","Some","A good deal","A great deal"]' in spec.replace(
        ", ", ","
    )


def test_a_response_with_nothing_but_an_id_does_not_take_the_page_down(store):
    """Half-written responses exist: the store is append-only and unvalidated."""
    store([{"participant_id": "p9", "group": "A", "study_variant": "contrastive_cooking_v1"}])

    at = _run()

    assert [m.value for m in at.metric][:2] == ["1", "1"]
    assert "No task-load ratings in these responses." in [e.value for e in at.info]


def test_the_paper_figure_is_offered_where_the_answers_it_draws_exist(store, quick_fits):
    """One button, and the export is built from the responses in view — so the
    filters above it apply to the file as much as to the charts."""
    store([THREE_STAGE])

    at = _run()

    assert "Paper figure" in _sections(at)
    # Collapsed, no model has been fit -- but the raw-responses figure needs
    # none, so it and its download are there regardless.
    assert [b.proto.label for b in at.get("download_button")] == [RAW]

    opened = _open_paper_figure(at)

    assert [b.proto.label for b in opened.get("download_button")] == [
        RAW,
        "Download paper figure (PDF)",
    ]


def test_a_store_too_small_to_fit_the_model_warns_instead_of_breaking_the_page(store):
    """A single response cannot support a graph-versus-videos effect, and the
    sampler says so from inside its own internals. Every panel above the figure
    is still the page's answer to "what did these responses say", so the fit
    that cannot happen costs its own section and nothing else.

    Needs the analysis extra: without it the page warns that the extra is
    missing instead, which is a different guard with its own test, and this one
    would assert a message no sampler was ever asked to produce."""
    pytest.importorskip("bambi")
    store([THREE_STAGE])

    opened = _open_paper_figure(_run())

    assert "Data-collection outcome" in _sections(opened)
    assert any("Could not fit the adjusted model" in w.value for w in opened.warning)
    # The adjusted figure is withheld -- its numbers do not exist -- while the
    # raw one remains, because a store too small to fit a model is not too
    # small to plot what it actually contains.
    assert [b.proto.label for b in opened.get("download_button")] == [RAW]


def test_a_layout_that_answered_none_of_the_figure_gets_no_button(store, quick_fits):
    """The production layout has no free choice and rates a session once, so
    there is no graph-versus-videos figure to draw for it — and four blank axes
    would read as a broken plot rather than as a layout that did not ask."""
    store([PRODUCTION, PRODUCTION_ALLOCATING])

    opened = _open_paper_figure(_run())

    # Neither figure: the production layout has no assigned-display stage to
    # draw raw and no randomized contrast to fit.
    assert not opened.get("download_button")
    assert any("randomized three-stage responses" in element.value for element in opened.info)


@pytest.fixture()
def arm_rollouts(tmp_path, monkeypatch):
    """Mount a rollout collection for every retrained policy under ``$DATA_PATH``.

    The regret panel reads its success rates off the canonical layout rather
    than from a table in the code (outcomes are hand-annotated and get
    corrected), so a test of the panel has to put rollouts on disk. Only
    ``episodes/metadata.yaml`` is written — it is all the outcome tally reads.
    """
    import yaml

    from behavior_graphs.viz.streamlit.user_study import regret

    def _mount(rates: dict[str, tuple[int, int]] | None = None) -> Path:
        root = tmp_path / "data"
        given = rates or {}
        for dataset in regret.REFERENCE_DATASETS:
            successes, trials = given.get(dataset, (30, 60))
            episodes = root / "rollouts" / dataset / "episodes"
            episodes.mkdir(parents=True, exist_ok=True)
            (episodes / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "source": {"kind": "test"},
                        "episodes": [
                            {"index": i, "length": 4, "success": i < successes}
                            for i in range(trials)
                        ],
                    }
                )
            )
        monkeypatch.setenv("DATA_PATH", str(root))
        return root

    return _mount


def test_the_regret_panel_scores_the_choices_against_the_retrained_policies(store, arm_rollouts):
    """The point of the panel: the choice is not just counted, it is priced.

    The participant picked the milk-left arrangement for set_table, which is
    mounted here as the best of the three, and the bowl-on-top arrangement for
    prepare_tray, which is mounted as 20 points behind the other one.
    """
    arm_rollouts(
        {
            "set_table_plusleft_pretrained_dit": (48, 60),
            "set_table_plusmiddle_pretrained_dit": (30, 60),
            "set_table_plusright_pretrained_dit": (24, 60),
            "prepare_tray_plustop": (42, 60),
            "prepare_tray_plusbottom": (54, 60),
        }
    )
    store([THREE_STAGE])

    at = _run()

    assert "Data-collection outcome" in _sections(at)
    (figure,) = _figures(at, "Regret of the data-collection choice")
    regrets = sorted(value for trace in figure["data"] for value in _numbers(trace, "y"))
    # Regret: the distance to the best option on offer, so two of these three
    # answers picked it and one left 20 points on the table. (The same answers
    # are 0/20/30 points of GAIN, which measures against collecting nothing
    # instead -- the panel reported that for a while and no longer does.)
    assert regrets == pytest.approx([0.0, 0.0, 20.0])


def test_the_regret_figure_is_offered_beside_the_study_figure(store, arm_rollouts, quick_fits):
    """Two paper figures now, and the regret one can be exported three ways."""
    arm_rollouts()
    store([THREE_STAGE])

    at = _run()
    assert [b.proto.label for b in at.get("download_button")] == [
        "Download regret figure (PDF)",
        RAW,
    ]

    opened = _open_paper_figure(at)

    assert [b.proto.label for b in opened.get("download_button")] == [
        "Download regret figure (PDF)",
        RAW,
        "Download paper figure (PDF)",
    ]
    modes = opened.radio[0]
    assert len(modes.options) == 3
    switched = modes.set_value("per_task").run()
    assert not switched.exception, switched.exception


def test_the_panel_names_the_policies_it_could_not_find(store, quick_fits):
    """No ``$DATA_PATH``, so no arm has a rate. "Nothing to show" over a store
    full of answered choices reads as the scoring being broken — it has to say
    which rollouts are missing instead."""
    store([THREE_STAGE])

    at = _run()

    said = " ".join(element.value for element in at.info)
    assert "No choice can be scored yet" in said
    assert "set_table_plusleft_pretrained_dit" in said
    # And no figure to download for a panel that has no numbers — while the
    # paper figure, which does not need them, is still offered.
    opened = _open_paper_figure(at)
    assert [b.proto.label for b in opened.get("download_button")] == [
        RAW,
        "Download paper figure (PDF)",
    ]


def test_a_task_missing_one_arm_is_left_out_rather_than_scored_against_the_rest(
    store, arm_rollouts
):
    """With one option unmeasured the best option is unknown, so every regret
    for that task would be understated by however good the missing arm is."""
    root = arm_rollouts(
        {
            "set_table_plusleft_pretrained_dit": (48, 60),
            "prepare_tray_plusbottom": (54, 60),
        }
    )
    import shutil

    shutil.rmtree(root / "rollouts" / "prepare_tray_plusbottom")
    store([THREE_STAGE])

    at = _run()

    said = " ".join(element.value for element in at.caption)
    assert "prepare_tray (prepare_tray_plusbottom)" in said
    (figure,) = _figures(at, "Regret of the data-collection choice")
    tasks = {value for trace in figure["data"] for value in trace["x"]}
    assert "Prepare the tray" not in tasks
    assert "Set the table" in tasks


def test_the_paper_figure_carries_a_decision_panel(store, arm_rollouts, quick_fits):
    """The figure's third panel: the ratings are what the displays cost to use,
    and this is what the decisions they led to were worth — the same
    graph-minus-videos contrast, read on an outcome nobody self-reported."""
    arm_rollouts(
        {
            "set_table_plusleft_pretrained_dit": (48, 60),
            "set_table_plusmiddle_pretrained_dit": (30, 60),
            "set_table_plusright_pretrained_dit": (24, 60),
            "prepare_tray_plustop": (42, 60),
            "prepare_tray_plusbottom": (54, 60),
        }
    )
    store([THREE_STAGE])

    opened = _open_paper_figure(_run())

    # The export is offered, which means the figure drew.
    assert any("paper figure" in button.label.lower() for button in opened.get("download_button"))
    assert not any("without its regret panel" in w.value for w in opened.warning)


def test_the_decision_panel_is_the_regret_answers_in_success_rate_points():
    """The same reading as the regret-only figure's pooled panel — the answers
    themselves, pooled per display — so the two figures do not report one
    quantity two ways. Points rather than a normalized score: it is the unit a
    reader can hold ("eight points of success rate left on the table") and what
    pooling them costs belongs in the caption, not in a second scale."""
    import matplotlib.pyplot as plt

    from behavior_graphs.viz.streamlit.user_study import figure_export

    rows = [
        {"phase": "graph", "task": "t", "regret": 0.0},
        {"phase": "graph", "task": "t", "regret": 10.0},
        {"phase": "videos", "task": "t", "regret": 6.0},
    ]

    fig = figure_export.study_figure([THREE_STAGE], rows)
    try:
        panel = fig.axes[-1]
        # Just the quantity: every regret panel is a distribution now, so there
        # is no mean to tell apart from the answers it averages.
        assert panel.get_ylabel().startswith("Regret")
        assert "success-rate points" in panel.get_ylabel()
        # Two lines, because a longer label is clipped at this panel height --
        # see `_REGRET_AXIS_LABEL`. The gain version of this label was
        # drawn clipped in every export until it was measured.
        assert panel.get_ylabel().count("\n") == 1
        # A dot per answer, in the same box-and-strip the rating panels use --
        # not a bar at the mean with a standard error around it.
        assert [
            sorted(round(float(y), 3) for _x, y in collection.get_offsets())
            for collection in panel.collections
        ] == [[0.0, 10.0], [6.0]]
        # Zero is the floor: regret is a distance to the best option available,
        # so an axis below it would imply a better answer than the best one.
        assert panel.get_ylim()[0] == 0
        # No printed numbers in the panel: it is a fifth of a row, and the box
        # and the axis already carry the reading.
        assert [text.get_text() for text in panel.texts] == []
        assert [label.get_text() for label in panel.get_xticklabels()] == [
            "Graph\nn=2",
            "Videos\nn=1",
        ]
    finally:
        plt.close(fig)


def _per_task_figure(at: AppTest) -> dict:
    """The per-task task-load chart, or a failure naming what was drawn."""
    figures = _figures(at, "Task load by dimension, task and display")
    assert len(figures) == 1, f"expected one per-task chart, got {len(figures)}"
    return figures[0]


def _facet_rows(figure: dict) -> list[str]:
    """The facet labels, top row first — which is what "longest first" means."""
    notes = [
        note
        for note in figure["layout"].get("annotations", [])
        if note.get("textangle") == 90 and note.get("text")
    ]
    return [note["text"] for note in sorted(notes, key=lambda note: -note["y"])]


def test_the_rating_axes_say_which_way_each_dimension_improves(store):
    """Under the ticks, and per dimension: the load items improve downward and
    confidence improves upward, so one note in the axis title would be
    backwards for one of the boxes drawn against it.

    Under the *x* ticks rather than in the y-axis title for a second reason —
    Plotly rotates that title, and an arrow rotates with it, so a down arrow
    set there comes out pointing sideways. The paper figures put theirs under
    the panel name for the same reason.
    """
    rated = dict(
        THREE_STAGE,
        nasa_tlx={
            task: {"temporal_demand": 40, "effort": 40, "confidence": 70}
            for task in ("cooking", "set_table", "prepare_tray")
        },
    )
    store([rated])
    at = _run()

    expected = [
        "Temporal demand \u2193",
        "Effort \u2193",
        "Confidence \u2191",
    ]
    (pooled,) = [
        figure
        for figure in _figures(at, "Task load by dimension")
        if figure["layout"]["title"]["text"] == "Task load by dimension"
    ]
    assert pooled["layout"]["xaxis"]["ticktext"] == expected
    # And on the faceted chart, whose dimension axis is drawn once under the
    # bottom row of facets.
    assert _per_task_figure(at)["layout"]["xaxis"]["ticktext"] == expected


def test_the_per_task_panel_splits_each_task_by_display(store):
    """The panel above pools the displays across tasks; this one must not — the
    reading it exists for is whether the display mattered more on the task that
    cost more, and a pooled box cannot say."""
    store([THREE_STAGE])
    at = _run()
    figure = _per_task_figure(at)

    assert "Task load per task" in _sections(at)
    # Boxes are displays, inside a row per task — not one box per task.
    assert {trace["name"] for trace in figure["data"]} <= {"videos", "graph"}
    assert set(_facet_rows(figure)) == {"Cook the meal", "Set the table", "Prepare the tray"}
    # The page's own two hues, the ones that mean the same two displays in every
    # other chart on it.
    assert {trace["name"]: trace["marker"]["color"] for trace in figure["data"]} == {
        "videos": "#1f77b4",
        "graph": "#ff7f0e",
    }


def test_the_per_task_panel_is_left_out_where_there_are_no_tasks(store):
    """A layout that rates a session once has one task and nothing to compare
    it with."""
    store([CONTRASTIVE])
    at = _run()

    assert "Task load per task" not in _sections(at)
    assert _figures(at, "Task load by dimension, task and display") == []


def test_the_task_rows_are_ordered_by_the_time_they_took(store):
    """Longest at the top, by the clock rather than by the temporal-demand
    item: "longer" is what the ratings are being read against, so it has to be
    the measured thing. A task no timing reached sorts last."""
    store([THREE_STAGE])

    # cooking took 300 s, prepare_tray 240 s, and set_table's stage has no
    # recorded duration at all.
    assert _facet_rows(_per_task_figure(_run())) == [
        "Cook the meal",
        "Prepare the tray",
        "Set the table",
    ]

    slower = dict(
        THREE_STAGE,
        participant_id="p6",
        # set_table's stage is now the long one, so the order flips.
        step_durations_seconds={"1": 100.0, "3": 900.0, "5": 240.0},
    )
    store([slower])

    assert _facet_rows(_per_task_figure(_run()))[0] == "Set the table"


def test_the_graph_is_the_left_box_in_both_task_load_charts(store):
    """`regret.PHASE_ORDER`'s order, which every other side-by-side drawing of
    the two displays uses: the graph is what the study is about, so it is the
    box a reader meets first. px would otherwise order them by whichever it met
    first in the frame, and the same two boxes would swap sides when a
    different response happened to load first."""
    # Whose first-seen display is videos, so an unordered chart would put
    # videos on the left.
    videos_first = dict(
        THREE_STAGE,
        participant_id="p7",
        evidence_order=["videos", "videos", "graph"],
        stage_modalities={"cooking": "videos", "set_table": "videos", "prepare_tray": "graph"},
    )
    store([videos_first])
    at = _run()

    pooled = _figures(at, "Task load by dimension")[0]
    assert [trace["name"] for trace in pooled["data"]] == ["graph", "videos"]

    # And inside every task's own row.
    per_task = _per_task_figure(at)
    by_row: dict[str, list[str]] = {}
    for trace in per_task["data"]:
        by_row.setdefault(trace.get("yaxis", "y"), []).append(trace["name"])
    assert by_row
    for row, names in by_row.items():
        # A task this fixture only shows in one display has only that box, and
        # it is not on the wrong side for being alone.
        assert names == [n for n in ("graph", "videos") if n in names], f"{row}: {names}"


def test_the_regret_figure_is_restricted_to_the_randomized_stages(store, arm_rollouts, quick_fits):
    """The export the paper takes covers the same evidence as every other figure
    in it: the stages whose display was *assigned*. The chart above it is the
    page exploring every scored answer, which is a different job — so the
    section says how many answers it left out rather than letting a reader
    wonder why a box here is not the box up there.

    ``THREE_STAGE`` read prepare tray in the display it chose, so exactly one of
    its three answers is dropped.
    """
    arm_rollouts()
    store([THREE_STAGE])

    at = _run()

    restriction = next(box for box in at.checkbox if "Randomized stages only" in box.label)
    assert restriction.value is True
    said = " ".join(element.value for element in at.caption)
    assert "Drawn from **2** of 3 scored answers" in said
    assert "The chart above includes them" in said

    # And it is the figure that changes, not just the words: a download URL is
    # a hash of the file's content, so a URL that moves is the export moving.
    before = next(b for b in at.get("download_button") if "regret figure" in b.label).proto.url
    everything = restriction.set_value(False).run()
    assert not everything.exception, everything.exception
    after = next(
        b for b in everything.get("download_button") if "regret figure" in b.label
    ).proto.url

    assert after != before
    # Nothing dropped, so nothing to account for.
    assert not any("are left out" in element.value for element in everything.caption)
