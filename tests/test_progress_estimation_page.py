"""The Progress Estimation page's rubric table and its averages.

Worth a page test rather than inspection: the table's whole claim is that every
row of a rubric block was measured on the SAME datasets, and the two ways that
silently breaks — a dataset scored by three of the four methods sneaking in, and
a rubric quietly losing datasets — both look like a plausible table.

The rubric table is rendered as HTML because `st.dataframe` cannot span a cell
over several rows, so the assertions here read the markup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "apps" / "demo" / "_pages" / "progress_estimation.py"

#: The four methods the rubric table compares, and the metrics each needs.
FOUR = ("graph_stage", "mlp", "robometer_4b", "topreward_ll")
METRICS = ("voc", "kendall", "pearson", "mae", "auroc")


def _metrics(methods, value: float = 0.5) -> dict:
    return {m: {k: [value, 0.0] for k in METRICS} for m in methods}


@pytest.fixture()
def results(tmp_path, monkeypatch):
    """A results root seeded per test, and the page pointed at it."""

    def _write(datasets: dict[str, dict]) -> Path:
        import streamlit as st

        st.cache_data.clear()
        for dataset, blob in datasets.items():
            (tmp_path / f"{dataset}_metrics.json").write_text(json.dumps(blob))
        monkeypatch.setenv("PROGRESS_RESULTS_ROOT", str(tmp_path))
        monkeypatch.setenv("PROGRESS_ROLLOUTS_ROOT", str(tmp_path / "rollouts"))
        return tmp_path

    return _write


def _run() -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _table_html(at: AppTest) -> str | None:
    """The rubric table's markup, or None if it was not drawn."""
    for element in at.markdown:
        value = element.value or ""
        if "<table" in value and "rowspan" in value:
            return value
    return None


def _ruled_rows(html: str) -> list[int]:
    """Row indices carrying a midrule, read off the emitted <style> block."""
    style = html[html.index("<style") : html.index("</style>")]
    rows: set[int] = set()
    for rule in style.split("}"):
        if "border-top" in rule:
            rows.update(int(n) for n in re.findall(r"_row(\d+)_col", rule))
    return sorted(rows)


def _block_starts(html: str) -> list[int]:
    """Row indices where a rubric block begins — the spanning cells."""
    body = html[html.index("<tbody>") :]
    return [
        i for i, row in enumerate(re.findall(r"<tr>(.*?)</tr>", body, re.S)) if 'rowspan="4"' in row
    ]


def _averages(at: AppTest):
    """The averages table.

    Matched on the SET metric labels, not on the index name: the page's main
    metrics table is also indexed by method, and it keys its columns by the
    lower-case metric names.
    """
    for element in at.dataframe:
        frame = element.value
        if getattr(frame.index, "name", None) == "method" and "VOC" in frame.columns:
            return frame
    return None


def test_each_rubric_is_one_cell_spanning_its_method_rows(results):
    """The multirow is the point: `st.dataframe` repeats the rubric on every row,
    which reads as four separate one-method groups."""
    results(
        {
            "libero_goal": _metrics(FOUR),
            "libero_object": _metrics(FOUR),
            "set_table_baseline": _metrics(FOUR),
            "cooking_baseline": _metrics(FOUR),
            "prepare_tray_baseline": _metrics(FOUR),
            "mimicgen_stack_d0": _metrics(FOUR),
        }
    )

    html = _table_html(_run())

    assert html is not None, "the rubric table was not drawn"
    # One spanning cell per rubric, four method rows tall -- plus one for the
    # average block at the foot.
    assert html.count('rowspan="4"') == 4
    # Counted datasets are in the row label, and droid pools the three
    # real-robot tasks — they are named by task on disk, not by platform.
    assert "libero (2)" in html
    assert "Real-world (3)" in html
    assert "mimicgen (1)" in html
    # A rule between blocks, but not above the first one. Counted by ROW:
    # pandas merges cells with identical CSS into one comma-separated selector,
    # so counting occurrences of the property measures its own grouping.
    assert _ruled_rows(html) == _block_starts(html)[1:]


def test_our_method_is_marked_and_the_metrics_are_set_as_acronyms(results):
    results({"libero_goal": _metrics(FOUR)})

    html = _table_html(_run())

    assert "(ours)" in html
    for label in ("VOC", "Kendall", "MAE", "AUROC"):
        assert f">{label}<" in html, f"{label} is not a column header"
    assert ">voc<" not in html and ">mae<" not in html


def test_the_dataset_count_is_only_in_the_label(results):
    """Not a column as well: it is the same number on all four rows of a block,
    so a column of it is four times the ink for one fact."""
    results({"libero_goal": _metrics(FOUR), "libero_object": _metrics(FOUR)})

    html = _table_html(_run())

    assert "libero (2)" in html
    # The header row is metrics only, and Pearson is not among the defaults.
    header = html[html.index("<thead>") : html.index("</thead>")]
    assert re.findall(r">(VOC|Kendall|Pearson|MAE|AUROC|n)<", header) == [
        "VOC",
        "Kendall",
        "MAE",
        "AUROC",
    ]


def test_a_dataset_missing_one_method_is_left_out_of_every_row(results):
    """Not just out of that method's row: the point of the table is that a
    rubric's rows are comparable, so a dataset three of the four were scored on
    would otherwise inflate those three against the fourth."""
    results(
        {
            "mimicgen_stack_d0": _metrics(FOUR, 0.9),
            # No topreward_ll. Its graph/mlp/robometer numbers must not count.
            "mimicgen_coffee_d0": _metrics(FOUR[:3], 0.1),
        }
    )

    at = _run()
    html = _table_html(at)

    assert "mimicgen (1)" in html
    assert "0.900" in html
    # 0.1 would drag any mean that counted the incomplete dataset.
    assert "0.100" not in html and "0.500" not in html
    # And it is reported, not silently dropped.
    assert any("mimicgen_coffee_d0" in (m.value or "") for m in at.markdown)


def test_the_safe_sets_are_out_of_the_table_entirely(results):
    """They carry no video, so no VLM baseline can be run on them; a row for
    them could only ever hold two of the four methods."""
    results(
        {
            "libero_goal": _metrics(FOUR),
            "safe_franka_close_the_door": _metrics(("graph_stage", "mlp")),
            "safe_widowx_lift_blue_cup": _metrics(("graph_stage", "mlp")),
        }
    )

    html = _table_html(_run())

    assert "libero (1)" in html
    assert "safe" not in html


def test_no_complete_dataset_says_so_instead_of_drawing_an_empty_table(results):
    results({"mimicgen_stack_d0": _metrics(("graph_stage", "mlp"))})

    at = _run()

    assert _table_html(at) is None
    assert any("scored by all four" in (i.value or "") for i in at.info)


def test_the_average_is_the_last_block_of_the_table(results):
    """A reader looks for it after the per-family rows, and the midrule above
    it already says it is not another family."""
    results(
        {
            "libero_goal": _metrics(FOUR, 0.8),
            "mimicgen_stack_d0": _metrics(FOUR, 0.4),
        }
    )

    at = _run()
    html = _table_html(at)

    blocks = re.findall(r'rowspan="4"[^>]*>([^<]+)<', html)
    assert blocks[-1] == "Average (2)", blocks
    assert len(blocks) == 3  # libero, mimicgen, average
    # Dataset-weighted by default, and each rubric holds one dataset here.
    assert "0.600" in html
    # And it is not also drawn as a table of its own.
    assert not any(
        getattr(el.value.index, "name", None) == "method" and "VOC" in el.value.columns
        for el in at.dataframe
    )


def test_the_average_can_weight_rubrics_instead_of_datasets(results):
    """The rubrics are very uneven, so which one is meant changes the number."""
    results(
        {
            # Two libero datasets at 0.9, one mimicgen at 0.3: dataset-weighted
            # is 0.7, rubric-weighted is 0.6.
            "libero_goal": _metrics(FOUR, 0.9),
            "libero_object": _metrics(FOUR, 0.9),
            "mimicgen_stack_d0": _metrics(FOUR, 0.3),
        }
    )

    at = _run()
    assert "0.700" in _table_html(at)

    at = at.radio(key="pe_rubric_weight").set_value("rubric").run()
    html = _table_html(at)

    assert "0.600" in html
    blocks = re.findall(r'rowspan="4"[^>]*>([^<]+)<', html)
    assert blocks[-1] == "Average (2 rubrics)", blocks


def test_clearing_the_metric_picker_drops_the_tables_not_the_page(results):
    """`st.stop()` here would take the episode browser off the page too."""
    results({"libero_goal": _metrics(FOUR)})

    at = _run()
    at = at.multiselect(key="pe_rubric_metrics").set_value([]).run()

    assert not at.exception, at.exception
    assert _table_html(at) is None
    assert any("at least one metric" in (i.value or "") for i in at.info)
    assert "Episode browser" in [h.value for h in at.header]


def test_the_metric_picker_chooses_the_columns(results):
    results({"libero_goal": _metrics(FOUR)})

    at = _run()
    at = at.multiselect(key="pe_rubric_metrics").set_value(["voc", "mae"]).run()

    html = _table_html(at)

    assert ">VOC<" in html and ">MAE<" in html
    assert ">Kendall<" not in html and ">AUROC<" not in html


def test_pearson_is_offered_even_though_it_is_not_a_default(results):
    """It is very nearly VOC printed twice (r = 0.989 over the datasets this
    table counts), so it is off by default — but dropping it from the page
    entirely would make a reader who wants it re-run the benchmark."""
    results({"libero_goal": _metrics(FOUR)})

    at = _run()

    # AppTest reports a multiselect's options as the FORMATTED labels, while
    # set_value below takes the underlying values.
    assert "Pearson" in at.multiselect(key="pe_rubric_metrics").options
    assert ">Pearson<" not in _table_html(at)

    at = at.multiselect(key="pe_rubric_metrics").set_value(["voc", "pearson"]).run()

    assert ">Pearson<" in _table_html(at)


def test_the_export_caption_counts_the_datasets_and_names_the_folds(results):
    """A caption is the part of a table a reader trusts without checking, so
    both numbers in it come from the artifacts: the dataset count from what the
    table actually counted, the fold count from each run's own report header."""
    store = results({"libero_goal": _metrics(FOUR), "mimicgen_stack_d0": _metrics(FOUR)})
    (store / "libero_goal_report.md").write_text(
        "# Progress-estimation benchmark: libero_goal (1000 eps, 3 fold(s), eval=334)\n"
    )
    (store / "mimicgen_stack_d0_report.md").write_text(
        "# Progress-estimation benchmark: mimicgen_stack_d0 (50 eps, 3 fold(s), eval=16)\n"
    )

    at = _run()
    captions = [c.value or "" for c in at.text_input]

    assert any(
        "over 2 datasets in total" in c and "means over 3 cross-validation folds" in c
        for c in captions
    ), captions


def test_a_caption_says_nothing_about_folds_when_the_runs_disagree(results):
    """One sentence cannot honestly cover a 3-fold run and a 5-fold one."""
    store = results({"libero_goal": _metrics(FOUR), "mimicgen_stack_d0": _metrics(FOUR)})
    (store / "libero_goal_report.md").write_text("# x (1000 eps, 3 fold(s), eval=334)\n")
    (store / "mimicgen_stack_d0_report.md").write_text("# y (50 eps, 5 fold(s), eval=10)\n")

    captions = [c.value or "" for c in _run().text_input]

    assert any("datasets in total" in c for c in captions)
    assert not any("cross-validation folds" in c for c in captions)


def test_the_rubric_column_says_what_its_number_counts(results):
    """ "Dataset group", because the parenthetical is a count of DATASETS: on
    most rows it is not a task count -- the real-world sets are three tasks with
    data variants, the robomimic ones five tasks in {ph, mh}."""
    results({"libero_goal": _metrics(FOUR), "set_table_baseline": _metrics(FOUR)})

    html = _table_html(_run())
    header = html[html.index("<thead>") : html.index("</thead>")]

    assert ">Source (#datasets)<" in header
    # Not tasks: on most rows that is a different number (the real-world sets
    # are three tasks with data variants, robomimic five in {ph, mh}).
    assert "Tasks" not in header
    # And the real-robot row does not read as the public DROID dataset.
    assert "Real-world (1)" in html
    assert "droid" not in html.lower()


def _sweep(datasets: dict[str, dict]) -> dict:
    """A `sweep_results.json` blob, as `sweep_progress_allds.py` writes one."""
    return datasets


def test_a_sweep_that_chose_k_on_the_eval_fold_is_not_displayed(results, tmp_path):
    """The number is a maximum over five readings of the data it calls unseen.
    Not a footnote: the table is what gets quoted."""
    store = results({"libero_goal": _metrics(FOUR)})
    leaky = {
        "voc_max_on_test": {
            "n_eps": 50,
            "best_k": 32,
            "ks_swept": [8, 16, 32],
            "per_k": {"32": {"graph_direct:stage": {m: 0.99 for m in METRICS}}},
        }
    }
    (store / "sweep_results.json").write_text(json.dumps(leaky))

    at = _run()
    captions = " ".join(c.value or "" for c in at.caption)
    options = [o for w in at.selectbox for o in (w.options or [])]

    assert "maximum over the data it calls unseen" in captions
    assert not any("sweep" in str(o).lower() for o in options), options


def test_a_nested_sweep_is_displayed(results):
    """Selection on an inner split of train is exactly what makes it reportable."""
    store = results({"libero_goal": _metrics(FOUR)})
    nested = {
        "libero_goal": {
            "n_eps": 50,
            "selection": "nested",
            "best_k": "selected",
            "k_per_fold": [8, 16, 8],
            "per_k": {"selected": {"graph_direct:stage": {m: 0.9 for m in METRICS}}},
        }
    }
    (store / "sweep_results.json").write_text(json.dumps(nested))

    at = _run()
    captions = " ".join(c.value or "" for c in at.caption)

    assert "maximum over the data it calls unseen" not in captions
