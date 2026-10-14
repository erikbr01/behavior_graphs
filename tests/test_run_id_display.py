"""Tests for showing the selected run's id in the picker sidebar.

The sidebar names a run by the hyperparameters it filters on, which is not what
anything else calls it: the sweep index, the run dirs on disk and the stage
caches are all keyed by the job hash. Without the hash on screen there is no way
to get from "the run I am looking at" to a row in `sweep_index.csv`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_SCRIPT = """
import behavior_graphs.viz.streamlit.data as data
data._run_index.clear()
data.select_run_dir()
"""


def _run(root, task: str, job_hash: str, k: int = 5):
    d = root / task / job_hash
    d.mkdir(parents=True)
    (d / "windows.json").write_text("{}")
    (d / "labels.npy").write_bytes(b"")
    (d / "metrics.json").write_text("{}")
    (d / "config.yaml").write_text(
        f"clustering: {{n_clusters: {k}}}\nwindowing: {{width: 5, stride: 2}}\n"
    )
    return d


def _picker(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_RUNS_ROOT", str(tmp_path))
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    monkeypatch.delenv("BEHAVIOR_GRAPHS_RUN_DIR", raising=False)
    at = AppTest.from_string(_SCRIPT, default_timeout=30)
    return at.run()


def test_the_picker_shows_the_selected_runs_job_hash(tmp_path, monkeypatch):
    _run(tmp_path, "set_table", "a1b2c3d4e5f6")
    at = _picker(tmp_path, monkeypatch)

    assert not at.exception
    assert "a1b2c3d4e5f6" in [c.value for c in at.code]


def test_the_shown_id_follows_the_selectors(tmp_path, monkeypatch):
    """Otherwise it is worse than nothing: a stale hash names the wrong run."""
    _run(tmp_path, "set_table", "aaaaaaaaaaaa", k=5)
    _run(tmp_path, "set_table", "bbbbbbbbbbbb", k=9)
    at = _picker(tmp_path, monkeypatch)

    shown = [c.value for c in at.code]
    assert len(shown) == 1
    (k_select,) = [s for s in at.sidebar.selectbox if s.label.startswith("K")]
    # AppTest reports options as strings but `value` as the picked object.
    other = next(v for v in k_select.options if v != str(k_select.value))
    at = k_select.select(other).run()

    assert not at.exception
    assert [c.value for c in at.code] != shown


def test_a_pasted_path_is_identified_too(tmp_path, monkeypatch):
    """The manual box takes a path; what comes back is still worth naming."""
    pasted = _run(tmp_path, "set_table", "ffffffffffff")
    _run(tmp_path, "set_table", "aaaaaaaaaaaa", k=9)
    at = _picker(tmp_path, monkeypatch)
    at = at.sidebar.text_input[0].set_value(str(pasted)).run()

    assert not at.exception
    assert [c.value for c in at.code] == ["ffffffffffff"]
