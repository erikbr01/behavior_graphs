"""A dataset's ``display_name`` is applied at render time only.

Its config key stays the sweep task-dir name, which is what keys the allowlist,
the best-per-K lookup and the selectors' widget/query params.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from behavior_graphs.viz.streamlit.data import (
    _data_group_labels,
    _resolve_task,
    dataset_display_name,
    display_dataset_names,
    task_display_name,
    task_display_names,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfgs(tmp_path, monkeypatch):
    """Writes a temp ``configs/data/shown.yaml`` and points the lookup at it."""

    def _make(body: str) -> Path:
        root = tmp_path / "configs"
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "data" / "shown.yaml").write_text(body)
        monkeypatch.setattr("behavior_graphs.viz.streamlit.data.configs_dir", lambda: root)
        monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "shown")
        return root

    return _make


def test_plain_string_entries_have_no_display_name(cfgs):
    cfgs("lift: /data/rollouts/lift\n")
    assert task_display_names() == {}
    assert task_display_name("lift") == "lift"


def test_display_name_is_read_from_the_dataset_entry(cfgs):
    cfgs("lift_x:\n  path: /data/rollouts/lift_x\n  display_name: lift\n")
    assert task_display_names() == {"lift_x": "lift"}
    assert task_display_name("lift_x") == "lift"


def test_unlabelled_tasks_pass_through(cfgs):
    """Only the exceptions are listed, so anything else must display as-is."""
    cfgs("lift_x:\n  path: /d/lift_x\n  display_name: lift\ncan: /d/can\n")
    assert task_display_name("can") == "can"
    assert task_display_name("a_task_not_in_any_config") == "a_task_not_in_any_config"


def test_blank_display_name_is_not_an_override(cfgs):
    """`display_name:` with nothing after it must not blank out the label."""
    cfgs("lift:\n  path: /d/lift\n  display_name:\n")
    assert task_display_name("lift") == "lift"


def test_labels_compose_through_defaults(cfgs):
    """A group inherits its parents' labels, as it does their datasets."""
    root = cfgs("defaults:\n  - parent\n  - _self_\ncan: /d/can\n")
    (root / "data" / "parent.yaml").write_text("lift_x:\n  path: /d/lift_x\n  display_name: lift\n")
    assert task_display_names() == {"lift_x": "lift"}


def test_group_entry_wins_over_an_inherited_namesake(cfgs):
    root = cfgs("defaults:\n  - parent\n  - _self_\nlift:\n  path: /d/lift\n  display_name: mine\n")
    (root / "data" / "parent.yaml").write_text("lift:\n  path: /d/lift\n  display_name: theirs\n")
    assert task_display_name("lift") == "mine"


def test_dataset_display_name_relabels_only_the_last_segment(cfgs):
    """Selectors label options with a recorded rollout_dir, not a bare name."""
    cfgs("lift_x:\n  path: /d/lift_x\n  display_name: lift\n")
    assert dataset_display_name("rollouts/lift_x") == "rollouts/lift"
    assert dataset_display_name("/data/rollouts/lift_x") == "/data/rollouts/lift"
    assert dataset_display_name("lift_x") == "lift"
    assert dataset_display_name("lift_x/rollouts") == "lift_x/rollouts"
    # Both helpers are passed bare as a selectbox format_func, so the pickers'
    # "(all datasets)" option goes through them and has to survive unchanged.
    assert dataset_display_name("(all datasets)") == "(all datasets)"
    assert task_display_name("(all datasets)") == "(all datasets)"


def test_a_shared_url_carrying_a_label_still_selects_the_run(cfgs):
    """`?task=` holds whatever was copied off the page, which is the label."""
    cfgs("lift_x:\n  path: /d/lift_x\n  display_name: lift\n")
    tasks = ["can", "lift_x"]
    assert _resolve_task("lift", tasks) == "lift_x"  # the label
    assert _resolve_task("lift_x", tasks) == "lift_x"  # the key
    # Neither: with format_func set, Streamlit compares labels and would accept
    # this as a value, leaving the page on a task that matches no run.
    assert _resolve_task("gone", tasks) == "can"


def test_unset_display_group_falls_back_to_all(monkeypatch):
    """A local checkout (no allowlist) must still show the same labels."""
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    assert task_display_name("set_table_baseline_pretrained_dit") == "set_table_baseline_dit"


def test_missing_group_file_means_no_labels(tmp_path, monkeypatch):
    """Absent configs/ is a bare-install case, not a reason to fail a page."""
    monkeypatch.setattr("behavior_graphs.viz.streamlit.data.configs_dir", lambda: tmp_path)
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    assert task_display_names() == {}


def test_repo_droid_group_drops_pretrained_from_the_set_table_runs():
    """The shipped labels: the four rgrl set_table runs read without 'pretrained'."""
    labels = _data_group_labels("droid", REPO / "configs")
    assert {n: v for n, v in labels.items() if v} == {
        "set_table_baseline_pretrained_dit": "set_table_baseline_dit",
        "set_table_plusleft_pretrained_dit": "set_table_plusleft_dit",
        "set_table_plusmiddle_pretrained_dit": "set_table_plusmiddle_dit",
        "set_table_plusright_pretrained_dit": "set_table_plusright_dit",
    }


def test_shipped_labels_keep_the_rollout_paths_intact():
    """The relabel is display-only: the dir on disk still has 'pretrained'."""
    cfg = yaml.safe_load((REPO / "configs" / "data" / "droid.yaml").read_text())
    path = cfg["set_table_baseline_pretrained_dit"]["path"]
    assert path.endswith("/rollouts/set_table_baseline_pretrained_dit")


def test_no_shipped_label_collides_with_a_real_dataset():
    """A label equal to another dataset's name would be two runs, one name."""
    labels = _data_group_labels("visualize", REPO / "configs")
    assert {v for v in labels.values() if v}.isdisjoint(set(labels))


def test_no_two_shipped_datasets_share_a_label():
    """Two identical options in the Task selector are indistinguishable."""
    shown = [v for v in _data_group_labels("visualize", REPO / "configs").values() if v]
    assert len(set(shown)) == len(shown)


def test_labelled_datasets_are_keyed_by_their_rollout_dir_name():
    """`dataset_display_name` looks a label up by the option's trailing dir name.

    That is only the same thing as the config key while a labelled dataset's key
    equals its path's last segment. Nothing in Hydra enforces that, so assert it
    here rather than adding a path index the pickers would have to consult.
    """
    cfg = yaml.safe_load((REPO / "configs" / "data" / "droid.yaml").read_text())
    labelled = {k: v for k, v in cfg.items() if isinstance(v, dict) and v.get("display_name")}
    assert labelled
    assert all(Path(v["path"]).name == k for k, v in labelled.items())


def test_unknown_key_in_an_entry_is_logged(cfgs, caplog):
    """A misspelled `display-name:` otherwise looks like a stale label cache."""
    cfgs("lift:\n  path: /d/lift\n  display-name: Lift\n")
    with caplog.at_level(logging.WARNING):
        assert task_display_names() == {}
    assert "display-name" in caplog.text and "'lift'" in caplog.text


def test_a_non_string_display_name_is_still_a_usable_label(cfgs):
    """Streamlit's format_func result must be a str; a YAML int/date isn't."""
    cfgs("lift:\n  path: /d/lift\n  display_name: 2024\n")
    assert task_display_name("lift") == "2024"


def test_an_unreadable_group_logs_and_shows_raw_names(tmp_path, monkeypatch, caplog):
    """A half-saved bind-mounted config must not take every page down."""
    root = tmp_path / "configs"
    (root / "data").mkdir(parents=True)
    (root / "data" / "shown.yaml").write_text("lift: [unclosed\n")
    monkeypatch.setattr("behavior_graphs.viz.streamlit.data.configs_dir", lambda: root)
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "shown")
    with caplog.at_level(logging.WARNING):
        assert task_display_name("lift") == "lift"
    assert "unreadable" in caplog.text
    # ...but the allowlist that same file feeds still refuses to guess.
    with pytest.raises(yaml.YAMLError):
        display_dataset_names()
