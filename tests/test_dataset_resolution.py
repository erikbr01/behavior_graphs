"""behavior_graphs.pipeline.dataset_resolution."""

import logging

import numpy as np
import pytest
from omegaconf import OmegaConf

from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.pipeline.dataset_resolution import (
    available_layers,
    resolve_layer,
    resolve_layer_if_applicable,
)


def test_available_layers_reports_actual_layers(tmp_path):
    rng = np.random.default_rng(0)
    write_rollout_dir(
        tmp_path / "ds", [3], [True], {"pi05": rng.normal(size=(3, 4)).astype(np.float32)}
    )
    assert available_layers([str(tmp_path / "ds")]) == {str(tmp_path / "ds"): {"pi05"}}


def test_available_layers_reports_empty_for_nonexistent_or_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    missing = tmp_path / "does_not_exist"
    result = available_layers([str(empty_dir), str(missing)])
    assert result == {str(empty_dir): set(), str(missing): set()}


def test_available_layers_logs_a_warning_for_real_errors_but_still_returns_empty(tmp_path, caplog):
    # A directory that IS a trajectory dir (has metadata.yaml) but with
    # corrupted YAML -- a real error, not "no data written yet".
    ds = tmp_path / "corrupted"
    (ds / "episodes").mkdir(parents=True)
    (ds / "episodes" / "metadata.yaml").write_text("not: valid: yaml: [")

    with caplog.at_level(logging.WARNING):
        result = available_layers([str(ds)])

    assert result == {str(ds): set()}
    # Wording is main's ("could not read embedding layers of ...") after the
    # merge; what matters is that it said something and named the dataset.
    assert any("embedding layers" in r.message and str(ds) in r.message for r in caplog.records)


def test_resolve_layer_still_works_after_dataset_resolution_move(tmp_path):
    rng = np.random.default_rng(0)
    write_rollout_dir(
        tmp_path / "ds", [3], [True], {"only_layer": rng.normal(size=(3, 4)).astype(np.float32)}
    )
    assert resolve_layer(tmp_path / "ds", None) == "only_layer"
    with pytest.raises(ValueError, match="not in"):
        resolve_layer(tmp_path / "ds", "missing_layer")


def test_resolve_layer_ambiguous_without_override_raises(tmp_path):
    rng = np.random.default_rng(0)
    write_rollout_dir(
        tmp_path / "ds",
        [3],
        [True],
        {
            "a": rng.normal(size=(3, 4)).astype(np.float32),
            "b": rng.normal(size=(3, 4)).astype(np.float32),
        },
    )
    with pytest.raises(ValueError, match="disambiguate"):
        resolve_layer(tmp_path / "ds", None)


class TestResolveLayerIfApplicable:
    def test_resolves_when_features_cfg_has_a_layer_field(self, tmp_path):
        rng = np.random.default_rng(0)
        write_rollout_dir(
            tmp_path / "ds",
            [3],
            [True],
            {"only_layer": rng.normal(size=(3, 4)).astype(np.float32)},
        )
        features_cfg = OmegaConf.create({"_target_": "x", "layer": None})
        assert resolve_layer_if_applicable(features_cfg, tmp_path / "ds", None) == "only_layer"

    def test_returns_none_when_features_cfg_has_no_layer_field(self, tmp_path):
        features_cfg = OmegaConf.create({"_target_": "x"})
        assert resolve_layer_if_applicable(features_cfg, tmp_path / "ds", "whatever") is None
