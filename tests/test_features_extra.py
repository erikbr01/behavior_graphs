"""Tests for raw observation/action storage and the derived feature extractors."""

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.envs.trajectory_dir import RolloutDirectory, write_rollout_dir
from behavior_graphs.stages import (
    ConcatFeatures,
    PolicyEmbeddings,
    StateActionFeatures,
    StateFeatures,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

LENGTHS = [12, 8, 15]
TOTAL = sum(LENGTHS)
D_OBS = 5
D_ACTION = 3
D_EMB = 7


@pytest.fixture
def rollouts(tmp_path):
    rng = np.random.default_rng(0)
    return write_rollout_dir(
        tmp_path / "r",
        LENGTHS,
        [True, False, None],
        embeddings={"pi05": rng.normal(size=(TOTAL, D_EMB)).astype(np.float32)},
        observations={"state": rng.normal(size=(TOTAL, D_OBS)).astype(np.float32)},
        actions=rng.normal(size=(TOTAL, D_ACTION)).astype(np.float32),
    )


@pytest.fixture
def bare_rollouts(tmp_path):
    """Rollouts without observations or actions (the pre-existing layout)."""
    rng = np.random.default_rng(1)
    return write_rollout_dir(
        tmp_path / "bare",
        LENGTHS,
        [True, False, None],
        embeddings={"pi05": rng.normal(size=(TOTAL, D_EMB)).astype(np.float32)},
    )


class TestLayoutRoundtrip:
    def test_observations_and_actions(self, rollouts):
        reread = RolloutDirectory(rollouts.path)
        assert reread.observation_keys() == ["state"]
        obs = reread.observations("state")
        assert obs.shape == (TOTAL, D_OBS)
        assert obs.dtype == np.float32
        actions = reread.actions()
        assert actions.shape == (TOTAL, D_ACTION)
        assert actions.dtype == np.float32

    def test_optional_arrays_absent(self, bare_rollouts):
        assert bare_rollouts.observation_keys() == []
        assert bare_rollouts.actions() is None
        with pytest.raises(FileNotFoundError, match="available"):
            bare_rollouts.observations("state")

    def test_unknown_observation_key(self, rollouts):
        with pytest.raises(FileNotFoundError, match="state"):
            rollouts.observations("nope")

    def test_misaligned_observations_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="rows"):
            write_rollout_dir(
                tmp_path / "r",
                [5, 6],
                [True, True],
                embeddings={},
                observations={"state": np.zeros((10, 2), dtype=np.float32)},
            )

    def test_misaligned_actions_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="rows"):
            write_rollout_dir(
                tmp_path / "r",
                [5, 6],
                [True, True],
                embeddings={},
                actions=np.zeros((10, 2), dtype=np.float32),
            )


class TestStateFeatures:
    def test_extract(self, rollouts):
        features = StateFeatures(key="state").extract(rollouts)
        assert features.dim == D_OBS
        assert features.num_samples == TOTAL
        assert np.allclose(features.features, rollouts.observations("state"))

    def test_missing_key(self, rollouts):
        with pytest.raises(FileNotFoundError, match="state"):
            StateFeatures(key="missing").extract(rollouts)


class TestStateActionFeatures:
    def test_extract(self, rollouts):
        features = StateActionFeatures(key="state", action_weight=2.0).extract(rollouts)
        assert features.dim == D_OBS + D_ACTION
        assert features.num_samples == TOTAL
        assert np.allclose(features.features[:, :D_OBS], rollouts.observations("state"))
        assert np.allclose(features.features[:, D_OBS:], 2.0 * rollouts.actions())

    def test_missing_actions(self, bare_rollouts, tmp_path):
        rng = np.random.default_rng(2)
        rollouts = write_rollout_dir(
            tmp_path / "obs_only",
            LENGTHS,
            [True, False, None],
            embeddings={},
            observations={"state": rng.normal(size=(TOTAL, D_OBS)).astype(np.float32)},
        )
        with pytest.raises(FileNotFoundError, match="actions"):
            StateActionFeatures(key="state").extract(rollouts)

    def test_missing_key(self, rollouts):
        with pytest.raises(FileNotFoundError, match="missing"):
            StateActionFeatures(key="missing").extract(rollouts)


class TestConcatFeatures:
    def test_extract(self, rollouts):
        extractor = ConcatFeatures([StateFeatures(key="state"), PolicyEmbeddings(layer="pi05")])
        features = extractor.extract(rollouts)
        assert features.dim == D_OBS + D_EMB
        assert features.num_samples == TOTAL
        assert np.allclose(features.features[:, :D_OBS], rollouts.observations("state"))
        assert np.allclose(features.features[:, D_OBS:], rollouts.embeddings("pi05"))

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            ConcatFeatures([])


class TestConfigs:
    def test_concat_config_instantiates_and_extracts(self, rollouts):
        cfg = OmegaConf.load(CONFIGS / "pipeline" / "features" / "state_and_embeddings.yaml")
        extractor = instantiate(cfg)
        assert isinstance(extractor, FeatureExtractor)
        assert all(isinstance(e, FeatureExtractor) for e in extractor.extractors)
        features = extractor.extract(rollouts)
        assert features.dim == D_OBS + D_EMB

    @pytest.mark.parametrize("name", ["state", "state_action"])
    def test_simple_configs_extract(self, name, rollouts):
        cfg = OmegaConf.load(CONFIGS / "pipeline" / "features" / f"{name}.yaml")
        extractor = instantiate(cfg)
        assert isinstance(extractor, FeatureExtractor)
        assert extractor.extract(rollouts).num_samples == TOTAL
