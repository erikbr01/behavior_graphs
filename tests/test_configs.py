"""Every config group entry must instantiate to the advertised interface."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from behavior_graphs.core.stages import (
    Clustering,
    DimReduction,
    FeatureExtractor,
    GraphConstruction,
    Windowing,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Group dir (relative to configs/) -> the interface its entries must satisfy.
# Paths, not bare names: the stage groups live under configs/pipeline/ and
# viz_reduction under configs/viz/, so a bare name globs an empty directory
# and silently parametrizes away to nothing.
GROUP_INTERFACES = {
    "pipeline/features": FeatureExtractor,
    "pipeline/windowing": Windowing,
    "pipeline/reduction": DimReduction,
    "viz/viz_reduction": DimReduction,
    "pipeline/clustering": Clustering,
    "pipeline/graph": GraphConstruction,
    # The clusterers the subclustering stage (and the app's on-demand
    # expansion) pick from by name.
    "pipeline/subclustering/algo": Clustering,
}


def _group_entries():
    for group, interface in GROUP_INTERFACES.items():
        entries = sorted((CONFIGS / group).glob("*.yaml"))
        assert entries, f"no configs found under configs/{group}"
        for path in entries:
            yield pytest.param(path, interface, id=f"{group}/{path.stem}")


@pytest.mark.parametrize("path,interface", _group_entries())
def test_config_instantiates(path: Path, interface: type) -> None:
    cfg = OmegaConf.load(path)
    target = cfg["_target_"]
    if "umap" in target:
        pytest.importorskip("umap")
    if "hdbscan" in target:
        pytest.importorskip("hdbscan")
    obj = instantiate(cfg)
    assert isinstance(obj, interface)


def test_evaluation_configs_instantiate() -> None:
    from behavior_graphs.core.stages import Evaluator

    for path in sorted((CONFIGS / "pipeline" / "evaluation").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        evaluators = instantiate(cfg.evaluators)
        for ev in evaluators:
            assert isinstance(ev, Evaluator)
            assert isinstance(ev.name, str) and ev.name


def test_subclustering_configs_instantiate() -> None:
    from behavior_graphs.core.stages import SubClustering

    paths = sorted((CONFIGS / "pipeline" / "subclustering").glob("*.yaml"))
    assert paths, "no subclustering stage configs"
    for path in paths:
        cfg = OmegaConf.load(path)
        stage = instantiate(cfg)
        assert isinstance(stage, SubClustering)
        assert isinstance(stage.name, str) and stage.name


def test_subclustering_is_off_by_default() -> None:
    """The stage must stay opt-in: enabling it rewrites the graph's terminals."""
    from behavior_graphs.pipeline.run import compose_pipeline_config

    cfg = compose_pipeline_config(["rollout_dir=/x"])
    assert cfg.get("subclustering") is None

    enabled = compose_pipeline_config(
        ["rollout_dir=/x", "pipeline/subclustering@subclustering=terminal"]
    )
    assert enabled.subclustering._target_.endswith("TerminalSubClustering")


def test_labeling_configs_instantiate() -> None:
    from behavior_graphs.core.stages import NodeLabeler

    for path in sorted((CONFIGS / "pipeline" / "labeling").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        labeler = instantiate(cfg)
        assert isinstance(labeler, NodeLabeler)
        assert isinstance(labeler.name, str) and labeler.name


def test_every_env_config_resolves_its_adapter() -> None:
    """The group `test_config_instantiates` never covered, and the gap that hid a bug.

    `envs/_hf_policy.py` carried a `from policy_graphs.core.env import Policy`
    left over from the package rename. That takes down `vla_foundry` and
    `robocasa365`, and with them four `configs/env/*.yaml` whose `_target_`
    can no longer be located -- `Robocasa365Adapter` among them, which the
    release explicitly keeps. Every test passed, because the venv the suite
    usually runs in has the OLD package installed and the import resolved.

    Locating the target is the whole check: these adapters take mandatory
    values (`dataset_path: ???`) and reach for submodules, so instantiating
    them here would test the environment rather than the config.
    """
    from hydra.utils import get_class, get_method

    from behavior_graphs.core.env import EnvAdapter

    paths = sorted((CONFIGS / "env").glob("*.yaml"))
    assert paths, "no env configs found"
    for path in paths:
        target = OmegaConf.load(path).get("_target_")
        assert target, f"{path.name} has no _target_"
        try:
            obj = get_class(target)
        except Exception:  # a factory function rather than a class
            obj = get_method(target)
            continue
        assert issubclass(obj, EnvAdapter), f"{path.name}: {target} is not an EnvAdapter"


@pytest.mark.parametrize(
    "config_name,overrides",
    [
        ("config", []),
        ("sweep", []),
        # Both leave `env` mandatory, so name one; the point is that the rest
        # of the defaults list resolves, not that a bare compose works.
        ("collect_rollouts", ["env=robomimic_lift_lowdim"]),
        ("train_policy", ["env=robomimic_lift_lowdim"]),
    ],
)
def test_entry_point_configs_compose(config_name: str, overrides: list[str]) -> None:
    """A defaults list naming a file that no longer exists is a dead entry point.

    `configs/progress_estimation/bench.yaml` sat broken this way after a
    method config was removed with its reference left in place: the default
    command could not compose at all, and nothing noticed.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
    assert cfg is not None
