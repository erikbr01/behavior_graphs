"""The bare pipeline command must work on more than one embedding family.

`configs/pipeline/features/policy_embedding.yaml` defaulted to `layer: pi05`,
which is LIBERO's tap and nothing else's. So the command README.md and
REPRODUCE.md both give as the shortest path from a rollout collection to a
graph --

    python -m behavior_graphs.pipeline.run rollout_dir=<...>

-- died with `FileNotFoundError: layer 'pi05' not found` on every other
family: `cond_proj` (DROID), `bottleneck_plan_t0` (sim), `obs_encoder`
(RoboCasa), `action_conditioning` (generalist VLAs). Four of five collections
tried. `resolve_layer` already existed for exactly this and had no caller.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from behavior_graphs.pipeline.dataset_resolution import resolve_layer_if_applicable


def _collection(root, layers: dict[str, int], n: int = 3):
    (root / "episodes").mkdir(parents=True)
    (root / "episodes" / "metadata.yaml").write_text(
        yaml.safe_dump({"episodes": [{"index": i, "length": 4, "success": True} for i in range(n)]})
    )
    (root / "policy_embeddings").mkdir()
    for name, dim in layers.items():
        np.savez(
            root / "policy_embeddings" / f"{name}.npz", embeddings=np.zeros((4 * n, dim), "float32")
        )
    return root


@pytest.mark.parametrize(
    "layer",
    ["cond_proj", "bottleneck_plan_t0", "obs_encoder", "action_conditioning", "pi05"],
)
def test_a_single_layer_resolves_whatever_it_is_called(tmp_path, layer):
    root = _collection(tmp_path / layer, {layer: 8})
    assert resolve_layer_if_applicable({"layer": None}, root, None) == layer


def test_two_layers_is_an_error_naming_both(tmp_path):
    """Guessing would silently featurize the wrong signal."""
    root = _collection(tmp_path / "two", {"cond_proj": 8, "dinov3": 16})
    with pytest.raises(ValueError, match="cond_proj.*dinov3|dinov3.*cond_proj"):
        resolve_layer_if_applicable({"layer": None}, root, None)


def test_an_explicit_layer_still_wins(tmp_path):
    root = _collection(tmp_path / "two", {"cond_proj": 8, "dinov3": 16})
    assert resolve_layer_if_applicable({"layer": "dinov3"}, root, "dinov3") == "dinov3"


def test_a_stage_with_no_layer_field_resolves_nothing(tmp_path):
    """Raw-state features have no tap to resolve; they must not be given one."""
    root = _collection(tmp_path / "one", {"cond_proj": 8})
    assert resolve_layer_if_applicable({}, root, None) is None


def test_the_shipped_default_is_auto():
    """A fixed default is right for exactly one of the five families."""
    import pathlib

    cfg = yaml.safe_load(
        (
            pathlib.Path(__file__).resolve().parents[1]
            / "configs/pipeline/features/policy_embedding.yaml"
        ).read_text()
    )
    assert cfg["layer"] is None, "a hardcoded layer breaks every family but one"
