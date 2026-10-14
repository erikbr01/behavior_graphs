"""Train a policy for one environment by delegating to its submodule.

Usage::

    python scripts/train_policy.py env=robomimic_lift_lowdim
    python scripts/train_policy.py env=robomimic_lift_lowdim \\
        train_overrides='{training.num_epochs: 50}'

All policy-learning code and hyperparameters live in the environment's
``third_party`` submodule; ``train_overrides`` are passed through verbatim to
the submodule's own config system.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested override dicts into dotted keys for the submodule."""
    flat = {}
    for key, value in d.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_policy")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    adapter = instantiate(cfg.env)
    overrides = _flatten(OmegaConf.to_container(cfg.train_overrides, resolve=True) or {})
    num_gpus = int(cfg.get("num_gpus", 1))
    if num_gpus > 1:
        log.info("training %s policy on %d GPUs -> %s", adapter.name, num_gpus, output_dir)
        policy = adapter.train_policy_multi_gpu(output_dir, num_gpus, **overrides)
    else:
        log.info("training %s policy -> %s", adapter.name, output_dir)
        policy = adapter.train_policy(output_dir, **overrides)
    log.info("checkpoint: %s", policy.checkpoint_path)


if __name__ == "__main__":
    main()
