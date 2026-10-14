"""Featurize trajectories (demos or existing rollouts) with a trained policy.

To featurize training demonstrations::

    python scripts/utils/extract_trajectory_features.py env=robomimic_lift_lowdim \\
        env.dataset_path=/path/to/low_dim_abs.hdf5 checkpoint=/path/to/ckpt

To featurize existing rollouts that were collected without embeddings::

    python scripts/utils/extract_trajectory_features.py env=robomimic_lift_lowdim \\
        checkpoint=/path/to/ckpt rollout_dir=/path/to/rollouts

Writes the canonical trajectory layout (one embedding row per timestep), which
is what the pipeline reads::

    python -m behavior_graphs.pipeline.run rollout_dir=<output>
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="extract_trajectory_features")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    adapter = instantiate(cfg.env)
    # Pass the raw string: gs:// checkpoint URIs must not go through Path().
    policy = adapter.load_policy(cfg.checkpoint)
    kwargs = OmegaConf.to_container(cfg.extract_kwargs, resolve=True) or {}

    if cfg.get("rollout_dir"):
        log.info("featurizing %s rollouts from %s -> %s", adapter.name, cfg.rollout_dir, output_dir)
        result = adapter.extract_rollout_features(
            policy, Path(cfg.rollout_dir), output_dir, **kwargs
        )
        kind = "rollout"
    else:
        log.info("featurizing %s demos -> %s", adapter.name, output_dir)
        result = adapter.extract_demo_features(policy, output_dir, **kwargs)
        kind = "demo"

    episodes = result.episodes()
    log.info(
        "%d %s episodes, %d samples, layers: %s",
        len(episodes),
        kind,
        sum(ep.num_samples for ep in episodes),
        result.embedding_layers(),
    )


if __name__ == "__main__":
    main()
