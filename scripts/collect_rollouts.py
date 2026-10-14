"""Collect policy rollouts with embeddings in the canonical layout.

Usage::

    python scripts/collect_rollouts.py env=robomimic_lift_lowdim \\
        checkpoint=/path/to/ckpt num_episodes=50

The resulting directory is the input to the behavior-graph pipeline
(``python -m behavior_graphs.pipeline.run rollout_dir=<output>``).
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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="collect_rollouts")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    adapter = instantiate(cfg.env)
    # Pass the raw string: gs:// checkpoint URIs must not go through Path().
    policy = adapter.load_policy(cfg.checkpoint)
    log.info("collecting %d %s episodes -> %s", cfg.num_episodes, adapter.name, output_dir)
    kwargs = OmegaConf.to_container(cfg.rollout_kwargs, resolve=True) or {}
    rollouts = adapter.collect_rollouts(
        policy, output_dir, num_episodes=cfg.num_episodes, seed=cfg.seed, **kwargs
    )
    episodes = rollouts.episodes()
    successes = sum(1 for ep in episodes if ep.success)
    log.info(
        "%d episodes (%d successes), layers: %s",
        len(episodes),
        successes,
        rollouts.embedding_layers(),
    )


if __name__ == "__main__":
    main()
