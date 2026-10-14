"""Write a vision-backbone embedding of episode video as a new layer.

The behavior-graph pipeline can embed frames on the fly
(``pipeline/features@features=dino_patch``), but everything else in the repo
addresses representations by *layer name* -- the demo app's layer picker, the
sweep runner's per-layer passes, the outcome classifiers -- and reads them out
of ``policy_embeddings/<layer>.npz``. This script materializes the frame
features into that file, so a dataset collected from an uninstrumented policy
gains a representation that those consumers can use, and no later job needs a
GPU::

    python scripts/utils/extract_video_features.py rollout_dir=<rollouts>
    python scripts/utils/extract_video_features.py rollout_dir=<rollouts> \\
        pipeline/features@features=dino_pooled layer=dino_pooled
    python scripts/utils/extract_video_features.py rollout_dir=<rollouts> \\
        features.model_name=facebook/dinov2-large features.grid=3 layer=dino_l3

The layer is added in place, beside whatever layers the directory already has;
an existing layer of the same name is kept unless ``overwrite=true``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from behavior_graphs.envs.trajectory_dir import (
    EMBEDDINGS_DIR,
    TrajectoryDirectory,
    add_embedding_layer,
)

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="extract_video_features")
def main(cfg: DictConfig) -> None:
    trajectory_dir = Path(cfg.rollout_dir)
    trajectories = TrajectoryDirectory(trajectory_dir)
    # Checked here as well as in add_embedding_layer, so an unusable layer name
    # fails before the GPU work rather than after it.
    if (trajectory_dir / EMBEDDINGS_DIR / f"{cfg.layer}.npz").exists() and not cfg.overwrite:
        raise FileExistsError(
            f"{trajectory_dir / EMBEDDINGS_DIR / f'{cfg.layer}.npz'} already exists; pass "
            "overwrite=true to replace it, or layer=<other name> to write a second "
            "representation"
        )

    extractor = instantiate(cfg.features)
    log.info(
        "embedding %d episodes of %s with %s -> layer '%s'",
        len(trajectories.episodes()),
        trajectory_dir,
        getattr(extractor, "model_name", type(extractor).__name__),
        cfg.layer,
    )
    features = extractor.extract(trajectories)

    target = add_embedding_layer(
        trajectory_dir, cfg.layer, features.features, overwrite=bool(cfg.overwrite)
    )
    log.info(
        "wrote %s: %d rows x %d dims (%.1f MB); layers now %s",
        target,
        features.num_samples,
        features.dim,
        target.stat().st_size / 1e6,
        trajectories.embedding_layers(),
    )


if __name__ == "__main__":
    main()
