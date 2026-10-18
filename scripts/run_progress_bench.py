"""Hydra entry point for the progress-estimation benchmark.

Instantiates a collection of methods + metrics + a split + a dataset from
`configs/progress_estimation/bench.yaml` and runs
:class:`~behavior_graphs.progress_estimation.runner.BenchmarkRunner`, writing
report.md / metrics.json / *_curves.npz to `out`. Replaces the old
`eval_progress_benchmark.py` (graph) and `vlm_reward_bench.py` (VLM) scripts —
graph vs VLM runs are just different method selections (and venvs):

    # graph methods (identity K128) on libero_goal, aligned eval:
    DATA_PATH=${DATA_PATH} PYTHONPATH=src python scripts/run_progress_bench.py \
      data.name=libero_goal backbone.clustering.n_clusters=128

    # a VLM baseline (robometer venv + GPU): select only that method
    ... '~methods.graph_stage' '~methods.graph_absorb' '~methods.frame_index' \
      +method@methods.robometer_4b=robometer_4b

Methods are added with ``+method@methods.<key>=<config>`` and removed with
``~methods.<key>``. See docs/progress_estimation.md for the Tab. II recipe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thread cap first: it only takes effect before numpy/sklearn initialise.
# See scripts/_threads.py for the measurement that made this necessary.
import _threads  # noqa: F401,E402  (import for its side effect)
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_graphs.progress_estimation.io import EpisodeSet, policy_embedding_layer  # noqa: E402
from behavior_graphs.progress_estimation.runner import BenchmarkRunner  # noqa: E402

# Root at configs/progress_estimation/, so `method`, `metric`, `backbone` and
# `split` are top-level groups that `+method@methods.<name>=<name>` can address.
# bench.yaml puts $BG_CONFIGS_ROOT on hydra's search path to reach the shared
# configs/data/* dataset map one directory up.
CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATH = str(CONFIGS_ROOT / "progress_estimation")
os.environ.setdefault("BG_CONFIGS_ROOT", str(CONFIGS_ROOT))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bench")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    # Resolve `auto` against policy_embeddings/ only. load_rollouts' own auto
    # ordering falls through to observations/state.npz, so a dataset without an
    # obs.npz silently gets its graph built on raw simulator state while the
    # report still labels the method "pi0.5 emb" -- which is how an earlier
    # cross-dataset pass came to be published wrong. sweep_progress_allds.py
    # has always used this guard; this entrypoint did not.
    layer = d.get("layer", "auto")
    if layer == "auto":
        layer = policy_embedding_layer(Path(d.rollout_dir))
    data = EpisodeSet.from_dir(
        name=d.name,
        rollout_dir=d.rollout_dir,
        layer=layer,
        suite=d.get("suite", ""),
        instruction=d.get("instruction", "") or "",
        instructions_json=d.get("instructions_json", None),
    )

    backbone = instantiate(cfg.backbone) if cfg.get("backbone") is not None else None
    methods = []
    for mc in cfg.methods.values():
        m = instantiate(mc)
        if getattr(m, "needs_backbone", False):
            m.backbone = backbone
        methods.append(m)
    metrics = [instantiate(mc) for mc in cfg.metrics.values()]
    split = instantiate(cfg.split)

    BenchmarkRunner(metrics, split, out_dir=cfg.out).run(data, methods)


if __name__ == "__main__":
    main()
