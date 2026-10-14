"""Hydra entry point for the behavior-graph pipeline.

Single run::

    python -m behavior_graphs.pipeline.run rollout_dir=/path/to/rollouts

Sweep (hydra multirun; one bundled artifact directory per job)::

    python -m behavior_graphs.pipeline.run -m \\
        rollout_dir=/path/to/rollouts \\
        windowing.width=1,5,8 \\
        clustering=kmeans,gmm clustering.n_clusters=5,10,20 \\
        reduction=pca,umap

Every stage is instantiated from its config group via ``_target_``, so any
constructor argument of any stage is sweepable from the command line.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.subclusters import SubClusterSet
from behavior_graphs.envs.trajectory_dir import RolloutDirectory
from behavior_graphs.pipeline.artifacts import RunArtifacts
from behavior_graphs.pipeline.cache import (
    StageCache,
    data_fingerprint,
    embedding_fingerprint,
    stage_key,
)
from behavior_graphs.pipeline.folds import (
    resolve_split,
    subset_windows,
    window_mask,
)
from behavior_graphs.pipeline.shared_store import SHARED_STAGES
from behavior_graphs.stages.features.episode_norm import apply_episode_norm
from behavior_graphs.stages.graph.smoothing import in_clustering_space

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")

#: A features cache entry above this is declined; see `_worth_caching`.
#: 4 GB keeps every rgrl tap (1.5 GB for legos, the largest) and declines
#: every DINOv3 block (12-27 GB).
FEATURES_CACHE_MAX_BYTES = 4 * 1024**3


def compose_pipeline_config(overrides, config_name: str = "config") -> DictConfig:
    """Compose a top-level config outside a ``@hydra.main`` entry point.

    Shared by the sweep runner, the GUI pipeline builder, and tests. Clears
    any stale GlobalHydra state first, so it is safe to call repeatedly and
    from long-lived processes (Streamlit, notebooks).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    gh = GlobalHydra.instance()
    if gh.is_initialized():
        # Inside an outer @hydra.main (e.g. scripts/run_sweep.py) or an
        # active initialize context — reuse it; configs/ is on the search path.
        return compose(config_name=config_name, overrides=list(overrides))
    gh.clear()
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def _load_subclusters(cache: StageCache, key: str) -> SubClusterSet | None:
    """Cache read for the subclustering stage, back into its own type."""
    payload = cache.get_json("subclusters", key)
    return None if payload is None else SubClusterSet.from_dict(payload)


def run_pipeline(
    cfg: DictConfig,
    rollouts: RolloutDirectory,
    run_dir: Path,
    shared_root: Path | None = None,
) -> PipelineResult:
    """Execute all pipeline stages and write artifacts to ``run_dir``.

    When ``cfg.cache_dir`` is set, every expensive stage output is reused
    from the content-addressed :class:`StageCache`: sweep jobs that share an
    upstream config (data + features + windowing + ...) only ever compute
    the stages where they differ.

    ``shared_root`` additionally stops them *storing* it twice: the stage cache
    avoids recomputing a window set, but every job still wrote its own 93 MB copy
    of the result. Passed by the sweep runner (the root is the sweep dir), so a
    sweep's jobs share one copy per distinct output while a single run stays
    self-contained. See :mod:`behavior_graphs.pipeline.shared_store`.
    """
    artifacts = RunArtifacts(run_dir, shared_root=shared_root)
    OmegaConf.save(cfg, artifacts.run_dir / "config.yaml", resolve=True)
    result = PipelineResult()
    timings = {}
    cache_hits: dict = {}

    def timed(stage_name: str, fn):
        start = time.perf_counter()
        out = fn()
        timings[stage_name] = round(time.perf_counter() - start, 3)
        log.info("%s done in %.2fs", stage_name, timings[stage_name])
        return out

    cache = StageCache(Path(cfg.cache_dir)) if cfg.get("cache_dir") else None

    def cfg_dict(node):
        return OmegaConf.to_container(node, resolve=True)

    # Opt-in fit/score separation. Without `cfg.split` every stage fits and
    # scores the same episodes, exactly as before.
    split = resolve_split(cfg.get("split"), len(rollouts.episodes()))
    if split is not None:
        log.info(
            "split %s: %d fit / %d score episodes", split.label, len(split.fit), len(split.score)
        )

    keys: dict = {}
    if cache is not None:
        fp = data_fingerprint(rollouts.path)
        features_cfg = cfg_dict(cfg.features)
        # Re-extracting a policy's taps rewrites policy_embeddings/<layer>.npz
        # in place, which no other part of this key notices. Appended only when
        # the config actually reads such a file, so state and vision-backbone
        # features keep the entries they already have.
        emb = embedding_fingerprint(rollouts.path, features_cfg)
        # `episode_norm` transforms the extractor's output, so it belongs in the
        # features key. Appended only when set, so every existing entry keeps
        # the key it already has and nothing re-extracts.
        ep_norm = str(cfg.get("episode_norm", "none") or "none")
        parts = [
            fp,
            features_cfg,
            *([emb] if emb else []),
            *([ep_norm] if ep_norm != "none" else []),
        ]
        keys["features"] = stage_key(*parts)
        keys["windows"] = stage_key(keys["features"], cfg_dict(cfg.windowing))
        # Features and windowing are split-independent, so folds keep sharing
        # them. Everything fitted on a subset carries the fit set in its key.
        split_id = split.identity() if split is not None else None
        keys["reduced"] = stage_key(keys["windows"], cfg_dict(cfg.reduction), split_id)
        keys["labels"] = stage_key(keys["reduced"], cfg_dict(cfg.clustering), split_id)
        keys["coords_2d"] = stage_key(keys["windows"], cfg_dict(cfg.viz_reduction))
        # Hand the shared stages' keys to the writer. The stage cache saves
        # recomputing these; this is what saves re-serialising them. Set here
        # rather than at construction because the keys need the resolved configs.
        artifacts.stage_keys = {k: v for k, v in keys.items() if k in SHARED_STAGES}
        # The subclusters key also needs the graph config and the stage's own
        # fingerprint, but the latter comes from the instantiated stage (it
        # resolves configs the top-level cfg only names) — set below, once the
        # stage exists.

    def _passthrough_reduction() -> bool:
        """Whether ``reduction`` hands back the array it was given.

        ``IdentityReduction`` with no normalizer returns the windowed features
        unchanged, so a ``reduced`` entry is a second copy of the ``windows``
        entry -- and a full-width one: 857 MB for a 512-d layer at stride 1
        against the ~90 MB a 100-component reduction costs. Measured across the
        eight ABC tasks that duplication is 165 GB, about 40% of all reduced-
        cache volume, and it is recreated on every sweep that includes an
        identity arm.

        Declining it costs nothing to recompute: the "computation" is
        ``np.asarray(features, float32)`` over an array the windows stage has
        already produced or loaded.
        """
        red = cfg.get("reduction") or {}
        if not str(red.get("_target_", "")).endswith("IdentityReduction"):
            return False
        return str(red.get("normalize", "none") or "none") == "none"

    def _worth_caching(stage: str, out) -> bool:
        """Whether a stage output earns the disk a cache entry costs.

        Two stages get declined. ``features`` when large:
        ``put_features`` writes ``np.savez_compressed`` of the array, so for an
        extractor that is itself a file read -- ``PolicyEmbeddings`` loads
        ``policy_embeddings/<layer>.npz`` -- the entry is a recompressed copy of
        the source and reading it back costs the same as reading the original.
        It buys nothing and duplicates the file.

        And ``reduced`` when the reduction is a pass-through; see
        :func:`_passthrough_reduction`.

        Measured: ``cache/features`` had grown to 156 GB, against a 4 GB total
        for every rgrl tap entry combined, because legos' DINOv3 block is 26.6
        GB and each of its two reading arms cached its own copy. Adding
        ``state_dinov3`` across eight tasks would have added ~130 GB more, and
        the disk went from 161 GB free to 31 GB in one sweep round.

        Vision-backbone features are the case this protects: those run a GPU
        model, so recomputing is expensive -- but pooled they are also well
        under the threshold, so they are still cached.
        """
        if stage == "reduced" and _passthrough_reduction():
            log.info(
                "reduced not cached: identity reduction returns the windows array, "
                "so the entry would duplicate windows (%s)",
                keys.get("windows", "?"),
            )
            return False
        if stage != "features":
            return True
        arr = getattr(out, "features", None)
        nbytes = int(getattr(arr, "nbytes", 0))
        if nbytes <= FEATURES_CACHE_MAX_BYTES:
            return True
        log.info(
            "features not cached (%.1f GB > %.1f GB): loading the source array "
            "costs the same as reading an entry",
            nbytes / 1e9,
            FEATURES_CACHE_MAX_BYTES / 1e9,
        )
        return False

    def cached(stage: str, load, store, compute):
        """Load a stage output from the cache or compute-and-store it."""
        if cache is not None:
            hit = load(keys[stage])
            cache_hits[stage] = hit is not None
            if hit is not None:
                timings[stage] = 0.0
                log.info("%s reused from cache (%s)", stage, keys[stage])
                return hit
        out = timed(stage, compute)
        if cache is not None and _worth_caching(stage, out):
            store(keys[stage], out)
        return out

    ep_norm_setting = str(cfg.get("episode_norm", "none") or "none")
    if ep_norm_setting != "none":
        log.info("episode_norm=%s applied to features before windowing", ep_norm_setting)
    features = instantiate(cfg.features)
    windowing = instantiate(cfg.windowing)
    graph_builder = instantiate(cfg.graph)

    result.features = cached(
        "features",
        lambda k: cache.get_features(k),
        lambda k, v: cache.put_features(k, v, meta=cfg_dict(cfg.features)),
        lambda: apply_episode_norm(features.extract(rollouts), ep_norm_setting),
    )
    result.windows = cached(
        "windows",
        lambda k: cache.get_windows(k),
        lambda k, v: cache.put_windows(k, v, meta=cfg_dict(cfg.windowing)),
        lambda: windowing.transform(result.features),
    )
    # Constructed outside the lambda so its own report of the fit survives the
    # call — a reducer that trains is the only thing that knows whether it
    # converged, and nothing downstream can tell a collapsed code from a good
    # one. Recorded into metrics below.
    reducer = instantiate(cfg.reduction)

    def _reduce() -> np.ndarray:
        # No-op for every reducer whose objective is features-only; the SAE's
        # temporal term needs to know which windows are adjacent in time.
        if split is None:
            reducer.set_window_context(result.windows)
            return reducer.fit_transform(result.windows.features)
        # Fit on the fit episodes, transform all of them: downstream artifacts
        # stay whole (the app and the graph read every window) while nothing
        # held out contributed to the embedding.
        m = window_mask(result.windows, split.fit)
        reducer.set_window_context(subset_windows(result.windows, m))
        reducer.fit(result.windows.features[m])
        return reducer.transform(result.windows.features)

    result.reduced = cached(
        "reduced",
        lambda k: cache.get_array("reduced", k),
        lambda k, v: cache.put_array("reduced", k, v, meta=cfg_dict(cfg.reduction)),
        _reduce,
    )
    # Constructed outside the lambda for the same two reasons as the reducer:
    # a sequence clusterer needs the window context, and its own report of the
    # fit (likelihood, dwell) is recorded below.
    clusterer = instantiate(cfg.clustering)

    def _cluster() -> np.ndarray:
        if split is None:
            clusterer.set_window_context(result.windows)
            return clusterer.fit_predict(result.reduced)
        m = window_mask(result.windows, split.fit)
        clusterer.set_window_context(subset_windows(result.windows, m))
        clusterer.fit(result.reduced[m])
        return clusterer.predict(result.reduced)

    result.labels = cached(
        "labels",
        lambda k: cache.get_array("labels", k),
        lambda k, v: cache.put_array("labels", k, v, meta=cfg_dict(cfg.clustering)),
        _cluster,
    )
    # Windows in the space the clustering was fitted on, not the raw features:
    # graph construction reads only window structure, but the stage's optional
    # smoothing scores distance to a cluster centroid, and UMAP does not preserve
    # distances. See `in_clustering_space`.
    graph_windows = in_clustering_space(result.windows, result.reduced)

    def _graph():
        if split is None:
            return graph_builder.build(result.labels, graph_windows)
        # The graph -- including sticky smoothing's centroids and the
        # degree-one merge map -- is learned from the fit episodes only, then
        # `relabel` reapplies that mapping to every window.
        m = window_mask(result.windows, split.fit)
        return graph_builder.build(result.labels[m], subset_windows(graph_windows, m))

    result.graph = timed("graph", _graph)
    # Graph-stage smoothing/pruning may relabel windows; prefer those labels.
    if split is None:
        final_labels = getattr(graph_builder, "last_labels_", None)
        if final_labels is not None:
            result.labels = np.asarray(final_labels)
    elif hasattr(graph_builder, "relabel"):
        result.labels = np.asarray(
            timed("graph:relabel", lambda: graph_builder.relabel(result.labels, graph_windows))
        )

    if result.reduced.shape[1] == 2:
        result.coords_2d = result.reduced
    else:
        result.coords_2d = cached(
            "coords_2d",
            lambda k: cache.get_array("coords_2d", k),
            lambda k, v: cache.put_array("coords_2d", k, v, meta=cfg_dict(cfg.viz_reduction)),
            lambda: instantiate(cfg.viz_reduction).fit_transform(result.windows.features),
        )

    # Optional sub-clustering step (cfg.subclustering, e.g.
    # pipeline/subclustering@subclustering=terminal). After the graph stage,
    # whose smoothing/pruning may have relabeled windows above, so its leaf
    # paths match the final trajectory tree; before evaluation, so an
    # evaluator can score the overlay without changing graph semantics.
    if cfg.get("subclustering"):
        subclusterer = instantiate(cfg.subclustering)
        # Chains off the final labels (the graph stage may relabel) plus the
        # graph config and the stage's full fingerprint — which folds in the
        # algo/reduction yamls split() reads by name, so editing one of those
        # files misses the cache instead of serving a stale split.
        if cache is not None:
            keys["subclusters"] = stage_key(
                keys["labels"], cfg_dict(cfg.graph), subclusterer.cache_fingerprint()
            )
        result.subclusters = cached(
            "subclusters",
            lambda k: _load_subclusters(cache, k),
            lambda k, v: cache.put_json(
                "subclusters", k, v.to_dict(), meta=cfg_dict(cfg.subclustering)
            ),
            lambda: subclusterer.split(result),
        )
        result.metrics[subclusterer.name] = subclusterer.summarize(result.subclusters)

    # Record the data provenance; the runs index and viz apps read this back.
    # RunArtifacts.save_metrics makes this relative to the data root (same
    # convention as configs/data/*.yaml) so runs stay portable across machines.
    # Only present when the fit actually ran: a cache hit returns the array
    # without constructing a fit, and inventing numbers for it would be worse
    # than the key being absent.
    reduction_fit = getattr(reducer, "fit_info_", None)
    if reduction_fit:
        result.metrics["reduction_fit"] = dict(reduction_fit)
    clustering_fit = getattr(clusterer, "fit_info_", None)
    if clustering_fit:
        result.metrics["clustering_fit"] = dict(clustering_fit)

    result.metrics["rollout_dir"] = str(rollouts.path)
    if split is not None:
        result.metrics["split"] = {
            "label": split.label,
            "n_fit": len(split.fit),
            "n_score": len(split.score),
            "score": split.score,
            # `fit` as well as `score`: anything comparing this run against a
            # baseline has to fit that baseline's own parameters on the same
            # episodes, and reconstructing them from the splits file means the
            # comparison could silently drift from what the run actually used.
            "fit": split.fit,
        }

    for evaluator in instantiate(cfg.evaluation.evaluators):
        result.metrics[evaluator.name] = timed(
            f"eval:{evaluator.name}", lambda e=evaluator: e.evaluate(result)
        )
    values = result.metrics.get("values", {}).get("node_values")
    if values is not None:
        result.node_values = {int(k): float(v) for k, v in values.items()}

    # Optional node-labeling step (cfg.labeling, e.g. pipeline/labeling@labeling=vlm).
    # Runs after evaluation so labelers can use node values as context.
    if cfg.get("labeling"):
        labeler = instantiate(cfg.labeling)
        result.metrics[labeler.name] = timed(
            f"label:{labeler.name}",
            lambda: labeler.label(result, run_dir=run_dir, rollout_dir=rollouts.path),
        )

    result.metrics["timings_s"] = timings
    if cache is not None:
        result.metrics["cache_hits"] = cache_hits
    if cfg.get("objective") is not None:
        # The objective is a _target_-instantiable Objective (linear combo of
        # metrics); its score() returns {score, feasible, ...}.
        objective = instantiate(cfg.objective)
        result.metrics["objective"] = objective.score(result.metrics)
    # A pass-through reduction's output is the windowed array it was handed, so
    # writing it would put a second full-width copy next to `windows` in the
    # shared store -- the same duplication `_worth_caching` declines for the
    # stage cache. Declined in both places or in neither, otherwise the saving
    # is half a saving.
    if _passthrough_reduction():
        artifacts.skip_shared.add("reduced")
    artifacts.save_result(result, save_features=cfg.save_features, data_path=cfg.get("data_path"))
    log.info("artifacts written to %s", artifacts.run_dir)
    return result


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> float:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    rollouts = RolloutDirectory(Path(cfg.rollout_dir))
    log.info("pipeline on %d episodes from %s", len(rollouts.episodes()), rollouts.path)
    result = run_pipeline(cfg, rollouts, run_dir)
    # Return a scalar so hydra sweepers (e.g. optuna) can optimize over it.
    # With an `objective` config this is the linear combination of metrics;
    # otherwise it falls back to the silhouette score.
    obj = result.metrics.get("objective", {}).get("score")
    if obj is not None:
        return float(obj)
    return float(result.metrics.get("cluster_quality", {}).get("silhouette", float("nan")))


if __name__ == "__main__":
    main()
