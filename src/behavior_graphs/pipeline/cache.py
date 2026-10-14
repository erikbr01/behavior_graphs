"""Content-addressed cache for intermediate pipeline stage outputs.

Graph creation is cheap, but the upstream stages (feature extraction,
windowing, reduction, clustering) dominate the cost of a sweep. Each stage
output is stored under a key that chains the data fingerprint with the
resolved configs of every stage that produced it::

    features_key = h(data_fp, features_cfg)
    windows_key  = h(features_key, windowing_cfg)
    reduced_key  = h(windows_key, reduction_cfg)
    labels_key   = h(reduced_key, clustering_cfg)
    coords_key   = h(windows_key, viz_reduction_cfg)

so two sweep jobs that differ only in clustering config share everything up
to ``reduced``, and a job that differs only in graph construction shares
everything. Entries are written atomically (temp dir + rename), which makes
concurrent sweep workers safe: the worst case is computing the same entry
twice.

Layout: ``<root>/<stage>/<key>/`` with the same file formats as
:class:`~behavior_graphs.pipeline.artifacts.RunArtifacts`, plus a ``meta.json``
recording the config that produced the entry (for debugging; never read
back as the source of truth — the key already encodes it).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.types import (
    EpisodeInfo,
    FeatureSet,
    WindowInfo,
    WindowSet,
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing config fragments."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stage_key(*parts: Any) -> str:
    """Chainable content hash over config fragments / upstream keys."""
    h = hashlib.sha256()
    for part in parts:
        h.update(canonical_json(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def data_fingerprint(trajectory_dir: Path) -> dict:
    """Cheap identity of a trajectory directory.

    Resolved path plus size/mtime of the episode metadata file: any
    re-collection rewrites ``episodes/metadata.yaml``, so this invalidates
    without hashing the (large) arrays.
    """
    p = Path(trajectory_dir).resolve()
    meta = p / "episodes" / "metadata.yaml"
    st = meta.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def embedding_layers_in(features_cfg: Any) -> list[str]:
    """Layers named by any ``PolicyEmbeddings`` extractor in a features config.

    Walks the config rather than reading a top-level ``layer``, because the
    extractor can be nested -- ``state_and_embeddings`` puts one inside a
    ``ConcatFeatures`` list.
    """
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            target = node.get("_target_")
            if isinstance(target, str) and target.endswith("policy_embedding.PolicyEmbeddings"):
                found.append(str(node.get("layer", "pi05")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)

    walk(features_cfg)
    return sorted(set(found))


def embedding_fingerprint(trajectory_dir: Path, features_cfg: Any) -> list[dict]:
    """Identity of every policy-embedding file a features config will read.

    :func:`data_fingerprint` covers ``episodes/metadata.yaml``, which a
    re-extraction does not touch: the extractor overwrites
    ``policy_embeddings/<layer>.npz`` in place. Without this, retraining a
    policy and re-extracting leaves the features stage key unchanged, so every
    job reuses the *previous* policy's features and reports its numbers under
    the new policy's name. Silently -- nothing errors, and the sweep just runs
    suspiciously fast.

    Observed: a legos sweep started 17:50 reused a features entry written at
    05:02, from a checkpoint two trainings earlier. Moving the sweep's job dirs
    aside does not help, because that only forces the *job* to re-run; the
    stage cache underneath it is keyed by config, not by content.

    Returns one entry per layer, sorted, so the hash is order-stable. Empty
    when the config reads no embedding file, and callers must then leave the
    key unchanged -- otherwise every state-features entry on the box would be
    invalidated for nothing.
    """
    from behavior_graphs.envs.trajectory_dir import EMBEDDINGS_DIR

    root = Path(trajectory_dir).resolve() / EMBEDDINGS_DIR
    out = []
    for layer in embedding_layers_in(features_cfg):
        path = root / f"{layer}.npz"
        if not path.exists():
            # Let the stage raise the useful FileNotFoundError, not this.
            continue
        st = path.stat()
        out.append({"layer": layer, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


class StageCache:
    """Save/load interface over the stage-output cache directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _entry(self, stage: str, key: str) -> Path:
        return self.root / stage / key

    def has(self, stage: str, key: str) -> bool:
        return self._entry(stage, key).is_dir()

    # ----------------------------------------------------------------- write
    def _commit(self, stage: str, key: str, write_fn, meta: dict | None) -> None:
        """Populate a temp dir via ``write_fn`` and rename it into place."""
        final = self._entry(stage, key)
        if final.exists():
            return
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=final.parent))
        try:
            write_fn(tmp)
            if meta is not None:
                (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        try:
            tmp.replace(final)
        except OSError:
            # Lost a race with a concurrent writer producing the same entry.
            shutil.rmtree(tmp, ignore_errors=True)

    def put_features(self, key: str, fs: FeatureSet, meta: dict | None = None) -> None:
        def write(d: Path) -> None:
            np.savez_compressed(d / "features.npz", features=fs.features)
            (d / "episodes.json").write_text(json.dumps([ep.to_dict() for ep in fs.episodes]))

        self._commit("features", key, write, meta)

    def put_windows(self, key: str, ws: WindowSet, meta: dict | None = None) -> None:
        def write(d: Path) -> None:
            np.savez_compressed(d / "windows.npz", features=ws.features)
            (d / "windows.json").write_text(
                json.dumps(
                    {
                        "episodes": [ep.to_dict() for ep in ws.episodes],
                        "windows": [w.to_dict() for w in ws.windows],
                    }
                )
            )

        self._commit("windows", key, write, meta)

    def put_array(self, stage: str, key: str, array: np.ndarray, meta: dict | None = None) -> None:
        def write(d: Path) -> None:
            np.save(d / "array.npy", array)

        self._commit(stage, key, write, meta)

    def put_json(self, stage: str, key: str, payload: Any, meta: dict | None = None) -> None:
        """Cache any JSON-serializable stage output (e.g. a SubClusterSet)."""

        def write(d: Path) -> None:
            (d / "value.json").write_text(json.dumps(payload, default=str))

        self._commit(stage, key, write, meta)

    # ------------------------------------------------------------------ read
    def get_features(self, key: str) -> FeatureSet | None:
        d = self._entry("features", key)
        if not d.is_dir():
            return None
        with np.load(d / "features.npz") as data:
            features = data["features"]
        episodes = [EpisodeInfo.from_dict(e) for e in json.loads((d / "episodes.json").read_text())]
        return FeatureSet(features=features, episodes=episodes)

    def get_windows(self, key: str) -> WindowSet | None:
        d = self._entry("windows", key)
        if not d.is_dir():
            return None
        with np.load(d / "windows.npz") as data:
            features = data["features"]
        meta = json.loads((d / "windows.json").read_text())
        return WindowSet(
            features=features,
            windows=[WindowInfo.from_dict(w) for w in meta["windows"]],
            episodes=[EpisodeInfo.from_dict(ep) for ep in meta["episodes"]],
        )

    def get_array(self, stage: str, key: str) -> np.ndarray | None:
        d = self._entry(stage, key)
        if not d.is_dir():
            return None
        return np.load(d / "array.npy")

    def get_json(self, stage: str, key: str) -> Any | None:
        d = self._entry(stage, key)
        if not d.is_dir():
            return None
        return json.loads((d / "value.json").read_text())
