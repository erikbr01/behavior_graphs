"""Reading and writing the per-run artifact directory.

Every pipeline run produces one directory::

    <run_dir>/
        config.yaml          # resolved hydra config
        features.npz         # optional (save_features=true): per-timestep features
        windows.npz          # window features
        windows.json         # window + episode metadata
        reduced.npz          # reduced window features (.npy in pre-2026-08 runs)
        shared.json          # only in a sweep job: which of the above are shared
        coords_2d.npy        # 2-D coordinates for visualization
        labels.npy           # cluster label per window
        graph.json           # behavior graph
        node_values.json     # MRP node values (if the values evaluator ran)
        node_labels.json     # VLM-assigned node labels (if vlm_labeler ran or labels set via MCP)
        subclusters.json     # per-node sub-cluster splits (if the subclustering stage ran)
        metrics.json         # all evaluator outputs
            scores.npz           # per-query-window metric arrays
            neighbors.npz        # top-k corpus indices + raw scores per query
            corpus.json          # corpus window/episode metadata + provenance
            cluster_values.json  # per-cluster summary of each metric

Visualization apps and downstream analysis only ever read this layout. A sweep
job's upstream stages live once under the sweep's ``_shared`` tree instead of in
the job dir, so such a run is self-contained only together with that tree; every
read here goes through :func:`~behavior_graphs.pipeline.shared_store.resolve`,
which finds either form.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.core.node_labels import load_node_labels, save_node_labels
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.subclusters import SubClusterSet
from behavior_graphs.core.types import EpisodeInfo, FeatureSet, WindowInfo, WindowSet
from behavior_graphs.pipeline.shared_store import (
    SharedStore,
    remove_local,
    resolve,
    write_pointer,
)

log = logging.getLogger(__name__)


def resolve_rollout_dir(raw: str | None, data_path: str | Path | None) -> Path | None:
    """Resolve a metrics.json ``rollout_dir`` value against the data root.

    The pipeline records rollout_dir relative to the repo-wide data root
    (``rollouts/<dataset>``, see :mod:`behavior_graphs.pipeline.run`); every
    reader must join it back. Pathlib's join handles both forms: a relative
    value lands under *data_path*, an absolute one (pre-convention runs)
    passes through unchanged.

    Absolute values used to get a second chance: re-rooted at *data_path*
    from their last ``rollouts`` component, so a run recorded on another
    machine still resolved. That guessed at intent -- and it could not tell a
    foreign path from a real local one, so on the machine that recorded the
    path it silently did nothing. :meth:`RunArtifacts.save_metrics`
    normalizes at write time instead, which no creation path can bypass.

    Returns None for an empty value; existence checks are otherwise the
    caller's concern.
    """
    if not raw:
        return None
    return Path(data_path or ".") / raw


class RunArtifacts:
    """Save/load interface over one run directory.

    ``shared_root`` opts the *writes* into the shared artifact store under that
    root (see :mod:`behavior_graphs.pipeline.shared_store`): upstream stage outputs
    every sibling job would otherwise duplicate are written once and pointed at.
    Reads never need it — :func:`resolve` finds a local copy or follows the
    pointer — so every existing reader keeps working against both layouts.
    """

    def __init__(
        self,
        run_dir: Path,
        shared_root: Path | None = None,
        stage_keys: dict[str, str] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.shared = SharedStore(shared_root) if shared_root is not None else None
        #: Stage-cache keys for the shared stages, when the caller has them.
        #: Passed down to `SharedStore.put_written` so a stage whose identical
        #: entry is already stored is not serialised and hashed again just to
        #: find that out. Optional: without them the store falls back to
        #: content-hashing every write, which is correct but is what made a
        #: large sweep I/O-bound.
        self.stage_keys = dict(stage_keys or {})
        #: Stages whose array this run should not write at all. Set by the
        #: caller for an output that is a copy of an upstream one it already
        #: wrote -- an identity reduction hands back the windowed array, so
        #: `reduced` would be a second full-width copy of `windows` (measured
        #: 347 MB against 356 MB for one whiteboard entry, 165 GB across the
        #: eight ABC tasks). The only reader of a run's `reduced` array guards
        #: on `array_path(...) is not None`, so absence is already handled.
        self.skip_shared: set[str] = set()
        # Skip the syscall entirely when the dir already exists: readers
        # (MCP/chat tools, viz apps) construct RunArtifacts against run dirs
        # that are frequently mounted read-only, where even a redundant
        # mkdir(exist_ok=True) raises OSError (EROFS) before the "already
        # exists" check can save it.
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ save
    def save_features(self, features: FeatureSet) -> None:
        np.savez_compressed(self.run_dir / "features.npz", features=features.features)

    def save_windows(self, windows: WindowSet) -> None:
        def write(target: Path) -> None:
            np.savez_compressed(target / "windows.npz", features=windows.features)
            meta = {
                "episodes": [ep.to_dict() for ep in windows.episodes],
                "windows": [w.to_dict() for w in windows.windows],
            }
            (target / "windows.json").write_text(json.dumps(meta))

        if self.shared is None:
            write(self.run_dir)
            return
        self._commit_shared("windows", write)

    #: Arrays written compressed rather than as a bare .npy. `reduced` is here
    #: because a sparse SAE code is 95-99.7% zeros and compresses 14-175x (a
    #: d=4096 code over 135k windows: 2.2 GB -> 29 MB). Dense reductions gain only
    #: ~15%, which is not why this exists but is not a reason to special-case
    #: them either. Per-job arrays (labels, coords_2d) stay uncompressed: they are
    #: a few hundred KB, and `labels.npy` in particular is what
    #: :func:`~behavior_graphs.pipeline.run_discovery.is_run_dir` looks for.
    _COMPRESSED_ARRAYS = frozenset({"reduced"})

    def save_array(self, name: str, array: np.ndarray) -> None:
        if name in self.skip_shared:
            return
        if name not in self._COMPRESSED_ARRAYS:
            np.save(self.run_dir / f"{name}.npy", array)
            return

        def write(target: Path) -> None:
            np.savez_compressed(target / f"{name}.npz", **{name: array})

        if self.shared is None:
            write(self.run_dir)
            return
        self._commit_shared(name, write)

    def _commit_shared(self, stage: str, write: Callable[[Path], None]) -> None:
        """Store ``stage`` in the shared tree, point this run at it, drop the local copy.

        The local copy has to go: a job dir being rewritten (a re-run of one that
        died mid-pipeline) already holds one, and a local copy wins in
        :func:`resolve`, so the run would keep reading the stale bytes.
        """
        assert self.shared is not None
        entry = self.shared.put_written(stage, write, stage_key=self.stage_keys.get(stage))
        write_pointer(self.run_dir, stage, entry)
        remove_local(self.run_dir, stage)

    def save_graph(self, graph: BehaviorGraph) -> None:
        graph.save(self.run_dir / "graph.json")

    def save_node_values(self, values: dict[int, float]) -> None:
        payload = {str(k): float(v) for k, v in values.items()}
        (self.run_dir / "node_values.json").write_text(json.dumps(payload, indent=2))

    def save_subclusters(self, subclusters: SubClusterSet) -> None:
        (self.run_dir / "subclusters.json").write_text(
            json.dumps(subclusters.to_dict(), default=_json_default)
        )

    def save_metrics(self, metrics: dict[str, Any], data_path: str | Path | None = None) -> None:
        """Write metrics.json, storing ``rollout_dir`` relative to *data_path*.

        Every writer goes through here, so no creation path can record an
        unportable absolute path. :func:`resolve_rollout_dir` is the inverse.
        A dir outside the root has no relative form, so it stays absolute.
        """
        raw = metrics.get("rollout_dir")
        if raw and data_path and Path(raw).is_absolute():
            try:
                rel = Path(raw).resolve().relative_to(Path(data_path).resolve())
                metrics = {**metrics, "rollout_dir": str(rel)}
            except ValueError:
                log.warning("rollout_dir %s is outside data_path %s", raw, data_path)
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=_json_default)
        )

    def save_result(
        self,
        result: PipelineResult,
        save_features: bool = False,
        data_path: str | Path | None = None,
    ) -> None:
        """Write every artifact the result carries (*data_path*: see save_metrics)."""
        if save_features and result.features is not None:
            self.save_features(result.features)
        if result.windows is not None:
            self.save_windows(result.windows)
        if result.reduced is not None:
            self.save_array("reduced", result.reduced)
        if result.coords_2d is not None:
            self.save_array("coords_2d", result.coords_2d)
        if result.labels is not None:
            self.save_array("labels", result.labels)
        if result.graph is not None:
            self.save_graph(result.graph)
        if result.node_values is not None:
            self.save_node_values(result.node_values)
        if result.subclusters is not None:
            self.save_subclusters(result.subclusters)
        if result.metrics:
            self.save_metrics(result.metrics, data_path=data_path)

    # ------------------------------------------------------------------ load
    def _dir_for(self, stage: str) -> Path:
        """Where ``stage``'s files live for this run — local, or the shared entry."""
        found = resolve(self.run_dir, stage)
        return found if found is not None else self.run_dir

    def load_windows_meta(self) -> dict:
        """Raw window + episode metadata dicts from ``windows.json``.

        Cheaper than :meth:`load_windows` when the ``windows.npz`` feature
        matrix is not needed (e.g. MCP tools serving window metadata).
        """
        return json.loads((self._dir_for("windows") / "windows.json").read_text())

    def load_windows(self) -> WindowSet:
        source = self._dir_for("windows")
        with np.load(source / "windows.npz") as data:
            features = data["features"]
        meta = json.loads((source / "windows.json").read_text())
        return WindowSet(
            features=features,
            windows=[WindowInfo.from_dict(w) for w in meta["windows"]],
            episodes=[EpisodeInfo.from_dict(ep) for ep in meta["episodes"]],
        )

    def array_path(self, name: str) -> Path | None:
        """The file the named array is in, local or shared; None if absent."""
        base = self._dir_for(name) if name in self._COMPRESSED_ARRAYS else self.run_dir
        suffix = "npz" if name in self._COMPRESSED_ARRAYS else "npy"
        path = base / f"{name}.{suffix}"
        return path if path.exists() else None

    def load_array(self, name: str) -> np.ndarray:
        path = self.array_path(name)
        if path is None:
            raise FileNotFoundError(f"{self.run_dir} has no {name}.npz or {name}.npy")
        if path.suffix == ".npz":
            with np.load(path) as data:
                return data[name]
        return np.load(path)

    def load_graph(self) -> BehaviorGraph:
        return BehaviorGraph.load(self.run_dir / "graph.json")

    def load_node_values(self) -> dict[int, float] | None:
        path = self.run_dir / "node_values.json"
        if not path.exists():
            return None
        return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}

    def load_subclusters(self) -> SubClusterSet | None:
        """Persisted sub-cluster splits, or None when the stage never ran."""
        path = self.run_dir / "subclusters.json"
        if not path.exists():
            return None
        return SubClusterSet.from_dict(json.loads(path.read_text()))

    def save_node_labels(self, labels: dict[int, dict]) -> None:
        save_node_labels(self.run_dir, labels)

    def load_node_labels(self) -> dict[int, dict]:
        return load_node_labels(self.run_dir)

    def load_metrics(self) -> dict[str, Any]:
        path = self.run_dir / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def load_result(self) -> PipelineResult:
        result = PipelineResult()
        # Probe through resolve(), not the run dir: a sweep job's windows and
        # reduced features are shared, and `reduced` is written compressed.
        if (self._dir_for("windows") / "windows.npz").exists():
            result.windows = self.load_windows()
        for name in ("reduced", "coords_2d", "labels"):
            if self.array_path(name) is not None:
                setattr(result, name, self.load_array(name))
        if (self.run_dir / "graph.json").exists():
            result.graph = self.load_graph()
        result.node_values = self.load_node_values()
        result.subclusters = self.load_subclusters()
        result.metrics = self.load_metrics()
        return result


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")
