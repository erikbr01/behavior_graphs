"""Canonical on-disk trajectory layout shared by all environments.

::

    <dir>/
        episodes/metadata.yaml          # per-episode length + success + provenance
        policy_embeddings/<layer>.npz   # (total_steps, d) float32
        observations/<key>.npz          # optional, (total_steps, d_obs) float32
        actions.npz                     # optional, (total_steps, d_action) float32
        media/ep{idx:04d}_{success|failure}.mp4            # optional, single view
        media/<camera>/ep{idx:04d}_{success|failure}.mp4   # optional, one dir per view

A platform with several cameras (DROID records a wrist and one or more
exterior views per episode) writes one subdirectory of ``media/`` per view,
named after the camera; the flat form is the same thing for a single unnamed
view and reads back under the name :data:`DEFAULT_CAMERA`. Both forms can
coexist, so a dataset assembled before per-camera media keeps working and can
gain views without rewriting what is already there. See
:meth:`TrajectoryDirectory.camera_names`.

``observations/<key>.npz`` holds raw per-timestep observation arrays (e.g.
``state.npz`` with the flattened low-dim state); ``actions.npz`` holds the
action EXECUTED at each step. Both are optional and aligned 1:1 with the
episode sample ranges, like embeddings.

``metadata.yaml``::

    source:                       # optional provenance block (free-form)
      kind: demos
      dataset: /path/to/low_dim_abs.hdf5
    episodes:
      - {index: 0, length: 124, success: true}
      - {index: 1, length: 98, success: false, demo_key: demo_7}

Unknown per-episode keys (e.g. ``demo_key`` for featurized demonstrations)
are preserved in ``EpisodeInfo.extra``.

:class:`TrajectoryDirectory` reads the layout for any trajectory source
(demonstrations featurized by ``extract_trajectory_features`` use it directly);
:class:`RolloutDirectory` is its policy-rollout subclass, returned by every
adapter's ``collect_rollouts``/``load_rollouts``. Env adapters write this
layout (directly or by translating their submodule's native output); the
pipeline only ever reads it. :class:`EpisodeSubset` is the one TrajectorySet
implementation that isn't disk-backed -- an in-memory view over a subset of
another TrajectorySet's episodes.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from behavior_graphs.core.env import RolloutSet, TrajectorySet
from behavior_graphs.core.types import EpisodeInfo, episodes_from_lengths
from behavior_graphs.core.yaml_io import load_yaml

EPISODES_DIR = "episodes"
EMBEDDINGS_DIR = "policy_embeddings"
OBSERVATIONS_DIR = "observations"
MEDIA_DIR = "media"
METADATA_FILE = "metadata.yaml"
ACTIONS_FILE = "actions.npz"

#: Name the single-view layout (``media/ep0000_success.mp4``) reads back under,
#: and the name that writes back to it. Reserved: a ``media/default/``
#: subdirectory is ignored when there are flat clips to name it after.
DEFAULT_CAMERA = "default"

#: An episode clip, in either media layout. The index is zero-padded to 4 by
#: the writers but not capped there, so this accepts any width.
_CLIP_RE = re.compile(r"^ep(\d+)_.*\.mp4$")

#: Per-episode metadata keys consumed by the reader itself; everything else
#: is provenance and lands in ``EpisodeInfo.extra``.
_RESERVED_EPISODE_KEYS = ("index", "length", "num_samples", "success")


class TrajectoryDirectory(TrajectorySet):
    """Read access to the canonical trajectory layout."""

    def __init__(self, path: Path, *, primary_camera: str | None = None) -> None:
        """*primary_camera* names the view that comes first, if the layout has it.

        Which view is "the" view is otherwise an accident of the alphabet — the
        subdirectories are scanned in sorted order — and a lot follows from it:
        :meth:`video_path` with no camera, and so every single-view consumer.
        Naming it is the display layer's business (``primary_camera:`` in
        ``configs/data/<group>.yaml``, read by
        :func:`behavior_graphs.viz.streamlit.data.primary_camera`), which is why
        this is an argument rather than something read here. A name the layout
        does not have changes nothing, so a group-wide preference is safe for a
        dataset that happens to lack that camera.
        """
        self._path = Path(path)
        self._primary_camera = primary_camera or None
        self._media: dict[str, dict[int, Path]] | None = None
        metadata_path = self._path / EPISODES_DIR / METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"no trajectory metadata at {metadata_path}")
        raw = load_yaml(metadata_path.read_text())
        self._source: dict[str, Any] = raw.get("source") or {}
        lengths, successes, extras = [], [], []
        for ep in raw["episodes"]:
            lengths.append(int(ep.get("length", ep.get("num_samples"))))
            successes.append(ep.get("success"))
            extras.append({k: v for k, v in ep.items() if k not in _RESERVED_EPISODE_KEYS})
        self._episodes = episodes_from_lengths(lengths, successes)
        for ep_info, extra in zip(self._episodes, extras):
            ep_info.extra.update(extra)

    @property
    def source(self) -> dict[str, Any]:
        """Provenance recorded by the writer (e.g. demo dataset path); may be empty."""
        return self._source

    @property
    def path(self) -> Path:
        return self._path

    def episodes(self) -> list[EpisodeInfo]:
        return self._episodes

    def embedding_layers(self) -> list[str]:
        emb_dir = self._path / EMBEDDINGS_DIR
        if not emb_dir.is_dir():
            return []
        return sorted(p.stem for p in emb_dir.glob("*.npz"))

    def embeddings(self, layer: str) -> np.ndarray:
        emb_path = self._path / EMBEDDINGS_DIR / f"{layer}.npz"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"layer '{layer}' not found; available: {self.embedding_layers()}"
            )
        return self._load_aligned(emb_path, "embeddings", f"embeddings '{layer}'")

    def observation_keys(self) -> list[str]:
        obs_dir = self._path / OBSERVATIONS_DIR
        if not obs_dir.is_dir():
            return []
        return sorted(p.stem for p in obs_dir.glob("*.npz"))

    def observations(self, key: str) -> np.ndarray:
        obs_path = self._path / OBSERVATIONS_DIR / f"{key}.npz"
        if not obs_path.exists():
            raise FileNotFoundError(
                f"observation key '{key}' not found; available: {self.observation_keys()}"
            )
        return self._load_aligned(obs_path, "observations", f"observations '{key}'")

    def actions(self) -> np.ndarray | None:
        actions_path = self._path / ACTIONS_FILE
        if not actions_path.exists():
            return None
        return self._load_aligned(actions_path, "actions", "actions")

    def _load_aligned(self, path: Path, npz_key: str, what: str) -> np.ndarray:
        """Load a (total_steps, d) float32 array and validate its alignment."""
        with np.load(path) as data:
            key = npz_key if npz_key in data else data.files[0]
            arr = np.asarray(data[key], dtype=np.float32)
        total = sum(ep.num_samples for ep in self._episodes)
        if arr.shape[0] != total:
            raise ValueError(f"{what} have {arr.shape[0]} rows but episodes sum to {total} steps")
        return arr

    def camera_names(self) -> list[str]:
        """The camera views that have at least one episode clip.

        Empty when there is no media at all, ``[DEFAULT_CAMERA]`` for the
        single-view layout, and one entry per ``media/<camera>/`` subdirectory
        otherwise. ``camera_names()[0]`` is the view :meth:`video_path` falls
        back to: the configured ``primary_camera`` when this reader was given one
        and the layout has it, else the flat view, else the first subdirectory
        alphabetically.
        """
        return list(self._media_index())

    def video_path(self, episode_idx: int, camera: str | None = None) -> Path | None:
        """One episode's clip, from *camera* or from the first view that has it.

        ``camera=None`` keeps the single-view contract every caller that
        predates multiple views relies on: whichever view comes first in
        :meth:`camera_names`, which for a flat ``media/`` is its only clip.
        """
        index = self._media_index()
        if camera is not None:
            return index.get(camera, {}).get(int(episode_idx))
        for by_episode in index.values():
            clip = by_episode.get(int(episode_idx))
            if clip is not None:
                return clip
        return None

    def video_paths(
        self, episode_idx: int, cameras: Sequence[str] | None = None
    ) -> dict[str, Path]:
        """``{camera: clip}`` for one episode, in :meth:`camera_names` order.

        Views this particular episode has no clip for are absent rather than
        mapped to None — a collection can lose one camera partway through.

        *cameras* restricts *and reorders* the result to the views named, mirroring
        :meth:`video_path`'s ``camera=`` argument, and drops the ones this episode
        lacks by the same rule. Here rather than at each caller because it is the
        layout's own rule: the contact sheets, the app's player panels and the
        thumbnail strips were each filtering their own way.
        """
        out: dict[str, Path] = {}
        for camera, by_episode in self._media_index().items():
            clip = by_episode.get(int(episode_idx))
            if clip is not None:
                out[camera] = clip
        if cameras is None:
            return out
        return {camera: out[camera] for camera in cameras if camera in out}

    def primary_view(self, episode_idx: int) -> tuple[str, Path] | None:
        """One episode's clip *and the view it came from*, or None.

        :meth:`video_path` answers the same question but drops the name, leaving
        callers that need it — to look up that camera's declared orientation, say
        — to guess at ``camera_names()[0]``. That guess is wrong for exactly the
        episodes this method exists for: the view it falls back to is the first
        that has *this* episode, which is not the first view overall when a
        camera missed one.
        """
        return next(iter(self.video_paths(episode_idx).items()), None)

    def _media_index(self) -> dict[str, dict[int, Path]]:
        """``{camera: {episode_idx: clip}}``, scanned once per reader.

        One ``scandir`` per view rather than a glob per episode: the apps build
        their episode index by asking for every episode's clip in turn, which
        walked the whole media directory each time — 1000 walks of a 1000-clip
        directory for one real-robot dataset.
        """
        if self._media is None:
            self._media = _scan_media(self._path / MEDIA_DIR, primary=self._primary_camera)
        return self._media


class RolloutDirectory(TrajectoryDirectory, RolloutSet):
    """Read access to canonical-layout policy rollouts."""


def _renumbered_episodes(
    entries: Sequence[tuple[EpisodeInfo, dict[str, Any]]],
) -> list[EpisodeInfo]:
    """Build a contiguous, renumbered ``EpisodeInfo`` list from source episodes.

    Shared by :class:`EpisodeSubset` and :class:`PooledTrajectorySet`; each
    ``entries`` pair is an original ``EpisodeInfo`` plus a provenance dict
    to merge into ``extra`` (``orig_index``, and for pooled sources
    ``source_index``).
    """
    episodes: list[EpisodeInfo] = []
    start = 0
    for new_idx, (src_ep, provenance) in enumerate(entries):
        extra = dict(src_ep.extra)
        extra.update(provenance)
        episodes.append(
            EpisodeInfo(
                index=new_idx,
                num_samples=src_ep.num_samples,
                sample_start_idx=start,
                sample_end_idx=start + src_ep.num_samples,
                success=src_ep.success,
                extra=extra,
            )
        )
        start += src_ep.num_samples
    return episodes


class EpisodeSubset(TrajectorySet):
    """Read-only, in-memory view over a subset of another TrajectorySet's episodes.

    Episodes are renumbered ``0..n-1`` in the given order (original index in
    ``extra["orig_index"]``); embeddings/observations/actions are sliced and
    concatenated to match. Used for train/held-out splits (see
    :func:`behavior_graphs.outcome_classifier.split.train_test_split_episodes`)
    without writing a new directory to disk.
    """

    def __init__(self, base: TrajectorySet, episode_indices: list[int]) -> None:
        self._base = base
        self._orig_indices = list(episode_indices)
        base_episodes = {ep.index: ep for ep in base.episodes()}
        entries = [
            (base_episodes[orig_idx], {"orig_index": orig_idx}) for orig_idx in self._orig_indices
        ]
        self._episodes = _renumbered_episodes(entries)
        self._base_episodes = base_episodes

    @property
    def path(self) -> Path:
        return self._base.path

    def episodes(self) -> list[EpisodeInfo]:
        return self._episodes

    def embedding_layers(self) -> list[str]:
        return self._base.embedding_layers()

    def embeddings(self, layer: str) -> np.ndarray:
        return self._gather(self._base.embeddings(layer))

    def observation_keys(self) -> list[str]:
        return self._base.observation_keys()

    def observations(self, key: str) -> np.ndarray:
        return self._gather(self._base.observations(key))

    def actions(self) -> np.ndarray | None:
        actions = self._base.actions()
        return None if actions is None else self._gather(actions)

    def video_path(self, episode_idx: int) -> Path | None:
        orig = self._episodes[episode_idx].extra["orig_index"]
        return self._base.video_path(orig)

    def _gather(self, array: np.ndarray) -> np.ndarray:
        rows = [
            array[
                self._base_episodes[orig].sample_start_idx : self._base_episodes[
                    orig
                ].sample_end_idx
            ]
            for orig in self._orig_indices
        ]
        return np.concatenate(rows, axis=0)


class PooledTrajectorySet(TrajectorySet):
    """Read-only, in-memory view pooling episodes from several TrajectorySets into one.

    Unlike :class:`EpisodeSubset` (a subset view over ONE base set), this
    concatenates episodes from several different base sets -- e.g. several
    tasks' rollout directories -- for cross-task generalization eval (fit on
    pooled "seen"-task episodes, predict on a separate "unseen" task; see
    ``docs/outcome_classifier_experiments.md``'s "Cross-benchmark validation").

    All sources must share the same embedding layers and observation keys.
    Episodes are renumbered ``0..n-1`` in argument order (``extra["source_index"]``/
    ``["orig_index"]`` preserve provenance). ``observations``/``actions``/
    ``video_path`` are forwarded to (and, for the first two, concatenated
    across) the sources, mirroring :class:`EpisodeSubset`.
    """

    def __init__(self, sources: Sequence[TrajectorySet]) -> None:
        if not sources:
            raise ValueError("no source TrajectorySets given")
        layers = sources[0].embedding_layers()
        for src in sources[1:]:
            if src.embedding_layers() != layers:
                raise ValueError(
                    f"embedding layers differ across sources: {layers} vs {src.embedding_layers()}"
                )
        obs_keys = sources[0].observation_keys()
        for src in sources[1:]:
            if src.observation_keys() != obs_keys:
                raise ValueError(
                    f"observation keys differ across sources: "
                    f"{obs_keys} vs {src.observation_keys()}"
                )
        self._sources = list(sources)
        self._layers = layers
        self._observation_keys = obs_keys
        entries = [
            (ep, {"source_index": src_idx, "orig_index": ep.index})
            for src_idx, src in enumerate(self._sources)
            for ep in src.episodes()
        ]
        self._episodes = _renumbered_episodes(entries)

    @property
    def path(self) -> Path:
        """Arbitrary (the first source's path) -- provenance lives in each episode's extra."""
        return self._sources[0].path

    def episodes(self) -> list[EpisodeInfo]:
        return self._episodes

    def embedding_layers(self) -> list[str]:
        return self._layers

    def embeddings(self, layer: str) -> np.ndarray:
        return np.concatenate([src.embeddings(layer) for src in self._sources], axis=0)

    def observation_keys(self) -> list[str]:
        return self._observation_keys

    def observations(self, key: str) -> np.ndarray:
        if key not in self._observation_keys:
            raise FileNotFoundError(
                f"observation key '{key}' not found; available: {self._observation_keys}"
            )
        return np.concatenate([src.observations(key) for src in self._sources], axis=0)

    def actions(self) -> np.ndarray | None:
        parts = [src.actions() for src in self._sources]
        if all(p is None for p in parts):
            return None
        if any(p is None for p in parts):
            raise ValueError(
                "some sources have actions() and others do not -- cannot pool "
                "inconsistent action availability"
            )
        return np.concatenate(parts, axis=0)

    def video_path(self, episode_idx: int) -> Path | None:
        extra = self._episodes[episode_idx].extra
        return self._sources[extra["source_index"]].video_path(extra["orig_index"])


def _clips_in(directory: Path) -> dict[int, Path]:
    """``{episode_idx: clip}`` for one view's directory.

    Sorted so that an episode with two clips (a stale ``_unknown`` left beside
    a relabeled ``_success``) resolves to the same one every time, matching the
    ``sorted(glob(...))[0]`` this replaced.

    A clip has to *resolve*, not merely be named like one. Views can be
    symlink farms into a source corpus -- the ABC study's datasets link every
    ``media/<view>/epNNNN_*.mp4`` into the ABC-130k tree -- and its ``left``
    and ``right`` links dangle, because those two sidecars are transient and
    get reaped to reclaim disk. Matching on the name alone counted 300 dead
    links per view as clips, so ``_scan_media`` kept the view and the app
    offered two camera buttons that could only render an empty panel, which is
    the exact thing its docstring promises not to do. ``is_file`` follows the
    link, and also rejects a directory that happens to match the pattern.

    That is one stat per matching entry, on a scan already made once per
    reader and cached above it.
    """
    clips: dict[int, Path] = {}
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return clips
    for entry in entries:
        match = _CLIP_RE.match(entry.name)
        if match and entry.is_file():
            clips.setdefault(int(match.group(1)), entry)
    return clips


def _view_subdirs(media: Path) -> list[Path]:
    """The per-camera subdirectories of a ``media/`` directory, sorted.

    One definition of "what counts as a view directory", so a reader and a
    relabel can't disagree about which ones they cover.
    """
    try:
        return sorted(p for p in media.iterdir() if p.is_dir())
    except OSError:
        return []


def _scan_media(media: Path, *, primary: str | None = None) -> dict[str, dict[int, Path]]:
    """``{camera: {episode_idx: clip}}`` for a ``media/`` directory.

    Views with no clips at all are dropped, so ``camera_names()`` never offers
    a picker entry that would render nothing. *primary* is moved to the front
    when it is one of the views that survived that — see
    :meth:`TrajectoryDirectory.__init__`; ordering is otherwise the flat view
    first, then the subdirectories alphabetically.
    """
    if not media.is_dir():
        return {}
    out: dict[str, dict[int, Path]] = {}
    flat = _clips_in(media)
    if flat:
        out[DEFAULT_CAMERA] = flat
    for subdir in _view_subdirs(media):
        if subdir.name == DEFAULT_CAMERA and flat:
            continue  # the name is taken by the flat clips it would shadow
        clips = _clips_in(subdir)
        if clips:
            out[subdir.name] = clips
    if primary in out and next(iter(out), None) != primary:
        # Rebuilt rather than reordered in place: dict order *is* the view order
        # every consumer reads, so it has to be the real thing, not a view of it.
        out = {primary: out[primary], **{c: v for c, v in out.items() if c != primary}}
    return out


def _media_dir(root: Path, camera: str) -> Path:
    """Where a view's clips go: ``media/`` itself for the flat/default one."""
    return root / MEDIA_DIR if camera == DEFAULT_CAMERA else root / MEDIA_DIR / camera


def write_trajectory_dir(
    path: Path,
    lengths: Sequence[int],
    successes: Sequence[bool | None],
    embeddings: dict[str, np.ndarray],
    video_paths: dict[int, Path | dict[str, Path]] | None = None,
    observations: dict[str, np.ndarray] | None = None,
    actions: np.ndarray | None = None,
    source: dict[str, Any] | None = None,
    episode_extras: Sequence[dict[str, Any]] | None = None,
) -> TrajectoryDirectory:
    """Write the canonical layout and return a reader over it.

    ``source`` and ``episode_extras`` record provenance (e.g. the demo
    dataset path and per-episode demo keys for featurized demonstrations).

    ``video_paths`` maps an episode to its clip, or — for a platform that
    records several camera views of the same episode — to ``{camera: clip}``.
    :data:`DEFAULT_CAMERA` writes back to the flat ``media/`` form, so a
    round-trip through :meth:`TrajectoryDirectory.video_paths` preserves the
    layout it read.
    """
    path = Path(path)
    if len(lengths) != len(successes):
        raise ValueError("lengths and successes must have equal length")
    if episode_extras is not None and len(episode_extras) != len(lengths):
        raise ValueError("episode_extras and lengths must have equal length")
    total = int(sum(lengths))
    for layer, emb in embeddings.items():
        if emb.shape[0] != total:
            raise ValueError(f"embeddings '{layer}' have {emb.shape[0]} rows, expected {total}")
    for key, obs in (observations or {}).items():
        if obs.shape[0] != total:
            raise ValueError(f"observations '{key}' have {obs.shape[0]} rows, expected {total}")
    if actions is not None and actions.shape[0] != total:
        raise ValueError(f"actions have {actions.shape[0]} rows, expected {total}")

    (path / EPISODES_DIR).mkdir(parents=True, exist_ok=True)
    episodes = []
    for i, (n, s) in enumerate(zip(lengths, successes)):
        entry: dict[str, Any] = {"index": i, "length": int(n), "success": s}
        if episode_extras is not None:
            entry.update(episode_extras[i])
        episodes.append(entry)
    metadata: dict[str, Any] = {"episodes": episodes}
    if source:
        metadata = {"source": dict(source), "episodes": episodes}
    (path / EPISODES_DIR / METADATA_FILE).write_text(yaml.safe_dump(metadata, sort_keys=False))

    (path / EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    for layer, emb in embeddings.items():
        np.savez_compressed(
            path / EMBEDDINGS_DIR / f"{layer}.npz",
            embeddings=np.asarray(emb, dtype=np.float32),
        )

    if observations:
        (path / OBSERVATIONS_DIR).mkdir(parents=True, exist_ok=True)
        for key, obs in observations.items():
            np.savez_compressed(
                path / OBSERVATIONS_DIR / f"{key}.npz",
                observations=np.asarray(obs, dtype=np.float32),
            )

    if actions is not None:
        np.savez_compressed(path / ACTIONS_FILE, actions=np.asarray(actions, dtype=np.float32))

    if video_paths:
        import shutil

        for ep_idx, spec in video_paths.items():
            clips = spec if isinstance(spec, dict) else {DEFAULT_CAMERA: spec}
            name = f"ep{ep_idx:04d}_{_success_tag(successes[ep_idx])}.mp4"
            for camera, src in clips.items():
                dst_dir = _media_dir(path, camera)
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst_dir / name)

    return TrajectoryDirectory(path)


def add_embedding_layer(
    path: Path, layer: str, embeddings: np.ndarray, overwrite: bool = False
) -> Path:
    """Add one ``policy_embeddings/<layer>.npz`` to an existing trajectory dir.

    Here rather than at the caller because every fact involved belongs to this
    module: the directory name, the ``embeddings`` array key, the float32
    dtype, the row-count alignment :meth:`TrajectoryDirectory.embeddings`
    validates on the way back, and the glob that decides what counts as a layer.

    The write goes through a temp file *outside* ``policy_embeddings/`` and is
    renamed into place: :meth:`TrajectoryDirectory.embedding_layers` globs that
    directory, so a half-written file beside the real ones would register
    permanently as a layer of its own (offered by the app's layer picker and by
    the sweep runner's layer auto-detection). The temp name keeps the ``.npz``
    suffix because :func:`numpy.savez_compressed` appends one to any path that
    lacks it, and stays inside *path* so the rename cannot cross a filesystem.
    """
    path = Path(path)
    reader = TrajectoryDirectory(path)
    total = sum(ep.num_samples for ep in reader.episodes())
    if embeddings.shape[0] != total:
        raise ValueError(
            f"embeddings '{layer}' have {embeddings.shape[0]} rows but the episodes of "
            f"{path} sum to {total} steps"
        )
    target = path / EMBEDDINGS_DIR / f"{layer}.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; pass overwrite=True to replace it, or a "
            "different layer name to keep both"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    replacing = target.exists()
    tmp = path / f".{layer}.tmp.npz"
    try:
        np.savez_compressed(tmp, embeddings=np.asarray(embeddings, dtype=np.float32))
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    if replacing:
        # Bump metadata.yaml's mtime, because that file *is* the data
        # fingerprint the stage cache keys on
        # (:func:`behavior_graphs.pipeline.cache.data_fingerprint` hashes its size
        # and mtime, not the arrays). Every other writer of this layout rewrites
        # metadata.yaml as a matter of course, so replacing a layer's contents
        # in place is the one way to change what a directory means without
        # changing its fingerprint -- and a pipeline run would then serve
        # features computed from the *previous* contents of this file. Only on
        # replacement: a brand-new layer name is a new features config, hence a
        # new cache key already, and touching the file would needlessly
        # invalidate every other layer's cached work for this dataset.
        meta = path / EPISODES_DIR / METADATA_FILE
        before_ns = meta.stat().st_mtime_ns
        meta.touch()
        if meta.stat().st_mtime_ns <= before_ns:
            # `touch()` takes its timestamp from the kernel's *coarse* clock,
            # which advances once per timer tick -- measured at 1 ms granularity
            # here (2000 touches produced 9 distinct mtimes). A replacement that
            # lands inside one tick leaves mtime_ns identical, so the fingerprint
            # would not move and a run would serve the stale features this whole
            # branch exists to prevent. Step it strictly past the old value.
            os.utime(meta, ns=(before_ns + 1, before_ns + 1))
    return target


def write_rollout_dir(path: Path, *args: Any, **kwargs: Any) -> RolloutDirectory:
    """Write canonical-layout policy rollouts and return a rollout reader."""
    write_trajectory_dir(path, *args, **kwargs)
    return RolloutDirectory(path)


def _success_tag(success: bool | None) -> str:
    return "success" if success else ("failure" if success is False else "unknown")


def set_episode_success(
    path: Path,
    episode_idx: int,
    success: bool | None,
    note: str | None = None,
) -> None:
    """Update one episode's success label in place (real-robot annotation).

    Real-world rollouts have no automatic success signal — outcomes are
    labeled inline by the on-robot collector
    (``third_party/diffusion_policy/scripts/droid_collect_rollouts.py``);
    this is the primitive for fixing a label afterwards.
    Rewrites ``episodes/metadata.yaml`` atomically (preserving provenance and
    per-episode extras) and renames the episode's media file so its
    ``_success``/``_failure``/``_unknown`` tag matches the new label.
    """
    path = Path(path)
    metadata_path = path / EPISODES_DIR / METADATA_FILE
    raw = load_yaml(metadata_path.read_text())
    entry = next(
        (ep for ep in raw["episodes"] if int(ep.get("index", -1)) == episode_idx),
        None,
    )
    if entry is None:
        raise KeyError(f"no episode with index {episode_idx} in {metadata_path}")
    entry["success"] = success
    if note is not None:
        entry["label_note"] = note
    tmp = metadata_path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
    tmp.replace(metadata_path)

    media = path / MEDIA_DIR
    if media.is_dir():
        # Every view, not just the flat one: a half-renamed multi-camera
        # episode would show the old outcome on whichever view the app picked.
        for view in [media, *_view_subdirs(media)]:
            for video in sorted(view.glob(f"ep{episode_idx:04d}_*.mp4")):
                target = view / f"ep{episode_idx:04d}_{_success_tag(success)}.mp4"
                if video != target:
                    video.replace(target)


def merge_trajectory_dirs(sources: Sequence[Path], dest: Path) -> TrajectoryDirectory:
    """Concatenate several canonical-layout directories into one.

    Built for incremental real-robot collection: each on-robot session
    records its own directory; merging them yields the single rollout dir
    the pipeline consumes. Episodes are re-indexed in source order; success
    labels, per-episode extras, videos, embeddings, observations, and
    actions all carry over. All sources must share the same embedding
    layers and observation keys; actions must be present in all sources or
    in none.
    """
    readers = [TrajectoryDirectory(Path(p)) for p in sources]
    if not readers:
        raise ValueError("no source directories given")

    layers = readers[0].embedding_layers()
    obs_keys = readers[0].observation_keys()
    for r in readers[1:]:
        if r.embedding_layers() != layers:
            raise ValueError(
                f"embedding layers differ: {r.path} has {r.embedding_layers()}, expected {layers}"
            )
        if r.observation_keys() != obs_keys:
            raise ValueError(
                f"observation keys differ: {r.path} has {r.observation_keys()}, expected {obs_keys}"
            )
    has_actions = [r.actions() is not None for r in readers]
    if any(has_actions) and not all(has_actions):
        raise ValueError("actions present in some sources but not all")

    lengths: list[int] = []
    successes: list[bool | None] = []
    extras: list[dict[str, Any]] = []
    video_paths: dict[int, Path | dict[str, Path]] = {}
    new_idx = 0
    for r in readers:
        for ep in r.episodes():
            lengths.append(ep.num_samples)
            successes.append(ep.success)
            extras.append(dict(ep.extra))
            # All views, keyed by camera — sources laid out flat come back as
            # {DEFAULT_CAMERA: clip} and are written flat again.
            clips = r.video_paths(ep.index)
            if clips:
                video_paths[new_idx] = clips
            new_idx += 1

    embeddings = {layer: np.concatenate([r.embeddings(layer) for r in readers]) for layer in layers}
    observations = {
        key: np.concatenate([r.observations(key) for r in readers]) for key in obs_keys
    } or None
    actions = np.concatenate([r.actions() for r in readers]) if all(has_actions) else None
    source = {
        "kind": "merged",
        "sources": [
            {"path": str(r.path), **({"source": r.source} if r.source else {})} for r in readers
        ],
    }

    return write_trajectory_dir(
        Path(dest),
        lengths,
        successes,
        embeddings,
        video_paths=video_paths or None,
        observations=observations,
        actions=actions,
        source=source,
        episode_extras=extras,
    )
