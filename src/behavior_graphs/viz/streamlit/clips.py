"""Shared helpers: resolve a panel's clips, then get them ready to play.

``mp4_player`` transcodes on demand, so a panel that shows six clips would
otherwise run six ffmpeg passes one after another with no feedback. Calling
:func:`prepare_clips` first runs them concurrently behind a single spinner, and
:func:`episode_clips` is the one place that turns episode indices into files on
disk; :func:`prepare_episode_clips` is that plus the pre-warm, and hands the
resolution back so the render loop that follows does not repeat it. Note the
five panels still look their entries up themselves for success/frame_count/fps,
so the join is centralised but the lookup is not.

:func:`episode_camera_clips` is the same join for a multi-camera dataset, where
one episode resolves to one clip *per view*, and :func:`prepare_entry_clips` is
the pre-warm over all of those views for panels that already hold the entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import streamlit as st

from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA
from behavior_graphs.viz import media


def prepare_episode_clips(
    episode_indices: Sequence[int] | Iterable[int],
    mp4_index: dict,
    mp4_dir: Path | None,
) -> list[tuple[int, dict, Path]]:
    """Resolve a panel's episodes to clips, pre-warm them, and hand them back.

    Returns ``[(episode_index, index_entry, clip_path)]`` for the episodes that
    actually have a video, in the order given, where the path is the episode's
    first camera view. Returning the resolution is the point: the render loop
    that follows needs exactly this, and every panel previously re-derived it
    inline right after asking for the pre-warm.
    """
    resolved = episode_clips(episode_indices, mp4_index, mp4_dir)
    prepare_entry_clips((entry for _, entry, _ in resolved), mp4_dir)
    return resolved


def prepare_entry_clips(entries: Iterable[dict], mp4_dir: Path | None) -> None:
    """Pre-warm every camera view of *entries*, in the orientation the app asks for.

    Every view, not just the one a panel returns: the players put all of a
    dataset's views in the DOM so the sidebar picker can reveal one without a
    rerun, so all of them need a proxy by the time the panel renders.

    The one place that pairs a clip with its camera's transform. A transform is
    part of the proxy's cache key, so warming the untransformed variant leaves
    the panel transcoding the rotated one on demand — without the spinner that
    was the point — and a caller that resolved the clips itself and forgot the
    transforms would hit exactly that, silently.
    """
    transforms = dataset_transforms(mp4_dir)
    prepare_clips(
        (path, transforms.get(camera))
        for entry in entries
        for camera, path in episode_camera_clips(entry, mp4_dir)
    )


def dataset_transforms(mp4_dir: Path | None) -> dict[str, dict]:
    """Configured per-camera transforms for the dataset *mp4_dir* belongs to.

    Here rather than at each call site so that the pre-warm and the render agree
    on which proxy variant they are talking about, and so the "a dataset is keyed
    by its rollout dir's name" convention lives in one module.
    """
    from behavior_graphs.viz.streamlit.data import camera_transforms

    return camera_transforms(Path(mp4_dir).name) if mp4_dir is not None else {}


def episode_camera_clips(
    entry: dict,
    mp4_dir: Path | None,
    cameras: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    """``[(camera, clip_path)]`` for one index entry — what ``mp4_player`` plays.

    *cameras* restricts and orders the result; views this episode has no clip
    for are skipped, so a dataset that lost a camera partway through still
    renders the views it does have. An entry built before per-camera media
    existed has only ``path``, which reads back as the single default view.
    """
    if mp4_dir is None:
        return []
    paths = entry.get("paths") or {DEFAULT_CAMERA: entry["path"]}
    names = list(paths) if cameras is None else [c for c in cameras if c in paths]
    return [(camera, Path(mp4_dir) / paths[camera]) for camera in names]


def episode_clips(
    episode_indices: Sequence[int] | Iterable[int],
    mp4_index: dict,
    mp4_dir: Path | None,
) -> list[tuple[int, dict, Path]]:
    """``[(episode_index, index_entry, clip_path)]`` for episodes that have a video."""
    from behavior_graphs.viz.streamlit.data import find_mp4_episode

    if mp4_dir is None:
        return []
    out: list[tuple[int, dict, Path]] = []
    for ep in episode_indices:
        entry = find_mp4_episode(int(ep), mp4_index)
        if entry:
            out.append((int(ep), entry, Path(mp4_dir) / entry["path"]))
    return out


def clip_paths(
    episode_indices: Sequence[int] | Iterable[int],
    mp4_index: dict,
    mp4_dir: Path | None,
) -> list[Path]:
    """Clip paths for the given episodes, skipping ones with no video.

    Centralised so that a change in how a clip path is derived lands in one
    place instead of the five panels that each grew their own copy of this
    comprehension. ``mp4_dir`` is None exactly when there are no videos (see
    :func:`~behavior_graphs.viz.streamlit.data.legacy_mp4_index`), which is why
    that case returns an empty list rather than paths under a fake root.
    """
    from behavior_graphs.viz.streamlit.data import find_mp4_episode

    if mp4_dir is None:
        return []
    out: list[Path] = []
    for ep in episode_indices:
        entry = find_mp4_episode(int(ep), mp4_index)
        if entry:
            out.append(Path(mp4_dir) / entry["path"])
    return out


def prepare_clips(
    items: Iterable[Path | tuple[Path, Mapping[str, object] | None]],
    *,
    thumb_widths: Sequence[int] = (media.POSTER_WIDTH,),
) -> None:
    """Build web proxies + posters for the clips a panel is about to show.

    *items* are clip paths, or ``(path, transform)`` pairs for views whose camera
    is configured with one (see :func:`prepare_entry_clips`) — a transform is
    baked into the proxy and keys it, so it picks which variant gets built.

    *thumb_widths* is passed to both the "does this still need work" check and
    the build, so the two can't disagree about which artifacts are wanted.
    """
    wanted: dict[Path, Mapping[str, object] | None] = {}
    for item in items:
        path, xform = item if isinstance(item, tuple) else (item, None)
        # First writer wins, like dict.fromkeys did: one clip listed twice is the
        # same work, and two views of one episode are different paths.
        wanted.setdefault(Path(path), xform)
    pending = [
        (p, xf)
        for p, xf in wanted.items()
        if media.needs_prep(p, thumb_widths=thumb_widths, transform=xf)
    ]
    if not pending:
        return
    n = len(pending)
    with st.spinner(f"Preparing {n} clip{'s' if n > 1 else ''} for playback…"):
        media.prewarm(pending, thumb_widths=thumb_widths)
