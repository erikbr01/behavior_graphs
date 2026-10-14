"""The panel that offers the clip on screen as an MP4 with its behavior bar.

Sits under the video player in the trajectory tree's selection panel, and on
the Figures page beside the graphs it belongs to. What it exports is what the
player is showing: the same episode, the same bands, the same colours — the
drawing is :mod:`behavior_graphs.viz.clip_figure`.

**Only where a figure is being made.** Two pages pass ``clip_export=True``
(Graph Demo and Figures) and nothing else does. The study apps are controlled
instruments — a participant offered a download is a participant doing something
the instrument does not measure — and the reviewer page is anonymous, where an
MP4 named after a run directory is a deanonymising artifact handed out by a page
whose whole job is not to carry one.

The build is behind a button rather than run on every rerun: an ffmpeg pass over
two views of a 40 s real-robot episode is seconds of work, and a panel that did
it eagerly would do it again every time the reader clicked a node. Once built,
the bytes are cached against the exact options that produced them, so the file
offered for download is always the one the options above describe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import streamlit as st

from behavior_graphs.viz import clip_figure
from behavior_graphs.viz.clip_figure import HORIZONTAL, VERTICAL, Band
from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS

#: How the orientation choice is put to the reader.
ORIENTATION_NAMES = {
    HORIZONTAL: "Side by side",
    VERTICAL: "Stacked",
}

#: The colour the player draws a single selected behavior's span in
#: (``bar1`` in :func:`~behavior_graphs.viz.streamlit.components.mp4_player`).
#: Reused here so a one-band export matches the strip it came from.
SLICE_COLOR = "#f5a623"


def _fingerprint(payload: Mapping[str, object]) -> str:
    """A cache key for one export."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _stat_signature(paths: Sequence[Path]) -> tuple:
    """Path, mtime and size per clip — what makes a probe's answer stale."""
    out = []
    for path in paths:
        try:
            stat = path.stat()
            out.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            out.append((str(path), None, None))
    return tuple(out)


@st.cache_data(show_spinner=False, max_entries=512)
def _probed_size(token: str, _views: Sequence, _orientation: str, _transforms: Mapping) -> tuple:
    """The export's pixel size, probed once per clip rather than per rerun.

    An expander's body runs whether or not it is open (the reason
    :func:`~behavior_graphs.viz.streamlit.lazy.lazy_section` exists), and this
    panel sits at the bottom of every video panel — so the uncached version
    forked an ffprobe per camera on every click in the graph above, to fill in a
    caption nobody had asked to see. Keyed on the clips' stat, so a re-collected
    episode still re-probes.
    """
    return clip_figure.export_size(
        list(_views), orientation=_orientation, view_transforms=_transforms
    )


@st.cache_data(show_spinner=False, max_entries=4)
def _rendered(token: str, _payload: dict) -> bytes | None:
    """One encoded clip, keyed on *token*.

    ``_payload`` is underscore-prefixed so Streamlit leaves it out of the hash
    — it holds ``Path`` and :class:`Band` objects, and *token* stands for them.
    Few entries: these are megabytes each, and the reader wants the last one
    back rather than the last fifty.
    """
    return clip_figure.export_clip(**_payload)


def slice_band(
    start: int | None,
    end: int | None,
    label: str | None,
    color: str = SLICE_COLOR,
) -> list[Band]:
    """A single selected behavior's span as the one band of a bar.

    The player draws this case as one orange block on a grey track rather than
    as a chain, and the export follows it: a mid-graph node is a claim about one
    stretch of the rollout, and padding the strip out with the behaviors either
    side would be the export asserting something the page did not show.
    """
    if start is None or end is None or end <= start:
        return []
    return [Band(int(start), int(end), label or "selected behavior", color)]


def bands_from_segments(segments: Sequence[Sequence[object]] | None) -> list[Band]:
    """The player's ``(start, end, label, color)`` tuples as :class:`Band`\\ s."""
    out = []
    for segment in segments or ():
        start, end, label, color = segment[0], segment[1], segment[2], segment[3]
        if end is None or start is None or int(end) <= int(start):
            continue
        out.append(Band(int(start), int(end), str(label), str(color)))
    return out


def episode_bands(
    labels: np.ndarray,
    metadata: Sequence[dict],
    ep_idx: int,
    *,
    cluster_names: Mapping[int, str] | None = None,
    colors: Sequence[str] = tuple(CLUSTER_COLORS),
    frame_count: int | None = None,
) -> list[Band]:
    """Every behavior this episode passes through, end to end.

    For a caller with no node selected — the Figures page, where the point is
    the whole rollout against the whole decomposition rather than one branch of
    it. Built from the run-length-collapsed window labels
    (:attr:`~behavior_graphs.viz.streamlit.episode_index.EpisodeIndex.rle`), so
    consecutive windows of one behavior are one band, and coloured by cluster id
    the way the demo's cluster timeline is — not by position in a path, which is
    what a *selection* is coloured by.
    """
    from behavior_graphs.viz.streamlit.episode_index import get_episode_index

    rle = get_episode_index(np.asarray(labels), list(metadata)).rle.get(int(ep_idx)) or []
    bands = []
    for i, (start, cluster) in enumerate(rle):
        end = rle[i + 1][0] if i + 1 < len(rle) else frame_count
        if end is None or end <= start:
            continue
        label = (cluster_names or {}).get(int(cluster)) or f"Behavior {int(cluster)}"
        bands.append(Band(int(start), int(end), str(label), colors[int(cluster) % len(colors)]))
    return bands


def clip_export_panel(
    views: Sequence[tuple[str, str | Path]],
    bands: Sequence[Band] = (),
    *,
    key: str,
    filename: str,
    total_frames: int | None = None,
    fps: int = 10,
    view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    visible_cameras: Sequence[str] | None = None,
    expanded: bool = False,
) -> None:
    """Draw the export controls for the clip the player above is showing.

    *views* is every camera of the episode, ``[(camera, path)]``, as the player
    is given them; *visible_cameras* is what the player currently has on screen,
    which is where the camera choice starts — a reader who narrowed the view
    picker meant it.
    """
    # Only views that are actually on disk: a cached episode index can outlive
    # a deleted rollout, and the honest answer to that is the same one the
    # player gives — say there is no video, rather than offer a button that
    # encodes nothing.
    views = [(camera, Path(path)) for camera, path in views if Path(path).is_file()]
    with st.expander("⬇ Export this clip as an MP4", expanded=expanded):
        if not views:
            st.info("No video for this episode, so there is nothing to export.")
            return
        if clip_figure.ffmpeg_exe() is None:
            st.info("No ffmpeg on this host, so a clip cannot be encoded here.")
            return

        cameras = [camera for camera, _ in views]
        chosen = cameras
        if len(cameras) > 1:
            default = [c for c in (visible_cameras or cameras) if c in cameras] or cameras
            chosen = st.multiselect(
                "Camera views to export",
                options=cameras,
                default=default,
                key=f"{key}_cams",
                help=(
                    "Starts at what the player above is showing. Every view is "
                    "the same episode, so they stay frame-aligned in the file."
                ),
            )
            if not chosen:
                st.caption("Pick at least one view to export.")
                return

        orientation = HORIZONTAL
        if len(chosen) > 1:
            orientation = st.radio(
                "Stack the views",
                (HORIZONTAL, VERTICAL),
                format_func=ORIENTATION_NAMES.__getitem__,
                horizontal=True,
                key=f"{key}_orientation",
                help=(
                    "Side by side reads as one moment from two angles, and is "
                    "what a double-column figure has room for. Stacked keeps "
                    "each view wide enough to see in a single column."
                ),
            )

        with_bar = True
        if bands:
            with_bar = st.checkbox(
                "Behavior bar below the video",
                value=True,
                key=f"{key}_with_bar",
                help=(
                    "The strip under the player, burned into the file: the "
                    "bands at the same fractions of the clip, the behavior "
                    "under the playhead named in full, the rest of the legend "
                    "dimmed. Off exports the views alone."
                ),
            )
        else:
            st.caption(
                "No behavior segments for this selection, so the export is the "
                "video alone. Click a behavior, a path or an outcome in the "
                "graph to get the bar."
            )

        export_views = [(camera, path) for camera, path in views if camera in set(chosen)]
        payload = {
            "views": export_views,
            "bands": [b for b in bands] if with_bar else [],
            "total_frames": total_frames,
            "fps": fps,
            "orientation": orientation,
            "view_transforms": {
                camera: dict(spec)
                for camera, spec in (view_transforms or {}).items()
                if camera in set(chosen)
            },
            "with_bar": with_bar,
        }
        token = _fingerprint(
            {
                "views": [str(path) for _, path in export_views],
                "bands": [(b.start, b.end, b.label, b.color) for b in payload["bands"]],
                "total_frames": total_frames,
                "fps": fps,
                "orientation": orientation,
                "transforms": payload["view_transforms"],
            }
        )

        size = _probed_size(
            _fingerprint(
                {
                    "stat": _stat_signature([path for _, path in export_views]),
                    "orientation": orientation,
                    "transforms": payload["view_transforms"],
                }
            ),
            export_views,
            orientation,
            payload["view_transforms"],
        )
        if size is not None:
            st.caption(f"`{size[0]}×{size[1]}` plus the bar, H.264, no audio.")

        built_key = f"{key}_built"
        if st.button(
            "Build the MP4",
            key=f"{key}_build",
            help="Encodes now. A two-view real-robot episode takes a few seconds.",
        ):
            st.session_state[built_key] = token
        # Only the export these options describe. Change a camera or the
        # orientation and the button comes back rather than the download
        # quietly serving the previous file.
        if st.session_state.get(built_key) != token:
            return
        with st.spinner("Encoding…"):
            data = _rendered(token, payload)
        if not data:
            st.session_state.pop(built_key, None)
            st.warning("ffmpeg could not encode this clip. The source may be truncated.")
            return
        st.video(data)
        st.download_button(
            f"⬇ {filename}.mp4 ({len(data) / 1e6:.1f} MB)",
            data=data,
            file_name=f"{filename}.mp4",
            mime="video/mp4",
            key=f"{key}_download",
        )
