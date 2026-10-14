"""Video frame extraction for VLM consumption.

Produces contact sheets — grids of timestamped frames — from episode MP4 files.
Contact sheets are the preferred input format for VLM node labeling (per Macrodata
Labs' finding that 20-frame 224px grids reduce API cost 12× vs per-frame calls
while maintaining or improving labeling quality).

A dataset can hold several views of every episode (``media/<camera>/`` — see
:meth:`behavior_graphs.envs.trajectory_dir.TrajectoryDirectory.camera_names`), and
:func:`contact_sheet_for_node` takes a *cameras* list to put more than one of
them in front of the model: each sampled episode contributes one labelled
row-group per view, all views sampled at the *same* timesteps so the rows read as
one moment seen from several angles rather than as separate phases. Passing
``cameras=None`` (the default) keeps the single-view behaviour every caller had
before, because each extra view multiplies the sheet's image tokens. The
transition and single-episode sheets are single-view: they are read by the MCP
tools and the report, which ask about one clip at a time.

Views are oriented the way the app shows them, via the same per-camera
``rotate``/``mirror`` config the web proxies bake in (*view_transforms*, from
:func:`behavior_graphs.viz.streamlit.data.camera_transforms`) — DROID's wrist
camera hangs upside down, and handing a VLM inverted frames is worse than
handing it none.

Requires the ``viz`` optional dependency group (imageio + imageio-ffmpeg + pillow).
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _require_imageio():
    try:
        import imageio.v3 as iio

        return iio
    except ImportError:
        raise ImportError(
            "imageio is required for frame extraction. "
            "Install with: pip install 'behavior_graphs[viz]'"
        )


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont

        return Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for contact sheet assembly. "
            "Install with: pip install 'behavior_graphs[viz]'"
        )


def _fit_tile(img, tile_px: int):
    """Scale *img* so its **shorter** side is *tile_px*, preserving aspect ratio.

    Not ``resize((tile_px, tile_px))``: that squashes non-square footage into a
    square, and several rollout sets are widescreen (cooking is 640x360,
    some sim footage 1280x720). A 16:9 frame squeezed to 1:1 compresses the horizontal
    by ~44% relative to the vertical — round objects become ellipses and the
    arm's proportions change — which is a poor thing to hand a model being
    asked what manipulation it is looking at.

    Shorter side rather than longer, so fixing the aspect never costs
    resolution. 640x360 at tile_px=224 becomes 398x224: the vertical is
    sampled exactly as finely as the old square resize did (360->224), and the
    horizontal stops being crushed. Scaling the longer side instead would give
    224x126 — undistorted but genuinely coarser than what it replaced. Square
    footage (libero 224, mimicgen 128) is unaffected either way.

    Costs more image tokens for widescreen sources, in proportion to how wide
    they are; lower *tile_px* to trade back.

    Tiles from one *view* stay a uniform size; a sheet blending several cameras
    with different aspect ratios does not, which is why
    :func:`make_contact_sheet` sizes its cells from the widest and tallest tile
    rather than from the first one.
    """
    from PIL import Image as _Image

    w, h = img.size
    if w == h == tile_px:
        return img
    scale = tile_px / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), _Image.LANCZOS)


def _oriented(img, rotate: int, mirror: bool):
    """Apply a camera's declared geometry to one frame, already normalized.

    The same geometry, from the same config, that
    :func:`behavior_graphs.viz.media.ensure_proxy` bakes into that view's web proxy
    — so a sheet shows a camera the way the app's players do. Without it DROID's
    wrist view reaches the VLM upside down, which reliably costs the label: an
    inverted gripper closing on an object does not read as a grasp.

    Mirror before rotation, matching
    :func:`behavior_graphs.viz.media.transform_filters`, so the two orderings cannot
    disagree; *rotate* is degrees clockwise, as
    :func:`behavior_graphs.viz.media.norm_transform` reports it. Applied before
    :func:`_fit_tile`, so *tile_px* governs the shorter side of the frame as
    *displayed* — a quarter turn swaps the axes.

    Takes the resolved pair rather than the config dict because it runs per frame
    while the normalization is constant for a whole clip.
    """
    from PIL import Image as _Image

    if mirror:
        img = img.transpose(_Image.FLIP_LEFT_RIGHT)
    if rotate:
        # norm_transform reports degrees *clockwise*; PIL rotates the other way.
        img = img.rotate(-rotate, expand=True)
    return img


def extract_episode_frames(
    video_path: Path,
    frame_indices: list[int],
    tile_px: int = 224,
    stamp_index: bool = True,
    episode_idx: int | None = None,
    transform: Mapping[str, object] | None = None,
    pad_missing: bool = False,
) -> list[np.ndarray | None]:
    """Read specific frames from an MP4 and resize to *tile_px* × *tile_px*.

    Args:
        video_path: Path to the episode MP4.
        frame_indices: 0-based frame indices to extract (in any order; deduplicated
            and clipped to valid range internally).
        tile_px: Shorter side of each output tile in pixels; the longer side
            follows the source aspect ratio (640x360 -> 398x224). 224px is
            optimal for VLMs per Macrodata Labs benchmark, but widescreen
            footage (cooking is 640x360) has detail to spare above it.
        stamp_index: If True, overlay the frame index as text on the tile.
            Visual timestamps outperform text-only maps for VLM grounding.
        episode_idx: If given, included in the stamp (``"ep{episode_idx}
            t{idx}"``) so a contact sheet blending frames from multiple
            episodes (e.g. :func:`contact_sheet_for_node`) doesn't read as one
            continuous, inexplicably-jumping sequence — the frame index alone
            resets/jumps at every episode boundary with no visual cue why.
        transform: This camera's declared ``rotate``/``mirror`` geometry — see
            :func:`_orient`. Applied before the resize.
        pad_missing: If True, an index that couldn't be read yields ``None`` in
            its own position instead of being dropped. Only useful when several
            views of one episode are being sampled at the same timesteps: a
            camera that stopped recording early would otherwise return a
            *shorter* list, sliding its row of the sheet left and silently
            breaking the column-per-timestep alignment that makes the views
            comparable in the first place. ``None`` is what
            :func:`make_contact_sheet` already draws as a blank cell.

    Returns:
        List of uint8 numpy arrays (H, W, 3) in the same order as
        *frame_indices*. Indices that fell out of range are skipped, or kept as
        ``None`` when *pad_missing*.
    """
    from behavior_graphs.viz.media import norm_transform

    iio = _require_imageio()
    Image, ImageDraw, ImageFont = _require_pillow()

    video_path = Path(video_path)
    if not video_path.exists():
        return [None] * len(frame_indices) if pad_missing else []

    # Constant for the whole clip: resolved once rather than per frame.
    rotate, mirror = norm_transform(transform)

    # Open the container once and read every requested frame from the same
    # handle — a per-index imread would reopen and re-seek the MP4 each time.
    idx_map: dict[int, np.ndarray] = {}
    with iio.imopen(video_path, "r", plugin="pyav") as vid:
        props = vid.properties()
        n_frames = props.shape[0] if props.shape and props.shape[0] else None

        for idx in sorted(set(frame_indices)):
            if n_frames is not None and idx >= n_frames:
                continue
            try:
                frame = vid.read(index=idx)
            except Exception:
                continue
            img = _fit_tile(_oriented(Image.fromarray(frame), rotate, mirror), tile_px)
            if stamp_index:
                draw = ImageDraw.Draw(img)
                text = f"ep{episode_idx} t{idx}" if episode_idx is not None else str(idx)
                # White text with black shadow for readability on any background
                draw.text((2, 2), text, fill=(0, 0, 0))
                draw.text((1, 1), text, fill=(255, 255, 255))
            idx_map[idx] = np.array(img)

    # Re-order to match the original frame_indices order
    if pad_missing:
        return [idx_map.get(i) for i in frame_indices]
    return [idx_map[i] for i in frame_indices if i in idx_map]


def make_contact_sheet(
    frames: list[np.ndarray | None],
    cols: int = 5,
    quality: int = 85,
    row_labels: dict[int, str] | None = None,
    tile_gap: int = 8,
    label_height: int = 26,
) -> bytes:
    """Composite *frames* into a grid image, returned as JPEG bytes.

    Args:
        frames: List of uint8 (H, W, 3) arrays. ``None`` leaves a blank cell —
            used both to pad a row so the *next* group starts fresh (see
            *row_labels*) and to hold a timestep's place in a view that has no
            frame there (see :func:`extract_episode_frames`'s ``pad_missing``).
            Sizes need not match: cells are as wide as the widest tile on the
            sheet and as tall as the tallest in their own row, with smaller tiles
            centred. Uniform tiles — one dataset, one camera — therefore lay out
            exactly as they always did, while a sheet mixing a 4:3 wrist view
            with a 16:9 exterior one stays in register instead of pasting every
            row at the first tile's width.
        cols: Number of columns in the grid.  4×5 = 20 frames is the sweet spot
            per Macrodata Labs benchmarks.
        quality: JPEG compression quality.
        row_labels: Optional ``{row_index: text}``. Draws *text* in a header
            band above that row (e.g. ``"Episode 3 - frames 12-45"``), so a
            sheet blending frames from multiple episodes/segments (e.g.
            :func:`contact_sheet_for_node`) shows *why* the frame index
            jumps at a group boundary instead of reading as one confusingly
            discontinuous sequence. Plain ASCII only — Pillow's default
            bitmap font (used here; no bundled TTF to depend on) renders
            Unicode punctuation like "·"/"–" as garbled boxes.
        tile_gap: Padding in pixels between tiles, both directions, and
            around the sheet's outer edge.
        label_height: Height in pixels of each row-label header band.

    Returns:
        JPEG bytes of the assembled contact sheet.
    """
    Image, ImageDraw, _ = _require_pillow()
    real_frames = [f for f in frames if f is not None]
    if not real_frames:
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (30, 30, 30)).save(buf, format="JPEG")
        return buf.getvalue()

    row_labels = row_labels or {}
    # Cell width is the widest tile on the sheet, because a column has to mean the
    # same thing on every row — that is what makes a column one timestep across
    # views. Height is per row, because nothing lines up vertically: giving every
    # row the tallest tile's height would pad a sheet of 16:9 rows to portrait
    # height for one 90°-rotated view, doubling its area in dark pixels and
    # re-sending that in both labeling passes.
    cell_w = max(f.shape[1] for f in real_frames)
    rows = (len(frames) + cols - 1) // cols
    width = cell_w * cols + tile_gap * (cols + 1)
    shortest = min(f.shape[0] for f in real_frames)
    row_h = [
        max(
            (f.shape[0] for f in frames[r * cols : (r + 1) * cols] if f is not None),
            # A row with no frames at all is a view that lost a whole row's worth
            # of consecutive timesteps (group padding never leaves one, since it
            # only tops up a row that already holds frames). Keep it visible, as
            # the gap it is, at the smallest tile height on the sheet — the
            # tallest would hand a stretch of missing footage the largest band.
            default=shortest,
        )
        for r in range(rows)
    ]

    # Each row's y-offset stacks: the gap, its own label band (if any), then
    # every prior row's own height + gap.
    row_y: list[int] = []
    y = tile_gap
    for r in range(rows):
        if r in row_labels:
            y += label_height
        row_y.append(y)
        y += row_h[r] + tile_gap
    height = y

    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for r, label in row_labels.items():
        if r < len(row_y):
            draw.text((tile_gap + 2, row_y[r] - label_height + 6), label, fill=(220, 220, 220))
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        r, c = divmod(i, cols)
        h, w = frame.shape[:2]
        x = tile_gap + c * (cell_w + tile_gap) + (cell_w - w) // 2
        sheet.paste(Image.fromarray(frame), (x, row_y[r] + (row_h[r] - h) // 2))
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def episode_views(traj, episode_idx: int, cameras: Sequence[str] | None) -> list[tuple[str, Path]]:
    """``[(camera, clip)]`` for one episode, honouring a *cameras* request.

    Both branches are the reader's own rules (see
    :meth:`~behavior_graphs.envs.trajectory_dir.TrajectoryDirectory.primary_view`
    and :meth:`~behavior_graphs.envs.trajectory_dir.TrajectoryDirectory.video_paths`);
    this is the sheet-shaped view of them. ``cameras=None`` is the
    pre-multi-camera contract — one clip, named after the view it came from so
    that view's configured orientation applies. A name no view has is the
    caller's problem to catch up front (see ``scripts/label_nodes.py``), because
    here it is indistinguishable from a camera this episode happens to lack.
    """
    if cameras is None:
        view = traj.primary_view(episode_idx)
        return [view] if view else []
    return list(traj.video_paths(episode_idx, cameras).items())


def _view_suffix(camera: str, multi_view: bool) -> str:
    """`` [wrist_image]`` for a row label, or nothing when the sheet has one view.

    A single-view sheet gains nothing from being told which view it is — the
    name is the same on every row — and the labels are drawn in Pillow's default
    bitmap font at a fixed band height, so the shorter they stay the better.
    """
    return f" [{camera}]" if multi_view else ""


def _episode_frame_count(video_path: Path) -> int:
    """Total frame count of an episode video, or 0 if unreadable."""
    try:
        iio = _require_imageio()
        with iio.imopen(video_path, "r", plugin="pyav") as vid:
            props = vid.properties()
            return props.shape[0] if (props.shape and props.shape[0]) else 0
    except Exception:
        return 0


def contact_sheet_for_node(
    run_dir: Path,
    rollout_dir: Path,
    node_id: int,
    frames_per_episode: int = 5,
    max_episodes: int = 4,
    tile_px: int = 224,
    cols: int = 5,
    episode_filter: set[int] | None = None,
    extend_to_episode_end: bool = False,
    cameras: Sequence[str] | None = None,
    view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    primary_camera: str | None = None,
) -> bytes:
    """Build a contact sheet for all windows belonging to *node_id*.

    Samples *frames_per_episode* frames from each of up to *max_episodes*
    episodes that visit the node, one labelled row-group per episode (per
    *camera* view, when more than one is asked for).  Frames are drawn from
    within each window's ``[start, end)`` timestep range.

    Every episode gets the same number of frames, so the sheet grows with how
    many episodes visit the node rather than each episode's coverage thinning
    out.  A node visited twice therefore reads at the same detail as one
    visited eight times.

    Args:
        run_dir: Pipeline run directory (contains ``labels.npy``, ``windows.json``).
        rollout_dir: Trajectory/rollout directory with ``media/`` videos.
        node_id: Cluster ID to visualize.
        frames_per_episode: Frames sampled from each episode row.
        max_episodes: Cap on distinct episodes to sample from, so the total
            tile count is at most ``frames_per_episode * max_episodes``.
        tile_px: Tile size in pixels.
        cols: Contact sheet column count.
        episode_filter: If given, only sample from episodes in this set (e.g.
            episodes with a specific outcome) — other episodes visiting the
            node are ignored entirely, as if they never visited it.
        extend_to_episode_end: If True, extend each sampled episode's covered
            range past this node's own window through to the episode's last
            recorded frame. A node's window ends wherever the clustering cut
            it, not wherever the episode itself ends — FAILURE/SUCCESS are
            synthetic terminal markers with no window of their own, so the
            actual failure/success moment (a drop, a wrong grasp, a knocked
            object) happens *after* the window and would otherwise never be
            sampled. Use this whenever the caller specifically wants to see
            what happens through to termination, not just this node's phase.
        cameras: Camera views to include, in the order given. ``None`` (the
            default) is one view per episode, as before multi-camera media
            existed; every extra name multiplies the sheet's tile count, so
            this opts in rather than defaulting to everything the dataset has.
            Each view of an episode is sampled at the *same* timesteps and gets
            its own labelled row-group, so a column holds one moment across
            views and the model can be told as much.
        view_transforms: ``{camera: {rotate, mirror}}`` display geometry, as
            :func:`behavior_graphs.viz.streamlit.data.camera_transforms` returns
            it. Views with no entry are used as recorded.
        primary_camera: Which view *cameras=None* means, as
            :func:`behavior_graphs.viz.streamlit.data.primary_camera` declares it.
            Handed to the reader rather than resolved here (this module does not
            read ``configs/``), so the one view a sheet shows by default is the
            one the app shows first instead of whichever ``media/<camera>/``
            sorts first.

    Returns:
        JPEG bytes of the contact sheet (empty dark image if no video found).
    """
    from behavior_graphs.pipeline.artifacts import RunArtifacts

    run_dir, rollout_dir = Path(run_dir), Path(rollout_dir)

    art = RunArtifacts(run_dir)
    labels = art.load_array("labels")
    windows_meta = art.load_windows_meta()["windows"]  # list of {episode_idx, start, end}

    # Collect windows belonging to this node
    node_windows = [w for w, lbl in zip(windows_meta, labels) if int(lbl) == node_id]
    if episode_filter is not None:
        node_windows = [w for w in node_windows if w["episode_idx"] in episode_filter]
    if not node_windows:
        return make_contact_sheet([])

    # Sample up to max_episodes distinct episodes
    seen_eps: dict[int, list[dict]] = {}
    for w in node_windows:
        ep = w["episode_idx"]
        seen_eps.setdefault(ep, []).append(w)
    sampled_eps = list(seen_eps.keys())[:max_episodes]

    frames_per_ep = max(1, frames_per_episode)
    all_frames: list[np.ndarray | None] = []
    row_labels: dict[int, str] = {}

    from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

    traj = TrajectoryDirectory(rollout_dir, primary_camera=primary_camera)

    transforms = view_transforms or {}
    # Whether the *sheet* holds several views, which is what the labeler is told
    # ("each episode appears once per camera view"). Per-episode counts would drop
    # the view name from the rows of an episode that lost a camera, leaving the
    # model to guess which angle an unqualified row is inside a sheet it was told
    # holds two.
    multi_view = bool(cameras) and len(cameras) > 1

    for ep_idx in sampled_eps:
        views = episode_views(traj, ep_idx, cameras)
        if not views:
            continue
        ep_windows = seen_eps[ep_idx]
        # Every timestep this node's (overlapping) windows actually cover in
        # this episode, then evenly spread frames_per_ep across positions in
        # that list -- NOT a fixed stride per window truncated to the first
        # frames_per_ep, which silently drops the back half of any visit
        # longer than the frame budget (a 9-timestep visit with a 5-frame
        # budget used to show only t0-4, never t5-8). Indexing by position
        # in the covered list rather than by raw timestep also gives fair
        # proportional coverage when a node's visit is discontiguous (e.g.
        # frames at t0-4 and, separately, t59-69) instead of wasting samples
        # on a numeric gap where the node isn't even present.
        covered = sorted({t for w in ep_windows for t in range(w["start"], w["end"])})
        if extend_to_episode_end and covered:
            # From the first view: the timesteps are the episode's, not any one
            # camera's, and a view that recorded fewer frames pads (see below)
            # rather than shortening the range for the views that have them.
            total = _episode_frame_count(views[0][1])
            if total > covered[-1] + 1:
                covered = sorted(set(covered) | set(range(covered[-1], total)))
        k = min(frames_per_ep, len(covered))
        # Endpoint-inclusive spacing (see _spread_frames) -- step = len/k
        # with positions up to k-1 never actually reaches covered[-1], which
        # would silently hide the tail of the range (e.g. the frames right
        # before a FAILURE transition) even after extend_to_episode_end has
        # correctly widened `covered` to include them.
        if k <= 1:
            all_ts = [covered[0]]
        else:
            step = (len(covered) - 1) / (k - 1)
            all_ts = sorted({covered[round(step * i)] for i in range(k)})
        for camera, video_path in views:
            # pad_missing only on a multi-view sheet, to keep the columns in
            # register; with one view it keeps the old "drop what you can't read"
            # behaviour, so a sheet doesn't grow blank cells for frames that were
            # never there. Sheet-wide rather than per-episode: an episode that
            # lost a camera still shares its columns with the episodes that
            # didn't.
            frames = extract_episode_frames(
                video_path,
                all_ts,
                tile_px=tile_px,
                transform=transforms.get(camera),
                pad_missing=multi_view,
            )
            if not any(f is not None for f in frames):
                continue
            # Pad to a fresh row so this group's header lines up with its
            # first tile rather than landing mid-row.
            if all_frames and len(all_frames) % cols != 0:
                all_frames.extend([None] * (cols - len(all_frames) % cols))
            row_labels[len(all_frames) // cols] = (
                f"Episode {ep_idx}{_view_suffix(camera, multi_view)} "
                f"- frames {all_ts[0]}-{all_ts[-1]}"
            )
            all_frames.extend(frames)

    return make_contact_sheet(all_frames, cols=cols, row_labels=row_labels)


def contact_sheets_for_transition(
    run_dir: Path,
    rollout_dir: Path,
    src_node_id: int,
    dst_node_id: int,
    n_episodes: int = 3,
    n_frames_per_ep: int = 10,
    tile_px: int = 224,
    cols: int = 5,
) -> list[bytes]:
    """Return per-episode contact sheets for episodes that traverse src→dst.

    Each sheet shows the frames within the src window followed by the dst window,
    giving the VLM a "before → after" view of the transition.

    Returns:
        List of JPEG bytes, one per episode (up to *n_episodes*).
    """
    from behavior_graphs.pipeline.artifacts import RunArtifacts

    run_dir, rollout_dir = Path(run_dir), Path(rollout_dir)

    art = RunArtifacts(run_dir)
    labels = art.load_array("labels")
    windows_meta = art.load_windows_meta()["windows"]

    # Group windows by episode, preserving order
    ep_windows: dict[int, list[tuple[int, dict]]] = {}
    for i, (w, lbl) in enumerate(zip(windows_meta, labels)):
        ep_windows.setdefault(w["episode_idx"], []).append((int(lbl), w))

    from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

    traj = TrajectoryDirectory(rollout_dir)
    sheets: list[bytes] = []

    for ep_idx, wlist in ep_windows.items():
        if len(sheets) >= n_episodes:
            break
        # Find consecutive (src, dst) pairs
        for j in range(len(wlist) - 1):
            lbl_a, w_a = wlist[j]
            lbl_b, w_b = wlist[j + 1]
            if lbl_a == src_node_id and lbl_b == dst_node_id:
                video_path = traj.video_path(ep_idx)
                if video_path is None:
                    break
                # Src window frames + dst window frames
                half = n_frames_per_ep // 2
                src_ts = _spread_frames(w_a["start"], w_a["end"], half)
                dst_ts = _spread_frames(w_b["start"], w_b["end"], half)
                src_frames = extract_episode_frames(video_path, src_ts, tile_px=tile_px)
                dst_frames = extract_episode_frames(video_path, dst_ts, tile_px=tile_px)
                if not src_frames or not dst_frames:
                    break
                frames: list[np.ndarray | None] = list(src_frames)
                row_labels = {0: f"Before - ep{ep_idx} frames {src_ts[0]}-{src_ts[-1]}"}
                if len(frames) % cols != 0:
                    frames.extend([None] * (cols - len(frames) % cols))
                row_labels[len(frames) // cols] = (
                    f"After - ep{ep_idx} frames {dst_ts[0]}-{dst_ts[-1]}"
                )
                frames.extend(dst_frames)
                sheets.append(make_contact_sheet(frames, cols=cols, row_labels=row_labels))
                break

    return sheets


def contact_sheet_for_episode(
    rollout_dir: Path,
    episode_idx: int,
    n_frames: int = 12,
    tile_px: int = 224,
    cols: int = 5,
) -> bytes:
    """Contact sheet of *n_frames* evenly-spaced frames from one episode.

    Args:
        rollout_dir: Trajectory/rollout directory with ``media/`` videos.
        episode_idx: Episode index to visualize.
        n_frames: Total frames to include (default 12, a good balance between
            coverage and VLM token cost).
        tile_px: Tile size in pixels.
        cols: Contact sheet column count.

    Returns:
        JPEG bytes of the contact sheet.

    Raises:
        FileNotFoundError: If no video is found for *episode_idx*.
    """
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory

    traj = RolloutDirectory(Path(rollout_dir))
    video_path = traj.video_path(episode_idx)
    if video_path is None:
        raise FileNotFoundError(f"No video found for episode {episode_idx} in {rollout_dir}")

    # Probe frame count and extract in a single open to avoid the double-open
    # that would result from calling extract_episode_frames (which probes again).
    iio = _require_imageio()
    Image, ImageDraw, _ = _require_pillow()

    idx_map: dict[int, np.ndarray] = {}
    selected: list[int] = []
    with iio.imopen(video_path, "r", plugin="pyav") as vid:
        props = vid.properties()
        total = props.shape[0] if (props.shape and props.shape[0]) else 0
        if total == 0:
            raise FileNotFoundError(
                f"Episode {episode_idx} video has no readable frames in {rollout_dir}"
            )
        selected = _spread_frames(0, total, n_frames)
        for idx in selected:
            try:
                frame = vid.read(index=idx)
            except Exception:
                continue
            img = _fit_tile(Image.fromarray(frame), tile_px)
            draw = ImageDraw.Draw(img)
            text = f"ep{episode_idx} t{idx}"
            draw.text((2, 2), text, fill=(0, 0, 0))
            draw.text((1, 1), text, fill=(255, 255, 255))
            idx_map[idx] = np.array(img)

    frames_out = [idx_map[i] for i in selected if i in idx_map]
    row_labels = {0: f"Episode {episode_idx} - frames {selected[0]}-{selected[-1]}"}
    return make_contact_sheet(frames_out, cols=cols, row_labels=row_labels)


def frames_for_episode_selection(
    rollout_dir: Path,
    episode_idx: int,
    frame_indices: list[int],
    tile_px: int = 224,
    cols: int = 5,
) -> bytes:
    """Contact sheet from explicit frame indices within one episode.

    Args:
        rollout_dir: Trajectory/rollout directory with ``media/`` videos.
        episode_idx: Episode index to read from.
        frame_indices: 0-based frame indices to include (clipped to valid range
            automatically; out-of-range indices are silently skipped).
        tile_px: Tile size in pixels.
        cols: Contact sheet column count.

    Returns:
        JPEG bytes of the contact sheet.

    Raises:
        FileNotFoundError: If no video is found for *episode_idx*.
    """
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory

    traj = RolloutDirectory(Path(rollout_dir))
    video_path = traj.video_path(episode_idx)
    if video_path is None:
        raise FileNotFoundError(f"No video found for episode {episode_idx} in {rollout_dir}")

    frames = extract_episode_frames(
        video_path, list(frame_indices), tile_px=tile_px, episode_idx=episode_idx
    )
    return make_contact_sheet(frames, cols=cols)


def _spread_frames(start: int, end: int, n: int) -> list[int]:
    """Uniformly sample *n* frame indices in ``[start, end)``, endpoint-inclusive.

    ``step = length / n`` with indices at ``start + step * i`` for
    ``i in range(n)`` never actually reaches ``end - 1`` (the last valid
    index) — the final sample lands at ``start + step * (n - 1)``, which is
    short of the true end by about one step's worth. That systematically
    hides whatever happens right at the end of the range (e.g. the actual
    moment of a rollout failure), which is usually exactly what a caller
    sampling "the whole range" wants to see. Spacing by ``(length - 1) /
    (n - 1)`` instead guarantees the last sample lands on ``end - 1``.
    """
    length = max(1, end - start)
    if n >= length:
        return list(range(start, end))
    if n <= 1:
        return [start]
    step = (length - 1) / (n - 1)
    return sorted({start + round(step * i) for i in range(n)})
