"""A rollout clip and its behavior bar, burned into one MP4.

The player on screen keeps these two things apart: an HTML ``<video>`` for each
camera, and, drawn under them in the same component, a strip of colored bands
saying which behavior the graph assigns to each moment of the clip (see
:mod:`behavior_graphs.viz.streamlit.components.mp4_player`). Reading the one
against the other *is* the claim a behavior graph makes, and it is the one part
of the app that cannot be screenshotted into a paper, a slide or a supplementary
video: the bands only mean anything while the playhead is moving across them.

So this module re-draws the strip frame by frame with PIL and lets ffmpeg stack
it under the video, giving a file that carries the annotation with it. What it
draws is deliberately the same reading as the player:

* the bands sit at the same fractions of the clip, computed frame → fraction the
  way the player's JS does (:func:`band_spans`) — via ``total_frames`` rather
  than ``fps``, because a VLA rollout is recorded at wall-clock speed and
  stepped at a few Hz, so ``total_frames / fps`` is not the clip's duration;
* the behavior under the playhead is named in full and the rest of the legend
  dims, which is what the on-screen badge on each view does;
* nothing is inferred that the player would not show. The bands are the ones the
  caller's selection in the graph produced.

Multiple cameras stack in one row or one column (:func:`stack_geometry`), which
is the choice a figure actually has to make: side by side reads as "the same
moment from two angles", stacked vertically fits a single-column figure.

No Streamlit here. The panel that calls this is
:mod:`behavior_graphs.viz.streamlit.components.clip_export`.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from behavior_graphs.viz.media import ffmpeg_exe, norm_transform, transform_filters

_log = logging.getLogger(__name__)

#: Stack the views side by side / one above the other.
HORIZONTAL = "horizontal"
VERTICAL = "vertical"

#: How wide an export may get, in pixels. A cap rather than a target: the views
#: are never upscaled, so a 256×256 simulation clip exports at its own size and
#: only a real-robot capture is scaled down to this.
MAX_WIDTH = 1280

#: The bar's colours — the player's, so an exported clip and the page it came
#: from read the same way.
BAR_BG = "#111318"
TRACK_BG = "#333333"
LABEL_DIM = "#9aa0a6"
LABEL_ON = "#ffffff"

#: Rows of legend the bar will grow to, at most. Behavior labels are sentences
#: the labeling model wrote, and a twelve-behavior path would otherwise give a
#: strip taller than the video it annotates.
MAX_LEGEND_ROWS = 4

#: Stands in for the dropped entries in :attr:`BarLayout.rows` when the legend
#: does not fit. A legend that stops has to say so — silently listing six of a
#: path's twenty behaviors reads as a complete key to the bar, in a file whose
#: whole purpose is to be believed. Not a band index, so nothing ever looks it
#: up in the backdrops.
MORE_SENTINEL = -1

#: x264 quality for the export. A little better than the web proxy's: this file
#: is what goes into a supplementary bundle, and the bands are flat colour that
#: compresses to nothing anyway.
CRF = 20


@dataclass(frozen=True)
class Band:
    """One behavior's span of a clip: ``[start, end)`` in policy steps."""

    start: int
    end: int
    label: str
    color: str


@dataclass(frozen=True)
class ClipInfo:
    """What ffmpeg says a clip is, as stored — before any camera transform."""

    width: int
    height: int
    duration: float
    fps: float


@dataclass(frozen=True)
class BarLayout:
    """Where everything on the behavior bar goes, in pixels."""

    width: int
    height: int
    track_top: int
    track_height: int
    legend_top: int
    line_height: int
    rows: tuple[tuple[int, ...], ...]
    font_size: int
    #: How many behaviors the legend had no room to name. Drawn as
    #: ``+N more`` where :data:`MORE_SENTINEL` appears in *rows*.
    dropped: int = 0


def _even(value: float) -> int:
    """*value* as an even integer — ``yuv420p`` has no odd dimensions."""
    return max(2, int(round(value / 2)) * 2)


# ── What ffmpeg says the clip is ─────────────────────────────────────────────


def probe_clip(path: str | Path) -> ClipInfo | None:
    """*path*'s pixel size, duration and frame rate, or None if unreadable.

    ``ffprobe`` when it is on PATH, and the header ffmpeg prints to stderr
    otherwise: :func:`~behavior_graphs.viz.media.ffmpeg_exe` may hand back the
    binary imageio-ffmpeg ships, which comes with no ffprobe beside it.
    """
    probed = _ffprobe(path)
    return probed if probed is not None else _probe_via_ffmpeg(path)


def _ffprobe(path: str | Path) -> ClipInfo | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=30,
            text=True,
        )
        if out.returncode != 0:
            return None
        probed = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    streams = probed.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    width, height = stream.get("width"), stream.get("height")
    # A stream carries no duration in some containers; the format's is the same
    # number for a single-stream clip, which is what these are.
    duration = _as_float(stream.get("duration")) or _as_float(
        (probed.get("format") or {}).get("duration")
    )
    fps = _as_ratio(stream.get("avg_frame_rate"))
    if not width or not height or not duration:
        return None
    return ClipInfo(int(width), int(height), duration, fps or 30.0)


def _probe_via_ffmpeg(path: str | Path) -> ClipInfo | None:
    """Parse the banner ``ffmpeg -i`` prints when asked to do nothing else."""
    exe = ffmpeg_exe()
    if exe is None:
        return None
    try:
        # -i with no output is an error by design; the header lands on stderr
        # before ffmpeg gives up, which is all this wants.
        out = subprocess.run(
            [exe, "-hide_banner", "-i", str(path)],
            capture_output=True,
            timeout=30,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stderr or ""
    size = re.search(r"Video:.*?[, ](\d{2,5})x(\d{2,5})[,\s]", text)
    clock = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    rate = re.search(r"([\d.]+)\s+fps", text)
    if size is None or clock is None:
        return None
    hours, minutes, seconds = clock.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        return None
    return ClipInfo(
        int(size.group(1)),
        int(size.group(2)),
        duration,
        float(rate.group(1)) if rate else 30.0,
    )


def _as_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _as_ratio(value: object) -> float | None:
    """``"30000/1001"`` → 29.97. ffprobe reports frame rates as a fraction."""
    if not isinstance(value, str) or "/" not in value:
        return _as_float(value)
    num, _, den = value.partition("/")
    top, bottom = _as_float(num), _as_float(den)
    return top / bottom if top and bottom else None


def oriented_size(info: ClipInfo, transform: Mapping[str, object] | None) -> tuple[int, int]:
    """*info*'s size after the camera's configured rotation.

    A quarter turn swaps the axes, so a stack laid out from the stored sizes
    would letterbox that view inside a box the shape of its unrotated self.
    """
    rotate, _ = norm_transform(transform)
    return (info.height, info.width) if rotate in (90, 270) else (info.width, info.height)


# ── Laying the views out ─────────────────────────────────────────────────────


def stack_geometry(
    sizes: Sequence[tuple[int, int]],
    orientation: str = HORIZONTAL,
    *,
    max_width: int = MAX_WIDTH,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """``([(w, h) per view], (out_w, out_h))`` for one row or one column.

    Views are matched on the axis they are joined along — one height for a row,
    one width for a column — at the smallest view's measure, so nothing is
    upscaled into blur to match a bigger camera. A row that comes out wider than
    *max_width* is scaled down as a whole, which keeps every view's share of the
    frame rather than cropping the last one.

    Every dimension is even: the export is ``yuv420p``, whose chroma planes are
    half-sized, and ffmpeg refuses odd ones.
    """
    if not sizes:
        raise ValueError("stack_geometry needs at least one view")
    if orientation == VERTICAL:
        width = _even(min(min(w for w, _ in sizes), max_width))
        scaled = [(width, _even(h * width / w)) for w, h in sizes]
        return scaled, (width, sum(h for _, h in scaled))

    height = float(min(h for _, h in sizes))
    scaled = [(_even(w * height / h), _even(height)) for w, h in sizes]
    total = sum(w for w, _ in scaled)
    if total > max_width:
        height *= max_width / total
        scaled = [(_even(w * height / h), _even(height)) for w, h in sizes]
        total = sum(w for w, _ in scaled)
    return scaled, (total, scaled[0][1])


# ── The bar ──────────────────────────────────────────────────────────────────


def band_spans(
    bands: Sequence[Band],
    *,
    total_frames: int | None,
    duration: float,
    fps: float,
) -> list[tuple[float, float]]:
    """Each band as a ``(left, right)`` fraction of the clip.

    Through ``total_frames`` when there is one, exactly as the player's
    ``_pct``/``_sec`` do: a policy step is a fraction of the run, and the clip is
    the same run end to end, so a band's share of the steps is its share of the
    video however far the recording's frame rate is from the policy's. Dividing
    by *fps* instead puts every band of a VLA rollout in the first fifth of it.
    """
    spans = []
    for band in bands:
        if total_frames and total_frames > 0:
            left, right = band.start / total_frames, band.end / total_frames
        elif duration > 0 and fps > 0:
            left, right = band.start / fps / duration, band.end / fps / duration
        else:
            left, right = 0.0, 0.0
        spans.append((min(max(left, 0.0), 1.0), min(max(right, 0.0), 1.0)))
    return spans


def load_font(size: int):
    """A scalable sans face, via matplotlib's bundled DejaVu.

    matplotlib is a hard dependency of everything else that draws here and it
    ships the font, so this needs nothing installed system-wide — unlike the
    Times-metric faces the paper figures want (see
    :mod:`behavior_graphs.viz.streamlit.user_study.figure_export`). Falls back to
    PIL's bitmap default, which is unreadably small but not a crash.
    """
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # pragma: no cover - no matplotlib, or no usable face
        return ImageFont.load_default()


def _text_width(font, text: str) -> float:
    try:
        return float(font.getlength(text))
    except AttributeError:  # pragma: no cover - PIL without getlength
        return float(font.getsize(text)[0])


def bar_layout(labels: Sequence[str], width: int, font=None) -> BarLayout:
    """Where the track, the playhead and each legend entry go, for *width*.

    The height is whatever the legend needs, up to :data:`MAX_LEGEND_ROWS` —
    computed here and *only* here, because ffmpeg has to be told the size of the
    strip before the first frame of it is drawn.
    """
    font_size = max(9, int(round(width / 100)))
    font = font or load_font(font_size)
    chip = max(6, font_size - 2)
    pad = max(6, font_size // 2)
    gap = font_size * 2

    rows: list[list[int]] = []
    used = 0.0
    for i, label in enumerate(labels):
        entry = chip + 5 + _text_width(font, label)
        if rows and used + gap + entry <= width - 2 * pad:
            rows[-1].append(i)
            used += gap + entry
        elif len(rows) < MAX_LEGEND_ROWS:
            rows.append([i])
            used = entry
        else:
            # Out of room: the bands stay, the legend stops. A strip that grows
            # past the video is worse than a legend that does not name every
            # behavior in a twenty-step path.
            break

    placed = sum(len(row) for row in rows)
    dropped = len(labels) - placed
    if dropped and rows:
        # One entry of the last row gives way to "+N more", so a legend that
        # stopped cannot be read as a complete key to the bands. It replaces an
        # entry rather than being appended: the row is full by construction, and
        # the marker is far shorter than any behavior sentence, so it fits
        # wherever the entry it displaces did.
        rows[-1] = [*rows[-1][:-1], MORE_SENTINEL]
        dropped += 1

    track_top = pad
    track_height = max(8, int(round(width / 90)))
    legend_top = track_top + track_height + pad + 2
    line_height = int(round(font_size * 1.5))
    height = legend_top + (line_height * len(rows) if rows else 0) + pad
    return BarLayout(
        width=width,
        height=_even(height),
        track_top=track_top,
        track_height=track_height,
        legend_top=legend_top,
        line_height=line_height,
        rows=tuple(tuple(row) for row in rows),
        font_size=font_size,
        dropped=dropped,
    )


def bar_backdrops(
    bands: Sequence[Band],
    spans: Sequence[tuple[float, float]],
    layout: BarLayout,
) -> dict[int | None, np.ndarray]:
    """One rendered bar per "which behavior is live", playhead excluded.

    The strip changes only when the behavior under the playhead changes, so the
    per-frame work is a copy and one line — rather than re-flowing a legend of
    sentences several hundred times.
    """
    font = load_font(layout.font_size)
    base = Image.new("RGB", (layout.width, layout.height), BAR_BG)
    draw = ImageDraw.Draw(base)
    pad = max(6, layout.font_size // 2)
    inner = layout.width - 2 * pad
    track_bottom = layout.track_top + layout.track_height
    draw.rectangle([pad, layout.track_top, pad + inner, track_bottom], fill=TRACK_BG)
    for band, (left, right) in zip(bands, spans, strict=True):
        x0 = pad + left * inner
        # At least a pixel wide: a behavior the policy passed through for one
        # step is still a band, and a zero-width rectangle is a missing one.
        x1 = pad + max(right * inner, left * inner + 1)
        draw.rectangle([x0, layout.track_top, x1, track_bottom], fill=band.color)

    chip = max(6, layout.font_size - 2)
    # Where each legend entry landed, so the "live" variants can redraw one
    # entry's text in white without re-flowing the row.
    slots: dict[int, tuple[float, int]] = {}
    for row_i, row in enumerate(layout.rows):
        x = float(pad)
        y = layout.legend_top + row_i * layout.line_height
        for i in row:
            if i == MORE_SENTINEL:
                # No colour chip: it stands for several behaviors, and a chip
                # would claim one of their colours for all of them.
                draw.text((x, y), f"+{layout.dropped} more", font=font, fill=LABEL_DIM)
                continue
            slots[i] = (x, y)
            draw.rectangle([x, y + 2, x + chip, y + 2 + chip], fill=bands[i].color)
            draw.text((x + chip + 5, y), bands[i].label, font=font, fill=LABEL_DIM)
            x += chip + 5 + _text_width(font, bands[i].label) + layout.font_size * 2

    backdrops: dict[int | None, np.ndarray] = {None: np.asarray(base, dtype=np.uint8).copy()}
    for i, (x, y) in slots.items():
        live = base.copy()
        # Over the dim copy of the same string: same glyphs, same place, so the
        # brighter one covers it exactly.
        ImageDraw.Draw(live).text((x + chip + 5, y), bands[i].label, font=font, fill=LABEL_ON)
        backdrops[i] = np.asarray(live, dtype=np.uint8).copy()
    return backdrops


def live_band(spans: Sequence[tuple[float, float]], position: float) -> int | None:
    """Index of the band *position* (a fraction of the clip) falls in.

    Half-open, except at the very end: the panels close the gaps between bands,
    so one band's end is the next one's start, and testing both ends inclusively
    names the behavior that just finished. The same rule the player's ``_mhl``
    applies, for the same reason.
    """
    for i, (left, right) in enumerate(spans):
        last = i == len(spans) - 1
        if position >= left and (position < right or (last and position <= right)):
            return i
    return None


def bar_frames(
    bands: Sequence[Band],
    layout: BarLayout,
    *,
    total_frames: int | None,
    duration: float,
    fps: float,
    out_fps: float,
) -> Iterator[np.ndarray]:
    """Yield the bar, one frame per output frame, playhead and all."""
    spans = band_spans(bands, total_frames=total_frames, duration=duration, fps=fps)
    backdrops = bar_backdrops(bands, spans, layout)
    pad = max(6, layout.font_size // 2)
    inner = layout.width - 2 * pad
    head = max(2, layout.width // 500)
    top = max(0, layout.track_top - 3)
    bottom = min(layout.height - 1, layout.track_top + layout.track_height + 3)
    n_frames = max(1, int(round(duration * out_fps)))
    for i in range(n_frames):
        position = min(1.0, (i / out_fps) / duration) if duration > 0 else 0.0
        # `.get`, because a band past `MAX_LEGEND_ROWS` has a coloured span on
        # the track but no legend entry to brighten — so there is no "live"
        # backdrop for it and the neutral one is the honest answer. Indexing
        # here raised KeyError the moment the playhead reached the 13th
        # behavior of a long path, and KeyError is not something `export_clip`
        # catches: it left the panel showing a traceback rather than a clip.
        frame = backdrops.get(live_band(spans, position), backdrops[None]).copy()
        x = int(round(pad + position * inner))
        frame[top : bottom + 1, max(0, x - head // 2) : max(1, x + head - head // 2)] = 255
        yield frame


# ── The export ───────────────────────────────────────────────────────────────


def filter_graph(
    view_sizes: Sequence[tuple[int, int]],
    *,
    transforms: Sequence[Sequence[str]],
    orientation: str,
    out_fps: float,
    bar_input: int | None,
) -> str:
    """The ``-filter_complex`` that turns the inputs into one frame.

    A string-returning function of its arguments so the geometry can be checked
    without running ffmpeg: a stack whose members disagree on height by one
    pixel fails at encode time with a message about the filter graph, which is a
    long way from the rounding that caused it.
    """
    parts = []
    for i, ((width, height), filters) in enumerate(zip(view_sizes, transforms, strict=True)):
        # fps first, so every view reaches the stack on one clock: the stack
        # filters take a frame from each input per output frame, and two cameras
        # recorded at different rates would otherwise drift apart.
        chain = [f"fps={out_fps:.6g}", *filters, f"scale={width}:{height}", "setsar=1"]
        parts.append(f"[{i}:v]{','.join(chain)}[v{i}]")
    if len(view_sizes) > 1:
        joined = "".join(f"[v{i}]" for i in range(len(view_sizes)))
        stack = "vstack" if orientation == VERTICAL else "hstack"
        parts.append(f"{joined}{stack}=inputs={len(view_sizes)}[stacked]")
        stacked = "[stacked]"
    else:
        stacked = "[v0]"
    if bar_input is None:
        parts.append(f"{stacked}null[out]")
    else:
        parts.append(f"{stacked}[{bar_input}:v]vstack=inputs=2[out]")
    return ";".join(parts)


def export_clip(
    views: Sequence[tuple[str, str | Path]],
    bands: Sequence[Band] = (),
    *,
    total_frames: int | None = None,
    fps: int = 10,
    orientation: str = HORIZONTAL,
    view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    with_bar: bool = True,
    max_width: int = MAX_WIDTH,
) -> bytes | None:
    """*views*, stacked, with the behavior bar burned in — as MP4 bytes.

    Returns None when there is nothing to encode or ffmpeg cannot do it: the
    caller is a panel offering a download, and a message there beats an
    exception on a page whose subject is the graph above it.

    *fps* is the policy's step rate, used only where a clip has no
    ``total_frames`` to scale the bands against; the output runs at the clips'
    own frame rate.
    """
    exe = ffmpeg_exe()
    paths = [(camera, Path(path)) for camera, path in views if Path(path).is_file()]
    if exe is None or not paths:
        return None

    infos = [probe_clip(path) for _, path in paths]
    if any(info is None for info in infos):
        return None
    transforms = [(view_transforms or {}).get(camera) for camera, _ in paths]
    sizes = [oriented_size(info, xform) for info, xform in zip(infos, transforms, strict=True)]
    view_sizes, (out_w, _out_h) = stack_geometry(sizes, orientation, max_width=max_width)

    # The longest view: a clip that ends early freezes on its last frame in the
    # grid rather than cutting the export short of the annotation it carries.
    duration = max(info.duration for info in infos)
    out_fps = max(1.0, min(60.0, max(info.fps for info in infos)))

    layout = bar_layout([band.label for band in bands], out_w) if (with_bar and bands) else None
    graph = filter_graph(
        view_sizes,
        transforms=[transform_filters(xform) for xform in transforms],
        orientation=orientation,
        out_fps=out_fps,
        bar_input=len(paths) if layout else None,
    )

    args = [exe, "-y", "-loglevel", "error"]
    for _, path in paths:
        args += ["-i", str(path)]
    if layout is not None:
        args += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{layout.width}x{layout.height}",
            "-framerate",
            f"{out_fps:.6g}",
            "-i",
            "-",
        ]
    args += [
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(CRF),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        "-shortest",
    ]

    with tempfile.TemporaryDirectory(prefix="pg-clip-") as tmp:
        out_path = Path(tmp) / "clip.mp4"
        try:
            proc = subprocess.Popen(
                [*args, str(out_path)],
                stdin=subprocess.PIPE if layout is not None else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            return None
        try:
            if layout is not None and proc.stdin is not None:
                stdin = proc.stdin
                frames = bar_frames(
                    bands,
                    layout,
                    total_frames=total_frames,
                    duration=duration,
                    fps=fps,
                    out_fps=out_fps,
                )
                try:
                    for frame in frames:
                        stdin.write(frame.tobytes())
                except BrokenPipeError:
                    # ffmpeg gave up before reading the whole strip; the
                    # returncode below is what decides the outcome either way.
                    pass
                finally:
                    try:
                        stdin.close()
                    except BrokenPipeError:
                        pass
                    # `communicate` flushes `proc.stdin`, and on Python 3.10
                    # flushing an already-closed handle raises ValueError (3.13
                    # tolerates it) — so the deployed container, which is 3.10,
                    # failed here after encoding the clip perfectly well. EOF is
                    # already signalled by the close above, so drop the handle
                    # and let `communicate` just drain stderr and reap.
                    proc.stdin = None
            _, errors = proc.communicate(timeout=600)
        except (subprocess.SubprocessError, OSError, ValueError):
            proc.kill()
            proc.communicate()
            return None
        except BaseException:
            # Anything else — a bug in the frame writer, an interrupt — still
            # has to reap the child: ffmpeg would otherwise sit blocked on an
            # open stdin, writing into a temp dir that is about to be deleted
            # underneath it, once per attempt. Re-raised rather than swallowed:
            # a failure that is not ffmpeg's is not "could not encode".
            proc.kill()
            proc.communicate()
            raise
        if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
            # The caller only gets None, so this is the one place the reason
            # exists. Logged rather than raised: a panel offering a download
            # should say "could not encode", not take the page down.
            _log.warning(
                "clip export failed (ffmpeg rc=%s): %s",
                proc.returncode,
                (errors or b"").decode("utf-8", "replace").strip()[:500] or "no stderr",
            )
            return None
        # Read inside the temp dir's lifetime: the bytes are what the download
        # button serves, so there is nothing to leave on disk afterwards.
        return out_path.read_bytes()


def export_size(
    views: Sequence[tuple[str, str | Path]],
    *,
    orientation: str = HORIZONTAL,
    view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    max_width: int = MAX_WIDTH,
) -> tuple[int, int] | None:
    """The pixel size the video part of :func:`export_clip` would come out at.

    Without encoding anything, for a panel that wants to say what a reader is
    about to wait for. The bar's own height is not included: it depends on how
    many behaviors the legend has to name.
    """
    paths = [(camera, Path(path)) for camera, path in views if Path(path).is_file()]
    if not paths:
        return None
    infos = [probe_clip(path) for _, path in paths]
    if any(info is None for info in infos):
        return None
    sizes = [
        oriented_size(info, (view_transforms or {}).get(camera))
        for info, (camera, _) in zip(infos, paths, strict=True)
    ]
    return stack_geometry(sizes, orientation, max_width=max_width)[1]
