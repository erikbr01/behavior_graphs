"""Web-delivery layer for rollout videos and thumbnails.

Rollout MP4s are written at capture resolution — the real-robot datasets are
1280×720 at ~12 Mbit/s, which is 10–50 MB for a 25 s episode. The apps play
them back in ~220 px-tall panels, so the source files are 50–200× larger than
anything the browser can show.

Two things happen here:

1. **Proxy transcode.** Each source video gets a small H.264 proxy
   (default 360p, CRF 26, no audio, ``+faststart``) written to a
   content-addressed disk cache. A 35 MB source becomes ~370 KB.
2. **URL delivery.** The proxy is registered with Streamlit's media file
   manager, which serves it from ``/media/<hash>.mp4`` with ``Accept-Ranges``
   support. Previously the players base64-inlined the raw bytes into the
   component HTML, so every single Streamlit rerun re-sent the whole video
   over the websocket. A URL is re-sent as ~40 bytes and the browser caches
   the body.

Everything degrades gracefully: no ffmpeg → serve the source; no Streamlit
runtime (tests, scripts) → fall back to a ``data:`` URI.

Environment variables:
    BEHAVIOR_GRAPHS_MEDIA_CACHE   Proxy cache dir (default ~/.cache/behavior_graphs/media).
    BEHAVIOR_GRAPHS_PROXY_HEIGHT  Proxy height in px (default 360).
    BEHAVIOR_GRAPHS_PROXY_CRF     x264 CRF for the proxy (default 26).
    BEHAVIOR_GRAPHS_PROXY_DISABLE Set to 1 to serve sources untouched.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

#: Bump when the encoder settings change so stale proxies are regenerated.
_CACHE_VERSION = "v1"

#: Sources at or below this size are already smaller than a proxy would be —
#: serve them directly (robomimic/mimicgen clips are 8–150 KB).
_SMALL_SOURCE_BYTES = 400_000

_MIME = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_log = logging.getLogger(__name__)

#: Cache filenames whose ffmpeg pass already failed in this process.
#:
#: Failure has to be remembered, not re-derived from "the artifact is still
#: missing": a truncated source (or a missing ffmpeg) can never produce one, so
#: needs_prep() would keep reporting "not ready" and every rerun would re-enter
#: the caller's spinner and re-run the same doomed ffmpeg pass. In-process only
#: — a restart is the natural way to retry after fixing the input.
_failed: set[str] = set()

#: Paths we have already warned about, so a per-rerun render doesn't log per rerun.
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg, *args)


#: Width of the poster frame shown before a clip starts loading.
POSTER_WIDTH = 480
#: Width of the tiles in the episode thumbnail strip.
STRIP_THUMB_WIDTH = 96


# ── Configuration ────────────────────────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def proxy_height() -> int:
    return _env_int("BEHAVIOR_GRAPHS_PROXY_HEIGHT", 360)


def proxy_crf() -> int:
    return _env_int("BEHAVIOR_GRAPHS_PROXY_CRF", 26)


def proxies_disabled() -> bool:
    return os.environ.get("BEHAVIOR_GRAPHS_PROXY_DISABLE", "").strip() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def cache_dir() -> Path:
    raw = os.environ.get("BEHAVIOR_GRAPHS_MEDIA_CACHE", "").strip()
    if raw:
        d = Path(raw)
    else:
        base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
        d = Path(base) / "behavior_graphs" / "media"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only home (locked-down container): fall back to /tmp so the
        # app still gets proxies, they just don't survive a restart.
        d = Path("/tmp/behavior_graphs_media")
        d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str | None:
    """ffmpeg binary, preferring the one imageio-ffmpeg ships."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ── Cache keys ───────────────────────────────────────────────────────────────


def _stat_key(src: Path) -> str:
    """Identity of the *content* at ``src`` — path, mtime and size."""
    try:
        st = src.stat()
        return f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return str(src)


def _digest(*parts: object) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:20]


def norm_transform(transform: Mapping[str, object] | None) -> tuple[int, bool]:
    """A camera's declared geometry as ``(rotate degrees clockwise, mirror)``.

    Public because two modules apply it: the ffmpeg filter chain below and
    :func:`behavior_graphs.viz.frames._orient`'s PIL path, which have to agree on
    both the normalization and the mirror-before-rotation order.

    Defensive rather than trusting: this is the last place the value is used, and
    anything unexpected reaching it (a hand-edited config, an older caller) is
    read as "no transform" instead of being handed to ffmpeg or folded into a
    cache key. Validation with the messages lives in
    :func:`behavior_graphs.viz.streamlit.data.camera_transforms`.
    """
    if not transform:
        return 0, False
    rotate = transform.get("rotate", 0)
    rotate = rotate % 360 if isinstance(rotate, int) and not isinstance(rotate, bool) else 0
    return (rotate if rotate in (90, 180, 270) else 0), bool(transform.get("mirror"))


def _xform_key(transform: Mapping[str, object] | None) -> tuple:
    """Cache-key parts for *transform* — empty when there is nothing to do.

    Empty rather than a placeholder so an untransformed clip keeps the key it
    had before transforms existed: the deployed cache is content-addressed and
    append-only (see :func:`prune`), and widening every key would strand all of
    it at once.
    """
    rotate, mirror = norm_transform(transform)
    if not rotate and not mirror:
        return ()
    return ("rot", rotate, "mir", int(mirror))


def transform_filters(transform: Mapping[str, object] | None) -> list[str]:
    """ffmpeg ``-vf`` entries for *transform*, mirror before rotation.

    A half turn is ``hflip,vflip`` rather than two transposes: same result, no
    dimension swap for the scaler to chase.

    Public alongside :func:`norm_transform` because a third consumer now applies
    the same geometry: the clip export
    (:mod:`behavior_graphs.viz.clip_figure`) builds its own filter chain, and a
    second transcription of "mirror, then rotate" is how an exported clip ends
    up a mirror image of the one on screen.
    """
    rotate, mirror = norm_transform(transform)
    filters = ["hflip"] if mirror else []
    if rotate == 90:
        filters.append("transpose=1")
    elif rotate == 270:
        filters.append("transpose=2")
    elif rotate == 180:
        filters += ["hflip", "vflip"]
    return filters


def proxy_path(
    src: Path,
    *,
    height: int | None = None,
    crf: int | None = None,
    transform: Mapping[str, object] | None = None,
) -> Path:
    h = proxy_height() if height is None else height
    c = proxy_crf() if crf is None else crf
    return (
        cache_dir() / f"{_digest(_CACHE_VERSION, _stat_key(src), h, c, *_xform_key(transform))}.mp4"
    )


def _effective_transform(
    transform: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """The transform that actually reaches the browser, not the configured one.

    With ``BEHAVIOR_GRAPHS_PROXY_DISABLE`` set nothing runs ffmpeg on the video, so
    it is served exactly as recorded — and a poster or strip tile decoded from
    the clip has to match it. Applied here, in the key, so the tile is *also*
    cached under the orientation actually produced: keying it as rotated and
    building it unrotated would hand back a stale upright tile once proxies came
    back on. Without this the picture stayed upright while its own poster turned
    180°, which is the disagreement baking the transform in exists to avoid.
    """
    if transform and transform_filters(transform) and proxies_disabled():
        return None
    return transform


def thumb_path(
    src: Path, *, width: int = 160, transform: Mapping[str, object] | None = None
) -> Path:
    transform = _effective_transform(transform)
    key = _digest(_CACHE_VERSION, _stat_key(src), "thumb", width, *_xform_key(transform))
    return cache_dir() / f"{key}.jpg"


# ── Transcoding ──────────────────────────────────────────────────────────────


def _run_ffmpeg(args: Sequence[str], dst: Path) -> bool:
    """Run ffmpeg writing to a temp file, then rename into place.

    The temp name carries a uuid, not just the pid: two threads of one
    Streamlit process (or the survey and demo containers, which both run
    streamlit as pid 1 and share the media_cache volume) can transcode the
    same source concurrently, and a shared temp path means two ``ffmpeg -y``
    runs interleaving writes into one file. The rename is atomic, so with a
    per-call temp the loser simply overwrites the winner's identical output
    instead of either of them publishing a spliced file.
    """
    exe = ffmpeg_exe()
    if exe is None:
        return False
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{dst.suffix}")
    try:
        proc = subprocess.run(
            [exe, "-y", "-loglevel", "error", *args, str(tmp)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            return False
        os.replace(tmp, dst)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _proxy_wanted(src: Path) -> bool:
    """True when ``src`` is big enough (and proxies enabled) to be worth a pass."""
    if proxies_disabled():
        return False
    try:
        return src.stat().st_size > _SMALL_SOURCE_BYTES
    except OSError:
        return False


def _pass_wanted(src: Path, transform: Mapping[str, object] | None) -> bool:
    """True when this source+orientation earns an ffmpeg pass.

    One predicate for :func:`ensure_proxy`, :func:`needs_prep` and
    :func:`prewarm`: a transform is correctness rather than an optimisation, so
    it earns a pass a clip's size would not — unless proxies are off altogether,
    which is an explicit operator choice this is not entitled to override. Three
    copies of that rule would let the spinner's accounting drift from what the
    build actually does.
    """
    return _proxy_wanted(src) or (bool(transform_filters(transform)) and not proxies_disabled())


def ensure_proxy(
    src: Path,
    *,
    height: int | None = None,
    crf: int | None = None,
    transform: Mapping[str, object] | None = None,
) -> Path:
    """Path to a web-sized proxy of ``src`` (the source itself if not worth it).

    A *transform* is baked in here rather than applied as a CSS transform on the
    ``<video>``: rotating that element rotates its native control bar with it,
    and the poster and thumbnail tiles decoded from the clip would still be the
    wrong way up. Baked, one orientation reaches every consumer, and the cache
    key below keeps the two variants apart.
    """
    src = Path(src)
    filters = transform_filters(transform)
    if not src.is_file():
        return src
    if not _pass_wanted(src, transform):
        if filters:  # only reachable with proxies disabled: nothing will rotate it
            _warn_once(
                f"unrotated:{src}",
                "BEHAVIOR_GRAPHS_PROXY_DISABLE is set, so %s is served as recorded — "
                "its configured rotation needs the ffmpeg pass",
                src,
            )
        return src

    dst = proxy_path(src, height=height, crf=crf, transform=transform)
    if dst.exists():
        return dst
    if dst.name in _failed:
        return src

    h = proxy_height() if height is None else height
    c = proxy_crf() if crf is None else crf
    ok = _run_ffmpeg(
        [
            "-i",
            str(src),
            # Never upscale, and keep width even (yuv420p requires it). Rotation
            # first: after a quarter turn it is the new height that has to fit.
            "-vf",
            ",".join([*filters, f"scale=-2:'min({h},ih)':flags=bicubic"]),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(c),
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
        ],
        dst,
    )
    if not ok:
        _failed.add(dst.name)
        _warn_once(
            f"proxy:{dst.name}",
            "could not transcode %s (ffmpeg %s); serving the source untranscoded",
            src,
            "missing" if ffmpeg_exe() is None else "failed",
        )
        return src
    return dst


def needs_prep(
    src: Path,
    *,
    thumb_widths: Sequence[int] = (POSTER_WIDTH,),
    transform: Mapping[str, object] | None = None,
) -> bool:
    """True when ``src`` still needs an ffmpeg pass that could actually succeed.

    *thumb_widths* must match what the caller will actually build (the same
    argument :func:`prewarm` takes) — asking about a width nobody requests
    reports work that will never be done.

    Artifacts whose pass already failed (:data:`_failed`) and the no-ffmpeg
    install both report False: the callers put this behind a spinner, so
    answering True for something unproducible turns every rerun into a fresh
    doomed transcode.
    """
    src = Path(src)
    if not src.is_file() or ffmpeg_exe() is None:
        return False
    if _pass_wanted(src, transform):
        proxy = proxy_path(src, transform=transform)
        if not proxy.exists() and proxy.name not in _failed:
            return True
    return any(_thumb_pending(src, w, transform=transform) for w in thumb_widths)


def _thumb_pending(
    src: Path,
    width: int,
    thumb: Path | None = None,
    *,
    transform: Mapping[str, object] | None = None,
) -> bool:
    """True when a tile at *width* is neither cached nor known-unproducible.

    The one predicate for "does this still need an ffmpeg pass" — pass *thumb*
    when the caller already derived the path, so a batch doesn't stat and hash
    it twice.

    The ``is_file`` check matters: ``ensure_thumb`` returns None for a missing
    source *without* recording a failure, so a clip deleted under a still-warm
    episode index would otherwise be reported pending on every rerun forever.
    """
    if not Path(src).is_file():
        return False
    thumb = thumb if thumb is not None else thumb_path(src, width=width, transform=transform)
    return not thumb.exists() and thumb.name not in _failed


def pending_thumbs(
    sources: Iterable[Path],
    *,
    width: int = STRIP_THUMB_WIDTH,
    transform: Mapping[str, object] | None = None,
) -> list[Path]:
    """The sources still needing a tile at *width*.

    Exposed so callers that show a progress spinner can size it without
    reimplementing the predicate — in particular without forgetting the
    :data:`_failed` check, which is what stops a permanently unproducible tile
    from being counted (and re-attempted) on every single rerun.
    """
    return [s for s in map(Path, sources) if _thumb_pending(s, width, transform=transform)]


def ensure_thumb(
    src: Path, *, width: int = 160, transform: Mapping[str, object] | None = None
) -> Path | None:
    """Small JPEG poster for ``src`` — its first frame, *transform* applied."""
    src = Path(src)
    if not src.is_file():
        return None
    # Matched to what the <video> will actually be served (see
    # _effective_transform), so the poster is never the other way up from it.
    transform = _effective_transform(transform)
    dst = thumb_path(src, width=width, transform=transform)
    if dst.exists():
        return dst
    if dst.name in _failed:
        return None

    # Decode from the proxy when one exists — it is ~100× smaller, so the
    # seek-and-decode is near-instant. Its own transform is already baked in, so
    # only a decode straight from the source still has to apply one.
    filters = []
    source = proxy_path(src, transform=transform)
    if not source.exists():
        source, filters = src, transform_filters(transform)

    args = [
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        ",".join([*filters, f"scale={width}:-2"]),
        "-q:v",
        "6",
    ]
    if _run_ffmpeg(args, dst):
        return dst
    _failed.add(dst.name)
    _warn_once(f"thumb:{dst.name}", "could not extract a thumbnail from %s", src)
    return None


def prewarm(
    sources: Iterable[Path | tuple[Path, Mapping[str, object] | None]],
    *,
    thumb_widths: Sequence[int] = (POSTER_WIDTH,),
    workers: int = 0,
    on_done=None,
) -> tuple[int, int]:
    """Build proxies (and posters) for many videos in parallel.

    ffmpeg is an external process, so threads scale fine here. Returns
    ``(n_done, n_total)``.

    *sources* are paths, or ``(path, transform)`` pairs when they don't all want
    the same orientation. Per source rather than one transform for the batch: a
    panel's views differ (a DROID episode's wrist camera is rotated and its
    exterior one is not), and a batch-wide transform made every caller split into
    one thread pool per camera — running sequentially the passes it exists to
    overlap.
    """
    items = [(Path(s[0]), s[1]) if isinstance(s, tuple) else (Path(s), None) for s in sources]
    if not items:
        return (0, 0)
    n_workers = workers or min(8, (os.cpu_count() or 4))

    def _one(item: tuple[Path, Mapping[str, object] | None]) -> bool:
        s, transform = item
        # Report what this call produced rather than re-deriving it from
        # needs_prep(), whose thumb check is tied to the poster width.
        # `==`, not `is`: ensure_proxy rebinds src = Path(src) internally, so
        # it never returns the caller's object and an identity test here is
        # unconditionally True — which reported a clean sweep for a run that
        # transcoded nothing at all.
        ok = not _pass_wanted(s, transform) or ensure_proxy(s, transform=transform) != s
        for w in thumb_widths:
            ok = ensure_thumb(s, width=w, transform=transform) is not None and ok
        if on_done is not None:
            on_done(s)
        return ok

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_one, items))
    return (sum(1 for r in results if r), len(items))


# ── Streamlit URL delivery ───────────────────────────────────────────────────


def _streamlit_url(path: Path, coordinates: str) -> str | None:
    """Register ``path`` with Streamlit's media manager and return its URL."""
    try:
        from streamlit.runtime import exists as _rt_exists
        from streamlit.runtime import get_instance
    except Exception:
        return None
    try:
        if not _rt_exists():
            return None
        mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
        return get_instance().media_file_mgr.add(str(path), mime, coordinates)
    except Exception:
        return None


def _data_uri(path: Path) -> str:
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def video_src(
    src: Path,
    *,
    coordinates: str,
    height: int | None = None,
    transform: Mapping[str, object] | None = None,
) -> str | None:
    """A ``<video src=…>`` value for ``src`` — a URL whenever possible.

    ``coordinates`` identifies the calling element so the media manager can
    release the previous file when the same player switches episodes.

    None when the clip cannot be served: the file is gone (a cached episode
    index can outlive a deleted rollout), or there is no Streamlit runtime to
    serve a URL and the file is too big to inline. Callers render their own
    "no video" state; the alternative for the second case is a multi-megabyte
    base64 payload over the websocket, which is the regression this module
    exists to prevent.
    """
    src = Path(src)
    if not src.is_file():
        _warn_once(f"missing:{src}", "no video at %s — nothing to serve", src)
        return None
    proxy = ensure_proxy(src, height=height, transform=transform)
    url = _streamlit_url(proxy, coordinates)
    if url:
        if proxy == src and _proxy_wanted(src):
            # Streamlit's media manager reads the whole file into memory and
            # keeps it for the session, so an untranscoded source is a real
            # memory cost, not just a bandwidth one.
            _warn_once(
                f"unproxied:{src}",
                "serving %s untranscoded (%.1f MB) — Streamlit holds it in memory "
                "for the session; check ffmpeg and BEHAVIOR_GRAPHS_PROXY_DISABLE",
                src,
                src.stat().st_size / 1e6,
            )
        return url
    try:
        size = proxy.stat().st_size
    except OSError:
        return None
    if size > _SMALL_SOURCE_BYTES:
        _warn_once(
            f"noinline:{proxy}",
            "no Streamlit runtime to serve %s (%.1f MB) and it is too big to "
            "inline as a data: URI — not rendering it",
            proxy,
            size / 1e6,
        )
        return None
    return _data_uri(proxy)


def poster_src(
    src: Path,
    *,
    coordinates: str,
    width: int = POSTER_WIDTH,
    transform: Mapping[str, object] | None = None,
) -> str | None:
    """A ``<video poster=…>`` value — shows instantly while the video loads."""
    thumb = ensure_thumb(Path(src), width=width, transform=transform)
    if thumb is None:
        return None
    url = _streamlit_url(thumb, coordinates)
    return url or _data_uri(thumb)


def thumb_srcs(
    sources: Sequence[Path | tuple[Path, Mapping[str, object] | None]],
    *,
    coordinates_prefix: str,
    width: int = STRIP_THUMB_WIDTH,
    on_pending=None,
) -> list[str | None]:
    """Poster URLs for a list of videos, generated in parallel.

    Used by the episode thumbnail strip, where a single selection can cover
    150+ episodes; doing this serially cost ~0.1 s per episode.

    *sources* are paths, or ``(path, transform)`` pairs. Per tile rather than one
    transform for the whole strip: a strip is *usually* many episodes of the same
    camera, but each tile is its own episode's first view, and a collection that
    lost a camera partway through (or merged sessions naming the wrist camera
    differently) has episodes whose first view is a different camera — those tiles
    would be built with a neighbour's orientation and come out upside down.
    """
    items = [(Path(s[0]), s[1]) if isinstance(s, tuple) else (Path(s), None) for s in sources]
    # One thumb_path() per source: each is a stat + a sha1, and this runs per
    # tile per rerun on a strip that can be 180 episodes long. *on_pending* lets
    # the caller show a sized progress message without a second pass of its own.
    paths = [thumb_path(s, width=width, transform=xf) for s, xf in items]
    missing = [(s, xf) for (s, xf), p in zip(items, paths) if _thumb_pending(s, width, p)]
    if missing:
        if on_pending is not None:
            on_pending(len(missing))
        n_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(lambda it: ensure_thumb(it[0], width=width, transform=it[1]), missing))
    return [
        _streamlit_url(p, f"{coordinates_prefix}:{i}") if p.exists() else None
        for i, p in enumerate(paths)
    ]


def prune(max_bytes: int) -> tuple[int, int]:
    """Trim the cache to *max_bytes*, least-recently-touched first.

    Returns ``(n_deleted, bytes_freed)``.

    This is the other half of a content-addressed cache. Every key folds in the
    source's mtime and size plus the encoder settings, so the cache is
    append-only by construction: re-collecting a rollout, changing
    ``BEHAVIOR_GRAPHS_PROXY_HEIGHT``, or any change to the key scheme strands
    every previous entry, and nothing else ever reclaims them. Streamlit's own
    caches don't reach in here — these are plain files under
    :func:`cache_dir`.

    Ordered by the newer of atime and mtime: atime is the real "least recently
    used" signal, but ``relatime`` mounts update it lazily, in which case mtime
    (creation time, for a content-addressed file) is the honest fallback.
    """
    try:
        entries = [(f, f.stat()) for f in cache_dir().iterdir() if f.is_file()]
    except OSError:
        return (0, 0)
    total = sum(st.st_size for _, st in entries)
    if total <= max_bytes:
        return (0, 0)

    entries.sort(key=lambda e: max(e[1].st_atime, e[1].st_mtime))
    freed = 0
    n = 0
    for f, st in entries:
        if total - freed <= max_bytes:
            break
        try:
            f.unlink()
        except OSError:  # pragma: no cover - racing another prune/reader
            continue
        freed += st.st_size
        n += 1
    return (n, freed)


def cache_stats() -> tuple[int, int]:
    """``(n_files, total_bytes)`` currently in the proxy cache."""
    n = 0
    total = 0
    try:
        for p in cache_dir().iterdir():
            if p.is_file():
                n += 1
                total += p.stat().st_size
    except OSError:
        pass
    return n, total
