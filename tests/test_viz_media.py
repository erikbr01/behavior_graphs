"""Tests for the video proxy layer and the shared episode index.

The proxy tests need ffmpeg; they skip cleanly without it. The episode-index
tests are pure Python and check the new lookups against the straightforward
scan they replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavior_graphs.viz import media
from behavior_graphs.viz.streamlit.episode_index import get_episode_index, signature
from script_loader import load_script

# ── Episode index ────────────────────────────────────────────────────────────


def _synthetic():
    """Two episodes: 0 succeeds A→B→C, 1 fails A→B→A."""
    labels, metadata = [], []
    plan = {
        0: ([0, 0, 1, 1, 2], True),
        1: ([0, 1, 1, 0], False),
    }
    for ep, (seq, success) in plan.items():
        for t, lab in enumerate(seq):
            labels.append(lab)
            metadata.append(
                {
                    "rollout_idx": ep,
                    "window_start": t,
                    "window_end": t + 2,
                    "success": success,
                }
            )
    return np.array(labels, dtype=np.int64), metadata


def test_episode_index_structure():
    labels, metadata = _synthetic()
    idx = get_episode_index(labels, metadata)

    assert idx.ep_key == "rollout_idx"
    assert idx.success == {0: True, 1: False}
    assert idx.length == {0: 5, 1: 4}
    assert idx.rle[0] == [(0, 0), (2, 1), (4, 2)]
    assert idx.rle[1] == [(0, 0), (1, 1), (3, 0)]


def test_episodes_for_node_matches_naive_scan():
    labels, metadata = _synthetic()
    idx = get_episode_index(labels, metadata)

    for node_id in (0, 1, 2):
        naive: dict[int, list[int]] = {}
        for i, m in enumerate(metadata):
            if int(labels[i]) == node_id:
                naive.setdefault(m["rollout_idx"], []).append(m["window_start"])
        expected = sorted((ep, min(ts), max(ts)) for ep, ts in naive.items())
        assert idx.episodes_for_node(node_id) == expected


def test_node_and_edge_stats():
    labels, metadata = _synthetic()
    idx = get_episode_index(labels, metadata)

    stats = {r["ep"]: r for r in idx.node_stats(0)}
    assert stats[0]["count"] == 2 and stats[0]["duration"] == 2
    # Episode 1 revisits cluster 0 at t=3, so the span covers the gap.
    assert stats[1]["count"] == 2 and stats[1]["duration"] == 4
    assert stats[1]["success"] is False

    edges = idx.edge_stats(0, 1)
    assert {e["ep"] for e in edges} == {0, 1}
    assert idx.edge_stats(1, 2) == [e for e in idx.edge_stats(1, 2) if e["ep"] == 0]
    assert idx.edge_stats(2, 0) == []


def test_per_frame_labels_covers_window_extent():
    labels, metadata = _synthetic()
    idx = get_episode_index(labels, metadata)

    pfl = idx.per_frame_labels(0, 6)
    # windows are [t, t+2); last-write-wins on sorted start.
    assert pfl.tolist() == [0, 0, 1, 1, 2, 2]
    assert idx.per_frame_labels(0, 0).size == 0


def test_signature_tracks_labels():
    labels, metadata = _synthetic()
    other = labels.copy()
    other[0] = 2
    assert signature(labels, metadata) == signature(labels.copy(), metadata)
    assert signature(labels, metadata) != signature(other, metadata)


# ── Media proxies ────────────────────────────────────────────────────────────


needs_ffmpeg = pytest.mark.skipif(media.ffmpeg_exe() is None, reason="ffmpeg not available")


@pytest.fixture()
def sample_video(tmp_path):
    """A lossless 720p test pattern — comfortably past the proxy threshold."""
    import subprocess

    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            media.ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30:duration=3",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # encoding a 3 s test pattern; a wedged ffmpeg should fail the test, not
        # hang the run (src/behavior_graphs/viz/media.py bounds its own calls too)
        timeout=120,
    )
    assert out.stat().st_size > 400_000, "fixture must exceed the small-source cutoff"
    return out


@needs_ffmpeg
def test_proxy_is_smaller_and_cached(tmp_path, sample_video, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()

    assert media.needs_prep(sample_video)
    proxy = media.ensure_proxy(sample_video, height=240, crf=30)
    assert proxy != sample_video
    assert proxy.exists()
    assert proxy.stat().st_size < sample_video.stat().st_size

    # Second call is a cache hit — same path, no new file.
    n_before, _ = media.cache_stats()
    assert media.ensure_proxy(sample_video, height=240, crf=30) == proxy
    assert media.cache_stats()[0] == n_before


@needs_ffmpeg
def test_small_sources_are_served_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()

    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"\x00" * 1024)
    assert media.ensure_proxy(tiny) == tiny


@needs_ffmpeg
def test_thumbnail_generation(tmp_path, sample_video, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()

    thumb = media.ensure_thumb(sample_video, width=96)
    assert thumb is not None and thumb.exists()
    assert thumb.stat().st_size > 0
    assert thumb.suffix == ".jpg"


@needs_ffmpeg
def test_proxy_key_changes_with_settings(tmp_path, sample_video, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()

    assert media.proxy_path(sample_video, height=240) != media.proxy_path(sample_video, height=360)
    assert media.proxy_path(sample_video, crf=20) != media.proxy_path(sample_video, crf=30)


def test_proxies_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
    media.cache_dir.cache_clear()

    src = tmp_path / "big.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.ensure_proxy(src) == src


def test_video_src_falls_back_to_data_uri_without_runtime(tmp_path, monkeypatch):
    """Outside a Streamlit runtime there is no /media endpoint to point at."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
    media.cache_dir.cache_clear()

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"abcdef")
    assert media.video_src(src, coordinates="t").startswith("data:video/mp4;base64,")


def test_video_src_returns_none_for_a_missing_clip(tmp_path, monkeypatch):
    """A cached episode index can outlive the file it points at."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()

    assert media.video_src(tmp_path / "gone.mp4", coordinates="t") is None


def test_video_src_refuses_to_inline_a_large_clip(tmp_path, monkeypatch):
    """Base64-inlining a full-size clip is the regression this module prevents."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
    media.cache_dir.cache_clear()

    src = tmp_path / "big.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.video_src(src, coordinates="t") is None


def test_needs_prep_gives_up_on_a_failed_pass(tmp_path, monkeypatch):
    """Otherwise every rerun re-enters the spinner for an unproducible file."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(media, "_failed", set())
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "/usr/bin/true")
    media.cache_dir.cache_clear()

    src = tmp_path / "truncated.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.needs_prep(src) is True

    media._failed.add(media.proxy_path(src).name)
    media._failed.add(media.thumb_path(src, width=480).name)
    assert media.needs_prep(src) is False


def test_needs_prep_is_false_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: None)
    media.cache_dir.cache_clear()

    src = tmp_path / "big.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.needs_prep(src) is False


def test_prewarm_counts_what_it_did_not_what_needs_prep_says(tmp_path, monkeypatch):
    """--no-thumbs asks for no poster, so it must not be counted as a failure."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
    media.cache_dir.cache_clear()

    src = tmp_path / "big.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.prewarm([src], thumb_widths=()) == (1, 1)


@needs_ffmpeg
def test_prewarm_without_thumbs_reports_success(tmp_path, sample_video, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", raising=False)
    monkeypatch.setattr(media, "_failed", set())
    media.cache_dir.cache_clear()

    assert media.prewarm([sample_video], thumb_widths=()) == (1, 1)
    assert media.proxy_path(sample_video).exists()


# ── Clip-path resolution ─────────────────────────────────────────────────────


def _mp4_index():
    episodes = [
        {"index": 0, "path": "media/ep0000_success.mp4", "success": True},
        {"index": 3, "path": "media/ep0003_failure.mp4", "success": False},
    ]
    return {"episodes": episodes, "by_index": {e["index"]: e for e in episodes}}


def test_find_mp4_episode_requires_the_by_index_map():
    """One contract: every index comes from data.py, so by_index is always there."""
    from behavior_graphs.viz.streamlit.data import empty_mp4_index, find_mp4_episode

    idx = _mp4_index()
    assert find_mp4_episode(3, idx)["path"].endswith("ep0003_failure.mp4")
    assert find_mp4_episode(9, idx) is None
    assert find_mp4_episode(0, empty_mp4_index()) is None

    # A hand-built index is a programming error, not a slow path to tolerate.
    with pytest.raises(KeyError):
        find_mp4_episode(3, {"episodes": idx["episodes"]})


def test_clip_paths_skips_episodes_without_video(tmp_path):
    from behavior_graphs.viz.streamlit.clips import clip_paths

    idx = _mp4_index()
    assert clip_paths([0, 9, 3], idx, tmp_path) == [
        tmp_path / "media/ep0000_success.mp4",
        tmp_path / "media/ep0003_failure.mp4",
    ]
    assert clip_paths([0, 3], idx, None) == []
    assert clip_paths([], idx, tmp_path) == []


def test_legacy_mp4_index_reports_absence_as_none(tmp_path, monkeypatch):
    """No videos -> no media dir, rather than a placeholder path to join onto."""
    from behavior_graphs.viz.streamlit.data import legacy_mp4_index

    assert legacy_mp4_index(None) == (None, {"episodes": [], "by_index": {}, "cameras": []})

    # A directory that is not a rollout dir at all.
    mp4_dir, index = legacy_mp4_index(tmp_path)
    assert mp4_dir is None and index["episodes"] == []


def test_pending_thumbs_skips_known_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(media, "_failed", set())
    media.cache_dir.cache_clear()

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.pending_thumbs([src], width=96) == [src]

    media._failed.add(media.thumb_path(src, width=96).name)
    assert media.pending_thumbs([src], width=96) == []


def test_needs_prep_honours_the_widths_asked_for(tmp_path, monkeypatch):
    """The strip tile and the poster are separate artifacts."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
    monkeypatch.setattr(media, "_failed", set())
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "/usr/bin/true")
    media.cache_dir.cache_clear()

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 2_000_000)
    assert media.needs_prep(src, thumb_widths=(96,)) is True
    assert media.needs_prep(src, thumb_widths=()) is False


def test_paths_token_is_content_derived():
    from behavior_graphs.viz.streamlit.data import paths_token

    assert paths_token(("a", "b")) == paths_token(("a", "b"))
    assert paths_token(("a", "b")) != paths_token(("b", "a"))
    # Not just a concatenation — separators can't be forged by a path.
    assert paths_token(("a", "b")) != paths_token(("ab",))


def test_thumb_srcs_reports_pending_once(tmp_path, monkeypatch):
    """The caller's spinner is sized from thumb_srcs' own pass, not a second one."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(media, "_failed", set())
    media.cache_dir.cache_clear()

    srcs = []
    for i in range(3):
        p = tmp_path / f"clip{i}.mp4"
        p.write_bytes(b"\x00" * 1024)
        srcs.append(p)

    seen: list[int] = []
    monkeypatch.setattr(media, "ensure_thumb", lambda s, width=0, transform=None: None)
    urls = media.thumb_srcs(srcs, coordinates_prefix="t", on_pending=seen.append)
    assert seen == [3]  # called once, with the real pending count
    assert len(urls) == len(srcs)  # and the result still aligns positionally


def test_episode_clips_returns_entry_and_path(tmp_path):
    from behavior_graphs.viz.streamlit.clips import episode_clips

    idx = _mp4_index()
    got = episode_clips([3, 9, 0], idx, tmp_path)
    assert [ep for ep, _, _ in got] == [3, 0]  # 9 has no video; order preserved
    assert got[0][1]["index"] == 3
    assert got[0][2] == tmp_path / "media/ep0003_failure.mp4"
    assert episode_clips([0], idx, None) == []


def test_prewarm_reports_failure_when_no_proxy_was_produced(tmp_path, monkeypatch):
    """`is` against ensure_proxy's return is always true — it rebinds Path(src).

    A structurally-true success flag reported `done: N/N` for a run that
    transcoded nothing, so this asserts the failing case, not the happy one.
    """
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", raising=False)
    monkeypatch.setattr(media, "_failed", set())
    media.cache_dir.cache_clear()

    # Big enough to want a proxy, but not decodable — ffmpeg will fail.
    src = tmp_path / "notavideo.mp4"
    src.write_bytes(b"\x00" * 500_000)

    assert media.prewarm([src], thumb_widths=()) == (0, 1)
    assert not media.proxy_path(src).exists()


def test_thumb_is_not_pending_for_a_missing_source(tmp_path, monkeypatch):
    """A deleted clip must not be reported pending on every rerun forever."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(media, "_failed", set())
    media.cache_dir.cache_clear()

    gone = tmp_path / "deleted.mp4"
    assert media.pending_thumbs([gone], width=96) == []

    seen: list[int] = []
    media.thumb_srcs([gone], coordinates_prefix="t", on_pending=seen.append)
    assert seen == []  # no spinner, no thread pool, no doomed ffmpeg pass


def test_prune_trims_to_the_budget_oldest_first(tmp_path, monkeypatch):
    """The content-addressed cache is append-only; prune is what reclaims it."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
    media.cache_dir.cache_clear()
    cache = media.cache_dir()

    import os

    for i in range(4):  # 4 × 1000 bytes, oldest first
        f = cache / f"{i}.mp4"
        f.write_bytes(b"\x00" * 1000)
        os.utime(f, (1_000_000 + i, 1_000_000 + i))

    assert media.cache_stats() == (4, 4000)

    # Under budget: a no-op.
    assert media.prune(4000) == (0, 0)
    assert media.cache_stats()[0] == 4

    # Trim to 2 files' worth; the two oldest go.
    assert media.prune(2000) == (2, 2000)
    assert sorted(p.name for p in cache.iterdir()) == ["2.mp4", "3.mp4"]


def test_prune_on_an_empty_or_missing_cache_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "nope"))
    media.cache_dir.cache_clear()
    assert media.prune(0) == (0, 0)


prewarm = load_script("prewarm_media")


def _dataset(root, name, *, cameras=(), recordings=1):
    """A rollout dir with per-camera media clips and the raw captures beside them."""
    from pathlib import Path

    ds: Path = root / name
    for camera in cameras or ("",):
        d = ds / "media" / camera if camera else ds / "media"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ep0000_success.mp4").write_bytes(b"\x00" * 10)
    for i in range(recordings):
        raw = ds / "recordings" / f"ep_{i}"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "wrist_image_left.mp4").write_bytes(b"\x00" * 10)
    return ds


def test_prewarm_all_skips_the_raw_captures(tmp_path):
    """`--all` warms what the apps serve; recordings/ is what the collector wrote."""
    _dataset(tmp_path, "droid_a", cameras=("exterior_image_1_left", "wrist_image_left"))
    _dataset(tmp_path, "sim_b")

    found = prewarm._discover(tmp_path, True)

    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        "droid_a/media/exterior_image_1_left/ep0000_success.mp4",
        "droid_a/media/wrist_image_left/ep0000_success.mp4",
        "sim_b/media/ep0000_success.mp4",
    ], "raw recordings/ captures are not served and must not be warmed"


def test_prewarm_one_dataset_covers_every_camera(tmp_path):
    ds = _dataset(tmp_path, "droid_a", cameras=("exterior_image_1_left", "wrist_image_left"))

    found = prewarm._discover(ds, False)

    assert [p.parent.name for p in found] == ["exterior_image_1_left", "wrist_image_left"]


def test_prewarm_rescues_a_layout_with_no_media_dir(tmp_path):
    """The fallback is what makes a nonstandard dir warm anything at all."""
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "clip.mp4").write_bytes(b"\x00" * 10)

    assert prewarm._discover(tmp_path, True) == [tmp_path / "loose" / "clip.mp4"]
