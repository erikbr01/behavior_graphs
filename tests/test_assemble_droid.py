"""Tests for assemble_droid_sync_recordings' destructive --overwrite path.

The script rebuilds episodes/, media/ and policy_embeddings/ in place. Those
are the only copy of the assembled dataset — the raw recordings/ tree can be
re-assembled, but a review clip costs an ffmpeg pass and the concatenated
embeddings cost an extraction — so a failed run must leave them untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "utils"))

from assemble_droid_sync_recordings import main  # noqa: E402

from behavior_graphs.envs.trajectory_dir import RolloutDirectory  # noqa: E402

LAYER = "cond_proj"


def _episode(
    rec: Path,
    name: str,
    *,
    n: int = 4,
    ok: bool = True,
    success: bool = True,
    cameras: tuple[str, ...] = (),
) -> None:
    d = rec / name
    d.mkdir(parents=True)
    if ok:
        (d / "metadata.yaml").write_text(yaml.safe_dump({"length": n, "success": success}))
        np.savez(d / f"embeddings_{LAYER}.npz", embeddings=np.zeros((n, 3), dtype=np.float32))
    else:  # what an interrupted write leaves behind: present but empty
        (d / "metadata.yaml").write_bytes(b"")
        (d / f"embeddings_{LAYER}.npz").write_bytes(b"")
    for camera in cameras:
        (d / f"{camera}.mp4").write_bytes(b"\x00" * 16)


def _assembled(root: Path, marker: str = "previous") -> None:
    for sub in ("episodes", "media", "policy_embeddings"):
        (root / sub).mkdir(parents=True)
        (root / sub / "sentinel.txt").write_text(marker)


def _run(root: Path, *extra: str) -> int:
    return main([str(root), "--layer", LAYER, *extra])


def test_incomplete_episodes_abort_without_touching_prior_output(tmp_path):
    """The regression: --overwrite used to rmtree before this check ran."""
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1")
    _episode(root / "recordings", "ep_2", ok=False)
    _assembled(root)
    with pytest.raises(SystemExit):
        _run(root, "--overwrite")
    for sub in ("episodes", "media", "policy_embeddings"):
        assert (root / sub / "sentinel.txt").read_text() == "previous", sub


def test_all_episodes_incomplete_aborts_without_touching_prior_output(tmp_path):
    """Every episode skipped left np.concatenate([]) to raise after the delete."""
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", ok=False)
    _assembled(root)
    with pytest.raises(SystemExit):
        _run(root, "--overwrite", "--skip-incomplete")
    assert (root / "episodes" / "sentinel.txt").exists()


def test_no_recordings_at_all_aborts_without_touching_prior_output(tmp_path):
    root = tmp_path / "ds"
    (root / "recordings").mkdir(parents=True)
    _assembled(root)
    with pytest.raises(SystemExit):
        _run(root, "--overwrite")
    assert (root / "media" / "sentinel.txt").exists()


def test_missing_output_without_overwrite_still_errors(tmp_path):
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1")
    _assembled(root)
    with pytest.raises(SystemExit):
        _run(root)
    assert (root / "episodes" / "sentinel.txt").exists()


def test_successful_overwrite_replaces_stale_output(tmp_path):
    """Stale clips must not survive: media/ is keyed by index *and* outcome."""
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", success=True)
    _assembled(root)
    (root / "media" / "ep0000_failure.mp4").write_text("stale")
    assert _run(root, "--overwrite", "--no-transcode") == 0
    assert not (root / "media" / "ep0000_failure.mp4").exists()
    assert not (root / "episodes" / "sentinel.txt").exists()
    meta = yaml.safe_load((root / "episodes" / "metadata.yaml").read_text())
    eps = meta["episodes"] if isinstance(meta, dict) else meta
    assert len(eps) == 1


def test_skip_incomplete_assembles_the_usable_subset(tmp_path):
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", n=4)
    _episode(root / "recordings", "ep_2", ok=False)
    _episode(root / "recordings", "ep_3", n=6)
    assert _run(root, "--skip-incomplete", "--no-transcode") == 0
    with np.load(root / "policy_embeddings" / f"{LAYER}.npz") as z:
        assert z[z.files[0]].shape[0] == 10


CAMERAS = ("exterior_image_1_left", "wrist_image")


def test_every_camera_gets_its_own_view_directory(tmp_path):
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", cameras=CAMERAS)
    _episode(root / "recordings", "ep_2", cameras=CAMERAS[:1])  # lost a camera

    assert _run(root, "--no-transcode") == 0

    media = root / "media"
    assert (media / "exterior_image_1_left" / "ep0000_success.mp4").is_file()
    assert (media / "wrist_image" / "ep0000_success.mp4").is_file()
    assert (media / "exterior_image_1_left" / "ep0001_success.mp4").is_file()
    # The episode that only recorded one view keeps only that one.
    assert not (media / "wrist_image" / "ep0001_success.mp4").exists()
    assert not list(media.glob("*.mp4"))


def test_one_named_camera_still_keeps_its_name(tmp_path):
    """The flat form would read back as an unnamed default view.

    Which costs it everything keyed by camera — the `cameras:` geometry in
    configs/data most of all, so a wrist-only export used to render upside down.
    """
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", cameras=CAMERAS)

    assert _run(root, "--no-transcode", "--camera", "wrist_image") == 0

    assert (root / "media" / "wrist_image" / "ep0000_success.mp4").is_file()
    assert not list((root / "media").glob("*.mp4"))
    # And it reads back under that name, not as the default view.
    rollouts = RolloutDirectory(root)
    assert rollouts.camera_names() == ["wrist_image"]
    assert rollouts.video_path(0, "wrist_image") is not None


def test_a_truncated_capture_costs_only_its_own_view(tmp_path):
    root = tmp_path / "ds"
    _episode(root / "recordings", "ep_1", cameras=CAMERAS)
    (root / "recordings" / "ep_1" / "wrist_image.mp4").write_bytes(b"")

    assert _run(root, "--no-transcode") == 0

    assert (root / "media" / "exterior_image_1_left" / "ep0000_success.mp4").is_file()
    assert not (root / "media" / "wrist_image").exists()
