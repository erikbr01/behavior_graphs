"""Generate tasks.json alongside an existing DROID VideoStore without re-converting.

Reads session order and task names from the raw h5 files (no SVO decode).
Identifies corrupt sessions by comparing the SVO frame count (header-only
read via pyzed) against the movement mask — the same check that
convert_droid_data skip_corrupt uses at full decode time.

Usage::

    LD_LIBRARY_PATH=${DATA_ROOT}/zed_sdk_extracted/lib \\
    python scripts/utils/write_droid_tasks_sidecar.py \\
        --raw-data-dir ${DATA_ROOT}/droid_raw \\
        --store-path   ${DATA_ROOT}/droid_converted \\
        --zed-settings-path ${DATA_ROOT}/zed_settings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "robo_dataset"))

log = logging.getLogger(__name__)

_MOVEMENT_KEY = "observation/controller_info/movement_enabled"


def _svo_frame_count(svo_dir: Path, serial: str, zed_settings_path: str | None) -> int | None:
    """Return number of frames in the SVO file via header-only open (no decode)."""
    import pyzed.sl as sl

    svo_path = None
    for f in sorted(svo_dir.iterdir()):
        if f.name.startswith(str(serial)) and ".svo" in f.name:
            svo_path = str(f)
            break
    if svo_path is None:
        return None

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    if zed_settings_path:
        init.optional_settings_path = zed_settings_path
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        cam.close()
        return None
    n = cam.get_svo_number_of_frames()
    cam.close()
    return n


def _is_corrupt(session: Path, serials: list[str], zed_settings_path: str | None) -> bool:
    """Return True if any camera's SVO frame count doesn't cover the movement mask."""
    import h5py

    h5_path = session / "trajectory.h5"
    with h5py.File(h5_path, "r") as f:
        if _MOVEMENT_KEY not in f:
            return False
        mask = f[_MOVEMENT_KEY][:].astype(bool)

    n_expected = int(mask.sum())
    svo_dir = session / "recordings" / "SVO"
    if not svo_dir.is_dir():
        return True

    for serial in serials:
        n_frames = _svo_frame_count(svo_dir, serial, zed_settings_path)
        if n_frames is None:
            return True
        # Count how many mask-True entries fall within the available SVO frames
        n_covered = int(mask[:n_frames].sum())
        if n_covered != n_expected:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--store-path", type=Path, required=True)
    parser.add_argument("--zed-settings-path", default=None)
    parser.add_argument(
        "--camera-source",
        choices=("svo", "none"),
        default="svo",
        help="'none' skips corrupt detection (faster but less accurate).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import h5py

    n_store_eps: int | None = None
    meta_path = args.store_path / "meta.json"
    if meta_path.exists():
        import json as _json

        with open(meta_path) as _f:
            n_store_eps = _json.load(_f)["n_episodes"]
        log.info("VideoStore has %d episodes", n_store_eps)
    else:
        log.warning("store meta.json not found — skipping episode count validation")

    # Discover sessions in the same order DroidStore uses
    sessions: list[Path] = []
    for root, _dirs, files in sorted(os.walk(args.raw_data_dir, followlinks=True)):
        if "trajectory.h5" in files:
            sessions.append(Path(root))
    sessions.sort()

    # Apply skip_empty (same logic as DroidStore)
    valid_sessions: list[Path] = []
    for s in sessions:
        with h5py.File(s / "trajectory.h5", "r") as f:
            if _MOVEMENT_KEY in f:
                T = int(f[_MOVEMENT_KEY][:].astype(bool).sum())
            else:
                act_keys = list(f.get("action", {}).keys())
                T = int(f[f"action/{act_keys[0]}"].shape[0]) if act_keys else 0
        if T == 0:
            log.info("skip empty: %s", s)
            continue
        valid_sessions.append(s)
    log.info("%d sessions after skip_empty", len(valid_sessions))

    # Discover camera serials from the first session
    serials: list[str] = []
    if args.camera_source == "svo":
        first_svo_dir = valid_sessions[0] / "recordings" / "SVO"
        if first_svo_dir.is_dir():
            seen: set[str] = set()
            for f in sorted(first_svo_dir.iterdir()):
                if ".svo" in f.name:
                    serial = f.name.split(".svo")[0]
                    if serial not in seen:
                        seen.add(serial)
                        serials.append(serial)
        log.info("camera serials: %s", serials)

    # Walk sessions, skip corrupt, collect task names
    tasks: dict[str, str] = {}
    ep_idx = 0
    n_corrupt = 0
    for s in valid_sessions:
        if args.camera_source == "svo" and serials:
            if _is_corrupt(s, serials, args.zed_settings_path):
                log.warning("skip corrupt: %s", s)
                n_corrupt += 1
                continue
        with h5py.File(s / "trajectory.h5", "r") as f:
            task = str(f.attrs.get("current_task", ""))
        tasks[str(ep_idx)] = task
        ep_idx += 1

    log.info("collected %d tasks (%d corrupt skipped)", ep_idx, n_corrupt)

    if n_store_eps is not None and ep_idx != n_store_eps:
        log.warning(
            "task count %d != store episode count %d — tasks.json may be misaligned",
            ep_idx,
            n_store_eps,
        )

    out_path = args.store_path / "tasks.json"
    with open(out_path, "w") as fp:
        json.dump(tasks, fp, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
