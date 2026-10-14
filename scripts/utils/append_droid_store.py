"""Append one VideoStore's episodes onto another, in place, without touching
any existing episode.

`convert_droid_data.py` is not incremental -- it refuses a non-empty
`--output` (see docs/droid_workflow.md). To add a new instruction variant (or
any additional sessions) to an existing store without re-converting or
duplicating what's already there: convert the new sessions into their own
small store first, then append that store onto the existing one with this
script.

Copies `videos/<key>/episode_*.mp4` and `data/<local_ep>/<lowdim_key>` from
`--src` into `--dst` at new, contiguous global episode indices starting at
dst's current `n_episodes` (from `meta.json`) -- it never overwrites an
existing dst episode file. Merges `tasks.json` (adds src's task labels under
the new indices) and bumps `n_episodes` in `meta.json`.

Everything is validated before the first byte is copied, and any failure
mid-copy unwinds what this run created -- there is no partial append and no
silently-degraded store. Specifically, this refuses to run unless:

* both stores use the per-episode layout `convert_droid_data.py` writes
  today (the flat `data/<key>/` + `episode_ends.npy` layout is deprecated;
  its readers derive `n_episodes` from `episode_ends.npy`, so appended
  episodes would be invisible),
* every `meta.json` field matches except `n_episodes` -- including
  `schemas`, which is the *only* place the `--action-space` choice is
  recorded (`action` is shape `[8]` for both joint-velocity and
  joint-position stores),
* both stores are internally consistent: exactly `n_episodes` videos per
  video key, one `data/<ep>/` dir per episode holding every lowdim key, no
  stray files (a leftover from an interrupted earlier run aborts the run),
* both stores have a `tasks.json` labelling *every* episode -- src's labels
  are also required to be non-empty. Writing labels for only part of a store
  would silently strip the base episodes' tasks and shrink any `task_filter`
  training set.

`--src` must be converted with the same `--camera`/`--action-space`/
`--image-size`/`--fps`/`--crf` flags as `--dst`, and must have gone through
the same `trim_droid_idle_beginnings.py` step (or not) as `--dst` --
trimming leaves no trace in `meta.json`, so it is the one incompatibility
this script cannot detect for you.

Usage (see docs/droid_workflow.md, "Adding a new instruction variant")::

    python scripts/utils/filter_droid_task_sessions.py \\
        --data-dir ${DATA_ROOT}/droid_raw \\
        --output   /tmp/cooking_carrot_sessions --task "cooking - carrot"
    python scripts/utils/convert_droid_data.py \\
        --data-dir /tmp/cooking_carrot_sessions \\
        --output   /tmp/cooking_carrot_store \\
        --camera wrist_image_left=14313307 --camera exterior_image_1_left=36716034 \\
        --zed-settings-path ${DATA_ROOT}/zed_settings \\
        --skip-empty --skip-corrupt
    python scripts/utils/trim_droid_idle_beginnings.py \\
        --store  /tmp/cooking_carrot_store \\
        --output /tmp/cooking_carrot_store_trimmed
    python scripts/utils/append_droid_store.py --dry-run \\
        --src /tmp/cooking_carrot_store_trimmed \\
        --dst ${DATA_ROOT}/droid_converted_cooking_trimmed
    python scripts/utils/append_droid_store.py \\
        --src /tmp/cooking_carrot_store_trimmed \\
        --dst ${DATA_ROOT}/droid_converted_cooking_trimmed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# meta.json fields that legitimately differ between the two stores. Every
# other field must match exactly -- unknown fields included, so a future
# converter flag that lands in meta.json fails loudly here rather than
# silently mixing two kinds of episodes.
_PER_STORE_META_FIELDS = frozenset({"n_episodes"})

_MISSING = object()


def _ep_name(ep: int) -> str:
    return f"{ep:06d}"


def _load_meta(store: Path, label: str) -> dict:
    meta_path = store / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{label} {store} is not a store: no meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    for field in ("keys", "shapes", "dtypes", "video_keys", "n_episodes"):
        if field not in meta:
            raise ValueError(f"{label} {store}: meta.json is missing required field {field!r}")
    if (store / "episode_ends.npy").exists() or "n_frames" in meta:
        raise ValueError(
            f"{label} {store} uses the deprecated flat store layout "
            "(data/<key>/ arrays + episode_ends.npy). Its readers derive n_episodes from "
            "episode_ends.npy, so appended episodes would never be visible. Re-convert it "
            "with convert_droid_data.py (or pass it through trim_droid_idle_beginnings.py, "
            "which rewrites into the per-episode layout) and append onto that instead."
        )
    return meta


def _compare_meta(dst_meta: dict, src_meta: dict) -> None:
    for field in sorted((set(dst_meta) | set(src_meta)) - _PER_STORE_META_FIELDS):
        dst_value = dst_meta.get(field, _MISSING)
        src_value = src_meta.get(field, _MISSING)
        if dst_value != src_value:
            dst_shown = "<absent>" if dst_value is _MISSING else repr(dst_value)
            src_shown = "<absent>" if src_value is _MISSING else repr(src_value)
            raise ValueError(
                f"meta mismatch on {field!r}: dst={dst_shown} src={src_shown} "
                "-- src must be converted with the same camera/action-space/image-size/fps/crf "
                "flags as dst (note: the action space lives in 'schemas', not in 'action''s shape)"
            )


def _verify_layout(store: Path, meta: dict, label: str) -> None:
    """Fail unless `store` holds exactly `n_episodes` complete episodes."""
    n_episodes = int(meta["n_episodes"])
    expected_eps = {_ep_name(ep) for ep in range(n_episodes)}

    for vk in meta["video_keys"]:
        video_dir = store / "videos" / vk
        if n_episodes and not video_dir.is_dir():
            raise FileNotFoundError(
                f"{label} {store}: missing videos/{vk}/ for {n_episodes} episodes"
            )
        found = {p.name for p in video_dir.glob("episode_*.mp4")} if video_dir.is_dir() else set()
        expected = {f"episode_{name}.mp4" for name in expected_eps}
        if found != expected:
            missing = sorted(expected - found)[:5]
            stray = sorted(found - expected)[:5]
            raise ValueError(
                f"{label} {store}: videos/{vk}/ does not match meta.json's n_episodes="
                f"{n_episodes} (missing {len(expected - found)}, e.g. {missing}; "
                f"unexpected {len(found - expected)}, e.g. {stray}) -- the store is inconsistent, "
                "likely a leftover from an interrupted run; clean it up before appending"
            )

    lowdim_keys = [k for k in meta["keys"] if k not in meta["video_keys"]]
    data_dir = store / "data"
    found_eps = {p.name for p in data_dir.iterdir() if p.is_dir()} if data_dir.is_dir() else set()
    if found_eps != expected_eps:
        raise ValueError(
            f"{label} {store}: data/ holds {len(found_eps)} episode dirs but meta.json says "
            f"n_episodes={n_episodes} (missing {sorted(expected_eps - found_eps)[:5]}, "
            f"unexpected {sorted(found_eps - expected_eps)[:5]}) -- the store is inconsistent, "
            "likely a leftover from an interrupted run; clean it up before appending"
        )
    for ep_name in sorted(expected_eps):
        for key in lowdim_keys:
            if not (data_dir / ep_name / key).is_dir():
                raise FileNotFoundError(
                    f"{label} {store}: data/{ep_name}/ is missing lowdim key {key!r} -- "
                    "the episode is incomplete and would blow up at training time; "
                    "re-convert the store"
                )


def _load_tasks(store: Path, n_episodes: int, label: str, require_nonempty: bool) -> dict[str, str]:
    """Load `store`'s tasks.json, requiring a label for every one of its episodes."""
    tasks_path = store / "tasks.json"
    if not tasks_path.is_file():
        if n_episodes == 0:
            return {}
        raise FileNotFoundError(
            f"{label} {store}: tasks.json is missing but the store has {n_episodes} episodes. "
            "Appending would write a tasks.json labelling only the new episodes, which silently "
            "strips every existing episode's task and shrinks any task_filter training set. "
            "Re-run conversion to regenerate it."
        )
    with open(tasks_path) as f:
        tasks = json.load(f)

    unlabeled = [ep for ep in range(n_episodes) if str(ep) not in tasks]
    if unlabeled:
        raise ValueError(
            f"{label} {store}: tasks.json labels {len(tasks)} episodes but is missing "
            f"{len(unlabeled)} of the store's {n_episodes} (e.g. {unlabeled[:5]}) -- "
            "task_filter would silently treat those as untasked; re-run conversion"
        )
    stray = [k for k in tasks if not k.isdigit() or int(k) >= n_episodes]
    if stray:
        raise ValueError(
            f"{label} {store}: tasks.json has {len(stray)} labels outside "
            f"[0, {n_episodes}) (e.g. {sorted(stray)[:5]}) -- the store is inconsistent"
        )
    if require_nonempty:
        empty = [ep for ep in range(n_episodes) if not str(tasks[str(ep)]).strip()]
        if empty:
            raise ValueError(
                f"{label} {store}: tasks.json has empty task strings for "
                f"{len(empty)} episodes (e.g. {empty[:5]}) -- appended episodes with no label "
                "are invisible to task_filter; re-convert from a filtered session tree"
            )
    return {str(k): v for k, v in tasks.items()}


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _validate(src: Path, dst: Path) -> tuple[dict, dict, dict[str, str], dict[str, str]]:
    """Run every check. Returns (src_meta, dst_meta, src_tasks, dst_tasks)."""
    src_resolved, dst_resolved = src.resolve(), dst.resolve()
    if src_resolved == dst_resolved:
        raise ValueError(
            f"--src and --dst are the same store ({src_resolved}) -- refusing to append it to itself"
        )
    if src_resolved in dst_resolved.parents or dst_resolved in src_resolved.parents:
        raise ValueError(
            f"--src {src_resolved} and --dst {dst_resolved} are nested -- refusing to append"
        )

    src_meta = _load_meta(src, "--src")
    dst_meta = _load_meta(dst, "--dst")
    _compare_meta(dst_meta, src_meta)

    if int(src_meta["n_episodes"]) == 0:
        raise ValueError(f"--src {src} has no episodes -- nothing to append")

    _verify_layout(src, src_meta, "--src")
    _verify_layout(dst, dst_meta, "--dst")

    src_tasks = _load_tasks(src, int(src_meta["n_episodes"]), "--src", require_nonempty=True)
    dst_tasks = _load_tasks(dst, int(dst_meta["n_episodes"]), "--dst", require_nonempty=False)
    return src_meta, dst_meta, src_tasks, dst_tasks


def append_store(src: Path, dst: Path, dry_run: bool = False) -> tuple[int, int]:
    """Append `src`'s episodes onto `dst` in place. Returns (start, n_appended)."""
    src_meta, dst_meta, src_tasks, dst_tasks = _validate(src, dst)

    start = int(dst_meta["n_episodes"])
    n_src = int(src_meta["n_episodes"])
    video_keys = list(dst_meta["video_keys"])

    # Plan every copy up front so a missing source or an occupied destination
    # aborts before anything is written.
    copies: list[tuple[Path, Path, bool]] = []  # (src, dst, is_dir)
    for local_ep in range(n_src):
        global_ep = start + local_ep
        for vk in video_keys:
            src_video = src / "videos" / vk / f"episode_{_ep_name(local_ep)}.mp4"
            dst_video = dst / "videos" / vk / f"episode_{_ep_name(global_ep)}.mp4"
            if dst_video.exists():
                raise FileExistsError(f"refusing to overwrite existing {dst_video}")
            copies.append((src_video, dst_video, False))
        src_data = src / "data" / _ep_name(local_ep)
        dst_data = dst / "data" / _ep_name(global_ep)
        if dst_data.exists():
            raise FileExistsError(f"refusing to overwrite existing {dst_data}")
        copies.append((src_data, dst_data, True))

    new_tasks = {str(start + local_ep): src_tasks[str(local_ep)] for local_ep in range(n_src)}

    if dry_run:
        print(
            f"[dry-run] would append {n_src} episodes from {src} -> {dst} at indices "
            f"[{start}, {start + n_src}): {len(copies)} files/dirs to copy"
        )
        for task, count in sorted(_count_tasks(new_tasks).items()):
            print(f"[dry-run]   {count:5d} episodes  {task!r}")
        return start, n_src

    created: list[tuple[Path, bool]] = []
    tasks_path = dst / "tasks.json"
    tasks_backup = tasks_path.read_bytes() if tasks_path.is_file() else None
    try:
        for src_path, dst_path, is_dir in copies:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            created.append((dst_path, is_dir))
            if is_dir:
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # meta.json is written last: bumping n_episodes is what makes the new
        # episodes visible, so it commits the append.
        _write_json_atomic(tasks_path, {**dst_tasks, **new_tasks})
        _write_json_atomic(dst / "meta.json", {**dst_meta, "n_episodes": start + n_src})
    except BaseException:
        for path, is_dir in reversed(created):
            if is_dir:
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        if tasks_backup is not None:
            tasks_path.write_bytes(tasks_backup)
        print(
            f"append failed -- removed the {len(created)} file(s)/dir(s) this run created; "
            f"{dst} still holds its original {start} episodes"
        )
        raise

    return start, n_src


def _count_tasks(tasks: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks.values():
        counts[task] = counts.get(task, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", required=True, type=Path, help="Small store to append (e.g. one new instruction)."
    )
    parser.add_argument(
        "--dst", required=True, type=Path, help="Existing store to append onto, modified in place."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every validation check and report what would be copied, without writing anything.",
    )
    args = parser.parse_args(argv)

    start, n_appended = append_store(args.src, args.dst, dry_run=args.dry_run)
    if not args.dry_run:
        print(
            f"appended {n_appended} episodes from {args.src} -> {args.dst} "
            f"at indices [{start}, {start + n_appended})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
