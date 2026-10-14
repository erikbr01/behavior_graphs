#!/usr/bin/env python3
"""Move a sweep's duplicated stage outputs into its shared store (or back).

A sweep's jobs share upstream stages by construction — one window set per
(dataset, width, stride), then every reduction × K on top of it — but each job
wrote its own copy. On ``sweep_all`` that was 7897 copies of ``windows.npz``
totalling 294 GB, holding 281 distinct contents worth 10 GB.

This rewrites such a tree in place to the layout in
:mod:`behavior_graphs.pipeline.shared_store`: one content-addressed copy under
``<root>/_shared/<stage>/``, and a ``shared.json`` of *relative* paths in each
run dir. No symlinks and no hardlinks, so ``cp -r`` / ``scp -r`` / ``rsync``
without ``-H`` / a plain zip all move the result to another machine intact.

Safe to interrupt and re-run, and safe to run while something is reading the
tree: each run is copied into the store, *then* given its pointer, and only then
does the local copy go. A reader therefore always sees either a local copy or a
pointer to content already in place — never neither — and an interrupt at any
point leaves a readable run rather than one whose file has moved but whose
pointer was never written. A run already pointing at the store is skipped.

    # what would change, and how much it saves
    python scripts/utils/dedupe_sweep_artifacts.py <sweep_dir> --dry-run

    # do it
    python scripts/utils/dedupe_sweep_artifacts.py <sweep_dir>

    # put it back — a run dir alone is not self-contained once shared
    python scripts/utils/dedupe_sweep_artifacts.py <sweep_dir> --inline

    # reclaim entries nothing points at any more (after deleting run dirs;
    # unlike the above, only when nothing is writing to the tree)
    python scripts/utils/dedupe_sweep_artifacts.py <sweep_dir> --gc
"""

from __future__ import annotations

import argparse
import filecmp
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from behavior_graphs.pipeline.shared_store import (  # noqa: E402
    SHARED_DIRNAME,
    SHARED_STAGES,
    SharedStore,
    content_key,
    has_stage,
    inline_shared,
    read_pointers,
    remove_local,
    write_pointer,
)

log = logging.getLogger("dedupe")


def run_dirs(root: Path):
    """Every run directory under *root*, skipping the shared tree itself."""
    for config in sorted(root.rglob("config.yaml")):
        run = config.parent
        if SHARED_DIRNAME in run.parts:
            continue
        yield run


def _local_files(run: Path, stage: str) -> dict[str, Path]:
    """The stage's files that this run holds locally, if it holds a complete set.

    A partial set is left alone: it is not something this can safely dedupe.
    """
    names = SHARED_STAGES.get(stage, ())
    present = {name: run / name for name in names if (run / name).exists()}
    return present if len(present) == len(names) else {}


def _drop_leftovers(run: Path, stage: str, rel: str | None, *, dry_run: bool) -> int:
    """Finish an interrupted delete; return the bytes it reclaimed.

    Step 3 removes the local files one at a time, so an interrupt can leave a
    partial set behind. resolve() ignores a partial set and reads the pointer, so
    the run is fine — but the space is not back until the leftovers go, and only
    files byte-identical to the entry's are safe to assume are leftovers.
    """
    stray = [run / name for name in SHARED_STAGES.get(stage, ()) if (run / name).exists()]
    if not stray:
        return 0
    entry = (run / rel).resolve() if rel else None
    if entry is None or not has_stage(entry, stage):
        log.warning("%s has a partial local %s and no usable pointer — skipping", run, stage)
        return 0
    freed = 0
    for path in stray:
        shared = entry / path.name
        if not shared.exists() or not filecmp.cmp(path, shared, shallow=False):
            log.warning("%s differs from %s — leaving it", path, shared)
            continue
        freed += path.stat().st_size
        if not dry_run:
            path.unlink()
    return freed


def dedupe(root: Path, *, dry_run: bool, stages: list[str]) -> None:
    store = SharedStore(root)
    freed = 0
    kept = 0
    moved: dict[str, set[str]] = {s: set() for s in stages}
    runs = 0
    for run in run_dirs(root):
        runs += 1
        pointers = read_pointers(run)
        for stage in stages:
            files = _local_files(run, stage)
            if not files:
                # Nothing to move: already shared, never produced, or a partial
                # set an interrupted run of this script left behind.
                freed += _drop_leftovers(run, stage, pointers.get(stage), dry_run=dry_run)
                continue
            size = sum(p.stat().st_size for p in files.values())
            key = content_key(files)
            first = key not in moved[stage]
            if first:
                kept += size
                moved[stage].add(key)
            else:
                freed += size
            if dry_run:
                continue
            # Order matters, and it is the whole reason this is safe to run
            # against a live tree: copy into the store, write the pointer while
            # the local copy is still there (resolve() prefers local, so nothing
            # changes for a reader yet), and only then remove the local copy.
            # Moving first would leave a window — and, on an interrupt, a
            # permanent state — where the run has neither.
            entry = store.put_files(stage, files, move=False, key=key)
            write_pointer(run, stage, entry)
            remove_local(run, stage)
    verb = "would free" if dry_run else "freed"
    print(f"{runs} run dirs scanned")
    for stage in stages:
        print(f"  {stage}: {len(moved[stage])} distinct entries")
    print(f"  kept once: {kept / 1e9:.1f} GB")
    print(f"  {verb}:  {freed / 1e9:.1f} GB")


def inline(root: Path, *, dry_run: bool) -> None:
    runs = 0
    added = 0
    for run in run_dirs(root):
        pointers = read_pointers(run)
        if not pointers:
            continue
        for stage in pointers:
            entry = (run / pointers[stage]).resolve()
            added += sum(
                (entry / name).stat().st_size
                for name in SHARED_STAGES.get(stage, ())
                if (entry / name).exists()
            )
        # inline_shared keeps the pointer (and warns) when an entry is missing,
        # so only count the runs it actually made self-contained.
        if dry_run or inline_shared(run):
            runs += 1
    verb = "would grow" if dry_run else "grew"
    print(f"{runs} run dirs made self-contained again; {verb} by {added / 1e9:.1f} GB")


def _reencode(args: tuple[str, str]) -> tuple[str, str, int, int]:
    """Worker: write a legacy entry's compressed twin and verify it round-trips.

    Returns ``(old entry, new entry, old size, new size)``. Writing is safe in
    parallel: ``put_written`` stages under a temp name and renames, and two
    workers can only collide on identical content, where either result is right.
    """
    import numpy as np

    root, entry_str = args
    entry = Path(entry_str)
    array = np.load(entry / "reduced.npy")
    new_entry = SharedStore(Path(root)).put_written(
        "reduced",
        lambda target, a=array: np.savez_compressed(target / "reduced.npz", reduced=a),
    )
    with np.load(new_entry / "reduced.npz") as data:
        if not np.array_equal(data["reduced"], array):
            raise RuntimeError(f"re-encode of {entry} does not round-trip")
    return (
        entry_str,
        str(new_entry),
        (entry / "reduced.npy").stat().st_size,
        (new_entry / "reduced.npz").stat().st_size,
    )


def recompress(root: Path, *, dry_run: bool, jobs: int = 1) -> None:
    """Re-encode legacy uncompressed ``reduced.npy`` as ``reduced.npz``.

    One-off, so a tree holds one format and the dual-format read paths can go.
    ``content_key`` hashes the filename with the bytes, so re-encoding gives an
    entry a new key: each is written, the runs pointing at the old entry are
    repointed, and only then is the old entry dropped — a reader always sees a
    pointer to an entry that exists.

    Compression is CPU-bound, so the encode runs across ``jobs`` processes while
    the parent does the repointing and deleting, keeping that ordering in one
    place.
    """
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    store = SharedStore(root)
    stage_dir = store.dir / "reduced"

    referenced: dict[Path, list[Path]] = {}
    for run in run_dirs(root):
        for stage, rel in read_pointers(run).items():
            if stage == "reduced":
                referenced.setdefault((run / rel).resolve(), []).append(run)

    todo: list[str] = []
    orphaned = 0
    if stage_dir.is_dir():
        for entry in sorted(stage_dir.iterdir()):
            if (
                not entry.is_dir()
                or entry.name.startswith(".")
                or not (entry / "reduced.npy").exists()
            ):
                continue
            if referenced.get(entry.resolve()):
                todo.append(str(entry))
            else:
                orphaned += 1  # --gc's business, not this pass's

    locals_ = [run for run in run_dirs(root) if (run / "reduced.npy").exists()]

    if dry_run:
        print(f"would convert {len(todo)} shared entries, {len(locals_)} local copies")
        if orphaned:
            print(f"  skipping {orphaned} orphaned entries (use --gc)")
        return

    before = after = 0
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for old, new, old_size, new_size in pool.map(
            _reencode, [(str(root), e) for e in todo], chunksize=1
        ):
            for run in referenced[Path(old).resolve()]:
                write_pointer(run, "reduced", Path(new))
            shutil.rmtree(old)
            before += old_size
            after += new_size
            done += 1
            if done % 200 == 0:
                log.info("%d/%d entries re-encoded", done, len(todo))

    for run in locals_:
        legacy = run / "reduced.npy"
        array = np.load(legacy)
        # Name it .npz: savez_compressed appends the extension otherwise, and the
        # rename would look for a file numpy never wrote.
        staged = run / ".reduced.tmp.npz"
        np.savez_compressed(staged, reduced=array)
        staged.replace(run / "reduced.npz")
        legacy.unlink()
        before += legacy.stat().st_size if legacy.exists() else 0
        after += (run / "reduced.npz").stat().st_size

    print(f"converted {done} shared entries, {len(locals_)} local copies")
    if orphaned:
        print(f"  skipped {orphaned} orphaned entries (use --gc)")
    if before:
        print(f"  {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB")


def gc(root: Path, *, dry_run: bool) -> None:
    """Delete shared entries no run points at any more.

    Deleting run dirs is how a sweep drops jobs, and once their outputs live in
    the store the space does not come back with them — the entry is only garbage
    once *every* run that referenced it is gone. So this is reachability: collect
    what is referenced, remove the rest.

    Deliberately not automatic on delete. An entry is shared by many runs, so
    "is anything still using this?" can only be answered by a full scan, and
    doing that per deleted run would make dropping a dataset quadratic.

    Unlike the migration, this is **not** safe to run while a sweep is writing to
    the tree: an entry stored between the scan and the sweep's ``write_pointer``
    looks unreferenced, and deleting it leaves that run dangling. Dot-prefixed
    names are skipped because they are the store's in-flight staging dirs.
    """
    store = SharedStore(root)
    if not store.dir.is_dir():
        print("no shared store here")
        return
    referenced: set[Path] = set()
    runs = 0
    for run in run_dirs(root):
        runs += 1
        for rel in read_pointers(run).values():
            referenced.add((run / rel).resolve())

    removed = kept = 0
    freed = 0
    for stage_dir in sorted(store.dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        for entry in sorted(stage_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.resolve() in referenced:
                kept += 1
                continue
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            removed += 1
            freed += size
            if not dry_run:
                shutil.rmtree(entry)
    verb = "would free" if dry_run else "freed"
    print(f"{runs} run dirs reference {len(referenced)} entries")
    print(f"  kept:    {kept}")
    print(f"  orphans: {removed}")
    print(f"  {verb}:  {freed / 1e9:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="sweep directory (or any run-dir tree)")
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="reverse: copy shared entries back into each run dir",
    )
    parser.add_argument(
        "--recompress",
        action="store_true",
        help="one-off: re-encode legacy uncompressed reduced.npy as reduced.npz",
    )
    parser.add_argument(
        "--gc",
        action="store_true",
        help="delete shared entries no run points at (after deleting run dirs; not while a sweep writes)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="parallel workers for --recompress (compression is CPU-bound)",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(SHARED_STAGES),
        help="limit to one stage (repeatable); default is all of them",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.root.is_dir():
        parser.error(f"{args.root} is not a directory")
    if args.recompress:
        recompress(args.root, dry_run=args.dry_run, jobs=args.jobs)
    elif args.gc:
        gc(args.root, dry_run=args.dry_run)
    elif args.inline:
        inline(args.root, dry_run=args.dry_run)
    else:
        dedupe(args.root, dry_run=args.dry_run, stages=args.stage or sorted(SHARED_STAGES))


if __name__ == "__main__":
    main()
