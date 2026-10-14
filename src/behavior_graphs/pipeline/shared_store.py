"""One copy per distinct stage output, in a stage-shaped tree beside the runs.

A run directory is meant to be self-contained, so every job writes its own
``windows.npz``. But a sweep's jobs share upstream stages by construction: a
dataset has one window set per (width, stride), and the grid then runs every
reduction × K on top of it. Measured on ``sweep_all``: 7897 run dirs holding
294 GB of ``windows.npz`` — with **281 distinct contents totalling 10 GB**. 96%
of it was the same bytes written again, a median of 28 times each, and 80% of
the whole sweep directory. The stage cache does not help: it avoids *recomputing*
a stage, then the run dir gets a fresh copy of the result anyway.

So the shared parts move up a level, into a tree shaped like the pipeline::

    <root>/
        _shared/
            windows/<content-key>/{windows.npz,windows.json}
            reduced/<content-key>/reduced.npz
        <dataset>/<job>/
            shared.json          # {"windows": "../../_shared/windows/<key>", ...}
            labels.npy, graph.json, metrics.json, ...

Entries are content-addressed, so identical output dedupes across datasets and
across configs that happen to agree, and a migration can verify byte-identity
before removing anything.

**Pointers are relative paths, deliberately.** Not symlinks and not hardlinks:
the whole point is that ``cp -r``, ``scp -r``, ``rsync`` (without ``-H``) and a
plain zip all move this tree to another machine intact. A symlink would dangle
and a hardlink would silently become N copies again. A relative path inside the
tree survives every one of those, and survives the tree being renamed or moved.

The cost is that a *single run directory* is no longer self-contained: copied on
its own, away from its ``_shared`` sibling, it has no window features.
:func:`inline_shared` re-materialises one (that is what
``scripts/utils/dedupe_sweep_artifacts.py --inline`` runs), and readers raise a
message saying so rather than a bare FileNotFoundError.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

log = logging.getLogger(__name__)

#: Directory under the store root holding the shared entries.
SHARED_DIRNAME = "_shared"

#: Written into a run dir: artifact name -> relative path of its shared entry.
POINTER_FILE = "shared.json"

#: Stage outputs worth sharing, and the files each one owns — what a *new* entry
#: is written as. Ordered as the pipeline produces them. Only these are ever
#: redirected; everything else in a run dir stays local, because it is per-job
#: and small.
SHARED_STAGES: dict[str, tuple[str, ...]] = {
    "windows": ("windows.npz", "windows.json"),
    "reduced": ("reduced.npz",),
}

#: How many hex characters of the content hash name an entry. 16 is 64 bits —
#: collision-free for any plausible number of artifacts, and short enough that
#: the paths stay readable in a pointer file.
_KEY_CHARS = 16


def content_key(files: dict[str, Path]) -> str:
    """A content hash over ``{name: path}``, stable across machines and runs.

    Names are folded in alongside the bytes, so two entries that happen to share
    one file but not another cannot collide.
    """
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        with open(files[name], "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()[:_KEY_CHARS]


class SharedStore:
    """The ``_shared`` tree under one root (a sweep dir, or any chosen root)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.dir = self.root / SHARED_DIRNAME

    def entry(self, stage: str, key: str) -> Path:
        return self.dir / stage / key

    # ------------------------------------------------------- stage-key aliases
    def _alias(self, stage: str, stage_key: str) -> Path:
        """Where the stage-cache key -> content key note lives.

        Under ``_shared/.stage_keys/`` rather than inside ``_shared/<stage>/``,
        so it is not mistaken for an entry by anything listing that directory.
        """
        return self.dir / ".stage_keys" / stage / f"{stage_key}.key"

    def lookup(self, stage: str, stage_key: str) -> Path | None:
        """The existing entry for a stage-cache key, without touching the bytes.

        The stage cache already content-addresses its output: the same
        ``stage_key`` is the same array, so the content key it produced last time
        is still correct. Recording that mapping is what lets a later job skip
        re-serialising an artifact it just read in order to discover that an
        identical copy is already stored.

        Returns None when there is no note, or the note points at an entry that
        has since been pruned -- in which case the caller writes it normally.
        """
        note = self._alias(stage, stage_key)
        if not note.exists():
            return None
        try:
            key = note.read_text().strip()
        except OSError:
            return None
        target = self.entry(stage, key)
        return target if key and _files_present(target, SHARED_STAGES[stage]) else None

    def remember(self, stage: str, stage_key: str, entry: Path) -> None:
        """Note that ``stage_key`` produced ``entry``. Best effort by design.

        A failure here costs the next job one redundant serialise, so it must
        never take the run down with it.
        """
        note = self._alias(stage, stage_key)
        try:
            note.parent.mkdir(parents=True, exist_ok=True)
            tmp = note.with_name(f".{note.name}.tmp")
            tmp.write_text(entry.name)
            os.replace(tmp, note)
        except OSError as e:
            log.warning("could not record stage-key alias %s: %s", note, e)

    def put_files(
        self,
        stage: str,
        files: dict[str, Path],
        *,
        move: bool = False,
        key: str | None = None,
    ) -> Path:
        """Commit ``{name: path}`` as a shared entry; return its directory.

        A second call with the same content is a no-op that returns the existing
        entry, so this is idempotent and safe to re-run over a half-migrated
        tree. ``move`` renames the sources in (used by the migration, where the
        source is one of the copies being replaced); otherwise they are copied.
        ``key`` skips re-hashing when the caller already has the content key.
        """
        key = key or content_key(files)
        target = self.entry(stage, key)
        if _files_present(target, files):
            if move:
                for path in files.values():
                    path.unlink(missing_ok=True)
            return target

        # Assemble under a temp name in the same directory, then rename into
        # place: a reader never sees a half-written entry, and two processes
        # racing on the same content both end up with the same bytes.
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=target.parent))
        try:
            for name, path in files.items():
                if move:
                    shutil.move(str(path), staging / name)
                else:
                    shutil.copy2(path, staging / name)
            try:
                os.replace(staging, target)
            except OSError:
                # `target` exists and is not empty, so the directory rename
                # cannot go through. Either another process won the race -- in
                # which case its bytes are this content and we are done -- or
                # the entry is *partial*, holding some of the file set and not
                # the rest. A partial entry used to be terminal: `_files_present`
                # rejects it, so it was neither reusable nor replaceable, and
                # every future job for that content raised ENOTEMPTY. Fill it in
                # file by file instead; each rename is atomic, so a reader sees
                # either the old file or the new one and never a torn write.
                if _files_present(target, files):
                    raise
                for name in files:
                    src = staging / name
                    if src.exists():
                        os.replace(src, target / name)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not _files_present(target, files):
                raise
        return target

    def put_written(
        self, stage: str, write: Callable[[Path], None], *, stage_key: str | None = None
    ) -> Path:
        """Commit whatever ``write(dir)`` produces for ``stage``.

        The writer runs into a temp directory, so the content key is computed
        from the bytes actually written rather than from any config that was
        meant to determine them.

        ``stage_key`` is the caller's stage-cache key for this output, and it is
        what makes a sweep affordable. Without it every job serialises its
        windows and reduced arrays in full and hashes them, only to find the
        identical entry already stored: measured on legos, 11.4s of a 22s job
        spent re-compressing and re-hashing a 1.7 GB window array that had just
        been read from cache. With it, the second and later jobs sharing an
        upstream stage write a pointer and nothing else.
        """
        if stage_key is not None:
            hit = self.lookup(stage, stage_key)
            if hit is not None:
                return hit
        expected = SHARED_STAGES[stage]
        staging = Path(tempfile.mkdtemp(prefix=f".{stage}.", dir=_ensure(self.dir / stage)))
        try:
            write(staging)
            files = {name: staging / name for name in expected if (staging / name).exists()}
            if not files:
                raise ValueError(f"{stage} writer produced none of {expected}")
            entry = self.put_files(stage, files, move=True)
            if stage_key is not None:
                self.remember(stage, stage_key, entry)
            return entry
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _files_present(entry: Path, files: Iterable[str] | dict[str, Path]) -> bool:
    """Whether ``entry`` holds exactly these named files."""
    names = files if not isinstance(files, dict) else files.keys()
    return entry.is_dir() and all((entry / name).exists() for name in names)


def has_stage(where: Path, stage: str) -> bool:
    """Whether a directory holds a complete ``stage`` file set."""
    return _files_present(where, SHARED_STAGES.get(stage, ()))


def _copy_atomic(source: Path, dest: Path) -> None:
    """Copy via a temp name: a truncated file would win over an intact pointer."""
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, dest)


# ------------------------------------------------------------------ pointers


def read_pointers(run_dir: Path) -> dict[str, str]:
    """``{artifact name: relative path}`` recorded in ``run_dir``, or empty."""
    path = Path(run_dir) / POINTER_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("unreadable %s: %s", path, e)
        return {}
    return {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def write_pointer(run_dir: Path, stage: str, entry: Path) -> None:
    """Record ``stage`` -> ``entry`` in ``run_dir``, as a relative path.

    Merges into any pointers already there, so stages can be committed one at a
    time as the pipeline reaches them.
    """
    run_dir = Path(run_dir)
    pointers = read_pointers(run_dir)
    pointers[stage] = os.path.relpath(Path(entry).resolve(), run_dir.resolve())
    (run_dir / POINTER_FILE).write_text(json.dumps(pointers, indent=2, sort_keys=True) + "\n")


def remove_local(run_dir: Path, stage: str) -> None:
    """Delete any local copy of ``stage``; only call once its pointer is written.

    A stale local copy wins in :func:`resolve`, so committing a pointer without
    this leaves readers on the old bytes.
    """
    for name in SHARED_STAGES.get(stage, ()):
        (Path(run_dir) / name).unlink(missing_ok=True)


def resolve(run_dir: Path, stage: str) -> Path | None:
    """Where ``stage``'s files are for this run: the run dir, or a shared entry.

    A *complete* local set always wins, so a run written before the store existed
    — or one deliberately inlined — reads exactly as it used to. A partial one
    does not: the migration removes the local files one by one, and an interrupt
    mid-delete must fall through to the pointer rather than pick a run dir that
    is missing half the stage. Returns None when the run has neither, which is a
    genuinely missing artifact.
    """
    run_dir = Path(run_dir)
    if has_stage(run_dir, stage):
        return run_dir
    rel = read_pointers(run_dir).get(stage)
    if rel is None:
        return None
    entry = (run_dir / rel).resolve()
    if has_stage(entry, stage):
        return entry
    raise FileNotFoundError(
        f"{run_dir} points at a shared '{stage}' entry that is not there: {entry}. "
        f"A run directory is not self-contained once its artifacts are shared — copy "
        f"the whole sweep directory (its '{SHARED_DIRNAME}' tree comes with it), or "
        f"run scripts/utils/dedupe_sweep_artifacts.py --inline on this run first."
    )


def inline_shared(run_dir: Path, stage: str | None = None) -> list[str]:
    """Copy shared entries back into ``run_dir``, making it self-contained again.

    Returns the stage names it materialised. The pointer is dropped only once the
    stage is verifiably complete locally: an entry that is gone or half there
    would otherwise take the run's last record of where its content lives, and
    the run would vanish from discovery instead of raising.
    """
    run_dir = Path(run_dir)
    pointers = read_pointers(run_dir)
    stages = [stage] if stage else sorted(pointers)
    done: list[str] = []
    for name in stages:
        rel = pointers.get(name)
        if rel is None:
            continue
        entry = (run_dir / rel).resolve()
        for filename in SHARED_STAGES.get(name, ()):
            source = entry / filename
            if source.exists():
                _copy_atomic(source, run_dir / filename)
        if not has_stage(run_dir, name):
            log.warning(
                "%s: shared '%s' entry %s is not there; keeping the pointer", run_dir, name, entry
            )
            continue
        pointers.pop(name)
        done.append(name)
    if done:
        if pointers:
            (run_dir / POINTER_FILE).write_text(
                json.dumps(pointers, indent=2, sort_keys=True) + "\n"
            )
        else:
            (run_dir / POINTER_FILE).unlink(missing_ok=True)
    return done
