"""One copy per distinct stage output, and every reader still finding it.

The property that matters most here is not the space saved but that the result
still *moves*: the whole point of relative pointers over symlinks or hardlinks is
that a plain recursive copy to another machine keeps working. That is what
``test_a_plain_recursive_copy_still_reads`` pins.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
from behavior_graphs.pipeline.artifacts import RunArtifacts
from behavior_graphs.pipeline.shared_store import (
    SHARED_DIRNAME,
    SharedStore,
    content_key,
    inline_shared,
    read_pointers,
    resolve,
)


def _windows(n: int, dim: int = 4, seed: int = 0) -> WindowSet:
    rng = np.random.default_rng(seed)
    return WindowSet(
        features=rng.normal(size=(n, dim)).astype(np.float32),
        windows=[WindowInfo(episode_idx=0, start=t, end=t + 1) for t in range(n)],
        episodes=episodes_from_lengths([n], [True]),
    )


class TestSharing:
    def test_identical_windows_are_stored_once(self, tmp_path):
        windows = _windows(20)
        for job in ("a", "b", "c"):
            RunArtifacts(tmp_path / job, shared_root=tmp_path).save_windows(windows)

        entries = list((tmp_path / SHARED_DIRNAME / "windows").iterdir())
        assert len(entries) == 1, entries
        # And no run kept its own copy.
        for job in ("a", "b", "c"):
            assert not (tmp_path / job / "windows.npz").exists()

    def test_different_windows_get_their_own_entry(self, tmp_path):
        RunArtifacts(tmp_path / "a", shared_root=tmp_path).save_windows(_windows(20, seed=0))
        RunArtifacts(tmp_path / "b", shared_root=tmp_path).save_windows(_windows(20, seed=1))

        assert len(list((tmp_path / SHARED_DIRNAME / "windows").iterdir())) == 2

    def test_every_job_reads_back_what_it_saved(self, tmp_path):
        windows = _windows(20)
        reduced = np.arange(40, dtype=np.float32).reshape(20, 2)
        artifacts = RunArtifacts(tmp_path / "job", shared_root=tmp_path)
        artifacts.save_windows(windows)
        artifacts.save_array("reduced", reduced)

        # A fresh reader, with no idea the store exists.
        reader = RunArtifacts(tmp_path / "job")
        assert np.array_equal(reader.load_windows().features, windows.features)
        assert np.array_equal(reader.load_array("reduced"), reduced)
        assert len(reader.load_windows_meta()["windows"]) == 20

    def test_load_result_finds_the_shared_artifacts(self, tmp_path):
        """It probes through resolve(): a deduped job dir has neither file."""
        windows = _windows(20)
        reduced = np.arange(40, dtype=np.float32).reshape(20, 2)
        artifacts = RunArtifacts(tmp_path / "job", shared_root=tmp_path)
        artifacts.save_windows(windows)
        artifacts.save_array("reduced", reduced)

        result = RunArtifacts(tmp_path / "job").load_result()

        assert result.windows is not None, "the VLM labeler needs these"
        assert np.array_equal(result.windows.features, windows.features)
        assert np.array_equal(result.reduced, reduced)

    def test_a_rerun_leaves_no_stale_local_copy(self, tmp_path):
        """A job that died mid-pipeline is re-run into the same dir."""
        run = tmp_path / "job"
        RunArtifacts(run).save_windows(_windows(20, seed=0))  # the dead run's copy
        fresh = _windows(20, seed=1)

        RunArtifacts(run, shared_root=tmp_path).save_windows(fresh)

        assert not (run / "windows.npz").exists()
        assert np.array_equal(RunArtifacts(run).load_windows().features, fresh.features)

    def test_a_run_without_a_shared_root_stays_self_contained(self, tmp_path):
        """The default is unchanged, so a single run is still one directory."""
        RunArtifacts(tmp_path / "job").save_windows(_windows(20))

        assert (tmp_path / "job" / "windows.npz").exists()
        assert not (tmp_path / SHARED_DIRNAME).exists()
        assert read_pointers(tmp_path / "job") == {}

    def test_per_job_arrays_are_never_shared(self, tmp_path):
        """Labels differ per job, so sharing them would be wrong, not just useless."""
        artifacts = RunArtifacts(tmp_path / "job", shared_root=tmp_path)
        artifacts.save_array("labels", np.zeros(5, dtype=np.int64))

        assert (tmp_path / "job" / "labels.npy").exists()
        assert "labels" not in read_pointers(tmp_path / "job")


class TestPortability:
    @pytest.fixture()
    def shared_tree(self, tmp_path) -> Path:
        root = tmp_path / "sweep"
        windows = _windows(20)
        for job in ("a", "b"):
            RunArtifacts(root / "ds" / job, shared_root=root).save_windows(windows)
        return root

    def test_nothing_is_a_symlink_or_a_hardlink(self, shared_tree):
        """Both would break on the copy paths this layout exists to survive."""
        files = [p for p in shared_tree.rglob("*") if p.is_file()]
        assert files
        assert not any(p.is_symlink() for p in files)
        assert all(p.stat().st_nlink == 1 for p in files)

    def test_a_plain_recursive_copy_still_reads(self, shared_tree, tmp_path):
        """`cp -r` — no -a, no --preserve=links, no rsync -H."""
        destination = tmp_path / "elsewhere"
        shutil.copytree(shared_tree, destination, symlinks=False)

        features = RunArtifacts(destination / "ds" / "a").load_windows().features
        assert features.shape == (20, 4)

    def test_pointers_are_relative(self, shared_tree):
        pointer = json.loads((shared_tree / "ds" / "a" / "shared.json").read_text())
        assert pointer["windows"].startswith("../../")

    def test_a_run_lifted_out_on_its_own_says_what_is_wrong(self, shared_tree, tmp_path):
        """Rather than a bare FileNotFoundError three frames deep in numpy."""
        lifted = tmp_path / "lifted"
        shutil.copytree(shared_tree / "ds" / "a", lifted)

        with pytest.raises(FileNotFoundError, match="not self-contained"):
            resolve(lifted, "windows")

    def test_inline_keeps_the_pointer_when_the_entry_is_gone(self, shared_tree):
        """Dropping it would erase the run's only record of where its data was."""
        run = shared_tree / "ds" / "a"
        shutil.rmtree(shared_tree / SHARED_DIRNAME)

        assert inline_shared(run) == []

        assert read_pointers(run) != {}
        with pytest.raises(FileNotFoundError, match="not self-contained"):
            resolve(run, "windows")

    def test_inline_makes_a_run_self_contained_again(self, shared_tree):
        run = shared_tree / "ds" / "a"

        assert inline_shared(run) == ["windows"]

        assert (run / "windows.npz").exists()
        assert not (run / "shared.json").exists()
        # The other job is untouched and still shares.
        assert read_pointers(shared_tree / "ds" / "b") != {}
        assert np.array_equal(
            RunArtifacts(run).load_windows().features,
            RunArtifacts(shared_tree / "ds" / "b").load_windows().features,
        )


class TestStoreMechanics:
    def test_a_complete_local_copy_wins_over_a_pointer(self, tmp_path):
        """So a half-migrated tree, or a deliberately inlined run, reads locally."""
        run = tmp_path / "job"
        RunArtifacts(run, shared_root=tmp_path).save_windows(_windows(20))
        shutil.copytree(resolve(run, "windows"), run, dirs_exist_ok=True)

        assert resolve(run, "windows") == run

    def test_a_partial_local_copy_does_not(self, tmp_path):
        """An interrupt between the migration's two unlinks must still read."""
        run = tmp_path / "job"
        windows = _windows(20)
        RunArtifacts(run, shared_root=tmp_path).save_windows(windows)
        (run / "windows.json").write_text("{}")  # the file step 3 had not reached

        assert resolve(run, "windows") != run
        assert np.array_equal(RunArtifacts(run).load_windows().features, windows.features)

    def test_committing_the_same_content_twice_is_a_no_op(self, tmp_path):
        store = SharedStore(tmp_path)
        source = tmp_path / "src"
        source.mkdir()
        (source / "reduced.npz").write_bytes(b"payload")

        first = store.put_files("reduced", {"reduced.npz": source / "reduced.npz"})
        (source / "reduced.npz").write_bytes(b"payload")
        second = store.put_files("reduced", {"reduced.npz": source / "reduced.npz"})

        assert first == second
        assert len(list((tmp_path / SHARED_DIRNAME / "reduced").iterdir())) == 1

    def test_the_content_key_covers_names_as_well_as_bytes(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"same")
        b.write_bytes(b"same")

        assert content_key({"one.npy": a}) != content_key({"two.npy": b})
        assert content_key({"one.npy": a}) == content_key({"one.npy": b})

    def test_a_missing_artifact_is_none_not_an_error(self, tmp_path):
        (tmp_path / "job").mkdir()

        assert resolve(tmp_path / "job", "windows") is None


class TestDiscovery:
    """The apps find runs by looking for their files; sharing moves one of them."""

    def test_a_shared_run_is_still_a_run(self, tmp_path):
        from behavior_graphs.pipeline.run_discovery import is_run_dir

        run = tmp_path / "ds" / "job"
        artifacts = RunArtifacts(run, shared_root=tmp_path)
        artifacts.save_windows(_windows(10))
        artifacts.save_array("labels", np.zeros(10, dtype=np.int64))

        assert not (run / "windows.json").exists(), "the point: it is not local"
        assert is_run_dir(run)

    def test_a_local_run_is_still_a_run(self, tmp_path):
        from behavior_graphs.pipeline.run_discovery import is_run_dir

        run = tmp_path / "job"
        artifacts = RunArtifacts(run)
        artifacts.save_windows(_windows(10))
        artifacts.save_array("labels", np.zeros(10, dtype=np.int64))

        assert is_run_dir(run)

    def test_a_directory_with_neither_is_not(self, tmp_path):
        from behavior_graphs.pipeline.run_discovery import is_run_dir

        (tmp_path / "empty").mkdir()

        assert not is_run_dir(tmp_path / "empty")


class TestMigrationIsSafeOnALiveTree:
    """The migration rewrites a directory the apps may be reading right now.

    So it must never leave a run with neither a local copy nor a usable pointer —
    not for an instant, and not permanently if it is interrupted. It gets that by
    ordering the three steps copy / point / delete, which these pin from the
    outside: after each individual step, the run still loads.
    """

    @staticmethod
    def _script():
        import importlib.util
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "scripts" / "utils"
        spec = importlib.util.spec_from_file_location(
            "dedupe_sweep_artifacts", path / "dedupe_sweep_artifacts.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _sweep(self, tmp_path: Path) -> Path:
        root = tmp_path / "sweep"
        windows = _windows(20)
        for job in ("a", "b"):
            run = root / "ds" / job
            RunArtifacts(run).save_windows(windows)  # local, i.e. pre-migration
            RunArtifacts(run).save_array("labels", np.zeros(20, dtype=np.int64))
            (run / "config.yaml").write_text("{}\n")
        return root

    def test_a_run_is_readable_at_every_step(self, tmp_path):
        root = self._sweep(tmp_path)
        run = root / "ds" / "a"
        store = SharedStore(root)
        files = {n: run / n for n in ("windows.npz", "windows.json")}
        before = RunArtifacts(run).load_windows().features

        entry = store.put_files("windows", files, move=False)
        assert np.array_equal(RunArtifacts(run).load_windows().features, before)

        from behavior_graphs.pipeline.shared_store import write_pointer

        write_pointer(run, "windows", entry)
        assert np.array_equal(RunArtifacts(run).load_windows().features, before)

        for path in files.values():
            path.unlink()
        assert np.array_equal(RunArtifacts(run).load_windows().features, before)

    def test_the_script_leaves_every_run_loadable(self, tmp_path):
        root = self._sweep(tmp_path)
        before = RunArtifacts(root / "ds" / "a").load_windows().features

        self._script().dedupe(root, dry_run=False, stages=["windows"])

        for job in ("a", "b"):
            run = root / "ds" / job
            assert not (run / "windows.npz").exists()
            assert np.array_equal(RunArtifacts(run).load_windows().features, before)

    def test_an_interrupted_delete_is_finished_on_the_next_run(self, tmp_path):
        """Step 3 unlinks one file at a time, so a kill can leave half a set."""
        root = self._sweep(tmp_path)
        run = root / "ds" / "a"
        script = self._script()
        script.dedupe(root, dry_run=False, stages=["windows"])
        shutil.copy2(resolve(run, "windows") / "windows.json", run / "windows.json")

        script.dedupe(root, dry_run=False, stages=["windows"])

        assert not (run / "windows.json").exists()
        assert RunArtifacts(run).load_windows().features.shape == (20, 4)

    def test_dry_run_changes_nothing(self, tmp_path):
        root = self._sweep(tmp_path)

        self._script().dedupe(root, dry_run=True, stages=["windows"])

        assert (root / "ds" / "a" / "windows.npz").exists()
        assert not (root / SHARED_DIRNAME).exists()


class TestGarbageCollection:
    """Space only comes back when the last run referencing an entry is gone."""

    @staticmethod
    def _script():
        import importlib.util
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "scripts" / "utils"
        spec = importlib.util.spec_from_file_location(
            "dedupe_sweep_artifacts", path / "dedupe_sweep_artifacts.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _sweep(self, tmp_path: Path) -> Path:
        """Two jobs sharing one window set, one job with its own."""
        root = tmp_path / "sweep"
        shared = _windows(20, seed=0)
        for job in ("a", "b"):
            RunArtifacts(root / "ds" / job, shared_root=root).save_windows(shared)
        RunArtifacts(root / "ds" / "c", shared_root=root).save_windows(_windows(20, seed=1))
        for job in ("a", "b", "c"):
            (root / "ds" / job / "config.yaml").write_text("{}\n")
        return root

    def test_an_entry_two_runs_share_survives_losing_one(self, tmp_path):
        root = self._sweep(tmp_path)
        shutil.rmtree(root / "ds" / "a")

        self._script().gc(root, dry_run=False)

        # b still needs it, so it is not garbage.
        assert RunArtifacts(root / "ds" / "b").load_windows().features.shape == (20, 4)
        assert len(list((root / SHARED_DIRNAME / "windows").iterdir())) == 2

    def test_an_entry_nothing_points_at_is_removed(self, tmp_path):
        root = self._sweep(tmp_path)
        shutil.rmtree(root / "ds" / "c")  # the only run using its window set

        self._script().gc(root, dry_run=False)

        assert len(list((root / SHARED_DIRNAME / "windows").iterdir())) == 1
        assert RunArtifacts(root / "ds" / "a").load_windows().features.shape == (20, 4)

    def test_gc_on_an_intact_sweep_removes_nothing(self, tmp_path):
        root = self._sweep(tmp_path)
        before = sorted(p.name for p in (root / SHARED_DIRNAME / "windows").iterdir())

        self._script().gc(root, dry_run=False)

        assert sorted(p.name for p in (root / SHARED_DIRNAME / "windows").iterdir()) == before

    def test_an_in_flight_staging_dir_is_left_alone(self, tmp_path):
        """A concurrent put_written stages into `_shared/<stage>/.<name>.*`."""
        root = self._sweep(tmp_path)
        staging = root / SHARED_DIRNAME / "windows" / ".windows.inflight"
        staging.mkdir()

        self._script().gc(root, dry_run=False)

        assert staging.is_dir()

    def test_dry_run_removes_nothing(self, tmp_path):
        root = self._sweep(tmp_path)
        shutil.rmtree(root / "ds" / "c")

        self._script().gc(root, dry_run=True)

        assert len(list((root / SHARED_DIRNAME / "windows").iterdir())) == 2


class TestCompression:
    """`reduced` is written compressed; the uncompressed form still reads.

    Sparse SAE codes are the reason: 95-99.7% zeros, so 14-175x smaller. The
    entries already on disk are .npy and stay that way, which is only safe if
    readers accept both.
    """

    @staticmethod
    def _sparse(n: int = 400, d: int = 256, k: int = 8) -> np.ndarray:
        rng = np.random.default_rng(0)
        out = np.zeros((n, d), dtype=np.float32)
        for row in range(n):
            out[row, rng.choice(d, size=k, replace=False)] = 1.0
        return out

    def test_only_the_compressed_form_is_recognised(self, tmp_path):
        """Back-compat was removed after every tree was converted, so an
        uncompressed array is simply not an artifact any more — better a clear
        absence than two definitions of what counts (which is what let a partial
        local set pass as complete)."""
        run = tmp_path / "job"
        run.mkdir()
        np.save(run / "reduced.npy", self._sparse())

        assert RunArtifacts(run).array_path("reduced") is None
        with pytest.raises(FileNotFoundError):
            RunArtifacts(run).load_array("reduced")

    def test_reduced_is_stored_compressed_and_reads_back_exactly(self, tmp_path):
        array = self._sparse()
        RunArtifacts(tmp_path / "job").save_array("reduced", array)

        assert (tmp_path / "job" / "reduced.npz").exists()
        assert not (tmp_path / "job" / "reduced.npy").exists()
        assert np.array_equal(RunArtifacts(tmp_path / "job").load_array("reduced"), array)

    def test_a_sparse_code_actually_gets_much_smaller(self, tmp_path):
        array = self._sparse()
        RunArtifacts(tmp_path / "job").save_array("reduced", array)

        compressed = (tmp_path / "job" / "reduced.npz").stat().st_size
        assert compressed < array.nbytes / 5, (compressed, array.nbytes)

    def test_load_result_finds_a_compressed_reduced(self, tmp_path):
        """It used to probe reduced.npy only, so every new run loaded None."""
        array = self._sparse()
        RunArtifacts(tmp_path / "job").save_array("reduced", array)

        assert np.array_equal(RunArtifacts(tmp_path / "job").load_result().reduced, array)

    def test_shared_and_compressed_together(self, tmp_path):
        array = self._sparse()
        RunArtifacts(tmp_path / "job", shared_root=tmp_path).save_array("reduced", array)

        entries = list((tmp_path / SHARED_DIRNAME / "reduced").iterdir())
        assert len(entries) == 1
        assert (entries[0] / "reduced.npz").exists()
        assert np.array_equal(RunArtifacts(tmp_path / "job").load_array("reduced"), array)

    def test_per_job_arrays_stay_uncompressed(self, tmp_path):
        """labels.npy in particular: is_run_dir looks for that exact name."""
        RunArtifacts(tmp_path / "job").save_array("labels", np.zeros(10, dtype=np.int64))

        assert (tmp_path / "job" / "labels.npy").exists()
        assert not (tmp_path / "job" / "labels.npz").exists()
