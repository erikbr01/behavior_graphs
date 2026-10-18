"""`LoadedRun.reduced` must find the array wherever the pipeline put it.

The regression: `load_run` guarded on `(run_dir / "reduced.npy").exists()`.
`reduced` is in `RunArtifacts._COMPRESSED_ARRAYS`, so it is written as `.npz`,
and a sweep puts it in the shared store rather than the run directory -- the
guard was wrong on both the suffix and the directory. It was therefore False
for every run ever written in this format, `reduced` was always None, and
`graph_demo` fell back to raw features for sticky smoothing: the mode measured
in `stages/graph/smoothing.py` as relabelling 13.7% of windows and growing the
trajectory tree by a median 56%.

On the data this repo was developed against, 3,921 `reduced.npz` files existed
in shared stores while the app believed not one run had reduced features.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.pipeline.artifacts import RunArtifacts


def _minimal_run(run_dir, reduced=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(run_dir)
    if reduced is not None:
        artifacts.save_array("reduced", reduced)
    return artifacts


def test_a_compressed_reduced_array_is_found(tmp_path):
    reduced = np.arange(12, dtype=np.float32).reshape(6, 2)
    artifacts = _minimal_run(tmp_path / "run", reduced)

    path = artifacts.array_path("reduced")
    assert path is not None, "the array was saved but array_path reports it absent"
    assert path.suffix == ".npz", "reduced is a compressed array; the old guard wanted .npy"
    assert not (tmp_path / "run" / "reduced.npy").exists(), (
        "nothing writes reduced.npy -- a guard on that filename is always False"
    )
    np.testing.assert_allclose(artifacts.load_array("reduced"), reduced)


def test_a_run_without_one_reports_absent(tmp_path):
    artifacts = _minimal_run(tmp_path / "run")
    assert artifacts.array_path("reduced") is None
