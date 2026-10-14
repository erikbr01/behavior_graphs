"""The episode bootstrap in ``consolidate_progress_results.py`` has to be paired.

``save_curves`` writes rows in fold-accumulation order, so a method scored with
``KFoldSplit(3)`` carries a shuffled ``idx`` while one scored with a single
``AlignedSplit`` fold carries ``0..n-1``. Resampling *positions* therefore
compares episode k of one method against a different episode of another. Nothing
downstream can notice: the Holm-adjusted p-values and the CLD letters come out
looking exactly as reasonable as if the pairing had been real.

The fixture below is built so an unpaired bootstrap gets the answer wrong: the
two methods are identical episode-for-episode, so every paired difference is
exactly zero and no pair can separate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from behavior_graphs.progress_estimation.io import save_curves

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def consolidate():
    spec = importlib.util.spec_from_file_location(
        "consolidate_progress_results", REPO / "scripts" / "consolidate_progress_results.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ramp(rate: float, n: int = 12) -> np.ndarray:
    """A monotone curve whose steepness is the episode's 'quality'."""
    return np.clip(np.linspace(0.0, rate, n), 0.0, 1.0)


def _write(curves_dir: Path, prefix: str, tag: str, order: list[int], rates: dict[int, float]):
    """One method's curves, written in `order` -- the fold order it was scored in.

    ``tag`` is the registry *infix* (``graph_direct_stage``), which is what
    ``resolve_curve_path`` globs for, not the registry key.
    """
    save_curves(
        curves_dir,
        prefix,
        tag,
        [_ramp(rates[e]) for e in order],
        [e % 3 != 0 for e in order],  # 2/3 succeed, keyed on the episode, not the row
        order,
    )


def test_the_bootstrap_pairs_on_episode_id_not_row_position(consolidate, tmp_path, monkeypatch):
    n = 24
    rng = np.random.default_rng(0)
    rates = {e: float(rng.uniform(0.3, 1.0)) for e in range(n)}

    # Same episodes, same curves, two different row orders -- as a KFold-scored
    # method and a single-fold method end up on disk.
    shuffled = list(rng.permutation(n))
    _write(tmp_path, "ds", "graph_direct_stage", list(range(n)), rates)
    _write(tmp_path, "ds", "graph_direct_absorb", shuffled, rates)
    assert shuffled != list(range(n))  # the fixture is actually exercising the bug

    monkeypatch.setattr(consolidate, "DATASETS", {"ds": "ds"})
    monkeypatch.setattr(consolidate, "METHOD_KEYS", ["graph_stage", "graph_absorb"])
    monkeypatch.setattr(consolidate, "N_BOOT", 200)

    results, methods = consolidate.compute_stats(tmp_path, np.random.default_rng(0))
    assert sorted(methods) == ["graph_absorb", "graph_stage"]

    # Identical episode-for-episode, so on a paired bootstrap every difference is
    # exactly zero: the two must agree to the last digit and share a CLD letter.
    # Pairing by row position compares different episodes, which makes them
    # differ and can separate them.
    for metric in ("voc", "kendall", "pearson", "mae"):
        means = results[metric]["means"]
        assert means["graph_stage"] == pytest.approx(means["graph_absorb"], abs=1e-12), metric
        cld = results[metric]["cld"]
        assert cld["graph_stage"] == cld["graph_absorb"], metric
