"""Tests for layer auto-detection and the stale-data guard.

Both cover silent-wrong-answer paths: a group whose datasets share no layer
used to sweep under the features config's default, and a dataset re-collected
in place used to keep its old results because nothing in a job's identity
depends on the episodes.
"""

from __future__ import annotations

import json

import pytest

from behavior_graphs.core.types import EpisodeInfo
from behavior_graphs.pipeline.sweep import check_data_digests, data_digest


class _Rollouts:
    def __init__(self, eps):
        self._eps = eps

    def episodes(self):
        return self._eps


def _r(*pairs):
    """Rollouts over (num_samples, success) pairs, using the real EpisodeInfo.

    Deliberately not a hand-rolled stand-in: an earlier stub exposed a
    ``length`` attribute EpisodeInfo doesn't have, so these tests passed while
    every real sweep raised AttributeError.
    """
    start = 0
    eps = []
    for n, s in pairs:
        eps.append(EpisodeInfo(len(eps), n, start, start + n, success=s))
        start += n
    return _Rollouts(eps)


# ------------------------------------------------------------------ digests
def test_digest_is_stable_for_identical_episodes():
    assert data_digest(_r((10, True), (20, False))) == data_digest(_r((10, True), (20, False)))


def test_digest_changes_with_an_added_episode():
    assert data_digest(_r((10, True))) != data_digest(_r((10, True), (20, False)))


def test_digest_changes_when_an_outcome_is_relabelled():
    assert data_digest(_r((10, True))) != data_digest(_r((10, False)))


def test_digest_changes_when_a_length_changes():
    assert data_digest(_r((10, True))) != data_digest(_r((11, True)))


def test_digest_ignores_unlabelled_vs_failed_distinction_not_at_all():
    """None and False are different outcomes and must not collide."""
    assert data_digest(_r((10, None))) != data_digest(_r((10, False)))


# ------------------------------------------------------------- staleness
def test_first_sweep_of_a_dataset_records_its_digest(tmp_path):
    out = check_data_digests(tmp_path, {"lift": _r((10, True))})
    assert set(out) == {"lift"}


def test_unchanged_dataset_passes(tmp_path):
    rolls = {"lift": _r((10, True))}
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": data_digest(rolls["lift"])}))
    assert check_data_digests(tmp_path, rolls)["lift"] == data_digest(rolls["lift"])


def _completed_job(sweep_dir, ds_name):
    """A job dir with metrics.json — what makes a dataset's digest matter."""
    d = sweep_dir / ds_name / "abc123"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text("{}")


def test_changed_dataset_is_refused(tmp_path):
    """The bug this guards: 168 stale job dirs would otherwise be kept."""
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": "stale00000000000"}))
    _completed_job(tmp_path, "lift")
    with pytest.raises(ValueError, match="changed since they were swept"):
        check_data_digests(tmp_path, {"lift": _r((10, True))})


def test_refusal_names_the_dir_to_delete(tmp_path):
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": "stale00000000000"}))
    _completed_job(tmp_path, "lift")
    with pytest.raises(ValueError, match=r"delete .*lift"):
        check_data_digests(tmp_path, {"lift": _r((10, True))})


def test_allow_stale_proceeds_but_keeps_the_recorded_digest(tmp_path):
    """Overwriting it would disable the guard for good after one opt-in run."""
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": "stale00000000000"}))
    _completed_job(tmp_path, "lift")
    out = check_data_digests(tmp_path, {"lift": _r((10, True))}, allow_stale=True)
    assert out["lift"] == "stale00000000000"


def test_deleting_the_job_dirs_clears_the_refusal(tmp_path):
    """The remedy the error prescribes has to actually unblock the sweep."""
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": "stale00000000000"}))
    _completed_job(tmp_path, "lift")
    with pytest.raises(ValueError):
        check_data_digests(tmp_path, {"lift": _r((10, True))})
    import shutil

    shutil.rmtree(tmp_path / "lift")
    out = check_data_digests(tmp_path, {"lift": _r((10, True))})
    assert out["lift"] == data_digest(_r((10, True)))


def test_changed_dataset_with_no_completed_jobs_is_not_stale(tmp_path):
    """Nothing to contradict, so a first sweep of new data must not be blocked."""
    (tmp_path / "data_digests.json").write_text(json.dumps({"lift": "stale00000000000"}))
    assert check_data_digests(tmp_path, {"lift": _r((10, True))})["lift"] == data_digest(
        _r((10, True))
    )


def test_other_datasets_digests_are_preserved(tmp_path):
    """Appending one dataset must not drop the rest of the map."""
    (tmp_path / "data_digests.json").write_text(json.dumps({"old": "keepme000000000"}))
    out = check_data_digests(tmp_path, {"new": _r((10, True))})
    assert out["old"] == "keepme000000000"


def test_unreadable_digest_file_is_fatal(tmp_path):
    """Treating it as empty silently re-records every dataset as current.

    That is the unnoticed-stale-results outcome the digests exist to catch, so
    an unreadable record must stop the sweep, not restart the bookkeeping.
    """
    (tmp_path / "data_digests.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        check_data_digests(tmp_path, {"lift": _r((10, True))})


def test_digest_file_of_the_wrong_shape_is_fatal(tmp_path):
    (tmp_path / "data_digests.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="dataset->digest mapping"):
        check_data_digests(tmp_path, {"lift": _r((10, True))})


# ------------------------------------------------------------ detect_layer
def _detect(paths):
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from run_sweep import detect_layers

    return detect_layers(paths)


def _layerless(overrides):
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from run_sweep import selects_layerless_features

    return selects_layerless_features(overrides)


def _ds(tmp_path, name, layers):
    d = tmp_path / name
    (d / "episodes").mkdir(parents=True)
    (d / "episodes" / "metadata.yaml").write_text("episodes: []\n")
    if layers:
        (d / "policy_embeddings").mkdir()
        for ly in layers:
            (d / "policy_embeddings" / f"{ly}.npz").write_bytes(b"")
    return str(d)


def test_layer_common_to_all_is_detected(tmp_path):
    paths = [_ds(tmp_path, "a", ["cond_proj"]), _ds(tmp_path, "b", ["cond_proj"])]
    assert _detect(paths) == ["cond_proj"]


def test_datasets_without_embeddings_are_ignored(tmp_path):
    """data=robomimic: three legacy dirs used to hide the other nine's layer."""
    paths = [
        _ds(tmp_path, "a", ["bottleneck_plan_t0"]),
        _ds(tmp_path, "b", ["bottleneck_plan_t0"]),
        _ds(tmp_path, "legacy", []),
    ]
    assert _detect(paths) == ["bottleneck_plan_t0"]


def test_no_embeddings_anywhere_means_no_override(tmp_path):
    """features=state reads no layer, so None is the honest answer."""
    assert _detect([_ds(tmp_path, "a", []), _ds(tmp_path, "b", [])]) == [None]


def test_disjoint_layers_sweep_every_layer_present(tmp_path):
    """`data=all` must stay usable: one pass per layer, not an abort.

    Raising here made data=all and data=visualize unsweepable without an
    explicit layers=, dry_run included.
    """
    paths = [_ds(tmp_path, "a", ["cond_proj"]), _ds(tmp_path, "b", ["pi05"])]
    assert _detect(paths) == ["cond_proj", "pi05"]


def test_partially_overlapping_layers_cover_the_union(tmp_path):
    paths = [
        _ds(tmp_path, "a", ["cond_proj", "obs_encoder"]),
        _ds(tmp_path, "b", ["pi05"]),
    ]
    assert _detect(paths) == ["cond_proj", "obs_encoder", "pi05"]


def test_layerless_features_stage_is_recognised():
    """features.layer= is a hard compose error against a stage without one."""
    assert _layerless(["pipeline/features=state"])
    assert _layerless(["features=state_action"])


def test_policy_embedding_features_stage_still_takes_a_layer():
    assert not _layerless(["pipeline/features=policy_embedding"])
    assert not _layerless(["clustering.n_clusters=5", "windowing.width=3"])
