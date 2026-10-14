"""Tests for the ABC study's config, splits and cell loader.

The splits tests carry most of the weight, because the failure they guard
against already happened once: the bottles policy trained on 76 of the 87
episodes the study reported as held out, because the study's split and the
policy's episode list were two artefacts that drifted apart.
"""

from __future__ import annotations

import json

import pytest

from behavior_graphs.ablation_study import cells, config, splits


def _stats(n: int, segs_per: int = 10, dur: float = 100.0) -> dict[int, dict[str, float]]:
    return {i: {"episodes": 1.0, "segments": float(segs_per), "duration": dur} for i in range(n)}


def _write_dataset(tmp_path, n_episodes: int, name: str = "toy"):
    """A dataset root with just enough on disk for splits to work."""
    root = tmp_path / name
    (root / "annotations").mkdir(parents=True)
    (root / "episodes").mkdir(parents=True)
    ann = [
        {
            "index": i,
            "wgo_id": f"episode_{i:04d}",
            "fps": 10.0,
            "duration_sec": 100.0,
            "segments": [
                {"start_sec": float(s), "end_sec": float(s + 10), "subtask": f"s{s}"}
                for s in range(0, 100, 10)
            ],
        }
        for i in range(n_episodes)
    ]
    (root / "annotations" / "segments.json").write_text(json.dumps(ann))
    (root / "episodes" / "metadata.yaml").write_text(
        json.dumps({"episodes": [{"wgo_id": f"episode_{i:04d}"} for i in range(n_episodes)]})
    )
    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "name": name,
                "task": "toy_task",
                "root": str(root),
                "sweep_dir": str(tmp_path / "sweep"),
                "split": {"seed": 0, "train": 0.6, "val": 0.2, "test": 0.2},
            }
        )
    )
    return config.load(cfg_path)


class TestSplitSpec:
    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            config.SplitSpec(train=0.6, val=0.2, test=0.3)

    def test_fractions_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            config.SplitSpec(train=1.0, val=0.0, test=0.0)

    def test_a_file_split_skips_fraction_validation(self):
        # A legacy file carries its own proportions; the fractions are unused.
        spec = config.SplitSpec(file="splits.json", train=0.9, val=0.9, test=0.9)
        assert spec.file == "splits.json"


class TestBalancedThreeWay:
    def test_partition_is_disjoint_and_total(self):
        st = _stats(100)
        named = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=0)
        allid = named["train"] + named["val"] + named["test"]
        assert sorted(allid) == sorted(st)
        assert len(set(allid)) == len(allid)

    def test_respects_requested_proportions(self):
        named = splits.balanced_three_way(_stats(100), (0.6, 0.2, 0.2), seed=0)
        assert (len(named["train"]), len(named["val"]), len(named["test"])) == (60, 20, 20)

    def test_balances_segments_not_just_episode_count(self):
        # Half the episodes carry ten times the segments. A partition balanced on
        # count alone would put a wildly uneven share of boundaries in test.
        st = _stats(100)
        for i in range(0, 100, 2):
            st[i]["segments"] = 100.0
        named = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=0)
        total = sum(st[i]["segments"] for i in st)
        share = sum(st[i]["segments"] for i in named["test"]) / total
        assert abs(share - 0.2) < 0.02

    def test_is_deterministic_in_seed(self):
        st = _stats(60)
        a = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=0)
        b = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=0)
        assert splits.digest(a) == splits.digest(b)

    def test_different_seeds_give_different_partitions(self):
        st = _stats(60)
        a = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=0)
        b = splits.balanced_three_way(st, (0.6, 0.2, 0.2), seed=1)
        assert splits.digest(a) != splits.digest(b)


class TestGenerateAndResolve:
    def test_round_trips_through_disk(self, tmp_path):
        cfg = _write_dataset(tmp_path, 50)
        payload = splits.generate(cfg)
        splits.write(cfg, payload)
        named = splits.resolve(cfg)
        assert sorted(named) == ["test", "train", "val"]
        assert splits.digest(named) == payload["sha256"]

    def test_resolve_rejects_overlapping_parts(self, tmp_path):
        cfg = _write_dataset(tmp_path, 30)
        payload = splits.generate(cfg)
        payload["val"] = payload["val"] + [payload["train"][0]]
        splits.write(cfg, payload)
        with pytest.raises(SystemExit, match="overlap"):
            splits.resolve(cfg)

    def test_missing_split_file_says_what_to_run(self, tmp_path):
        cfg = _write_dataset(tmp_path, 20)
        with pytest.raises(SystemExit, match="study splits"):
            splits.resolve(cfg)

    def test_generate_refuses_when_the_split_is_a_file(self, tmp_path):
        cfg = _write_dataset(tmp_path, 20)
        cfg = dataclass_replace(cfg, split=config.SplitSpec(file="splits.json"))
        with pytest.raises(ValueError, match="read from a file"):
            splits.generate(cfg)


class TestRgrlTrainIds:
    def test_excludes_every_held_out_episode(self, tmp_path):
        cfg = _write_dataset(tmp_path, 50)
        splits.write(cfg, splits.generate(cfg))
        order = tmp_path / "abc_episode_order.json"
        order.write_text(json.dumps({"episode_dirs": [f"episode_{i:04d}" for i in range(50)]}))
        payload = splits.rgrl_train_ids(cfg, order)
        named = splits.resolve(cfg)
        assert payload["count"] == len(named["train"])
        held = set(named["val"]) | set(named["test"])
        assert not set(payload["episode_ids"]) & held
        assert payload["n_held_out_excluded"] == len(held)

    def test_pins_the_id_list_with_a_digest(self, tmp_path):
        cfg = _write_dataset(tmp_path, 40)
        splits.write(cfg, splits.generate(cfg))
        order = tmp_path / "order.json"
        order.write_text(json.dumps({"episode_dirs": [f"episode_{i:04d}" for i in range(40)]}))
        a = splits.rgrl_train_ids(cfg, order)
        b = splits.rgrl_train_ids(cfg, order)
        assert a["expected_sha256"] == b["expected_sha256"]
        assert a["split_sha256"] == b["split_sha256"]


class TestCells:
    def _rows(self):
        return [
            cells.Cell(
                dir=f"/j/{i}",
                layer="state" if i % 2 else "rgrl_dit",
                reduction="pca",
                n_components=10,
                clusterer="hdbscan" if i == 0 else "kmeans",
                k=200 if i == 0 else 10,
                width=4,
                stride=4,
                smoothing="none",
                min_segment=str(ms),
                score=0.1 * i + 0.01 * ms,
                density=1.0,
                gain=0.0,
            )
            for i in range(4)
            for ms in range(2)
        ]

    def test_per_job_keeps_the_best_min_segment(self):
        rows = self._rows()
        jobs = cells.per_job(rows)
        assert len(jobs) == 4
        # ms=1 always scores higher here, so every survivor should be that level.
        assert {c.min_segment for c in jobs} == {"1"}

    def test_hdbscan_k_is_tagged_so_it_never_averages_with_a_count(self):
        rows = self._rows()
        labels = {c.k_label() for c in rows}
        assert "mcs200" in labels
        assert "200" not in labels

    def test_eta_squared_is_zero_when_an_axis_explains_nothing(self):
        rows = self._rows()
        # every cell shares one width, so width can explain no variance
        assert cells.eta_squared(rows, "width") == pytest.approx(0.0)

    def test_eta_squared_is_one_when_an_axis_explains_everything(self):
        rows = [
            cells.Cell(
                dir=f"/j/{i}",
                layer=f"L{i}",
                reduction="p",
                n_components=1,
                clusterer="k",
                k=1,
                width=1,
                stride=1,
                smoothing="none",
                min_segment="0",
                score=float(i),
                density=1.0,
                gain=0.0,
            )
            for i in range(5)
        ]
        assert cells.eta_squared(rows, "layer") == pytest.approx(1.0)

    def test_plateau_includes_ties_within_the_margin(self):
        rows = self._rows()
        top = max(c.score for c in rows)
        near = cells.plateau(rows, 0.05)
        assert all(c.score >= top - 0.05 for c in near)
        assert max(c.score for c in near) == top

    def test_levels_sort_numerically_where_possible(self):
        rows = [
            cells.Cell(
                dir=f"/j/{k}",
                layer="s",
                reduction="p",
                n_components=1,
                clusterer="k",
                k=k,
                width=1,
                stride=1,
                smoothing="none",
                min_segment="0",
                score=0.0,
                density=1.0,
                gain=0.0,
            )
            for k in (10, 4, 32, 6)
        ]
        assert cells.levels(rows, "k") == ["4", "6", "10", "32"]


def dataclass_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


class TestConfigLoading:
    def test_ships_configs_for_the_three_reported_tasks(self):
        """The three DROID deployments Tab. III averages over, and only those."""
        assert {"settable", "preparetray", "cooking"} <= set(config.available())

    def test_every_task_generates_sixty_twenty_twenty(self):
        for name in ("settable", "preparetray", "cooking"):
            cfg = config.load(name)
            assert cfg.split.file is None, name
            assert (cfg.split.train, cfg.split.val, cfg.split.test) == (0.6, 0.2, 0.2), name

    def test_the_three_tasks_share_one_k_range(self):
        """A maximum over unequal searches is not a comparison -- see paper_ablation."""
        grids = {config.load(n).grid_for("cond_proj") for n in ("settable", "preparetray")}
        assert grids == {"ablation_droid"}

    def test_unknown_dataset_lists_what_exists(self):
        with pytest.raises(SystemExit, match="available"):
            config.load("no_such_dataset")
