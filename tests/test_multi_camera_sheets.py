"""Multi-camera contact sheets, and the two places a user picks the views.

The point of putting several cameras on one sheet is that a column holds one
moment across views. So the tests that matter are about alignment (the grid, the
timesteps, the row labels) and about the model being *told* what it is looking
at — a two-view sheet with no explanation reads as two phases of the episode.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from behavior_graphs.envs.trajectory_dir import RolloutDirectory, write_rollout_dir
from behavior_graphs.viz.frames import (
    _oriented,
    _view_suffix,
    contact_sheet_for_node,
    episode_views,
    make_contact_sheet,
)
from script_loader import load_script


def _tile(h: int, w: int, value: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


#: An analysis-shaped response, for analyze()'s own JSON contract.
_ANALYSIS = {
    "task_summary": "s",
    "success_modes": [],
    "failure_modes": [],
    "similar_pairs": [],
    "overall_assessment": "a",
}


class _StubClient:
    """A VLM that always answers, optionally recording the prompts it was sent.

    One definition rather than one per test: the ``complete(system, content,
    max_tokens)`` signature is the labeler's contract with every adapter, and four
    copies of it turn one change to that call into a four-site edit.
    """

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __init__(self, response: dict | None = None, prompts: list[str] | None = None) -> None:
        self._response = response or {"reasoning": "r", "label": "a thing", "description": "d"}
        self._prompts = prompts

    def complete(self, system, content, max_tokens=None):
        if self._prompts is not None:
            self._prompts.append(system)
        return json.dumps(self._response)


def _sheet_size(frames, cols: int = 2, tile_gap: int = 8) -> tuple[int, int]:
    from PIL import Image

    sheet = make_contact_sheet(frames, cols=cols, tile_gap=tile_gap)
    return Image.open(io.BytesIO(sheet)).size


def _clips(rollout_dir: Path, cameras: list[str] | None, episodes: range | list[int]) -> None:
    """Write empty clip files in the flat or per-camera media layout."""
    media = rollout_dir / "media"
    for camera in cameras or [None]:
        target = media if camera is None else media / camera
        target.mkdir(parents=True, exist_ok=True)
        for ep in episodes:
            (target / f"ep{ep:04d}_success.mp4").write_bytes(b"")


@pytest.fixture
def rollouts(tmp_path) -> Path:
    """A two-view rollout dir: three episodes, both cameras on all of them."""
    path = tmp_path / "set_table_two_cam"
    write_rollout_dir(
        path, [6, 6, 6], [True, True, False], {"cond_proj": np.zeros((18, 4), np.float32)}
    )
    _clips(path, ["exterior_image_1", "wrist_image_left"], range(3))
    return path


class TestGridHoldsMixedTileSizes:
    """Views of one episode differ in aspect ratio; the grid can't assume one."""

    def test_cells_come_from_the_largest_tile_not_the_first(self):
        # A 4:3 wrist tile first, a 16:9 exterior tile second: laying the grid
        # out on frames[0] would make the sheet too narrow to hold the second.
        assert _sheet_size([_tile(224, 298), _tile(224, 398)]) == (398 * 2 + 8 * 3, 224 + 8 * 2)

    def test_uniform_tiles_lay_out_exactly_as_before(self):
        assert _sheet_size([_tile(224, 224)] * 4) == (224 * 2 + 8 * 3, 224 * 2 + 8 * 3)

    def test_a_blank_cell_does_not_size_the_grid(self):
        # None is a missing frame, not a zero-size tile.
        assert _sheet_size([_tile(224, 398), None]) == (398 * 2 + 8 * 3, 224 + 8 * 2)

    def test_one_portrait_view_does_not_stretch_the_landscape_rows(self):
        """Row height is the row's own, or a 90°-rotated view pads the whole sheet.

        ``camera_transforms`` supports ``rotate: 90``, whose tiles come out
        portrait. Sizing every row to the tallest tile on the sheet would give the
        landscape rows a 398px-tall cell — nearly doubling the sheet's area in dark
        padding, on every node, re-sent in both passes.
        """
        landscape, portrait = _tile(224, 398), _tile(398, 224)

        size = _sheet_size([landscape, landscape, portrait, portrait])

        # Width still global (a column means one timestep on every row); height
        # per row: 224 for the landscape row, 398 for the portrait one.
        assert size == (398 * 2 + 8 * 3, 224 + 398 + 8 * 3)


class TestOrientMatchesTheProxies:
    """A view is shown the way the app shows it, or the label pays for it.

    Through ``norm_transform`` rather than around it: the config dict is what a
    caller actually has, and the pair the sheets apply has to come out of the same
    normalization the ffmpeg filter chain uses.
    """

    def _orient(self, img, transform):
        from behavior_graphs.viz.media import norm_transform

        return _oriented(img, *norm_transform(transform))

    @staticmethod
    def _corner_marked(w: int, h: int):
        from PIL import Image

        img = Image.new("RGB", (w, h), (0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))  # top-left
        return img

    def test_half_turn_moves_the_corner_opposite(self):
        # DROID's wrist camera: rotate 180 in configs/data/droid.yaml.
        img = self._orient(self._corner_marked(4, 3), {"rotate": 180})
        assert img.size == (4, 3)
        assert img.getpixel((3, 2)) == (255, 0, 0)

    def test_quarter_turn_is_clockwise_and_swaps_the_axes(self):
        # _norm_transform documents degrees clockwise; PIL rotates the other
        # way, so a sign slip here would silently tilt every wrist frame.
        img = self._orient(self._corner_marked(4, 3), {"rotate": 90})
        assert img.size == (3, 4)
        assert img.getpixel((2, 0)) == (255, 0, 0)

    def test_mirror_applies_before_rotation(self):
        # Same order as media.transform_filters: hflip carries the top-left pixel
        # to the top-right, and the quarter turn then carries it to the bottom
        # of the last column. Rotating first would land it at (0, 0) instead.
        img = self._orient(self._corner_marked(4, 3), {"rotate": 90, "mirror": True})
        assert img.size == (3, 4)
        assert img.getpixel((2, 3)) == (255, 0, 0)

    @pytest.mark.parametrize(
        "transform", [None, {}, {"rotate": 0}, {"rotate": 45}, {"rotate": True}]
    )
    def test_nothing_to_do_leaves_the_frame_alone(self, transform):
        img = self._orient(self._corner_marked(4, 3), transform)
        assert img.size == (4, 3)
        assert img.getpixel((0, 0)) == (255, 0, 0)


class TestViewResolution:
    def test_no_cameras_asked_for_is_one_clip_as_before(self, rollouts):
        traj = RolloutDirectory(rollouts)
        views = episode_views(traj, 0, None)
        assert len(views) == 1
        # The name is the view video_path actually fell back to, so its
        # configured orientation still applies.
        camera, path = views[0]
        assert camera == traj.camera_names()[0]
        assert path == traj.video_path(0)

    def test_an_explicit_list_restricts_and_orders(self, rollouts):
        traj = RolloutDirectory(rollouts)
        views = episode_views(traj, 1, ["wrist_image_left", "exterior_image_1"])
        assert [c for c, _ in views] == ["wrist_image_left", "exterior_image_1"]
        views = episode_views(traj, 1, ["wrist_image_left"])
        assert [c for c, _ in views] == ["wrist_image_left"]

    def test_a_view_this_episode_lacks_is_skipped_not_fatal(self, tmp_path):
        # A collection can lose one camera partway through; the episodes that
        # do have it are still worth showing.
        path = tmp_path / "partial"
        write_rollout_dir(path, [4, 4], [True, True], {"x": np.zeros((8, 2), np.float32)})
        _clips(path, ["exterior_image_1"], [0, 1])
        _clips(path, ["wrist_image_left"], [0])
        traj = RolloutDirectory(path)
        asked = ["exterior_image_1", "wrist_image_left"]
        assert len(episode_views(traj, 0, asked)) == 2
        assert [c for c, _ in episode_views(traj, 1, asked)] == ["exterior_image_1"]

    def test_the_default_view_is_named_after_the_clip_it_returns(self, tmp_path):
        """Not ``camera_names()[0]``, which is only the same view when it has the clip.

        ``video_path`` falls back to the first view that has *this episode*. Named
        from the camera list instead, an episode the exterior camera missed would
        pair the wrist clip with the exterior's name — and so with the exterior's
        (absent) rotation, sending the wrist view to the VLM upside down.
        """
        path = tmp_path / "exterior_gap"
        write_rollout_dir(path, [4, 4], [True, True], {"x": np.zeros((8, 2), np.float32)})
        _clips(path, ["exterior_image_1"], [0])
        _clips(path, ["wrist_image_left"], [0, 1])
        traj = RolloutDirectory(path)

        ((camera, clip),) = episode_views(traj, 1, None)

        assert camera == "wrist_image_left", "the name has to match the clip"
        assert clip == traj.video_path(1)
        assert camera != traj.camera_names()[0]

    def test_an_unknown_name_yields_no_view(self, rollouts):
        assert episode_views(RolloutDirectory(rollouts), 0, ["nope"]) == []

    def test_no_media_at_all(self, tmp_path):
        path = tmp_path / "bare"
        write_rollout_dir(path, [4], [True], {"x": np.zeros((4, 2), np.float32)})
        traj = RolloutDirectory(path)
        assert episode_views(traj, 0, None) == []
        assert episode_views(traj, 0, ["exterior_image_1"]) == []

    @pytest.mark.parametrize(
        ("camera", "multi_view", "expected"),
        [("wrist", True, " [wrist]"), ("wrist", False, "")],
    )
    def test_row_labels_only_name_the_view_when_there_are_several(
        self, camera, multi_view, expected
    ):
        assert _view_suffix(camera, multi_view) == expected


def _run_dir(tmp_path, rollout_dir: Path, lengths: list[int] | None = None) -> Path:
    """A run whose windows put every episode's frames 0-5 on node 0."""
    from behavior_graphs.core.graph import BehaviorGraph
    from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
    from behavior_graphs.pipeline.artifacts import RunArtifacts

    lengths = lengths or [6, 6, 6]
    windows = [WindowInfo(episode_idx=ep, start=0, end=6) for ep in range(len(lengths))]
    labels = np.zeros(len(windows), dtype=np.int64)
    window_set = WindowSet(
        features=np.zeros((len(windows), 4), np.float32),
        windows=windows,
        episodes=episodes_from_lengths(lengths, [True] * (len(lengths) - 1) + [False]),
    )
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.save_windows(window_set)
    artifacts.save_array("labels", labels)
    artifacts.save_graph(BehaviorGraph.from_window_labels(labels, window_set))
    artifacts.save_metrics({"rollout_dir": str(rollout_dir)})
    return artifacts.run_dir


class TestSheetForNodeAcrossViews:
    """Frames are faked: what's under test is the layout, not pyav."""

    @staticmethod
    def _record(monkeypatch) -> list[dict]:
        calls: list[dict] = []

        def fake_extract(video_path, frame_indices, **kw):
            calls.append({"path": Path(video_path), "ts": list(frame_indices), **kw})
            return [_tile(224, 398) for _ in frame_indices]

        monkeypatch.setattr("behavior_graphs.viz.frames.extract_episode_frames", fake_extract)
        return calls

    @staticmethod
    def _labels(monkeypatch) -> dict:
        seen: dict = {}

        def fake_sheet(frames, cols=5, row_labels=None, **kw):
            seen["frames"] = list(frames)
            seen["row_labels"] = dict(row_labels or {})
            return b"jpeg"

        monkeypatch.setattr("behavior_graphs.viz.frames.make_contact_sheet", fake_sheet)
        return seen

    def test_each_view_gets_its_own_labelled_row_group(self, tmp_path, rollouts, monkeypatch):
        self._record(monkeypatch)
        seen = self._labels(monkeypatch)
        contact_sheet_for_node(
            _run_dir(tmp_path, rollouts),
            rollouts,
            0,
            frames_per_episode=5,
            cameras=["exterior_image_1", "wrist_image_left"],
        )
        labels = list(seen["row_labels"].values())
        assert len(labels) == 6, "3 episodes x 2 views"
        assert any("Episode 0 [exterior_image_1]" in lb for lb in labels)
        assert any("Episode 0 [wrist_image_left]" in lb for lb in labels)
        # Grouped by episode, so an episode's views sit next to each other.
        assert [lb.split()[1] for lb in labels] == ["0", "0", "1", "1", "2", "2"]

    def test_views_are_sampled_at_the_same_timesteps(self, tmp_path, rollouts, monkeypatch):
        calls = self._record(monkeypatch)
        self._labels(monkeypatch)
        contact_sheet_for_node(
            _run_dir(tmp_path, rollouts),
            rollouts,
            0,
            frames_per_episode=4,
            cameras=["exterior_image_1", "wrist_image_left"],
        )
        by_episode: dict[str, list[list[int]]] = {}
        for call in calls:
            by_episode.setdefault(call["path"].name, []).append(call["ts"])
        for name, timesteps in by_episode.items():
            assert len({tuple(t) for t in timesteps}) == 1, f"{name} sampled differently per view"

    def test_each_view_is_extracted_with_its_own_transform(self, tmp_path, rollouts, monkeypatch):
        calls = self._record(monkeypatch)
        self._labels(monkeypatch)
        contact_sheet_for_node(
            _run_dir(tmp_path, rollouts),
            rollouts,
            0,
            cameras=["exterior_image_1", "wrist_image_left"],
            view_transforms={"wrist_image_left": {"rotate": 180}},
        )
        wrist = [c for c in calls if "wrist" in str(c["path"])]
        exterior = [c for c in calls if "exterior" in str(c["path"])]
        assert wrist and exterior
        assert all(c["transform"] == {"rotate": 180} for c in wrist)
        assert all(not c["transform"] for c in exterior)

    def test_several_views_pad_missing_frames_to_stay_in_register(
        self, tmp_path, rollouts, monkeypatch
    ):
        # A camera that stopped recording early would otherwise return a
        # shorter list and slide its whole row left.
        calls = self._record(monkeypatch)
        self._labels(monkeypatch)
        run_dir = _run_dir(tmp_path, rollouts)
        contact_sheet_for_node(
            run_dir, rollouts, 0, cameras=["exterior_image_1", "wrist_image_left"]
        )
        assert all(c["pad_missing"] for c in calls)

        calls.clear()
        contact_sheet_for_node(run_dir, rollouts, 0, cameras=["exterior_image_1"])
        assert not any(c["pad_missing"] for c in calls), "one view keeps the old behaviour"

    def test_an_episode_missing_a_view_still_names_the_one_it_has(self, tmp_path, monkeypatch):
        """Every row of a multi-view sheet is qualified, even a lone one.

        The prompt tells the model each episode appears once per view. An
        unqualified ``Episode 1 - frames …`` inside that sheet leaves it guessing
        which angle the row is, or reading it as one of the two it expects.
        """
        path = tmp_path / "partial"
        write_rollout_dir(path, [6, 6], [True, True], {"x": np.zeros((12, 4), np.float32)})
        _clips(path, ["exterior_image_1"], [0, 1])
        _clips(path, ["wrist_image_left"], [0])
        calls = self._record(monkeypatch)
        seen = self._labels(monkeypatch)

        contact_sheet_for_node(
            _run_dir(tmp_path, path, lengths=[6, 6]),
            path,
            0,
            cameras=["exterior_image_1", "wrist_image_left"],
        )

        labels = list(seen["row_labels"].values())
        assert len(labels) == 3, "2 views of ep0, 1 of ep1"
        assert all("[" in lb for lb in labels), labels
        assert "Episode 1 [exterior_image_1]" in labels[2]
        # And its frames still pad, so its columns line up with ep0's rows.
        assert all(c["pad_missing"] for c in calls)

    def test_single_view_labels_are_unchanged(self, tmp_path, rollouts, monkeypatch):
        self._record(monkeypatch)
        seen = self._labels(monkeypatch)
        contact_sheet_for_node(_run_dir(tmp_path, rollouts), rollouts, 0)
        labels = list(seen["row_labels"].values())
        assert len(labels) == 3
        assert all("[" not in lb for lb in labels)
        assert labels[0].startswith("Episode 0 - frames ")

    def test_a_view_with_no_readable_frames_is_dropped_not_labelled(
        self, tmp_path, rollouts, monkeypatch
    ):
        def fake_extract(video_path, frame_indices, **kw):
            if "wrist" in str(video_path):
                return [None] * len(frame_indices)
            return [_tile(224, 398) for _ in frame_indices]

        monkeypatch.setattr("behavior_graphs.viz.frames.extract_episode_frames", fake_extract)
        seen = self._labels(monkeypatch)
        contact_sheet_for_node(
            _run_dir(tmp_path, rollouts),
            rollouts,
            0,
            cameras=["exterior_image_1", "wrist_image_left"],
        )
        labels = list(seen["row_labels"].values())
        assert labels and all("wrist" not in lb for lb in labels)


class TestLabelerIsToldWhatItIsLookingAt:
    """The prompt half. Without it a two-view sheet reads as two phases."""

    @staticmethod
    def _labeler_prompts(tmp_path, rollouts, cameras) -> list[str]:
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        prompts: list[str] = []

        VLMLabeler(
            model="fake",
            client=_StubClient(prompts=prompts),
            task_description="set the table",
            cameras=cameras,
        ).label(result, run_dir=run_dir, rollout_dir=rollouts)
        return prompts

    def test_both_passes_are_warned_the_rows_are_simultaneous(self, tmp_path, rollouts):
        prompts = self._labeler_prompts(
            tmp_path, rollouts, ["exterior_image_1", "wrist_image_left"]
        )
        assert prompts, "expected draft + refine calls"
        for i, prompt in enumerate(prompts):
            assert "SAME timesteps" in prompt, f"call {i} was not told what the rows mean"
            assert "wrist_image_left" in prompt, f"call {i} does not name the views"

    @pytest.mark.parametrize("cameras", [None, ["exterior_image_1"]])
    def test_a_single_view_run_says_nothing_about_views(self, tmp_path, rollouts, cameras):
        for prompt in self._labeler_prompts(tmp_path, rollouts, cameras):
            assert "camera view" not in prompt

    def test_an_empty_camera_list_is_read_as_the_default(self):
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        # Not "no views", which would silently produce text-only labels.
        assert VLMLabeler(model="fake", cameras=[]).cameras is None

    def test_the_views_reach_the_sheet_builder(self, tmp_path, rollouts, monkeypatch):
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        seen: list[dict] = []

        def fake_sheet(**kw):
            seen.append(kw)
            return b"jpeg"

        monkeypatch.setattr("behavior_graphs.viz.frames.contact_sheet_for_node", fake_sheet)

        VLMLabeler(
            model="fake",
            client=_StubClient(),
            task_description="set the table",
            cameras=["exterior_image_1", "wrist_image_left"],
            view_transforms={"wrist_image_left": {"rotate": 180}},
        ).label(result, run_dir=run_dir, rollout_dir=rollouts)

        assert seen, "no contact sheet was built"
        assert all(k["cameras"] == ["exterior_image_1", "wrist_image_left"] for k in seen)
        assert all(k["view_transforms"] == {"wrist_image_left": {"rotate": 180}} for k in seen)

    def test_the_pipeline_stage_still_orients_its_views(self, tmp_path, rollouts, monkeypatch):
        """Hydra can set ``cameras:`` but has nowhere to put a transform map.

        Left to the caller alone, ``labeling.cameras=[…,wrist_image_left]`` sent
        every wrist frame to the model upside down — the config knob was
        documented and half broken.
        """
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        seen: list[dict] = []
        monkeypatch.setattr(
            "behavior_graphs.viz.streamlit.data.camera_transforms",
            lambda dataset: {"wrist_image_left": {"rotate": 180}},
        )
        monkeypatch.setattr(
            "behavior_graphs.viz.frames.contact_sheet_for_node",
            lambda **kw: seen.append(kw) or b"jpeg",
        )

        VLMLabeler(
            model="fake",
            client=_StubClient(),
            task_description="set the table",
            cameras=["exterior_image_1", "wrist_image_left"],
        ).label(result, run_dir=run_dir, rollout_dir=rollouts)

        assert seen, "no contact sheet was built"
        assert all(k["view_transforms"] == {"wrist_image_left": {"rotate": 180}} for k in seen)

    def test_a_caller_that_passes_transforms_is_not_second_guessed(self, tmp_path, rollouts):
        """An explicit map wins: the app resolved it for the group it is displaying."""
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        labeler = VLMLabeler(model="fake", view_transforms={"wrist_image_left": {"rotate": 90}})
        transforms, _ = labeler._view_config(rollouts)
        assert transforms == {"wrist_image_left": {"rotate": 90}}

    def test_analyze_passes_a_tile_size_at_all(self, tmp_path, rollouts, monkeypatch):
        """It used to omit tile_px, which is a required positional: a TypeError
        out of analyze() on every run that had frames to show."""
        from behavior_graphs.core.node_labels import make_label_entry, upsert_node_labels
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        upsert_node_labels(run_dir, {0: make_label_entry(label="a thing", source="test")})
        monkeypatch.setattr(
            "behavior_graphs.viz.frames.contact_sheet_for_node", lambda **kw: b"jpeg"
        )

        analysis = VLMLabeler(model="fake", client=_StubClient(_ANALYSIS), tile_px=320).analyze(
            result, run_dir=run_dir, rollout_dir=rollouts
        )
        assert analysis["task_summary"] == "s"

    def test_a_camera_no_view_matches_stops_the_run(self, tmp_path, rollouts):
        """Only the CLI validated names; Hydra and the chat tool did not.

        An unmatched name is not inert: every episode resolves to no clip, so each
        node's sheet is the blank placeholder make_contact_sheet returns for no
        frames — a *valid* JPEG, so it reads as a real sheet all the way to the API
        and the run pays full price to label black squares.
        """
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        labeler = VLMLabeler(
            model="fake", client=_StubClient(), cameras=["exterior_image_1", "wrist_image"]
        )

        with pytest.raises(ValueError) as exc:
            labeler.label(result, run_dir=run_dir, rollout_dir=rollouts)

        assert "wrist_image" in str(exc.value)
        assert "wrist_image_left" in str(exc.value), "the error should list what is available"

    def test_a_failed_sheet_is_not_remembered_as_no_sheet(self, tmp_path, rollouts, monkeypatch):
        """A transient failure must not decide the node has no frames all run.

        The cache exists so analyze() reuses label()'s work; caching the failure
        would make one pyav hiccup permanent for that node.
        """
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        calls: list[int] = []

        def flaky(**kw):
            calls.append(kw["node_id"])
            if len(calls) == 1:
                raise RuntimeError("pyav said no")
            return b"jpeg"

        monkeypatch.setattr("behavior_graphs.viz.frames.contact_sheet_for_node", flaky)
        labeler = VLMLabeler(model="fake", client=_StubClient())

        assert labeler._sheet_b64(run_dir, rollouts, 0) is None
        assert labeler._sheet_b64(run_dir, rollouts, 0) is not None, "the failure was cached"

    def test_analyze_reuses_the_sheets_label_just_built(self, tmp_path, rollouts, monkeypatch):
        """Same run, same nodes, same settings — building them twice is pure waste.

        Every sampled frame of every view decoded, resized and composited a second
        time, per node; ``scripts/label_nodes.py --analyze`` does both in one
        process.
        """
        from behavior_graphs.pipeline.artifacts import RunArtifacts
        from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

        run_dir = _run_dir(tmp_path, rollouts)
        result = RunArtifacts(run_dir).load_result()
        built: list[int] = []
        monkeypatch.setattr(
            "behavior_graphs.viz.frames.contact_sheet_for_node",
            lambda **kw: built.append(kw["node_id"]) or b"jpeg",
        )

        labeler = VLMLabeler(model="fake", client=_StubClient(), task_description="set the table")
        labeler.label(result, run_dir=run_dir, rollout_dir=rollouts)
        after_label = list(built)
        labeler._client = _StubClient(_ANALYSIS)
        labeler.analyze(result, run_dir=run_dir, rollout_dir=rollouts)

        assert built == after_label, "analyze() rebuilt sheets label() had already built"
        assert len(built) == len(set(built)), "one sheet per node, not one per pass"


label_nodes = load_script("label_nodes")


class TestScriptCameraFlag:
    """`--camera` is validated up front: an unmatched view buys empty sheets."""

    def test_valid_views_pass_through_in_order(self, rollouts):
        cameras = label_nodes._resolve_cameras(rollouts, ["wrist_image_left", "exterior_image_1"])
        assert cameras == ["wrist_image_left", "exterior_image_1"]

    def test_no_flag_leaves_the_default_view_to_the_labeler(self, rollouts):
        """``None``, not the primary view spelled out.

        A one-element list also *restricts*: an episode that camera missed would
        contribute no row, where the default falls back to a view that has it.
        """
        assert label_nodes._resolve_cameras(rollouts, None) is None

    def test_an_unknown_view_exits_rather_than_labelling_without_it(self, rollouts, capsys):
        with pytest.raises(SystemExit) as exc:
            label_nodes._resolve_cameras(rollouts, ["exterior_image_1", "wrsit_image_left"])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "wrsit_image_left" in err
        assert "wrist_image_left" in err, "the error should list what is available"

    def test_list_cameras_names_every_view(self, rollouts):
        assert label_nodes._camera_names(rollouts) == ["exterior_image_1", "wrist_image_left"]
