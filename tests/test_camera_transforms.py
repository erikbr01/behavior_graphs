"""Per-camera display geometry declared in ``configs/data/``.

A camera whose mounting does not match how its frames are stored (DROID's wrist
camera hangs upside down) is corrected on the way to the browser: the config is
read for every panel, so it must never raise, and the correction is baked into
the derived web proxy rather than into the recorded clips or onto the ``<video>``
element — whose native controls would turn upside down with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA  # noqa: E402
from behavior_graphs.viz import media  # noqa: E402
from behavior_graphs.viz.streamlit.components import mp4_player as player_mod  # noqa: E402
from behavior_graphs.viz.streamlit.data import (  # noqa: E402
    _data_group_entries,
    camera_transforms,
    primary_camera,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfgs(tmp_path, monkeypatch):
    """Writes ``configs/data/<name>.yaml`` files and selects one as the group."""

    def _make(group: str, **bodies: str) -> Path:
        root = tmp_path / "configs"
        (root / "data").mkdir(parents=True, exist_ok=True)
        for name, body in bodies.items():
            (root / "data" / f"{name}.yaml").write_text(body)
        monkeypatch.setattr("behavior_graphs.viz.streamlit.data.configs_dir", lambda: root)
        monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", group)
        return root

    return _make


class TestConfig:
    def test_no_cameras_block_means_no_transforms(self, cfgs):
        cfgs("shown", shown="lift: /d/lift\n")

        assert camera_transforms("lift") == {}

    def test_a_group_block_applies_to_its_datasets(self, cfgs):
        cfgs("shown", shown="cameras:\n  wrist: {rotate: 180}\nlift: /d/lift\ncan: /d/can\n")

        assert camera_transforms("lift") == {"wrist": {"rotate": 180}}
        assert camera_transforms("can") == {"wrist": {"rotate": 180}}

    def test_a_dataset_can_override_one_camera(self, cfgs):
        cfgs(
            "shown",
            shown=(
                "cameras:\n  wrist: {rotate: 180}\n"
                "lift: /d/lift\n"
                "can:\n  path: /d/can\n  cameras:\n    wrist: {rotate: 90, mirror: true}\n"
            ),
        )

        assert camera_transforms("lift") == {"wrist": {"rotate": 180}}
        assert camera_transforms("can") == {"wrist": {"rotate": 90, "mirror": True}}

    def test_a_dataset_can_turn_a_group_default_off(self, cfgs):
        """``rotate: 0`` is the only way to say "this one is mounted the other way"."""
        cfgs(
            "shown",
            shown=(
                "cameras:\n  wrist: {rotate: 180}\n"
                "can:\n  path: /d/can\n  cameras:\n    wrist: {rotate: 0}\n"
            ),
        )

        assert camera_transforms("can") == {}

    def test_a_group_block_does_not_reach_a_sibling_groups_datasets(self, cfgs):
        """The point of group scoping: droid's wrist rotation must not hit LIBERO."""
        cfgs(
            "top",
            top="defaults:\n  - droidish\n  - simish\n  - _self_\n",
            droidish="cameras:\n  wrist: {rotate: 180}\nreal_robot: /d/real\n",
            simish="sim_task: /d/sim\n",
        )

        assert camera_transforms("real_robot") == {"wrist": {"rotate": 180}}
        assert camera_transforms("sim_task") == {}

    def test_the_selected_group_corrects_what_it_inherits(self, cfgs):
        cfgs(
            "top",
            top="defaults:\n  - base\n  - _self_\ncameras:\n  wrist: {rotate: 90}\n",
            base="cameras:\n  wrist: {rotate: 180}\nlift: /d/lift\n",
        )

        assert camera_transforms("lift") == {"wrist": {"rotate": 90}}

    @pytest.mark.parametrize(
        "spec",
        [
            "wrist: {rotate: 45}",  # not a quarter turn
            "wrist: {rotate: yes}",  # YAML bool
            "wrist: {rotate: '180'}",  # a string, not a number
            "wrist: {mirror: 1}",  # not a bool
            "wrist: [180]",  # not a mapping
            "wrist: {rotate: 180, rotat: 90}",  # typo alongside a good key
        ],
    )
    def test_a_bad_spec_is_logged_and_dropped_not_raised(self, cfgs, caplog, spec):
        cfgs("shown", shown=f"cameras:\n  {spec}\nlift: /d/lift\n")

        with caplog.at_level(logging.WARNING):
            transforms = camera_transforms("lift")

        assert caplog.records, "a rejected spec has to say so"
        # The one valid key in the last case survives its typo'd neighbour.
        assert transforms in ({}, {"wrist": {"rotate": 180}})

    def test_a_rejected_dataset_spec_falls_back_to_the_group_default(self, cfgs, caplog):
        """A typo must not silently *cancel* the rotation it failed to change."""
        cfgs(
            "shown",
            shown=(
                "cameras:\n  wrist: {rotate: 180}\n"
                "can:\n  path: /d/can\n  cameras:\n    wrist: {rotate: 45}\n"
            ),
        )

        with caplog.at_level(logging.WARNING):
            transforms = camera_transforms("can")

        assert caplog.records, "the rejected spec still has to say so"
        assert transforms == {"wrist": {"rotate": 180}}, (
            "an unusable override is dropped, not treated as `rotate: 0`"
        )

    def test_re_declaring_an_inherited_dataset_keeps_the_group_block(self, cfgs):
        """visualize.yaml re-lists datasets droid.yaml declares; geometry must survive."""
        cfgs(
            "top",
            top="defaults:\n  - droidish\n  - _self_\nreal_robot: /d/real\n",
            droidish="cameras:\n  wrist: {rotate: 180}\nreal_robot: /d/real\n",
        )

        assert camera_transforms("real_robot") == {"wrist": {"rotate": 180}}

    def test_an_unknown_dataset_has_no_transforms(self, cfgs):
        cfgs("shown", shown="cameras:\n  wrist: {rotate: 180}\nlift: /d/lift\n")

        assert camera_transforms("a_dataset_in_no_config") == {}

    def test_no_configs_directory_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "behavior_graphs.viz.streamlit.data.configs_dir", lambda: tmp_path / "nope"
        )

        assert camera_transforms("lift") == {}

    def test_the_repos_droid_group_rotates_the_wrist_cameras(self):
        """The shipped config, since that is what the deployment renders."""
        entries = _data_group_entries("droid", REPO / "configs")

        for name, entry in entries.items():
            assert entry["cameras"] == {
                "wrist_image_left": {"rotate": 180},
                "wrist_image": {"rotate": 180},
            }, name

    def test_the_sim_groups_are_untouched(self):
        entries = _data_group_entries("visualize", REPO / "configs")

        assert entries["libero_goal"]["cameras"] == {}
        assert entries["mimicgen_square_d0"]["cameras"] == {}
        assert entries["libero_goal"]["primary_camera"] is None


class TestDefaultView:
    """``default: true`` on a ``cameras:`` entry names the view shown first."""

    def test_the_flagged_camera_is_the_default(self, cfgs):
        cfgs(
            "shown",
            shown=(
                "cameras:\n  exterior: {default: true}\n  wrist: {rotate: 180}\nlift: /d/lift\n"
            ),
        )

        assert primary_camera("lift") == "exterior"

    def test_the_flag_is_not_a_transform(self, cfgs):
        """It must not reach the media layer, which would key a second variant."""
        cfgs(
            "shown",
            shown="cameras:\n  exterior: {default: true}\n  wrist: {rotate: 180}\nlift: /d/lift\n",
        )

        assert camera_transforms("lift") == {"wrist": {"rotate": 180}}, (
            "a camera listed only to be the default carries no geometry"
        )

    def test_a_flagged_camera_can_also_be_rotated(self, cfgs):
        cfgs("shown", shown="cameras:\n  wrist: {rotate: 180, default: true}\nlift: /d/lift\n")

        assert primary_camera("lift") == "wrist"
        assert camera_transforms("lift") == {"wrist": {"rotate": 180}}

    def test_nothing_flagged_leaves_the_order_alone(self, cfgs):
        cfgs("shown", shown="cameras:\n  wrist: {rotate: 180}\nlift: /d/lift\n")

        assert primary_camera("lift") is None

    def test_a_dataset_can_move_the_default(self, cfgs):
        cfgs(
            "shown",
            shown=(
                "cameras:\n  exterior: {default: true}\n"
                "can:\n  path: /d/can\n  cameras:\n    wrist: {rotate: 180, default: true}\n"
            ),
        )

        assert primary_camera("can") == "wrist"

    def test_overriding_one_cameras_geometry_keeps_the_groups_default(self, cfgs):
        """Saying "this wrist is mounted differently" says nothing about view order."""
        cfgs(
            "shown",
            shown=(
                "cameras:\n  exterior: {default: true}\n  wrist: {rotate: 180}\n"
                "can:\n  path: /d/can\n  cameras:\n    wrist: {rotate: 90}\n"
            ),
        )

        assert primary_camera("can") == "exterior"
        assert camera_transforms("can") == {"wrist": {"rotate": 90}}

    def test_two_defaults_are_logged_and_the_first_wins(self, cfgs, caplog):
        cfgs(
            "shown",
            shown=(
                "cameras:\n  exterior: {default: true}\n  wrist: {default: true}\nlift: /d/lift\n"
            ),
        )

        with caplog.at_level(logging.WARNING):
            chosen = primary_camera("lift")

        assert chosen == "exterior"
        assert any("default" in r.message for r in caplog.records)

    @pytest.mark.parametrize("bad", ["1", "'yes'", "[true]"])
    def test_a_non_boolean_flag_is_logged_and_dropped(self, cfgs, caplog, bad):
        cfgs("shown", shown=f"cameras:\n  exterior: {{default: {bad}}}\nlift: /d/lift\n")

        with caplog.at_level(logging.WARNING):
            chosen = primary_camera("lift")

        assert chosen is None
        assert caplog.records

    def test_the_shipped_droid_group_defaults_to_the_left_exterior(self):
        entries = _data_group_entries("droid", REPO / "configs")

        assert entries, "the droid group lists no datasets"
        for name, entry in entries.items():
            assert entry["primary_camera"] == "exterior_image_1_left", name


class TestPlayerMarkup:
    """The player hands each view's transform to the media layer, not to CSS.

    Transforming the ``<video>`` element would turn its native control bar
    upside down along with the picture, and leave the poster (decoded from the
    clip) disagreeing with what is on screen.
    """

    @pytest.fixture()
    def calls(self, monkeypatch):
        """Records ``(camera, transform)`` per view, for video and poster alike."""
        seen: dict[str, list] = {"video": [], "poster": [], "html": []}

        def _video(src, *, coordinates, height=None, transform=None):
            seen["video"].append((coordinates.rsplit(":", 1)[-1], transform))
            return f"/media/{src.name}"

        def _poster(src, *, coordinates, width=0, transform=None):
            seen["poster"].append((coordinates.rsplit(":", 1)[-1], transform))
            return None

        monkeypatch.setattr(player_mod, "video_src", _video)
        monkeypatch.setattr(player_mod, "poster_src", _poster)
        monkeypatch.setattr(
            player_mod.components, "html", lambda html, **kw: seen["html"].append(html)
        )
        return seen

    def test_each_view_is_served_with_its_own_transform(self, calls, tmp_path):
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4"), ("wrist", tmp_path / "w.mp4")],
            key="k",
            view_transforms={"wrist": {"rotate": 180}},
        )

        assert calls["video"] == [("exterior", None), ("wrist", {"rotate": 180})]
        # The poster is a frame of the same clip, so it gets the same treatment.
        assert calls["poster"] == [("exterior", None), ("wrist", {"rotate": 180})]

    def test_the_element_itself_is_never_transformed(self, calls, tmp_path):
        player_mod.mp4_player(
            videos=[("wrist", tmp_path / "w.mp4")],
            key="k",
            view_transforms={"wrist": {"rotate": 90, "mirror": True}},
        )

        (html,) = calls["html"]
        assert "transform:" not in html  # would rotate the controls with the picture
        assert "data-rot" not in html

    def test_a_transform_for_a_camera_this_episode_lacks_is_ignored(self, calls, tmp_path):
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4")],
            key="k",
            view_transforms={"wrist": {"rotate": 180}},
        )

        assert calls["video"] == [("exterior", None)]

    def test_no_transforms_at_all_passes_none(self, calls, tmp_path):
        player_mod.mp4_player(videos=[("wrist", tmp_path / "w.mp4")], key="k")

        assert calls["video"] == [("wrist", None)]

    def test_the_single_view_call_path_takes_no_transform(self, calls, tmp_path):
        """``video_path=`` callers (the study's fixed clips) are unaffected."""
        player_mod.mp4_player(tmp_path / "a.mp4", key="k")

        assert calls["video"] == [(DEFAULT_CAMERA, None)]


class TestMediaLayer:
    """Baking it in: the cache key, the ffmpeg filters, and what forces a pass."""

    @pytest.fixture(autouse=True)
    def _cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BEHAVIOR_GRAPHS_MEDIA_CACHE", str(tmp_path / "cache"))
        media.cache_dir.cache_clear()
        yield
        media.cache_dir.cache_clear()

    def test_an_untransformed_clip_keeps_its_old_cache_key(self, tmp_path):
        """The deployed cache is append-only, so keys must not widen for everyone."""
        src = tmp_path / "a.mp4"
        src.write_bytes(b"\x00" * 32)

        assert media.proxy_path(src, transform=None) == media.proxy_path(src)
        assert media.proxy_path(src, transform={}) == media.proxy_path(src)
        assert media.proxy_path(src, transform={"rotate": 0}) == media.proxy_path(src)
        assert media.thumb_path(src, transform={"rotate": 0}) == media.thumb_path(src)

    def test_each_orientation_is_its_own_entry(self, tmp_path):
        src = tmp_path / "a.mp4"
        src.write_bytes(b"\x00" * 32)

        keys = {
            media.proxy_path(src, transform=t)
            for t in (None, {"rotate": 180}, {"rotate": 90}, {"rotate": 90, "mirror": True})
        }

        assert len(keys) == 4
        assert media.thumb_path(src, transform={"rotate": 180}) != media.thumb_path(src)

    @pytest.mark.parametrize(
        ("spec", "filters"),
        [
            (None, []),
            ({}, []),
            ({"rotate": 0}, []),
            ({"rotate": 180}, ["hflip", "vflip"]),
            ({"rotate": 90}, ["transpose=1"]),
            ({"rotate": 270}, ["transpose=2"]),
            ({"mirror": True}, ["hflip"]),
            ({"rotate": 90, "mirror": True}, ["hflip", "transpose=1"]),
            ({"rotate": 45}, []),  # rejected upstream; ignored here too
            ({"rotate": True}, []),  # a YAML bool is not 1 degree
        ],
    )
    def test_the_filter_chain(self, spec, filters):
        assert media.transform_filters(spec) == filters

    def test_a_transform_earns_a_pass_a_small_clip_would_not(self, tmp_path, monkeypatch):
        """Rotation is correctness, so it is not skipped as an optimisation."""
        src = tmp_path / "small.mp4"
        src.write_bytes(b"\x00" * 16)  # below _SMALL_SOURCE_BYTES
        ran: list[list[str]] = []

        def _fake_ffmpeg(args, dst):
            ran.append(list(args))
            dst.write_bytes(b"\x00" * 8)
            return True

        monkeypatch.setattr(media, "_run_ffmpeg", _fake_ffmpeg)

        assert media.ensure_proxy(src) == src  # untransformed: not worth a pass
        assert ran == []
        out = media.ensure_proxy(src, transform={"rotate": 180})
        assert out != src
        assert "hflip,vflip" in ",".join(ran[0])

    def test_disabling_proxies_still_disables_them(self, tmp_path, monkeypatch):
        """An explicit operator switch is not overridden by a config rotation."""
        src = tmp_path / "small.mp4"
        src.write_bytes(b"\x00" * 16)
        monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
        monkeypatch.setattr(
            media, "_run_ffmpeg", lambda *a, **k: pytest.fail("should not transcode")
        )

        assert media.ensure_proxy(src, transform={"rotate": 180}) == src

    def test_a_pending_check_asks_about_the_orientation_it_will_build(self, tmp_path, monkeypatch):
        src = tmp_path / "a.mp4"
        src.write_bytes(b"\x00" * 32)
        monkeypatch.setattr(media, "ffmpeg_exe", lambda: "/usr/bin/ffmpeg")
        # Warm only the untransformed proxy and poster.
        media.proxy_path(src).write_bytes(b"\x00")
        media.thumb_path(src, width=media.POSTER_WIDTH).write_bytes(b"\x00")

        assert not media.needs_prep(src)
        assert media.needs_prep(src, transform={"rotate": 180}), (
            "the rotated variant is a different cache entry and still has to be built"
        )

    def test_one_batch_covers_views_with_different_orientations(self, tmp_path, monkeypatch):
        """A panel's views differ, and they must not need a pool each to be warmed."""
        upright = tmp_path / "exterior.mp4"
        rotated = tmp_path / "wrist.mp4"
        for src in (upright, rotated):
            # Over _SMALL_SOURCE_BYTES, so the untransformed one earns a pass on
            # size alone and the batch really does contain two orientations.
            src.write_bytes(b"\x00" * 500_000)
        built: dict[str, str] = {}

        def _fake_ffmpeg(args, dst):
            built[Path(args[1]).name] = ",".join(args)
            dst.write_bytes(b"\x00" * 8)
            return True

        monkeypatch.setattr(media, "_run_ffmpeg", _fake_ffmpeg)

        assert media.prewarm([upright, (rotated, {"rotate": 180})], thumb_widths=()) == (2, 2)
        assert "hflip,vflip" not in built["exterior.mp4"]
        assert "hflip,vflip" in built["wrist.mp4"]
        assert media.proxy_path(upright).exists()
        assert media.proxy_path(rotated, transform={"rotate": 180}).exists()

    def test_a_poster_is_never_the_other_way_up_from_its_own_video(self, tmp_path, monkeypatch):
        """With proxies off the video is served as recorded, so the tile must be too."""
        src = tmp_path / "wrist.mp4"
        src.write_bytes(b"\x00" * 500_000)
        monkeypatch.setenv("BEHAVIOR_GRAPHS_PROXY_DISABLE", "1")
        ran: list[str] = []

        def _fake_ffmpeg(args, dst):
            ran.append(",".join(args))
            dst.write_bytes(b"\x00" * 8)
            return True

        monkeypatch.setattr(media, "_run_ffmpeg", _fake_ffmpeg)

        assert media.ensure_proxy(src, transform={"rotate": 180}) == src  # unrotated video
        media.ensure_thumb(src, width=media.POSTER_WIDTH, transform={"rotate": 180})

        assert not any("hflip" in r for r in ran), "the poster followed the video, not the config"
        # Cached as what was produced, so re-enabling proxies does not hand back
        # this upright tile under the rotated key.
        assert media.thumb_path(src, width=media.POSTER_WIDTH, transform={"rotate": 180}) == (
            media.thumb_path(src, width=media.POSTER_WIDTH)
        )

    def test_a_strip_can_mix_orientations_tile_by_tile(self, tmp_path, monkeypatch):
        """Tiles are each episode's first view, which is not always one camera."""
        upright = tmp_path / "a.mp4"
        rotated = tmp_path / "b.mp4"
        for src in (upright, rotated):
            src.write_bytes(b"\x00" * 500_000)
        built: dict[str, str] = {}

        def _fake_ffmpeg(args, dst):
            built[Path(args[1]).name] = ",".join(args)
            dst.write_bytes(b"\x00" * 8)
            return True

        monkeypatch.setattr(media, "_run_ffmpeg", _fake_ffmpeg)
        monkeypatch.setattr(media, "_streamlit_url", lambda p, c: f"/media/{p.name}")

        urls = media.thumb_srcs([upright, (rotated, {"rotate": 180})], coordinates_prefix="strip:t")

        assert all(urls)
        assert "hflip" not in built["a.mp4"]
        assert "hflip,vflip" in built["b.mp4"]
