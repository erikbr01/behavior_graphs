"""Multi-camera support in the app stack: index, players, sidebar picker.

The layout side of it (``media/<camera>/``) is covered in
``test_trajectory_dir.py``; this is what the Streamlit stack does with it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA, write_rollout_dir  # noqa: E402
from behavior_graphs.viz.streamlit import clips as clips_mod  # noqa: E402
from behavior_graphs.viz.streamlit.components import camera_views  # noqa: E402
from behavior_graphs.viz.streamlit.components import mp4_player as player_mod  # noqa: E402


def _rollout(tmp_path, video_paths):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00" * 16)
    emb = {"l": np.zeros((22, 4), dtype=np.float32)}
    write_rollout_dir(
        tmp_path / "r",
        [10, 12],
        [True, False],
        emb,
        video_paths={ep: spec(src) for ep, spec in video_paths.items()},
    )
    return tmp_path / "r"


class TestIndex:
    def test_single_view_index_is_unchanged(self, tmp_path):
        from behavior_graphs.viz.streamlit.data import _build_mp4_index, mp4_index_cameras

        index = _build_mp4_index(str(_rollout(tmp_path, {0: lambda s: s, 1: lambda s: s})))

        assert mp4_index_cameras(index) == [DEFAULT_CAMERA]
        assert index["by_index"][0]["path"] == "media/ep0000_success.mp4"
        assert index["by_index"][0]["paths"] == {DEFAULT_CAMERA: "media/ep0000_success.mp4"}

    def test_named_views_are_listed_and_joined_per_episode(self, tmp_path):
        from behavior_graphs.viz.streamlit.data import _build_mp4_index, mp4_index_cameras

        rollout = _rollout(
            tmp_path,
            {
                0: lambda s: {"wrist": s, "exterior": s},
                1: lambda s: {"exterior": s},
            },
        )
        index = _build_mp4_index(str(rollout))

        assert mp4_index_cameras(index) == ["exterior", "wrist"]
        assert index["by_index"][0]["paths"] == {
            "exterior": "media/exterior/ep0000_success.mp4",
            "wrist": "media/wrist/ep0000_success.mp4",
        }
        # "path" stays the first view, for the panels that show only one.
        assert index["by_index"][1]["path"] == "media/exterior/ep0001_failure.mp4"
        assert "wrist" not in index["by_index"][1]["paths"]

    def test_the_configured_default_view_leads_the_index(self, tmp_path):
        """What the config buys: the picker order, and the single-view "path"."""
        from behavior_graphs.viz.streamlit.data import _build_mp4_index, mp4_index_cameras

        rollout = _rollout(
            tmp_path,
            {0: lambda s: {"wrist": s, "exterior": s}, 1: lambda s: {"wrist": s, "exterior": s}},
        )

        index = _build_mp4_index(str(rollout), "wrist")

        assert mp4_index_cameras(index) == ["wrist", "exterior"]
        assert index["by_index"][0]["path"] == "media/wrist/ep0000_success.mp4"
        # Unflagged, it falls back to the alphabetical order.
        assert mp4_index_cameras(_build_mp4_index(str(rollout))) == ["exterior", "wrist"]

    def test_cameras_of_an_index_without_the_key(self):
        """An index cached by an older build has no "cameras"."""
        from behavior_graphs.viz.streamlit.data import mp4_index_cameras

        assert mp4_index_cameras({"episodes": [], "by_index": {}}) == []


class TestEpisodeCameraClips:
    def test_orders_and_restricts_by_camera(self, tmp_path):
        entry = {"paths": {"wrist": "media/wrist/a.mp4", "exterior": "media/exterior/a.mp4"}}

        assert clips_mod.episode_camera_clips(entry, tmp_path, ["exterior", "wrist"]) == [
            ("exterior", tmp_path / "media/exterior/a.mp4"),
            ("wrist", tmp_path / "media/wrist/a.mp4"),
        ]
        # A view this episode has no clip for is skipped, not faked.
        assert clips_mod.episode_camera_clips(entry, tmp_path, ["exterior", "gone"]) == [
            ("exterior", tmp_path / "media/exterior/a.mp4"),
        ]

    def test_entry_without_per_camera_paths(self, tmp_path):
        entry = {"path": "media/a.mp4"}

        assert clips_mod.episode_camera_clips(entry, tmp_path) == [
            (DEFAULT_CAMERA, tmp_path / "media/a.mp4")
        ]

    def test_no_media_dir_resolves_to_nothing(self):
        assert clips_mod.episode_camera_clips({"path": "a.mp4"}, None) == []


class TestPlayerMarkup:
    """What ``mp4_player`` emits for one view vs several."""

    @pytest.fixture()
    def rendered(self, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(
            player_mod,
            "video_src",
            lambda src, *, coordinates, height=None, transform=None: f"/media/{src.name}",
        )
        monkeypatch.setattr(
            player_mod, "poster_src", lambda src, *, coordinates, width=0, transform=None: None
        )
        monkeypatch.setattr(player_mod.components, "html", lambda html, **kw: captured.append(html))
        return captured

    def test_single_view_has_no_sync_layer(self, rendered, tmp_path):
        player_mod.mp4_player(tmp_path / "a.mp4", key="k")

        (html,) = rendered
        assert html.count("<video id=") == 1
        assert "BroadcastChannel" not in html

    def test_every_view_is_rendered_and_synced(self, rendered, tmp_path):
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4"), ("wrist", tmp_path / "w.mp4")],
            key="k",
            visible_cameras=["wrist"],
        )

        (html,) = rendered
        # Both views are in the DOM — the picker reveals one by CSS, so a view
        # left out here could never come back without a rerun.
        assert html.count("<video id=") == 2
        assert "/media/e.mp4" in html and "/media/w.mp4" in html
        assert 'data-cam="exterior"' in html and 'data-cam="wrist"' in html
        assert "BroadcastChannel" in html
        # Only the selected one starts out visible.
        exterior, wrist = html.split('data-cam="exterior"')[1], html.split('data-cam="wrist"')[1]
        assert "display:none;" in exterior.split("</div>")[0]
        assert "display:none;" not in wrist.split("</div>")[0]

    def test_peers_are_never_seeked_past_their_own_end(self, rendered, tmp_path):
        """Views within 0.5s align on absolute time, so the longer one overruns.

        Unclamped, the browser clamps the assignment instead, the resulting
        'seeked' no longer matches what was sent, and it reads as the user
        scrubbing — dragging the reference view back and trading seeks over the
        last fraction of a second, so it never reaches 'ended'.
        """
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4"), ("wrist", tmp_path / "w.mp4")],
            key="k",
        )

        (html,) = rendered
        align = html.split("function _align(")[1].split("function ")[0]
        assert "o.duration" in align and "t=o.duration" in align

    def test_the_sync_channel_is_scoped_not_process_wide(self, rendered, tmp_path):
        """A bare global name would reach players in other browser tabs."""
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4"), ("wrist", tmp_path / "w.mp4")],
            key="k",
        )

        (html,) = rendered
        assert f'BroadcastChannel("{camera_views.channel_name()}")' in html

    def test_selection_naming_no_available_view_falls_back(self, rendered, tmp_path):
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4")], key="k", visible_cameras=["wrist"]
        )

        (html,) = rendered
        assert "display:none;" not in html.split('data-cam="exterior"')[1].split("</div>")[0]

    def test_segment_overlay_lands_on_every_view(self, rendered, tmp_path):
        player_mod.mp4_player(
            videos=[("exterior", tmp_path / "e.mp4"), ("wrist", tmp_path / "w.mp4")],
            key="k",
            slice_start=2,
            slice_end=5,
            total_frames=10,
        )

        (html,) = rendered
        assert html.count('id="segbadge_') == 2
        assert html.count(">behavior inactive<") == 2

    def test_unavailable_clips_report_instead_of_rendering(self, rendered, tmp_path, monkeypatch):
        monkeypatch.setattr(player_mod, "video_src", lambda src, **kw: None)
        infos: list[str] = []
        monkeypatch.setattr(player_mod.st, "info", infos.append)

        player_mod.mp4_player(videos=[("exterior", tmp_path / "e.mp4")], key="k")

        assert rendered == []
        assert "e.mp4" in infos[0]

    def test_no_views_at_all_reports_instead_of_raising(self, rendered, monkeypatch):
        """``episode_camera_clips`` returns [] when there is no media dir."""
        infos: list[str] = []
        monkeypatch.setattr(player_mod.st, "info", infos.append)

        player_mod.mp4_player(videos=[], key="k")

        assert rendered == []
        assert infos == ["Video unavailable"]

    def test_needs_a_source(self):
        with pytest.raises(ValueError, match="video_path or videos"):
            player_mod.mp4_player(key="k")


_SELECTOR_APP = """
import streamlit as st
from behavior_graphs.viz.streamlit.components.camera_views import (
    render_camera_selector,
    selected_cameras,
)

CAMERAS = st.session_state.get("_cameras", ["exterior", "wrist"])
with st.sidebar:
    shown = render_camera_selector(CAMERAS)
st.text("shown=" + ",".join(shown))
st.text("live=" + ",".join(selected_cameras(CAMERAS)))
"""


class TestSidebarSelector:
    @staticmethod
    def _texts(at):
        return {t.value.split("=")[0]: t.value.split("=")[1] for t in at.text}

    def test_defaults_to_every_view(self):
        at = AppTest.from_string(_SELECTOR_APP).run()

        assert at.sidebar.multiselect[0].label == "Camera views"
        assert self._texts(at)["shown"] == "exterior,wrist"

    def test_choosing_a_view_narrows_what_the_players_show(self):
        at = AppTest.from_string(_SELECTOR_APP).run()

        at.sidebar.multiselect[0].set_value(["wrist"]).run()

        # Both the return value and the live read the video panels use.
        assert self._texts(at)["shown"] == "wrist"
        assert self._texts(at)["live"] == "wrist"

    def test_empty_selection_keeps_one_view(self):
        at = AppTest.from_string(_SELECTOR_APP).run()

        at.sidebar.multiselect[0].set_value([]).run()

        assert self._texts(at)["shown"] == "exterior"

    def test_single_view_dataset_draws_no_picker(self):
        at = AppTest.from_string(_SELECTOR_APP)
        at.session_state["_cameras"] = ["exterior"]
        at.run()

        assert not at.sidebar.multiselect
        assert self._texts(at)["shown"] == "exterior"

    def test_another_datasets_selection_does_not_apply_here(self):
        """The key is scoped to the view set, so choices can't cross datasets.

        Otherwise a narrowed selection made on the graph demo would silently
        apply to another robot's panels on the pages that read it but draw no
        picker — collapsing them to one view for no visible reason.
        """
        from behavior_graphs.viz.streamlit.components.camera_views import state_key

        at = AppTest.from_string(_SELECTOR_APP)
        at.session_state[state_key(["other_left", "other_right"])] = ["other_left"]
        at.run()

        assert not at.exception
        assert self._texts(at)["shown"] == "exterior,wrist"

    def test_the_choice_is_stored_under_the_view_sets_own_key(self):
        """Same pair of cameras, same key: the choice follows the robot.

        Which is what lets a panel on a page with no picker (Sweep Analysis, the
        study apps) honour a choice made on the graph demo for the *same* views.
        """
        from behavior_graphs.viz.streamlit.components.camera_views import state_key

        at = AppTest.from_string(_SELECTOR_APP).run()
        at.sidebar.multiselect[0].set_value(["wrist"]).run()

        assert at.session_state[state_key(["exterior", "wrist"])] == ["wrist"]
        assert self._texts(at)["live"] == "wrist"
