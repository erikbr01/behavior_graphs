"""Tests for the demo app's graph-construction modes.

Covers the graph-JSON serialization used by the SVG component (probability
floor + rare-edge flagging) and drives the graph demo page end-to-end with
streamlit's AppTest: count-pruning default, switch to state merging, alpha
slider.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.core.graph import (  # noqa: E402
    FAILURE_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths  # noqa: E402
from behavior_graphs.pipeline.artifacts import RunArtifacts  # noqa: E402
from behavior_graphs.viz.streamlit.user_study.graph_plot import _build_graph_json  # noqa: E402

PAGE = Path(__file__).parent.parent / "apps" / "demo" / "_pages" / "graph_demo.py"

# 120 successes take 0 -> 1 -> 2; one failure diverges to 0 -> 3, so the
# outlier edges have prob 1/121 < the renderer's default 1% floor.
_SEQS = [[0, 1, 2]] * 120 + [[0, 3]]
_OUTCOMES = [True] * 120 + [False]


def _make_window_set(label_seqs, successes, dim=4):
    lengths = [len(seq) for seq in label_seqs]
    episodes = episodes_from_lengths(lengths, successes)
    windows = []
    for ep_idx, seq in enumerate(label_seqs):
        for t in range(len(seq)):
            windows.append(WindowInfo(episode_idx=ep_idx, start=t, end=t + 1))
    rng = np.random.default_rng(0)
    features = rng.normal(size=(len(windows), dim)).astype(np.float32)
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in label_seqs])
    return labels, WindowSet(features=features, windows=windows, episodes=episodes)


@pytest.fixture()
def rare_failure_graph():
    labels, windows = _make_window_set(_SEQS, _OUTCOMES)
    return BehaviorGraph.from_window_labels(labels, windows)


class TestBuildGraphJson:
    @staticmethod
    def _pos(graph):
        return {nid: (0.0, float(i)) for i, nid in enumerate(graph.nodes)}

    def test_default_prob_floor_hides_outlier_edge(self, rare_failure_graph):
        data = json.loads(_build_graph_json(rare_failure_graph, self._pos(rare_failure_graph)))
        edges = {(e["src"], e["tgt"]) for e in data["edges"]}
        assert (0, 3) not in edges  # 1/121 < the default 1% floor

    def test_zero_floor_keeps_and_flags_rare_edges(self, rare_failure_graph):
        data = json.loads(
            _build_graph_json(
                rare_failure_graph,
                self._pos(rare_failure_graph),
                prob_floor=0.0,
                rare_edge_max_count=2,
            )
        )
        edges = {(e["src"], e["tgt"]): e for e in data["edges"]}
        assert edges[(0, 3)].get("rare") is True
        assert edges[(3, FAILURE_NODE_ID)].get("rare") is True
        assert "rare" not in edges[(0, 1)]  # high-count edges unflagged

    def test_rare_flag_off_by_default(self, rare_failure_graph):
        data = json.loads(
            _build_graph_json(rare_failure_graph, self._pos(rare_failure_graph), prob_floor=0.0)
        )
        assert all("rare" not in e for e in data["edges"])


@pytest.fixture()
def demo_run_dir(tmp_path):
    """Synthetic run dir with a 1-in-121 divergent failure."""
    labels, windows = _make_window_set(_SEQS, _OUTCOMES)
    graph = BehaviorGraph.from_window_labels(labels, windows)
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.save_windows(windows)
    artifacts.save_array("labels", labels)
    artifacts.save_graph(graph)
    artifacts.save_metrics({"rollout_dir": str(tmp_path / "rollouts")})
    return artifacts.run_dir


class TestGraphDemoApp:
    def _open_markov_view(self, run_dir) -> AppTest:
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.run()
        assert not at.exception
        viz = next(sb for sb in at.selectbox if sb.label == "Visualization")
        viz.set_value("markov_svg_bfs")
        at.run()
        assert not at.exception
        return at

    def test_none_mode_default(self, demo_run_dir, monkeypatch):
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUN_DIR", str(demo_run_dir))
        at = self._open_markov_view(demo_run_dir)
        mode = next(r for r in at.radio if r.label == "Simplification")
        assert mode.value == "None (folding only)"
        # No count/state-merging simplification is active by default — folding
        # (tree-only) is the only default simplification, so nothing is hidden.
        # And with nothing hidden the page says nothing: a caption reporting two
        # zeroes is a row of the page spent on "no news".
        assert not any("edges hidden" in (c.value or "") for c in at.caption)

    def test_flicker_smoothing_is_reachable_without_state_merging(self, demo_run_dir, monkeypatch):
        """It used to be nested inside the state-merging branch.

        So the only way to stop labels oscillating was to also change how the
        graph was constructed — though smoothing runs before any graph is built
        and every mode inherits its labels.
        """
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUN_DIR", str(demo_run_dir))
        at = self._open_markov_view(demo_run_dir)

        assert next(r for r in at.radio if r.label == "Simplification").value.startswith("None")
        smooth = next(c for c in at.checkbox if c.label.startswith("Smooth label flicker"))
        # Off by default: over sweep_all it shrinks the tree on 60% of runs and
        # enlarges it on 26%, so there is no default that is right in general.
        assert smooth.value is False

        smooth.set_value(True).run()
        assert not at.exception
        # And it says what it did, rather than silently rewriting which windows —
        # and so which video clips — belong to each node.
        assert any("Flicker smoothing on" in (c.value or "") for c in at.caption)

    def test_count_pruning_mode(self, demo_run_dir, monkeypatch):
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUN_DIR", str(demo_run_dir))
        at = self._open_markov_view(demo_run_dir)
        mode = next(r for r in at.radio if r.label == "Simplification")
        mode.set_value("Count pruning (hide rare edges)")
        at.run()
        assert not at.exception
        threshold = next(s for s in at.slider if s.label.startswith("Hide transitions"))
        threshold.set_value(2)
        at.run()
        assert not at.exception
        # The divergent failure path has count 1 < 2, so it gets hidden.
        assert any(
            "edges hidden" in (c.value or "") and "0 edges hidden" not in (c.value or "")
            for c in at.caption
        )

    def test_video_panel_control_only_on_views_that_have_a_panel(self, demo_run_dir, monkeypatch):
        """A control that changes nothing is worse than no control.

        The sunburst and icicle views render no selection panel, so there is no
        panel to place — the radio is offered only where it acts.
        """
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUN_DIR", str(demo_run_dir))
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.run()
        assert not at.exception

        def _panel_radio(app):
            return [r for r in app.radio if r.label == "Video panel"]

        assert _panel_radio(at), "the default trajectory tree has a selection panel"

        viz = next(sb for sb in at.selectbox if sb.label == "Visualization")
        viz.set_value("tree_sunburst")
        at.run()
        assert not at.exception
        assert not _panel_radio(at)

        next(sb for sb in at.selectbox if sb.label == "Visualization").set_value("markov_svg_bfs")
        at.run()
        assert not at.exception
        assert _panel_radio(at)

    def test_the_panel_choice_survives_a_view_without_a_panel(self, demo_run_dir, monkeypatch):
        """Streamlit drops a widget's key on any run that doesn't render it.

        So the choice has to be remembered outside the radio, or a detour through
        the sunburst and back would quietly move the panel to the right again.
        """
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUN_DIR", str(demo_run_dir))
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.run()
        assert not at.exception

        next(r for r in at.radio if r.label == "Video panel").set_value("below")
        at.run()
        assert not at.exception

        for view in ("tree_sunburst", "tree_native_svg"):
            next(sb for sb in at.selectbox if sb.label == "Visualization").set_value(view)
            at.run()
            assert not at.exception

        radio = next(r for r in at.radio if r.label == "Video panel")
        assert radio.value == "below"
