"""The Policy Comparison page's behavior-graph section.

Two things are worth pinning here. One is the matching: a policy's graph is
found by a *layout convention* (a sweep files its runs under a folder named
after the rollout dataset), and getting it wrong draws one policy's behavior
under another's heading — a mistake nothing on screen would give away. The
other is the narrowing: the section's whole claim is that every graph on it was
built the same way, so an axis that fails to filter, or one that filters a
policy out silently, breaks the comparison rather than merely looking odd.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from behavior_graphs.envs.trajectory_dir import write_rollout_dir  # noqa: E402
from behavior_graphs.viz.streamlit import comparison_graphs as cg  # noqa: E402
from behavior_graphs.viz.streamlit.data import PARAM_LABELS  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "apps" / "demo" / "_pages" / "policy_comparison.py"


def _row(task: str, path: str | None = None, **params) -> dict:
    """One run as the run index reports it: every axis present, mostly None."""
    row = {param: None for param, _ in PARAM_LABELS}
    row.update(task=task, path=path or f"/runs/{task}/{len(params)}", has_metrics=True)
    row.update(params)
    return row


class TestMatchingPoliciesToRuns:
    def test_a_policys_runs_are_the_ones_under_its_sweep_folder(self, monkeypatch):
        rows = [_row("lift", "/runs/lift/b"), _row("lift", "/runs/lift/a"), _row("square")]
        monkeypatch.setattr(cg, "run_index_rows", lambda: rows)

        found = cg.runs_by_dataset(["lift", "square"])

        assert [r["path"] for r in found["lift"]] == ["/runs/lift/a", "/runs/lift/b"]
        assert len(found["square"]) == 1

    def test_a_policy_with_no_runs_still_gets_a_place_in_the_grid(self, monkeypatch):
        """Dropping it makes it indistinguishable from one nobody selected."""
        monkeypatch.setattr(cg, "run_index_rows", lambda: [_row("lift")])

        assert cg.runs_by_dataset(["lift", "never_swept"])["never_swept"] == []

    def test_a_qualified_dataset_key_finds_the_runs_filed_under_its_name(self, monkeypatch):
        """``rollout_outcomes`` qualifies colliding basenames; runs are not."""
        monkeypatch.setattr(cg, "run_index_rows", lambda: [_row("lift")])

        assert len(cg.runs_by_dataset(["sweep_a/lift"])["sweep_a/lift"]) == 1


class TestAxes:
    def test_only_the_axes_the_policies_differ_on_get_a_selector(self):
        runs = {
            "a": [_row("a", k=5, w=16), _row("a", k=10, w=16)],
            "b": [_row("b", k=5, w=16)],
        }

        assert [param for param, _ in cg.varying_axes(runs)] == ["k"]

    def test_the_axes_come_out_in_the_pickers_order(self):
        """So the section reads like the sidebar, K first rather than by dict order."""
        runs = {
            "a": [
                _row("a", k=5, w=8, representation="cond_proj"),
                _row("a", k=10, w=16, representation="dinov2-base patch2x2"),
            ]
        }

        assert [param for param, _ in cg.varying_axes(runs)] == ["representation", "k", "w"]

    def test_the_default_setting_is_the_one_the_most_policies_share(self):
        """Otherwise the grid opens with one graph and three "no run" cards."""
        runs = {
            "a": [_row("a", k=5), _row("a", k=10)],
            "b": [_row("b", k=10)],
            "c": [_row("c", k=10)],
        }

        assert cg.widest_value(runs, "k") == 10

    def test_a_tie_goes_to_the_smaller_value(self):
        runs = {"a": [_row("a", k=5)], "b": [_row("b", k=10)]}

        assert cg.widest_value(runs, "k") == 5

    def test_narrowing_keeps_only_the_runs_built_at_that_setting(self):
        runs = {"a": [_row("a", k=5), _row("a", k=10)], "b": [_row("b", k=10)]}

        narrowed = cg.narrow(runs, "k", 10)

        assert [r["k"] for r in narrowed["a"]] == [10]
        assert [r["k"] for r in narrowed["b"]] == [10]

    def test_a_run_that_never_recorded_the_axis_is_not_a_run_that_agrees(self):
        """A density-clustered run has no K; showing it as the K=10 graph would
        put a differently-built graph under the section's own claim."""
        runs = {"a": [_row("a", k=None), _row("a", k=10)]}

        assert [r["k"] for r in cg.narrow(runs, "k", 10)["a"]] == [10]

    def test_an_axis_no_remaining_run_recorded_offers_nothing_to_choose(self):
        """Which is how a policy swept without the sticky arm stays reachable:
        the page skips an empty axis rather than narrowing everything away."""
        runs = {"a": [_row("a", smoothing="none", sticky_lambda=None)]}

        assert cg.axis_options(runs, "sticky_lambda") == []


class TestResolvingOneRun:
    """Which of a policy's remaining candidates actually gets drawn."""

    @staticmethod
    def _run_dir(root: Path, name: str, rollout_dir: str | None) -> str:
        d = root / name
        d.mkdir(parents=True)
        if rollout_dir is not None:
            (d / "metrics.json").write_text(json.dumps({"rollout_dir": rollout_dir}))
        return str(d)

    def test_the_run_that_names_this_dataset_wins(self, tmp_path):
        """The name match is a convention; metrics.json is the record."""
        wrong = self._run_dir(tmp_path, "aaa", "rollouts/other")
        right = self._run_dir(tmp_path, "bbb", "rollouts/lift")
        candidates = [_row("lift", wrong), _row("lift", right)]

        picked = cg.resolve_run(candidates, tmp_path / "rollouts" / "lift", str(tmp_path))

        assert picked["path"] == right

    def test_a_policy_whose_runs_all_name_another_dataset_draws_nothing(self, tmp_path):
        candidates = [_row("lift", self._run_dir(tmp_path, "aaa", "rollouts/other"))]

        assert cg.resolve_run(candidates, tmp_path / "rollouts" / "lift", str(tmp_path)) is None

    def test_a_run_that_never_said_is_taken_at_its_word(self, tmp_path):
        """No metrics.json at all: an interrupted run is still a good graph."""
        candidates = [_row("lift", self._run_dir(tmp_path, "aaa", None), has_metrics=False)]

        picked = cg.resolve_run(candidates, tmp_path / "rollouts" / "lift", str(tmp_path))

        assert picked is not None

    def test_the_finished_run_beats_the_crashed_retry(self, tmp_path):
        """A retried job leaves two run dirs sharing every hyperparameter."""
        crashed = self._run_dir(tmp_path, "aaa", None)
        finished = self._run_dir(tmp_path, "bbb", "rollouts/lift")
        candidates = [
            _row("lift", crashed, has_metrics=False),
            _row("lift", finished),
        ]

        picked = cg.resolve_run(candidates, tmp_path / "rollouts" / "lift", str(tmp_path))

        assert picked["path"] == finished

    def test_no_candidates_means_no_graph(self):
        assert cg.resolve_run([], Path("/data/rollouts/lift"), "/data") is None


class TestDemoLinks:
    ROW = {
        "path": "/data/sweeps/all/lift/abc123",
        "task": "lift",
        "k": 10,
        "w": 16,
        "reduction": "pca",
        "sticky_lambda": None,
    }

    def test_the_link_carries_the_run_itself(self):
        """Not just its axes: several runs can share them, and the link has to
        land on the graph that was clicked."""
        params = cg.demo_query_params(self.ROW)

        assert params["run_dir"] == "/data/sweeps/all/lift/abc123"
        assert (params["task"], params["k"], params["w"]) == ("lift", "10", "16")

    def test_an_axis_the_run_never_recorded_is_left_out(self):
        """An empty ``sticky_lambda=`` seeds the demo's selector with nothing."""
        assert "sticky_lambda" not in cg.demo_query_params(self.ROW)

    def test_the_link_keeps_the_password_it_arrived_with(self):
        """The deployment is gated and a link is a fresh page load, so without
        this, following one lands on the password prompt."""
        assert cg.demo_query_params(self.ROW, {"pwd": "hunter 2"})["pwd"] == "hunter 2"

    def test_the_link_is_relative_to_the_page_it_is_drawn_on(self):
        """An absolute /graph_demo breaks under a path prefix."""
        url = cg.demo_url(self.ROW, {"pwd": "hunter 2"})

        assert url.startswith("./?")
        assert "pwd=hunter+2" in url
        assert "run_dir=%2Fdata%2Fsweeps%2Fall%2Flift%2Fabc123" in url

    def test_the_graph_demo_is_still_the_page_served_at_the_app_root(self):
        """The links point at the root because that is where Streamlit serves
        the default page — asking for its own ``url_path`` instead answers "Page
        not found" over the top of it. Promoting another page to default has to
        fail here rather than by quietly redirecting every link to it."""
        import ast

        home = (PAGE.parents[1] / "Home.py").read_text()
        defaults = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(home))
            if isinstance(node, ast.Call)
            and any(kw.arg == "default" for kw in node.keywords)
            and getattr(node.func, "attr", None) == "Page"
        ]

        assert len(defaults) == 1, defaults
        assert "graph_demo.py" in defaults[0], defaults[0]
        assert cg.GRAPH_DEMO_URL == "./"


def _write_run(run_dir: Path, *, k: int, rollout_dir: str, episodes: int = 6) -> None:
    """A run dir the section can draw a tree from.

    The tree reads window labels and window metadata, so those are what a
    fixture has to write — no features (the loader stands them in) and no
    graph.json (the tree builds its own from the labels).
    """
    run_dir.mkdir(parents=True)
    per_episode = 4
    windows, episode_meta, labels = [], [], []
    for ep in range(episodes):
        for step in range(per_episode):
            windows.append({"episode_idx": ep, "start": step, "end": step + 1})
            # A different behavior at each step, so the tree has depth, and a
            # different last one per episode, so it also branches.
            labels.append(step if step < per_episode - 1 else (ep % k))
        episode_meta.append(
            {
                "index": ep,
                "num_samples": per_episode,
                "sample_start_idx": ep * per_episode,
                "sample_end_idx": (ep + 1) * per_episode,
                "success": ep % 2 == 0,
            }
        )
    (run_dir / "windows.json").write_text(
        json.dumps({"windows": windows, "episodes": episode_meta})
    )
    np.save(run_dir / "labels.npy", np.array(labels, dtype=np.int64))
    (run_dir / "config.yaml").write_text(
        f"clustering: {{n_clusters: {k}}}\nwindowing: {{width: 16, stride: 4}}\n"
    )
    (run_dir / "metrics.json").write_text(json.dumps({"rollout_dir": rollout_dir}))


class TestTreeInputs:
    def test_the_tree_reads_labels_and_window_metadata(self, tmp_path):
        _write_run(tmp_path / "run", k=3, rollout_dir="rollouts/lift")
        cg.tree_inputs.clear()

        labels, metadata = cg.tree_inputs(str(tmp_path / "run"))

        assert len(labels) == len(metadata) == 24
        assert {m["rollout_idx"] for m in metadata} == set(range(6))
        assert metadata[0]["success"] is True

    def test_it_does_not_read_the_feature_matrix(self, tmp_path):
        """37 MB a run in the deployment's sweep, for numbers no tree looks at —
        so a run whose windows.npz is absent still draws."""
        _write_run(tmp_path / "run", k=3, rollout_dir="rollouts/lift")
        cg.tree_inputs.clear()

        assert not (tmp_path / "run" / "windows.npz").exists()
        assert cg.tree_inputs(str(tmp_path / "run")) is not None

    def test_an_unreadable_run_comes_back_empty_rather_than_raising(self, tmp_path):
        """One broken run must not take the other policies' cards down."""
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "windows.json").write_text("{not json")
        cg.tree_inputs.clear()

        assert cg.tree_inputs(str(tmp_path / "run")) is None

    def test_labels_that_do_not_match_the_windows_are_refused(self, tmp_path):
        """Mismatched artifacts would mislabel every window after the shortfall."""
        _write_run(tmp_path / "run", k=3, rollout_dir="rollouts/lift")
        np.save(tmp_path / "run" / "labels.npy", np.zeros(5, dtype=np.int64))
        cg.tree_inputs.clear()

        assert cg.tree_inputs(str(tmp_path / "run")) is None


class TestSectionOnThePage:
    """Drive the real page, against a data root holding rollouts and runs."""

    @staticmethod
    def _store(tmp_path: Path, monkeypatch) -> None:
        from behavior_graphs.viz.streamlit import rollout_outcomes as ro

        for cache in (ro._scan_rollouts_root, ro._scan_indexed_runs, ro._outcomes_cached):
            cache.clear()
        cg.tree_inputs.clear()
        cg._recorded_rollout_dir.clear()

        rollouts = tmp_path / "rollouts"
        write_rollout_dir(rollouts / "policy_a", [10] * 8, [True] * 6 + [False] * 2, {})
        write_rollout_dir(rollouts / "policy_b", [10] * 8, [True] * 2 + [False] * 6, {})
        # policy_a was swept over two Ks, policy_b over one: the K selector has
        # to exist, and the setting both share has to be the one it opens on.
        for policy, ks in (("policy_a", (5, 10)), ("policy_b", (10,))):
            for k in ks:
                _write_run(
                    tmp_path / "sweeps" / policy / f"k{k}",
                    k=k,
                    rollout_dir=f"rollouts/{policy}",
                )

        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        monkeypatch.setenv("BEHAVIOR_GRAPHS_RUNS_ROOT", str(tmp_path / "sweeps"))
        monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)

    @classmethod
    def _open_section(cls, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest

        cls._store(tmp_path, monkeypatch)
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.query_params["pwd"] = "letmein"
        at.run()
        assert not at.exception, at.exception
        # The section is lazy, so nothing below exists until its header is clicked.
        at.button(key="pc_graphs_lazy_toggle").click().run()
        assert not at.exception, at.exception
        return at

    def test_every_policy_gets_a_graph_and_a_link(self, tmp_path, monkeypatch):
        at = self._open_section(tmp_path, monkeypatch)

        assert len(at.get("component_instance")) == 2
        urls = [el.proto.url for el in at.get("link_button")]
        assert len(urls) == 2
        assert all(u.startswith("./?") for u in urls)

    def test_the_links_point_at_the_run_each_graph_was_drawn_from(self, tmp_path, monkeypatch):
        at = self._open_section(tmp_path, monkeypatch)

        urls = [el.proto.url for el in at.get("link_button")]
        # The shared K, so both policies resolve to their own k10 run.
        assert [u for u in urls if "policy_a%2Fk10" in u]
        assert [u for u in urls if "policy_b%2Fk10" in u]
        assert all("k=10" in u for u in urls)

    def test_the_links_carry_the_password_the_page_was_opened_with(self, tmp_path, monkeypatch):
        at = self._open_section(tmp_path, monkeypatch)

        assert all("pwd=letmein" in el.proto.url for el in at.get("link_button"))

    def test_every_setting_lives_in_the_sidebar(self, tmp_path, monkeypatch):
        """Where the Graph Demo keeps the same selectors, leaving the page to the
        pictures."""
        at = self._open_section(tmp_path, monkeypatch)

        assert "pc_graph_k" in {element.key for element in at.sidebar.selectbox}
        assert {"pc_graph_per_row", "pc_graph_fold_pct"} <= {
            element.key for element in [*at.sidebar.radio, *at.sidebar.slider]
        }
        # The page's own selectboxes are the statistical test settings, no more.
        assert not [element for element in at.main.selectbox if "pc_graph" in (element.key or "")]

    def test_the_k_selector_opens_on_the_setting_both_policies_have(self, tmp_path, monkeypatch):
        at = self._open_section(tmp_path, monkeypatch)

        assert at.session_state["pc_graph_k"] == 10
        assert not [c.value for c in at.caption if "No run at these settings" in c.value]

    def test_choosing_a_setting_only_one_policy_has_says_so(self, tmp_path, monkeypatch):
        at = self._open_section(tmp_path, monkeypatch)

        at.selectbox(key="pc_graph_k").set_value(5).run()

        assert not at.exception, at.exception
        assert len(at.get("component_instance")) == 1
        assert any("No run at these settings" in c.value for c in at.caption)

    def test_a_policy_with_no_runs_at_all_does_not_take_the_section_down(
        self, tmp_path, monkeypatch
    ):
        """Rollouts exist for it; nothing was ever swept over them."""
        self._store(tmp_path, monkeypatch)
        write_rollout_dir(tmp_path / "rollouts" / "policy_c", [10] * 4, [True] * 4, {})

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PAGE), default_timeout=120).run()
        at.button(key="pc_graphs_lazy_toggle").click().run()

        assert not at.exception, at.exception
        assert any("No run at these settings" in c.value for c in at.caption)
        assert len(at.get("component_instance")) == 2
