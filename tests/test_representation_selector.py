"""The picker's Representation selector: which features a graph was built from.

Two runs on different representations are graphs of different things, so this
is a top-level sidebar choice like Task and K rather than a slot in the
advanced expander. Before the policy-free feature source existed the picker
only offered "Layer", which is blank for a run whose features never came from a
policy -- so those runs were indistinguishable in the sidebar from the policy
runs of the same task.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.data import representation_label  # noqa: E402

_SCRIPT = """
import behavior_graphs.viz.streamlit.data as data
data._run_index.clear()
data._varying_params.clear()
data.select_run_dir()
"""

POLICY = """clustering: {n_clusters: %d}
windowing: {width: 5, stride: 1}
features: {_target_: behavior_graphs.stages.features.policy_embedding.PolicyEmbeddings, layer: cond_proj}
"""

DINO = """clustering: {n_clusters: %d}
windowing: {width: 5, stride: 1}
features:
  _target_: behavior_graphs.stages.features.vision_backbone.VisionBackboneFeatures
  model_name: facebook/dinov2-base
  pooling: grid
  grid: 2
"""


def _run(root, sweep: str, task: str, job_hash: str, config: str):
    d = root / sweep / task / job_hash
    d.mkdir(parents=True)
    (d / "windows.json").write_text("{}")
    (d / "labels.npy").write_bytes(b"")
    (d / "metrics.json").write_text("{}")
    (d / "config.yaml").write_text(config)
    return d


def _picker(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_GRAPHS_RUNS_ROOT", str(tmp_path))
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    monkeypatch.delenv("BEHAVIOR_GRAPHS_RUN_DIR", raising=False)
    return AppTest.from_string(_SCRIPT, default_timeout=30).run()


class TestLabel:
    def test_policy_embeddings_are_named_by_layer(self):
        assert (
            representation_label(
                {
                    "_target_": "behavior_graphs.stages.features.policy_embedding.PolicyEmbeddings",
                    "layer": "cond_proj",
                }
            )
            == "cond_proj"
        )

    @pytest.mark.parametrize(
        "features,expected",
        [
            ({"pooling": "grid", "grid": 2}, "dinov2-base patch2x2"),
            ({"pooling": "grid", "grid": 4}, "dinov2-base patch4x4"),
            # grid 1 *is* mean pooling; naming it "patch1x1" would offer the
            # viewer a distinction that does not exist.
            ({"pooling": "grid", "grid": 1}, "dinov2-base mean"),
            ({"pooling": "mean"}, "dinov2-base mean"),
            ({"pooling": "cls"}, "dinov2-base cls"),
            (
                {"pooling": "grid", "grid": 2, "cameras": ["exterior", "wrist"]},
                "dinov2-base patch2x2 x2views",
            ),
        ],
    )
    def test_vision_backbone_names_the_backbone_and_pooling(self, features, expected):
        cfg = {
            "_target_": "behavior_graphs.stages.features.vision_backbone.VisionBackboneFeatures",
            "model_name": "facebook/dinov2-base",
            **features,
        }
        assert representation_label(cfg) == expected

    def test_the_index_carries_the_label_as_a_column(self):
        """The picker and best-per-K read one precomputed value, not two guesses.

        `flatten_config` labels the *raw* config, before pandas can widen
        `features.grid` to a float or the camera list can be JSON-dumped --
        which is what a second, index-side reconstruction of this label got
        wrong (`patch2.0x2.0`, `x21views`). Pinned here because the whole
        selector depends on the two sides agreeing exactly: Streamlit silently
        drops a selectbox value that is not among the options.
        """
        import pandas as pd

        from behavior_graphs.model_selection.index import flatten_config

        dino = {
            "features": {
                "_target_": "behavior_graphs.stages.features.vision_backbone.VisionBackboneFeatures",
                "model_name": "facebook/dinov2-base",
                "pooling": "grid",
                "grid": 2,
                "cameras": ["exterior", "wrist"],
            }
        }
        policy = {
            "features": {
                "_target_": "behavior_graphs.stages.features.policy_embedding.PolicyEmbeddings",
                "layer": "cond_proj",
            }
        }
        rows = [flatten_config(dino), flatten_config(policy)]
        assert [r["representation"] for r in rows] == [
            "dinov2-base patch2x2 x2views",
            "cond_proj",
        ]
        # A mixed population is what used to widen features.grid to float; the
        # label column is immune because it is computed before the frame is.
        frame = pd.DataFrame(rows)
        assert frame["features.grid"].dtype == float
        assert list(frame["representation"]) == [r["representation"] for r in rows]

    def test_other_extractors_and_unknowns(self):
        assert (
            representation_label(
                {"_target_": "behavior_graphs.stages.features.state.StateFeatures"}
            )
            == "state"
        )
        # Unreadable config: None keeps the run out of the selector's value set
        # rather than filing it under a wrong name.
        assert representation_label({}) is None
        assert representation_label(None) is None


class TestSelector:
    def test_absent_when_every_run_shares_one_representation(self, tmp_path, monkeypatch):
        """A selector with one option is noise; the sweep-wide varying rule covers it."""
        _run(tmp_path, "sweep_all", "set_table", "a" * 12, POLICY % 5)
        _run(tmp_path, "sweep_all", "set_table", "b" * 12, POLICY % 9)
        at = _picker(tmp_path, monkeypatch)

        assert not at.exception
        assert not [s for s in at.sidebar.selectbox if s.label == "Representation"]

    def test_offers_both_and_filters_to_the_picked_one(self, tmp_path, monkeypatch):
        """Runs from *different sweep dirs* merge under one task, so this is the
        only thing that tells a policy graph from a video-feature graph."""
        _run(tmp_path, "sweep_all", "set_table", "c" * 12, POLICY % 5)
        _run(tmp_path, "dino_droid", "set_table", "d" * 12, DINO % 5)
        at = _picker(tmp_path, monkeypatch)

        assert not at.exception
        (rep,) = [s for s in at.sidebar.selectbox if s.label == "Representation"]
        assert list(rep.options) == ["cond_proj", "dinov2-base patch2x2"]

        # The sidebar names the resolved run by its job hash (_show_run_id).
        assert "c" * 12 in "".join(c.value for c in at.code)
        at = (
            at.sidebar.selectbox(key="run_dir_select_representation")
            .select("dinov2-base patch2x2")
            .run()
        )
        assert not at.exception
        assert "d" * 12 in "".join(c.value for c in at.code)

    def test_layer_selector_no_longer_stands_in_for_it(self, tmp_path, monkeypatch):
        """The video runs have no layer at all, so Layer cannot separate them."""
        _run(tmp_path, "sweep_all", "set_table", "e" * 12, POLICY % 5)
        _run(tmp_path, "dino_droid", "set_table", "f" * 12, DINO % 5)
        at = _picker(tmp_path, monkeypatch)

        labels = [s.label for s in at.sidebar.selectbox]
        assert "Representation" in labels
        assert "Layer" not in labels  # one value across the index: not a varying axis
