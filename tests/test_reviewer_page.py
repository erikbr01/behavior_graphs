"""behavior_graphs.viz.streamlit.reviewer — what the try.behavior-graphs.com page offers.

The page itself is a Streamlit script; what is worth testing is the decisions it
makes before drawing anything: which families exist, which K values a dataset
has, and which of a dataset's many swept runs is the one to draw.
"""

from pathlib import Path

from behavior_graphs.viz.streamlit import reviewer
from behavior_graphs.viz.streamlit.data import _group_families


def _row(task, k, path, **kw):
    """One run-index row, with the keys `reviewer` reads and nothing else."""
    return {"task": task, "k": k, "path": path, "has_metrics": True, **kw}


# ── Families out of the configs/data/ tree ───────────────────────────────────


def _write_group(root: Path, name: str, body: str) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / f"{name}.yaml").write_text(body)


def test_families_are_the_groups_that_declare_datasets_not_the_aggregators(tmp_path):
    # `all` names other groups and declares nothing, the way the real one does.
    _write_group(tmp_path, "all", "defaults:\n  - libero\n  - droid\n  - _self_\n")
    _write_group(
        tmp_path, "libero", "libero_goal: /r/libero_goal\nlibero_object: /r/libero_object\n"
    )
    _write_group(
        tmp_path, "droid", "cameras:\n  wrist:\n    rotate: 180\nset_table: /r/set_table\n"
    )

    families = _group_families("all", tmp_path, set())

    # "all" itself is not a family — offered as one, it is the whole corpus in
    # a dropdown that exists to narrow the corpus.
    assert families == {
        "libero": ["libero_goal", "libero_object"],
        "droid": ["set_table"],
    }
    # And `cameras:` is configuration, not a dataset.
    assert "cameras" not in families["droid"]


def test_families_descend_through_a_second_aggregator(tmp_path):
    # `visualize` is `all` with entries commented out — two levels of naming
    # before the first group that declares anything.
    _write_group(tmp_path, "visualize", "defaults:\n  - all\n  - _self_\n")
    _write_group(tmp_path, "all", "defaults:\n  - libero\n  - _self_\n")
    _write_group(tmp_path, "libero", "libero_goal: /r/libero_goal\n")

    assert _group_families("visualize", tmp_path, set()) == {"libero": ["libero_goal"]}


def test_a_group_that_both_includes_and_declares_keeps_its_own_datasets(tmp_path):
    _write_group(tmp_path, "visualize", "defaults:\n  - libero\n  - _self_\nextra: /r/extra\n")
    _write_group(tmp_path, "libero", "libero_goal: /r/libero_goal\n")

    assert _group_families("visualize", tmp_path, set()) == {
        "libero": ["libero_goal"],
        "visualize": ["extra"],
    }


def test_a_defaults_cycle_terminates(tmp_path):
    _write_group(tmp_path, "a", "defaults:\n  - b\n  - _self_\n")
    _write_group(tmp_path, "b", "defaults:\n  - a\n  - _self_\nds: /r/ds\n")

    assert _group_families("a", tmp_path, set()) == {"b": ["ds"]}


# ── Families against what is actually mounted ────────────────────────────────


def test_families_drop_datasets_and_groups_with_no_runs():
    families = {"libero": ["libero_goal", "libero_object"], "safe": ["safe_franka_lift"]}

    assert reviewer.families_with_runs(families, {"libero_goal"}) == {"libero": ["libero_goal"]}


def test_orphan_tasks_are_the_ones_no_family_claims():
    families = {"libero": ["libero_goal"]}

    assert reviewer.orphan_tasks(families, {"libero_goal", "unswept_thing"}) == ["unswept_thing"]
    assert reviewer.orphan_tasks(families, {"libero_goal"}) == []


# ── K ────────────────────────────────────────────────────────────────────────


def test_k_is_the_pinned_configs_k_wherever_one_exists():
    """A predefined config decides K; nothing is chosen automatically over it."""
    rows = [_row("set_table", 11, "/sweeps/set_table/pinned"), _row("set_table", 20, "/s/better")]
    # Model selection likes K=20 here, and does not get a say.
    best = {"set_table": {20: {"path": "/s/better", "score": 0.99}}}
    pinned = [_pin("set_table", "/sweeps/set_table/pinned")]

    assert reviewer.preferred_k("set_table", rows, pinned, best) == 11


def test_k_is_auto_chosen_only_for_a_dataset_no_config_pins():
    rows = [_row("libero_goal", 5, "/r/k5"), _row("libero_goal", 15, "/r/k15")]
    best = {"libero_goal": {5: {"score": 0.4}, 15: {"score": 0.8}}}

    # The best-scoring K, not the first — K=5 on a dataset nobody chose it for
    # is a coarse graph presented as the intended one.
    assert reviewer.preferred_k("libero_goal", rows, [], best) == 15
    # And with no scores to go on, the smallest is all there is.
    assert reviewer.preferred_k("libero_goal", rows, [], None) == 5


def test_a_pin_this_deployment_cannot_serve_falls_through_to_auto():
    """The older sessions pin runs under $DATA_PATH/runs, which a sweeps-only
    stack has never indexed. Those are not a veto on choosing."""
    rows = [_row("robomimic_lift_mh", 10, "/data/sweeps/robomimic_lift_mh/abc")]
    pinned = [_pin("robomimic_lift_mh", "/data/runs/robomimic_lift_mh_dp")]

    assert reviewer.preferred_k("robomimic_lift_mh", rows, pinned, None) == 10


def test_k_values_are_ascending_deduplicated_and_per_task():
    rows = [
        _row("a", 11, "/r/1"),
        _row("a", 5, "/r/2"),
        _row("a", 5, "/r/3"),
        _row("a", None, "/r/4"),
        _row("b", 20, "/r/5"),
    ]

    assert reviewer.k_values("a", rows) == [5, 11]
    assert reviewer.k_values("missing", rows) == []


# ── Which run to draw ────────────────────────────────────────────────────────


def test_resolve_run_draws_the_run_model_selection_named_not_one_like_it():
    # The tie is the point: these two agree on every hyperparameter the picker
    # exposes and are different runs. Matching on params would pick either.
    rows = [
        _row("a", 11, "/sweeps/a/twin", w=8, reduction="umap"),
        _row("a", 11, "/sweeps/a/winner", w=8, reduction="umap"),
    ]
    best = {"a": {11: {"w": 8, "reduction": "umap", "score": 0.84, "path": "/sweeps/a/winner"}}}

    assert reviewer.resolve_run("a", 11, rows, best) == Path("/sweeps/a/winner")


def test_resolve_run_finds_the_winner_across_a_bind_mount():
    # best_per_k.csv is written by the sweep against the host's paths and read
    # by an app that has the same tree mounted somewhere else.
    rows = [_row("a", 11, "/data/sweeps/a/winner"), _row("a", 11, "/data/sweeps/a/twin")]
    best = {"a": {11: {"path": "${DATA_PATH}/sweeps/a/winner"}}}

    assert reviewer.resolve_run("a", 11, rows, best) == Path("/data/sweeps/a/winner")


def test_resolve_run_falls_back_when_the_named_winner_is_not_mounted():
    rows = [_row("a", 11, "/r/only")]

    assert reviewer.resolve_run("a", 11, rows, {"a": {11: {"path": "/r/gone"}}}) == Path("/r/only")
    assert reviewer.resolve_run("a", 11, rows, None) == Path("/r/only")


def test_resolve_run_prefers_a_run_that_finished():
    # A crashed job retried under a new hash leaves two runs with the same
    # hyperparameters, one of them without metrics.json.
    rows = [
        _row("a", 11, "/r/crashed", has_metrics=False),
        _row("a", 11, "/r/complete"),
    ]

    assert reviewer.resolve_run("a", 11, rows, None) == Path("/r/complete")


def test_resolve_run_returns_none_when_nothing_matches():
    assert reviewer.resolve_run("a", 99, [_row("a", 11, "/r/1")], None) is None


def test_the_pinned_run_wins_over_the_best_scoring_one():
    # The run the participants read carries the node labels the study showed;
    # a higher-scoring twin at the same K does not, and draws "B6" for every
    # node.
    rows = [_row("set_table", 11, "/sweeps/set_table/pinned"), _row("set_table", 11, "/s/better")]
    best = {"set_table": {11: {"path": "/s/better", "score": 0.9}}}
    pinned = [_pin("set_table", "/sweeps/set_table/pinned")]

    assert reviewer.resolve_run("set_table", 11, rows, best, pinned) == Path(
        "/sweeps/set_table/pinned"
    )


def test_a_pinned_dataset_at_another_k_gets_the_best_scoring_run():
    # The pin is per-run, not per-dataset: move the slider off the study's K
    # and there is no participant's tree to be faithful to.
    rows = [_row("set_table", 11, "/sweeps/set_table/pinned"), _row("set_table", 15, "/s/k15")]
    pinned = [_pin("set_table", "/sweeps/set_table/pinned")]

    assert reviewer.resolve_run("set_table", 15, rows, None, pinned) == Path("/s/k15")


# ── Where the page opens ─────────────────────────────────────────────────────


def _pin(task, run_dir, **params):
    return reviewer.PinnedGraph(
        task=task, label=task, run_dir=Path(run_dir), graph_params=dict(params)
    )


def test_the_page_opens_on_the_first_pinned_graph_at_its_own_k():
    pinned = [_pin("set_table", "/sweeps/set_table/pinned"), _pin("cooking", "/sweeps/c/x")]
    rows = [
        _row("set_table", 11, "/sweeps/set_table/pinned"),
        _row("set_table", 5, "/sweeps/set_table/other"),
        _row("cooking", 15, "/sweeps/c/x"),
    ]
    families = {"libero": ["libero_goal"], "droid": ["set_table", "cooking"]}

    assert reviewer.default_selection(pinned, families, rows) == ("droid", "set_table", 11)


def test_the_page_falls_through_to_the_first_dataset_when_nothing_is_pinned():
    rows = [_row("libero_goal", 7, "/r/1")]
    families = {"libero": ["libero_goal"]}

    assert reviewer.default_selection([], families, rows) == ("libero", "libero_goal", 7)


def test_the_page_skips_a_pinned_graph_whose_dataset_is_not_mounted():
    pinned = [_pin("set_table", "/sweeps/set_table/pinned")]
    rows = [_row("libero_goal", 7, "/r/1")]

    assert reviewer.default_selection(pinned, {"libero": ["libero_goal"]}, rows) == (
        "libero",
        "libero_goal",
        7,
    )


def test_no_default_when_nothing_has_a_k():
    assert reviewer.default_selection([], {"libero": ["libero_goal"]}, []) is None


# ── How it is drawn ──────────────────────────────────────────────────────────


def test_a_pinned_task_is_drawn_with_its_configs_own_settings():
    pinned = [_pin("prepare_tray", "/r/1", min_branch=1, color_mode="outcome")]

    params = reviewer.display_params("prepare_tray", pinned)

    # prepare_tray's shallower min_branch is what makes its tree legible, and
    # that does not stop being true one notch along the K slider.
    assert params["min_branch"] == 1
    # The page's own quiet defaults still apply to everything the config's block
    # does not mention.
    assert params["show_fold_toggles"] is False
    assert params["video_panel"] == "below"


def test_a_pinned_config_supplies_the_graph_and_not_the_pages_chrome():
    """A session yaml configures a study page; it does not get to configure this
    one. Only `CONFIG_KEYS` — what the tree *is* — are taken from it."""
    pinned = [
        _pin(
            "prepare_tray",
            "/r/1",
            min_branch=1,
            # All chrome, and all of it the study app's business:
            show_paths=True,
            show_fold_toggles=True,
            video_panel="right",
            shuffle_rollout_indices=True,
        )
    ]

    params = reviewer.display_params("prepare_tray", pinned)

    assert params["min_branch"] == 1
    assert params["show_paths"] is False
    assert params["show_fold_toggles"] is False
    assert params["video_panel"] == "below"
    assert "shuffle_rollout_indices" not in params


def test_an_unpinned_task_gets_the_pages_defaults():
    params = reviewer.display_params("libero_goal", [_pin("prepare_tray", "/r/1", min_branch=1)])

    assert params == reviewer.DISPLAY_FALLBACK
    assert params is not reviewer.DISPLAY_FALLBACK  # a copy, not the module's dict


def test_family_label_falls_back_to_the_group_name():
    assert reviewer.family_label("libero") == "LIBERO"
    assert reviewer.family_label("a_benchmark_added_later") == "a_benchmark_added_later"


# ── The repo's own configs ───────────────────────────────────────────────────


def test_the_studys_three_stages_lead_and_every_pin_is_one_dataset(monkeypatch):
    """Read against the checkout's real `configs/viz/study/`, because the order
    is the point and a fixture would be asserting on itself.

    No DATA_PATH needed: `resolve_path` leaves an unset `${oc.env:…}` in place,
    and the task is the run dir's parent either way.
    """
    configs = Path(__file__).resolve().parents[1] / "configs" / "viz" / "study"
    monkeypatch.setenv("STUDY_SESSIONS_DIR", str(configs))

    pinned = reviewer.pinned_graphs()

    assert [p.task for p in pinned[:3]] == [
        "set_table_baseline_pretrained_dit",
        "prepare_tray_baseline",
        "cooking_baseline_dit_200",
    ]
    # One graph per dataset: two sessions pin robomimic_transport_mh_dp
    # (`transport_mh_jan28` and its `_k10` sibling) and the first by config
    # order wins, so the page never has to choose between two pinned graphs of
    # one dataset.
    tasks = [p.task for p in pinned]
    assert len(tasks) == len(set(tasks)), tasks
    # Every pin carries a run and a label, so nothing resolves to a blank option.
    assert all(p.run_dir.name and p.label for p in pinned)


def test_a_flat_layout_pin_learns_its_dataset_from_the_index(monkeypatch, tmp_path):
    """`runs/<name>` keeps the dataset in the run dir, not its parent — read off
    the parent it would be the literal "runs".

    The session is written here rather than taken from `configs/viz/study/`: the
    sessions that shipped a flat `runs/<name>` pin were the sim-era pilots, and
    the behaviour has to stay covered after they were dropped from the release.
    """
    sessions = tmp_path / "study"
    sessions.mkdir()
    (sessions / "flat_pin.yaml").write_text(
        "task: robomimic_lift_mh\n"
        "title: Flat layout pin\n"
        "label: Flat layout pin\n"
        "run_dir: /data/runs/robomimic_lift_mh_dp\n"
        "rollout_dir: /data/rollouts/robomimic_lift_mh\n"
        "strategies: []\n"
    )
    monkeypatch.setenv("STUDY_SESSIONS_DIR", str(sessions))
    monkeypatch.setenv("DATA_PATH", "/data")

    rows = [_row("robomimic_lift_mh", 10, "/data/runs/robomimic_lift_mh_dp")]
    tasks = [p.task for p in reviewer.pinned_graphs(rows)]

    assert "robomimic_lift_mh" in tasks
    # A pin whose dataset is not in the index keeps the parent guess — "runs",
    # which matches no dataset and so can never be drawn.
    assert tasks.count("runs") <= 1


def test_the_behavior_timeline_under_each_clip_is_on():
    """The banded strip below the videos — what shows a reader that the graph's
    decomposition matches what the robot does. The study turned it off because a
    participant reading the segmentation off the video is not reading it off the
    graph; a reviewer is not being measured."""
    from behavior_graphs.viz.streamlit.components.graph_config import tree_kwargs_from_config

    params = reviewer.display_params("libero_goal", [])

    assert params["show_behavior_annotations"] is True
    # And it survives the trip through the kwarg translation, which renames it.
    assert tree_kwargs_from_config(params)["show_annotations"] is True


def test_a_pinned_config_cannot_turn_the_timeline_back_off():
    """It is chrome, so it is not in CONFIG_KEYS — the study's sessions all set
    it False and must not drag that onto this page."""
    pinned = [_pin("prepare_tray", "/r/1", show_behavior_annotations=False)]

    assert reviewer.display_params("prepare_tray", pinned)["show_behavior_annotations"] is True


def test_the_studys_graphs_are_the_variants_stages_in_its_own_order(monkeypatch):
    """Not the first three of ``pinned_graphs``. That list leads with the
    stages and then carries every other session yaml, so taking three by
    position keeps saying "these are the study's" after a fourth config is
    added or the variant goes missing.
    """

    class Stage:
        def __init__(self, task, label):
            self.task = task
            self.label = label

    class Variant:
        stages = [
            Stage("cooking", "Cook the meal"),
            Stage("set_table", "Set the table"),
        ]

    pins = [
        reviewer.PinnedGraph("set_table_baseline_dit", "Set the table", Path("a"), {}),
        reviewer.PinnedGraph("cooking_baseline_200", "Cook the meal", Path("b"), {}),
        reviewer.PinnedGraph("square_mh", "square_mh_feb5", Path("c"), {}),
    ]
    monkeypatch.setattr(reviewer, "pinned_graphs", lambda *a, **k: pins)
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_studies.registry.get_variant",
        lambda _id: Variant(),
    )
    found = reviewer.study_graphs([])
    # The variant's order, and the session yaml that is not a stage is absent.
    assert [graph.label for graph in found] == ["Cook the meal", "Set the table"]


def test_a_stage_whose_run_is_not_mounted_is_simply_left_out(monkeypatch):
    class Stage:
        def __init__(self, task, label):
            self.task = task
            self.label = label

    class Variant:
        stages = [Stage("set_table", "Set the table"), Stage("cooking", "Cook the meal")]

    monkeypatch.setattr(
        reviewer,
        "pinned_graphs",
        lambda *a, **k: [
            reviewer.PinnedGraph("set_table_baseline_dit", "Set the table", Path("a"), {})
        ],
    )
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_studies.registry.get_variant",
        lambda _id: Variant(),
    )
    assert [graph.label for graph in reviewer.study_graphs([])] == ["Set the table"]


def test_no_variant_means_no_study_graphs(monkeypatch):
    """A page that then shows three arbitrary graphs under a sentence about
    the study is worse than one that says it cannot find them."""
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_studies.registry.get_variant",
        lambda _id: None,
    )
    assert reviewer.study_graphs([]) == []


def test_a_pin_carries_the_task_description_the_study_showed(monkeypatch, tmp_path):
    """A graph of a task a reader cannot name is a graph of nothing, and the
    words are already in the session yaml the study read them from."""
    session = tmp_path / "set_table_contrastive.yaml"
    session.write_text(
        "task: set_table_contrastive\n"
        "run_dir: /runs/abc\n"
        "task_description: >-\n"
        "  The robot should place the yellow pumpkin on the plate.\n"
    )

    class Variant:
        graph_params: dict = {}
        stages = []

    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_study.sessions.sessions_dir", lambda: tmp_path
    )
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_study.sessions.resolve_run_dir",
        lambda _sess: Path("/runs/abc"),
    )
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.user_studies.registry.get_variant", lambda _id: Variant()
    )
    (pin,) = reviewer.pinned_graphs([])
    # One line, whatever the yaml's folding did to it.
    assert pin.description == "The robot should place the yellow pumpkin on the plate."
