"""The URL a participant is sent: one pinned variant, and no sidebar.

``apps/study/Home.py`` is the deployed participant-facing app. Until now it could
only run the original Group A / B study, so a new variant was reachable only
through the researcher demo page — where the sidebar offers a study-version
dropdown, a group dropdown and every other page of the lab app, all of which a
participant can click.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.core.graph import BehaviorGraph  # noqa: E402
from behavior_graphs.core.types import (  # noqa: E402
    WindowInfo,
    WindowSet,
    episodes_from_lengths,
)
from behavior_graphs.envs.trajectory_dir import write_rollout_dir  # noqa: E402
from behavior_graphs.pipeline.artifacts import RunArtifacts  # noqa: E402
from behavior_graphs.viz.streamlit.user_studies import closed  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "study" / "Home.py"

#: The sessions ``configs/viz/study_variants/three_stage_v1.yaml`` names, one
#: per stage. The variant is read from the real configs dir -- it is part of
#: what these tests cover -- so the stand-in yamls have to answer to its names.
_SESSIONS = ("set_table_contrastive", "prepare_tray_contrastive", "cooking_contrastive")

_SEQS = [[0, 1, 2]] * 6 + [[0, 3]] * 2
_OUTCOMES = [True] * 6 + [False] * 2


def _run_dir(path: Path) -> Path:
    """A minimal run dir with the artifacts ``load_run`` insists on."""
    lengths = [len(s) for s in _SEQS]
    windows = [
        WindowInfo(episode_idx=ep, start=t, end=t + 1)
        for ep, seq in enumerate(_SEQS)
        for t in range(len(seq))
    ]
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in _SEQS])
    features = np.random.default_rng(0).normal(size=(len(windows), 4)).astype(np.float32)

    artifacts = RunArtifacts(path)
    artifacts.save_windows(
        WindowSet(
            features=features, windows=windows, episodes=episodes_from_lengths(lengths, _OUTCOMES)
        )
    )
    artifacts.save_array("labels", labels)
    artifacts.save_graph(
        BehaviorGraph.from_window_labels(
            labels,
            WindowSet(
                features=features,
                windows=windows,
                episodes=episodes_from_lengths(lengths, _OUTCOMES),
            ),
        )
    )
    artifacts.save_metrics({"rollout_dir": "rollouts/stand_in"})
    return artifacts.run_dir


def _rollout_dir(path: Path) -> Path:
    """A rollout dir the mp4 index can read: trajectory metadata plus one clip
    per episode. The clips are empty files -- nothing here plays them."""
    lengths = [len(s) for s in _SEQS]
    write_rollout_dir(
        path,
        lengths,
        _OUTCOMES,
        {"cond_proj": np.zeros((sum(lengths), 4), dtype=np.float32)},
    )
    media = path / "media"
    media.mkdir(parents=True, exist_ok=True)
    for ep in range(len(_SEQS)):
        outcome = "success" if _OUTCOMES[ep] else "failure"
        (media / f"ep{ep:04d}_{outcome}.mp4").write_bytes(b"")
    return path


@pytest.fixture(autouse=True)
def _no_password(monkeypatch, tmp_path):
    monkeypatch.delenv("APP_PASSWORD_SHA256", raising=False)
    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
    # The participant route is closed unless told otherwise, so every test of
    # what it serves has to say so -- see `user_studies/closed.py` and
    # `test_study_closed.py`. Set here rather than in `_run`, because the
    # warm-up tests below build their own `AppTest`.
    monkeypatch.setenv(closed.ACCEPTING_ENV, "1")


@pytest.fixture(autouse=True)
def _study_data(monkeypatch, tmp_path):
    """Stand-in study data, so these tests do not need the real data root.

    The layout renders nothing until all three stages' artifacts load, so a
    shell with no ``$DATA_PATH`` used to fail here on an assertion about the
    introduction's wording -- the intro never got drawn, and the reason was two
    layers away from what the test said it was checking. One synthetic run,
    named three times over, is enough: what these tests are about is the route
    (which variant is served, and that it writes a record), not the evidence.
    """
    run = _run_dir(tmp_path / "run")
    rollouts = _rollout_dir(tmp_path / "rollouts" / "stand_in")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for name in _SESSIONS:
        (sessions / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "task": name,
                    "title": f"{name} (stand-in)",
                    "task_description": "A stand-in task, for the route tests.",
                    "label": name,
                    "strategies": [],
                    "ranking": {"top_n": 3},
                    "run_dir": str(run),
                    "rollout_dir": str(rollouts),
                }
            )
        )
    monkeypatch.setenv("STUDY_SESSIONS_DIR", str(sessions))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setenv("BEHAVIOR_GRAPHS_RUNS_ROOT", str(tmp_path))


def _run(monkeypatch, variant: str | None = None, group: str | None = None) -> AppTest:
    if variant is None:
        monkeypatch.delenv("STUDY_VARIANT", raising=False)
    else:
        monkeypatch.setenv("STUDY_VARIANT", variant)
    if group is None:
        monkeypatch.delenv("STUDY_VARIANT_GROUP", raising=False)
    else:
        monkeypatch.setenv("STUDY_VARIANT_GROUP", group)
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_the_pinned_variant_is_what_a_participant_gets(monkeypatch):
    """Pinned by the environment, not picked from a dropdown they could change."""
    at = _run(monkeypatch, variant="three_stage_v1")

    assert not at.exception, [e.value for e in at.exception]
    # The three-stage study's own first page, and no study-version selector.
    said = " ".join(markdown.value for markdown in at.markdown)
    assert "three different robot tasks" in said
    assert not [box for box in at.selectbox if box.label == "Study version"]


def test_the_record_says_it_came_from_the_participant_route(monkeypatch, tmp_path):
    """The lab app renders these same layouts, and this one writes a record on
    the first page load — so without a marker a researcher reading the wording
    leaves something indistinguishable from a participant who quit on the
    introduction, in the funnel and in the completed-of-started metric."""
    at = _run(monkeypatch, variant="three_stage_v1")

    assert not at.exception, [e.value for e in at.exception]
    from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore

    written = LocalResponseStore(tmp_path).list_all()
    assert written, "the record is written before the participant has clicked anything"
    assert {r["origin"] for r in written} == {"participant"}


def test_a_browser_with_nothing_remembered_gets_a_new_attempt(monkeypatch):
    """The recall is a cookie read on the connection, not a page: a first-time
    participant sees the study itself, with no extra load in front of it."""
    at = _run(monkeypatch, variant="three_stage_v1")

    assert not at.exception
    said = " ".join(markdown.value for markdown in at.markdown)
    assert "three different robot tasks" in said


def test_the_sidebar_is_hidden(monkeypatch):
    """It carries the condition, the session id and the lab app's other pages."""
    at = _run(monkeypatch, variant="three_stage_v1")

    styles = " ".join(markdown.value for markdown in at.markdown if "stSidebar" in markdown.value)
    assert "display:none" in styles


def test_a_variant_that_does_not_exist_says_so_instead_of_serving_nothing(monkeypatch):
    at = _run(monkeypatch, variant="no_such_variant_v9")

    assert not at.exception
    assert any("not a study variant" in error.value for error in at.error)


def test_a_group_that_does_not_exist_names_the_ones_that_do(monkeypatch):
    at = _run(monkeypatch, variant="three_stage_v1", group="group_z")

    assert not at.exception
    assert any("randomized" in error.value for error in at.error)


def test_a_warm_up_hit_writes_no_record(monkeypatch, tmp_path):
    """The layout writes a record on first page load, so a warm-up that went
    through the ordinary route would leave an abandoned attempt in the funnel
    after every restart, and the completed-of-started metric would drift down
    with each redeploy."""
    from behavior_graphs.viz.streamlit.user_studies.warmup import WARMUP_MARKER, WARMUP_PARAM

    monkeypatch.setenv("STUDY_VARIANT", "three_stage_v1")
    monkeypatch.delenv("STUDY_VARIANT_GROUP", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params[WARMUP_PARAM] = "1"
    at = at.run()

    assert not at.exception
    assert not list(tmp_path.glob("*.json")), "a warm-up is not a participant"
    said = " ".join(str(element.value) for element in at.markdown)
    assert WARMUP_MARKER in said
