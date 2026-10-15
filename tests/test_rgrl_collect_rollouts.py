"""The rgrl adapter collects rollouts rather than denying it can.

This file exists because of a documentation bug that cost real time. The
adapter raised ``NotImplementedError`` saying "rgrl doesn't write the canonical
trajectory layout yet", written six weeks *after* the submodule gained a
collector that does exactly that. Two separate investigations believed the
message instead of the submodule, and the second one only looked again because
a human insisted.

The underlying trap is still live: the submodule's working tree can sit on a
branch that predates the collector while the recorded gitlink points at one
that has it, so reading the checkout tells you nothing about what the pin
supports. These tests pin the behaviour the adapter must have in both cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_graphs.envs.rgrl_env import RgrlAdapter, RgrlCheckpoint


@pytest.fixture()
def adapter(tmp_path: Path) -> RgrlAdapter:
    """An adapter over a fake submodule laid out like the pinned ref."""
    sub = tmp_path / "third_party" / "rgrl"
    (sub / "rgrl" / "inference").mkdir(parents=True)
    (sub / "configs" / "rollout_env").mkdir(parents=True)
    (sub / "train.py").touch()
    (sub / "rgrl" / "inference" / "collect_rollouts.py").touch()
    for backend in ("droid", "robomimic"):
        (sub / "configs" / "rollout_env" / f"{backend}.yaml").touch()
    (tmp_path / "scripts" / "utils").mkdir(parents=True)
    (tmp_path / "scripts" / "utils" / "assemble_droid_sync_recordings.py").touch()
    return RgrlAdapter(
        data="lift_state", target="lift_state", env="robomimic_lift", repo_root=str(tmp_path)
    )


@pytest.fixture()
def ckpt(tmp_path: Path) -> RgrlCheckpoint:
    return RgrlCheckpoint(tmp_path / "run" / "latest.pt")


def _capture(monkeypatch):
    """Record both subprocess invocations instead of running them."""
    calls: list[tuple[str, list]] = []
    monkeypatch.setattr(
        RgrlAdapter,
        "_run_in_submodule",
        lambda self, args, log_path: calls.append(("submodule", list(args))),
    )
    monkeypatch.setattr(
        "behavior_graphs.envs.rgrl_env.run_streaming",
        lambda cmd, cwd, log_path: calls.append(("repo", list(cmd))),
    )
    monkeypatch.setattr(
        "behavior_graphs.envs.rgrl_env.RolloutDirectory", lambda path: ("rollout_dir", path)
    )
    return calls


def test_it_does_not_raise_not_implemented(adapter, ckpt, tmp_path, monkeypatch):
    """The regression itself: this used to refuse unconditionally."""
    _capture(monkeypatch)
    adapter.collect_rollouts(ckpt, tmp_path / "out", num_episodes=2)


def test_sim_backend_is_selected_and_the_tree_is_assembled(adapter, ckpt, tmp_path, monkeypatch):
    calls = _capture(monkeypatch)
    out = tmp_path / "out"
    adapter.collect_rollouts(ckpt, out, num_episodes=25, seed=7, capture_layers=["cond_proj"])

    where, argv = calls[0]
    assert where == "submodule"
    assert "-m" in argv and "rgrl.inference.collect_rollouts" in argv
    assert "rollout_env=robomimic" in argv, "a sim env must pick the robomimic backend"
    assert f"output_dir={out.resolve()}" in argv
    assert "num_episodes=25" in argv
    assert "rollout_env.seed=7" in argv
    assert "capture_layers=[cond_proj]" in argv

    where, cmd = calls[1]
    assert where == "repo", "the assembler imports this repo, not the submodule's env"
    assert cmd[1].endswith("assemble_droid_sync_recordings.py")
    assert "--layer" in cmd and "cond_proj" in cmd


def test_a_checkout_without_the_collector_says_so(adapter, ckpt, tmp_path, monkeypatch):
    """The branch trap, named explicitly rather than failing obscurely."""
    _capture(monkeypatch)
    (Path(adapter.submodule_dir) / "rgrl" / "inference" / "collect_rollouts.py").unlink()
    with pytest.raises(FileNotFoundError, match="pin"):
        adapter.collect_rollouts(ckpt, tmp_path / "out", num_episodes=2)


def test_an_unknown_rollout_env_lists_what_exists(adapter, ckpt, tmp_path, monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(ValueError, match="robomimic"):
        adapter.collect_rollouts(
            ckpt, tmp_path / "out", num_episodes=2, rollout_env="no_such_backend"
        )


def test_assemble_false_keeps_the_raw_recordings_tree(adapter, ckpt, tmp_path, monkeypatch):
    """Async-streaming sessions need offline extraction before assembly."""
    calls = _capture(monkeypatch)
    adapter.collect_rollouts(
        ckpt, tmp_path / "out", num_episodes=2, capture_layers=["enc"], assemble=False
    )
    assert [where for where, _ in calls] == ["submodule"], "no assembler run was expected"


def test_assemble_false_returns_a_path_not_a_rolloutset(adapter, ckpt, tmp_path, monkeypatch):
    """Regression: it used to build a RolloutDirectory over the un-assembled tree.

    `episodes/metadata.yaml` is written *by* assembly, so constructing a
    RolloutSet before step 2 raised FileNotFoundError. A real 3-episode sim
    collection hit this; the unit test missed it because it had mocked
    RolloutDirectory out, which is precisely the object that failed. So this one
    does NOT mock it -- the point is that nothing tries to construct it at all.
    """
    monkeypatch.setattr(RgrlAdapter, "_run_in_submodule", lambda self, args, log_path: None)
    out = tmp_path / "out"
    got = adapter.collect_rollouts(
        ckpt, out, num_episodes=2, capture_layers=["enc"], assemble=False
    )
    assert got == out.resolve(), "assemble=False should hand back the raw output dir"
    assert not (out / "episodes").exists(), "nothing should have been assembled"


def test_no_capture_layers_warns_that_the_graph_would_have_nothing(
    adapter, ckpt, tmp_path, monkeypatch, caplog
):
    _capture(monkeypatch)
    adapter.collect_rollouts(ckpt, tmp_path / "out", num_episodes=2)
    assert any(
        "no capture_layers" in r.message or "capture_layers" in r.getMessage()
        for r in caplog.records
    )


def test_the_module_no_longer_claims_rgrl_writes_nothing():
    """The false sentence, kept out by name."""
    import inspect

    from behavior_graphs.envs import rgrl_env

    source = inspect.getsource(rgrl_env)
    assert "doesn't write the canonical trajectory layout" not in source
    assert "collect_rollouts" in (rgrl_env.__doc__ or "")
