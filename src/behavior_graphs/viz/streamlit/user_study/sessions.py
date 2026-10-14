"""Shared session-config loading for the group A/B study apps.

Both study apps present the same sidebar (participant ID + session selector),
resolve the same session yaml fields, and seed the same ``st.session_state``
keys.  Keeping that logic here guarantees the two study conditions cannot
silently diverge in path resolution, rollout-dir overrides, or submit gating.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
import yaml

REPO_ROOT = Path.cwd()


def sessions_dir() -> Path:
    """Directory holding study session yamls ($STUDY_SESSIONS_DIR override)."""
    return Path(
        os.environ.get("STUDY_SESSIONS_DIR", "").strip()
        or (REPO_ROOT / "configs" / "viz" / "study")
    )


def list_sessions(directory: Path) -> tuple[list[Path], dict[str, str]]:
    """Session yaml files in *directory* plus their display labels."""
    files = sorted(directory.glob("*.yaml")) if directory.is_dir() else []
    labels = {f.stem: yaml.safe_load(f.read_text()).get("label", f.stem) for f in files}
    return files, labels


DEFAULT_SESSION = "set_table_baseline"


def default_session_index(files: list[Path]) -> int:
    """Selectbox index for :data:`DEFAULT_SESSION`, else the first session."""
    stems = [f.stem for f in files]
    return stems.index(DEFAULT_SESSION) if DEFAULT_SESSION in stems else 0


def resolve_session_choice(
    configured_session: str | None,
    files: list[Path],
    labels: dict[str, str],
    *,
    key: str,
) -> str:
    """Resolve a study session, treating an explicitly configured one as authoritative.

    Study variants describe which session supplies their evidence. Showing a second,
    independent selector for that same value makes it easy to accidentally run a
    variant with the wrong task. Legacy callers without a valid configured session
    retain the manual selector as a fallback.
    """
    stems = [f.stem for f in files]
    if configured_session in stems:
        return configured_session

    return st.sidebar.selectbox(
        "Session",
        options=stems,
        index=default_session_index(files),
        format_func=lambda session: labels.get(session, session),
        key=key,
    )


def resolve_path(p: str) -> Path:
    """Substitute ``${oc.env:VAR}`` references and anchor relative paths at the repo root."""
    p = re.sub(r"\$\{oc\.env:([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), str(p))
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_run_dir(sess: dict) -> Path | None:
    """Run dir from the session yaml, else $BEHAVIOR_GRAPHS_RUN_DIR."""
    raw = sess.get("run_dir") or os.environ.get("BEHAVIOR_GRAPHS_RUN_DIR", "").strip()
    return resolve_path(str(raw)) if raw else None


def load_session_artifacts(sess_path: Path, data_path: str | None = None):
    """Load a session yaml, its run artifacts, and the legacy mp4 index.

    Returns ``(sess, config_path, run, mp4_dir, index, errors)``; ``run`` is
    None (and ``errors`` non-empty) when anything failed to load.
    """
    from behavior_graphs.viz.streamlit.data import empty_mp4_index, legacy_mp4_index, load_run

    sess = yaml.safe_load(sess_path.read_text())
    config_path = resolve_path(sess["study_config"]) if "study_config" in sess else sess_path

    errors: list[str] = []
    run_dir = resolve_run_dir(sess)
    if run_dir is None:
        errors.append("No run_dir in the session config and $BEHAVIOR_GRAPHS_RUN_DIR unset.")
    if not config_path.exists():
        errors.append(f"Study config not found: {config_path}")

    run = None
    mp4_dir, index = None, empty_mp4_index()
    if run_dir is not None and not errors:
        try:
            run = load_run(str(run_dir), data_path=data_path)
        except Exception as e:  # missing artifacts, bad path, ...
            errors.append(f"Failed to load run dir {run_dir}: {e}")
        if run is not None:
            rd_override = sess.get("rollout_dir")
            effective_rollout_dir = (
                resolve_path(str(rd_override)) if rd_override else run.rollout_dir
            )
            mp4_dir, index = legacy_mp4_index(effective_rollout_dir)
            if not index["episodes"]:
                errors.append(f"No episode videos found in rollout dir of run {run_dir}.")
    return sess, config_path, run, mp4_dir, index, errors


def seed_common_state(pfx: str, sess: dict, cfg: dict, mp4_dir: Path, index: dict) -> None:
    """Populate the session-state keys shared by both study apps."""
    st.session_state[f"{pfx}_index"] = index
    st.session_state[f"{pfx}_strategies"] = cfg["strategies"]
    st.session_state[f"{pfx}_budget"] = cfg.get("budget", {}).get("total_demos", 500)
    st.session_state[f"{pfx}_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
    st.session_state[f"{pfx}_ranking"] = cfg.get("ranking")
    st.session_state[f"{pfx}_mp4_dir"] = str(mp4_dir)
    st.session_state[f"{pfx}_rollout_limit"] = sess.get("rollout_time_limit_seconds", 600)
    st.session_state[f"{pfx}_hide_episode_index"] = bool(sess.get("hide_episode_index", False))
    # demo_videos_dir: explicit override or default to <mp4_dir>/demo_videos
    dvd = sess.get("demo_videos_dir")
    st.session_state[f"{pfx}_demo_videos_dir"] = (
        str(resolve_path(dvd)) if dvd else str(mp4_dir / "demo_videos")
    )
    st.session_state[f"{pfx}_task_description"] = sess.get("task_description", "")


def task_description_markdown(task_description: str) -> str:
    """The intro page's task blurb, with a generic fallback."""
    return (
        f"\n{task_description}"
        if task_description
        else "\nThe robot must complete a manipulation task."
    )


def submit_allowed(responses: dict) -> bool:
    """Warn about unanswered survey questions; True when submission may proceed."""
    unanswered = [k for k, v in responses.items() if v is None]
    if unanswered:
        st.warning(f"Please answer all questions before submitting ({len(unanswered)} remaining).")
    return not unanswered
