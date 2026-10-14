"""Sidebar picker for which camera views the video players show.

A dataset can carry several views of every episode (``media/<camera>/`` in the
canonical rollout layout — see
:meth:`behavior_graphs.envs.trajectory_dir.TrajectoryDirectory.camera_names`).
Choosing between them has to leave playback alone: a Streamlit rerun rebuilds
every component iframe from scratch, which restarts each video at frame 0 and
throws away the sync the players maintain between views. So the change never
becomes a rerun of the page:

* the picker is wrapped in :func:`streamlit.fragment`, so selecting a view
  reruns the sidebar alone — the page script, and with it every video iframe,
  is not re-executed;
* the fragment then broadcasts the selection on a ``BroadcastChannel`` that
  every mounted player listens on (see
  :func:`~behavior_graphs.viz.streamlit.components.mp4_player._sync_js`), and the
  players show and hide panels already in their DOM. Component iframes are
  ``srcdoc`` with ``allow-same-origin``, so they share the app's origin and see
  the same channel — which is also why the name is scoped per session (see
  :func:`channel_name`): same-origin reaches other browser tabs too.

Which is why the players render *every* view the dataset has, hidden ones
included: revealing a panel client-side only works if it is already there. The
cost is one web proxy per view per shown clip, whether or not it is on screen.

The picker is on the graph demo's sidebar; the panels that read its selection
also render on Sweep Analysis and in the study apps, which don't draw one (a
nested fragment isn't allowed inside Sweep Analysis's tree fragment, and the
study apps are controlled instruments). Those pages therefore show the default —
every view — and :func:`state_key` keeps a choice made on the demo from leaking
into a *different* dataset's panels there.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

#: Base BroadcastChannel name the picker and the players agree on, before
#: :func:`channel_name` scopes it to the session.
CHANNEL = "behavior_graphs_camera_views"

#: Prefix of the session-state key holding the selected view names.
STATE_KEY = "camera_views"


def channel_name() -> str:
    """The picker's channel for *this* session, not a process-wide one.

    ``BroadcastChannel`` reaches every same-origin browsing context, so a bare
    global name crosses browser tabs: with two tabs open on datasets that share
    camera names — the normal DROID case, where every dataset has
    ``exterior_image_1_left`` and a wrist view — narrowing the picker in one tab
    hid views in the other, whose own picker still showed them selected. That is
    the cross-page leak :func:`state_key` scopes the Python side against, so the
    JS side is scoped the same way, by the one thing shared by the picker and the
    players of one browser tab and by nothing else: the Streamlit session id.

    Falls back to the bare name outside a script run (tests, a bare import),
    where there is no session to scope to and nothing else to collide with.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
    except Exception:  # pragma: no cover - Streamlit internals moved
        return CHANNEL
    return f"{CHANNEL}:{ctx.session_id}" if ctx is not None else CHANNEL


def state_key(cameras: Sequence[str], prefix: str = STATE_KEY) -> str:
    """Session-state key for a selection over exactly *cameras*.

    *prefix* separates one selection from another over the same views — the VLM
    annotation modal keeps its own, because what you watch and what you pay a
    model to look at are different choices — while keeping this scoping rule in
    one place for both.

    Scoped to the view names, not global, because a session moves between
    datasets and pages: the picker lives on the graph demo, while the panels that
    read the selection also render on Sweep Analysis and in the study apps. A
    single key would let a choice made against one robot's views silently apply
    to another's — and, when the names don't overlap at all, collapse those
    pages to one view chosen by a picker that isn't even on screen. Keying by the
    view set means datasets that share views share the choice, which is what you
    want when it *is* the same pair of cameras, and nothing else interferes.
    """
    return f"{prefix}:" + ",".join(cameras)


def render_camera_selector(cameras: Sequence[str]) -> list[str]:
    """Draw the picker where it is called and return the views to show.

    Renders nothing at all for a dataset with fewer than two views — there is
    nothing to choose — and hands back what it was given.
    """
    cameras = list(cameras)
    if len(cameras) < 2:
        return cameras
    _selector(cameras, state_key(cameras))
    return selected_cameras(cameras)


def narrowed(chosen: Sequence[str], cameras: Sequence[str]) -> list[str]:
    """*chosen*, or the first view when it is empty.

    An emptied picker is a real selection, not an unset one, and every consumer
    degrades the same way: to the single view the picker says it is showing. Shared
    so the players, this module's own reader and the annotation modal cannot drift
    on what "nothing selected" means.
    """
    return list(chosen) or list(cameras[:1])


def selected_cameras(cameras: Sequence[str]) -> list[str]:
    """The views to show, read live from session state.

    Read at render time rather than passed down from the page: the video panels
    are fragments, and a fragment rerun re-executes with the arguments it was
    *first* called with, so a selection threaded through as an argument goes
    stale the moment the user changes it.
    """
    cameras = list(cameras)
    if len(cameras) < 2:
        return cameras
    stored = st.session_state.get(state_key(cameras))
    if stored is None:
        return cameras  # nothing picked yet: the picker's own default
    # An emptied picker is a real selection, not an unset one, and falls back to
    # the same single view the picker says it is showing.
    return narrowed([c for c in stored if c in cameras], cameras)


def player_views(entry: dict, mp4_dir: Path | None, mp4_index: dict) -> dict[str, Any]:
    """The ``videos`` / ``visible_cameras`` / ``view_transforms`` a player needs.

    Every view goes in, so the picker can reveal one without a rerun; only the
    selected ones start out visible.
    """
    from behavior_graphs.viz.streamlit.clips import dataset_transforms, episode_camera_clips
    from behavior_graphs.viz.streamlit.data import mp4_index_cameras

    cameras = mp4_index_cameras(mp4_index)
    return {
        "videos": episode_camera_clips(entry, mp4_dir, cameras or None),
        "visible_cameras": selected_cameras(cameras),
        "view_transforms": dataset_transforms(mp4_dir),
    }


@st.fragment
def _selector(cameras: list[str], key: str) -> None:
    """The widget itself, in a fragment so its reruns stop at the sidebar.

    Returns nothing because a fragment's return value is None on its own
    reruns; callers read :func:`selected_cameras` instead.
    """
    chosen = st.multiselect(
        "Camera views",
        options=cameras,
        default=cameras,
        key=key,
        help=(
            "Views to show side by side. Playback stays in sync across them, and "
            "changing the selection applies to the players already on screen "
            "without restarting them."
        ),
    )
    if not chosen:
        st.caption(f"No view selected — showing `{cameras[0]}`.")
    _broadcast(narrowed(chosen, cameras))


def _broadcast(cameras: Sequence[str]) -> None:
    """Push the selection to every mounted player.

    A zero-height component whose HTML embeds the selection: it is re-mounted
    (and so re-runs) exactly when the selection changes, and re-broadcasting an
    unchanged one would be a no-op anyway — the players' handler is idempotent.
    """
    payload = json.dumps({"cameras": list(cameras)})
    components.html(
        f"""<script>
(function(){{
  try{{
    var bus=new BroadcastChannel({json.dumps(channel_name())});
    bus.postMessage({payload});
    bus.close();
  }}catch(err){{}}
}})();
</script>""",
        height=0,
    )
