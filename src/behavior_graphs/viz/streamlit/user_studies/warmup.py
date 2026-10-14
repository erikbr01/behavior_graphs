"""Paying a fresh process's start-up cost before a participant does.

Streamlit runs every session in one process, so the first script run after a
restart imports the whole study (numpy, plotly, the graph component, the media
layer) and takes the first read of each run's artifacts. Everything after it
reuses both: ``sys.modules`` is process-wide and :func:`behavior_graphs.viz
.streamlit.data.load_run` is an ``st.cache_resource``. Measured against the
deployment, three separate browsers into a freshly restarted app:

    person 1, cold process    12,638 ms
    person 2                   1,386 ms
    person 3                   1,699 ms

One person pays, and twelve seconds of blank page is enough for a recruited
participant to close the tab. So ``deploy/warm_study.py`` sends that first hit
deliberately, before anybody is invited.

What matters here is that a warm-up is **not a participant**. The layout writes
a record on first page load, so a warm-up that went through the ordinary route
would leave an abandoned attempt in the funnel after every restart, and the
completed-of-started metric would drift down each time the app was redeployed.
``apps/study/Home.py`` therefore checks for :data:`WARMUP_PARAM` before it hands
over to the variant's renderer, which is the only thing that writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: Query parameter that marks a request as a warm-up rather than a visit.
WARMUP_PARAM = "warmup"

#: Printed on the warmed page, so the script can tell a warm-up that worked from
#: one that reached an error page and still returned HTTP 200.
WARMUP_MARKER = "study-warm-ok"


def warm(variant: Any) -> list[str]:
    """Load what the first participant would otherwise wait for.

    Returns the sessions successfully loaded. Never raises: a warm-up is an
    optimisation, and one that fails should leave a deployable app behind, not
    take the restart down with it. A session it could not load is logged and
    skipped, and the participant who meets it pays the cost this avoids.
    """
    from behavior_graphs.viz.streamlit.data import resolve_data_path
    from behavior_graphs.viz.streamlit.user_study.sessions import (
        load_session_artifacts,
        sessions_dir,
    )

    warmed: list[str] = []
    for stage in getattr(variant, "stages", None) or []:
        name = getattr(stage, "session_name", "")
        if not name:
            continue
        try:
            load_session_artifacts(
                Path(sessions_dir()) / f"{name}.yaml", data_path=resolve_data_path()
            )
        except Exception as error:  # a missing run, an unreadable mount, ...
            _log.warning("warm-up could not load session %s: %s", name, error)
            continue
        warmed.append(name)
    return warmed
