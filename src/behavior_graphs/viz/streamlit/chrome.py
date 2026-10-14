"""Page chrome shared by the apps that show one thing and nothing else.

The study app hides the sidebar because a participant has no business seeing
which condition they are in; the reviewer app hides it because there is nothing
to put on it. Both want the same rule applied the same way, and the way is not
obvious (see :func:`hide_sidebar`), so it lives here rather than being copied.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

#: Keyed so the CSS below can address the injector: Streamlit gives a container
#: with a key the class ``st-key-<key>``.
_INJECTOR_KEY = "hide_sidebar_css"

_HIDE_SIDEBAR_CSS = (
    "[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl']{display:none !important;}"
    # The injector is a zero-height iframe, but a zero-height block still costs
    # one of the main column's 16px flex gaps, which pushed every participant
    # page down by one. Taken out of the flow rather than `display:none`d, so
    # there is no question of whether a hidden iframe still runs its script; the
    # layout wrapper Streamlit puts around a keyed container is the flex child,
    # so that is what has to move.
    f"[data-testid='stLayoutWrapper']:has(> .st-key-{_INJECTOR_KEY}), .st-key-{_INJECTOR_KEY}"
    "{position:absolute !important;width:0 !important;height:0 !important;overflow:hidden !important;}"
)


def hide_sidebar() -> None:
    """Hide the sidebar for the rest of the page, durably.

    Hidden with CSS rather than by not rendering it, so the same renderers serve
    the lab app, where the sidebar's controls are the point.

    Written twice, deliberately. The ``st.markdown`` rule is an app-body element
    and a fragment rerun sweeps away app-body elements it did not itself
    re-emit, this ``<style>`` among them, with nothing putting it back until the
    next full rerun — measured on a study walk-through, back when the
    data-collection cards were a fragment: after picking a card the sidebar was
    visible, with the participant and session ids on it. The graph is a fragment
    too, so the hazard stands wherever this is used. A node in the document's
    ``<head>`` is outside that reconciliation, so it survives; it is written
    once, guarded by its id, since the component reloads with the page.

    Call it after ``st.set_page_config`` (pass
    ``initial_sidebar_state="collapsed"`` there as a belt to this brace: if the
    rule ever fails to apply, a collapsed sidebar is a better failure than an
    open one).
    """
    st.markdown(f"<style>{_HIDE_SIDEBAR_CSS}</style>", unsafe_allow_html=True)
    with st.container(key=_INJECTOR_KEY):
        components.html(
            f"""<script>
              const marker = "study-hide-sidebar";
              const outer = window.parent.document;
              if (outer && !outer.getElementById(marker)) {{
                const rule = outer.createElement("style");
                rule.id = marker;
                rule.textContent = {json.dumps(_HIDE_SIDEBAR_CSS)};
                outer.head.appendChild(rule);
              }}
            </script>""",
            height=0,
        )
