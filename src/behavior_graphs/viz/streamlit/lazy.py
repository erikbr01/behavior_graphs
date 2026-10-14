"""Collapsible sections whose body is only *computed* when open.

``st.expander`` is a purely visual container: by default the code inside it runs
on every rerun whether or not the panel is open. Several panels in these apps
index thousands of run directories or render a whole trajectory tree, so a
collapsed expander was costing seconds per interaction for output nobody could
see.

For an expander, Streamlit itself has the answer — pass ``on_change="rerun"``
and read ``.open`` off the returned container::

    panel = st.expander("📊 Model selection", on_change="rerun")
    if panel.open:
        with panel:
            render_model_selection(...)

(The ``on_change`` is what makes it stateful; without it ``.open`` is None.)

:func:`lazy_section` covers the case Streamlit does not: a section whose body is
top-level page content rather than something that should read as nested, so an
expander's chrome would be wrong::

    if lazy_section("Raw scores", key="sa_raw"):
        st.dataframe(...)
"""

from __future__ import annotations

import streamlit as st


def lazy_section(
    label: str,
    key: str,
    *,
    default_open: bool = False,
    help: str | None = None,
) -> bool:
    """Render a collapsible header button; return True when the body should render."""
    state_key = f"_lazy_open_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = bool(default_open)

    def _toggle() -> None:
        st.session_state[state_key] = not st.session_state[state_key]

    # Read before drawing the button: the on_click callback runs before the
    # rerun, so this value is already the post-click state.
    is_open = bool(st.session_state[state_key])
    st.button(
        f"{'▼' if is_open else '▶'}  {label}",
        key=f"{key}_lazy_toggle",
        on_click=_toggle,
        use_container_width=True,
        help=help,
    )
    return is_open
