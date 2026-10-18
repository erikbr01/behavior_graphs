"""Streamlit AppTest accessors that moved between the versions we support.

`pyproject.toml` floors streamlit at 1.58 and resolves to whatever is current,
so the suite has to read an AppTest the same way across that range. Two
accessors changed and both are reached only from tests:

* images were ``at.get("imgs")`` and are now ``at.image``;
* the session-state mapping was the private ``filtered_state`` and is now a
  public ``to_dict()``.

Prefer the newer, public spelling and fall back, so neither end of the
supported range fails on an API that was never the product's.
"""

from __future__ import annotations

from typing import Any


def images(at: Any) -> list:
    """Every image element on the page."""
    try:
        return list(at.image)
    except AttributeError:
        return list(at.get("imgs"))


def session_values(at: Any) -> dict:
    """The session state as a plain mapping."""
    state = at.session_state
    to_dict = getattr(state, "to_dict", None)
    if to_dict is not None:
        return dict(to_dict())
    return dict(state.filtered_state)
