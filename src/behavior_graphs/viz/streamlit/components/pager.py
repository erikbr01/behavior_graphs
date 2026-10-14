"""Paging state for the two places that show clips a page at a time.

The nav rows differ — the video browser wants one above and below its grid, the
graph's selection panels want one compact row of arrows — but the state maths is
the same, and the part worth having in one place is the clamp: a stored page
index has to be brought back inside the range *before* it is used, because the
number of pages changes under it. Switching the video panel from beside the graph
to below it takes the clips-per-page from 1 to 3, and the page index left over
from the narrow layout used to point past the end, which reached
``st.columns(0)`` — an exception where the panel should have been.
"""

from __future__ import annotations

import streamlit as st


def page_slice(items: list, per_page: int, page_key: str) -> tuple[list, int, int]:
    """``(this page's items, page index, page count)`` for ``items``.

    Reads and writes back ``st.session_state[page_key]``, clamped to the current
    page count so a stale index cannot survive a change in ``per_page``.
    """
    per_page = max(1, per_page)
    total_pages = max(1, (len(items) + per_page - 1) // per_page)
    page = max(0, min(int(st.session_state.get(page_key, 0)), total_pages - 1))
    st.session_state[page_key] = page
    start = page * per_page
    return items[start : start + per_page], page, total_pages
