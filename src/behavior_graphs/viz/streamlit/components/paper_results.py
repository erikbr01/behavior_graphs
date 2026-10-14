"""Read the exported results once, and say so clearly when one is missing.

:mod:`behavior_graphs.results` is the data; this is the few lines of page plumbing
around it, shared by the result pages of ``apps/reviewer`` so none of them
invents its own failure message.

``cache_resource`` rather than ``cache_data``: the document is read-only and the
same for every session, and ``cache_data`` would deep-copy a 90 kB structure per
call per rerun to protect a mutation nobody makes.
"""

from __future__ import annotations

import streamlit as st

from behavior_graphs import results


@st.cache_resource(show_spinner=False)
def _loaded() -> tuple[dict, str]:
    """``(document, problem)`` — the failure cached alongside the success.

    A page that cannot read the export must say *that*, not "this section is
    missing": the two have different fixes, and the one that matters more is
    the one a re-export does not solve.
    """
    try:
        return results.load(), ""
    except results.ResultsUnavailable as error:
        return {}, str(error)


def document() -> dict:
    """The whole export, ``{}`` where it could not be read."""
    return _loaded()[0]


def section(name: str) -> dict:
    """One section, or stop the page with the reason.

    Stops rather than returning empty: every caller draws charts of this, and a
    page of empty axes reads as a study that found nothing.
    """
    found, problem = _loaded()
    if problem:
        st.error(problem)
        st.stop()
    try:
        return results.section(found, name)
    except results.ResultsUnavailable as error:
        st.error(str(error))
        st.stop()
    return {}  # pragma: no cover - st.stop raises


@st.cache_resource(show_spinner=False)
def _loaded_curves() -> tuple[dict, str]:
    try:
        return results.load_curves(), ""
    except results.ResultsUnavailable as error:
        return {}, str(error)


def curves() -> dict:
    """The predicted progress curves, or stop the page with the reason.

    Separate from :func:`section` because the curves are a separate file. A
    checkout holding the document and not the curves is a real state -- the
    curves are a quarter of a megabyte of bulk and the document is the part a
    person reviews -- so the message has to name the missing file rather than a
    section key.
    """
    found, problem = _loaded_curves()
    if problem:
        st.error(problem)
        st.stop()
    return found
