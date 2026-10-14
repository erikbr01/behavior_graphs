"""Shared password gate for the public-facing Streamlit apps (demo + study)."""

from __future__ import annotations

import hashlib
import hmac
import os

import streamlit as st


def password_gate(label: str = "Password", *, title: str | None = None) -> bool:
    """Return True if the request is authorized.

    If APP_PASSWORD_SHA256 is unset/empty, the app is open (local dev).
    Otherwise, a ``?pwd=`` query param whose SHA-256 hex digest matches the
    env var auto-authorizes the request (for sharing deep links without
    passing the password out of band separately) -- accepted tradeoff: unlike
    the manual prompt below, this puts the plaintext password in the URL,
    which lands in proxy access logs and browser history. Failing that,
    render a password prompt (preceded by ``title``, if given) and return
    False until the visitor enters a password whose SHA-256 hex digest
    matches the env var. Comparison is constant-time in both paths.

    A plain text_input is used instead of st.form: Enter submits either way,
    and the form's submit button arrives one render frame after the form
    itself, which flashes Streamlit's red "Missing Submit Button" warning on
    slow (proxied) first loads.
    """
    expected = os.environ.get("APP_PASSWORD_SHA256", "").strip().lower()
    if not expected:
        return True
    if st.session_state.get("_authed"):
        return True

    qp_pw = st.query_params.get("pwd")
    if qp_pw:
        got = hashlib.sha256(qp_pw.encode("utf-8")).hexdigest()
        if hmac.compare_digest(got, expected):
            st.session_state["_authed"] = True
            return True
        # Wrong/stale ?pwd= falls through to the normal prompt silently (no
        # st.error) -- the user didn't type anything, so don't scold them for
        # a shared link having drifted; they can just type the real password.

    if title:
        st.title(title)
    pw = st.text_input(label, type="password")
    if pw:
        got = hashlib.sha256(pw.encode("utf-8")).hexdigest()
        if hmac.compare_digest(got, expected):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False
