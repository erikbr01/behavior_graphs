"""Recovering a study session from the browser after the tab is closed.

The session id is in the URL, which survives a refresh but not a closed window:
someone who clicks the recruitment link again arrives at the bare link and looks
like a new participant, with half an hour of answers stranded in the store under
an id nobody holds. A cookie is what is left when the URL is gone.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest

from behavior_graphs.viz.streamlit.user_studies.browser_session import (
    COOKIE_PREFIX,
    MAX_AGE_SECONDS,
    cookie_name,
    remember_script,
    remembered_session,
)


class TestTheCookieItIsKeptIn:
    def test_two_participants_on_one_browser_do_not_share_a_session(self):
        """A lab laptop runs one person and then the next. Sharing the cookie
        would offer the second participant the first one's half-finished
        attempt."""
        assert cookie_name("three_stage_v1", "alice") != cookie_name("three_stage_v1", "bob")

    def test_two_studies_on_one_origin_do_not_share_a_session(self):
        assert cookie_name("three_stage_v1", "p") != cookie_name("contrastive_v1", "p")

    def test_it_is_namespaced_away_from_whatever_else_the_origin_sets(self):
        assert cookie_name("three_stage_v1", "p").startswith(f"{COOKIE_PREFIX}_")

    def test_a_link_with_no_participant_id_still_has_a_cookie(self):
        """Pilots and hand-built links have no id; one anonymous bucket is the
        best available, and no cookie at all would mean no recovery."""
        assert cookie_name("three_stage_v1", "")

    @pytest.mark.parametrize(
        "participant", ["a b", "a;b", "a=b", "a\nb", "../../etc", "p" * 200, "🙂"]
    )
    def test_a_hostile_participant_id_still_makes_a_legal_cookie_name(self, participant):
        """The id comes off the query string. A separator that survived into the
        name would let a link write some other cookie, or none at all."""
        name = cookie_name("three_stage_v1", participant)

        assert re.fullmatch(r"[A-Za-z0-9_-]+", name)
        assert len(name) < 200


class TestReadingItBack:
    def _cookies(self, mapping):
        return patch(
            "behavior_graphs.viz.streamlit.user_studies.browser_session.st.context",
            type("Ctx", (), {"cookies": mapping})(),
        )

    def test_the_id_this_browser_last_carried(self):
        with self._cookies({cookie_name("three_stage_v1", "alice"): "abc123"}):
            assert remembered_session("three_stage_v1", "alice") == "abc123"

    def test_another_participant_s_cookie_is_not_this_participant_s_session(self):
        with self._cookies({cookie_name("three_stage_v1", "bob"): "bobs_id"}):
            assert remembered_session("three_stage_v1", "alice") == ""

    def test_a_browser_that_sent_nothing_recalls_nothing(self):
        with self._cookies({}):
            assert remembered_session("three_stage_v1", "alice") == ""

    def test_no_cookies_at_all_is_not_an_error(self):
        """``st.context`` raises outside a request. Somebody trying to start a
        study should not meet a traceback because of it."""
        with self._cookies(None):
            assert remembered_session("three_stage_v1", "alice") == ""

    def test_a_context_that_raises_is_not_an_error_either(self):
        class Exploding:
            @property
            def cookies(self):
                raise RuntimeError("no request context")

        with patch(
            "behavior_graphs.viz.streamlit.user_studies.browser_session.st.context", Exploding()
        ):
            assert remembered_session("three_stage_v1", "alice") == ""


class TestWhatTheBrowserIsTold:
    """Asserted on the script rather than the page: ``AppTest`` renders a
    component as an opaque element with no readable content."""

    def test_it_writes_the_id_under_this_participant_s_cookie(self):
        emitted = remember_script(cookie_name("three_stage_v1", "alice"), "abc123")

        assert "document.cookie" in emitted
        assert cookie_name("three_stage_v1", "alice") in emitted
        assert '"abc123"' in emitted

    def test_it_survives_the_tab_closing_and_then_some(self):
        """A cookie with no max-age dies with the browser session, which is the
        one event this whole module exists for."""
        emitted = remember_script("n", "id")

        assert f"max-age={MAX_AGE_SECONDS}" in emitted
        assert MAX_AGE_SECONDS >= 24 * 60 * 60

    def test_it_is_sent_on_arrival_from_a_recruitment_link(self):
        """Lax, not Strict: the link is a cross-site top-level navigation, and
        Strict would withhold the cookie on exactly the visit that needs it."""
        emitted = remember_script("n", "id")

        assert "SameSite=Lax" in emitted
        assert "SameSite=Strict" not in emitted

    def test_it_is_secure_over_https_and_not_over_plain_http(self):
        """Setting Secure unconditionally makes a developer on localhost watch
        the browser silently drop it."""
        emitted = remember_script("n", "id")

        assert 'location.protocol === "https:"' in emitted
        assert '"; Secure"' in emitted

    def test_the_path_is_the_whole_app_not_the_page_it_was_set_on(self):
        assert "path=/" in remember_script("n", "id")

    def test_a_value_with_a_quote_in_it_cannot_break_out_of_the_script(self):
        """The id is ours, but the name is built from a URL parameter."""
        hostile = '"); alert(1); //'
        emitted = remember_script(cookie_name("v", hostile), hostile)

        quoted = re.findall(r'"(?:[^"\\]|\\.)*"', emitted)
        assert json.dumps(cookie_name("v", hostile)) in quoted
        assert json.dumps(hostile) in quoted
        # And the value is escaped again on the way into the header.
        assert "encodeURIComponent" in emitted

    def test_a_browser_refusing_cookies_does_not_break_the_page(self):
        """Losing the copy costs recovery, not the study."""
        emitted = remember_script("n", "id")

        assert "try {" in emitted and "catch (e) {}" in emitted
