"""NASA Task Load Index (TLX) survey block, rendered in the user-study pages.

The six NASA-TLX dimensions with the canonical low / high anchors, except
``temporal_demand``, which asks how long the work took rather than how much time
*pressure* there was: nothing in this study sets a pace, so the canonical item
had nothing to measure. A score from this block is therefore not comparable to a
published TLX score on that dimension.

The dimensions are
each rated 0-100 in steps of 20 (see :data:`RATING_POINTS`). The helper returns
the responses as a dict the page can serialize into the submission JSON, with
``None`` for a dimension nobody answered.

Reference: Hart & Staveland (1988), "Development of NASA-TLX (Task Load
Index): Results of empirical and theoretical research".
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

# (key, accessible name, low_label, high_label). The second field is not
# drawn: the question in _ITEM_DESCRIPTIONS is the heading, and this is what the
# radio is called for anyone reading the page with a screen reader.
_NASA_TLX_ITEMS: list[tuple[str, str, str, str]] = [
    (
        "mental_demand",
        "Mental Demand",
        "Very Low",
        "Very High",
    ),
    (
        "physical_demand",
        "Physical Demand",
        "Very Low",
        "Very High",
    ),
    (
        "temporal_demand",
        "Time Taken",
        "Very Little",
        "Lots of Time",
    ),
    (
        "performance",
        "Performance",
        "Perfect",
        "Failure",
    ),
    (
        "effort",
        "Effort",
        "Very Low",
        "Very High",
    ),
    (
        "frustration",
        "Frustration",
        "Very Low",
        "Very High",
    ),
    # Not a NASA-TLX dimension. Rendered by this block because it belongs
    # directly under the load items a page asks and shares their scale, but a
    # TLX score must not be computed over it — hence its absence from the
    # default item list, which stays the canonical six.
    (
        "confidence",
        "Confidence",
        "Not at all confident",
        "Completely confident",
    ),
]

_ITEM_DESCRIPTIONS: dict[str, str] = {
    "mental_demand": "How much mental and perceptual activity was required (e.g. thinking, deciding, looking, searching)? Was the task easy or demanding, simple or complex?",
    "physical_demand": "How much physical activity was required (e.g. clicking, scrolling)? Was the task easy or demanding, slow or brisk?",
    # Not the canonical TLX item, which asks about time *pressure* — nothing in
    # this study sets a pace, so participants had none to report. Asking how
    # long the work took is the thing this study can act on.
    "temporal_demand": "How much time did answering take?",
    "performance": "How successful do you think you were in accomplishing the goals of the task set by the experimenter? How satisfied were you with your performance? (Note: lower is better.)",
    # Directer than the canonical item, which asks about work "to accomplish
    # your level of performance" -- a participant answering questions about a
    # policy has no performance level to reason about.
    "effort": "How much effort was it to answer the questions?",
    "frustration": "How insecure, discouraged, irritated, stressed, and annoyed (vs. secure, gratified, content, relaxed, and complacent) did you feel during the task?",
    # How sure they are of the answers they just gave, which is the thing a
    # display is supposed to move and the load items do not measure: two
    # participants can spend the same effort and come away differently certain.
    "confidence": "How confident are you in the answers you just gave?",
}


#: The points a rating can take: 0-100 in steps of 20, six of them. NASA-TLX's
#: own instrument is a 20-interval visual-analog line, which is what the slider
#: was imitating; a six-point scale is coarser but is what a participant can
#: actually distinguish, and it makes two answers comparable rather than two
#: pixel positions.
RATING_STEP = 20
RATING_POINTS = tuple(range(0, 101, RATING_STEP))

#: What a caller gets by asking for nothing: the canonical six, and not
#: ``confidence``, which this block can render but which is not part of the
#: instrument. A default that quietly included it would put a non-TLX item into
#: every legacy page's score.
TLX_DEFAULT_ITEMS = (
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
)


def render_nasa_tlx(key_prefix: str, items: Sequence[str] | None = None) -> dict[str, int | None]:
    """Render the NASA-TLX block and return ``{item_key: rating or None}``.

    *items* selects and orders the dimensions by key, defaulting to all six. A
    study asking the same block twice per participant pays for every dimension
    twice, so a page may keep only the ones it will report on -- the returned
    dict then has only those keys, and a raw TLX score is no longer comparable
    to a six-dimension one.

    Nothing is pre-selected, and an unanswered dimension comes back ``None``.
    The slider this replaced started every dimension at 50 and recorded that
    value whether or not the participant ever touched it, which is
    indistinguishable in the data from a deliberate "middling" answer.
    """
    wanted = list(items) if items is not None else list(TLX_DEFAULT_ITEMS)
    unknown = [key for key in wanted if key not in _ITEM_DESCRIPTIONS]
    if unknown:  # a typo would otherwise silently drop a dimension
        raise ValueError(f"unknown NASA-TLX items: {unknown}")  # noqa: TRY003
    chosen = [item for item in _NASA_TLX_ITEMS if item[0] in set(wanted)]
    chosen.sort(key=lambda item: wanted.index(item[0]))
    st.markdown(f"Please answer each of the {len(chosen)} questions below.")
    scope = f"{key_prefix}_scale"
    # A horizontal radio group is a flex row that starts at the left edge of
    # whatever it is in, which put the scale hard against the left margin and
    # left the anchors stranded from the points they anchor.
    st.markdown(
        # Prefix match: each item's container is `st-key-<scope>_<item>`, so a
        # selector for the scope alone matches nothing.
        # One centred row per dimension: anchor, the points, anchor. Streamlit
        # stacks a container's children vertically, so the row is CSS -- and its
        # children have to be content-width, or the anchors sit a column away
        # from the points they anchor (which is what st.columns did).
        f"<style>[class*='st-key-{scope}_']"
        "{ flex-direction: row !important; align-items: center;"
        "  justify-content: center; gap: 0.55rem; }"
        f"[class*='st-key-{scope}_'] > div[class*='stElementContainer']"
        "{ width: auto; }"
        f"[class*='st-key-{scope}_'] div[role='radiogroup']"
        "{ gap: 0.4rem; }</style>",
        unsafe_allow_html=True,
    )
    responses: dict[str, int | None] = {}
    for item_key, label, low, high in chosen:
        # The question itself is the heading. It used to be a short title with
        # the question under it as a caption — "Time Taken", then "How much time
        # did answering take?" — which is the same thing said twice, the second
        # time in the smaller type. The title survives as the radio's accessible
        # name, which is what a screen reader announces for the scale.
        st.markdown(f"**{_ITEM_DESCRIPTIONS[item_key]}**")
        with st.container(key=f"{scope}_{item_key}"):
            st.markdown(_anchor(low), unsafe_allow_html=True)
            # Radio, not a slider: the points are the answer set, and a slider
            # cannot be left unanswered -- it always reports its handle.
            responses[item_key] = st.radio(
                label,
                options=RATING_POINTS,
                index=None,
                horizontal=True,
                key=f"{key_prefix}_{item_key}",
                label_visibility="collapsed",
            )
            st.markdown(_anchor(high), unsafe_allow_html=True)
    return responses


def _anchor(text: str) -> str:
    """One end of the scale, as small grey text beside its end point."""
    return f"<span style='color:#8b8b8b;font-size:0.85em;white-space:nowrap;'>{text}</span>"
