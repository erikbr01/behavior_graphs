"""Custom Likert-scale survey blocks for the data-collection user study.

Three blocks, all on a 5-point scale (1 = Do not agree, 5 = Fully agree):

  Block 1  Behavior-Graph Interaction
           Only meaningful for participants who saw the graph. Rendered
           in the graph-condition page after the graph exploration steps
           and before strategy selection.

  Block 2  Strategy-Selection Experience
           Universal. Rendered after the strategy-allocation step.

  Block 3  Final Assessment
           Q14/Q15 universal (prior experience). Q16/Q17 only for the
           graph condition (would-use + wish-for-more questions).
"""

from __future__ import annotations

import streamlit as st

# ── Question banks ───────────────────────────────────────────────────────────
# (key, question_text)

_BLOCK1_GRAPH_INTERACTION: list[tuple[str, str]] = [
    (
        "b1_q1_overview",
        "The behavior graph gave me a clear overview of how the robot executes the task.",
    ),
    (
        "b1_q2_modes",
        "I was able to identify distinct behavioral modes of the robot from the graph.",
    ),
    ("b1_q3_videos", "The video examples helped me interpret what each graph node represents."),
    (
        "b1_q4_failure_transitions",
        "I understand which transitions in the graph are associated with task failure.",
    ),
    (
        "b1_q5_confidence",
        "I feel confident about which behavioral modes I identified as most in need of "
        "additional data collection.",
    ),
    (
        "b1_q6_easy_to_identify",
        "I found it easy to identify behavioral modes where additional data collection "
        "would be most beneficial.",
    ),
    (
        "b1_q7_graph_to_strategy",
        "The behavior graph helped me understand what kind of data collection "
        "strategies would improve the robot.",
    ),
]

_BLOCK2_STRATEGY: list[tuple[str, str]] = [
    ("b2_q8_confidence", "I felt confident about the data collection strategies I selected."),
    (
        "b2_q9_easy_choice",
        "I found it easy to choose between the different data collection strategy options.",
    ),
    (
        "b2_q10_addresses_failure",
        "I felt that the strategies I selected would address the robot's failure modes.",
    ),
    (
        "b2_q11_understood_impact",
        "I understood how different data collection strategies would affect the robot's "
        "performance.",
    ),
    (
        "b2_q12_targeted",
        "I felt that my strategy choices were targeted at the right behavioral modes.",
    ),
    ("b2_q13_intuitive", "The strategy selection process felt intuitive."),
]

_BLOCK3_FINAL_UNIVERSAL: list[tuple[str, str]] = [
    ("b3_q14_prior_demo_experience", "I have prior experience collecting robot demonstrations."),
    ("b3_q15_prior_robot_experience", "I have prior experience working with robots."),
]

_BLOCK3_FINAL_GRAPH_ONLY: list[tuple[str, str]] = [
    (
        "b3_q16_would_use",
        "I would use a behavior graph tool to guide my data collection strategy "
        "decisions in future projects.",
    ),
    ("b3_q17_more_interaction", "I wished for more ways to interact with the behavior graph."),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _render_likert_block(items: list[tuple[str, str]], key_prefix: str) -> dict[str, int]:
    """Render a sequence of 5-point Likert questions; return ``{key: rating}``."""
    responses: dict[str, int] = {}
    for item_key, prompt in items:
        st.markdown(prompt)
        # End-labels sit outside the radio (not as per-option captions) so
        # "Do not agree" doesn't widen option 1's column and throw off the
        # spacing between the five circles. The radio packs its 5 options
        # against the left edge of radio_col, so a wide radio_col leaves
        # dead space only on the right (next to "Fully agree"); using equal
        # thirds and right-aligning the left label keeps both labels close
        # to roughly equidistant from their nearest circle.
        left, radio_col, right = st.columns([1, 1, 1], vertical_alignment="center")
        with left:
            st.caption('<div style="text-align:right">Do not agree</div>', unsafe_allow_html=True)
        with radio_col:
            responses[item_key] = st.radio(
                prompt,
                options=[1, 2, 3, 4, 5],
                index=None,
                horizontal=True,
                key=f"{key_prefix}_{item_key}",
                label_visibility="collapsed",
            )
        with right:
            # Same wrapper markup as the left label (rather than plain
            # st.caption) so both columns have identical DOM structure --
            # otherwise the two captions render at slightly different
            # heights and vertical_alignment="center" centers them
            # against each other, not against the radio circles.
            st.caption('<div style="text-align:left">Fully agree</div>', unsafe_allow_html=True)
    return responses


def render_block1_graph_interaction(key_prefix: str) -> dict[str, int]:
    """Block 1 — administered after the graph-exploration stage."""
    st.markdown(
        "Please rate each statement on a 5-point scale (1 = Do not agree, 5 = Fully agree)."
    )
    return _render_likert_block(_BLOCK1_GRAPH_INTERACTION, key_prefix)


def render_block2_strategy(key_prefix: str) -> dict[str, int]:
    """Block 2 — administered after the strategy-selection stage."""
    st.markdown(
        "Please rate each statement on a 5-point scale (1 = Do not agree, 5 = Fully agree)."
    )
    return _render_likert_block(_BLOCK2_STRATEGY, key_prefix)


def render_block3_final(key_prefix: str, *, include_graph_questions: bool) -> dict[str, int]:
    """Block 3 — prior-experience + (optional) graph-tool questions."""
    st.markdown(
        "Please rate each statement on a 5-point scale (1 = Do not agree, 5 = Fully agree)."
    )
    items = list(_BLOCK3_FINAL_UNIVERSAL)
    if include_graph_questions:
        items += _BLOCK3_FINAL_GRAPH_ONLY
    return _render_likert_block(items, key_prefix)
