"""Registry of selectable user-study setups/versions, surfaced as one page.

This package is deliberately separate from ``behavior_graphs.viz.streamlit.
user_study`` (singular), which holds the original Group A / Group B study
script and is still what ``study.*`` deploys. New study designs are drafted
here as their own variant; the original study is itself wired in as the
``production_group_ab`` variant (``variants/production/render.py``) rather
than duplicated, so there remains exactly one copy of its logic. All variants
are surfaced through a single "User Study" page in the demo app
(``apps/demo/_pages/user_study.py``), whose "Setup" dropdown picks the
variant and "Group / page" dropdown picks the group/condition within it.

Structure:

  types.py            StudyGroupSpec / StudyStageSpec / StudyVariantSpec — the
                       shape a variant and its groups/pages/tasks take.
  registry.py          Discovers variant configs under
                       ``configs/viz/study_variants/*.yaml`` and pairs each
                       with its renderer.
  evidence.py          The two displays themselves — the rollout-video grid and
                       the behavior graph, with the copy above each and the
                       restatement of the task. Shared by every variant,
                       because a difference in that copy is a difference in the
                       stimulus.
  exit_questions.py    The closing page (prior experience + open remarks) and
                       the reader the submit page records it through.
  step_names.py        What each numbered step of a layout is called.
  variants/<name>/     One subpackage per study version/setup: its own
                       question bank(s) and step-machine renderer (or, for
                       ``production``, a thin ``runpy`` delegation to the
                       original script). Components shared with the original
                       study (video browser, graph explorer, response
                       storage, session loading, step-flow helpers) are
                       imported from ``behavior_graphs.viz.streamlit.user_study``
                       rather than duplicated.

What a variant owns is its *design*: how many pages, in what order, with which
questions, and what the response records. What it does not own is the evidence
or the closing questions — those come from the modules above, so two layouts
can be compared to each other rather than only to themselves.

To add a new variant: drop a yaml under configs/viz/study_variants/, add a
``variants/<name>/`` subpackage with a ``render(variant, group)`` function,
and wire it into ``registry._render_dispatch``.
"""

from __future__ import annotations
