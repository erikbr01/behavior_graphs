"""Shape of a study variant/setup and the groups (pages) within it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StudyGroupSpec:
    """One condition/page within a study variant (e.g. "Group A")."""

    id: str
    label: str
    # Variant-specific ordering of that variant's presentation phases, e.g.
    # ["videos", "graph"] for the order-effect variant. Opaque to the
    # registry — only the variant's own render() interprets it.
    order: list[str]


@dataclass(frozen=True)
class StudyStageSpec:
    """One task a variant walks a participant through, in a multi-task layout.

    A single-task variant names its evidence with the variant-level
    ``session``/``question_set``; a variant that shows several tasks in a row
    needs one of each per task, so those move in here. ``label`` is what the
    task is called on a page and in a response — the session yaml's own title
    ("Set Table Task: Contrastive Study") is about the study, not the task.
    """

    task: str
    label: str
    session_name: str
    question_set: str | None = None
    #: Per-task behavior-graph overrides on top of the variant's ``graph:``
    #: block. prepare_tray's graph is shallower than the other two and wants a
    #: ``min_branch`` of its own.
    graph_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StudyVariantSpec:
    """One study setup/version, made up of one or more groups."""

    id: str
    title: str
    description: str
    session_name: str
    groups: list[StudyGroupSpec]
    render: Callable[[StudyVariantSpec, StudyGroupSpec], None]
    # Fixed behavior-graph display settings (view_mode/min_branch/color_mode/...)
    # from the variant yaml's `graph:` block. Study participants get one
    # preset view with no controls to fiddle with; only opaque to the
    # registry -- each variant's render() picks which keys it honors.
    graph_params: dict = field(default_factory=dict)
    # Optional task-specific question bank. None keeps the legacy Likert bank.
    question_set: str | None = None
    # The tasks a multi-task variant walks through, in the order the yaml lists
    # them -- which is not the order any participant sees them in when the
    # variant draws that per session. Empty for the single-task variants, which
    # carry their one task in `session_name`/`question_set` above.
    stages: list[StudyStageSpec] = field(default_factory=list)

    def group(self, group_id: str) -> StudyGroupSpec | None:
        return next((g for g in self.groups if g.id == group_id), None)
