"""Human annotations over recorded trajectories.

Two kinds, sharing one on-disk layout (:mod:`.store`):

:mod:`.subtasks`
    Subtask boundaries — the ground truth a behavior graph's segmentation is
    scored against.
:mod:`.failure_modes`
    Which way a failed rollout went wrong, in the study's own wordings — the
    ground truth the study's failure-mode rankings are scored against.
"""
