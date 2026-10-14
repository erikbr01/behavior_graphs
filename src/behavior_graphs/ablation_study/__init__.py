"""The ABC subtask-boundary ablation study, as a module rather than a script pile.

Entry point is ``scripts/ablation_study/study.py``; everything here is importable and
testable. A dataset is described by one file in ``configs/ablation_study/`` and nothing
downstream hardcodes a path.
"""

from behavior_graphs.ablation_study import config, splits

__all__ = ["config", "splits"]
