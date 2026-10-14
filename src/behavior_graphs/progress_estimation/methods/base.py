"""Base class for progress-estimation methods.

A method produces, for a held-out set of episodes, one chronological progress
curve per episode. Two shapes both fit this interface: graph methods fit a
behavior graph on the train split (``fit``) and label the eval split
(``predict``); VLM methods are training-free (``fit`` loads the model, ``predict``
scores each episode's video). ``modality`` tells the runner which data the eval
view must supply (policy embeddings vs mp4 frames).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from behavior_graphs.progress_estimation.io import METHOD_REGISTRY, EpisodeView


class ProgressMethod(ABC):
    name: str = ""
    modality: str = "embedding"  # "embedding" | "video"
    needs_backbone: bool = False

    @property
    def output_tag(self) -> str:
        """The ``*_curves.npz`` infix this method writes (registry tag, else name)."""
        info = METHOD_REGISTRY.get(self.name)
        return info.tag if info else self.name

    def fit(self, train: EpisodeView) -> None:  # noqa: D401 - default no-op
        """Fit on the train split (default: nothing, for training-free methods)."""

    def reset(self) -> None:  # noqa: D401 - default no-op
        """Clear any fitted state before a new fold (default: nothing)."""

    @abstractmethod
    def predict(self, eval: EpisodeView) -> list[np.ndarray]:
        """One chronological progress curve per eval episode, in ``eval.indices`` order."""
