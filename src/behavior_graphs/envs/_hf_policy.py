"""A :class:`~behavior_graphs.core.env.Policy` that is just a checkpoint reference.

Adapters for policies served out of process -- the model runs behind an HTTP
server in its own environment, and this repo never loads its weights -- have no
in-process policy object to hand back. What they do have is the checkpoint the
server was started from, which is the only thing downstream needs: it goes into
the rollout directory's metadata so a run records which weights produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from behavior_graphs.core.env import Policy


class HFCheckpointPolicy(Policy):
    """A HuggingFace checkpoint reference (hub repo id or local path)."""

    def __init__(self, checkpoint: str, metadata: dict[str, Any] | None = None) -> None:
        self._checkpoint = str(checkpoint)
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("checkpoint", self._checkpoint)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self._checkpoint)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
