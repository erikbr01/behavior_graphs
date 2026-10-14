"""Core data types shared across the behavior-graph pipeline.

All pipeline stages communicate through these types. They are deliberately
simple (numpy arrays + dataclasses) so that every stage output can be
serialized into a run's artifact directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeInfo:
    """Metadata for one episode within a flat (n_samples, ...) array.

    Samples of episode ``index`` occupy rows ``[sample_start_idx, sample_end_idx)``
    of the associated feature array.
    """

    index: int
    num_samples: int
    sample_start_idx: int
    sample_end_idx: int
    success: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "num_samples": self.num_samples,
            "sample_start_idx": self.sample_start_idx,
            "sample_end_idx": self.sample_end_idx,
            "success": self.success,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeInfo:
        return cls(
            index=int(d["index"]),
            num_samples=int(d["num_samples"]),
            sample_start_idx=int(d["sample_start_idx"]),
            sample_end_idx=int(d["sample_end_idx"]),
            success=d.get("success"),
            extra=d.get("extra", {}),
        )


def episodes_from_lengths(
    lengths: list[int], successes: list[bool | None] | None = None
) -> list[EpisodeInfo]:
    """Build contiguous EpisodeInfo entries from per-episode lengths."""
    episodes = []
    start = 0
    for i, n in enumerate(lengths):
        success = successes[i] if successes is not None else None
        episodes.append(
            EpisodeInfo(
                index=i,
                num_samples=n,
                sample_start_idx=start,
                sample_end_idx=start + n,
                success=success,
            )
        )
        start += n
    return episodes


@dataclass
class FeatureSet:
    """Per-timestep features for a set of episodes.

    ``features[i]`` belongs to the episode whose ``[sample_start_idx,
    sample_end_idx)`` range contains ``i``; the timestep within the episode is
    ``i - sample_start_idx``.
    """

    features: np.ndarray  # (n_samples, d) float32
    episodes: list[EpisodeInfo]

    def __post_init__(self) -> None:
        n = sum(ep.num_samples for ep in self.episodes)
        if n != self.features.shape[0]:
            raise ValueError(
                f"episode lengths sum to {n} but features have {self.features.shape[0]} rows"
            )

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])


@dataclass
class WindowInfo:
    """One window of consecutive timesteps within a single episode."""

    episode_idx: int
    start: int  # timestep within episode, inclusive
    end: int  # timestep within episode, exclusive

    def to_dict(self) -> dict[str, Any]:
        return {"episode_idx": self.episode_idx, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowInfo:
        return cls(episode_idx=int(d["episode_idx"]), start=int(d["start"]), end=int(d["end"]))


@dataclass
class WindowSet:
    """Aggregated features per window. Windows never cross episode boundaries."""

    features: np.ndarray  # (n_windows, d)
    windows: list[WindowInfo]
    episodes: list[EpisodeInfo]

    def __post_init__(self) -> None:
        if len(self.windows) != self.features.shape[0]:
            raise ValueError(
                f"{len(self.windows)} windows but features have {self.features.shape[0]} rows"
            )

    @property
    def num_windows(self) -> int:
        return int(self.features.shape[0])

    def episode_success(self, episode_idx: int) -> bool | None:
        return self.episodes[episode_idx].success


def consecutive_window_pairs(windows: WindowSet) -> np.ndarray:
    """``(n_pairs, 2)`` index pairs of windows adjacent in time within an episode.

    Ordered by (episode, start), so a pair is two windows a reducer or smoother
    should treat as neighbours. Pairs never cross an episode boundary.
    """
    episode = np.array([w.episode_idx for w in windows.windows])
    start = np.array([w.start for w in windows.windows])
    order = np.lexsort((start, episode))
    same = episode[order[:-1]] == episode[order[1:]]
    return np.stack([order[:-1][same], order[1:][same]], axis=1)


def node_progress_range(
    node_id: int,
    labels: np.ndarray,
    windows: list[WindowInfo],
    episodes: list[EpisodeInfo],
) -> tuple[float, float] | None:
    """Episode-relative position range [min, max] of a node's windows.

    Uses window start / episode length as a proxy for task progress (WARP-RM
    insight: time ≠ progress, but relative position is a useful contextual
    hint, e.g. for VLM labeling).  Returns None when the node has no windows
    in episodes of known length.
    """
    ep_lengths = {ep.index: ep.num_samples for ep in episodes}
    positions = [
        w.start / ep_lengths[w.episode_idx]
        for w, lbl in zip(windows, labels)
        if int(lbl) == node_id and ep_lengths.get(w.episode_idx, 0) > 0
    ]
    if not positions:
        return None
    return float(min(positions)), float(max(positions))
