"""One pass over ``(labels, metadata)``, reused by every panel that needs it.

The video panels, the node/transition inspector and the per-frame timeline
strip each used to walk the full per-window metadata list on every Streamlit
rerun — and several of them once *per rendered clip*. On a real-robot run
that is ~20 k windows scanned a dozen times for a single click.

This module builds every per-episode structure the panels want in a single
pass and caches it, so the panels do dictionary lookups instead of scans.

The cache key is a cheap content signature (see :func:`signature`) rather
than the arrays themselves: Streamlit would otherwise hash ~1.5 MB of dicts
on every lookup.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import streamlit as st


@dataclass
class EpisodeIndex:
    """Per-episode views of a (labels, metadata) pair."""

    ep_key: str
    #: episode → [(window_start, window_end, label)] sorted by start
    windows: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    #: episode → outcome (None when unknown)
    success: dict[int, bool | None] = field(default_factory=dict)
    #: episode → last window_start + 1 (the legacy notion of "length")
    length: dict[int, int] = field(default_factory=dict)
    #: episode → run-length-collapsed [(window_start, label)], noise dropped
    rle: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    #: label → episode → [window_start, …]
    by_label: dict[int, dict[int, list[int]]] = field(default_factory=dict)

    # ── Queries ──────────────────────────────────────────────────────────

    def episodes_for_node(self, node_id: int) -> list[tuple[int, int, int]]:
        """``[(episode, first_window_start, last_window_start), …]``."""
        out = []
        for ep, tss in self.by_label.get(int(node_id), {}).items():
            out.append((ep, tss[0], tss[-1]))
        return sorted(out, key=lambda x: x[0])

    def node_stats(self, node_id: int) -> list[dict]:
        """Per-episode duration / count / position stats for one node."""
        out = []
        for ep, tss in self.by_label.get(int(node_id), {}).items():
            ts_min, ts_max = tss[0], tss[-1]
            length = self.length.get(ep, ts_max + 1) or 1
            out.append(
                {
                    "ep": ep,
                    "duration": ts_max - ts_min + 1,
                    "count": len(tss),
                    "ts_start": ts_min,
                    "ts_end": ts_max,
                    "position": (ts_min + ts_max) / 2 / length,
                    "success": self.success.get(ep),
                }
            )
        return out

    def edge_stats(self, src_id: int, tgt_id: int) -> list[dict]:
        """Per-episode stats for the first ``src → tgt`` transition."""
        out = []
        for ep, rle in self.rle.items():
            length = self.length.get(ep, 1) or 1
            for j in range(len(rle) - 1):
                if rle[j][1] == src_id and rle[j + 1][1] == tgt_id:
                    ts_trans = rle[j + 1][0]
                    out.append(
                        {
                            "ep": ep,
                            "ts_src": rle[j][0],
                            "ts_tgt": ts_trans,
                            "position": ts_trans / length,
                            "success": self.success.get(ep),
                        }
                    )
                    break
        return out

    def per_frame_labels(self, ep_idx: int, total_frames: int) -> np.ndarray:
        """Per-frame cluster label array for one episode (-1 = unlabeled).

        Overlapping windows resolve last-write-wins on sorted window_start,
        matching the previous behaviour.
        """
        if total_frames <= 0:
            return np.array([], dtype=np.int64)
        out = np.full(total_frames, -1, dtype=np.int64)
        for ws, we, lbl in self.windows.get(ep_idx, ()):
            if lbl < 0:
                continue
            out[max(0, ws) : min(total_frames, we)] = lbl
        return out


def _build(labels: np.ndarray, metadata: Sequence[dict], ep_key: str) -> EpisodeIndex:
    idx = EpisodeIndex(ep_key=ep_key)
    windows: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    by_label: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    labels_list = labels.tolist()
    for i, m in enumerate(metadata):
        ep = m.get(ep_key, -1)
        ws = int(m.get("window_start", m.get("timestep", 0)))
        we = int(m.get("window_end", ws + 1))
        lbl = int(labels_list[i])
        windows[ep].append((ws, we, lbl))
        if ep not in idx.success:
            idx.success[ep] = m.get("success")
        prev = idx.length.get(ep, 0)
        if ws + 1 > prev:
            idx.length[ep] = ws + 1

    for ep, wins in windows.items():
        wins.sort()
        idx.windows[ep] = wins
        rle: list[tuple[int, int]] = []
        for ws, _we, lbl in wins:
            if lbl < 0:
                continue
            by_label[lbl][ep].append(ws)
            if not rle or rle[-1][1] != lbl:
                rle.append((ws, lbl))
        idx.rle[ep] = rle

    idx.by_label = {lbl: dict(eps) for lbl, eps in by_label.items()}
    return idx


def signature(labels: np.ndarray, metadata: Sequence[dict]) -> str:
    """Cheap content signature for a (labels, metadata) pair.

    Hashing the label array is ~0.05 ms; the metadata list is summarised by
    its length plus three sampled entries, which is enough to separate runs
    without serialising megabytes of dicts on every rerun.
    """
    h = hashlib.blake2b(digest_size=16)
    arr = np.ascontiguousarray(labels)
    h.update(arr.tobytes())
    h.update(str(arr.dtype).encode())
    n = len(metadata)
    h.update(str(n).encode())
    for i in (0, n // 2, n - 1):
        if 0 <= i < n:
            h.update(repr(sorted(metadata[i].items())).encode())
    return h.hexdigest()


# cache_resource, not cache_data: the index is read-only and holds tens of
# thousands of tuples, so we want the object handed back by reference rather
# than unpickled on every lookup.
@st.cache_resource(show_spinner=False, max_entries=32)
def _cached(sig: str, ep_key: str, _labels: np.ndarray, _metadata: Sequence[dict]) -> EpisodeIndex:
    return _build(_labels, _metadata, ep_key)


def get_episode_index(labels: np.ndarray, metadata: Sequence[dict]) -> EpisodeIndex:
    """Cached :class:`EpisodeIndex` for a (labels, metadata) pair."""
    ep_key = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"
    return _cached(signature(labels, metadata), ep_key, labels, metadata)
