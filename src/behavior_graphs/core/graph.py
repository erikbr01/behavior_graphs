"""Behavior graph: an empirical transition graph over behavioral clusters.

Ported from policy_doctor's ``behavior_graph`` package and adapted to the
:class:`~behavior_graphs.core.types.WindowSet` types. The graph is the central
artifact of the pipeline and serializes to/from plain JSON.
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from behavior_graphs.core.types import WindowSet

START_NODE_ID = -2
END_NODE_ID = -3
SUCCESS_NODE_ID = -4
FAILURE_NODE_ID = -5

TERMINAL_NODE_IDS = frozenset({END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})
SPECIAL_NODE_IDS = frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS

_SPECIAL_NODE_NAMES = {
    START_NODE_ID: "START",
    END_NODE_ID: "END",
    SUCCESS_NODE_ID: "SUCCESS",
    FAILURE_NODE_ID: "FAILURE",
}


def node_display_name(node_id: int) -> str:
    """Human-readable name for a node id: START/END/SUCCESS/FAILURE or ``Behavior N``."""
    return _SPECIAL_NODE_NAMES.get(node_id, f"Behavior {node_id}")


Path_ = tuple[list[int], float, list[tuple[int, int, float]]]


def sorted_episode_window_indices(labels: np.ndarray, windows: WindowSet) -> dict[int, list[int]]:
    """Per-episode window indices (into ``labels``/``windows.windows``), sorted
    by window start, with noise-labeled (``-1``) windows dropped.

    The shared traversal order behind :func:`collapse_episode_sequences` and
    anything else that needs the same "which windows, in what order" view of
    an episode -- e.g. ``outcome_classifier.classify``'s collapsed-run
    construction, which additionally carries a feature vector and window
    count per run, not just the label sequence this function's other caller
    needs.
    """
    labels = np.asarray(labels)
    per_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, w in enumerate(windows.windows):
        if int(labels[i]) == -1:
            continue
        per_episode[w.episode_idx].append((w.start, i))
    return {
        ep_idx: [i for _, i in sorted(entries, key=lambda x: x[0])]
        for ep_idx, entries in per_episode.items()
    }


def collapse_episode_sequences(labels: np.ndarray, windows: WindowSet) -> dict[int, list[int]]:
    """Per-episode sequence of cluster visits, collapsing consecutive repeats.

    Noise windows (label ``-1``) are dropped before collapsing. Shared by
    graph construction and by anything that needs the same "sequence of
    behavioral clusters" a rollout visited (e.g. held-out outcome
    prediction) without going through the full graph build.
    """
    labels = np.asarray(labels)
    collapsed: dict[int, list[int]] = {}
    for ep_idx, indices in sorted_episode_window_indices(labels, windows).items():
        seq = [int(labels[i]) for i in indices]
        run = [seq[0]]
        for label in seq[1:]:
            if label != run[-1]:
                run.append(label)
        collapsed[ep_idx] = run
    return collapsed


@dataclass
class BehaviorNode:
    cluster_id: int
    name: str
    num_windows: int
    num_episodes: int
    episode_indices: list[int] = field(default_factory=list)

    @property
    def is_start(self) -> bool:
        return self.cluster_id == START_NODE_ID

    @property
    def is_end(self) -> bool:
        return self.cluster_id == END_NODE_ID

    @property
    def is_success(self) -> bool:
        return self.cluster_id == SUCCESS_NODE_ID

    @property
    def is_failure(self) -> bool:
        return self.cluster_id == FAILURE_NODE_ID

    @property
    def is_terminal(self) -> bool:
        return self.cluster_id in TERMINAL_NODE_IDS

    @property
    def is_special(self) -> bool:
        return self.cluster_id in SPECIAL_NODE_IDS

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "num_windows": self.num_windows,
            "num_episodes": self.num_episodes,
            "episode_indices": self.episode_indices,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BehaviorNode:
        return cls(
            cluster_id=int(d["cluster_id"]),
            name=d["name"],
            num_windows=int(d["num_windows"]),
            num_episodes=int(d["num_episodes"]),
            episode_indices=[int(i) for i in d.get("episode_indices", [])],
        )


@dataclass
class BehaviorGraph:
    """Directed transition graph over behavioral clusters plus START/SUCCESS/FAILURE/END.

    Edges carry empirical transition counts/frequencies. The graph is NOT
    guaranteed to be Markovian; value
    computations treat it as a Markov reward process as an approximation.
    """

    nodes: dict[int, BehaviorNode]
    transition_counts: dict[int, dict[int, int]]
    transition_probs: dict[int, dict[int, float]]
    num_episodes: int

    # ------------------------------------------------------------------ build
    @classmethod
    def from_window_labels(
        cls,
        labels: np.ndarray,
        windows: WindowSet,
        cluster_names: dict[int, str] | None = None,
    ) -> BehaviorGraph:
        """Build the graph from per-window cluster labels.

        Noise windows (label ``-1``) are skipped. Consecutive equal labels
        within an episode collapse into one visit. Episodes transition from
        START into their first cluster and from their last cluster into
        SUCCESS/FAILURE (if the episode outcome is known) or END.
        """
        labels = np.asarray(labels)
        outcomes = {ep.index: ep.success for ep in windows.episodes}
        has_outcomes = any(v is not None for v in outcomes.values())

        collapsed = collapse_episode_sequences(labels, windows)

        counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for ep_idx, seq in collapsed.items():
            counts[START_NODE_ID][seq[0]] += 1
            for j in range(len(seq) - 1):
                counts[seq[j]][seq[j + 1]] += 1
            outcome = outcomes.get(ep_idx) if has_outcomes else None
            if outcome is True:
                counts[seq[-1]][SUCCESS_NODE_ID] += 1
            elif outcome is False:
                counts[seq[-1]][FAILURE_NODE_ID] += 1
            else:
                counts[seq[-1]][END_NODE_ID] += 1

        probs: dict[int, dict[int, float]] = {}
        for src, targets in counts.items():
            total = sum(targets.values())
            probs[src] = {tgt: c / total for tgt, c in targets.items()}

        window_counts: dict[int, int] = defaultdict(int)
        episode_sets: dict[int, set[int]] = defaultdict(set)
        for i, w in enumerate(windows.windows):
            label = int(labels[i])
            if label == -1:
                continue
            window_counts[label] += 1
            episode_sets[label].add(w.episode_idx)

        episode_ids = sorted(collapsed.keys())
        nodes: dict[int, BehaviorNode] = {
            START_NODE_ID: BehaviorNode(START_NODE_ID, "START", 0, len(episode_ids), episode_ids)
        }
        groups = {
            SUCCESS_NODE_ID: ("SUCCESS", [e for e in episode_ids if outcomes.get(e) is True]),
            FAILURE_NODE_ID: ("FAILURE", [e for e in episode_ids if outcomes.get(e) is False]),
            END_NODE_ID: ("END", [e for e in episode_ids if outcomes.get(e) is None]),
        }
        if not has_outcomes:
            groups = {END_NODE_ID: ("END", episode_ids)}
        for nid, (name, eps) in groups.items():
            if eps:
                nodes[nid] = BehaviorNode(nid, name, 0, len(eps), eps)

        for c_id in sorted({int(c) for c in labels if c != -1}):
            name = (cluster_names or {}).get(c_id, f"Behavior {c_id}")
            nodes[c_id] = BehaviorNode(
                cluster_id=c_id,
                name=name,
                num_windows=window_counts[c_id],
                num_episodes=len(episode_sets[c_id]),
                episode_indices=sorted(episode_sets[c_id]),
            )
        return cls(
            nodes=nodes,
            transition_counts={k: dict(v) for k, v in counts.items()},
            transition_probs=probs,
            num_episodes=len(episode_ids),
        )

    # ------------------------------------------------------------- accessors
    @property
    def terminal_node_ids(self) -> set[int]:
        return {nid for nid in self.nodes if nid in TERMINAL_NODE_IDS}

    @property
    def cluster_nodes(self) -> dict[int, BehaviorNode]:
        return {k: v for k, v in self.nodes.items() if not v.is_special}

    def get_outgoing_transitions(self, node_id: int) -> list[tuple[int, int, float]]:
        """(target_id, count, probability) for edges leaving ``node_id``."""
        targets = self.transition_probs.get(node_id, {})
        counts = self.transition_counts.get(node_id, {})
        return [(tgt, counts.get(tgt, 0), p) for tgt, p in targets.items()]

    def get_incoming_transitions(self, node_id: int) -> list[tuple[int, int, float]]:
        """(source_id, count, probability) for edges entering ``node_id``."""
        result = []
        for src, targets in self.transition_probs.items():
            if node_id in targets:
                cnt = self.transition_counts.get(src, {}).get(node_id, 0)
                result.append((src, cnt, targets[node_id]))
        return result

    # ---------------------------------------------------------------- values
    def compute_values(
        self,
        gamma: float = 0.99,
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
        reward_end: float = 0.0,
    ) -> dict[int, float]:
        """Per-node values, treating the graph as a Markov reward process.

        An approximation: the underlying behavior sequences need not be
        Markovian.
        """
        terminal_rewards = {
            SUCCESS_NODE_ID: reward_success,
            FAILURE_NODE_ID: reward_failure,
            END_NODE_ID: reward_end,
        }
        terminal_ids = self.terminal_node_ids
        nonterminal = sorted(nid for nid in self.nodes if nid not in terminal_ids)
        index = {nid: i for i, nid in enumerate(nonterminal)}
        n = len(nonterminal)
        if n == 0:
            return {nid: terminal_rewards.get(nid, 0.0) for nid in self.nodes}
        P = np.zeros((n, n))
        b = np.zeros(n)
        for nid in nonterminal:
            i = index[nid]
            for tgt, p in self.transition_probs.get(nid, {}).items():
                if tgt in index:
                    P[i, index[tgt]] = p
                elif tgt in terminal_ids:
                    b[i] += p * terminal_rewards.get(tgt, 0.0)
        v = np.linalg.solve(np.eye(n) - gamma * P, gamma * b)
        values = {nid: float(v[index[nid]]) for nid in nonterminal}
        for nid, reward in terminal_rewards.items():
            values[nid] = reward
        return values

    def compute_window_values(
        self,
        labels: np.ndarray,
        windows: WindowSet,
        node_values: dict[int, float],
        gamma: float = 0.99,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-window Q-values, advantages, and next-cluster ids.

        Q of a window in a cluster segment is ``gamma * V(next segment's
        cluster)``; the advantage is ``Q - V(current cluster)``.
        """
        labels = np.asarray(labels)
        n = len(labels)
        outcomes = {ep.index: ep.success for ep in windows.episodes}
        has_outcomes = any(v is not None for v in outcomes.values())

        per_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for i, w in enumerate(windows.windows):
            per_episode[w.episode_idx].append((w.start, i, int(labels[i])))

        q_values = np.zeros(n)
        advantages = np.zeros(n)
        next_cluster = np.full(n, -1, dtype=int)
        for ep_idx, items in per_episode.items():
            items.sort(key=lambda x: x[0])
            segments: list[tuple[int, list[int]]] = []
            for _, idx, label in items:
                if label == -1:
                    continue
                if segments and segments[-1][0] == label:
                    segments[-1][1].append(idx)
                else:
                    segments.append((label, [idx]))
            if not segments:
                continue
            outcome = outcomes.get(ep_idx) if has_outcomes else None
            terminal = (
                SUCCESS_NODE_ID
                if outcome is True
                else (FAILURE_NODE_ID if outcome is False else END_NODE_ID)
            )
            for s, (seg_label, seg_indices) in enumerate(segments):
                nxt = segments[s + 1][0] if s < len(segments) - 1 else terminal
                q = gamma * node_values.get(nxt, 0.0)
                adv = q - node_values.get(seg_label, 0.0)
                for idx in seg_indices:
                    q_values[idx] = q
                    advantages[idx] = adv
                    next_cluster[idx] = nxt
        return q_values, advantages, next_cluster

    # ----------------------------------------------------------------- paths
    def _loops_for_path(self, path: list[int]) -> list[tuple[int, int, float]]:
        """Edges from nodes in *path* back to an earlier node in the same path."""
        path_set = set(path)
        loops = []
        for i, node in enumerate(path):
            for tgt, edge_prob in self.transition_probs.get(node, {}).items():
                if tgt in path_set and tgt not in TERMINAL_NODE_IDS:
                    if path.index(tgt) < i:
                        loops.append((node, tgt, edge_prob))
        return loops

    def _search_paths(
        self,
        max_paths: int,
        min_probability: float,
        min_edge_probability: float,
        accept: Callable[[list[int]], bool] | None = None,
    ) -> list[tuple[list[int], float]]:
        """Best-first search core shared by :meth:`enumerate_paths` and
        :meth:`enumerate_paths_to_terminal`.

        If *accept* is given, only a completed path for which
        ``accept(path)`` is true counts toward *max_paths* — the search
        keeps expanding past non-matching completions instead of stopping
        early. This matters when one terminal dominates in probability
        (e.g. a 97% success rate): stopping as soon as *max_paths* paths to
        *any* terminal are found would return only success paths, and a
        caller filtering for FAILURE afterward would wrongly see none, even
        though failure paths exist deeper in the search.
        """
        terminals = self.terminal_node_ids
        results: list[tuple[list[int], float]] = []
        counter = 0
        heap: list[tuple[float, int, list[int], set[int]]] = [
            (-1.0, 0, [START_NODE_ID], {START_NODE_ID})
        ]
        # A rare/unreachable terminal must not search forever; the state
        # space is bounded (simple paths over `len(self.nodes)` nodes), so
        # a generous cap only ever triggers when there's truly nothing left.
        max_expansions = max(20_000, max_paths * 4_000)
        expansions = 0
        while heap and len(results) < max_paths and expansions < max_expansions:
            expansions += 1
            neg_prob, _, path, visited = heapq.heappop(heap)
            prob = -neg_prob
            current = path[-1]
            if current in terminals:
                if prob >= min_probability and (accept is None or accept(path)):
                    results.append((list(path), prob))
                continue
            for tgt, edge_prob in self.transition_probs.get(current, {}).items():
                if edge_prob < min_edge_probability:
                    continue
                if tgt in visited and tgt not in terminals:
                    continue
                counter += 1
                heapq.heappush(heap, (-prob * edge_prob, counter, path + [tgt], visited | {tgt}))
        return results

    def enumerate_paths(
        self,
        max_paths: int = 50,
        min_probability: float = 0.0,
        min_edge_probability: float = 0.0,
    ) -> list[Path_]:
        """Most-probable simple START→terminal paths (best-first search).

        Each result is ``(node_ids, path_probability, loop_edges)`` where
        ``loop_edges`` are edges from path nodes back to earlier path nodes.
        """
        results = self._search_paths(max_paths, min_probability, min_edge_probability)
        return [(path, prob, self._loops_for_path(path)) for path, prob in results]

    def enumerate_paths_to_terminal(
        self,
        terminal_id: int,
        max_paths: int = 50,
        min_probability: float = 0.0,
        min_edge_probability: float = 0.0,
    ) -> list[Path_]:
        """Most-probable simple START→*terminal_id* paths.

        Searches until *max_paths* paths ending specifically at
        *terminal_id* are found, not just *max_paths* paths to any
        terminal — see :meth:`_search_paths` for why that distinction
        matters when terminals have very different probabilities.
        """
        results = self._search_paths(
            max_paths,
            min_probability,
            min_edge_probability,
            accept=lambda path: path[-1] == terminal_id,
        )
        return [(path, prob, self._loops_for_path(path)) for path, prob in results]

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "transition_counts": {
                str(k): {str(t): c for t, c in v.items()} for k, v in self.transition_counts.items()
            },
            "transition_probs": {
                str(k): {str(t): p for t, p in v.items()} for k, v in self.transition_probs.items()
            },
            "num_episodes": self.num_episodes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BehaviorGraph:
        return cls(
            nodes={int(n["cluster_id"]): BehaviorNode.from_dict(n) for n in d["nodes"]},
            transition_counts={
                int(k): {int(t): int(c) for t, c in v.items()}
                for k, v in d["transition_counts"].items()
            },
            transition_probs={
                int(k): {int(t): float(p) for t, p in v.items()}
                for k, v in d["transition_probs"].items()
            },
            num_episodes=int(d["num_episodes"]),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> BehaviorGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))
