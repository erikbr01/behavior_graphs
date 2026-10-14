"""Graph simplification methods for the viz stack: smoothing, pruning, layout.

Rollout level is assumed; ``metadata`` is the legacy per-window dict list
(see ``behavior_graphs.viz.compat``).

Each function is a pure transformation. Two flavors:
  - **Label transformations** take ``(cluster_labels, metadata, ...)`` and return new labels.
  - **Graph transformations** take a ``BehaviorGraph`` (plus labels for re-derivation) and
    return a new graph (and possibly new labels).

See ``docs/graph_simplification_brainstorm.md`` for the method catalogue and rationale.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    SPECIAL_NODE_IDS,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
)
from behavior_graphs.core.trajectory_paths import collapse_episode_paths
from behavior_graphs.core.types import WindowSet
from behavior_graphs.stages.graph.simplify import apply_merge_map
from behavior_graphs.stages.graph.smoothing import in_clustering_space  # noqa: F401  (re-exported)
from behavior_graphs.viz.compat import graph_from_labels_meta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode_groups(
    labels: np.ndarray, metadata: list[dict], level: str = "rollout"
) -> dict[int, list[int]]:
    """Return {episode_idx: [sample_indices_in_time_order]}."""
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, i))
    out: dict[int, list[int]] = {}
    for ep, vals in eps.items():
        vals.sort()
        out[ep] = [i for _, i in vals]
    return out


def _cluster_centroids(labels: np.ndarray, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """Mean embedding per cluster id (ignores noise -1)."""
    out: dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        out[int(c)] = embeddings[labels == c].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Smoothing (label-level)
# ---------------------------------------------------------------------------


def hmm_smooth(
    embeddings: np.ndarray,
    metadata: list[dict],
    n_states: int,
    level: str = "rollout",
    covariance_type: str = "diag",
    n_iter: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Gaussian-HMM smoothing: fit on all windows, Viterbi-decode per-episode.

    Returns new labels (0..n_states-1). The HMM jointly estimates state means
    and a transition matrix; the diagonal of that matrix is the learned
    self-persistence bias.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn required: pip install hmmlearn") from e

    eps = _episode_groups(np.zeros(len(metadata), dtype=np.int32), metadata, level)
    # Build episode-grouped sequences for fit (concatenate, with lengths)
    seqs = []
    lengths = []
    for ep, idxs in eps.items():
        seqs.append(embeddings[idxs])
        lengths.append(len(idxs))
    X = np.concatenate(seqs, axis=0)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X, lengths=lengths)

    out = np.full(len(metadata), -1, dtype=np.int64)
    offset = 0
    for ep, idxs in eps.items():
        T = len(idxs)
        seq_X = X[offset : offset + T]
        offset += T
        states = model.predict(seq_X)
        for i, t in enumerate(idxs):
            out[t] = int(states[i])
    return out


# ---------------------------------------------------------------------------
# Graph-level transformations
# ---------------------------------------------------------------------------


def prune_edges_by_count(graph: BehaviorGraph, min_count: int = 2) -> BehaviorGraph:
    """Drop edges with count < min_count; renormalize probabilities."""
    new_counts: dict[int, dict[int, int]] = {}
    for src, targets in graph.transition_counts.items():
        kept = {t: c for t, c in targets.items() if c >= min_count}
        if kept:
            new_counts[src] = kept
    new_probs: dict[int, dict[int, float]] = {}
    for src, targets in new_counts.items():
        total = sum(targets.values())
        new_probs[src] = {t: c / total for t, c in targets.items()}
    return BehaviorGraph(
        nodes=dict(graph.nodes),
        transition_counts=new_counts,
        transition_probs=new_probs,
        num_episodes=graph.num_episodes,
    )


def prune_edges_by_prob(graph: BehaviorGraph, min_prob: float = 0.1) -> BehaviorGraph:
    """Drop edges with probability < min_prob; renormalize remaining."""
    new_counts: dict[int, dict[int, int]] = {}
    for src, targets in graph.transition_counts.items():
        total = sum(targets.values())
        if total == 0:
            continue
        kept = {t: c for t, c in targets.items() if c / total >= min_prob}
        if kept:
            new_counts[src] = kept
    new_probs: dict[int, dict[int, float]] = {}
    for src, targets in new_counts.items():
        total = sum(targets.values())
        new_probs[src] = {t: c / total for t, c in targets.items()}
    return BehaviorGraph(
        nodes=dict(graph.nodes),
        transition_counts=new_counts,
        transition_probs=new_probs,
        num_episodes=graph.num_episodes,
    )


def merge_similar_centroids(
    labels: np.ndarray,
    embeddings: np.ndarray,
    metadata: list[dict],
    sim_threshold: float = 0.9,
    level: str = "rollout",
) -> np.ndarray:
    """Merge clusters whose centroids are cosine-similar above ``sim_threshold``.

    Iterates: at each step find the most-similar pair, merge (keep smaller id),
    until no pair exceeds threshold. Returns relabeled array.
    """
    cur = labels.copy()
    while True:
        centroids = _cluster_centroids(cur, embeddings)
        if len(centroids) < 2:
            return cur
        ids = sorted(centroids.keys())
        C = np.stack([centroids[c] for c in ids])
        norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        Cn = C / norms
        sims = Cn @ Cn.T
        np.fill_diagonal(sims, -np.inf)
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        max_sim = sims[i, j]
        if max_sim < sim_threshold:
            return cur
        # Merge j into i (smaller id wins)
        a, b = sorted([ids[i], ids[j]])
        cur = np.where(cur == b, a, cur)


def stable_phase_prune(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
) -> tuple[BehaviorGraph, np.ndarray]:
    """ENAP-style stable-phase pruning.

    Merge node ``q'`` into ``q`` when:
      1. ``q`` has a self-loop (sees itself in the cluster sequence — actually,
         in run-length-collapsed graphs there are no self-loops; we use a
         stricter definition: ``q'`` is reached *only* from ``q`` and has very
         few outgoing destinations).
      2. The (q → q') transition is much more frequent than (q' → anywhere
         else), i.e., q' is "downstream" of q.

    Concretely: merge q' into q when (count(q→q') / sum(counts out of q'))
    > 0.5 AND q' has a single dominant incoming source q.
    """
    counts = {s: dict(t) for s, t in graph.transition_counts.items()}
    # Build incoming view
    incoming: dict[int, dict[int, int]] = defaultdict(dict)
    for s, targets in counts.items():
        for t, c in targets.items():
            incoming[t][s] = c

    merge_map: dict[int, int] = {}
    for q_prime in list(graph.nodes.keys()):
        if q_prime in SPECIAL_NODE_IDS or q_prime in TERMINAL_NODE_IDS:
            continue
        inc = incoming.get(q_prime, {})
        if not inc:
            continue
        # Identify dominant incoming source
        q, qc = max(inc.items(), key=lambda kv: kv[1])
        if q in SPECIAL_NODE_IDS or q in TERMINAL_NODE_IDS:
            continue
        # Mass on dominant incoming edge
        if qc / max(1, sum(inc.values())) < 0.6:
            continue
        # Outgoing from q': mostly back into the system or terminal?
        out_qprime = counts.get(q_prime, {})
        total_out = sum(out_qprime.values())
        if total_out == 0:
            merge_map[q_prime] = q
            continue
        # If most of q's outgoing mass to q' is much larger than q's other
        # outgoing flows, consider q' a "stable continuation" of q
        out_q = counts.get(q, {})
        share_to_qprime = qc / max(1, sum(out_q.values()))
        if share_to_qprime > 0.3:
            merge_map[q_prime] = q

    if not merge_map:
        return graph, labels

    new_labels = apply_merge_map(labels, merge_map)
    new_graph = graph_from_labels_meta(new_labels, metadata)
    return new_graph, new_labels


# ---------------------------------------------------------------------------
# Alternative clusterings
# ---------------------------------------------------------------------------


def auto_k_kmeans(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = (4, 15),
    random_state: int = 42,
) -> tuple[np.ndarray, int, dict[int, float]]:
    """Sweep K, return labels for the K with the best silhouette score.

    Returns ``(labels, best_k, scores_by_k)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    best_k = k_range[0]
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    # Subsample for silhouette (it's O(n^2))
    n = embeddings.shape[0]
    rng = np.random.default_rng(random_state)
    sub = rng.choice(n, size=min(3000, n), replace=False)
    X_sub = embeddings[sub]
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(embeddings).astype(np.int64)
        if k == 1:
            scores[k] = 0.0
            continue
        sub_labels = labels[sub]
        if len(np.unique(sub_labels)) < 2:
            scores[k] = 0.0
            continue
        sc = float(silhouette_score(X_sub, sub_labels))
        scores[k] = sc
        if sc > best_score:
            best_score = sc
            best_k = k
            best_labels = labels
    if best_labels is None:
        # fallback
        km = KMeans(n_clusters=k_range[0], n_init=10, random_state=random_state)
        best_labels = km.fit_predict(embeddings).astype(np.int64)
    return best_labels, best_k, scores


def spectral_transition_clustering(
    labels: np.ndarray,
    metadata: list[dict],
    n_macro: int = 8,
    level: str = "rollout",
    random_state: int = 42,
) -> np.ndarray:
    """Build the micro-transition matrix, spectral-cluster it into macro-clusters.

    Idea: even if individual clusters are noisy, the *transition affinity* between
    them carries the true behavioral grouping — clusters that flow into each
    other should be one macro-mode.
    """
    from sklearn.cluster import SpectralClustering

    # Build symmetric affinity = transition counts + transpose
    micro_ids = sorted(set(int(c) for c in labels if c != -1))
    id_to_idx = {c: i for i, c in enumerate(micro_ids)}
    K = len(micro_ids)
    if K <= n_macro:
        # Nothing to do; identity mapping
        return labels.copy()

    M = np.zeros((K, K), dtype=np.float64)
    eps = _episode_groups(labels, metadata, level)
    for ep, idxs in eps.items():
        seq = labels[idxs]
        # collapse run-length
        if len(seq) == 0:
            continue
        collapsed = [int(seq[0])]
        for v in seq[1:]:
            if v == -1:
                continue
            if int(v) != collapsed[-1]:
                collapsed.append(int(v))
        for a, b in zip(collapsed[:-1], collapsed[1:]):
            if a in id_to_idx and b in id_to_idx:
                M[id_to_idx[a], id_to_idx[b]] += 1.0
                M[id_to_idx[b], id_to_idx[a]] += 1.0  # symmetric for spectral
    # Add small self affinity so isolated clusters don't break Laplacian
    np.fill_diagonal(M, M.diagonal() + 1.0)
    # Spectral clustering on the affinity
    sc = SpectralClustering(
        n_clusters=n_macro,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    macro = sc.fit_predict(M)
    micro_to_macro = {micro_ids[i]: int(macro[i]) for i in range(K)}

    out = labels.copy()
    for i, c in enumerate(labels):
        if int(c) in micro_to_macro:
            out[i] = micro_to_macro[int(c)]
    return out


def change_point_segmentation(
    embeddings: np.ndarray,
    metadata: list[dict],
    n_macro: int,
    penalty: float = 10.0,
    level: str = "rollout",
    random_state: int = 42,
) -> np.ndarray:
    """Detect change-points per episode, then cluster the resulting segments.

    For each episode: run PELT (ruptures library) to find change-points; each
    inter-changepoint segment gets a single label assigned by KMeans on the
    segment's mean embedding.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("ruptures required: pip install ruptures") from e
    from sklearn.cluster import KMeans

    eps = _episode_groups(np.zeros(len(metadata), dtype=np.int32), metadata, level)
    # Step 1: detect change-points, collect segment means
    segments: list[tuple[list[int], np.ndarray]] = []  # (indices_in_segment, mean_emb)
    for ep, idxs in eps.items():
        X = embeddings[idxs]
        if len(X) < 4:
            segments.append((idxs, X.mean(axis=0)))
            continue
        algo = rpt.Pelt(model="l2").fit(X)
        bkps = algo.predict(pen=penalty)
        bkps = [0] + bkps  # include start
        for i in range(len(bkps) - 1):
            lo, hi = bkps[i], bkps[i + 1]
            seg_idx = idxs[lo:hi]
            if not seg_idx:
                continue
            segments.append((seg_idx, X[lo:hi].mean(axis=0)))
    # Step 2: KMeans on segment means
    seg_means = np.stack([s[1] for s in segments])
    km = KMeans(n_clusters=n_macro, n_init=10, random_state=random_state)
    seg_labels = km.fit_predict(seg_means).astype(np.int64)

    out = np.full(len(metadata), -1, dtype=np.int64)
    for (seg_idx, _), lab in zip(segments, seg_labels):
        for i in seg_idx:
            out[i] = int(lab)
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def temporal_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
) -> dict[int, tuple[float, float]]:
    """Layout where each cluster node's x = median fraction-of-episode-length.

    START is pinned at x_min, terminals at x_max. Vertical positions are
    spread to avoid overlaps within the same x-band. Timesteps are normalized
    to [0, 1] per episode so episodes of different lengths contribute on the
    same scale.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # First pass: episode-length per episode
    ep_max_t: dict[int, float] = defaultdict(float)
    for m in metadata:
        ep = m[ep_key]
        t = float(m.get("timestep", m.get("window_start", 0)))
        if t > ep_max_t[ep]:
            ep_max_t[ep] = t

    cluster_timesteps: dict[int, list[float]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        t = float(m.get("timestep", m.get("window_start", 0)))
        ep = m[ep_key]
        ep_len = max(1.0, ep_max_t[ep])
        cluster_timesteps[c].append(t / ep_len)  # fraction 0..1

    # Compute per-cluster median fraction; use rank-based x to guarantee
    # even spread along the horizontal axis (raw medians often cluster around
    # mid-trajectory which produces visual overlap).
    cluster_median: dict[int, float] = {
        c: float(np.median(ts)) for c, ts in cluster_timesteps.items()
    }
    if not cluster_median:
        return {}
    # Sort clusters by median; assign rank 0..K-1.
    sorted_clusters = sorted(cluster_median.keys(), key=lambda c: cluster_median[c])
    K = len(sorted_clusters)
    cluster_rank: dict[int, int] = {c: i for i, c in enumerate(sorted_clusters)}

    pos: dict[int, tuple[float, float]] = {}
    inner_min = x_min + (x_max - x_min) * 0.1
    inner_max = x_min + (x_max - x_min) * 0.9
    inner_range = inner_max - inner_min

    for c in graph.nodes:
        if c == START_NODE_ID:
            pos[c] = (x_min, 0.0)
        elif c in TERMINAL_NODE_IDS:
            pos[c] = (x_max, 0.0)
        elif c in cluster_rank:
            # Blend rank-position (even) with raw median (preserves "tied" clusters)
            rank_x = inner_min + inner_range * (cluster_rank[c] / max(1, K - 1))
            pos[c] = (rank_x, 0.0)
        else:
            # No data → place mid
            pos[c] = ((x_min + x_max) / 2, 0.0)

    # Resolve y by binning nodes into x-bands and spreading vertically.
    # Tight x-tolerance so we only spread truly-overlapping nodes.
    x_tolerance = (x_max - x_min) * 0.03
    # Sort by x, then by id within
    nodes_by_x = sorted(
        [
            (nid, x, y)
            for nid, (x, y) in pos.items()
            if nid not in TERMINAL_NODE_IDS and nid != START_NODE_ID
        ],
        key=lambda v: (v[1], v[0]),
    )
    bins: list[list[int]] = []
    last_x: float | None = None
    for nid, x, _ in nodes_by_x:
        if last_x is None or abs(x - last_x) > x_tolerance:
            bins.append([nid])
            last_x = x
        else:
            bins[-1].append(nid)
    # Spread y within each bin; spacing scales with bin size
    for bin_nodes in bins:
        n = len(bin_nodes)
        if n == 1:
            continue
        bin_nodes = sorted(bin_nodes, key=lambda nid: nid)
        y_spacing = 0.7 if n <= 4 else 0.5
        for i, nid in enumerate(bin_nodes):
            y = (i - (n - 1) / 2.0) * y_spacing
            pos[nid] = (pos[nid][0], y)

    # Pin success ABOVE failure if both exist
    if SUCCESS_NODE_ID in pos and FAILURE_NODE_ID in pos:
        pos[SUCCESS_NODE_ID] = (pos[SUCCESS_NODE_ID][0], 1.0)
        pos[FAILURE_NODE_ID] = (pos[FAILURE_NODE_ID][0], -1.0)
    if END_NODE_ID in pos and END_NODE_ID != SUCCESS_NODE_ID:
        pos[END_NODE_ID] = (pos[END_NODE_ID][0], 0.0)

    return pos


def sugiyama_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
    y_scale: float = 1.2,
) -> dict[int, tuple[float, float]]:
    """Layered (Sugiyama) layout using NetworkX multipartite_layout.

    Builds a *layered DAG* where each cluster node's layer = its rank by
    median fraction-of-episode-length. Edges within and across layers are
    routed by NetworkX; the resulting y-positions minimise visual crossings
    while keeping x meaningful (= temporal rank).

    START is pinned left, terminals right.
    """
    import networkx as nx

    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    ep_max_t: dict[int, float] = defaultdict(float)
    for m in metadata:
        ep = m[ep_key]
        t = float(m.get("timestep", m.get("window_start", 0)))
        if t > ep_max_t[ep]:
            ep_max_t[ep] = t

    cluster_med: dict[int, float] = {}
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        t = float(m.get("timestep", m.get("window_start", 0)))
        ep_len = max(1.0, ep_max_t[m[ep_key]])
        by_cluster[c].append(t / ep_len)
    for c, ts in by_cluster.items():
        cluster_med[c] = float(np.median(ts))

    sorted_clusters = sorted(cluster_med, key=lambda c: cluster_med[c])
    rank = {c: i + 1 for i, c in enumerate(sorted_clusters)}
    K = len(sorted_clusters)

    G = nx.DiGraph()
    layer_of: dict[int, int] = {}
    for nid in graph.nodes:
        if nid == START_NODE_ID:
            layer_of[nid] = 0
        elif nid in TERMINAL_NODE_IDS:
            layer_of[nid] = K + 1
        else:
            layer_of[nid] = rank.get(nid, K // 2 + 1)
        G.add_node(nid, layer=layer_of[nid])
    for src, targets in graph.transition_probs.items():
        for tgt, p in targets.items():
            if src in G and tgt in G and p > 0:
                G.add_edge(src, tgt, weight=p)

    raw = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    # multipartite_layout returns x in [-0.5, 0.5]-ish — we re-scale.
    xs = [raw[n][0] for n in raw]
    ys = [raw[n][1] for n in raw]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    span_x = max(1e-6, x_hi - x_lo)
    span_y = max(1e-6, y_hi - y_lo)
    pos: dict[int, tuple[float, float]] = {}
    for n in raw:
        rx, ry = raw[n]
        nx_norm = (rx - x_lo) / span_x
        ny_norm = (ry - y_lo) / span_y - 0.5
        pos[n] = (x_min + nx_norm * (x_max - x_min), ny_norm * y_scale * 4)

    # Force terminals
    if SUCCESS_NODE_ID in pos and FAILURE_NODE_ID in pos:
        pos[SUCCESS_NODE_ID] = (x_max, max(2.0, pos[SUCCESS_NODE_ID][1]))
        pos[FAILURE_NODE_ID] = (x_max, min(-2.0, pos[FAILURE_NODE_ID][1]))
    return pos


def force_directed_x_pinned_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
    iterations: int = 200,
    seed: int = 42,
) -> dict[int, tuple[float, float]]:
    """Spring layout with x fixed to temporal rank.

    Each node's x is set by its median timestep rank; spring_layout then
    only adjusts y to minimise edge length. The result preserves the
    temporal axis while letting y-positions drift to avoid edges passing
    through nodes.
    """
    import networkx as nx

    base = temporal_layout(
        graph,
        labels,
        metadata,
        level=level,
        x_min=x_min,
        x_max=x_max,
    )
    # Build a NetworkX graph with weights so spring layout treats high-
    # probability edges as stronger springs.
    G = nx.DiGraph()
    for nid in graph.nodes:
        G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        for tgt, p in targets.items():
            if src in G and tgt in G and p > 0:
                G.add_edge(src, tgt, weight=float(p))

    # spring_layout treats fixed positions in raw (x, y); we want only x
    # fixed. Workaround: run unconstrained spring, then override x.
    initial = {
        nid: np.array([x, np.random.default_rng(seed + nid).uniform(-0.5, 0.5)])
        for nid, (x, _) in base.items()
    }
    new_pos = nx.spring_layout(
        G.to_undirected(),
        pos=initial,
        iterations=iterations,
        seed=seed,
        k=0.6,
    )
    # Rescale y to a readable range, restore x from base
    ys = [v[1] for v in new_pos.values()]
    y_lo, y_hi = min(ys), max(ys)
    span = max(1e-6, y_hi - y_lo)
    out: dict[int, tuple[float, float]] = {}
    for nid, (x_base, _) in base.items():
        y_raw = new_pos[nid][1]
        y_norm = (y_raw - y_lo) / span - 0.5  # in [-0.5, 0.5]
        out[nid] = (x_base, y_norm * 5.0)
    # Force terminals
    if SUCCESS_NODE_ID in out and FAILURE_NODE_ID in out:
        out[SUCCESS_NODE_ID] = (x_max, 2.0)
        out[FAILURE_NODE_ID] = (x_max, -2.0)
    if START_NODE_ID in out:
        out[START_NODE_ID] = (x_min, 0.0)
    return out


def build_trajectory_tree(
    labels: np.ndarray,
    metadata: list[dict],
    level: str = "rollout",
) -> list[dict]:
    """Build a prefix-tree (trie) of run-length-collapsed cluster sequences.

    Each tree node is identified by its full path from the root (a tuple of
    cluster IDs). Stats:
      - n_episodes: how many episodes pass through this prefix
      - n_success / n_failure: outcome breakdown
      - parent_path: the prefix-1 path (so we can render it as a tree)
      - visit_fraction: this node's n_episodes as a fraction of the root's
        total n_episodes (1.0 for the root itself) — computed once here
        rather than downstream by every consumer that wants to know "what
        share of the whole run took this path?"

    Returns a flat list of nodes; the root has parent_path=None.

    The path "(c1, c2, c3, SUCCESS)" means "an episode that visited clusters
    c1 → c2 → c3 and succeeded."
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    success: dict[int, bool | None] = {}
    for m in metadata:
        if "success" in m and m[ep_key] not in success:
            success[m[ep_key]] = m["success"]

    # Per-episode run-length-collapsed sequence with its terminal token
    # appended. Shared with the subclustering stage (which works off a
    # WindowSet rather than these metadata dicts) so the leaves it persists
    # are the same leaves this trie renders.
    sequences = collapse_episode_paths(
        [m[ep_key] for m in metadata],
        [m.get("timestep", m.get("window_start", 0)) for m in metadata],
        labels,
        success,
    )

    # Build trie. node_counts[path] = (n_episodes, n_success, n_failure).
    # Also track which episode indices pass through each prefix so node
    # panels can list videos for the episodes that took that path.
    counts: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0, 0])
    ep_through: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for ep_id, seq, succ in sequences:
        for depth in range(1, len(seq) + 1):
            prefix = seq[:depth]
            counts[prefix][0] += 1
            ep_through[prefix].append(ep_id)
            if succ is True:
                counts[prefix][1] += 1
            elif succ is False:
                counts[prefix][2] += 1

    # Root node (empty path)
    nodes: list[dict] = [
        {
            "path": (),
            "label": "START",
            "cluster_id": START_NODE_ID,
            "depth": 0,
            "n_episodes": len(sequences),
            "n_success": sum(1 for _, _, s in sequences if s is True),
            "n_failure": sum(1 for _, _, s in sequences if s is False),
            "parent_path": None,
            "episode_indices": [ep_id for ep_id, _, _ in sequences],
            "visit_fraction": 1.0,
        }
    ]
    root_n_episodes = len(sequences)
    for path, (n_ep, n_succ, n_fail) in counts.items():
        cid = path[-1]
        nodes.append(
            {
                "path": path,
                "label": f"Behavior {cid}"
                if cid >= 0
                else (
                    "SUCCESS"
                    if cid == SUCCESS_NODE_ID
                    else "FAILURE"
                    if cid == FAILURE_NODE_ID
                    else "END"
                ),
                "cluster_id": cid,
                "depth": len(path),
                "n_episodes": n_ep,
                "n_success": n_succ,
                "n_failure": n_fail,
                "parent_path": path[:-1],
                "episode_indices": list(ep_through[path]),
                "visit_fraction": n_ep / root_n_episodes if root_n_episodes else 0.0,
            }
        )
    return nodes


#: Sentinel cluster id for a collapsed "+N rare variants" tree node. Distinct
#: from every real behavior id (0..K-1) and from TERMINAL_NODE_IDS.
RARE_COLLAPSE_CLUSTER_ID = -100


def _is_rare_child(node: dict, min_visit_fraction: float) -> bool:
    """Whether ``node`` is one of its parent's foldable rare children.

    The one place the fold rule lives, so :func:`collapse_rare_children` and
    :func:`bucket_reveal_paths` cannot drift into folding one set of children
    and revealing another.
    """
    return (
        node["visit_fraction"] < min_visit_fraction and node["cluster_id"] not in TERMINAL_NODE_IDS
    )


def bucket_reveal_paths(
    nodes: list[dict],
    min_visit_fraction: float | None,
    parent_path: tuple,
) -> set[tuple]:
    """The ``expanded`` entries that open ``parent_path``'s summary node fully.

    A summary node stands for its parent's rare children *and their whole
    subtrees*, so that is what expanding it should show. Revealing one layer
    instead leaves every branch point inside it below the threshold — each folds
    again on the next render, and the user clicks "+3 rare variants" only to be
    handed "+2 rare variants" one level down, as many times as the subtree is
    deep.

    Everything inside the bucket is rarer than the child that got folded, so no
    node in there can survive the threshold on its own: the whole subtree of each
    rare child goes in, rather than walking level by level.

    Terminals are left out. Only parent paths mean anything here — folding is
    skipped where a *branch point* is listed — and a caller that also reads this
    set as "nodes whose subtree is showing" would give a ✓/✗ leaf a collapse
    badge for children it does not have.
    """
    parent = tuple(parent_path)
    if not min_visit_fraction or min_visit_fraction <= 0:
        return {parent}

    # Every rare child is `parent + (cluster_id,)`, so "is this path inside the
    # bucket" is one slice and one set lookup rather than a prefix comparison
    # against each rare child in turn.
    rare_ids = {
        tuple(nd["path"])[-1]
        for nd in nodes
        if nd["parent_path"] is not None
        and tuple(nd["parent_path"]) == parent
        and _is_rare_child(nd, min_visit_fraction)
    }
    depth = len(parent)
    reveal = {parent}
    for nd in nodes:
        if nd["cluster_id"] in TERMINAL_NODE_IDS:
            continue
        path = tuple(nd["path"])
        if len(path) > depth and path[depth] in rare_ids and path[:depth] == parent:
            reveal.add(path)
    return reveal


def collapse_rare_children(
    nodes: list[dict],
    min_visit_fraction: float | None,
    expanded: frozenset[tuple] = frozenset(),
) -> list[dict]:
    """Bound trajectory-tree branching without ever hiding an outcome.

    At each branch point, fold every child accounting for less than
    ``min_visit_fraction`` of the *whole run's* episodes — and that child's
    *entire subtree* — into one synthetic "+N rare variants" leaf carrying
    the aggregate episode/outcome counts and the full episode-index list of
    everything folded. This is an episode-count threshold expressed as a
    fraction of the total instead of a flat number, so the same threshold
    means the same thing whether the run has 50 episodes or 5,000 — a flat
    count like "5" is aggressive pruning on a small run and nearly a no-op
    on a large one. Children at or above the fraction always keep their own
    full subtree, however many of them there are: this only folds branches
    that are actually rare in absolute terms, unlike a fixed per-branch cap
    (e.g. "keep the top 3"), which at a wide, roughly-uniform branch point
    (e.g. the root, splitting into K similarly-sized initial behaviors)
    would misleadingly label a majority of the data as a rare afterthought
    just for not being in the top few.

    Nothing is deleted: the summary node is a real, clickable tree node whose
    video panel lists every folded episode — unlike count pruning (which can
    disconnect a whole rare-outcome tail), a rare branch is always one click
    away instead of gone. Each summary node also carries
    ``n_folded_branches`` — how many direct children it stands in for — so
    callers can report "N branches folded" without having to infer it from
    the node count (a folded branch with its own multi-node subtree removes
    more nodes than branches, so node-count deltas overstate that number).

    Rare *terminal* children (✓/✗/END) are never folded, however many of them
    a branch point has. A terminal has no subtree to hide, so bucketing one
    buys no simplification, and what it costs is the outcome itself — a
    "+2 rare variants" node that expands into nothing but "✗ FAILURE" put the
    one thing the tree is read for behind a click. Only rare children with
    real subtrees are worth folding, which is also what keeps the promise in
    this function's first line: every outcome stays on the canvas.

    ``expanded`` is the set of branch-point paths the caller has chosen to
    reveal — folding is skipped there, so their children render as real nodes
    instead of one aggregate leaf. Each entry undoes the fold at exactly one
    branch point: a revealed child with rare grandchildren of its own gets its
    own fresh summary node unless it is in ``expanded`` too. Callers that mean
    "open this bucket all the way" pass the whole subtree — see
    :func:`bucket_reveal_paths`, which works out which paths that is.

    Each node carries its own ``visit_fraction`` — its share of the root's
    total ``n_episodes`` — computed once where the tree is built
    (:func:`build_trajectory_tree`), so folding is a plain per-node
    threshold check here rather than a downstream recomputation.

    ``min_visit_fraction`` of ``None`` or <= 0 disables collapsing (returns
    ``nodes`` unchanged). The root (``parent_path is None``) is never folded.
    """
    if not min_visit_fraction or min_visit_fraction <= 0:
        return nodes

    children_by_parent: dict[tuple, list[dict]] = defaultdict(list)
    for nd in nodes:
        if nd["parent_path"] is not None:
            children_by_parent[tuple(nd["parent_path"])].append(nd)

    folded_child_paths: set[tuple] = set()
    fold_buckets: dict[tuple, list[dict]] = {}
    for parent_path, children in children_by_parent.items():
        if parent_path in expanded:
            continue
        fold = [c for c in children if _is_rare_child(c, min_visit_fraction)]
        if not fold:
            continue
        for c in fold:
            folded_child_paths.add(tuple(c["path"]))
        fold_buckets[parent_path] = fold

    if not fold_buckets:
        return nodes

    def _is_folded_or_descendant(path: tuple) -> bool:
        return any(path[: len(fp)] == fp for fp in folded_child_paths)

    kept = [nd for nd in nodes if not _is_folded_or_descendant(tuple(nd["path"]))]

    for parent_path, fold in fold_buckets.items():
        # parent_path may itself be one of the folded nodes (it had its own
        # rare children folded into a bucket, but parent_path is *also* a
        # rare child of *its* parent and got folded away in turn) — that
        # bucket would be an orphan with no surviving parent to attach to,
        # since its content is already absorbed into the ancestor's bucket.
        if _is_folded_or_descendant(parent_path):
            continue
        n_episodes = sum(c["n_episodes"] for c in fold)
        n_success = sum(c["n_success"] for c in fold)
        n_failure = sum(c["n_failure"] for c in fold)
        episode_indices: list[int] = []
        for c in fold:
            episode_indices.extend(c.get("episode_indices", []))
        summary_path = parent_path + (RARE_COLLAPSE_CLUSTER_ID,)
        kept.append(
            {
                "path": summary_path,
                "label": f"+{len(fold)} rare variant{'s' if len(fold) != 1 else ''} ({n_episodes} eps)",
                "cluster_id": RARE_COLLAPSE_CLUSTER_ID,
                "depth": len(summary_path),
                "n_episodes": n_episodes,
                "n_success": n_success,
                "n_failure": n_failure,
                "parent_path": parent_path,
                "episode_indices": episode_indices,
                "n_folded_branches": len(fold),
            }
        )
    return kept


def collapse_manual(nodes: list[dict], collapsed: frozenset[tuple]) -> list[dict]:
    """Hide the descendants of every manually-collapsed node.

    Unlike ``collapse_rare_children`` (which aggregates *sibling* branches
    below a size threshold into one synthetic summary leaf), this lets the
    user hide the subtree below *any* single node they choose, regardless of
    its size. The node itself is kept — it still shows its own stats and a
    "+" badge — only its descendants disappear, so re-expanding it later
    (dropping it from ``collapsed``) restores exactly what was there before.

    A path that is itself a descendant of another collapsed path is already
    removed by its ancestor, so it never needs its own removal pass — nested
    collapse states simply stay dormant in ``collapsed`` until the ancestor
    is re-expanded, at which point they re-apply automatically.
    """
    if not collapsed:
        return nodes

    def _is_hidden(path: tuple) -> bool:
        return any(len(path) > len(cp) and path[: len(cp)] == cp for cp in collapsed)

    return [nd for nd in nodes if not _is_hidden(tuple(nd["path"]))]


#: What a trajectory tree costs to draw: how many nodes, and how deep.
TreeSize = tuple[int, int]


def tree_size(nodes: list[dict]) -> TreeSize:
    """``(node count, max depth)`` of a trajectory tree."""
    return len(nodes), max((nd["depth"] for nd in nodes), default=0)


def smoothing_effect(
    labels: np.ndarray,
    windows: WindowSet,
    metadata: list[dict],
    sticky_lambda: float,
    *,
    level: str = "rollout",
) -> tuple[TreeSize, TreeSize]:
    """``(size unsmoothed, size after sticky smoothing)`` for one run.

    Priced per run because the answer is not general. Sticky-decode smoothing
    removes the A→B→A label oscillation that overlapping windows produce and that
    run-length collapse cannot, which usually shrinks the tree — but it decides
    each episode's path from that episode's own features, so paths can come out
    shorter *and* less alike, which a prefix tree charges for in branching.

    Measured over all 176 best-per-K runs in ``sweep_all`` (44 datasets, K in
    5/10/15/20) at λ=25: the tree shrinks on 60% of runs, is unchanged on 14%,
    and grows on 26% — worst case 26 → 58 nodes. How much a run oscillates
    barely predicts which (Spearman +0.10), so neither a fixed default nor a
    rule over the oscillation rate can stand in for measuring it. Outcomes are
    never at risk: an episode's terminal comes from its recorded success, and
    0/176 runs changed one.

    The reason no λ generalizes: in :func:`sticky_decode` λ is a flat penalty
    against unary costs normalized *per episode* by their own standard deviation,
    so its effective strength moves with K and with how far apart the centroids
    sit. The band that helps therefore differs per run — below it the tree grows,
    above it real behaviors collapse into one run (see the tests).
    """
    from behavior_graphs.stages.graph.smoothing import smooth_labels

    smoothed = smooth_labels(labels, windows, "sticky", sticky_lambda=sticky_lambda)
    return (
        tree_size(build_trajectory_tree(labels, metadata, level=level)),
        tree_size(build_trajectory_tree(smoothed, metadata, level=level)),
    )
