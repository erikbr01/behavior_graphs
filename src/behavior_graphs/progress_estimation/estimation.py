"""Graph-derived progress labels for reward-model supervision (SARM-style).

Ported from ``online_rl/graph_guided/progress.py`` onto the behavior_graphs
types. SARM (arXiv:2509.25358) shows that dense progress labels for
manipulation reward models must be a function of STATE, not of frame index:
in variable-length episodes the same physical state gets frame-index labels
anywhere in [0.2, 0.8], and that label noise is what breaks the reward model.
SARM buys state-alignment with human effort — annotators segment demos into
semantic subtasks, giving (a) a stage sequence and (b) dataset-level
stage-duration priors used as cumulative progress offsets.

The behavior graph produces both ingredients with zero annotation:

  clusters                 -> stages
  segment order on         -> stage order (mean normalized time over
  successful episodes         successful episodes)
  mean time-fraction per   -> SARM's duration priors alpha_k
  cluster on successes
  P(absorb at SUCCESS | c) -> a calibrated monotone progress coordinate that
                              SARM approximates with cumulative priors

The core functions are plain numpy over per-episode label/probability
sequences (``list[np.ndarray]``), so they stay agnostic of the pipeline
stack; :func:`episode_label_sequences` bridges from the flat per-window
labels + :class:`~behavior_graphs.core.types.WindowSet` representation, and
:func:`success_absorption` bridges from
:class:`~behavior_graphs.core.graph.BehaviorGraph`.

All sequences are WINDOW-level (one entry per graph window), the same
granularity the clustering stage labels; see
:func:`behavior_graphs.progress_estimation.scoring.expand_to_steps` for going back to
per-timestep scores.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from behavior_graphs.core.graph import (
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
)
from behavior_graphs.core.types import WindowSet

__all__ = [
    "StageSchedule",
    "stage_schedule",
    "stage_progress",
    "absorption_progress",
    "frame_index_progress",
    "chunk_deltas",
    "rabc_weights",
    "sarm_soft_clip",
    "same_state_label_std",
    "success_auroc",
    "monotonicity",
    "success_absorption",
    "episode_label_sequences",
]


# --------------------------------------------------------------- adapters


def episode_label_sequences(labels: np.ndarray, windows: WindowSet) -> list[np.ndarray]:
    """Split flat per-window labels into per-episode sequences in temporal order.

    Returns one int array per entry of ``windows.episodes`` (in that order);
    episodes without windows yield an empty array. Noise labels (``-1``) are
    kept so window alignment is preserved — downstream progress functions read
    them as zero progress.
    """
    labels = np.asarray(labels)
    per_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, w in enumerate(windows.windows):
        per_episode[w.episode_idx].append((w.start, int(labels[i])))
    seqs = []
    for ep in windows.episodes:
        seq = sorted(per_episode.get(ep.index, []), key=lambda x: x[0])
        seqs.append(np.array([label for _, label in seq], dtype=np.int64))
    return seqs


def success_absorption(graph: BehaviorGraph) -> dict[int, float]:
    """P(eventually absorb at SUCCESS | currently at node), per cluster node.

    Standard absorbing-chain solve ``q = (I - P_NN)^-1 b`` with
    ``b[i] = P(i -> SUCCESS)`` in one step, over ``graph.transition_probs``.
    SUCCESS/FAILURE (and END, if present) are absorbing; nodes with no
    outgoing edges absorb with probability 0. Undiscounted — this is
    reachability, not value, so it separates "doomed" from merely "far away".

    Special (negative-id) nodes are part of the solve (START feeds the
    chain) but excluded from the returned dict.
    """
    transition_probs = graph.transition_probs
    node_ids = set(transition_probs)
    for targets in transition_probs.values():
        node_ids.update(targets)
    nonterminal = sorted(n for n in node_ids if n not in TERMINAL_NODE_IDS)
    index = {nid: i for i, nid in enumerate(nonterminal)}
    n = len(nonterminal)
    if n == 0:
        return {}

    P = np.zeros((n, n))
    b = np.zeros(n)
    for nid in nonterminal:
        i = index[nid]
        for tgt, p in transition_probs.get(nid, {}).items():
            if tgt in index:
                P[i, index[tgt]] = p
            elif tgt == SUCCESS_NODE_ID:
                b[i] += p
    q = np.linalg.solve(np.eye(n) - P, b)
    # Numerical noise can push slightly outside [0, 1].
    q = np.clip(q, 0.0, 1.0)
    return {nid: float(q[index[nid]]) for nid in nonterminal if nid >= 0}


# ------------------------------------------------------------------ schemes


@dataclass(frozen=True)
class StageSchedule:
    """Cluster -> (base offset, span) mapping derived from successful episodes.

    ``base[c]`` is the cumulative progress reached when stage ``c`` begins and
    ``span[c]`` the progress mass of the stage itself (SARM's alpha_k prior),
    so labels inside stage ``c`` ramp over ``[base[c], base[c] + span[c]]``.
    Clusters never visited on a successful episode are off the success path:
    they get base 0 and span 0 (floor label — "no measurable progress").
    """

    order: tuple[int, ...]
    base: dict[int, float]
    span: dict[int, float]


def stage_schedule(
    label_seqs: list[np.ndarray],
    successes: list[bool],
) -> StageSchedule:
    """Derive stage order and duration priors from successful episodes.

    Order = clusters sorted by their mean normalized time of occurrence over
    successful episodes (SARM's annotated subtask order, discovered instead of
    annotated). Priors = mean fraction of episode time spent in each cluster
    over successful episodes, normalized to sum to 1.
    """
    succ = [np.asarray(seq) for seq, ok in zip(label_seqs, successes) if ok]
    succ = [seq for seq in succ if len(seq)]
    if not succ:
        raise ValueError("stage_schedule needs at least one successful episode")

    clusters = sorted({int(c) for seq in succ for c in seq if c >= 0})
    mean_time: dict[int, list[float]] = {c: [] for c in clusters}
    frac: dict[int, list[float]] = {c: [] for c in clusters}
    for seq in succ:
        n = len(seq)
        t_norm = np.arange(n) / max(n - 1, 1)
        for c in clusters:
            mask = seq == c
            if mask.any():
                mean_time[c].append(float(t_norm[mask].mean()))
            frac[c].append(float(mask.mean()))

    order = tuple(sorted(clusters, key=lambda c: float(np.mean(mean_time[c]))))
    raw = np.array([float(np.mean(frac[c])) for c in order])
    spans = raw / raw.sum()

    base: dict[int, float] = {}
    span: dict[int, float] = {}
    cum = 0.0
    for c, s in zip(order, spans):
        base[c] = cum
        span[c] = float(s)
        cum += float(s)
    return StageSchedule(order=order, base=base, span=span)


def stage_progress(
    labels: np.ndarray,
    schedule: StageSchedule,
    normalize_visited: bool = False,
) -> np.ndarray:
    """SARM-style label: cumulative stage base + linear ramp within each
    contiguous same-cluster segment.

    Revisiting an earlier stage drops the label back to that stage's band —
    that is the desired behavior (regressions must read as regressions;
    SARM needed synthetic "rewind" augmentation to teach this).

    ``normalize_visited`` rebuilds the bands over only the clusters THIS episode
    actually enters, keeping the schedule's global cluster order and using its
    global spans as relative weights. Motivation, measured on sim-bottles: with
    K=64 an episode visits only ~3 clusters, so ~95% of the [0, 1] range belongs
    to bands it never enters and the curve advances by *jumping over* them. That
    puts 62% of all forward progress into 1% of frames, which is
    fine for a ranking metric (VOC 0.977) and ruinous for anything that
    differentiates the curve — the RA-BC weights it produced had ESS 0.066.

    This is deliberately NOT a rescale to each segment's observed duration: that
    would make progress identical to normalized time, scoring perfectly on both
    VOC and weight-flatness while carrying no information from the graph at all.
    Order and relative span still come from the fitted schedule; only the unvisited
    bands are removed.
    """
    labels = np.asarray(labels)
    out = np.zeros(len(labels), dtype=np.float64)

    base, span = schedule.base, schedule.span
    if normalize_visited:
        # Segment boundaries in occurrence order, then bands over the visited
        # clusters only, ordered by the schedule's global base (its stage order).
        visited = sorted({int(c) for c in labels}, key=lambda c: schedule.base.get(c, 0.0))
        weights = np.array([schedule.span.get(c, 0.0) for c in visited], dtype=np.float64)
        if weights.sum() <= 0:  # no span information: fall back to equal bands
            weights = np.ones(len(visited), dtype=np.float64)
        weights = weights / weights.sum()
        base, span = {}, {}
        cum = 0.0
        for c, wgt in zip(visited, weights):
            base[c] = cum
            span[c] = float(wgt)
            cum += float(wgt)

    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        c = int(labels[i])
        b = base.get(c, 0.0)
        s = span.get(c, 0.0)
        seg_len = j - i
        # ramp midpoint-aligned: single-window segment sits at the band center
        ramp = (np.arange(seg_len) + 0.5) / seg_len
        out[i:j] = b + s * ramp
        i = j
    return out


def absorption_progress(
    labels: np.ndarray | None,
    absorb: dict[int, float],
    soft_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Progress = P(eventually reach SUCCESS | current cluster).

    Hard variant: look up the assigned cluster. Soft variant: expectation over
    the soft assignment ``(n_windows, n_clusters)`` — smoother near cluster
    boundaries. Unknown clusters read 0 (off-graph = no evidence of progress).
    """
    if soft_probs is not None:
        soft_probs = np.asarray(soft_probs)
        vec = np.array([absorb.get(c, 0.0) for c in range(soft_probs.shape[1])])
        return soft_probs @ vec
    if labels is None:
        raise ValueError("need labels or soft_probs")
    labels = np.asarray(labels)
    return np.array([absorb.get(int(c), 0.0) for c in labels])


def frame_index_progress(n: int) -> np.ndarray:
    """The baseline SARM argues against: progress = normalized frame index."""
    if n <= 1:
        return np.zeros(max(n, 1))
    return np.arange(n) / (n - 1)


# ------------------------------------------------------------------ RA-BC


def sarm_soft_clip(
    r: np.ndarray,
    kappa: float,
    mean: float,
    std: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """SARM Eq. 9 weighting of a reward signal ``r``.

        w = 1                       r > kappa
            clip(w_tilde, 0, 1)     0 <= r <= kappa
            0                       r < 0
        w_tilde = (r - (mu - 2 sigma)) / (4 sigma)

    The ``r < 0 -> 0`` branch is the point: without it a backtracking chunk lands
    mid-ramp (~0.5 for mu > 0 data) instead of being dropped, which flattens the
    weight distribution and blunts the whole scheme.

    ``mean``/``std`` are passed in rather than taken from ``r`` so that calling
    this per-episode gives the same answer as calling it once on the concatenated
    array -- the moments must be the dataset's, not the episode's.
    """
    w_tilde = np.clip((r - (mean - 2.0 * std)) / (4.0 * std + eps), 0.0, 1.0)
    return np.where(r > kappa, 1.0, np.where(r >= 0.0, w_tilde, 0.0))


def chunk_deltas(progress_seqs: list[np.ndarray], delta: int) -> list[np.ndarray]:
    """Progress gained over the action chunk starting at each step, per episode.

    ``r_i = p[i + delta] - p[i]``, with the index clipped at the episode tail.
    """
    out = []
    for p in progress_seqs:
        p = np.asarray(p)
        idx = np.minimum(np.arange(len(p)) + delta, len(p) - 1)
        out.append(p[idx] - p)
    return out


def rabc_weights(
    progress_seqs: list[np.ndarray],
    delta: int,
    kappa: float | None = None,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Offline RA-BC sample weights (SARM sec. RA-BC), one weight per entry.

    Per-sample signal is the progress delta over the action-chunk horizon,
    r_i = p[i + delta] - p[i] (delta clipped at the episode tail), mapped through
    :func:`sarm_soft_clip`. Statistics are computed over the whole dataset in one
    pass (the online paper uses Welford running stats; offline the batch moments
    are the same numbers). ``kappa`` defaults to the dataset mean delta, SARM's
    medium-selectivity setting.
    """
    deltas = chunk_deltas(progress_seqs, delta)
    flat = np.concatenate(deltas)
    mu, sigma = float(flat.mean()), float(flat.std())
    k = mu if kappa is None else kappa
    return [sarm_soft_clip(r, k, mu, sigma, eps=eps) for r in deltas]


# ------------------------------------------------------------------ metrics


def same_state_label_std(
    features: np.ndarray,
    episode_ids: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 8,
    n_queries: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """SARM's Table-1 argument as a number: label spread over near-identical
    states drawn from *different* episodes.

    For each sampled query window, take its ``n_neighbors`` nearest windows in
    feature space, keep only cross-episode neighbors, and record the standard
    deviation of their labels (query included). Returns the mean std over
    queries with at least 3 cross-episode neighbors. A state-aligned labeling
    scheme scores low; frame-index labeling scores high.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = rng or np.random.default_rng(0)
    features = np.asarray(features)
    n = len(features)
    q_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    # over-fetch so that same-episode neighbors can be discarded
    nn = NearestNeighbors(n_neighbors=min(4 * n_neighbors, n), algorithm="kd_tree")
    nn.fit(features)
    _, nbrs = nn.kneighbors(features[q_idx])

    stds = []
    for qi, row in zip(q_idx, nbrs):
        cross = [j for j in row if episode_ids[j] != episode_ids[qi]][:n_neighbors]
        if len(cross) < 3:
            continue
        stds.append(float(np.std(np.append(labels[cross], labels[qi]))))
    return float(np.mean(stds)) if stds else float("nan")


def success_auroc(scores: np.ndarray, successes: np.ndarray) -> float:
    """AUROC of an episode-level score against the success flag.

    Rank-based (Mann-Whitney), ties get half credit. 0.5 = chance. This is the
    threshold-free version of SARM's SE/PSE/FE rollout classification: can the
    label scheme tell successful rollouts from failed ones at all?
    """
    scores = np.asarray(scores, dtype=float)
    successes = np.asarray(successes, dtype=bool)
    # An episode with no score at all leaves the comparison rather than losing
    # it. NaN is False for both `>` and `==`, so a NaN used to contribute zero
    # to the numerator while still counting in the denominator -- a silent
    # downward bias whose size is the fraction of episodes that failed to
    # decode.
    scored = np.isfinite(scores)
    pos, neg = scores[successes & scored], scores[(~successes) & scored]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def monotonicity(
    progress_seqs: list[np.ndarray],
    successes: list[bool],
) -> float:
    """Mean Spearman correlation of label vs window index over successful
    episodes. On successes, real progress does move forward in time, so a
    usable scheme should score well ABOVE 0 here — this metric keeps the
    consistency metric honest (labeling everything 0.5 is perfectly
    consistent and perfectly useless).
    """
    from scipy.stats import spearmanr

    rhos = []
    for p, ok in zip(progress_seqs, successes):
        if not ok or len(p) < 3:
            continue
        # [0] = correlation across scipy versions (.statistic is new-only)
        rho = spearmanr(np.arange(len(p)), p)[0]
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def progress_regression_metrics(
    progress_seqs: list[np.ndarray],
    successes: list[bool],
) -> dict[str, float]:
    """Paper-comparable progress metrics (TOPReward arXiv:2602.19313, ROBOMETER
    arXiv:2603.02115), averaged over SUCCESSFUL episodes vs linear-interpolation
    GT progress 0->1: voc (Spearman rank vs frame order), kendall (tau-b),
    pearson (vs linear GT), mae (|pred-GT|, both in [0,1]). Ported from the
    graph_guided prototype; see PROGRESS_ESTIMATION_LOG."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    voc, ken, pear, mae = [], [], [], []
    for p, ok in zip(progress_seqs, successes):
        if not ok or len(p) < 3:
            continue
        n = len(p)
        gt = np.linspace(0.0, 1.0, n)
        pc = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        idx = np.arange(n)
        # index [0] = correlation across scipy versions (.statistic is new-only)
        rho = spearmanr(idx, pc)[0]
        tau = kendalltau(idx, pc)[0]
        r = pearsonr(pc, gt)[0] if pc.std() > 1e-9 else 0.0
        if np.isfinite(rho):
            voc.append(float(rho))
        if np.isfinite(tau):
            ken.append(float(tau))
        if np.isfinite(r):
            pear.append(float(r))
        mae.append(float(np.mean(np.abs(pc - gt))))
    m = lambda a: float(np.mean(a)) if a else float("nan")  # noqa: E731
    return {"voc": m(voc), "kendall": m(ken), "pearson": m(pear), "mae": m(mae)}


def per_episode_progress_metrics(seq: np.ndarray) -> dict[str, float]:
    """VOC/Kendall/Pearson/MAE for a SINGLE episode's progress curve vs the
    linear GT ramp (for bootstrapping over episodes without re-running models)."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    p = np.clip(np.asarray(seq, dtype=float), 0.0, 1.0)
    n = len(p)
    if n < 3:
        return {
            "voc": float("nan"),
            "kendall": float("nan"),
            "pearson": float("nan"),
            "mae": float("nan"),
        }
    gt = np.linspace(0.0, 1.0, n)
    idx = np.arange(n)
    r = pearsonr(p, gt)[0] if p.std() > 1e-9 else 0.0
    return {
        "voc": float(spearmanr(idx, p)[0]),
        "kendall": float(kendalltau(idx, p)[0]),
        "pearson": float(r),
        "mae": float(np.mean(np.abs(p - gt))),
    }
