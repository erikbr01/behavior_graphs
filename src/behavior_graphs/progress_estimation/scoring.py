"""Demo-curation scoring schemes over graph-derived progress labels.

Ported from ``online_rl/scripts/score_demos_rabc.py`` as pure functions over
per-episode label / progress sequences, so each scheme is testable on
synthetic arrays without fitting a graph. ``scripts/curation/score_demos.py`` wires
them to a fitted graph bundle and a robomimic demo hdf5.

Schemes (each returns one score array per episode, granularity as noted):

- :func:`ramp_scores`     RA-BC soft-clip weights on progress deltas
- :func:`relu_scores`     relu(progress delta), mean-normalized
- :func:`length_scores`   1/T prior, constant per episode (per-STEP)
- :func:`seg_trim_scores` data diet: fractional dwell caps from successful
                          rollouts, 3rd+ revisits of a cluster dropped
- :func:`stage_balance_scores` inverse stage-mass tempered by ** 0.5,
                          capped at 5x median
- :func:`oracle_scores`   externally given per-episode weights (robomimic
                          mask groups), constant per episode (per-STEP)
- :func:`shuffled_scores` seed-fixed permutation control of any flat score

:func:`expand_to_steps` maps window-level scores back to per-timestep scores
by interpolating between window centers (clamped at the episode edges),
matching the upsampling in the source script. :func:`ess` and
:func:`zero_frac` are the summary statistics reported per scheme.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.progress_estimation.estimation import rabc_weights

__all__ = [
    "ramp_scores",
    "relu_scores",
    "length_scores",
    "dwell_fraction_stats",
    "seg_trim_scores",
    "stage_balance_scores",
    "oracle_scores",
    "shuffled_scores",
    "expand_to_steps",
    "ess",
    "zero_frac",
]


# ------------------------------------------------------------ step mapping


def expand_to_steps(
    per_window_scores: np.ndarray,
    window_width: int,
    stride: int,
    episode_length: int,
) -> np.ndarray:
    """Map window-level scores back to per-step scores for one episode.

    Windows are the ones :class:`~behavior_graphs.stages.windowing.SlidingWindow`
    produces: starts ``0, stride, 2*stride, ...`` up to
    ``episode_length - window_width`` (a single whole-episode window when the
    episode is shorter than ``window_width``). Per-step scores interpolate
    linearly between window CENTERS and clamp to the nearest window outside
    the first/last center — the same upsampling the source script uses, which
    also covers the gaps an undersampling stride (stride > width) leaves.
    """
    scores = np.asarray(per_window_scores, dtype=np.float64)
    if episode_length <= 0:
        return np.zeros(0)
    if episode_length < window_width:
        starts = np.array([0])
        centers = np.array([(episode_length - 1) / 2.0])
    else:
        starts = np.arange(0, episode_length - window_width + 1, stride)
        centers = starts + (window_width - 1) / 2.0
    if len(scores) != len(starts):
        raise ValueError(
            f"got {len(scores)} window scores but episode_length={episode_length} "
            f"with width={window_width}, stride={stride} yields {len(starts)} windows"
        )
    t = np.arange(episode_length, dtype=np.float64)
    return np.interp(t, centers, scores)


# ------------------------------------------------------------------ stats


def ess(weights: np.ndarray) -> float:
    """Normalized effective sample size ``(sum w)^2 / (N * sum w^2)`` in (0, 1]."""
    w = np.asarray(weights, dtype=np.float64)
    denom = len(w) * float((w**2).sum())
    if denom <= 0:
        return 0.0
    return float(w.sum() ** 2 / denom)


def zero_frac(weights: np.ndarray, threshold: float = 0.05) -> float:
    """Fraction of samples with (near-)zero weight — the "starved" mass."""
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return 0.0
    return float((w < threshold).mean())


# ---------------------------------------------------------------- schemes


def ramp_scores(
    progress_seqs: list[np.ndarray],
    delta: int,
    kappa: float | None = None,
) -> list[np.ndarray]:
    """SARM RA-BC weights: soft clip ramp on progress deltas (see
    :func:`~behavior_graphs.progress_estimation.estimation.rabc_weights`)."""
    return rabc_weights(progress_seqs, delta=delta, kappa=kappa)


def relu_scores(
    progress_seqs: list[np.ndarray],
    delta: int,
    eps: float = 1e-9,
) -> list[np.ndarray]:
    """relu(progress delta over the horizon), normalized to global mean 1.

    Stall/backtrack chunks get 0, productive chunks weight proportional to
    progress rate; per-episode total mass ~ total progress, so efficient
    (short) episodes get higher per-step weight without a length hack.
    """
    seqs = []
    for p in progress_seqs:
        p = np.asarray(p, dtype=np.float64)
        idx = np.minimum(np.arange(len(p)) + delta, len(p) - 1)
        seqs.append(np.maximum(p[idx] - p, 0.0))
    flat = np.concatenate(seqs)
    return [w / (flat.mean() + eps) for w in seqs]


def length_scores(episode_lengths: list[int]) -> list[np.ndarray]:
    """Per-STEP score ``1/T`` per episode, normalized to global mean 1.

    The cheap heuristic every graph scheme must be compared against — worse
    operators take roughly twice as long.
    """
    seqs = [np.full(n, 1.0 / max(n, 1), dtype=np.float64) for n in episode_lengths]
    flat = np.concatenate(seqs) if seqs else np.zeros(0)
    mean = flat.mean() if len(flat) else 1.0
    return [w / mean for w in seqs]


def _segments(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Contiguous same-label runs as ``(label, window_indices)``; noise (-1) skipped.

    A noise window *ends* the run it interrupts rather than being transparent to
    it. Skipping it silently merged two separate visits into one non-contiguous
    segment, so ``[3, -1, 3]`` read as a single dwell of 2 instead of two dwells
    of 1 -- which understated the dwell distribution in
    :func:`dwell_fraction_stats`, hid the revisit from
    :func:`seg_trim_scores`'s loop detector, and let its ``wt[idx[cap:]]`` zero
    windows that were not adjacent to the over-long part.
    """
    labels = np.asarray(labels)
    segments: list[tuple[int, list[int]]] = []
    previous: int | None = None
    for i, label in enumerate(labels):
        label = int(label)
        if label == -1:
            previous = None  # whatever follows starts a new run
            continue
        if segments and previous == label:
            segments[-1][1].append(i)
        else:
            segments.append((label, [i]))
        previous = label
    return [(label, np.array(idx)) for label, idx in segments]


def dwell_fraction_stats(
    label_seqs: list[np.ndarray],
    successes: list[bool],
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-cluster (mean, std) of FRACTIONAL segment dwell on successful episodes.

    Dwell is measured in fractions of the episode's window count: unit-free,
    so stats from subsampled rollouts transfer to full-rate demos (absolute
    window caps trimmed 80% of demo steps on ~8.5x-subsampled transport
    rollouts).
    """
    dwell: dict[int, list[float]] = {}
    for seq, ok in zip(label_seqs, successes):
        if not ok or len(seq) == 0:
            continue
        for label, idx in _segments(seq):
            dwell.setdefault(label, []).append(len(idx) / max(len(seq), 1))
    mu = {c: float(np.mean(v)) for c, v in dwell.items()}
    sd = {c: float(np.std(v) + 1e-6) for c, v in dwell.items()}
    return mu, sd


def seg_trim_scores(
    label_seqs: list[np.ndarray],
    dwell_mu: dict[int, float],
    dwell_sd: dict[int, float],
    max_revisits: int = 2,
) -> list[np.ndarray]:
    """Data diet: per-WINDOW 0/1 weights trimming over-long dwells and loops.

    Within each episode, a segment's dwell is capped at
    ``ceil((mu + sd) * n_windows)`` windows (fractional cap learned from
    successful rollouts via :func:`dwell_fraction_stats`; clusters without
    stats are uncapped), and the ``max_revisits + 1``-th and later revisits of
    a cluster are dropped entirely. Everything kept stays at weight 1.
    """
    out = []
    for seq in label_seqs:
        wt = np.ones(len(seq), dtype=np.float64)
        visits: dict[int, int] = {}
        for label, idx in _segments(seq):
            visits[label] = visits.get(label, 0) + 1
            if visits[label] > max_revisits:
                wt[idx] = 0.0
                continue
            cap_frac = (dwell_mu[label] + dwell_sd[label]) if label in dwell_mu else 1.0
            cap = max(1, int(np.ceil(cap_frac * len(seq))))
            if len(idx) > cap:
                wt[idx[cap:]] = 0.0
        out.append(wt)
    return out


def stage_balance_scores(
    label_seqs: list[np.ndarray],
    power: float = 0.5,
    cap_multiple: float = 5.0,
    eps: float = 1e-9,
) -> list[np.ndarray]:
    """Batch composition: per-WINDOW weight ~ inverse stage mass, tempered.

    Class-balancing with graph stages as classes: weight
    ``(1 / mass[c]) ** power`` (``power=0.5`` tempers the correction), capped
    at ``cap_multiple`` times the median weight so rare clusters cannot blow
    up, then normalized to global mean 1. Stage mass is counted over the
    given sequences themselves (the demo set being balanced).
    """
    # Noise is not a stage, so it is not a class to balance: counting -1 in the
    # histogram made one pseudo-class whose size then boosted or suppressed
    # every unlabelled window. Those windows take a neutral weight of 1 instead
    # and are normalized along with the rest.
    stage_mass: dict[int, int] = {}
    for seq in label_seqs:
        for c in np.asarray(seq):
            c = int(c)
            if c == -1:
                continue
            stage_mass[c] = stage_mass.get(c, 0) + 1
    seqs = [
        np.array(
            [1.0 if int(c) == -1 else (1.0 / stage_mass[int(c)]) ** power for c in seq],
            dtype=np.float64,
        )
        for seq in label_seqs
    ]
    flat = np.concatenate(seqs) if seqs else np.zeros(0)
    if len(flat) == 0:
        return seqs
    cap = cap_multiple * float(np.median(flat))
    seqs = [np.minimum(w, cap) for w in seqs]
    mean = float(np.concatenate(seqs).mean())
    return [w / (mean + eps) for w in seqs]


def oracle_scores(
    episode_lengths: list[int],
    episode_weights: list[float],
) -> list[np.ndarray]:
    """Per-STEP scores from externally given per-episode quality weights
    (e.g. robomimic operator-proficiency mask groups). Not normalized — the
    group weights are meaningful absolute values.
    """
    if len(episode_lengths) != len(episode_weights):
        raise ValueError(f"{len(episode_lengths)} lengths but {len(episode_weights)} weights")
    return [
        np.full(n, float(w), dtype=np.float64) for n, w in zip(episode_lengths, episode_weights)
    ]


def shuffled_scores(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Seed-fixed permutation of a flat score array: an information-free
    control with an identical weight histogram."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(scores))
