"""Scoring for subtask-boundary agreement. One implementation, several callers.

Split out of ``subtask_boundaries`` so the pipeline evaluator and the offline
re-scorer share it exactly. Anything that measures a graph against the subtask
annotations should import from here; two implementations of "the metric" is how a
comparison quietly stops being a comparison.

Four things this gets right that a boundary-F1-and-nothing-else does not:

**The tolerance is frozen, not derived per score set.** Deriving tau from the
episodes being scored makes every fold a slightly different metric -- measured on
ABC bottles, 0.5062s on cv4 against 0.5410s on cv0, a 6.8% spread, so a
cross-fold mean averages over different measurements. Callers pass ``tol``
explicitly; the recommended value comes from the *train* half once.

**Gain is reported with density, and a density band is available.** Raw F1 is
gameable by emitting more boundaries: on this data K 6->32 moves precision
0.219->0.224 while recall goes 0.346->0.527. Subtracting a count-matched uniform
predictor removes that, but it opens the opposite exploit -- a sparse predictor
has a *weak* uniform reference, so gain flatters it. 87 of 5,176 swept jobs score
gain > +0.02 at density < 0.6 that way. Neither direction is a method working, so
``in_band`` marks whether a row is in a plausible density regime at all.

**Subtask identity is scored, not just boundary position.** The annotations carry
25 distinct subtask strings and a boundary-only metric discards all of them, so a
graph that finds the right change points while assigning meaningless clusters is
indistinguishable from one that separates pick from place. Frame-wise ARI and NMI
against ground-truth subtask identity, plus segmental F1 at IoU thresholds, close that.

**Non-task segments are handled explicitly.** `mistake`, `reset both arms`,
`adjust` and `trim` are 540 of 6,102 segments (8.8%) and are not subtask
structure. Under ``gt_set="task"`` their spans become *ignore regions*:
boundaries adjacent to them are not required, and predictions inside them are
neither credited nor penalised -- which is the honest treatment, where simply
deleting the boundaries would leave a prediction inside an operator error counted
as a false positive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Segment labels that are not subtask structure. Matched case-insensitively on
#: the stripped string.
NON_TASK_LABELS = frozenset({"mistake", "reset both arms", "adjust", "trim"})

#: IoU thresholds for segmental F1. The first four match the CPU study; 0.90 is
#: a rung above the criterion, so "is 0.75 still leaving headroom" is a number
#: rather than an argument.
#:
#: Adding one costs almost nothing: ``segmental_counts`` pairs predictions to
#: ground truth by *intersection* and computes the IoU matrix once, so a
#: threshold is a single comparison over an array that already exists. Which is
#: also why this is a property of the metric rather than of a dataset --
#: ``EvalSpec.iou_thresholds`` selects which rungs a report shows, it does not
#: decide which are computed.
#:
#: Expect 0.90 to floor. Measured on the cooking grid (864 cells, gripper-derived
#: ground truth), the *best* cell averages 0.646 matched IoU and the median is
#: 0.530 -- so a 0.90 threshold sits above everything the pipeline produces and
#: separates nothing. It earns its place by saying that, not by ranking.
IOU_THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 0.90)


def load_annotations(root: Path | str) -> list[dict]:
    path = Path(root) / "annotations" / "segments.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; boundary scoring needs gt subtask segments "
            "(written by scripts/ablation_study/build_*)"
        )
    return json.loads(path.read_text())


def build_ground_truth(root: Path | str, gt_set: str = "task") -> dict[int, dict[str, Any]]:
    """``episode index -> {fps, end_sec, boundaries, ignore, labels}``.

    ``gt_set="task"`` drops non-task segments to ignore regions (the primary
    view); ``"all"`` keeps every annotated segment as structure.
    """
    if gt_set not in ("task", "all"):
        raise ValueError(f"gt_set must be 'task' or 'all'; got {gt_set!r}")
    out: dict[int, dict[str, Any]] = {}
    for ep in load_annotations(root):
        segs = ep["segments"]
        if not segs:
            continue
        is_task = [s["subtask"].strip().lower() not in NON_TASK_LABELS for s in segs]
        end_sec = float(segs[-1]["end_sec"])

        if gt_set == "all":
            bounds = [float(s["end_sec"]) for s in segs[:-1]]
            ignore: list[tuple[float, float]] = []
        else:
            # A boundary is required only between two consecutive *task*
            # segments; the span of any non-task segment is ignored entirely.
            bounds = [
                float(segs[i]["end_sec"])
                for i in range(len(segs) - 1)
                if is_task[i] and is_task[i + 1]
            ]
            ignore = [
                (float(s["start_sec"]), float(s["end_sec"]))
                for s, ok in zip(segs, is_task)
                if not ok
            ]

        # Per-frame subtask identity, for the label-agreement metrics.
        fps = float(ep["fps"])
        n = int(round(end_sec * fps))
        vocab: dict[str, int] = {}
        frame_lab = np.full(n, -1, dtype=np.int32)
        for s, ok in zip(segs, is_task):
            if gt_set == "task" and not ok:
                continue
            name = s["subtask"].strip()
            cid = vocab.setdefault(name, len(vocab))
            a = int(round(float(s["start_sec"]) * fps))
            b = int(round(float(s["end_sec"]) * fps))
            frame_lab[max(0, a) : min(n, b)] = cid

        out[int(ep["index"])] = {
            "fps": fps,
            "end_sec": end_sec,
            "boundaries": np.asarray(bounds, dtype=np.float64),
            "ignore": ignore,
            "frame_labels": frame_lab,
            "segments": [
                (float(s["start_sec"]), float(s["end_sec"]))
                for s, ok in zip(segs, is_task)
                if gt_set == "all" or ok
            ],
        }
    return out


def drop_ignored(times: np.ndarray, ignore: list[tuple[float, float]]) -> np.ndarray:
    """Predictions falling inside an ignore region, removed.

    Not counted as false positives: the method is not being asked about those
    spans, so charging it for guessing there would penalise it for the
    annotation's bookkeeping.
    """
    if not ignore or times.size == 0:
        return times
    keep = np.ones(times.shape, dtype=bool)
    for a, b in ignore:
        keep &= ~((times >= a) & (times <= b))
    return times[keep]


def merge_short(bounds: np.ndarray, span: float, min_sec: float) -> np.ndarray:
    """Drop boundaries that would create a segment shorter than ``min_sec``."""
    if min_sec <= 0 or bounds.size == 0:
        return bounds
    kept: list[float] = []
    last = 0.0
    for b in np.sort(bounds):
        if b - last < min_sec or span - b < min_sec:
            continue
        kept.append(float(b))
        last = float(b)
    return np.asarray(kept, dtype=np.float64)


def match_count(pred: np.ndarray, gt: np.ndarray, tol: float) -> int:
    """Greedy one-to-one matches within ``tol`` seconds, nearest pair first."""
    if pred.size == 0 or gt.size == 0:
        return 0
    cost = np.abs(pred[:, None] - gt[None, :])
    hits = 0
    used_p: set[int] = set()
    used_g: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if cost[i, j] > tol:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(int(i))
        used_g.add(int(j))
        hits += 1
    return hits


def uniform_boundaries(span: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    return np.asarray([span * (k + 1) / (n + 1) for k in range(n)], dtype=np.float64)


def _f1(tp: float, n_pred: int, n_gt: int) -> float:
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _segments_from(
    bounds: np.ndarray, span: float, ignore: list[tuple[float, float]] | None = None
) -> list[tuple[float, float]]:
    """Segments implied by ``bounds``, clipped out of any ignore region.

    Without the clipping a predicted segment straddling a `mistake` span is
    compared, as one long interval, against the task segments on either side --
    so it scores poor IoU with both and the method is penalised for a region the
    boundary metric explicitly excused it from. Gold segments never cross an
    ignore region, so predicted ones must not either.
    """
    edges = np.concatenate(([0.0], np.sort(bounds), [span]))
    segs = [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if not ignore:
        return segs
    for lo, hi in sorted(ignore):
        cut: list[tuple[float, float]] = []
        for a, b in segs:
            if b <= lo or a >= hi:
                cut.append((a, b))
                continue
            if a < lo:
                cut.append((a, lo))
            if b > hi:
                cut.append((hi, b))
        segs = cut
    return [(a, b) for a, b in segs if b - a > 1e-9]


def segmental_counts(
    pred_bounds: np.ndarray, gt: dict[str, Any]
) -> dict[Any, tuple[int, int, int]]:
    """COCO-style segment detection: match first, then classify matches by IoU.

    The structure is object detection's, with the one substitution temporal
    segments force:

    1. **Assign.** Each predicted segment is assigned to the ground-truth segment it
       overlaps most, by *absolute intersection*. A prediction overlapping no ground-truth
       segment is unassigned. IoU plays no part in the assignment -- it decides
       how well a prediction did, not what it was attempting.
    2. **Suppress duplicates.** Several predictions can be assigned to one ground-truth
       segment; the one with the largest intersection represents it and the rest
       are duplicate false positives. This is COCO's rule, and it is what stops
       over-segmentation from being free.
    3. **Classify.** A representative is a true positive if *its own* IoU clears
       the threshold, otherwise a false positive. Gold segments with no true
       positive are false negatives.

    So FP = n_pred - TP (unassigned + duplicates + representatives that missed the
    threshold) and FN = n_gt - TP, both charged through the precision and recall
    denominators.

    COCO orders candidates by confidence, which is also what breaks its duplicate
    ties. Segments carry no confidence, so intersection serves both roles; IoU is
    reserved for the classification step, which is the property that matters --
    the threshold never influences the pairing, so true-positive counts are
    monotone in it and one pairing serves every threshold.

    Returns ``{threshold: (tp, n_pred, n_gt)}`` plus ``mean_matched_iou``.
    """
    ps = _segments_from(pred_bounds, gt["end_sec"], gt.get("ignore"))
    gs = gt["segments"]
    if not ps or not gs:
        empty: dict[Any, Any] = {t: (0, len(ps), len(gs)) for t in IOU_THRESHOLDS}
        empty["mean_matched_iou"] = float("nan")
        return empty

    a = np.asarray([s[0] for s in ps], dtype=np.float64)[:, None]
    b = np.asarray([s[1] for s in ps], dtype=np.float64)[:, None]
    c = np.asarray([s[0] for s in gs], dtype=np.float64)[None, :]
    d = np.asarray([s[1] for s in gs], dtype=np.float64)[None, :]
    inter = np.clip(np.minimum(b, d) - np.maximum(a, c), 0.0, None)
    union = np.maximum(b, d) - np.minimum(a, c)
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

    # 1. assign by intersection; -1 means the prediction overlaps nothing
    best_g = inter.argmax(axis=1)
    assigned = np.where(inter[np.arange(inter.shape[0]), best_g] > 0, best_g, -1)

    # 2. one representative per ground-truth segment, by largest intersection
    rep_iou: list[float] = []
    for g_idx in range(len(gs)):
        cand = np.nonzero(assigned == g_idx)[0]
        if cand.size == 0:
            continue
        rep = cand[int(np.argmax(inter[cand, g_idx]))]
        rep_iou.append(float(iou[rep, g_idx]))

    # 3. classify each representative against every threshold
    reps = np.asarray(rep_iou, dtype=np.float64)
    out: dict[Any, Any] = {t: (int((reps >= t).sum()), len(ps), len(gs)) for t in IOU_THRESHOLDS}
    out["mean_matched_iou"] = float(reps.mean()) if reps.size else float("nan")
    return out


def label_agreement(
    labels: np.ndarray,
    windows: Any,
    gt: dict[int, dict[str, Any]],
    episodes: list[int],
) -> dict[str, float]:
    """Frame-wise ARI and NMI between predicted window labels and ground-truth subtask identity.

    Boundary metrics say *when* behaviour changed; these say whether the graph's
    states correspond to the annotated subtasks at all. On a task that is ~17
    repetitions of pick-then-place, a method can score well on boundaries while
    its clusters carry no subtask meaning, and only this separates the two.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    want = set(episodes)
    pred_frames: list[np.ndarray] = []
    gt_frames: list[np.ndarray] = []
    for i, w in enumerate(windows.windows):
        ep = w.episode_idx
        if ep not in want or ep not in gt:
            continue
        lab = int(labels[i])
        if lab == -1:
            continue
        g = gt[ep]["frame_labels"]
        a, b = int(w.start), min(int(w.end), g.size)
        if b <= a:
            continue
        seg = g[a:b]
        ok = seg >= 0
        if not ok.any():
            continue
        pred_frames.append(np.full(int(ok.sum()), lab, dtype=np.int64))
        gt_frames.append(seg[ok].astype(np.int64))
    if not pred_frames:
        return {"ari": float("nan"), "nmi": float("nan"), "n_frames": 0}
    p = np.concatenate(pred_frames)
    q = np.concatenate(gt_frames)
    return {
        "ari": float(adjusted_rand_score(q, p)),
        "nmi": float(normalized_mutual_info_score(q, p)),
        "n_frames": int(p.size),
    }


def score_boundaries(
    preds: dict[int, np.ndarray],
    gt: dict[int, dict[str, Any]],
    episodes: list[int],
    tol: float,
    min_segment: float = 0.0,
    density_band: tuple[float, float] = (0.8, 1.5),
    with_segmental: bool = True,
) -> dict[str, Any]:
    """Boundary + segmental metrics over ``episodes``, at one tolerance and min_segment.

    ``episodes`` is a LIST and may repeat an index -- a paired bootstrap resamples
    with replacement -- so everything accumulates over the list rather than being
    keyed by episode.
    """
    tp = tp_u = n_pred = n_gt = 0
    per_ep_f1: list[float] = []
    seg_hits = {t: 0 for t in IOU_THRESHOLDS}
    seg_np = {t: 0 for t in IOU_THRESHOLDS}
    seg_ng = {t: 0 for t in IOU_THRESHOLDS}
    # The same count-matched uniform reference the boundary metric uses, scored
    # segmentally. Needed because a trivial predictor is *good* on this task:
    # equally spaced boundaries at the method's own count reach segF1@0.25 ~0.87
    # and segF1@0.5 ~0.55, so a raw segmental F1 says almost nothing at those
    # thresholds. At IoU 0.75 the reference is only ~0.13 and there is real
    # headroom, which makes segmental_gain@0.75 the discriminating extent metric.
    segu_hits = {t: 0 for t in IOU_THRESHOLDS}
    segu_np = {t: 0 for t in IOU_THRESHOLDS}
    # Threshold-free summary of pairing quality, and its count-matched reference.
    miou: list[float] = []
    miou_u: list[float] = []

    for i in episodes:
        g = gt[i]
        p = drop_ignored(np.asarray(preds.get(i, np.empty(0)), dtype=np.float64), g["ignore"])
        p = merge_short(p, g["end_sec"], min_segment)
        gb = g["boundaries"]
        h = match_count(p, gb, tol)
        hu = match_count(uniform_boundaries(g["end_sec"], p.size), gb, tol)
        tp += h
        tp_u += hu
        n_pred += int(p.size)
        n_gt += int(gb.size)
        per_ep_f1.append(_f1(h, int(p.size), int(gb.size)))
        if with_segmental:
            # Segment extents do not depend on the boundary-match tolerance, so a
            # caller sweeping tolerances should ask for this once. It dominates
            # the cost of this function.
            sc_p = segmental_counts(p, g)
            miou.append(sc_p.pop("mean_matched_iou"))
            for t, (hh, npr, ngo) in sc_p.items():
                seg_hits[t] += hh
                seg_np[t] += npr
                seg_ng[t] += ngo
            pu = uniform_boundaries(g["end_sec"], int(p.size))
            sc_u = segmental_counts(pu, g)
            miou_u.append(sc_u.pop("mean_matched_iou"))
            for t, (hh, npr, _) in sc_u.items():
                segu_hits[t] += hh
                segu_np[t] += npr

    f1 = _f1(tp, n_pred, n_gt)
    f1_u = _f1(tp_u, n_pred, n_gt)
    density = n_pred / n_gt if n_gt else float("nan")
    out: dict[str, Any] = {
        "tolerance": tol,
        "min_segment": min_segment,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "density": density,
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
        "f1": f1,
        "uniform_f1": f1_u,
        "gain": f1 - f1_u,
        # Macro: episodes range 19-56 segments here, so a pooled F1 weights the
        # longest episode ~2.9x the shortest.
        "macro_f1": float(np.mean(per_ep_f1)) if per_ep_f1 else 0.0,
        "in_band": bool(density_band[0] <= density <= density_band[1]),
    }
    if with_segmental:
        for t in IOU_THRESHOLDS:
            s = _f1(seg_hits[t], seg_np[t], seg_ng[t])
            su = _f1(segu_hits[t], segu_np[t], seg_ng[t])
            out[f"segmental_f1@{t}"] = s
            out[f"segmental_uniform_f1@{t}"] = su
            out[f"segmental_gain@{t}"] = s - su
        good = [v for v in miou if np.isfinite(v)]
        good_u = [v for v in miou_u if np.isfinite(v)]
        out["mean_matched_iou"] = float(np.mean(good)) if good else float("nan")
        out["mean_matched_iou_uniform"] = float(np.mean(good_u)) if good_u else float("nan")
        out["mean_matched_iou_gain"] = (
            out["mean_matched_iou"] - out["mean_matched_iou_uniform"]
            if good and good_u
            else float("nan")
        )
    return out
