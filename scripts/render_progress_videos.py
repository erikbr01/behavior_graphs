"""Render progress-estimation result videos: play each episode's frames next to a
live progress plot that sweeps in time, overlaying every method's predicted
progress (graph-stage + the VLM baselines) against the linear GT (successes).

For each selected episode we composite [robot frame | progress plot with a moving
time cursor] and write an mp4. Successful and failed episodes are rendered into
separate folders — on failures there is no linear GT (the task never completes),
so we drop the diagonal and it's visible whether each signal correctly plateaus.

CPU only. Reads the saved pool500 curve npzs (idx-keyed) via the shared io loader
+ method registry (colors/infixes), and the episode mp4s.

    PYTHONPATH=src python scripts/render_progress_videos.py \
        --suite lg_pool500 --rollout-dir ${DATA_PATH}/rollouts/libero_goal \
        --n 10 --out-dir analysis/out/progress_videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from behavior_graphs.progress_estimation.io import (  # noqa: E402
    METHOD_REGISTRY,
    load_curves,
    read_frames,
    resolve_curve_path,
)

# Methods overlaid in the videos (registry keys); graph-absorb intentionally omitted.
RENDER_METHODS = ["graph_stage", "topreward_ll", "robometer_4b", "robometer_fft"]


def load_method_curves(curves_dir, suite):
    """display_label -> {idx: curve}; plus {idx: flag}."""
    out, flags = {}, {}
    for m in RENDER_METHODS:
        p = resolve_curve_path(curves_dir, suite, m)
        if p is None:
            continue
        c = load_curves(p)
        label = METHOD_REGISTRY[m].display
        out[label] = {
            int(i): np.clip(np.asarray(cv, float), 0, 1) for i, cv in zip(c.idx, c.curves)
        }
        for i, fl in zip(c.idx, c.flags):
            flags[int(i)] = bool(fl)
    return out, flags


def _colors():
    return {METHOD_REGISTRY[m].display: METHOD_REGISTRY[m].color for m in RENDER_METHODS}


def to_frame_timeline(curve, T):
    """Interpolate a per-window curve onto T frame positions in [0,1]."""
    L = len(curve)
    if L < 2:
        return np.full(T, float(curve[0]) if L else 0.0)
    return np.interp(np.linspace(0, 1, T), np.linspace(0, 1, L), curve)


def render_episode(mp4, curves_by_method, success, out_path, fps=15, height=560):
    frames = read_frames(mp4, rgb=False)  # BGR for cv2 compositing
    T = len(frames)
    if T < 3:
        return False
    h0, w0 = frames[0].shape[:2]
    H = height
    rw = int(round(w0 * H / h0))
    robot = [cv2.resize(f, (rw, H), interpolation=cv2.INTER_CUBIC) for f in frames]
    series = {m: to_frame_timeline(c, T) for m, c in curves_by_method.items()}
    tn = np.linspace(0, 1, T)

    dpi = 100
    pw = int(H * 1.55)
    fig = plt.figure(figsize=(pw / dpi, H / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    colors = _colors()
    lines = []
    for m, ys in series.items():
        lines.append(ax.plot(tn, ys, color=colors.get(m, "gray"), lw=2.6, label=m)[0])
    if success:
        lines.append(ax.plot([0, 1], [0, 1], "k--", lw=1.6, alpha=0.6, label="linear GT")[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("episode time", fontsize=15)
    ax.set_ylabel("predicted progress", fontsize=15)
    ax.set_title(("SUCCESS" if success else "FAILURE") + f"  ·  {out_path.stem}", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(lines),
        fontsize=12,
        framealpha=0.9,
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.24)
    cursor = ax.axvline(0, color="k", lw=2.0)
    dots = {
        m: ax.plot([0], [series[m][0]], "o", color=colors.get(m, "gray"), ms=9)[0] for m in series
    }

    vw = None
    for t in range(T):
        cursor.set_xdata([tn[t], tn[t]])
        for m in series:
            dots[m].set_data([tn[t]], [series[m][t]])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plot_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        plot_bgr = cv2.resize(plot_bgr, (int(H * plot_bgr.shape[1] / plot_bgr.shape[0]), H))
        panel = cv2.hconcat([robot[t], plot_bgr])
        if vw is None:
            vw = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (panel.shape[1], panel.shape[0]),
            )
        vw.write(panel)
    vw.release()
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", type=Path, default=Path("analysis/out/progress_bench_results"))
    ap.add_argument("--suite", default="lg_pool500")
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/out/progress_videos"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument(
        "--height", type=int, default=560, help="output panel height (robot view upscaled)"
    )
    args = ap.parse_args()

    curves, flags = load_method_curves(args.curves_dir, args.suite)
    if not curves:
        raise SystemExit(f"no curves for suite {args.suite} in {args.curves_dir}")
    media = args.rollout_dir / "media"
    succ = sorted(i for i, f in flags.items() if f)[: args.n]
    fail = sorted(i for i, f in flags.items() if not f)[: args.n]
    print(f"selected {len(succ)} success + {len(fail)} failure episodes")

    for group, idxs in [("success", succ), ("failure", fail)]:
        outg = args.out_dir / group
        outg.mkdir(parents=True, exist_ok=True)
        for i in idxs:
            tag = "succ" if flags[i] else "fail"
            mp4 = media / f"ep{i:04d}_{tag}.mp4"
            if not mp4.exists():
                cand = list(media.glob(f"ep{i:04d}_*.mp4"))
                mp4 = cand[0] if cand else None
            if mp4 is None:
                print(f"  ep{i:04d}: no mp4, skip")
                continue
            cbm = {m: curves[m][i] for m in curves if i in curves[m]}
            out_path = outg / f"{args.suite}_ep{i:04d}_{tag}.mp4"
            ok = render_episode(mp4, cbm, flags[i], out_path, fps=args.fps, height=args.height)
            print(f"  {'wrote' if ok else 'skip '} {out_path}")
    print(f"done -> {args.out_dir}/{{success,failure}}/")


if __name__ == "__main__":
    main()
