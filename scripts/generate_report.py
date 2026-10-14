"""Generate a self-contained HTML report for a labeled behavior graph run.

Usage:
    python scripts/generate_report.py <run_dir> [options]

Example:
    python scripts/generate_report.py \\
        deploy/data/runs/transport_mh_jan28__agg_first__infembed_w3_s1_seed0_kmeans_k10 \\
        --rollout-dir deploy/data/rollouts/transport_mh_jan28 \\
        --out /tmp/report.html
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------


def _conf_badge(conf: float | None) -> str:
    if conf is None:
        return ""
    pct = int(conf * 100)
    color = "#22c55e" if pct >= 85 else "#f59e0b" if pct >= 65 else "#ef4444"
    return f'<span class="badge" style="background:{color}">{pct}%</span>'


def _card(
    nid: int,
    label: dict,
    sheet_b64: str | None,
    progress: tuple | None,
    preds: list[int],
    succs: list[int],
    node_labels: dict[int, dict],
) -> str:
    name = label.get("label", f"Behavior {nid}")
    desc = label.get("description", "")
    conf = label.get("confidence")
    src = label.get("source", "")
    reas = label.get("reasoning", "")

    prog_str = ""
    if progress:
        lo, hi = int(progress[0] * 100), int(progress[1] * 100)
        prog_str = f"{lo}–{hi}% through episodes"

    def node_chip(nid2: int) -> str:
        lbl = node_labels.get(nid2, {}).get("label")
        text = f"{lbl} ({nid2})" if lbl else f"Behavior {nid2}"
        return f'<span class="chip">{text}</span>'

    preds_html = " ".join(node_chip(p) for p in preds) or '<span class="none">—</span>'
    succs_html = " ".join(node_chip(s) for s in succs) or '<span class="none">—</span>'

    img_html = ""
    if sheet_b64:
        img_html = f'<img src="data:image/jpeg;base64,{sheet_b64}" alt="contact sheet node {nid}">'
    else:
        img_html = '<div class="no-img">no frames</div>'

    reas_html = ""
    if reas:
        reas_html = f"""
        <details>
          <summary>Reasoning</summary>
          <p class="reasoning">{reas}</p>
        </details>"""

    return f"""
    <div class="card" id="node-{nid}">
      <div class="card-img">{img_html}</div>
      <div class="card-body">
        <div class="card-header">
          <span class="node-id">Behavior {nid}</span>
          {_conf_badge(conf)}
        </div>
        <h2>{name}</h2>
        <p class="desc">{desc}</p>
        <div class="meta">
          {'<span class="meta-item">⏱ ' + prog_str + "</span>" if prog_str else ""}
          {'<span class="meta-item">🤖 ' + src + "</span>" if src else ""}
        </div>
        <div class="flow">
          <div class="flow-row"><span class="flow-label">From</span>{preds_html}</div>
          <div class="flow-row"><span class="flow-label">To</span>{succs_html}</div>
        </div>
        {reas_html}
      </div>
    </div>"""


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #0f172a; color: #e2e8f0; padding: 2rem; }
h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
.subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(680px, 1fr)); gap: 1.25rem; }
.card { background: #1e293b; border-radius: 12px; overflow: hidden;
        display: flex; flex-direction: row; border: 1px solid #334155; }
.card-img { flex: 0 0 260px; background: #0f172a; display: flex; align-items: center;
            justify-content: center; padding: 0.5rem; }
.card-img img { width: 100%; border-radius: 6px; display: block; }
.no-img { color: #475569; font-size: 0.8rem; padding: 1rem; text-align: center; }
.card-body { flex: 1; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.6rem;
             min-width: 0; }
.card-header { display: flex; align-items: center; gap: 0.5rem; }
.node-id { font-size: 0.75rem; font-weight: 600; color: #64748b;
           text-transform: uppercase; letter-spacing: 0.05em; }
.badge { font-size: 0.7rem; font-weight: 700; color: #fff; padding: 0.15rem 0.45rem;
         border-radius: 999px; }
h2 { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; line-height: 1.3; }
.desc { color: #94a3b8; font-size: 0.875rem; line-height: 1.55; }
.meta { display: flex; gap: 1rem; flex-wrap: wrap; }
.meta-item { font-size: 0.75rem; color: #64748b; }
.flow { display: flex; flex-direction: column; gap: 0.3rem; margin-top: 0.25rem; }
.flow-row { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; }
.flow-label { font-size: 0.7rem; font-weight: 600; color: #475569;
              text-transform: uppercase; letter-spacing: 0.05em; min-width: 2rem; }
.chip { font-size: 0.72rem; background: #0f172a; border: 1px solid #334155;
        color: #94a3b8; padding: 0.15rem 0.5rem; border-radius: 6px; white-space: nowrap; }
.none { color: #334155; font-size: 0.75rem; }
details { margin-top: 0.25rem; }
summary { font-size: 0.75rem; color: #64748b; cursor: pointer; user-select: none; }
summary:hover { color: #94a3b8; }
.reasoning { font-size: 0.8rem; color: #64748b; line-height: 1.55;
             margin-top: 0.4rem; white-space: pre-wrap; }
"""


def render_html(
    run_dir: Path, rollout_dir: Path | None, frames_per_episode: int, max_episodes: int
) -> str:
    from behavior_graphs.core.graph import START_NODE_ID, TERMINAL_NODE_IDS
    from behavior_graphs.core.node_labels import load_node_labels
    from behavior_graphs.core.types import node_progress_range
    from behavior_graphs.pipeline.artifacts import RunArtifacts
    from behavior_graphs.stages.labeling.vlm_labeler import _neighbor_ids

    artifacts = RunArtifacts(run_dir)
    graph = artifacts.load_graph()
    windows = artifacts.load_windows()
    labels_arr = artifacts.load_array("labels")
    node_labels = load_node_labels(run_dir)
    metrics = artifacts.load_metrics()

    run_name = run_dir.name

    # Infer rollout_dir from metrics if not given
    if rollout_dir is None:
        rd_str = metrics.get("rollout_dir")
        rollout_dir = Path(rd_str) if rd_str else None

    behavior_nodes = sorted(graph.cluster_nodes)

    # Compute episode-relative position once per node, for ordering and cards.
    prog_by_nid = {
        nid: node_progress_range(nid, labels_arr, windows.windows, windows.episodes)
        for nid in behavior_nodes
    }

    def _midpoint(p: tuple[float, float] | None) -> float:
        return (p[0] + p[1]) / 2 if p else 0.5

    behavior_nodes.sort(key=lambda nid: _midpoint(prog_by_nid[nid]))

    cards = []
    for nid in behavior_nodes:
        label = node_labels.get(nid, {})
        prog = prog_by_nid[nid]
        preds, succs = _neighbor_ids(graph, nid)
        preds = [p for p in preds if p not in TERMINAL_NODE_IDS and p != START_NODE_ID]
        succs = [s for s in succs if s not in TERMINAL_NODE_IDS and s != START_NODE_ID]

        sheet_b64 = None
        if rollout_dir and rollout_dir.is_dir():
            try:
                from behavior_graphs.viz.frames import contact_sheet_for_node

                jpeg = contact_sheet_for_node(
                    run_dir,
                    rollout_dir,
                    nid,
                    frames_per_episode=frames_per_episode,
                    max_episodes=max_episodes,
                )
                sheet_b64 = base64.standard_b64encode(jpeg).decode()
            except Exception as e:
                print(f"  Warning: could not generate sheet for node {nid}: {e}", file=sys.stderr)

        cards.append(_card(nid, label, sheet_b64, prog, preds, succs, node_labels))

    labeled_count = len(node_labels)
    subtitle = (
        f"{run_name} &nbsp;·&nbsp; "
        f"{labeled_count}/{len(behavior_nodes)} nodes labeled &nbsp;·&nbsp; "
        f"{graph.num_episodes} episodes"
    )

    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Behavior Graph — {run_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Behavior Graph Report</h1>
  <p class="subtitle">{subtitle}</p>
  <div class="grid">
    {cards_html}
  </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rollout-dir", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="Output path (default: <run_dir>/report.html)"
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=5,
        help="Frames sampled from each episode row of a node's contact sheet",
    )
    parser.add_argument("--max-episodes", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out = args.out or run_dir / "report.html"

    print(f"Generating report for {run_dir.name}...")
    html = render_html(run_dir, args.rollout_dir, args.frames_per_episode, args.max_episodes)
    out.write_text(html)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
