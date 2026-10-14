"""Structural statistics of the behavior graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator


class GraphStatsEvaluator(Evaluator):
    """Node/edge counts, success rate, branching statistics, and top paths."""

    name = "graph_stats"

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        graph = result.graph
        cluster_nodes = graph.cluster_nodes

        out_degrees = []
        entropies = []
        for nid in cluster_nodes:
            probs = list(graph.transition_probs.get(nid, {}).values())
            out_degrees.append(len(probs))
            if probs:
                p = np.asarray(probs)
                entropies.append(float(-(p * np.log(p)).sum()))

        successes = [ep.success for ep in result.windows.episodes]
        known = [s for s in successes if s is not None]
        success_rate = float(np.mean([s is True for s in known])) if known else None

        top_paths = [
            {"path": [int(n) for n in path], "probability": float(prob)}
            for path, prob, _ in graph.enumerate_paths(max_paths=5)
        ]
        return {
            "num_cluster_nodes": len(cluster_nodes),
            "num_edges": sum(len(t) for t in graph.transition_probs.values()),
            "num_episodes": graph.num_episodes,
            "success_rate": success_rate,
            "mean_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "mean_transition_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "top_paths": top_paths,
        }
