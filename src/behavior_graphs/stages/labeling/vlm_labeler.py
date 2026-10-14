"""Two-pass VLM node labeler with pluggable backend adapters.

Implements the seeded-relabeling pipeline (Macrodata Labs) combined with
reasoning-before-estimation (ProcVLM) and episode-relative progress context
(WARP-RM).  Labels are written to ``<run_dir>/node_labels.json`` (the
canonical, editable artifact); only a small summary is returned for the
metrics dict.

The model string selects the backend automatically via ``make_vlm_client()``:

    claude-opus-4-8          → Anthropic   (pip install anthropic)
    gemini-2.0-flash         → Gemini      (pip install google-genai)
    ollama:llava             → Ollama      (pip install openai; Ollama running)
    ollama:llava@<url>       → Ollama at a custom URL
    openai:<model>@<url>     → any OpenAI-compat endpoint (LM Studio, vLLM, …)
    gpt-4o                   → OpenAI      (pip install openai)
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.graph import (
    FAILURE_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
    node_display_name,
)
from behavior_graphs.core.graph_analysis import save_graph_analysis
from behavior_graphs.core.node_labels import load_node_labels, make_label_entry, upsert_node_labels
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import NodeLabeler
from behavior_graphs.core.types import node_progress_range
from behavior_graphs.pipeline.artifacts import resolve_rollout_dir
from behavior_graphs.vlm.adapters import VLMClient, make_vlm_client

log = logging.getLogger(__name__)


def _sheet_blocks(caption: str | None, b64: str) -> list[dict]:
    """Content blocks for one contact sheet: an optional caption plus the image."""
    blocks: list[dict] = []
    if caption:
        blocks.append({"type": "text", "text": caption})
    blocks.append(
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
    )
    return blocks


def _parse_vlm_json(text: str, context: str = "") -> dict:
    """Parse a VLM JSON response, stripping an optional markdown code fence.

    Re-raises as ``ValueError`` carrying *context* and the raw response: a bare
    ``JSONDecodeError`` surfaces in the UI as "Expecting value: line 1 column 1
    (char 0)", which says nothing about which call produced what.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
    where = f" for {context}" if context else ""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"VLM returned unparseable JSON{where} ({exc}). Raw response: {text[:400]!r}"
        ) from exc
    # Valid JSON that isn't an object is just as unusable, and models do wrap
    # the object in a list. Caught here so callers fail with the raw response
    # rather than an AttributeError several steps later.
    if not isinstance(parsed, dict):
        raise ValueError(
            f"VLM returned {type(parsed).__name__}, not a JSON object{where}. "
            f"Raw response: {text[:400]!r}"
        )
    return parsed


def _graph_summary(graph: BehaviorGraph, node_values: dict[int, float] | None) -> str:
    """Build a compact text description of the graph for VLM context."""
    lines = [
        f"Behavior graph: {graph.num_episodes} total episodes, "
        f"{len(graph.cluster_nodes)} behavior clusters.",
        "",
        "Nodes:",
    ]
    for nid, node in sorted(graph.nodes.items()):
        val_str = ""
        if node_values and nid in node_values:
            val_str = f"  value={node_values[nid]:.2f}"
        lines.append(
            f"  {node_display_name(nid)} (id={nid}): "
            f"{node.num_windows} windows across {node.num_episodes} episodes{val_str}"
        )
    lines.append("")
    lines.append("Transitions (probability > 0.05):")
    for src, probs in sorted(graph.transition_probs.items()):
        for dst, prob in sorted(probs.items(), key=lambda x: -x[1]):
            if prob >= 0.05:
                lines.append(f"  {node_display_name(src)} → {node_display_name(dst)}: {prob:.0%}")
    return "\n".join(lines)


def _progress_desc(progress: tuple[float, float] | None) -> str:
    if progress is None:
        return "unknown"
    lo, hi = progress
    pct_lo = int(lo * 100)
    pct_hi = int(hi * 100)
    if pct_lo == pct_hi:
        return f"~{pct_lo}% through episodes"
    return f"{pct_lo}–{pct_hi}% through episodes"


def _views_hint(cameras: Sequence[str] | None) -> str:
    """Prompt text explaining a multi-view sheet, or nothing for one view.

    Load-bearing, not decoration. A two-camera sheet has two labelled
    row-groups per episode covering *the same timesteps* from different angles;
    with nothing said about that, the model reads the second group as a later
    phase of the episode and describes a manipulation sequence that never
    happened. The row labels carry the view names, so naming them here is what
    ties the two together.
    """
    if not cameras or len(cameras) < 2:
        return ""
    names = ", ".join(cameras)
    return (
        f"\nEach episode appears once per camera view ({names}) — one labelled "
        "row-group each, sampled at the SAME timesteps. Rows of the same episode "
        "are simultaneous views of one moment from different angles, not "
        "successive phases; read them together and do not describe a view change "
        "as something the robot did.\n"
    )


def _neighbor_ids(
    graph: BehaviorGraph,
    node_id: int,
) -> tuple[list[int], list[int]]:
    """Return (predecessor_ids, successor_ids) for *node_id*."""
    preds = [
        src for src, _, _ in graph.get_incoming_transitions(node_id) if src not in TERMINAL_NODE_IDS
    ]
    succs = [tgt for tgt, _, _ in graph.get_outgoing_transitions(node_id)]
    return preds, succs


class VLMLabeler(NodeLabeler):
    """Auto-label behavior graph nodes using a VLM with two-pass seeded relabeling.

    Pass 1: generate draft labels with reasoning from contact-sheet images and
    graph context.  Pass 2: refine each draft label with the VLM seeing its
    own prior output as additional context (seeded relabeling per Macrodata
    Labs — improves accuracy 61% → 78%).

    Args:
        model: Model string — selects backend automatically. Examples:
            ``claude-opus-4-8``, ``gemini-2.0-flash``, ``ollama:llava``,
            ``openai:Qwen/Qwen2-VL-7B@http://localhost:1234/v1``.
        frames_per_episode: Frames sampled from each episode that visits a
            node. Every episode row gets the same count, so a node seen in
            more episodes gets a taller sheet rather than thinner coverage.
        max_episodes_per_node: Max episodes to sample frames from per node.
        tile_px: Shorter side of each contact-sheet tile. Raise it for
            higher-resolution footage (cooking is 640x360, some sim 1280x720);
            libero is already 224 and mimicgen only 128, so it buys nothing
            there. Image tokens scale with tile area.
        task_description: Optional free-text hint about the task (e.g.
            "pick up the cup and place it on the plate").  Shown to the VLM
            as additional context.
        client: Pre-built VLMClient. If provided, ``model`` is used only for
            the output metadata (the client's own model is used for requests).
        cameras: Camera views to build every contact sheet from, in the order
            given (see
            :meth:`~behavior_graphs.envs.trajectory_dir.TrajectoryDirectory.camera_names`).
            ``None`` uses one view per episode, which is what a single-camera
            dataset has anyway. A wrist view is often what settles whether an
            exterior view's ambiguous approach was a grasp, but each extra view
            multiplies image tokens per node by roughly one — the sheets are
            re-sent in both passes — so this is opt-in.
        view_transforms: ``{camera: {rotate, mirror}}`` display geometry for
            those views, from
            :func:`behavior_graphs.viz.streamlit.data.camera_transforms`.
            Resolved by the entry point rather than here: it comes from
            ``configs/data/``, which is the app layer's to read, and a stage
            takes its inputs as arguments.
    """

    name = "vlm_labeler"

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        frames_per_episode: int = 5,
        max_episodes_per_node: int = 4,
        task_description: str = "",
        client: VLMClient | None = None,
        data_path: str | None = None,
        thinking_level: str | None = None,
        tile_px: int = 224,
        cameras: Sequence[str] | None = None,
        view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.model = model
        self.frames_per_episode = frames_per_episode
        self.max_episodes_per_node = max_episodes_per_node
        self.task_description = task_description
        self._client = client
        self.data_path = data_path
        self.thinking_level = thinking_level
        self.tile_px = tile_px
        # An empty list means "no views", which no caller can want and which
        # would silently produce text-only labels; read it as the default.
        self.cameras = list(cameras) if cameras else None
        self.view_transforms = dict(view_transforms or {})
        #: Sheets already built this run, keyed by (run, rollout dir, node).
        self._sheets: dict[tuple[str, str, int], str | None] = {}
        #: Resolved (transforms, primary view) per rollout dir — constant per run,
        #: and every node's sheet would otherwise re-resolve it.
        self._view_configs: dict[str, tuple[Mapping[str, Mapping[str, object]], str | None]] = {}

    @property
    def _multi_view(self) -> bool:
        """Whether a sheet holds several views — the one place that is decided."""
        return bool(self.cameras) and len(self.cameras) > 1

    def _view_config(
        self, rollout_dir: Path
    ) -> tuple[Mapping[str, Mapping[str, object]], str | None]:
        """``(transforms, primary view)`` for this dataset, from ``configs/data/``.

        A caller's explicit *view_transforms* wins — the app resolved them for the
        group it is displaying, and the sheets have to match its players. Resolved
        here otherwise because Hydra can name ``cameras:`` in
        ``configs/pipeline/labeling/vlm.yaml`` but has nowhere to put a transform
        map, so a ``labeling.cameras=[…,wrist_image_left]`` run would otherwise
        send every wrist frame upside down.

        Two caveats worth knowing, both inherited from the display layer:
        ``camera_transforms`` reads the group named by
        ``$BEHAVIOR_GRAPHS_DISPLAY_DATA``, so a dataset outside that group resolves
        to no geometry at all; and without the viz extra there is nothing to read.
        Both degrade to "as recorded" rather than failing the run, and both are
        logged. Ultimately this belongs in a non-UI config module the stage and the
        app can share — see ``docs/vlm_annotations.md``.
        """
        if str(rollout_dir) in self._view_configs:
            return self._view_configs[str(rollout_dir)]
        resolved = self._read_view_config(rollout_dir)
        self._view_configs[str(rollout_dir)] = resolved
        return resolved

    def _read_view_config(
        self, rollout_dir: Path
    ) -> tuple[Mapping[str, Mapping[str, object]], str | None]:
        try:
            from behavior_graphs.viz.streamlit.data import camera_transforms, primary_camera
        except ImportError:
            log.warning(
                "behavior_graphs.viz is not installed, so configured camera geometry "
                "cannot be read: sheets will show every view as recorded"
            )
            return self.view_transforms, None
        dataset = Path(rollout_dir).name
        transforms = self.view_transforms or camera_transforms(dataset)
        if self.cameras and not transforms:
            log.info(
                "no camera geometry configured for %s in the display group "
                "($BEHAVIOR_GRAPHS_DISPLAY_DATA); views go on the sheet as recorded",
                dataset,
            )
        return transforms, primary_camera(dataset)

    def _sheet_b64(self, run_dir: Path, rollout_dir: Path, node_id: int) -> str | None:
        """This node's contact sheet, built once per run and remembered.

        ``analyze()`` asks for the same sheets ``label()`` just built, from
        identical inputs — every sampled frame of every view decoded, resized and
        composited a second time, per node. Cached on the labeler because that is
        the object whose lifetime is the run.
        """
        key = (str(run_dir), str(rollout_dir), int(node_id))
        if key in self._sheets:
            return self._sheets[key]
        transforms, primary = self._view_config(rollout_dir)
        try:
            from behavior_graphs.viz.frames import contact_sheet_for_node

            jpeg = contact_sheet_for_node(
                run_dir=run_dir,
                rollout_dir=rollout_dir,
                node_id=node_id,
                frames_per_episode=self.frames_per_episode,
                max_episodes=self.max_episodes_per_node,
                tile_px=self.tile_px,
                cameras=self.cameras,
                view_transforms=transforms,
                primary_camera=primary,
            )
            sheet = base64.standard_b64encode(jpeg).decode()
        except Exception:
            # Not cached: a transient failure (a pyav hiccup, a clip rewritten
            # under a live server) should not decide that this node has no sheet
            # for the rest of the run, and the reason belongs in the log — a node
            # labeled from text alone otherwise looks like a node with no frames.
            log.warning("contact sheet for node %s failed to build", node_id, exc_info=True)
            return None
        self._sheets[key] = sheet
        return sheet

    def _check_cameras(self, rollout_dir: Path) -> None:
        """Fail on a camera view the dataset does not have, before paying for it.

        ``scripts/label_nodes.py`` validates its own ``--camera`` flags, but the
        other two entry points — Hydra's ``labeling.cameras=[…]`` and
        :func:`behavior_graphs.chat.annotate.annotate_nodes` — hand names straight
        here, and a name no view matches is not inert: every episode resolves to no
        clip, so each node's sheet is the blank placeholder
        :func:`~behavior_graphs.viz.frames.make_contact_sheet` returns for no frames.
        That is a *valid* JPEG, so it reads as a real sheet all the way to the API,
        and the run pays full price to label black squares while the prompt still
        claims both views are there. Easy name to get wrong, too: DROID collections
        disagree on the wrist view's spelling, which is why
        ``configs/data/droid.yaml`` carries both.
        """
        if not self.cameras:
            return
        from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

        available = TrajectoryDirectory(rollout_dir).camera_names()
        unknown = [c for c in self.cameras if c not in available]
        if unknown:
            raise ValueError(
                f"no such camera view(s) in {rollout_dir}: {', '.join(unknown)}. "
                f"Available: {', '.join(available) or '(none — no media/ clips)'}"
            )

    def _fallback_rollout_dir(self, result: PipelineResult) -> Path:
        """Recorded rollout dir resolved against the data root.

        Without it the VLM labels nodes with no frames at all, which is not
        useful, so a failed resolution is an error rather than a fallback to
        text-only labeling.
        """
        raw = result.metrics.get("rollout_dir")
        resolved = resolve_rollout_dir(raw, self.data_path)
        if resolved is None or not resolved.is_dir():
            raise FileNotFoundError(
                f"rollout_dir {raw!r} resolves to {resolved} which does not "
                "exist — cannot extract frames for VLM labeling. Recorded "
                "paths are relative to the data root; pass data_path (or set "
                "$DATA_PATH)."
            )
        return resolved

    def _infer_task(
        self,
        result: PipelineResult,
        rollout_dir: Path,
        client: VLMClient,
        n_episodes: int = 3,
        frames_per_episode: int = 20,
    ) -> str | None:
        """Infer the task description from full contact sheets of successful episodes."""
        from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory
        from behavior_graphs.viz.frames import (
            episode_views,
            extract_episode_frames,
            make_contact_sheet,
        )

        if result.windows is None:
            return None

        successful_eps = [ep.index for ep in result.windows.episodes if ep.success is True]
        if not successful_eps:
            return None

        transforms, primary = self._view_config(rollout_dir)
        traj = TrajectoryDirectory(rollout_dir, primary_camera=primary)
        content: list[Any] = []
        for ep_idx in successful_eps[:n_episodes]:
            ep_info = result.windows.episodes[ep_idx]
            length = ep_info.num_samples
            indices = [int(length * i / frames_per_episode) for i in range(frames_per_episode)]
            # One sheet per view rather than per episode, each captioned with its
            # camera: the sheets go in a single message, so the model still sees
            # the whole episode from every angle, and no sheet mixes views —
            # which is what would make "the sequence" ambiguous here.
            for camera, video_path in episode_views(traj, ep_idx, self.cameras):
                frames = extract_episode_frames(
                    video_path,
                    indices,
                    tile_px=self.tile_px,
                    transform=transforms.get(camera),
                )
                if not frames:
                    continue
                sheet = make_contact_sheet(frames)
                b64 = base64.standard_b64encode(sheet).decode()
                view = f" — {camera} view" if self._multi_view else ""
                content += _sheet_blocks(f"[Successful episode {ep_idx}{view}]", b64)

        if not content:
            return None

        content.append(
            {
                "type": "text",
                "text": (
                    "These contact sheets show complete successful episodes of a robot task. "
                    "Each sheet spans the full episode from start to finish.\n"
                    + (
                        "An episode captioned with several camera views appears once per "
                        "view — the same episode from different angles, not several "
                        "episodes.\n"
                        if self._multi_view
                        else ""
                    )
                    + "\nIn 1–2 sentences, describe the task the robot is performing. "
                    "Be specific about objects, containers, and the sequence of actions. "
                    "Respond with only the task description, no preamble."
                ),
            }
        )
        return client.complete(
            system="You are an expert robotics analyst. Describe robot tasks concisely and accurately.",
            content=content,
        )

    def label(
        self,
        result: PipelineResult,
        run_dir: Path | None = None,
        rollout_dir: Path | None = None,
    ) -> dict[str, Any]:
        client = self._client or make_vlm_client(self.model, self.thinking_level)

        if result.graph is None:
            return {"error": "No graph in PipelineResult — run GraphConstruction first."}
        if result.labels is None or result.windows is None:
            return {"error": "labels and windows required for frame extraction."}

        # Fall back to the run's recorded rollout dir for standalone use.
        # Recorded paths are relative to the data root (see pipeline.run).
        if rollout_dir is None and result.metrics:
            rollout_dir = self._fallback_rollout_dir(result)
        if rollout_dir is not None:
            self._check_cameras(rollout_dir)

        graph = result.graph
        labels = np.asarray(result.labels)
        windows = result.windows
        node_values = result.node_values

        # Behavior nodes only (exclude special nodes)
        behavior_node_ids = list(graph.cluster_nodes)

        if not behavior_node_ids:
            return {"nodes_labeled": 0, "model": self.model, "labels": {}}

        graph_text = _graph_summary(graph, node_values)

        # Use provided task description, or infer from successful episodes.
        task_description = self.task_description
        inferred_task: str | None = None
        if not task_description and rollout_dir is not None:
            # Optional enrichment: labeling works without it, so a blank or
            # failed inference must not cost the caller the whole run.
            try:
                inferred_task = self._infer_task(result, rollout_dir, client)
            except Exception as exc:
                log.warning("Task inference failed, labeling without a task hint: %s", exc)
                inferred_task = None
            task_description = inferred_task or ""
        task_hint = f"\nTask context: {task_description}" if task_description else ""
        # For the per-node user messages, where it sits beside the actual ask
        # rather than at the top of a system prompt followed by the whole graph.
        task_line = f"Task: {task_description}\n" if task_description else ""
        views_hint = _views_hint(self.cameras)

        # Per-node context: contact sheet + progress
        node_ctx: dict[int, dict] = {}
        for nid in behavior_node_ids:
            progress = node_progress_range(nid, labels, windows.windows, windows.episodes)
            sheet_b64 = None
            if rollout_dir is not None and run_dir is not None:
                sheet_b64 = self._sheet_b64(run_dir, rollout_dir, nid)
            preds, succs = _neighbor_ids(graph, nid)
            node_ctx[nid] = {
                "progress": progress,
                "sheet_b64": sheet_b64,
                "preds": preds,
                "succs": succs,
            }

        # ---------------------------------------------------------------- Pass 1: draft labels
        draft_labels: dict[int, dict] = {}
        errors: dict[int, str] = {}
        for nid in behavior_node_ids:
            ctx = node_ctx[nid]
            progress_str = _progress_desc(ctx["progress"])
            preds_str = ", ".join(node_display_name(p) for p in ctx["preds"]) or "none"
            succs_str = ", ".join(node_display_name(s) for s in ctx["succs"]) or "none"

            content: list[Any] = []

            # Predecessor contact sheets for three-context hierarchy
            for pred_id in ctx["preds"][:2]:
                pred_b64 = node_ctx.get(pred_id, {}).get("sheet_b64")
                if pred_b64:
                    content += _sheet_blocks(
                        f"[Previous behavior — {node_display_name(pred_id)} contact sheet]",
                        pred_b64,
                    )

            # Current node contact sheet
            if ctx["sheet_b64"]:
                content += _sheet_blocks(
                    f"[Current behavior — {node_display_name(nid)} contact sheet]",
                    ctx["sheet_b64"],
                )

            # Successor contact sheets
            for succ_id in ctx["succs"][:2]:
                if succ_id in TERMINAL_NODE_IDS:
                    continue
                succ_b64 = node_ctx.get(succ_id, {}).get("sheet_b64")
                if succ_b64:
                    content += _sheet_blocks(
                        f"[Next behavior — {node_display_name(succ_id)} contact sheet]",
                        succ_b64,
                    )

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Analyze behavior cluster {node_display_name(nid)} (id={nid}).\n"
                        f"{task_line}"
                        f"Episode-relative position: {progress_str}.\n"
                        f"Incoming from: {preds_str}. Outgoing to: {succs_str}.\n\n"
                        "First, reason step-by-step:\n"
                        "1. What completed manipulation event does this cluster represent? "
                        "(What object state changed? What was the robot doing?)\n"
                        "2. How does this cluster relate to the clusters before and after it?\n"
                        "3. What task stage does this cluster represent (early, mid, late)?\n\n"
                        "Then assign:\n"
                        "- A SHORT label (3–6 words, atomic event, e.g. 'reaching for handle')\n"
                        "- A one-sentence description\n\n"
                        "Respond in JSON: "
                        '{"reasoning": "...", "label": "...", "description": "..."}'
                    ),
                }
            )

            # Per node, not per run: a single bad response used to propagate
            # out of label() before anything was written, so N-1 good nodes
            # were paid for and thrown away. Skip the node, keep the rest.
            try:
                text = client.complete(
                    system=(
                        "You are an expert robotics analyst. You analyze robot policy behavior "
                        "clusters from video contact sheets (grids of frames with timestep "
                        "numbers stamped on each tile). Each frame grid shows the robot's "
                        f"behavior during that cluster.\n{views_hint}{task_hint}\n\n"
                        f"Full behavior graph context:\n{graph_text}"
                    ),
                    content=content,
                )
                draft_labels[nid] = _parse_vlm_json(
                    text, f"the draft label of {node_display_name(nid)}"
                )
            except Exception as exc:
                errors[nid] = str(exc)
                log.warning("Draft label for %s failed: %s", node_display_name(nid), exc)

        # ---------------------------------------------------------------- Pass 2: seeded refinement
        # Only nodes that drafted successfully — writing a placeholder
        # "Behavior 7" over a hand-checked label would be worse than skipping.
        final_labels: dict[int, dict] = {}
        for nid in list(draft_labels):
            draft = draft_labels[nid]
            draft_label = draft.get("label", f"Behavior {nid}")
            draft_reasoning = draft.get("reasoning", "")

            ctx = node_ctx[nid]
            content2: list[Any] = []
            if ctx["sheet_b64"]:
                content2 += _sheet_blocks(None, ctx["sheet_b64"])
            content2.append(
                {
                    "type": "text",
                    "text": (
                        f"Refine the label for behavior cluster {node_display_name(nid)}.\n"
                        # Repeated next to the judgment itself, not just in the
                        # system prompt: this pass decides the label that gets
                        # written, and "provide a better label" is meaningless
                        # without knowing what task the robot is performing.
                        f"{task_line}"
                        f"\nInitial reasoning: {draft_reasoning}\n"
                        f"Initial label: '{draft_label}'\n\n"
                        "Review the contact sheet above. Does the initial label accurately "
                        "describe the completed manipulation event shown, in the context of "
                        "the task?\n"
                        "- If yes, confirm it (you may tweak wording for clarity).\n"
                        "- If no, provide a better label.\n\n"
                        "Use the task's own vocabulary for objects and actions where the "
                        "frames support it.\n\n"
                        "Respond in JSON: "
                        '{"label": "...", "description": "...", "confidence": 0.0–1.0}'
                    ),
                }
            )

            # Pass 1 already produced a usable label, so a refine failure
            # degrades to the draft instead of losing the node.
            try:
                text2 = client.complete(
                    system=(
                        "You are an expert robotics analyst refining a behavior cluster label."
                        # Same reason the task hint is repeated here: this pass
                        # decides the written label off the same multi-view sheet,
                        # so it needs the same warning about what its rows mean.
                        f"\n{views_hint}{task_hint}\n\n"
                        f"Full behavior graph context:\n{graph_text}"
                    ),
                    content=content2,
                )
                refined = _parse_vlm_json(text2, f"the refined label of {node_display_name(nid)}")
            except Exception as exc:
                errors[nid] = str(exc)
                log.warning(
                    "Refine for %s failed, keeping the draft label: %s",
                    node_display_name(nid),
                    exc,
                )
                refined = {}
            final_labels[nid] = {
                "label": refined.get("label", draft_label),
                "description": refined.get("description", ""),
                "confidence": float(refined.get("confidence", 0.8)),
                "reasoning": draft_reasoning,
            }

        # Write labels to node_labels.json if we have a run_dir
        if run_dir is not None:
            upsert_node_labels(
                run_dir,
                {
                    nid: make_label_entry(
                        label=entry["label"],
                        description=entry["description"],
                        source=self.model,
                        reasoning=entry["reasoning"],
                        confidence=entry["confidence"],
                    )
                    for nid, entry in final_labels.items()
                },
            )

        # node_labels.json is the single source of truth (it can be edited
        # later, e.g. via the MCP label_node tool) — return only a summary.
        out: dict[str, Any] = {"model": self.model, "nodes_labeled": len(final_labels)}
        if inferred_task:
            out["inferred_task"] = inferred_task
        if errors:
            # Surfaced rather than logged only: a partial pass still writes
            # node_labels.json, so the caller has to be told what it missed.
            out["node_errors"] = {node_display_name(nid): msg for nid, msg in errors.items()}
        return out

    def analyze(
        self,
        result: PipelineResult,
        run_dir: Path | None = None,
        rollout_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Generate a graph-level analysis report and write it to graph_analysis.json.

        Requires existing node labels in node_labels.json (run label() first).
        Produces: task summary, success/failure modes, similar cluster pairs,
        and an overall assessment.  Returns the analysis dict.
        """
        from datetime import datetime, timezone

        client = self._client or make_vlm_client(self.model, self.thinking_level)

        if result.graph is None:
            return {"error": "No graph in PipelineResult."}

        if rollout_dir is None and result.metrics:
            rollout_dir = self._fallback_rollout_dir(result)
        if rollout_dir is not None:
            self._check_cameras(rollout_dir)

        graph = result.graph
        node_values = result.node_values
        task_hint = self.task_description or ""

        # Load existing labels (must run label() first)
        existing_labels: dict[int, dict] = {}
        if run_dir is not None:
            existing_labels = load_node_labels(run_dir)

        behavior_node_ids = list(graph.cluster_nodes)

        # ── Top paths (success and failure) ──────────────────────────────────
        success_paths = graph.enumerate_paths_to_terminal(SUCCESS_NODE_ID, max_paths=5)
        failure_paths = graph.enumerate_paths_to_terminal(FAILURE_NODE_ID, max_paths=5)

        def _path_str(node_ids: list[int]) -> str:
            return " → ".join(node_display_name(n) for n in node_ids)

        paths_text = "Top success paths:\n"
        for path_ids, prob, _ in success_paths:
            paths_text += f"  {_path_str(path_ids)} ({prob:.0%})\n"
        paths_text += "\nTop failure paths:\n"
        for path_ids, prob, _ in failure_paths:
            paths_text += f"  {_path_str(path_ids)} ({prob:.0%})\n"

        # ── Node labels summary ───────────────────────────────────────────────
        success_node_ids: set[int] = {
            nid for path_ids, _, _ in success_paths for nid in path_ids if nid >= 0
        }

        labels_text = "Behavior cluster labels (all nodes):\n"
        for nid in sorted(behavior_node_ids):
            entry = existing_labels.get(nid, {})
            lbl = entry.get("label", f"Behavior {nid}")
            desc = entry.get("description", "")
            labels_text += f"  Behavior {nid}: {lbl}"
            if desc:
                labels_text += f" — {desc}"
            labels_text += "\n"

        success_labels_text = "Behavior cluster labels (success-path nodes only):\n"
        for nid in sorted(success_node_ids):
            entry = existing_labels.get(nid, {})
            lbl = entry.get("label", f"Behavior {nid}")
            desc = entry.get("description", "")
            success_labels_text += f"  Behavior {nid}: {lbl}"
            if desc:
                success_labels_text += f" — {desc}"
            success_labels_text += "\n"

        graph_text = _graph_summary(graph, node_values)

        # ── Build content: one contact sheet per behavior node ────────────────
        content: list[Any] = []
        node_sheets: dict[int, str | None] = {}
        for nid in behavior_node_ids:
            b64 = None
            if rollout_dir is not None and run_dir is not None:
                # Reuses what label() built in this process (see _sheet_b64), and
                # takes its sampling settings off self — passing them positionally
                # is how tile_px came to be missing here, which made analyze() a
                # TypeError on every run that had frames to show at all.
                b64 = self._sheet_b64(run_dir, rollout_dir, nid)
            node_sheets[nid] = b64
            entry = existing_labels.get(nid, {})
            lbl = entry.get("label", f"Behavior {nid}")
            if b64:
                content += _sheet_blocks(f"[Behavior {nid}: {lbl}]", b64)

        content.append(
            {
                "type": "text",
                "text": (
                    f"{graph_text}\n\n"
                    f"{labels_text}\n\n"
                    f"{success_labels_text}\n\n"
                    f"{paths_text}\n\n"
                    + (f"Task context: {task_hint}\n\n" if task_hint else "")
                    + "Analyze this robot behavior graph and provide:\n\n"
                    "1. task_summary: 2–3 sentences describing what the robot is trying to do "
                    "and the overall structure of the policy. Base this strictly on the "
                    "success-path node labels listed above — do not introduce objects, actions, "
                    "or goals not evidenced by those labels.\n\n"
                    "2. success_modes: list of objects, one per distinct success pattern. Each has:\n"
                    '   - "path_str": human-readable path (e.g. "B2 → B3 → SUCCESS")\n'
                    '   - "node_ids": list of integer node IDs along the path\n'
                    '   - "probability": float (0–1)\n'
                    '   - "description": 1–2 sentences — what the robot does right on this path\n\n'
                    "3. failure_modes: list of objects, one per distinct failure pattern, "
                    "sorted by probability descending (most common first). Each has:\n"
                    '   - "description": 1–2 sentences — where and why episodes fail\n'
                    '   - "key_node_ids": list of integer node IDs most associated with this failure\n'
                    '   - "probability": float (0–1) — estimated fraction of all episodes that hit this failure\n\n'
                    "4. similar_pairs: list of behavior cluster pairs that look visually similar "
                    "(only include genuinely similar pairs). Each has:\n"
                    '   - "node_a": integer node ID\n'
                    '   - "node_b": integer node ID\n'
                    '   - "distinction": 1 sentence describing the key visual difference\n\n'
                    "5. overall_assessment: 2–3 sentences — key insights about the policy, "
                    "bottlenecks, or surprising patterns.\n\n"
                    "Respond ONLY in JSON with keys: "
                    "task_summary, success_modes, failure_modes, similar_pairs, overall_assessment."
                ),
            }
        )

        text = client.complete(
            system=(
                "You are an expert robotics analyst. Analyze the behavior graph of a robot "
                "policy from contact-sheet images and graph statistics. Be precise and concise. "
                f"Respond only in valid JSON.{_views_hint(self.cameras)}"
            ),
            content=content,
        )

        analysis = _parse_vlm_json(text, "the graph analysis")
        analysis["model"] = self.model
        analysis["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Ensure failure_modes are sorted by probability descending so index 0 = dominant
        if isinstance(analysis.get("failure_modes"), list):
            analysis["failure_modes"].sort(key=lambda m: m.get("probability", 0), reverse=True)

        if run_dir is not None:
            save_graph_analysis(run_dir, analysis)

        return analysis
