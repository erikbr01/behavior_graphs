"""Shared IO and registries for the progress-estimation benchmark.

Single source of truth for everything the runner, the methods, and the analysis
scripts used to each re-implement: rollout/embedding loading, mp4 frame decoding,
the per-episode progress-curve ``.npz`` schema, the per-suite instruction maps,
and the method display/color/tag registry.

An :class:`EpisodeSet` owns a rollout directory (success flags always; policy
embeddings and mp4 frames loaded lazily, since graph methods need embeddings and
VLM methods need video). :meth:`EpisodeSet.view` hands a method a subset of
episodes (a train or eval split) as an :class:`EpisodeView`; episodes are indexed
by their position in ``episodes/metadata.yaml``, which equals the ``epNNNN`` mp4
filename index in the canonical rollout layout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "load_rollouts",
    "policy_embedding_layer",
    "list_episodes",
    "read_frames",
    "prefix_samples",
    "save_curves",
    "load_curves",
    "Curves",
    "EpisodeSet",
    "EpisodeView",
    "resolve_instructions",
    "metadata_instructions",
    "LIBERO_SUITES",
    "SINGLE_TASK_INSTRUCTIONS",
    "METHOD_REGISTRY",
    "MethodInfo",
    "resolve_curve_path",
]


# --------------------------------------------------------------------- rollout IO
def load_rollouts(rollout_dir: Path, layer: str = "auto"):
    """Load per-episode observation arrays + success flags from a rollout dir.

    Picks a data layer npz (``auto`` = policy_embeddings/obs -> observations/state
    -> first policy_embeddings glob) and slices it into one array per episode using
    the ``episodes/metadata.yaml`` lengths. Returns ``(episodes, flags, layer_name)``.
    """
    import yaml

    rollout_dir = Path(rollout_dir)
    meta = yaml.safe_load((rollout_dir / "episodes" / "metadata.yaml").read_text())
    candidates = {
        "obs": rollout_dir / "policy_embeddings" / "obs.npz",
        "state": rollout_dir / "observations" / "state.npz",
    }
    if layer == "auto":
        path = next((p for p in candidates.values() if p.exists()), None)
        if path is None:
            embs = sorted((rollout_dir / "policy_embeddings").glob("*.npz"))
            path = embs[0] if embs else None
    elif layer in candidates:
        path = candidates[layer]
    else:
        path = rollout_dir / "policy_embeddings" / f"{layer}.npz"
    if path is None or not path.exists():
        raise FileNotFoundError(f"no data layer found in {rollout_dir} ({layer})")
    z = np.load(path)
    obs = z[list(z.keys())[0]]
    episodes, flags, off = [], [], 0
    for ep in meta["episodes"]:
        episodes.append(obs[off : off + ep["length"]])
        flags.append(bool(ep["success"]))
        off += ep["length"]
    assert off == len(obs), f"metadata total {off} != rows {len(obs)} in {path}"
    return episodes, flags, path.name


def policy_embedding_layer(rollout_dir: Path) -> str:
    """The policy-embedding file stem under ``policy_embeddings/`` (the method's
    premise input), so callers never silently fall back to raw simulator state via
    :func:`load_rollouts`'s ``auto`` ordering. Prefers a real policy embedding over
    a generic ``obs`` dump."""
    embs = sorted((Path(rollout_dir) / "policy_embeddings").glob("*.npz"))
    if not embs:
        raise FileNotFoundError(f"no policy_embeddings/*.npz in {rollout_dir}")
    non_obs = [p for p in embs if p.stem != "obs"]
    return (non_obs or embs)[0].stem


def list_episodes(rollout_dir: Path):
    """Return ``[(index, mp4_path, success)]`` from ``media/ep*.mp4`` + metadata."""
    import yaml

    rollout_dir = Path(rollout_dir)
    meta = yaml.safe_load((rollout_dir / "episodes" / "metadata.yaml").read_text())
    succ = {e["index"]: bool(e["success"]) for e in meta["episodes"]}
    out = []
    for mp4 in sorted((rollout_dir / "media").glob("ep*.mp4")):
        idx = int(mp4.stem[2:6])
        out.append((idx, mp4, succ.get(idx, mp4.stem.endswith("succ"))))
    return out


#: Preferred camera when a rollout dir stores media per view. A progress
#: estimator watching a wrist camera sees the gripper, not the task, so the
#: scene view is the one to hand it; the rest are fallbacks in the order a
#: DROID dir tends to carry them.
_CAMERA_PREFERENCE = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "agentview",
    "wrist_image_left",
)


def _episode_mp4(
    rollout_dir: Path,
    idx: int,
    success: bool | None = None,
    prefer_camera: str | None = None,
) -> Path | None:
    """The clip for one episode, flat layout or per-camera.

    Two layouts are in use. Sim dirs write ``media/ep0000_success.mp4``; DROID
    dirs write ``media/<camera>/ep0000_success.mp4``, one subdir per view. Only
    the flat form was searched, so every DROID dataset raised "no mp4 for
    episode 0" the moment a video method reached it -- 11 of them, including
    every set_table and prepare_tray run.
    """
    media = Path(rollout_dir) / "media"
    roots = [media]
    # Preferred cameras first, then any other subdir, so an unfamiliar rig
    # still resolves rather than failing.
    subdirs = [d for d in sorted(media.iterdir()) if d.is_dir()] if media.is_dir() else []
    # An explicitly configured camera outranks the built-in preference order: a
    # method told which view to score should not silently be handed another.
    if prefer_camera and (media / prefer_camera).is_dir():
        roots.append(media / prefer_camera)
    roots += [media / c for c in _CAMERA_PREFERENCE if (media / c).is_dir()]
    roots += [d for d in subdirs if d.name not in _CAMERA_PREFERENCE]
    for root in roots:
        if success is not None:
            p = root / f"ep{idx:04d}_{'succ' if success else 'fail'}.mp4"
            if p.exists():
                return p
        cand = sorted(root.glob(f"ep{idx:04d}_*.mp4"))
        if cand:
            return cand[0]
    return None


def read_frames(mp4_path: Path, rgb: bool = True) -> list[np.ndarray]:
    """Decode an mp4 to a list of HxWx3 uint8 frames (RGB if ``rgb`` else BGR)."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if rgb else bgr)
    cap.release()
    return frames


def prefix_samples(frames: list[np.ndarray], query_fracs: np.ndarray, n_ctx: int):
    """For each q, uniformly sample ``n_ctx`` frames from the prefix ``frames[:qT]``
    (always including the last prefix frame). Returns ``list[list[frame]]``."""
    T = len(frames)
    out = []
    for q in query_fracs:
        end = max(1, int(round(q * T)))
        idx = np.unique(np.linspace(0, end - 1, min(n_ctx, end)).round().astype(int))
        out.append([frames[i] for i in idx])
    return out


# ---------------------------------------------------------------- curve npz schema
@dataclass
class Curves:
    """Per-episode progress curves as saved/loaded from an ``*_curves.npz``."""

    curves: list[np.ndarray]
    flags: np.ndarray
    idx: np.ndarray
    instr: np.ndarray | None = None


def save_curves(out_dir: Path, name: str, tag: str, curves, flags, idx, instr=None) -> Path:
    """Write ``{out_dir}/{name}_{tag}_curves.npz`` in the canonical schema
    (object ``curves`` + ``flags`` + ``idx`` [+ ``instr``]). Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{tag}_curves.npz"
    payload = dict(
        curves=np.array([np.asarray(c, float) for c in curves], dtype=object),
        flags=np.asarray(flags),
        idx=np.asarray(idx),
    )
    if instr is not None:
        payload["instr"] = np.asarray(instr)
    # No `allow_pickle=True`: np.savez only grew that keyword in numpy 2.4, and
    # the diffusion_policy env is on 1.23 where `**kwds` swallows it and writes
    # a bogus array called `allow_pickle` into every file. Object arrays are
    # pickled by savez regardless, which is what the ragged `curves` needs.
    np.savez(path, **payload)
    return path


def load_curves(path: Path) -> Curves:
    """Load an ``*_curves.npz`` into a :class:`Curves` (curves as float arrays)."""
    z = np.load(path, allow_pickle=True)
    curves = [np.asarray(c, float) for c in z["curves"]]
    flags = np.asarray(z["flags"]).astype(bool)
    idx = np.asarray(z["idx"]) if "idx" in z else np.arange(len(curves))
    instr = np.asarray(z["instr"]) if "instr" in z else None
    return Curves(curves=curves, flags=flags, idx=idx, instr=instr)


# ------------------------------------------------------------------- instructions
# LIBERO per-suite instructions == the task.language the pi0.5 policy was prompted
# with (task_name underscores->spaces). Episodes are contiguously blocked by task:
# task_id = episode_index // trials_per_task.
LIBERO_SUITES: dict[str, list[str]] = {
    "libero_goal": [
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside",
        "put the bowl on top of the cabinet",
        "push the plate to the front of the stove",
        "put the cream cheese in the bowl",
        "turn on the stove",
        "put the bowl on the plate",
        "put the wine bottle on the rack",
    ],
    "libero_object": [
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the orange juice and place it in the basket",
    ],
    "libero_spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ],
}

# Single-task datasets: one instruction for the whole suite (VLM baselines).
SINGLE_TASK_INSTRUCTIONS: dict[str, str] = {
    "robomimic_lift_mh": "lift the red cube",
    "robomimic_lift_ph": "lift the red cube",
    "robomimic_can_mh": "pick up the can and place it in the correct bin",
    "robomimic_can_ph": "pick up the can and place it in the correct bin",
    "robomimic_square_mh": "pick up the square nut and place it on the peg",
    "robomimic_square_ph": "pick up the square nut and place it on the peg",
    "robomimic_transport_mh": (
        "transport the hammer to the target bin and clear the trash from the target bin"
    ),
    "robomimic_transport_ph": (
        "transport the hammer to the target bin and clear the trash from the target bin"
    ),
    "robomimic_tool_hang_ph": "assemble the frame and hang the wrench on the hook",
    # ---- DROID real robot -------------------------------------------------
    # Phrased from the VLM behaviour annotations in each dataset's
    # `node_labels.json` (Graph Demo), so the wording describes what the robot
    # is actually seen doing rather than what a dataset name suggests. The
    # variants differ only in where the milk goes, which is what the rgrl
    # selection ("prepare plate - milk right/middle/left") encodes.
    "set_table_baseline_pretrained_dit": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk beside the plate"
    ),
    "set_table_plusleft_pretrained_dit": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk to the left of the plate"
    ),
    "set_table_plusmiddle_pretrained_dit": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk in the middle, beside the plate"
    ),
    "set_table_plusright_pretrained_dit": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk to the right of the plate"
    ),
    "set_table_baseline": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk beside the plate"
    ),
    "set_table_baseline_v2_sync": (
        "set the table: place the pumpkin from the metal bowl onto the plate "
        "and put the chocolate milk beside the plate"
    ),
    "set_table_baseline_v2_plusleft_sync": (
        "set the table: place the pumpkin onto the plate and put the chocolate "
        "milk to the left of the plate"
    ),
    "set_table_baseline_v2_plusmiddle_sync": (
        "set the table: place the pumpkin onto the plate and put the chocolate "
        "milk in the middle, beside the plate"
    ),
    "set_table_baseline_v2_plusright_sync": (
        "set the table: place the pumpkin onto the plate and put the chocolate "
        "milk to the right of the plate"
    ),
    "prepare_tray_baseline": (
        "prepare the tray: slide the tray out from under the shelf, then place "
        "the bowl and the fork on top of the tray"
    ),
    "prepare_tray_baseline_dp": (
        "prepare the tray: slide the tray out from under the shelf, then place "
        "the bowl and the fork on top of the tray"
    ),
    "prepare_tray_baseline_rgrl": (
        "prepare the tray: slide the tray out from under the shelf, then place "
        "the bowl and the fork on top of the tray"
    ),
    "prepare_tray_plusbottom": (
        "prepare the tray: slide the tray out from under the shelf, then place "
        "the bowl and the fork on top of the tray"
    ),
    "prepare_tray_plustop": (
        "prepare the tray: slide the tray out from under the shelf, then place "
        "the bowl and the fork on top of the tray"
    ),
    "cooking_baseline": (
        "cook: put the ingredients from the bowls into the pot and place the lid on the pot"
    ),
    "cooking_baseline_dit_200": (
        "cook: put the ingredients from the bowls into the pot and place the lid on the pot"
    ),
    "cooking_baseline_carrot_dit": (
        "cook: put the carrot into the pot and place the lid on the pot"
    ),
    "cooking_baseline_corn_dit": ("cook: put the corn into the pot and place the lid on the pot"),
    "cooking_baseline_lid_dit": (
        "cook: put the ingredients into the pot and place the lid on the pot"
    ),
    # ---- MimicGen ---------------------------------------------------------
    # These hdf5s record an `env_name` (Coffee_D0, HammerCleanup_D0, ...) but no
    # language field, so the wording below is the published MimicGen task
    # definition for that env rather than anything read out of the data. The env
    # name is what pins which task each one is.
    "mimicgen_coffee_d0": "insert the coffee pod into the machine and close the lid",
    "mimicgen_hammer_cleanup_d0": ("open the drawer, put the hammer inside, and close the drawer"),
    "mimicgen_mug_cleanup_d0": "open the drawer, put the mug inside, and close the drawer",
    "mimicgen_square_d0": "pick up the square nut and place it on the peg",
    "mimicgen_stack_d0": "stack the red cube on top of the green cube",
    "mimicgen_stack_three_d0": "stack the three cubes into a single tower",
    "mimicgen_threading_d0": "pick up the needle and thread it onto the rod",
    "mimicgen_three_piece_assembly_d0": (
        "assemble the three-piece structure by stacking the pieces in order"
    ),
    # transport_mh_salvage250 is robomimic transport_mh data, so it takes the
    # same instruction as the dataset it was salvaged from.
    "transport_mh_salvage250": (
        "transport the hammer to the target bin and clear the trash from the target bin"
    ),
    # ---- other sim --------------------------------------------------------
    "vla_foundry_mug": "pick up the mug and place it on the target",
    "vla_foundry_qwen3_mug": "pick up the mug and place it on the target",
    # NOT listed, deliberately: libero_pro_spatial_with_mug,
    # libero_plus_perturbations, libero_pro_perturbations. Those are MULTI-task
    # suites, so one instruction for the whole dataset would be wrong on most of
    # its episodes -- they need a per-task list in LIBERO_SUITES, the way
    # libero_goal/object/spatial have one.
}


def metadata_instructions(episodes: Sequence[dict]) -> dict[int, str]:
    """Per-episode task strings the rollout metadata already carries, if any.

    The perturbation suites record the exact string the policy was prompted with
    on every episode (``prompt``, with ``task_name`` as a fallback), which is
    strictly better provenance than any table maintained here. Suites that
    record no language at all yield an empty map, so a caller can treat this as
    "nothing known" rather than having to probe the fields itself.
    """
    out: dict[int, str] = {}
    for i, ep in enumerate(episodes):
        text = ep.get("prompt") or ep.get("language_instruction") or ""
        if not text and ep.get("task_name"):
            text = str(ep["task_name"]).replace("_", " ")
        if text:
            out[i] = str(text)
    return out


def resolve_instructions(
    n_episodes: int,
    suite: str = "",
    instruction: str = "",
    instructions_json: Path | None = None,
    from_metadata: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Per-episode instruction map. Precedence per episode: instructions_json
    (idx as str key) > suite (episode_index // trials_per_task) > single instruction
    (a bare suite name resolves via SINGLE_TASK_INSTRUCTIONS) > the task string the
    rollout metadata recorded.

    Metadata ranks LAST deliberately, even though it is the most trustworthy
    source, because promoting it would silently restate the question every
    already-scored dataset was asked and invalidate its numbers. As a fallback it
    only ever fills episodes that would otherwise have been asked about no task
    at all -- which is what the three perturbation suites were getting, and why
    their VLM scores came out negative.
    """
    per_idx: dict[int, str] = {}
    if instructions_json:
        raw = json.loads(Path(instructions_json).read_text())
        per_idx = {int(k): v for k, v in raw.items()}
    suite_map: dict[int, str] = {}
    if suite in LIBERO_SUITES:
        tasks = LIBERO_SUITES[suite]
        per = max(1, n_episodes // len(tasks))
        suite_map = {i: tasks[min(i // per, len(tasks) - 1)] for i in range(n_episodes)}
    default = instruction or SINGLE_TASK_INSTRUCTIONS.get(suite, "")
    meta = dict(from_metadata or {})
    return {
        i: per_idx.get(i) or suite_map.get(i) or default or meta.get(i, "")
        for i in range(n_episodes)
    }


# ---------------------------------------------------------------------- EpisodeSet
class EpisodeSet:
    """A rollout directory as an indexable set of episodes (position == mp4 index).

    Success flags load eagerly from metadata; policy embeddings and mp4 frames load
    lazily on first access. ``instructions`` maps episode position -> task string.
    """

    def __init__(self, name, rollout_dir, flags, layer="auto", instructions=None):
        self.name = name
        self.rollout_dir = Path(rollout_dir)
        self.flags = [bool(f) for f in flags]
        self.layer = layer
        self.instructions = dict(instructions or {})
        self._embeddings: list[np.ndarray] | None = None
        self.layer_name: str | None = None

    @classmethod
    def from_dir(
        cls,
        name,
        rollout_dir,
        layer="auto",
        suite="",
        instruction="",
        instructions_json=None,
    ) -> EpisodeSet:
        import yaml

        rollout_dir = Path(rollout_dir)
        meta = yaml.safe_load((rollout_dir / "episodes" / "metadata.yaml").read_text())
        flags = [bool(e["success"]) for e in meta["episodes"]]
        instr = resolve_instructions(
            len(flags),
            suite,
            instruction,
            instructions_json,
            from_metadata=metadata_instructions(meta["episodes"]),
        )
        return cls(name, rollout_dir, flags, layer=layer, instructions=instr)

    @property
    def n(self) -> int:
        return len(self.flags)

    def embeddings(self) -> list[np.ndarray]:
        if self._embeddings is None:
            eps, flags, layer_name = load_rollouts(self.rollout_dir, self.layer)
            if len(flags) != len(self.flags):
                raise ValueError(
                    f"{self.name}: embedding episodes {len(flags)} != metadata {len(self.flags)}"
                )
            self._embeddings, self.layer_name = eps, layer_name
        return self._embeddings

    def frames(self, idx: int, rgb: bool = True) -> list[np.ndarray]:
        mp4 = _episode_mp4(self.rollout_dir, idx, self.flags[idx])
        if mp4 is None:
            raise FileNotFoundError(f"no mp4 for episode {idx} in {self.rollout_dir}/media")
        return read_frames(mp4, rgb=rgb)

    def instruction(self, idx: int) -> str:
        return self.instructions.get(idx, "")

    def view(self, indices) -> EpisodeView:
        return EpisodeView(self, list(indices))


class EpisodeView:
    """A subset (train or eval split) of an :class:`EpisodeSet`. ``indices`` are
    original episode positions; accessors preserve that order."""

    def __init__(self, parent: EpisodeSet, indices: list[int]):
        self.parent = parent
        self.indices = list(indices)

    @property
    def flags(self) -> list[bool]:
        return [self.parent.flags[i] for i in self.indices]

    def embeddings(self) -> list[np.ndarray]:
        allemb = self.parent.embeddings()
        return [allemb[i] for i in self.indices]

    def frames(self, idx: int, rgb: bool = True) -> list[np.ndarray]:
        return self.parent.frames(idx, rgb=rgb)

    def instruction(self, idx: int) -> str:
        return self.parent.instruction(idx)

    def __len__(self) -> int:
        return len(self.indices)


# ------------------------------------------------------------------ method registry
@dataclass(frozen=True)
class MethodInfo:
    """Display + provenance metadata for one progress method.

    ``tag`` is the ``*_curves.npz`` infix NEW runs write; ``infixes`` are the
    candidate infixes to LOAD (``tag`` first, then historical aliases). ``display``
    is the table label; ``input_label`` the modality shown in tables.
    """

    tag: str
    color: str
    input_label: str
    display: str
    infixes: tuple[str, ...]


METHOD_REGISTRY: dict[str, MethodInfo] = {
    "graph_stage": MethodInfo(
        "graph_direct_stage", "#1f77b4", "pi0.5 emb", "graph-stage", ("graph_direct_stage",)
    ),
    "graph_absorb": MethodInfo(
        "graph_direct_absorb", "#9467bd", "pi0.5 emb", "graph-absorb", ("graph_direct_absorb",)
    ),
    "frame_index": MethodInfo(
        "frame_index", "#7f7f7f", "GT order", "frame-index", ("frame_index",)
    ),
    # Plain "topreward": this is the published method, run through the authors'
    # own vendored code, so it is the one a reader means. The in-house
    # reimplementation below keeps the qualifier, since it is the variant that
    # needs explaining.
    "topreward_ll": MethodInfo("topreward_ll", "#ff7f0e", "RGB 8B", "topreward", ("topreward_ll",)),
    "topreward": MethodInfo(
        "topreward", "#ffbb78", "RGB 8B", "topreward-reimpl", ("topreward", "top_topreward")
    ),
    "robometer_4b": MethodInfo(
        "rbm4b_robometer",
        "#2ca02c",
        "RGB 4B",
        "robometer-4b",
        ("rbm4b_robometer", "rm4b_robometer"),
    ),
    "robometer_fft": MethodInfo(
        "rbmlibero_robometer",
        "#d62728",
        "RGB 4B",
        "robometer-fft",
        ("rbmlibero_robometer", "rmfft_robometer"),
    ),
    "dummy": MethodInfo("dummy", "#17becf", "harness", "dummy", ("dummy",)),
    "mlp": MethodInfo("mlp", "#e377c2", "policy emb", "mlp-regress", ("mlp",)),
    # Same method as `graph_stage`, different EXPERIMENT: the graph is fit on the
    # policy's own training demonstrations rather than on a held-out fold of the
    # rollouts, then every rollout is scored. A separate registry key rather than
    # a flag because it has to appear as its own row next to the k-fold number --
    # that comparison is the whole point of the experiment.
    "graph_stage_demofit": MethodInfo(
        "graph_direct_stage_demofit",
        "#1f77b4",
        "policy emb, demo-fit",
        "graph-stage (demo-fit)",
        ("graph_direct_stage_demofit",),
    ),
}


#: Every spelling of a method that should resolve to its registry key: the key
#: itself, its write tag, and each historical load alias -- with ``:`` folded to
#: ``_`` so the SCHEME spelling matches too. Built once; the registry is a
#: module-level constant.
_METHOD_ALIASES: dict[str, str] = {
    alias.replace(":", "_"): key
    for key, info in METHOD_REGISTRY.items()
    for alias in (key, info.tag, *info.infixes)
}


def canonical_method(method: str) -> str:
    """The registry key for ``method``, whichever of its spellings you hold.

    One method wears three names and they are not interchangeable by accident:
    ``metrics.json`` keys the graph rows by SCHEME (``graph_direct:stage`` --
    see ``report.STAGE``), the registry is keyed by METHOD (``graph_stage``),
    and the curve files carry the INFIX (``graph_direct_stage``). A caller
    holding the scheme spelling used to miss the registry entirely, fall back to
    globbing ``*_graph_direct:stage_curves.npz``, find nothing, and drop the
    method with no error -- which is how the graph curves went missing from the
    episode browser while their metrics sat in the table above it.

    Unknown names are returned unchanged, so this never invents a mapping.
    """
    return _METHOD_ALIASES.get(method.replace(":", "_"), method)


def discover_curve_methods(curves_dir: Path, prefix: str) -> tuple[str, ...]:
    """Registry keys for every ``{prefix}_*_curves.npz`` actually on disk.

    A method's metrics and its curves are written by different passes and one
    can exist without the other -- ``lg_pool500`` has robometer curves and no
    robometer entry in its ``metrics.json``. Anything that reads ``metrics.json``
    to decide what to PLOT therefore silently omits those curves, so ask the
    directory instead. Infixes that match no registry entry are returned as they
    were found rather than dropped.
    """
    curves_dir = Path(curves_dir)
    found = set()
    for path in curves_dir.glob(f"{prefix}_*_curves.npz"):
        infix = path.name[len(prefix) + 1 : -len("_curves.npz")]
        key = canonical_method(infix)
        # Only registry-known infixes. Dataset names nest -- `cooking_baseline`
        # is a prefix of `cooking_baseline_carrot_dit` -- so this glob also
        # matches the LONGER dataset's files, and the leftover text there is
        # another dataset's name plus a method, not a method. Without this the
        # picker filled up with phantoms like `carrot_dit_frame_index`.
        # The cost is that a method with curves but no registry entry is not
        # discovered here; it still arrives via metrics.json, which is keyed by
        # method rather than parsed out of a filename.
        if key in METHOD_REGISTRY:
            found.add(key)
    return tuple(sorted(found))


def resolve_curve_path(curves_dir: Path, prefix: str, method: str) -> Path | None:
    """First existing ``{prefix}_{infix}_curves.npz`` over a method's candidate
    infixes (tag first, then historical aliases), or None.

    ``method`` may be a registry key, a write tag or a scheme name; see
    :func:`canonical_method`.
    """
    curves_dir = Path(curves_dir)
    info = METHOD_REGISTRY.get(canonical_method(method))
    infixes = info.infixes if info else (method.replace(":", "_"),)
    for inf in infixes:
        p = curves_dir / f"{prefix}_{inf}_curves.npz"
        if p.exists():
            return p
    return None
