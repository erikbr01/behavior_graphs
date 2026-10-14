"""VLM reward-model progress methods (TOPReward, ROBOMETER).

These are vision models scored directly on each episode's mp4: ``fit`` loads the
model (heavy deps imported lazily, so the configs instantiate on CPU without a
GPU/VLM), ``predict`` returns one progress curve per eval episode. Ported from the
old ``scripts/vlm_reward_bench.py`` ``RewardModel`` tree; the scoring internals are
unchanged.

Model CODE for ROBOMETER/faithful-TOPReward comes from the vendored
``third_party/robometer`` submodule; its heavy runtime deps still come from the
robometer venv. Run VLM methods in that venv on a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from behavior_graphs.progress_estimation.io import prefix_samples
from behavior_graphs.progress_estimation.methods.base import ProgressMethod

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROBOMETER_DEFAULT = str(_REPO_ROOT / "third_party" / "robometer")


class _VLMBase(ProgressMethod):
    modality = "video"

    def score_episode(self, frames: list, instruction: str) -> np.ndarray:
        raise NotImplementedError

    def predict(self, eval) -> list[np.ndarray]:
        curves = []
        for idx in eval.indices:
            frames = eval.frames(idx, rgb=True)
            # NaN, not 0.0: too few frames to score is missing data, and 0.0 is
            # an emphatic prediction of failure. The metrics drop a NaN episode
            # from their pool; a zero used to drag the AUROC down as though the
            # method had confidently called it.
            if len(frames) < 3:
                curves.append(np.array([np.nan]))
                continue
            curve = np.asarray(self.score_episode(frames, eval.instruction(idx)), float)
            curves.append(np.clip(curve, 0.0, 1.0) if curve.size else np.array([np.nan]))
        return curves


class Dummy(_VLMBase):
    """CPU harness check: a perfect chronological ramp (VOC/Pearson~1, MAE~0)."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def score_episode(self, frames: list, instruction: str) -> np.ndarray:
        n = min(len(frames), 32)
        return np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])


class TopRewardProbe(_VLMBase):
    """Training-free TOPReward reimplementation: teacher-forced P(yes)/(P(yes)+P(no))
    over growing video prefixes (our own logprob probe, kept for reproducibility of
    the reimpl numbers; the faithful recipe is :class:`TopRewardLL`)."""

    QUERY = (
        "You are shown frames from a robot performing a task.\n"
        "Task: {instr}.\n"
        "Question: Has the task been completed successfully? "
        "Answer with a single word: yes or no."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        name: str = "topreward",
        device: str = "cuda",
        max_side: int = 384,
        n_queries: int = 16,
        n_ctx: int = 15,
    ):
        self.name = name
        self.model_id, self.device, self.max_side = model_id, device, max_side
        self.n_queries, self.n_ctx = n_queries, n_ctx
        self.model = self.processor = None
        self._yes_ids = self._no_ids = None

    def fit(self, train=None) -> None:
        # Idempotent: these methods are training-free, so `fit` only loads
        # weights. The runner calls it once per fold and a cross-dataset run
        # calls it once per dataset on top of that -- reloading an 8B
        # checkpoint ~100 times costs more than the scoring does.
        if getattr(self, "model", None) is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        tok = self.processor.tokenizer

        def first_ids(words):
            ids = set()
            for w in words:
                enc = tok(w, add_special_tokens=False)["input_ids"]
                if enc:
                    ids.add(enc[0])
            return sorted(ids)

        self._yes_ids = first_ids(["yes", " yes", "Yes", " Yes", "YES"])
        self._no_ids = first_ids(["no", " no", "No", " No", "NO"])
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("could not resolve yes/no token ids")

    def _pil(self, frames):
        from PIL import Image

        out = []
        for f in frames:
            im = Image.fromarray(f)
            if max(im.size) > self.max_side:
                s = self.max_side / max(im.size)
                im = im.resize((int(im.width * s), int(im.height * s)))
            out.append(im)
        return out

    def score_episode(self, frames: list, instruction: str) -> np.ndarray:
        torch = self._torch
        query = self.QUERY.format(instr=instruction)
        query_fracs = np.linspace(1.0 / self.n_queries, 1.0, self.n_queries)
        prefixes = prefix_samples(frames, query_fracs, self.n_ctx)
        scores = []
        for pf in prefixes:
            images = self._pil(pf)
            msg = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": im} for im in images]
                    + [{"type": "text", "text": query}],
                }
            ]
            text = self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            py = torch.logsumexp(logp[self._yes_ids], 0)
            pn = torch.logsumexp(logp[self._no_ids], 0)
            scores.append(float(torch.sigmoid(py - pn).item()))
        return np.asarray(scores)


class TopRewardLL(_VLMBase):
    """FAITHFUL TOPReward via the authors' own code (through ROBOMETER's framework):
    minmax-normalized log-likelihood of ``"<instruction> ... The answer is: True"``
    over growing prefix videos. Config matches ROBOMETER's baseline."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        name: str = "topreward_ll",
        repo: str = _ROBOMETER_DEFAULT,
        max_frames: int = 64,
        num_prefix_samples: int = 15,
        reduction: str = "mean",
        add_chat_template: bool = True,
        fps: float = 2.0,
    ):
        self.name = name
        self.model_id, self.repo = model_id, repo
        self.max_frames, self.num_prefix_samples = max_frames, num_prefix_samples
        self.reduction, self.add_chat_template, self.fps = reduction, add_chat_template, fps
        self._top = None

    def fit(self, train=None) -> None:
        # Idempotent: these methods are training-free, so `fit` only loads
        # weights. The runner calls it once per fold and a cross-dataset run
        # calls it once per dataset on top of that -- reloading an 8B
        # checkpoint ~100 times costs more than the scoring does.
        if self._top is not None:
            return
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from robometer.evals.baselines.topreward import TopReward

        self._top = TopReward(
            model_path=self.model_id,
            max_frames=self.max_frames,
            num_prefix_samples=self.num_prefix_samples,
            reduction=self.reduction,
            add_chat_template=self.add_chat_template,
            fps=self.fps,
        )

    def score_episode(self, frames: list, instruction: str) -> np.ndarray:
        frames_arr = np.asarray(frames, dtype=np.uint8)
        return np.asarray(self._top.compute_progress(frames_arr, instruction), dtype=float)


def _downscale(frames: np.ndarray, long_side: int | None) -> np.ndarray:
    """Shrink frames so their longer edge is at most *long_side*, aspect kept.

    Vision-token count grows with pixels, so a 1920x960 clip costs ~37x what a
    224x224 one does at the same frame budget. Left alone, a cross-dataset table
    would compare methods that had seen wildly different amounts of image -- and
    the big ones simply do not fit.
    """
    if not long_side:
        return frames
    import cv2

    h, w = frames.shape[1:3]
    if max(h, w) <= long_side:
        return frames
    scale = long_side / max(h, w)
    size = (int(round(w * scale)), int(round(h * scale)))
    return np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])


class Robometer(_VLMBase):
    """ROBOMETER's released trained reward model: one forward on the whole trajectory
    -> per-frame progress (+ a success head, stashed on ``last_success``). Wraps the
    repo's own inference internals so preprocessing/collation match exactly."""

    def __init__(
        self,
        ckpt: str = "robometer/Robometer-4B",
        name: str = "robometer_4b",
        repo: str = _ROBOMETER_DEFAULT,
        device: str = "cuda",
        max_frames: int | None = None,
        image_long_side: int | None = None,
    ):
        self.name = name
        self.ckpt = ckpt or "robometer/Robometer-4B"
        self.repo, self.device = repo, device
        # ROBOMETER's own eval linspace-subsamples each trajectory to a small
        # frame budget (eval_utils default 16); the model was trained on short
        # budgets, NOT full episodes. Feeding raw 400-1817-frame sim-bottles
        # episodes through the Qwen3-VL vision tower in one forward OOMs a 24GB
        # GPU, so set max_frames to replicate their subsampling for both fidelity
        # and memory. None = whole trajectory (matches short LIBERO episodes).
        self.max_frames = max_frames
        # Frames are also capped by their LONG SIDE. The frame budget alone is
        # not enough: this zoo runs from 224x224 (LIBERO) to 1920x960
        # (vla_foundry), an 8x spread in vision tokens per frame, and the big
        # ones OOM 24GB at any useful frame count. Capping makes the token
        # budget comparable across datasets as well as survivable, which the
        # raw resolutions never were. None = leave frames as they are.
        self.image_long_side = image_long_side
        self.last_success = None

    def fit(self, train=None) -> None:
        # Idempotent, as in the other VLM methods: training-free, so `fit` only
        # loads weights, and the runner calls it once per fold per dataset.
        if getattr(self, "_model", None) is not None:
            return
        import torch

        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self._torch = torch
        self._compute = compute_batch_outputs
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._device = dev
        cfg, tok, proc, model = load_model_from_hf(model_path=self.ckpt, device=dev)
        model = model.to(torch.bfloat16)
        model.eval()
        model.requires_grad_(False)  # inference only: no autograd graph (24GB-GPU OOM)
        self._cfg, self._tok = cfg, tok
        self._model = model
        self._collate = setup_batch_collator(proc, tok, cfg, is_eval=True)
        loss = getattr(cfg, "loss", None)
        self._discrete = (
            (getattr(loss, "progress_loss_type", "l2").lower() == "discrete") if loss else False
        )
        self._bins = getattr(loss, "progress_discrete_bins", None) or getattr(
            cfg.model, "progress_discrete_bins", 10
        )

    def score_episode(self, frames: list, instruction: str) -> np.ndarray:
        from robometer.data.dataset_types import ProgressSample, Trajectory

        frames_arr = np.asarray(frames, dtype=np.uint8)
        # Replicate ROBOMETER's own eval: linspace-subsample to the frame budget
        # BEFORE the forward. The model was trained on short budgets, and full-length
        # episodes (1000+ frames) OOM the Qwen3-VL vision tower on a 24GB GPU.
        if self.max_frames is not None and frames_arr.shape[0] > self.max_frames:
            sel = np.linspace(0, frames_arr.shape[0] - 1, self.max_frames).round().astype(int)
            frames_arr = frames_arr[sel]
        frames_arr = _downscale(frames_arr, self.image_long_side)
        T = int(frames_arr.shape[0])
        traj = Trajectory(
            frames=frames_arr,
            frames_shape=tuple(frames_arr.shape),
            task=instruction,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        batch = self._collate([ProgressSample(trajectory=traj, sample_type="progress")])
        pin = batch["progress_inputs"]
        mdtype = next(self._model.parameters()).dtype
        for k, v in pin.items():
            if hasattr(v, "to"):
                v = v.to(self._device)
                if k.startswith("pixel_values") and v.is_floating_point():
                    v = v.to(mdtype)
                pin[k] = v
        res = self._compute(
            self._model,
            self._tok,
            pin,
            sample_type="progress",
            is_discrete_mode=self._discrete,
            num_bins=self._bins,
        )
        prog = res.get("progress_pred", [])
        succ = (res.get("outputs_success", {}) or {}).get("success_probs", [])
        self.last_success = np.asarray(succ[0], float) if succ and len(succ) else None
        return np.asarray(prog[0], float) if prog and len(prog) else np.array([], float)
