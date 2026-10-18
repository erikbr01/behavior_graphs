"""Feature extractor that embeds episode video with a frozen vision backbone.

Every other extractor in this package reads an array the *policy* wrote: the
canonical layout's ``policy_embeddings/<layer>.npz`` comes out of a forward
pass through a trained checkpoint, which makes a trained policy a hard
prerequisite for building a behavior graph at all. This one reads the episode
*videos* instead and embeds them with a general-purpose vision backbone
(DINOv2/DINOv3 by default), so a graph can be built from any recorded
trajectory set -- a policy nobody instrumented, a stack whose internals are
not reachable, a human demonstration -- at the cost of describing what the
scene looks like rather than what the policy was thinking. See
``docs/vision_backbone_features.md``.

**Patch tokens, not the CLS token.** A ViT's CLS token (and the mean over its
patch tokens) is an image-level summary trained to be invariant to exactly the
thing a behavior graph needs to see: *where* in the frame the arm and the
objects are. Two frames of the same kitchen with the gripper at the pot and at
the drawer are near-identical under mean pooling and far apart under the patch
grid. :meth:`pooling="grid"` therefore average-pools the patch grid down to a
coarse ``grid x grid`` map and keeps it laid out, trading token count for
retained spatial structure; ``pooling="mean"``/``"cls"`` are the image-level
baselines to compare against, and ``"tokens"`` keeps every patch (256 x 768
for DINOv2-base at 224px, which is more than the reduction stage can use).

Frames are matched to timesteps by proportional division: an episode's
``num_samples`` rows are laid over its ``n_frames`` video frames in order, and
each row takes the *first* frame of its span by default -- with action chunking
one policy call spans several rendered env steps (mimicgen: 30 rows, 120
frames), and the first frame of the span is the observation that call actually
conditioned on. See :func:`frame_for_step`.

Requires the ``dino`` optional dependency group (torch + transformers +
imageio + pillow). The backbone is loaded on first :meth:`extract`, never in
the constructor, so this stage stays instantiable (and its config testable)
on a machine with no model weights.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.env import TrajectorySet
from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.core.types import EpisodeInfo, FeatureSet

log = logging.getLogger(__name__)

#: Pooling of the ``(grid_h, grid_w, d)`` patch-token map into one vector.
POOLINGS = ("grid", "mean", "cls", "tokens")

#: Which frame of a timestep's span of video frames represents it.
FRAME_IN_STEP = ("first", "middle", "last")

#: What every DINO/ViT checkpoint we care about normalizes with, and the
#: fallback when a repo ships no preprocessor config.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def frame_for_step(step: int, num_steps: int, num_frames: int, where: str = "first") -> int:
    """Video frame index representing timestep *step* of a *num_steps* episode.

    The episode's steps are laid over its frames proportionally -- step ``t``
    spans frames ``[t * r, (t + 1) * r)`` for ``r = num_frames / num_steps`` --
    and *where* picks one frame out of that span. The two counts differ
    whenever the recorder writes one frame per *env* step while the trajectory
    stores one row per *policy* call: an action-chunked policy executes the
    whole chunk between calls (mimicgen rollouts are 4 frames per row), and
    indexing the video by row would then only ever read the first quarter of
    every episode.

    ``"first"`` is the default because it is the frame the policy actually saw:
    the chunk was computed from the observation at the start of its span.
    ``"last"`` sees the outcome of the chunk instead, and ``"middle"`` splits
    the difference; all three coincide for a recorder that wrote one frame per
    row, which is the common case.
    """
    if where not in FRAME_IN_STEP:
        raise ValueError(f"frame_in_step must be one of {FRAME_IN_STEP}, got {where!r}")
    ratio = num_frames / num_steps
    lo = int(step * ratio)
    # At least one frame per step even when the video is *shorter* than the
    # trajectory (a truncated recording): consecutive steps then share a frame
    # rather than reading past the end.
    hi = max(lo + 1, int((step + 1) * ratio))
    if where == "first":
        idx = lo
    elif where == "last":
        idx = hi - 1
    else:
        idx = (lo + hi - 1) // 2
    return min(max(idx, 0), num_frames - 1)


class VisionBackboneFeatures(FeatureExtractor):
    """Per-timestep features from a frozen vision backbone over episode video.

    Args:
        model_name: HuggingFace model id loaded with ``AutoModel``, e.g.
            ``facebook/dinov2-small|base|large`` (ungated) or
            ``facebook/dinov3-vitb16-pretrain-lvd1689m`` (gated -- needs an
            accepted license on the hub). Any ViT whose ``last_hidden_state``
            is ``[prefix tokens..., patch tokens...]`` works; the number of
            prefix tokens (CLS, plus DINOv3's registers) is inferred from the
            patch grid rather than assumed.
        pooling: One of :data:`POOLINGS`. ``"grid"`` average-pools the patch
            map to ``grid x grid`` and flattens it (spatial structure kept),
            ``"mean"`` averages every patch, ``"cls"`` takes the CLS token,
            ``"tokens"`` keeps every patch token.
        grid: Side of the pooled patch map for ``pooling="grid"``. ``1`` is
            exactly ``"mean"``; ``2``-``4`` is the useful range (a 2x2 map of
            DINOv2-base is 3072 dims, 4x4 is 12288).
        cameras: Views to embed, concatenated in the order given. ``None``
            (default) is the one view :meth:`TrajectorySet.video_path` returns,
            which is the only thing every trajectory set supports; naming
            several requires a reader with ``video_paths`` (the canonical
            directory has it) and multiplies both cost and feature width.
        image_size: Frames are resized to ``image_size x image_size`` before
            the backbone, *without* center cropping: the processor default
            (resize short side, crop) would throw away the sides of a
            widescreen rollout view, and for spatial features the whole field
            of view matters more than the aspect ratio. Must be a multiple of
            the backbone's patch size; larger means a finer patch grid and
            more compute.
        frame_in_step: Which frame of a timestep's span represents it; see
            :func:`frame_for_step`.
        batch_size: Frames per backbone forward pass. Batches are filled
            across episode boundaries, so this is a pure memory/throughput
            knob.
        decode_workers: Clips decoded (and resized) in parallel threads while
            this thread runs the forward passes. Video decode dominates the
            wall clock -- see :meth:`_fill_camera` -- so this is the knob that
            matters for throughput; it cannot change the features.
        device: Torch device string; defaults to CUDA when available.
        l2_normalize: Unit-normalize each output row -- the whole row, so a
            multi-camera run is normalized once across its views rather than
            per view. Makes the features scale-free before the reduction stage,
            at the cost of discarding activation magnitude.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        pooling: str = "grid",
        grid: int = 2,
        cameras: list[str] | None = None,
        image_size: int = 224,
        frame_in_step: str = "first",
        batch_size: int = 64,
        decode_workers: int = 8,
        device: str | None = None,
        l2_normalize: bool = False,
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if pooling == "grid" and grid < 1:
            raise ValueError(f"grid must be >= 1, got {grid}")
        if frame_in_step not in FRAME_IN_STEP:
            raise ValueError(f"frame_in_step must be one of {FRAME_IN_STEP}, got {frame_in_step!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if decode_workers < 1:
            raise ValueError(f"decode_workers must be >= 1, got {decode_workers}")
        if isinstance(cameras, str):
            # `cameras: exterior` in a yaml would otherwise become one view per
            # character and fail per episode with "no video for camera 'e'".
            raise ValueError(
                f"cameras must be a list of view names, got the string {cameras!r}; "
                f"write it as [{cameras}]"
            )
        self.model_name = model_name
        self.pooling = pooling
        self.grid = int(grid)
        self.cameras = list(cameras) if cameras else None
        self.image_size = int(image_size)
        self.frame_in_step = frame_in_step
        self.batch_size = int(batch_size)
        self.decode_workers = int(decode_workers)
        self.device = device
        self.l2_normalize = bool(l2_normalize)
        self._backbone: _Backbone | None = None

    # ------------------------------------------------------------------ stage
    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return ``(total_steps, d)`` frame features, aligned with the episodes."""
        episodes = trajectories.episodes()
        cameras = self.cameras or [None]
        clips_per_camera = self._clips(trajectories, episodes, cameras)
        backbone = self._load()
        per_camera = backbone.output_dim
        total = sum(ep.num_samples for ep in episodes)
        features = np.zeros((total, per_camera * len(cameras)), dtype=np.float32)
        log.info(
            "%s: %d frames x %d view(s) -> %d dims (%s)",
            self.model_name,
            total,
            len(cameras),
            features.shape[1],
            self._pooling_repr(),
        )
        for camera_pos, clips in enumerate(clips_per_camera):
            columns = slice(camera_pos * per_camera, (camera_pos + 1) * per_camera)
            self._fill_camera(backbone, episodes, clips, features, columns)
        if self.l2_normalize:
            # Once over the finished row rather than per view: each view is
            # filled by its own pass, so normalizing inside the backbone would
            # leave a two-camera row with norm sqrt(2) and make the flag mean
            # something different at one view than at two.
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features /= np.maximum(norms, 1e-6)
        return FeatureSet(features, episodes)

    # -------------------------------------------------------------- internals
    def _pooling_repr(self) -> str:
        return f"grid {self.grid}x{self.grid}" if self.pooling == "grid" else self.pooling

    def _load(self) -> _Backbone:
        if self._backbone is None:
            self._backbone = _Backbone(
                self.model_name,
                pooling=self.pooling,
                grid=self.grid,
                image_size=self.image_size,
                device=self.device,
            )
        return self._backbone

    def _clips(
        self,
        trajectories: TrajectorySet,
        episodes: Sequence[EpisodeInfo],
        cameras: Sequence[str | None],
    ) -> list[dict[int, Path]]:
        """One ``{episode index: clip}`` map per camera, in *cameras* order.

        A missing clip is fatal rather than zero-filled: the pipeline's
        contract is one feature row per timestep, and rows of zeros would
        cluster together into a plausible-looking node that means nothing but
        "this episode was not recorded". Resolved up front, before the
        backbone is loaded, so a dataset with a gap fails in a second rather
        than after the first camera's worth of GPU work.
        """
        per_camera: list[dict[int, Path]] = []
        for camera in cameras:
            clips: dict[int, Path] = {}
            for episode in episodes:
                clip = self._clip(trajectories, episode, camera)
                if clip is None:
                    named = f" for camera {camera!r}" if camera is not None else ""
                    available = getattr(trajectories, "camera_names", lambda: [])()
                    raise FileNotFoundError(
                        f"episode {episode.index} of {trajectories.path} has no video"
                        f"{named}; {type(self).__name__} embeds recorded frames, so "
                        "every episode needs a clip"
                        + (f" (available views: {available})" if camera is not None else "")
                    )
                clips[episode.index] = clip
            per_camera.append(clips)
        return per_camera

    def _clip(
        self, trajectories: TrajectorySet, episode: EpisodeInfo, camera: str | None
    ) -> Path | None:
        """One episode's clip from one view, or None if it has none.

        ``camera=None`` goes through :meth:`TrajectorySet.video_path`, the only
        clip accessor every trajectory set implements; a named view needs a
        multi-view reader, which is duck-typed because ``video_paths`` lives on
        the canonical directory rather than on the ABC.
        """
        if camera is None:
            return trajectories.video_path(episode.index)
        video_paths = getattr(trajectories, "video_paths", None)
        if video_paths is None:
            raise TypeError(
                f"{type(trajectories).__name__} exposes no video_paths(); "
                f"cameras={self.cameras} needs a multi-view reader such as "
                "behavior_graphs.envs.trajectory_dir.TrajectoryDirectory"
            )
        return video_paths(episode.index, [camera]).get(camera)

    def _fill_camera(
        self,
        backbone: _Backbone,
        episodes: Sequence[EpisodeInfo],
        clips: dict[int, Path],
        features: np.ndarray,
        columns: slice,
    ) -> None:
        """Embed one view of every episode into ``features[:, columns]``.

        Decoding runs in a thread pool over *clips* while this thread does the
        forward passes, because decoding, not the GPU, is where the time went
        before this: a serial loop left the GPU idle the large majority of the
        time. Clips decode near-linearly in parallel (ffmpeg drops the GIL),
        which is where the default for *decode_workers* comes from;
        frame-level threading *within* a clip measured slower than serial on
        clips this short, so it is not used. Parallel decode returns the same
        frames as serial (there is a test for it). See
        ``docs/vision_backbone_features.md`` for the numbers.

        Batches span episode boundaries: clips are short (tens to hundreds of
        frames) and a per-episode flush would leave the GPU running half-empty
        batches for the whole dataset.
        """
        batch: list[np.ndarray] = []
        rows: list[list[int]] = []

        def flush() -> None:
            if not batch:
                return
            vectors = backbone.embed(batch)
            for vector, targets in zip(vectors, rows):
                features[targets, columns] = vector
            batch.clear()
            rows.clear()

        for ep, prepared in self._decoded_episodes(backbone, episodes, clips):
            for frame, frame_rows in prepared:
                batch.append(frame)
                rows.append(frame_rows)
                if len(batch) >= self.batch_size:
                    flush()
            log.debug("embedded episode %d, %d frames", ep.index, len(prepared))
        flush()

    def _decoded_episodes(
        self,
        backbone: _Backbone,
        episodes: Sequence[EpisodeInfo],
        clips: dict[int, Path],
    ):
        """Yield ``(episode, [(resized frame, rows)])`` in episode order.

        A sliding window of at most ``decode_workers + 1`` in-flight episodes,
        rather than submitting every episode at once: each pending result holds
        one episode's frames *already resized to* ``image_size`` (150 KB each,
        so ~150 MB for a 1000-step DROID episode, against 2.7 GB if they were
        held at 720p), and an unbounded queue of them would decode the whole
        dataset into memory before the first forward pass.
        """
        from concurrent.futures import ThreadPoolExecutor

        def decode(ep: EpisodeInfo):
            clip = clips[ep.index]
            reader = _clip_reader(clip)
            wanted = self._frame_rows(ep, reader.num_frames, clip.name)
            # Resized as they arrive, so what this future holds is the small
            # version: 150 KB a frame instead of 2.7 MB at 720p. Frame and rows
            # travel as one pair, so nothing downstream can misalign them.
            return ep, [
                (backbone.resize(frame), rows)
                for frame, rows in _frames_for_rows(reader, wanted, clip.name)
            ]

        with ThreadPoolExecutor(self.decode_workers) as pool:
            pending: list = []
            queue = list(episodes)
            while queue or pending:
                while queue and len(pending) <= self.decode_workers:
                    pending.append(pool.submit(decode, queue.pop(0)))
                yield pending.pop(0).result()

    def _frame_rows(self, episode: EpisodeInfo, num_frames: int, name: str) -> dict[int, list[int]]:
        """``{frame index: global feature rows}`` for one episode's clip.

        A dict of lists rather than one row per frame because a step-to-frame
        map is not injective in either direction: several steps share a frame
        when the video is shorter than the trajectory, and most frames are
        wanted by no step at all when it is longer.
        """
        if num_frames == 0:
            raise ValueError(f"could not read any frames from {name}")
        if num_frames < episode.num_samples:
            log.warning(
                "episode %d has %d steps but only %d frames in %s; steps will share frames",
                episode.index,
                episode.num_samples,
                num_frames,
                name,
            )
        wanted: dict[int, list[int]] = {}
        for step in range(episode.num_samples):
            frame_idx = frame_for_step(
                step, episode.num_samples, num_frames, where=self.frame_in_step
            )
            wanted.setdefault(frame_idx, []).append(episode.sample_start_idx + step)
        return wanted


class _Backbone:
    """A loaded ViT plus the preprocessing and pooling around it.

    Split out of the stage so the torch/transformers import stays inside the
    call that needs weights, and so the pooling arithmetic (the part with the
    interesting failure modes) can be tested against a stub model.
    """

    def __init__(
        self,
        model_name: str,
        *,
        pooling: str,
        grid: int,
        image_size: int,
        device: str | None,
    ) -> None:
        torch = _require_torch()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - dependency-group guard
            raise ImportError(
                "transformers is required to embed frames with a vision backbone. "
                "Install with: uv pip install -e '.[dino]'"
            ) from exc

        self.torch = torch
        self.pooling = pooling
        self.grid = grid
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        mean, std = _normalization(model_name)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {image_size} is not a multiple of {model_name}'s patch size "
                f"{self.patch_size}; the patch grid would not tile the frame"
            )
        self.patch_grid = image_size // self.patch_size
        self.hidden_size = int(self.model.config.hidden_size)
        log.info(
            "loaded %s on %s: %dx%d patches of %dpx, hidden %d",
            model_name,
            self.device,
            self.patch_grid,
            self.patch_grid,
            self.patch_size,
            self.hidden_size,
        )

    @property
    def output_dim(self) -> int:
        if self.pooling == "grid":
            return self.grid * self.grid * self.hidden_size
        if self.pooling == "tokens":
            return self.patch_grid * self.patch_grid * self.hidden_size
        return self.hidden_size

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """One RGB uint8 frame -> ``image_size x image_size`` uint8.

        Separate from :meth:`embed` because this is the half that runs in the
        decoder threads (see
        :meth:`VisionBackboneFeatures._decoded_episodes`): it is pure CPU work
        with the GIL dropped inside PIL, and doing it there both parallelizes
        it and shrinks what a pending episode holds in memory.
        """
        from PIL import Image

        size = (self.image_size, self.image_size)
        return np.asarray(Image.fromarray(frame).resize(size, Image.BICUBIC))

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """``(len(frames), output_dim)`` float32 features for :meth:`resize`d frames."""
        torch = self.torch
        pixel_values = torch.from_numpy(self._normalize(frames)).to(self.device)
        with torch.inference_mode():
            hidden = self.model(pixel_values=pixel_values).last_hidden_state
        return self._pool(hidden).float().cpu().numpy()

    def _normalize(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Resized uint8 frames -> normalized ``(B, 3, S, S)`` float32.

        The transpose happens *during* the uint8->float conversion rather than
        after it: ``stack().astype(float32).transpose()`` leaves a
        non-contiguous 38 MB view that the three arithmetic passes then walk
        with a stride-3 inner loop, which measured 165 ms a batch against 40 ms
        for this form -- in front of a 110 ms forward pass, i.e. most of the
        GPU's idle time. The operations and their order are unchanged, so this
        is the same arithmetic on the same values (checked directly against the
        strided form). Fusing the two constants into one multiply-add, or
        normalizing on the GPU (~1 ms a batch), would be faster still but would
        genuinely change the arithmetic, so neither is done here.

        A caveat that belongs on the record rather than in a claim of exact
        reproducibility: the *features* are not guaranteed byte-identical
        across machines or GPU conditions even from identical frames, because a
        fp32 convolution picks its algorithm partly by available workspace.
        Re-extracting a dataset first embedded on a GPU another job was filling
        reproduced 1600 of 1604 rows exactly, and four rows -- the one
        undersized tail batch -- to within 0.09. Repeated runs on an
        unconstrained GPU are exactly reproducible, and no comparison in this
        repo comes closer to the float than the metrics do.
        """
        batch = np.ascontiguousarray(np.stack(frames).transpose(0, 3, 1, 2), dtype=np.float32)
        batch /= 255.0
        batch -= self.mean
        batch /= self.std
        return batch

    def _pool(self, hidden: Any) -> Any:
        """``(B, tokens, d)`` -> ``(B, output_dim)``, keeping the layout documented above.

        The prefix tokens (CLS, and DINOv3's four registers) are identified by
        subtracting the patch count from the sequence length rather than
        assumed to be a single CLS, so a backbone with registers does not
        silently shift the whole patch map by four positions.
        """
        num_patches = self.patch_grid * self.patch_grid
        prefix = hidden.shape[1] - num_patches
        if prefix < 0:
            raise ValueError(
                f"backbone returned {hidden.shape[1]} tokens but a "
                f"{self.patch_grid}x{self.patch_grid} patch grid needs {num_patches}; "
                "image_size and the model's expected input disagree"
            )
        if self.pooling == "cls":
            if prefix == 0:
                raise ValueError(
                    f"pooling='cls' but {self.model.config.model_type} returned no prefix token"
                )
            return hidden[:, 0, :]
        patches = hidden[:, prefix:, :]
        if self.pooling == "mean":
            return patches.mean(dim=1)
        if self.pooling == "tokens":
            return patches.reshape(patches.shape[0], -1)
        # grid: (B, N, d) -> (B, d, g, g) -> (B, g, g, d) -> (B, g*g*d), so the
        # vector reads row-major over the pooled cells with each cell's channels
        # contiguous. Adaptive pooling rather than a reshape: the patch grid is
        # not generally divisible by `grid` (16 patches into a 3x3 map).
        batch, _, dim = patches.shape
        maps = patches.transpose(1, 2).reshape(batch, dim, self.patch_grid, self.patch_grid)
        pooled = self.torch.nn.functional.adaptive_avg_pool2d(maps, (self.grid, self.grid))
        return pooled.permute(0, 2, 3, 1).reshape(batch, -1)


def _normalization(model_name: str) -> tuple[Sequence[float], Sequence[float]]:
    """The backbone's channel mean/std, read straight from its hub config.

    Not through ``AutoImageProcessor``: the processor classes require
    torchvision as of transformers 5, and the resize/crop they would apply is
    not the one this stage wants anyway (see
    :class:`VisionBackboneFeatures`'s ``image_size``) -- the two normalization
    constants are the only thing worth taking from them. A repo without a
    preprocessor config falls back to the ImageNet statistics every DINO
    variant is trained with.
    """
    try:
        import json

        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_name, "preprocessor_config.json")) as fh:
            config = json.load(fh)
        return (
            config.get("image_mean") or IMAGENET_MEAN,
            config.get("image_std") or IMAGENET_STD,
        )
    except Exception as exc:
        # Reported at warning level with the cause, because two very different
        # things land here: a repo that genuinely ships no preprocessor config
        # (every DINO variant, whose stats *are* the ImageNet ones), and a
        # transient hub failure for a backbone whose stats are not. The second
        # embeds a whole dataset with the wrong channel statistics, and since
        # normalization is not part of the cache key the entry is otherwise
        # indistinguishable from a correct one.
        log.warning(
            "could not read %s's preprocessor config (%s: %s); falling back to ImageNet "
            "normalization, which is wrong for a backbone that does not use it",
            model_name,
            type(exc).__name__,
            exc,
        )
        return IMAGENET_MEAN, IMAGENET_STD


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - dependency-group guard
        raise ImportError(
            "torch is required to embed frames with a vision backbone. "
            "Install with: uv pip install -e '.[dino]'"
        ) from exc


def _require_imageio():
    try:
        import imageio.v3 as iio

        return iio
    except ImportError as exc:  # pragma: no cover - dependency-group guard
        raise ImportError(
            "imageio is required to read episode video. Install with: uv pip install -e '.[dino]'"
        ) from exc


#: Frames per decoder call. The torchcodec reader can fetch a whole episode in
#: one call, but a 1000-step DROID episode is 2.7 GB of 720p frames held at
#: once; measured throughput is flat from 16 to unbounded (970-1130 frames/s on
#: DROID footage), so this trades nothing for a memory ceiling.
_DECODE_CHUNK = 32


#: The decoder class this process settled on, probed once by
#: :func:`_clip_reader`. Racing decode threads may both probe; they reach the
#: same answer, so the duplicate work is a wasted import attempt and nothing
#: more.
_DECODER: type[_ClipReader] | None = None


def _pick_decoder() -> type[_ClipReader]:
    """Choose a decode backend for this process, and say which, once.

    torchcodec (what ``robo_dataset``'s VideoStore reads with) fetches an
    arbitrary list of frame indices in one C++ call, so it skips the RGB
    conversion and the Python-level copy of every frame the stage does *not*
    want -- which is most of them whenever a policy call spans several
    rendered env steps. It is several times faster on every dataset family
    here, most of all where the ratio is highest; see
    ``docs/vision_backbone_features.md`` for the measurements.

    The two backends deliver *the same frames* -- verified frame by frame over
    whole datasets, including the truncated-clip tail -- which is why the
    decoder is not part of the features cache key: choosing one or the other
    cannot change which pixels the backbone sees. (That is a statement about
    frames, not about float equality of the features themselves; see
    :meth:`_Backbone._normalize`.)

    Any failure to reach ``VideoDecoder`` demotes the process to PyAV, not
    just ``ImportError``: torchcodec needs FFmpeg's shared libraries on the
    system (PyAV's bundled build does not expose them) and raises
    ``RuntimeError`` from its own loader when they are missing -- an
    installed-but-unusable torchcodec is the likeliest way this goes wrong,
    and it is exactly the case the fallback exists for.
    """
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:
        log.warning("frame decode backend: PyAV (torchcodec unusable: %s)", exc)
        return _PyavReader
    log.info("frame decode backend: torchcodec")
    return _TorchcodecReader


def _clip_reader(clip: Path) -> _ClipReader:
    """Open *clip* with the backend this process picked, or PyAV if that fails.

    The per-clip fallback covers a container torchcodec cannot open but PyAV
    can; it warns rather than staying silent, because the same clip then costs
    a great deal more to read and that is worth seeing in a log.
    """
    global _DECODER
    if _DECODER is None:
        _DECODER = _pick_decoder()
    try:
        return _DECODER(clip)
    except Exception as exc:
        if _DECODER is _PyavReader:
            raise
        log.warning("%s: %s failed (%s); falling back to PyAV", clip.name, _DECODER.__name__, exc)
        return _PyavReader(clip)


class _ClipReader:
    """One clip's frame count plus access to chosen frame indices."""

    num_frames: int

    def frames(self, indices: Sequence[int]):
        """Yield ``(index, frame)`` in ascending order, for the indices given.

        Indices the stream turns out not to have are simply absent from the
        output; :func:`_frames_for_rows` backfills those rows. A reader may
        decode in batches internally (torchcodec does, to bound memory) but
        the caller sees one frame at a time, because that is all it wants.
        """
        raise NotImplementedError


class _TorchcodecReader(_ClipReader):
    """torchcodec's batched index API (``get_frames_at``)."""

    def __init__(self, clip: Path) -> None:
        from torchcodec.decoders import VideoDecoder

        # NHWC directly, so a frame arrives in the layout PIL and the rest of
        # the stage use; one ffmpeg thread per clip because the parallelism
        # that pays here is over clips (see _fill_camera), and robo_dataset hit
        # thread exhaustion letting each decoder auto-thread.
        self._decoder = VideoDecoder(str(clip), dimension_order="NHWC", num_ffmpeg_threads=1)
        self.num_frames = len(self._decoder)

    def frames(self, indices: Sequence[int]):
        wanted = [i for i in indices if 0 <= i < self.num_frames]
        for start in range(0, len(wanted), _DECODE_CHUNK):
            chunk = wanted[start : start + _DECODE_CHUNK]
            try:
                batch = self._decoder.get_frames_at(indices=chunk)
            except (RuntimeError, IndexError):
                # The frame count came from the container, and this stream
                # delivers fewer than it promised (an interrupted recording).
                # PyAV's sequential read just ends early here; torchcodec
                # raises on the first index it cannot decode, so retry the
                # chunk one frame at a time to keep the ones before it and let
                # _frames_for_rows backfill the rest.
                #
                # Which exception depends on the version: torchcodec raised
                # RuntimeError for an undecodable index and now raises
                # IndexError ("Invalid frame index=N for stream_index=0").
                # Catching only the old one turned a backfillable short clip
                # into a crash on any current install.
                yield from self._one_at_a_time(chunk)
                return
            yield from zip(chunk, batch.data.numpy())

    def _one_at_a_time(self, chunk: Sequence[int]):
        for idx in chunk:
            try:
                frame = self._decoder.get_frames_at(indices=[idx])
            except (RuntimeError, IndexError):
                return
            yield idx, frame.data.numpy()[0]


class _PyavReader(_ClipReader):
    """Sequential decode through imageio's PyAV plugin, dropping unwanted frames.

    One forward pass rather than a seek per index: PyAV has no batched index
    API, and seeking a normal-GOP MP4 per frame re-decodes from the preceding
    keyframe every time -- worse than reading through the clip once.
    """

    def __init__(self, clip: Path) -> None:
        self._clip = clip
        self._iio = _require_imageio()
        with self._iio.imopen(clip, "r", plugin="pyav") as video:
            shape = video.properties().shape
        declared = int(shape[0]) if shape and shape[0] else 0
        # A container need not store a frame count: imageio reports the video
        # stream's `frames` field, which is 0 for MPEG-TS, fragmented MP4 and
        # anything a killed recorder left behind. Iterating still delivers every
        # frame, so count them rather than declaring the episode unreadable --
        # this is the fallback decoder, i.e. the machine where torchcodec is
        # missing, and a second decode pass beats refusing the dataset.
        self.num_frames = declared or self._count_frames()
        if not declared and self.num_frames:
            log.info(
                "%s stores no frame count; counted %d frames by decoding",
                clip.name,
                self.num_frames,
            )

    def _count_frames(self) -> int:
        return sum(1 for _ in self._iter())

    def _iter(self):
        """One decode pass. ``thread_count=1`` for the same reason
        :class:`_TorchcodecReader` pins it: the parallelism that pays is over
        clips, and imageio's default (one thread per core) would multiply by
        ``decode_workers``."""
        return self._iio.imiter(self._clip, plugin="pyav", thread_count=1)

    def frames(self, indices: Sequence[int]):
        wanted = set(int(i) for i in indices)
        if not wanted:
            return
        last = max(wanted)
        for idx, frame in enumerate(self._iter()):
            if idx in wanted:
                yield idx, np.asarray(frame)
            if idx >= last:
                break


def _frames_for_rows(reader: _ClipReader, wanted: dict[int, list[int]], name: str):
    """Yield ``(frame, rows)`` for one clip, backfilling frames it turns out to lack.

    An MP4 container's declared frame count is not always its decodable one --
    a rollout recorder that was killed mid-write leaves a clip whose header
    promises more than the stream delivers. Since :meth:`_frame_rows` reads
    the declared count, the last few steps of such an episode can ask for
    frames that never arrive; those rows take the last frame that did, which
    is what the episode ended on anyway. Leaving them at zero instead would
    quietly plant a row of zeros in the middle of the feature matrix, and a
    cluster of them looks exactly like a real behavior.
    """
    remaining = dict(wanted)
    last: np.ndarray | None = None
    for idx, frame in reader.frames(sorted(wanted)):
        last = frame
        yield frame, remaining.pop(idx)
    if not remaining:
        return
    if last is None:
        raise ValueError(f"could not decode any frame of {name}")
    stranded = [row for rows in remaining.values() for row in rows]
    log.warning(
        "%s declares %d frames but decoded fewer; %d step(s) reuse its last frame",
        name,
        max(wanted) + 1,
        len(stranded),
    )
    yield last, stranded
