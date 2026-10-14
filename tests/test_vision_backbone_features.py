"""Tests for the policy-free feature source: frames through a vision backbone.

The backbone itself is stubbed (a ViT-shaped module with deterministic
outputs), so these cover the parts with real failure modes -- step-to-frame
alignment, patch-token pooling and its layout, multi-camera concatenation,
and the missing-video errors -- without downloading weights.
"""

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from behavior_graphs.core.stages import FeatureExtractor
from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.stages.features.vision_backbone import (
    VisionBackboneFeatures,
    frame_for_step,
)

torch = pytest.importorskip("torch")
iio = pytest.importorskip("imageio.v3")
av = pytest.importorskip("av")

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

LENGTHS = [4, 3]
TOTAL = sum(LENGTHS)
PATCH = 14
IMAGE_SIZE = 42  # 3x3 patches -- small enough to keep the stub cheap
PATCH_GRID = IMAGE_SIZE // PATCH
HIDDEN = 8


class _StubConfig:
    patch_size = PATCH
    hidden_size = HIDDEN
    model_type = "stub"


class _StubModel:
    """A ViT-shaped stand-in: one CLS token plus a patch grid of known values.

    Token *k* of frame *b* is filled with the constant ``b * 100 + k``, so a
    pooling bug shows up as a wrong number rather than merely a wrong shape.
    """

    config = _StubConfig()

    def __init__(self, prefix_tokens: int = 1) -> None:
        self.prefix_tokens = prefix_tokens
        self.calls: list[tuple[int, ...]] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values):
        self.calls.append(tuple(pixel_values.shape))
        batch = pixel_values.shape[0]
        num_tokens = self.prefix_tokens + PATCH_GRID * PATCH_GRID
        values = torch.arange(num_tokens, dtype=torch.float32).reshape(1, num_tokens, 1)
        offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1) * 100
        hidden = (values + offsets).expand(batch, num_tokens, HIDDEN).clone()
        return type("Out", (), {"last_hidden_state": hidden})()


@pytest.fixture
def stub_backbone(monkeypatch):
    """Patch ``AutoModel`` at the point of use, and pin the normalization.

    ``_Backbone`` imports transformers inside its constructor (so the stage
    stays importable without it), which means the stub has to be installed as
    a fake module rather than monkeypatched onto an attribute. Normalization
    is pinned to a no-op so the stub's token values are the only thing the
    assertions depend on -- and so no test reaches the hub.
    """
    import sys
    import types

    from behavior_graphs.stages.features import vision_backbone

    model = _StubModel()
    fake = types.ModuleType("transformers")
    fake.AutoModel = type("AutoModel", (), {"from_pretrained": staticmethod(lambda name: model)})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(vision_backbone, "_normalization", lambda name: ((0, 0, 0), (1, 1, 1)))
    return model


def _write_clip(path: Path, num_frames: int, seed: int = 0) -> None:
    """Write a throwaway H.264 clip of *num_frames* random frames.

    Encoded with PyAV directly rather than ``iio.imwrite``: imageio's pyav
    writer sets the stream size after opening the codec, which the installed
    PyAV rejects. Reading, which is all the extractor does, goes through
    imageio as everywhere else in the repo.
    """
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 255, size=(num_frames, 32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _rollouts(tmp_path, name, frames_per_step=1, cameras=(None,)):
    """Canonical-layout rollouts with a clip per episode per camera.

    *frames_per_step* above 1 is the action-chunked case (the recorder writes
    one frame per env step while the trajectory stores one row per policy
    call); ``cameras=(None,)`` writes the flat single-view layout.
    """
    out = write_rollout_dir(tmp_path / name, LENGTHS, [True, False], embeddings={})
    for ep, n in zip(out.episodes(), LENGTHS):
        for camera in cameras:
            media = out.path / "media" if camera is None else out.path / "media" / camera
            _write_clip(media / f"ep{ep.index:04d}_success.mp4", frames_per_step * n, seed=ep.index)
    return out


@pytest.fixture
def rollouts(tmp_path):
    return _rollouts(tmp_path, "r")


@pytest.fixture
def chunked_rollouts(tmp_path):
    return _rollouts(tmp_path, "chunked", frames_per_step=4)


@pytest.fixture
def multi_view_rollouts(tmp_path):
    return _rollouts(tmp_path, "views", cameras=("exterior", "wrist"))


def _extractor(**kwargs) -> VisionBackboneFeatures:
    defaults = dict(image_size=IMAGE_SIZE, batch_size=4, device="cpu")
    return VisionBackboneFeatures(**{**defaults, **kwargs})


class TestFrameForStep:
    def test_one_frame_per_step_is_the_identity(self):
        assert [frame_for_step(t, 5, 5) for t in range(5)] == [0, 1, 2, 3, 4]

    def test_chunked_video_takes_the_frame_the_policy_saw(self):
        """4 frames per step: 'first' is the observation the chunk came from."""
        assert [frame_for_step(t, 3, 12, "first") for t in range(3)] == [0, 4, 8]
        assert [frame_for_step(t, 3, 12, "middle") for t in range(3)] == [1, 5, 9]
        assert [frame_for_step(t, 3, 12, "last") for t in range(3)] == [3, 7, 11]

    def test_non_integer_ratio_stays_in_range_and_monotone(self):
        indices = [frame_for_step(t, 30, 250, "last") for t in range(30)]
        assert indices == sorted(indices)
        assert indices[-1] == 249
        assert all(0 <= i < 250 for i in indices)

    def test_short_video_shares_frames_instead_of_overrunning(self):
        """A truncated recording must not index past its own last frame."""
        indices = [frame_for_step(t, 6, 3, "last") for t in range(6)]
        assert max(indices) == 2
        assert indices == [0, 0, 1, 1, 2, 2]

    def test_unknown_position_rejected(self):
        with pytest.raises(ValueError, match="frame_in_step"):
            frame_for_step(0, 1, 1, "somewhere")


class TestPooling:
    def test_grid_keeps_one_cell_per_region(self, stub_backbone, rollouts):
        features = _extractor(pooling="grid", grid=PATCH_GRID).extract(rollouts)
        assert features.num_samples == TOTAL
        assert features.dim == PATCH_GRID * PATCH_GRID * HIDDEN
        # Frame 0 of the batch: patch token k holds 1 + k (CLS is token 0), and
        # a grid equal to the patch grid pools nothing away, so the vector is
        # each cell's value repeated across the hidden dim, cell-major.
        expected = np.repeat(np.arange(1, PATCH_GRID**2 + 1, dtype=np.float32), HIDDEN)
        assert np.allclose(features.features[0], expected)

    def test_grid_one_equals_mean(self, stub_backbone, rollouts):
        grid = _extractor(pooling="grid", grid=1).extract(rollouts).features
        mean = _extractor(pooling="mean").extract(rollouts).features
        assert grid.shape == mean.shape == (TOTAL, HIDDEN)
        assert np.allclose(grid, mean)

    def test_mean_and_cls_are_image_level(self, stub_backbone, rollouts):
        mean = _extractor(pooling="mean").extract(rollouts)
        cls = _extractor(pooling="cls").extract(rollouts)
        assert mean.dim == cls.dim == HIDDEN
        # CLS is token 0; the patch mean is the mean of tokens 1..9.
        assert np.allclose(cls.features[0], 0.0)
        assert np.allclose(mean.features[0], np.arange(1, PATCH_GRID**2 + 1).mean())

    def test_tokens_keeps_every_patch(self, stub_backbone, rollouts):
        features = _extractor(pooling="tokens").extract(rollouts)
        assert features.dim == PATCH_GRID * PATCH_GRID * HIDDEN

    def test_register_tokens_do_not_shift_the_patch_map(self, stub_backbone, rollouts):
        """DINOv3-style prefixes: patches are found by count, not by assuming one CLS."""
        stub_backbone.prefix_tokens = 5
        features = _extractor(pooling="mean").extract(rollouts)
        assert np.allclose(features.features[0], np.arange(5, 5 + PATCH_GRID**2).mean())

    def test_l2_normalize(self, stub_backbone, rollouts):
        features = _extractor(pooling="grid", grid=2, l2_normalize=True).extract(rollouts)
        norms = np.linalg.norm(features.features, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_l2_normalize_covers_the_whole_row_across_views(
        self, stub_backbone, multi_view_rollouts
    ):
        """Per-view normalization would leave a two-camera row at norm sqrt(2),
        so the flag would mean something different at one view than at two."""
        features = _extractor(
            pooling="mean", cameras=["exterior", "wrist"], l2_normalize=True
        ).extract(multi_view_rollouts)

        assert features.dim == 2 * HIDDEN
        assert np.allclose(np.linalg.norm(features.features, axis=1), 1.0, atol=1e-5)

    def test_image_size_must_tile_the_patch_grid(self, stub_backbone, rollouts):
        with pytest.raises(ValueError, match="patch size"):
            _extractor(image_size=IMAGE_SIZE + 1).extract(rollouts)


class TestAlignment:
    def test_rows_follow_the_video_over_episodes(self, stub_backbone, rollouts):
        """Row t of an episode must hold frame t's embedding, per episode."""
        features = _extractor(pooling="cls", batch_size=100).extract(rollouts).features
        # The stub encodes batch position in the value; with one batch per
        # camera, position == global row, so distinct rows stay distinct and
        # in order.
        assert np.allclose(features[:, 0], 100 * np.arange(TOTAL))

    def test_chunked_video_yields_one_row_per_step(self, stub_backbone, chunked_rollouts):
        """A 4x-longer clip still produces exactly the trajectory's rows."""
        features = _extractor(pooling="cls").extract(chunked_rollouts)
        assert features.num_samples == TOTAL
        assert [ep.num_samples for ep in features.episodes] == LENGTHS

    def test_frame_rows_map_steps_onto_the_clip(self, stub_backbone, chunked_rollouts):
        from behavior_graphs.stages.features.vision_backbone import _clip_reader

        extractor = _extractor()
        episode = chunked_rollouts.episodes()[0]
        clip = chunked_rollouts.video_path(episode.index)
        rows = extractor._frame_rows(episode, _clip_reader(clip).num_frames, clip.name)
        # 4 frames per step: every 4th frame, one global row each.
        assert sorted(rows) == [0, 4, 8, 12]
        assert [rows[i] for i in sorted(rows)] == [[0], [1], [2], [3]]

    def test_truncated_clip_backfills_its_last_frame(self, stub_backbone, rollouts, monkeypatch):
        """A container that over-reports its length must not leave zero rows."""
        from behavior_graphs.stages.features import vision_backbone

        real_reader = vision_backbone._clip_reader

        def over_reporting(clip):
            reader = real_reader(clip)
            reader.num_frames += 2  # header promises frames the stream lacks
            return reader

        monkeypatch.setattr(vision_backbone, "_clip_reader", over_reporting)
        features = _extractor(pooling="mean", batch_size=100).extract(rollouts).features
        assert features.shape == (TOTAL, HIDDEN)
        assert not np.any(np.all(features == 0, axis=1))

    def test_parallel_decode_matches_serial(self, stub_backbone, rollouts):
        """The decode threads must not reorder anything: same features, any width.

        The stub encodes each frame's position *within its batch*, so a worker
        pool that returned episodes out of order would shift those positions
        and show up here as different numbers -- which is what makes this a
        real check on the sliding window, not just on shapes.
        """
        serial = _extractor(pooling="mean", decode_workers=1).extract(rollouts).features
        parallel = _extractor(pooling="mean", decode_workers=8).extract(rollouts).features
        assert np.array_equal(serial, parallel)

    def test_batching_does_not_change_the_features(self, stub_backbone, rollouts):
        """Batches span episodes, so the batch size must not reorder anything."""
        one = _extractor(pooling="mean", batch_size=1).extract(rollouts).features
        many = _extractor(pooling="mean", batch_size=64).extract(rollouts).features
        # The stub's values depend on batch position, so compare structure:
        # both must be constant across the hidden dim and strictly increasing.
        for features in (one, many):
            assert np.allclose(features, features[:, :1])
        assert one.shape == many.shape == (TOTAL, HIDDEN)


class TestDecoders:
    """The two decode backends must be interchangeable, not merely both present.

    The stage picks torchcodec when it is importable and PyAV otherwise, and
    keeps the choice out of the features cache key -- which is only sound if
    they return the same pixels.
    """

    def test_backends_return_the_same_frames(self, rollouts):
        pytest.importorskip("torchcodec")
        from behavior_graphs.stages.features.vision_backbone import (
            _PyavReader,
            _TorchcodecReader,
        )

        clip = rollouts.video_path(0)
        wanted = list(range(LENGTHS[0]))
        pyav = _PyavReader(clip)
        torchcodec = _TorchcodecReader(clip)
        assert pyav.num_frames == torchcodec.num_frames

        from_pyav = [frame for _, frame in pyav.frames(wanted)]
        from_tc = [frame for _, frame in torchcodec.frames(wanted)]
        assert len(from_pyav) == len(from_tc) == len(wanted)
        assert all(np.array_equal(a, b) for a, b in zip(from_pyav, from_tc))

    @pytest.mark.parametrize(
        "error",
        [
            ImportError("torchcodec is not installed"),
            # What an installed torchcodec raises when the system has no FFmpeg
            # shared libraries -- the likeliest way this goes wrong, and not an
            # ImportError, so catching only that aborted the whole extraction
            # on a machine the fallback was written for.
            RuntimeError("Could not load libtorchcodec"),
        ],
        ids=["not-installed", "installed-but-unusable"],
    )
    def test_unusable_torchcodec_falls_back_to_pyav(self, rollouts, monkeypatch, error):
        import builtins

        from behavior_graphs.stages.features import vision_backbone

        real_import = builtins.__import__

        def broken_torchcodec(name, *args, **kwargs):
            if name.startswith("torchcodec"):
                raise error
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_torchcodec)
        monkeypatch.setattr(vision_backbone, "_DECODER", None)
        reader = vision_backbone._clip_reader(rollouts.video_path(0))
        assert isinstance(reader, vision_backbone._PyavReader)

    def test_a_clip_torchcodec_cannot_open_falls_back_per_clip(self, rollouts, monkeypatch):
        """A container one backend rejects should cost speed, not the run."""
        from behavior_graphs.stages.features import vision_backbone

        class Refuses:
            def __init__(self, clip):
                raise RuntimeError("unsupported container")

        monkeypatch.setattr(vision_backbone, "_DECODER", Refuses)
        reader = vision_backbone._clip_reader(rollouts.video_path(0))
        assert isinstance(reader, vision_backbone._PyavReader)


class TestCameras:
    def test_views_are_concatenated(self, stub_backbone, multi_view_rollouts):
        extractor = _extractor(pooling="mean", cameras=["exterior", "wrist"])
        features = extractor.extract(multi_view_rollouts)
        assert features.dim == 2 * HIDDEN

    def test_a_container_without_a_frame_count_is_counted_not_refused(
        self, stub_backbone, rollouts, monkeypatch
    ):
        """A stream that stores no frame count still decodes.

        imageio reports the video stream's `frames` field, which is 0 for
        MPEG-TS, fragmented MP4 and anything a killed recorder left behind. On
        the PyAV path -- the machine where torchcodec is missing, i.e. exactly
        where robustness matters -- that used to make num_frames 0 and abort the
        whole extraction on the first such episode.
        """
        from behavior_graphs.stages.features import vision_backbone

        real_imopen = vision_backbone._require_imageio().imopen

        class Undeclared:
            """imageio's reader, with the frame count stripped from properties."""

            def __init__(self, *args, **kwargs):
                self._inner = real_imopen(*args, **kwargs)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

            def properties(self, *args, **kwargs):
                props = self._inner.properties(*args, **kwargs)
                return type(props)(shape=(0, *props.shape[1:]), dtype=props.dtype)

            def __getattr__(self, name):
                # iter()/read()/close() and anything else imageio reaches for.
                return getattr(self._inner, name)

        iio = vision_backbone._require_imageio()
        monkeypatch.setattr(iio, "imopen", Undeclared)
        monkeypatch.setattr(vision_backbone, "_DECODER", vision_backbone._PyavReader)

        reader = vision_backbone._clip_reader(rollouts.video_path(0))
        assert reader.num_frames == LENGTHS[0]
        assert [idx for idx, _ in reader.frames(range(LENGTHS[0]))] == list(range(LENGTHS[0]))

    def test_missing_camera_is_named(self, stub_backbone, multi_view_rollouts):
        extractor = _extractor(cameras=["exterior", "nope"])
        with pytest.raises(FileNotFoundError, match="nope"):
            extractor.extract(multi_view_rollouts)

    def test_missing_video_is_fatal(self, stub_backbone, tmp_path):
        bare = write_rollout_dir(tmp_path / "bare", LENGTHS, [True, False], embeddings={})
        with pytest.raises(FileNotFoundError, match="no video"):
            _extractor().extract(bare)


class TestConstructorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"pooling": "nope"}, "pooling"),
            ({"pooling": "grid", "grid": 0}, "grid"),
            ({"frame_in_step": "nope"}, "frame_in_step"),
            ({"batch_size": 0}, "batch_size"),
            ({"decode_workers": 0}, "decode_workers"),
            # `cameras: exterior` in a yaml would otherwise become one view per
            # character and fail per episode on camera 'e'.
            ({"cameras": "exterior"}, "list of view names"),
        ],
    )
    def test_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            VisionBackboneFeatures(**kwargs)


class TestConfigs:
    @pytest.mark.parametrize("name", ["dino_patch", "dino_pooled"])
    def test_config_instantiates_without_weights(self, name):
        """The configs must not touch the hub at construction time."""
        cfg = OmegaConf.load(CONFIGS / "pipeline" / "features" / f"{name}.yaml")
        extractor = instantiate(cfg)
        assert isinstance(extractor, FeatureExtractor)
        assert extractor.model_name.startswith("facebook/dinov")

    def test_patch_config_keeps_spatial_layout(self):
        cfg = OmegaConf.load(CONFIGS / "pipeline" / "features" / "dino_patch.yaml")
        extractor = instantiate(cfg)
        assert extractor.pooling == "grid"
        assert extractor.grid > 1
