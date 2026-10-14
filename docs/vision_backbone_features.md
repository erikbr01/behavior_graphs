# Behavior graphs without a policy: DINO patch features

Every other feature source in the pipeline starts from
`policy_embeddings/<layer>.npz` — a forward pass through a trained
checkpoint. That makes a trained, *instrumented* policy a precondition for
building a graph at all: no checkpoint, no embeddings, no graph. It also ties
the representation to one policy, so two policies' graphs live in different
spaces and can only be compared through their behavior, never their features.

`VisionBackboneFeatures`
(`src/behavior_graphs/stages/features/vision_backbone.py`) removes that
precondition. It reads the episode **videos** every rollout directory already
stores and embeds them with a frozen general-purpose vision backbone —
DINOv2 by default:

```bash
python -m behavior_graphs.pipeline.run rollout_dir=<rollouts> \
    pipeline/features@features=dino_patch
```

What you get is a graph over *what the scene looked like* rather than over
what the policy was thinking. That is weaker evidence about the policy's
internal structure and stronger evidence about the task: it applies to a
policy nobody instrumented, a stack whose internals are unreachable (an API
model), and human demonstrations, and it puts several policies on one shared
representation.

## Patch tokens, not the CLS token

A ViT's CLS token — and the mean over its patch tokens — is an image-level
summary trained to be invariant to exactly what a behavior graph needs to
see: *where* the arm and the objects are. Two frames of the same kitchen with
the gripper at the pot and at the drawer are nearly identical under mean
pooling. So `pooling="grid"` average-pools the 16×16 patch map down to a
coarse `grid × grid` map and keeps it laid out, spending feature width on
retained spatial structure:

| `pooling` | Width (DINOv2-base) | What it keeps |
|---|---|---|
| `grid`, `grid: 2` (default) | 4 × 768 = 3072 | Quadrant-level layout |
| `grid`, `grid: 4` | 16 × 768 = 12288 | Finer layout, 4× the width |
| `mean` (= `grid: 1`) | 768 | Image-level content only |
| `cls` | 768 | Image-level content only |
| `tokens` | 256 × 768 = 196608 | Every patch (impractical downstream) |

The vector reads row-major over the pooled cells with each cell's channels
contiguous, so a cell is a contiguous slice — the layout the reduction stage
sees.

Two configs ship, so the claim above is testable rather than assumed:
`dino_patch` (2×2 grid) and `dino_pooled` (its mean-pooled baseline).
`configs/grid/dino.yaml` sweeps them against each other over the standard
window/reduction/K dimensions (270 jobs per dataset):

```bash
python scripts/run_sweep.py grid=full 'datasets={coffee: <rollouts>}'
```

The policy's own embeddings are deliberately *not* in that grid: the sweep
runner injects `features.layer=<layer>` only when every features option takes
a layer, so a mixed grid would hand a layer to an extractor that has none.
Run the baseline as a second pass into the same sweep dir and compare:

```bash
python scripts/run_sweep.py grid=full 'datasets={coffee: <rollouts>}' \
    layer=<policy layer> sweep_dir=<same dir>
```

To put a frame extractor into any other grid, name it in `extra_overrides`:

```bash
python scripts/run_sweep.py grid=quick 'datasets={coffee: <rollouts>}' \
    'extra_overrides=[pipeline/features@features=dino_patch]'
```

## Does it work?

The paper's ablation (Tab. III) is the measured answer, over the three DROID
tasks of the user study and against gripper-derived subtask boundaries: frozen
DINOv3 reaches $F_1(0.75)$ of 0.113 averaged over reductions, against 0.132 for
proprioception and 0.226 for the policy's own embeddings. It is a usable
representation rather than a better one — which is the useful result, since it
is the only one of the three that needs no policy checkpoint at all.

Reproduce it with `scripts/ablation_study/paper_ablation.py`; the grid that
produced it is `configs/grid/ablation_droid.yaml` and the representation is
`dinov3` in each task's `configs/ablation_study/<task>.yaml`.

## Frames and timesteps

A trajectory row is one *policy call*; a video frame is one *env step*. With
action chunking those differ — mimicgen rollouts are 4 frames per row — so
the episode's rows are laid over its frames proportionally and each row takes
one frame of its span (`frame_in_step`):

- `first` (default) — the observation the chunk was actually computed from.
- `last` — what the chunk led to.
- `middle` — between the two.

All three coincide when the recorder wrote one frame per row. A clip whose
container over-reports its length (an interrupted recording) backfills its
last decoded frame with a warning, rather than leaving rows of zeros that
would cluster into a plausible-looking node meaning "not recorded". A missing
clip is a hard error: the pipeline's contract is one feature row per
timestep.

Frames are resized to `image_size × image_size` with **no** center crop. The
processor default (resize the short side, then crop) would throw away the
sides of a widescreen view; for spatial features the whole field of view
matters more than the aspect ratio.

## Multiple views

`cameras: [exterior_image_1_left, wrist_image_left]` embeds several views of
each episode and concatenates them, in the order given (DROID's wrist view
plus an exterior one). `null`, the default, is the single view the trajectory
reader returns — the only thing every `TrajectorySet` implementation
supports. Every extra view multiplies both cost and feature width.

## Cost: it is a video-decoding job, not a GPU job

Profiled on DROID footage (640×360, batch 64), the three stages of the loop
are wildly unbalanced. (The two tables below were measured at different times
on a shared box, so read the ratios rather than the absolutes — what matters is
which stage is orders of magnitude off the others, and that ordering was stable
across every run.)

| stage | throughput | share of a serial loop |
|---|---|---|
| H.264 decode, 1 thread | 48 f/s | 66% |
| PIL resize to 224² | 112 f/s | 28% |
| DINOv2-base forward, fp32 | 577 f/s | 5% |

So the GPU is not the constraint and fp16 (1435 f/s) would buy nothing. Two
things fix the parts that are:

**Decode with torchcodec** (`_clip_reader`), the same decoder
`robo_dataset`'s `VideoStore` reads with. Its `get_frames_at(indices=[...])`
fetches exactly the frames this stage wants in one C++ call, skipping the RGB
conversion and Python-level copy of every frame it would otherwise decode and
throw away — which is most of them whenever one policy call spans several
rendered env steps. Measured on one worker, wanted-frames/s:

| dataset (frames per step) | PyAV sequential | torchcodec |
|---|---|---|
| set_table DIT (1×, 640×360) | 145 | 554 |
| mimicgen (4×, 128²) | 105 | 663 |
| vla_foundry (7×, 1920×960) | 10.8 | 20.1 |

**Decode clips in parallel** (`decode_workers`, default 8), since ffmpeg drops
the GIL: 32 → 128 → 235 f/s at 1 → 4 → 8 workers. Resizing happens inside
those workers too, which both parallelizes it and keeps a pending episode at
150 KB a frame instead of 2.7 MB.

Together, end to end on 720p footage: **23 → 163 rows/s**. Frames from the two
decoders are bit-identical (verified on all four dataset families, and by
`tests/test_vision_backbone_features.py`), which is why the decoder is *not*
part of the cache key — entries computed under either backend are
interchangeable. torchcodec needs FFmpeg's shared libraries on the system; without them it is
installed but unusable and raises from its own loader rather than failing to
import, so the stage demotes itself to PyAV on *any* failure to reach the
decoder — and again per clip, for a container one backend rejects and the
other reads. Either way it only gets slower, and it says so in the log.

**Reproducibility.** The decoders return the same *frames* — checked frame by
frame over whole datasets (coffee 1604, set_table 27 396
frames: no mismatches), which is what licenses keeping the decoder out of the
cache key. The *features* are a weaker claim: repeated extractions on this box
are exactly reproducible, and `batch_size` provably does not change a value,
but a fp32 convolution picks its algorithm partly by available GPU workspace,
so re-extracting a dataset first embedded on a GPU another job was filling
reproduced 1600 of 1604 rows exactly and four — the one undersized tail batch —
to within 0.09. Nothing downstream reads features that closely; it is recorded
here so nobody later reads byte-equality into a cache hit.

**Camera orientation is deliberately not applied.** `configs/data/<group>.yaml`
declares `rotate`/`mirror` per view (DROID's wrist camera hangs upside down) and
the app bakes it into its proxies, but that is display geometry a stage cannot
see, and a consistently inverted camera is still consistent within a dataset —
which is all the clustering needs. A backbone is not rotation-invariant, so if
wrist views ever join the feature set this is worth revisiting.

CUDA decode (NVDEC) is not used: the wheel we pin rejects a CUDA device, and
with decode fixed the forward pass is the limit anyway. Measured after both
changes, on the same DROID footage: torchcodec decode alone reaches 9970 f/s at
8 workers and 1550 f/s with the resize, against a forward pass that does 577
f/s on an idle GPU and **85 f/s on a GPU another job is saturating** — so an
extraction that looks slow is now worth checking with `nvidia-smi` before
blaming the decoder. `_clip_reader` logs which backend it picked, once per
process, for the same reason. fp16 would roughly double the forward (1435 f/s
measured) at the cost of changing every feature value, so it is deliberately
not offered while a study is in flight.

## Caching and materializing a layer

The features stage is content-addressed like every other
(`pipeline/cache.py`), so a sweep embeds a dataset once and reuses it — but
note that the whole stage config is the cache key, `batch_size`,
`decode_workers` and `device` included: retuning throughput re-embeds the
dataset (identical values, wasted time).

Everything in the repo *outside* the pipeline addresses representations by
layer name (the demo app's layer picker, the sweep runner's per-layer passes,
the outcome classifiers). To make the frame features available to those, bake
them into the trajectory directory as another layer:

```bash
python scripts/utils/extract_video_features.py rollout_dir=<rollouts>
# -> <rollouts>/policy_embeddings/dino_patch.npz

python scripts/utils/extract_video_features.py rollout_dir=<rollouts> \
    pipeline/features@features=dino_pooled layer=dino_pooled
```

The layer is added in place beside the existing ones; an existing layer of
the same name is kept unless `overwrite=true`. Afterwards the dataset behaves
like any other instrumented one — `features.layer=dino_patch`, no GPU in the
loop.

## Backbones

Any HF ViT whose `last_hidden_state` is `[prefix tokens…, patch tokens…]`
works; the prefix count (CLS, plus DINOv3's four registers) is inferred from
the patch grid rather than assumed, so a backbone with registers does not
silently shift the patch map.

- `facebook/dinov2-small|base|large` — ungated, the default.
- `facebook/dinov3-vitb16-pretrain-lvd1689m` — needs an accepted license on
  the hub and `transformers>=4.56`.

Install the dependencies with `uv pip install -e '.[dino]'` (torch,
transformers, imageio, pillow).
