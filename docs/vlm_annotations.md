# VLM Node Annotations

Behavior graph nodes can be auto-labeled with a VLM using contact sheets
(grids of frames sampled from the rollout videos). Labels are written to
`<run_dir>/node_labels.json` and immediately visible in the demo app's node
inspector.

The labeler uses a two-pass seeded-relabeling pipeline (Macrodata Labs): pass
1 generates chain-of-thought draft labels from contact sheets and graph
context; pass 2 refines each draft by re-prompting the VLM with its own prior
output. This improves accuracy from ~61% → ~78% over single-pass labeling.

## One-time setup

**1. Configure environment variables**

```bash
bash scripts/setup_env.sh        # writes .env with DATA_PATH + Vertex AI defaults
source .env
```

The script writes a root `.env` (gitignored) with:

```bash
export DATA_PATH=...
export GOOGLE_CLOUD_PROJECT=gcp-driven-data
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
```

Edit `GOOGLE_CLOUD_PROJECT` if your GCP project differs.

**2. Authenticate with Google Cloud**

```bash
gcloud auth application-default login
```

This writes Application Default Credentials to
`~/.config/gcloud/application_default_credentials.json`. The `google-genai`
SDK picks them up automatically — no API key file needed.

**3. Install the VLM dependency**

```bash
uv pip install google-genai
```

## Label a run

```bash
source .env
python scripts/label_nodes.py <run_dir> \
    --rollout-dir <rollout_dir> \
    --task "pick up the hammer and place it in the right container"
```

On a multi-camera dataset the sheets are built from one view unless you name
more:

```bash
python scripts/label_nodes.py <run_dir> --list-cameras
python scripts/label_nodes.py <run_dir> \
    --camera exterior_image_1_left --camera wrist_image_left
```

See [Camera views](#camera-views) for what the extra views buy and cost. A
`--camera` the rollout dir doesn't have is an error rather than an empty row —
a typo would otherwise buy a full-price pass over sheets missing the view it
was run for.

The default model is `gemini-3.5-flash` via Vertex AI. Override with
`--model`:

| `--model` value | Backend | Install |
|---|---|---|
| `gemini-3.5-flash` *(default)* | Vertex AI / Google AI Studio | `google-genai` |
| `claude-opus-4-8` | Anthropic | `anthropic` |
| `ollama:<model>` | Ollama (local) | `openai` + Ollama running |
| `openai:<model>@<url>` | Any OpenAI-compat endpoint (LM Studio, vLLM, …) | `openai` |
| `gpt-4o` | OpenAI | `openai` |

For Ollama or LM Studio, no cloud auth is needed — just make sure the server
is running and the model is pulled.

## Label a run from the demo app

The Graph Demo page has a **🏷️ VLM annotations** button in the sidebar, under
the run picker, labeled with the run's progress (`3/20 labeled`). It
opens a modal to pick the VLM and give an optional annotation prompt — the
same task context `--task` passes on the CLI; leave it blank to let the VLM
infer the task from successful episodes. A multi-camera dataset also gets a
**Camera views** picker there.

**Sampling settings** holds the rest of the labeler's knobs, defaulted from
`configs/pipeline/labeling/vlm.yaml`:

| Setting | Meaning |
|---|---|
| Reasoning effort | Gemini's `thinking_level` — `minimal`/`low`/`medium`/`high`. Billed as output tokens and usually the bulk of a run's cost. Hidden for non-Gemini models, which have no equivalent. |
| Frames per episode | Frames sampled from each episode that visits a node, forming one labeled row of that node's contact sheet. |
| Max episodes | Cap on how many episodes contribute a row, so a node's sheet holds at most `frames_per_episode × max_episodes × views` tiles. |

**Camera views** sits above the expander, on datasets that have more than one —
see [Camera views](#camera-views).

Each node gets exactly one contact sheet, with one labeled row-group per
episode (`Episode 7 - frames 12-31`). Frames are counted *per episode*, not as
a total split across them, so a node visited twice reads at the same detail as
one visited eight times — the sheet grows instead of the coverage thinning.

The model list comes from `configs/vlm/models.yaml`; the preselected one is
`model:` from `configs/pipeline/labeling/vlm.yaml`, so the modal, the CLI and
the pipeline stage all default to the same labeler.

Runs that already have labels show how many, and the action button turns
orange **Overwrite annotations** — the labeler relabels every behavior node,
so existing labels are replaced.

## Camera views

A dataset whose `media/` has per-camera subdirectories (see
`docs/architecture.md`, and `TrajectoryDirectory.camera_names()`) can put more
than one view of each episode on a node's sheet. Every sampled episode then
contributes **one labeled row-group per view**, sampled at the *same* timesteps:

```
Episode 7 [exterior_image_1_left] - frames 12-31
Episode 7 [wrist_image_left] - frames 12-31
```

A column therefore holds one moment across views, and both labeling passes are
told as much — without that, a model reads the second group as a later phase of
the episode and describes a sequence that never happened. Views are oriented
the way the app's players show them, using the same per-camera
`rotate`/`mirror` from `configs/data/` that the web proxies bake in: DROID's
wrist camera hangs upside down, and inverted frames cost more label quality
than they add.

Where to set it:

| Entry point | How |
|---|---|
| Graph Demo modal | **Camera views** multiselect, opening on the first view the sidebar is showing |
| `scripts/label_nodes.py` | `--camera NAME`, repeatable and ordered; `--list-cameras` to see what a rollout dir has |
| Pipeline stage | `cameras:` in `configs/pipeline/labeling/vlm.yaml`, or `labeling.cameras=[a,b]` |

The default everywhere is **one** view: the dataset's declared `primary_camera`
(the one the app shows first) on the CLI and in the pipeline — resolved by
`VLMLabeler` itself, so all three entry points agree — and the first view the
sidebar is showing in the modal. Either way it falls back to whatever
`TrajectoryDirectory.video_path()` resolves for a dataset that declares none,
and an episode the default view happens to lack still contributes the view it
does have.

That resolution reads the group named by `$BEHAVIOR_GRAPHS_DISPLAY_DATA`, the same
one the app displays. A dataset outside that group therefore gets no configured
geometry and no declared default view — logged, and the views go on the sheet as
recorded. Moving `configs/data/` reading into a non-UI module, so a labeling run
does not depend on a display filter, is the obvious follow-up.

That is all a single-camera dataset has, and the cheap default for one with
more: **each extra view adds a row-group per episode to every node's sheet, and
the sheets are re-sent in both passes**, so two views roughly doubles the image
tokens of a labeling run. A wrist view earns that when the exterior one leaves
grasps ambiguous — but it is always something you asked for, never what the
button does on its own. The modal's tile-count caption multiplies by the
selection so the cost is visible before the button.

The picker is deliberately separate from the sidebar's camera picker, which
drives the video players: what you watch and what you pay a VLM to look at are
different decisions.

## Generate an HTML report

```bash
python scripts/generate_report.py <run_dir> \
    --rollout-dir <rollout_dir> \
    --out report.html
```

Produces a self-contained HTML file with contact-sheet images, labels,
confidence badges, episode-progress ranges, and labeled transition chips for
every behavior node, ordered by episode position.

## Run as a pipeline stage

Add `vlm` to the labeling config for automatic labeling during a
pipeline run:

```bash
python -m behavior_graphs.pipeline.run \
    rollout_dir=<rollout_dir> \
    pipeline/labeling@labeling=vlm \
    labeling.task_description="pick up the block"
```

Config: `configs/pipeline/labeling/vlm.yaml`.

Default settings:

```yaml
model: gemini-3.5-flash
frames_per_episode: 5
max_episodes_per_node: 4
tile_px: 224
cameras: null          # one view per episode; e.g. [exterior_image_1_left, wrist_image_left]
thinking_level: high
task_description: ""
```

## Architecture

| File | Role |
|---|---|
| `src/behavior_graphs/vlm/adapters.py` | `VLMClient` ABC + adapters for Gemini, Anthropic, OpenAI-compat |
| `src/behavior_graphs/vlm/__init__.py` | Re-exports `VLMClient`, `make_vlm_client` |
| `src/behavior_graphs/viz/streamlit/models.py` | Model choices from `configs/`, shared by the chat and annotation UIs |
| `src/behavior_graphs/viz/streamlit/components/annotation_dialog.py` | Graph Demo's annotate button + modal |
| `src/behavior_graphs/stages/labeling/vlm_labeler.py` | `VLMLabeler` pipeline stage (`NodeLabeler` ABC) |
| `src/behavior_graphs/viz/frames.py` | Contact-sheet extraction from MP4 episodes, one row-group per episode per camera view |
| `src/behavior_graphs/core/node_labels.py` | Atomic JSON read/write for `node_labels.json` |
| `scripts/label_nodes.py` | Standalone labeler CLI |
| `scripts/generate_report.py` | HTML report generator |
| `configs/pipeline/labeling/vlm.yaml` | Hydra config for the pipeline labeling stage |
