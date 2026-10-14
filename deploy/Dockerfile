# Behavior Graphs — Streamlit demo + user-study apps.
#
# Build (from the repo root — the Dockerfile lives in deploy/ but needs the
# whole repo as context):
#   docker build -f deploy/Dockerfile -t behavior-graphs .
# Run:    docker run -p 8501:8501 -v /path/to/data:/data behavior-graphs
# Open:   http://localhost:8501
#
# Or, for both apps at once, use deploy/docker-compose.yml.
#
# The image bundles the behavior_graphs package (installed with uv), the apps/
# Streamlit entrypoints and all config YAMLs. Run data (/data/runs,
# /data/sweeps, /data/rollouts) is NOT baked in: it is bind-mounted at runtime,
# so a baked-in copy would only be shadowed. Pass --build-arg DATA_MODE=full to
# copy whatever is staged at deploy/data/ into the image instead, for a host
# with no filesystem to mount from.

ARG DATA_MODE=empty

FROM python:3.10-slim AS base

# System deps for plotting + video frame extraction.
# - libgl1: cv2 / imageio image decoding (needed by mp4 thumbnail extraction)
# - libglib2.0-0: Qt-style image plugins
# - ffmpeg: imageio video reading
# - curl: container healthcheck
# - g++: PyTensor, under the Bayesian models in the `analysis` extra, compiles
#   each model's log-density to C. Without a compiler it says so in a warning
#   and falls back to its Python implementation, which took a fit measured in
#   seconds on a developer's machine past a minute in this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package first (better Docker layer caching). uv replaces the
# old conda env machinery. Editable install: at runtime this imports
# straight from /app/src, so the docker-compose.override.yml bind mount of
# src/ actually takes effect without a rebuild — a non-editable install
# freezes a site-packages copy at build time and silently ignores it.
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -e ".[viz,analysis,vlm]"

# Times-metric TrueType faces (Tinos, from croscore), for the paper figure the
# survey analytics page exports — see
# `viz/streamlit/user_study/figure_export.py`. Without them matplotlib falls
# back to its bundled DejaVu Serif and the figure does not match the body text
# of the IEEE template it goes into. A layer of its own, and deliberately
# *after* the uv install: putting it up with the other apt packages would
# invalidate the dependency install on every rebuild for the sake of 4 MB of
# fonts. TrueType matters too — `pdf.fonttype: 42` cannot embed a CFF-flavoured
# OpenType face and silently falls back to Type 3, which IEEE PDF eXpress
# rejects.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-croscore \
    && rm -rf /var/lib/apt/lists/*

# App entrypoints + study configs.
COPY apps /app/apps
COPY configs /app/configs
# Streamlit runtime settings (websocket compression, minimal toolbar, …).
COPY .streamlit /app/.streamlit
# Media pre-warming CLI.
COPY scripts/prewarm_media.py /app/scripts/prewarm_media.py

# ── Run data: empty placeholders (default — runtime bind mount supplies the
# real thing) vs. baked-in bundle (DATA_MODE=full). Selecting via a stage
# name means BuildKit never even reads deploy/data/ off disk in the default
# case — the data-full stage isn't part of the build graph, so it's skipped
# entirely rather than merely unused. ──────────────────────────────────────
FROM base AS data-empty
RUN mkdir -p /data/runs /data/sweeps /data/rollouts

FROM base AS data-full
COPY deploy/data/ /data/

FROM data-${DATA_MODE} AS runtime

ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/data
ENV BEHAVIOR_GRAPHS_RUNS_ROOT=/data/runs
ENV STUDY_RESULTS_DIR=/app/study_results
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
# Web-sized video proxies live here (see behavior_graphs.viz.media). Mount a
# volume over it so transcodes survive container restarts.
ENV BEHAVIOR_GRAPHS_MEDIA_CACHE=/media-cache
RUN mkdir -p /media-cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "apps/demo/Home.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
