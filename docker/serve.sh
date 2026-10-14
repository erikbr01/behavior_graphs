#!/usr/bin/env bash
# Serve a trained diffusion_policy checkpoint over gRPC.
#
# Usage:
#   CHECKPOINT=droid/set_table_trimmed_h2/checkpoints/latest.ckpt GPU=0 ./docker/serve.sh
#
# Required env vars:
#   CHECKPOINT   Path to the checkpoint file relative to CHECKPOINT_DIR
#
# Optional env vars:
#   GPU             Host GPU index (default: 0)
#   PORT            Port to listen on (default: 5001)
#   CHECKPOINT_DIR  Host path containing checkpoints (default: ${OUTPUT_PATH})
#   IMAGE_TRAIN     Image name (default: dp-train)
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required, e.g. CHECKPOINT=droid/set_table/checkpoints/latest.ckpt}"
GPU="${GPU:-0}"
PORT="${PORT:-5001}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_PATH}}"
IMAGE_TRAIN="${IMAGE_TRAIN:-dp-train}"

docker run --rm \
  --gpus "device=$GPU" \
  -p "$PORT:$PORT" \
  -v "$CHECKPOINT_DIR":/checkpoints \
  "$IMAGE_TRAIN" \
  python -m diffusion_policy.real_world.droid.policy_server \
    --checkpoint "/checkpoints/$CHECKPOINT" \
    --device cuda:0 \
    --port "$PORT"
