#!/usr/bin/env bash
# Run diffusion_policy training inside dp-train.
#
# Usage:
#   CONFIG=set_table_trimmed_h2 GPU=0 ./docker/train.sh
#
# Required env vars:
#   CONFIG   Hydra config dir name under configs/image/droid/
#   GPU      Host GPU index (default: 0)
#
# Optional env vars:
#   CHECKPOINT_DIR   Host path for checkpoints (default: ${OUTPUT_PATH})
#   WANDB_API_KEY    Passed through to the container
#   IMAGE_TRAIN      Image name (default: dp-train)
set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required, e.g. CONFIG=set_table_trimmed_h2}"
GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_PATH}}"
IMAGE_TRAIN="${IMAGE_TRAIN:-dp-train}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

docker run --rm \
  --gpus "device=$GPU" \
  --ipc=host \
  ${WANDB_API_KEY:+-e WANDB_API_KEY="$WANDB_API_KEY"} \
  -v "$CHECKPOINT_DIR":/workspace/third_party/diffusion_policy/data/outputs \
  "$IMAGE_TRAIN" \
  python train.py \
    --config-dir="configs/image/droid/$CONFIG" \
    --config-name=config.yaml \
    training.device=cuda:0
