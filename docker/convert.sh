#!/usr/bin/env bash
# Convert raw DROID sessions to a robo_dataset store.
#
# Usage:
#   # MP4 sessions (no ZED SDK required):
#   RAW_DATA_DIR=/data/raw CONVERTED_DIR=/data/out ./docker/convert.sh
#
#   # SVO sessions (ZED SDK bind-mounted from host):
#   RAW_DATA_DIR=/data/raw CONVERTED_DIR=/data/out ZED=1 ./docker/convert.sh
#
# Optional env vars:
#   RAW_DATA_DIR     Source sessions dir (default: ${DATA_ROOT}/droid_raw)
#   CONVERTED_DIR    Output store dir   (default: ${DATA_ROOT}/droid_converted)
#   ZED              Set to 1 to use dp-convert-zed with ZED SDK (default: 0)
#   ZED_SDK_LIB_DIR  Host path to extracted ZED SDK libs (default: ${DATA_ROOT}/zed_sdk_extracted/lib)
#   ZED_SETTINGS_DIR Host path to ZED calibration files  (default: ${DATA_ROOT}/zed_settings)
#   CONVERT_ARGS     Extra flags passed to convert_droid_data.py
#   IMAGE_CONVERT     Image name for MP4 conversion (default: dp-convert)
#   IMAGE_CONVERT_ZED Image name for ZED conversion  (default: dp-convert-zed)
set -euo pipefail

RAW_DATA_DIR="${RAW_DATA_DIR:-${DATA_ROOT}/droid_raw}"
CONVERTED_DIR="${CONVERTED_DIR:-${DATA_ROOT}/droid_converted}"
ZED="${ZED:-0}"
ZED_SDK_LIB_DIR="${ZED_SDK_LIB_DIR:-${DATA_ROOT}/zed_sdk_extracted/lib}"
ZED_SETTINGS_DIR="${ZED_SETTINGS_DIR:-${DATA_ROOT}/zed_settings}"
CONVERT_ARGS="${CONVERT_ARGS:-}"
IMAGE_CONVERT="${IMAGE_CONVERT:-dp-convert}"
IMAGE_CONVERT_ZED="${IMAGE_CONVERT_ZED:-dp-convert-zed}"

if [ "$ZED" = "1" ]; then
  docker run --rm \
    --gpus all \
    -v "$ZED_SDK_LIB_DIR":/opt/zed/lib \
    -v "$ZED_SETTINGS_DIR":/data/zed_settings \
    -v "$RAW_DATA_DIR":/data/droid_raw \
    -v "$CONVERTED_DIR":/data/droid_converted \
    "$IMAGE_CONVERT_ZED" \
    python scripts/utils/convert_droid_data.py \
      --data-dir /data/droid_raw \
      --output   /data/droid_converted \
      --zed-settings-path /data/zed_settings \
      $CONVERT_ARGS
else
  docker run --rm \
    --gpus all \
    -v "$RAW_DATA_DIR":/data/droid_raw \
    -v "$CONVERTED_DIR":/data/droid_converted \
    "$IMAGE_CONVERT" \
    python scripts/utils/convert_droid_data.py \
      --data-dir /data/droid_raw \
      --output   /data/droid_converted \
      $CONVERT_ARGS
fi
