#!/usr/bin/env bash
# Set up offline DROID SVO->VideoStore conversion with NO sudo and NO conda,
# reusing the existing uv training venv (third_party/diffusion_policy/.venv).
#
# The raw DROID sessions ship ZED SVO2 recordings (no MP4s), so
# convert_droid_data.py must decode them via pyzed. pyzed is only a binding to
# the ZED SDK's C++ shared libs, so we extract the SDK to a user prefix (no
# system install) and point LD_LIBRARY_PATH at it. depth_mode=NONE in
# DroidStore._read_svo_frames keeps the AI/depth path (TensorRT, cuDNN) out of
# the picture entirely, so only libsl_zed + libturbojpeg are needed.
#
# Target: Ubuntu 24.04, CUDA 12.8+ (RTX 50-series ok), Python 3.9 venv.
# Re-runnable: existing artifacts are detected and skipped.
#
# After it finishes, run the printed conversion command.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_PY="$REPO/third_party/diffusion_policy/.venv/bin/python"
SDK_DIR="${ZED_SDK_DIR:-$HOME/zed_sdk_extracted}"          # extracted SDK (full tree)
SETTINGS_DIR="${ZED_SETTINGS_DIR:-$REPO/data/zed_settings}"  # SN<serial>.conf calibration
SDK_URL="https://download.stereolabs.com/zedsdk/5.4/cu12/ubuntu24"   # -> cuda12.8 v5.4 runfile
PYZED_WHL_URL="https://download.stereolabs.com/zedsdk/5.4/whl/linux_x86_64/pyzed-5.4-cp39-cp39-linux_x86_64.whl"
CAMERA_SERIALS="${CAMERA_SERIALS:-14313307 36716034}"       # wrist_image_left, exterior_image_1_left

[ -x "$VENV_PY" ] || { echo "training venv missing: $VENV_PY (run diffusion_policy/scripts/setup_env_uv.sh first)" >&2; exit 1; }

# 1. ZED SDK — extract the Makeself runfile without executing it (no sudo).
if [ -f "$SDK_DIR/lib/libsl_zed.so" ]; then
    echo ">> ZED SDK already extracted at $SDK_DIR"
else
    echo ">> downloading + extracting ZED SDK 5.4 -> $SDK_DIR"
    run="$(mktemp -t zed_sdk.XXXXXX.run)"
    curl -fL# "$SDK_URL" -o "$run"
    chmod +x "$run"
    "$run" --noexec --keep --nox11 --noprogress --target "$SDK_DIR" >/dev/null
    rm -f "$run"
fi

# 2. libturbojpeg.so.0 — libsl_zed's only otherwise-unmet dep. Extract from the
#    Ubuntu package into the SDK lib dir (no sudo, no system install).
if [ ! -f "$SDK_DIR/lib/libturbojpeg.so.0" ]; then
    echo ">> staging libturbojpeg.so.0"
    tmp="$(mktemp -d)"
    ( cd "$tmp" && apt-get download libturbojpeg0 >/dev/null 2>&1 \
        && dpkg-deb -x libturbojpeg0*.deb x )
    real="$(find "$tmp" /opt/conda/pkgs -name 'libturbojpeg.so.0.*' 2>/dev/null | head -1)"
    [ -n "$real" ] || { echo "could not locate libturbojpeg.so.0.* to stage" >&2; exit 1; }
    cp -f "$real" "$SDK_DIR/lib/libturbojpeg.so.0"
    rm -rf "$tmp"
fi

# 3. pyzed into the venv — prebuilt wheel, --no-deps so numpy stays pinned (<2)
#    for the training stack. numpy-2-built wheels are ABI-compatible with 1.x.
if ! LD_LIBRARY_PATH="$SDK_DIR/lib" "$VENV_PY" -c "import pyzed.sl" 2>/dev/null; then
    echo ">> installing pyzed 5.4 into the venv (--no-deps)"
    whl="$(mktemp -d)/pyzed-5.4-cp39-cp39-linux_x86_64.whl"
    curl -fL# "$PYZED_WHL_URL" -o "$whl"
    "$VENV_PY" -m pip install --no-deps --force-reinstall "$whl"
    rm -f "$whl"
fi

# 4. Camera calibration .conf files (factory calibration by serial).
mkdir -p "$SETTINGS_DIR"
for sn in $CAMERA_SERIALS; do
    dst="$SETTINGS_DIR/SN${sn}.conf"
    if [ -s "$dst" ]; then
        echo ">> SN${sn}.conf present"
    else
        curl -fsSL "https://calib.stereolabs.com/?SN=${sn}" -o "$dst"
        echo ">> fetched SN${sn}.conf ($(wc -c <"$dst") bytes)"
    fi
done

echo
echo ">> setup complete. pyzed: $(LD_LIBRARY_PATH="$SDK_DIR/lib" "$VENV_PY" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null)"
cat <<EOF

Convert the DROID dataset with (32 workers; ~9% of sessions have empty/corrupt
recordings and are skipped):

  LD_LIBRARY_PATH="$SDK_DIR/lib" \\
  "$VENV_PY" "$REPO/scripts/utils/convert_droid_data.py" \\
      --data-dir "$REPO/data/droid_raw" \\
      --output   "$REPO/data/droid_converted" \\
      --camera-source svo \\
      --camera wrist_image_left=14313307 \\
      --camera exterior_image_1_left=36716034 \\
      --zed-settings-path "$SETTINGS_DIR" \\
      --workers 32 --skip-empty --skip-corrupt
EOF
