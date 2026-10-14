#!/usr/bin/env bash
# Build Docker images for the behavior_graphs workflow stack.
#
# Usage:
#   ./docker/build.sh [TARGET]
#
# TARGET: all (default) | train | convert | convert-zed | test | integration-test
#
# Image names can be overridden via environment variables:
#   IMAGE_TRAIN=dp-train IMAGE_CONVERT=dp-convert ./docker/build.sh all
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DOCKER_DIR="$REPO_ROOT/docker"
TARGET="${1:-all}"

IMAGE_TRAIN="${IMAGE_TRAIN:-dp-train}"
IMAGE_CONVERT="${IMAGE_CONVERT:-dp-convert}"
IMAGE_CONVERT_ZED="${IMAGE_CONVERT_ZED:-dp-convert-zed}"
IMAGE_TEST="${IMAGE_TEST:-dp-test}"
IMAGE_INTEGRATION_TEST="${IMAGE_INTEGRATION_TEST:-dp-integration-test}"

_build_train() {
  docker build -f "$DOCKER_DIR/Dockerfile.train" -t "$IMAGE_TRAIN" "$REPO_ROOT"
}
_build_convert() {
  docker build -f "$DOCKER_DIR/Dockerfile.convert" \
    --build-arg BASE_IMAGE="$IMAGE_TRAIN" -t "$IMAGE_CONVERT" "$REPO_ROOT"
}
_build_convert_zed() {
  docker build -f "$DOCKER_DIR/Dockerfile.convert-zed" \
    --build-arg BASE_IMAGE="$IMAGE_CONVERT" -t "$IMAGE_CONVERT_ZED" "$REPO_ROOT"
}
_build_test() {
  docker build -f "$DOCKER_DIR/Dockerfile.test" -t "$IMAGE_TEST" "$REPO_ROOT"
}
_build_integration_test() {
  docker build -f "$DOCKER_DIR/Dockerfile.integration-test" \
    --build-arg BASE_IMAGE="$IMAGE_CONVERT" -t "$IMAGE_INTEGRATION_TEST" "$REPO_ROOT"
}

case "$TARGET" in
  train)
    _build_train ;;
  convert)
    _build_train && _build_convert ;;
  convert-zed)
    _build_train && _build_convert && _build_convert_zed ;;
  test)
    _build_test ;;
  integration-test)
    _build_train && _build_convert && _build_integration_test ;;
  all)
    _build_train
    _build_convert
    _build_convert_zed
    _build_test
    _build_integration_test
    ;;
  *)
    echo "Usage: $0 [all|train|convert|convert-zed|test|integration-test]" >&2
    exit 1 ;;
esac
