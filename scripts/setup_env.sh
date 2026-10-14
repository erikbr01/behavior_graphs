#!/usr/bin/env bash
# Write .env with local paths. Re-run whenever you move the data directory.
# Usage: bash scripts/setup_env.sh [data_path]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Default: data/ sibling of the repo root, then common mount points.
if [[ $# -ge 1 ]]; then
    DATA_PATH="$1"
elif [[ -d "$REPO_ROOT/../data" ]]; then
    DATA_PATH="$(cd "$REPO_ROOT/../data" && pwd)"
elif [[ -d "${DATA_PATH}" ]]; then
    DATA_PATH="${DATA_PATH}"
else
    DATA_PATH="$REPO_ROOT/data"
fi

cat > "$ENV_FILE" <<EOF
export DATA_PATH=$DATA_PATH

# Google Cloud / Vertex AI (for Gemini VLM node labeling)
export GOOGLE_CLOUD_PROJECT=gcp-driven-data
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
EOF

echo "Wrote $ENV_FILE"
echo "  DATA_PATH=$DATA_PATH"
echo "  GOOGLE_CLOUD_PROJECT=gcp-driven-data"
echo ""
echo "Source it in your shell:  source .env"
echo "Or export inline:         export DATA_PATH=$DATA_PATH"
