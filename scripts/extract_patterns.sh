#!/usr/bin/env bash
set -euo pipefail

# Move to the directory where the script is run
cd "${PBS_O_WORKDIR:-$(pwd)}"

# Source config.env if it exists
if [[ -f config.env ]]; then
  source config.env
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

MODE="${EXTRACT_MODE:-process}"
CROP="${EXTRACT_CROP:-386 60 642 238}"
NEW_REPO_ID="${EXTRACT_NEW_REPO_ID:-yen-0/so101-writei-patterns}"

ARGS=(
  --repo_id "${DATASET_REPO_ID:-k1000dai/so101-writei}"
  --mode "${MODE}"
)

if [[ -n "${CROP}" ]]; then
  # Parse coordinates (e.g. "100 100 300 300")
  ARGS+=(--crop ${CROP})
fi

if [[ -n "${NEW_REPO_ID}" ]]; then
  ARGS+=(--new_repo_id "${NEW_REPO_ID}")
  ARGS+=(--push_to_hub)
fi

uv run python src/lerobot/scripts/lerobot_extract_patterns.py "${ARGS[@]}"
