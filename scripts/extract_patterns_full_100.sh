#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f config.env ]]; then
  source config.env
fi
if [[ -f config.shared.env ]]; then
  source config.shared.env
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

export EXTRACT_SOURCE_REPO_ID="${FULL_DIGITS_DATASET_REPO_ID:-k1000dai/so101-write}"
export EXTRACT_OUTPUT_DIR="${REPO_ROOT}/target_drawings_full_100"
export EXTRACT_MAX_EPISODES=100

uv run python src/lerobot/scripts/lerobot_extract_patterns.py
