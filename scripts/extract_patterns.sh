#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Source optional local secrets first, then tracked non-secret config.
if [[ -f config.env ]]; then
  source config.env
fi
if [[ -f config.shared.env ]]; then
  source config.shared.env
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

uv run python src/lerobot/scripts/lerobot_extract_patterns.py
