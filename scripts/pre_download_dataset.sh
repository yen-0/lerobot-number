#!/usr/bin/env bash
# Script to pre-download and cache datasets/models using the Apptainer container environment.
set -euo pipefail

# 1. Move to the workspace directory
cd "$(dirname "$0")/.."

# 2. Source configuration
if [[ ! -f config.env ]]; then
  if [[ -f config.env.example ]]; then
    echo "[$(date -Is)] creating config.env from config.env.example..."
    cp config.env.example config.env
  else
    echo "[$(date -Is)] Error: Missing config.env" >&2
    exit 1
  fi
fi
source config.env

# 3. Prepare cache directories
mkdir -p "${HF_HOME}"
mkdir -p "${HF_LEROBOT_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"

# 4. Load apptainer module if it exists/is needed on the host
if command -v module &> /dev/null && [[ -n "${APPTAINER_MODULE:-}" ]]; then
  module load "${APPTAINER_MODULE}" || true
fi

# 5. Export variables for Apptainer
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi

DATASET_REPO="${DATASET_REPO_ID:-k1000dai/so101-write}"
MODEL_REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

echo "[$(date -Is)] Starting pre-download for dataset: ${DATASET_REPO}"
apptainer exec \
  --bind "$(pwd):$(pwd):rw" \
  --bind "$HOME:$HOME:rw" \
  --writable-tmpfs \
  --pwd "$(pwd)" \
  "${APPTAINER_IMAGE}" \
  huggingface-cli download "${DATASET_REPO}" --repo-type dataset

echo "[$(date -Is)] Starting pre-download for VLM model: ${MODEL_REPO}"
apptainer exec \
  --bind "$(pwd):$(pwd):rw" \
  --bind "$HOME:$HOME:rw" \
  --writable-tmpfs \
  --pwd "$(pwd)" \
  "${APPTAINER_IMAGE}" \
  huggingface-cli download "${MODEL_REPO}"

echo "[$(date -Is)] Pre-download complete! All assets cached in ${HF_HOME}"
