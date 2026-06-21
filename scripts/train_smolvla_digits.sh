#!/usr/bin/env bash
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

DATASET_REPO_ID="${DATASET_REPO_ID:-k1000dai/so101-writei}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_so101_digits}"
JOB_NAME="${JOB_NAME:-smolvla_so101_digits}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MNIST_EXAMPLES_PER_DIGIT="${MNIST_EXAMPLES_PER_DIGIT:-64}"
MNIST_CACHE_DIR="${MNIST_CACHE_DIR:-}"
DIGIT_MAP="${DIGIT_MAP:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

ARGS=(
  --dataset.repo_id "${DATASET_REPO_ID}"
  --output_dir "${OUTPUT_DIR}"
  --job_name "${JOB_NAME}"
  --policy.device "${DEVICE}"
  --steps "${STEPS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --mnist_examples_per_digit "${MNIST_EXAMPLES_PER_DIGIT}"
)

if [[ -n "${MNIST_CACHE_DIR}" ]]; then
  ARGS+=(--mnist_cache_dir "${MNIST_CACHE_DIR}")
fi

if [[ -n "${DIGIT_MAP}" ]]; then
  ARGS+=(--digit_map "${DIGIT_MAP}")
fi

if [[ "${PUSH_TO_HUB}" == "true" ]]; then
  if [[ -z "${POLICY_REPO_ID}" ]]; then
    echo "POLICY_REPO_ID is required when PUSH_TO_HUB=true" >&2
    exit 1
  fi
  ARGS+=(--push_to_hub --policy.repo_id "${POLICY_REPO_ID}")
fi

uv run python examples/training/train_smolvla_digits.py "${ARGS[@]}"
