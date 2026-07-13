#!/usr/bin/env bash
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

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

DATASET_REPO_ID="${DATASET_REPO_ID:-yen-0/so101-write-5-kadokawa}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_so101_digits}"
JOB_NAME="${JOB_NAME:-smolvla_so101_digits}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MNIST_EXAMPLES_PER_DIGIT="${MNIST_EXAMPLES_PER_DIGIT:-64}"
MNIST_CACHE_DIR="${MNIST_CACHE_DIR:-}"
USE_MNIST="${USE_MNIST:-false}"
USE_TARGET_DRAWING="${USE_TARGET_DRAWING:-true}"
BLUE_WORLD_FILTER="${BLUE_WORLD_FILTER:-false}"
DIGIT_MAP="${DIGIT_MAP:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-300}"
DIAGNOSTIC_DUMP_INTERVAL="${DIAGNOSTIC_DUMP_INTERVAL:-0}"
TARGET_DRAWINGS_DIR="${TARGET_DRAWINGS_DIR:-target_drawings}"
BLUE_WORLD_HUE_MIN="${BLUE_WORLD_HUE_MIN:-0.55}"
BLUE_WORLD_HUE_MAX="${BLUE_WORLD_HUE_MAX:-0.75}"
BLUE_WORLD_SATURATION_MIN="${BLUE_WORLD_SATURATION_MIN:-0.2}"
BLUE_WORLD_VALUE_MIN="${BLUE_WORLD_VALUE_MIN:-0.05}"
BLUE_WORLD_CLEANUP_PASSES="${BLUE_WORLD_CLEANUP_PASSES:-1}"
BLUE_WORLD_MIN_BLUE_NEIGHBORS="${BLUE_WORLD_MIN_BLUE_NEIGHBORS:-1}"
BLUE_WORLD_FILL_HOLE_NEIGHBORS="${BLUE_WORLD_FILL_HOLE_NEIGHBORS:-6}"
HUB_ONLY="${HUB_ONLY:-true}"
RESUME="${RESUME:-false}"
HUB_ONLY="true"
PUSH_TO_HUB="true"

ARGS=(
  --dataset.repo_id "${DATASET_REPO_ID}"
  --output_dir "${OUTPUT_DIR}"
  --job_name "${JOB_NAME}"
  --policy.device "${DEVICE}"
  --steps "${STEPS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --mnist_examples_per_digit "${MNIST_EXAMPLES_PER_DIGIT}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --heartbeat_timeout "${HEARTBEAT_TIMEOUT}"
  --diagnostic_dump_interval "${DIAGNOSTIC_DUMP_INTERVAL}"
  --target_drawings_dir "${TARGET_DRAWINGS_DIR}"
  --policy.blue_world_hue_min "${BLUE_WORLD_HUE_MIN}"
  --policy.blue_world_hue_max "${BLUE_WORLD_HUE_MAX}"
  --policy.blue_world_saturation_min "${BLUE_WORLD_SATURATION_MIN}"
  --policy.blue_world_value_min "${BLUE_WORLD_VALUE_MIN}"
  --policy.blue_world_cleanup_passes "${BLUE_WORLD_CLEANUP_PASSES}"
  --policy.blue_world_min_blue_neighbors "${BLUE_WORLD_MIN_BLUE_NEIGHBORS}"
  --policy.blue_world_fill_hole_neighbors "${BLUE_WORLD_FILL_HOLE_NEIGHBORS}"
)

if [[ "${HUB_ONLY}" == "true" ]]; then
  ARGS+=(--hub_only)
elif [[ "${RESUME}" == "true" ]]; then
  ARGS+=(--resume)
fi

if [[ "${FREEZE_VISION_ENCODER}" == "true" ]]; then
  ARGS+=(--policy.freeze_vision_encoder)
else
  ARGS+=(--no-policy.freeze_vision_encoder)
fi

if [[ "${TRAIN_EXPERT_ONLY}" == "true" ]]; then
  ARGS+=(--policy.train_expert_only)
else
  ARGS+=(--no-policy.train_expert_only)
fi

if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  ARGS+=(--policy.gradient_checkpointing)
else
  ARGS+=(--no-policy.gradient_checkpointing)
fi

if [[ "${USE_MNIST}" == "true" ]]; then
  ARGS+=(--use-mnist)
else
  ARGS+=(--no-use-mnist)
fi

if [[ "${USE_TARGET_DRAWING}" == "true" ]]; then
  ARGS+=(--use-target-drawing)
else
  ARGS+=(--no-use-target-drawing)
fi

if [[ "${BLUE_WORLD_FILTER}" == "true" ]]; then
  ARGS+=(--policy.blue_world_filter)
else
  ARGS+=(--no-policy.blue_world_filter)
fi

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
