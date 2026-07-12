#!/usr/bin/env bash
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [[ -f config.env ]]; then
  source config.env
fi
if [[ -f config.shared.env ]]; then
  source config.shared.env
fi

export TEACHER_REPO_ID="${TEACHER_REPO_ID:-yen-0/smolvla-so101-digits-0707}"
export HUB_REPO_ID="${HUB_REPO_ID:-yen-0/mnist-distill-0707}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/mnist_distill}"
export JOB_NAME="${JOB_NAME:-mnist_distill}"
export DEVICE="${DEVICE:-cuda}"
export TEACHER_DEVICE="${TEACHER_DEVICE:-${DEVICE}}"
export STEPS="${STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export TEMPERATURE="${TEMPERATURE:-2.0}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
export STUDENT_HIDDEN_DIM="${STUDENT_HIDDEN_DIM:-64}"
export STUDENT_DROPOUT="${STUDENT_DROPOUT:-0.1}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export EVAL_FREQ="${EVAL_FREQ:-500}"
export LOG_FREQ="${LOG_FREQ:-50}"
export SEED="${SEED:-42}"
export MNIST_ROOT="${MNIST_ROOT:-.cache/mnist}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
export HUB_ONLY="${HUB_ONLY:-true}"

if [[ -n "${CACHE_DIR:-}" ]]; then
  export CACHE_DIR
fi

if [[ -n "${TRAIN_SUBSET:-}" ]]; then
  export TRAIN_SUBSET
fi

if [[ -n "${EVAL_SUBSET:-}" ]]; then
  export EVAL_SUBSET
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

qsub -V "$(dirname "$0")/mnist.pbs"
