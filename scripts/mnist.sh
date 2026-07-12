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
export HUB_REPO_ID="${HUB_REPO_ID:-yen-0/smolvla-0707-no-training-mnist-probe}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/no_training_mnist_probe_0707}"
export JOB_NAME="${JOB_NAME:-no_training_mnist_probe_0707}"
export DEVICE="${DEVICE:-cuda}"
export TEACHER_DEVICE="${TEACHER_DEVICE:-${DEVICE}}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SEED="${SEED:-42}"
export MNIST_ROOT="${MNIST_ROOT:-.cache/mnist}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
export HUB_ONLY="${HUB_ONLY:-true}"

if [[ -n "${CACHE_DIR:-}" ]]; then
  export CACHE_DIR
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

if [[ -n "${LOCAL_FILES_ONLY:-}" ]]; then
  export LOCAL_FILES_ONLY
fi

qsub -V "$(dirname "$0")/mnist.pbs"
