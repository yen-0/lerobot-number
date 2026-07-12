#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "${ROOT_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config.env"
fi

if [[ -f "${ROOT_DIR}/config.shared.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config.shared.env"
fi

for script in \
  "${ROOT_DIR}/scripts/train_smolvla_digits.pbs" \
  "${ROOT_DIR}/scripts/train_smolvla_digits_blue_world.pbs"
do
  if [[ ! -f "${script}" ]]; then
    echo "missing PBS script: ${script}" >&2
    exit 1
  fi
done

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available in PATH" >&2
  exit 1
fi

echo "Using source dataset: ${DATASET_REPO_ID:-yen-0/so101-write-5-kadokawa}"
echo "Using blue dataset: ${FILTERED_DATASET_REPO_ID:-yen-0/so101-write-5-kadokawa-blue-world}"
echo "Using target-image policy repo: ${POLICY_REPO_ID:-yen-0/smolvla-so101-digits-0707}"
echo "Using blue-world policy repo: ${BLUE_POLICY_REPO_ID:-yen-0/smolvla-so101-digits-0707-blue-world}"

target_job="$(qsub "${ROOT_DIR}/scripts/train_smolvla_digits.pbs")"
echo "Submitted target-image job: ${target_job}"

blue_job="$(qsub "${ROOT_DIR}/scripts/train_smolvla_digits_blue_world.pbs")"
echo "Submitted blue-world filter+training job: ${blue_job}"
