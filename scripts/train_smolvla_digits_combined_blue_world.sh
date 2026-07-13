#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config.env"
fi

if [[ -f "${ROOT_DIR}/config.shared.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config.shared.env"
fi

PBS_SCRIPT="${ROOT_DIR}/scripts/train_smolvla_digits_combined_blue_world.pbs"
if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "missing PBS script: ${PBS_SCRIPT}" >&2
  exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available in PATH" >&2
  exit 1
fi

echo "Using full dataset: ${FULL_DIGITS_DATASET_REPO_ID:-k1000dai/so101-write}"
echo "Using focused dataset: ${FOCUSED_DIGITS_DATASET_REPO_ID:-yen-0/so101-write-5-kadokawa}"
echo "Using merged blue-world dataset: ${COMBINED_BLUE_WORLD_DATASET_REPO_ID:-yen-0/so101-write-combined-blue-world}"
echo "Using merged policy repo: ${COMBINED_BLUE_POLICY_REPO_ID:-yen-0/smolvla-so101-digits-combined-blue-world}"

job_id="$(qsub "${PBS_SCRIPT}")"
echo "Submitted combined blue-world job: ${job_id}"
