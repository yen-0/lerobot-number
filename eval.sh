#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/config.env" ]]; then
  source "${REPO_ROOT}/config.env"
fi
if [[ -f "${REPO_ROOT}/config.shared.env" ]]; then
  source "${REPO_ROOT}/config.shared.env"
fi

EVAL_POLICY_REPO_ID="yen-0/smolvla-so101-digits-0707"
EVAL_POLICY_REVISION="fb91d44a811352c6fb4392d818fb6bedba93ad6c"

TARGET_DRAWING_PATH="${TARGET_DRAWING_PATH:-${REPO_ROOT}/target_drawings/episode_0.png}"
export TARGET_DRAWING_PATH

ROLL_OUT_TASK="${DATASET_SINGLE_TASK:-write2}"
ROLLOUT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROLLOUT_OWNER="${EVAL_POLICY_REPO_ID%%/*}"
ROLLOUT_TASK_SLUG="$(printf '%s' "${ROLL_OUT_TASK}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g')"
ROLLOUT_TASK_SLUG="${ROLLOUT_TASK_SLUG:-task}"
ROLLOUT_DATASET_REPO_ID="${ROLLOUT_OWNER}/rollout-${ROLLOUT_TASK_SLUG}-${ROLLOUT_TIMESTAMP}"

uv run lerobot-rollout \
--strategy.type=episodic \
--inference.type="${INFERENCE_TYPE:-rtc}" \
--robot.type=so101_follower \
--robot.port=/dev/follower_arm \
--robot.id=my_awesome_follower_arm \
--robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0, width: 640, height: 480, fps: 30}, top :{type: intelrealsense, serial_number_or_name : '138422075876', width: 848, height: 480, fps: 30}}" \
--display_data=true \
--dataset.repo_id="${ROLLOUT_DATASET_REPO_ID}" \
--dataset.num_episodes=10 \
--dataset.single_task="${ROLL_OUT_TASK}" \
--dataset.streaming_encoding=true \
--dataset.encoder_threads=2 \
--policy.path="${EVAL_POLICY_REPO_ID}" \
--policy.pretrained_revision="${EVAL_POLICY_REVISION}" \
--policy.n_action_steps=50
