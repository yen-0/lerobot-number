#!/usr/bin/env bash
set -uo pipefail

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
INFERENCE_TYPE="${INFERENCE_TYPE:-rtc}"
DATASET_NUM_EPISODES=10
GOAL_IMAGE_DIR="${REPO_ROOT}/target_drawings_tasks"
TASK_DRAW_TIAN=$'draw\u7530'

EXPERIMENTS=(base blue combined)
TASKS=(writeA draw15 draw55 drawSquare drawCircle drawFace drawHuman "${TASK_DRAW_TIAN}")
TASK_MODES=(with_goal no_goal)

declare -A EXPERIMENT_REVISION=(
  [base]="fb91d44a811352c6fb4392d818fb6bedba93ad6c"
  [blue]="40eb626c785eac9f16b6afb20b3b1dcfa0e88e32"
  [combined]="725ad96569110dc3d62c6dee08476d977deb3b8d"
)

declare -A TASK_IMAGE_PATH=(
  [writeA]="${GOAL_IMAGE_DIR}/write_a.png"
  [draw15]="${GOAL_IMAGE_DIR}/draw_15.png"
  [draw55]="${GOAL_IMAGE_DIR}/draw_55.png"
  [drawSquare]="${GOAL_IMAGE_DIR}/draw_square.png"
  [drawCircle]="${GOAL_IMAGE_DIR}/draw_circle.png"
  [drawFace]="${GOAL_IMAGE_DIR}/draw_face.png"
  [drawHuman]="${GOAL_IMAGE_DIR}/draw_human.png"
)
TASK_IMAGE_PATH["${TASK_DRAW_TIAN}"]="${GOAL_IMAGE_DIR}/draw_tian.png"

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g'
}

run_rollout() {
  local experiment="$1"
  local task_name="$2"
  local mode="$3"
  local target_image="$4"
  local revision="$5"
  local owner task_slug dataset_repo_id timestamp

  if [[ "${mode}" == "with_goal" && ! -f "${target_image}" ]]; then
    echo "Missing goal image for ${task_name}: ${target_image}" >&2
    return 1
  fi

  owner="${EVAL_POLICY_REPO_ID%%/*}"
  task_slug="$(slugify "${task_name}")"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  dataset_repo_id="${owner}/rollout-${experiment}-${task_slug}-${mode}-${timestamp}"

  if [[ "${mode}" == "with_goal" ]]; then
    export TARGET_DRAWING_PATH="${target_image}"
  else
    unset TARGET_DRAWING_PATH
  fi

  echo "[$(date -Is)] Starting ${experiment} rollout for ${task_name} (${mode})"
  if [[ "${mode}" == "with_goal" ]]; then
    echo "[$(date -Is)] Goal image: ${TARGET_DRAWING_PATH}"
  else
    echo "[$(date -Is)] Goal image: <none>"
  fi
  echo "[$(date -Is)] Dataset repo: ${dataset_repo_id}"

  uv run lerobot-rollout \
    --strategy.type=episodic \
    --inference.type="${INFERENCE_TYPE}" \
    --robot.type=so101_follower \
    --robot.port=/dev/follower_arm \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0, width: 640, height: 480, fps: 30}, top :{type: intelrealsense, serial_number_or_name : '138422075876', width: 848, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="${dataset_repo_id}" \
    --dataset.num_episodes="${DATASET_NUM_EPISODES}" \
    --dataset.single_task="${task_name}" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --policy.path="${EVAL_POLICY_REPO_ID}" \
    --policy.pretrained_revision="${revision}" \
    --policy.n_action_steps=50
}

failures=()

for experiment in "${EXPERIMENTS[@]}"; do
  for task_name in "${TASKS[@]}"; do
    for mode in "${TASK_MODES[@]}"; do
      if ! run_rollout "${experiment}" "${task_name}" "${mode}" "${TASK_IMAGE_PATH[${task_name}]}" "${EXPERIMENT_REVISION[${experiment}]}"; then
        failures+=("${experiment}:${task_name}:${mode}")
        echo "[$(date -Is)] Failed ${experiment}:${task_name}:${mode}" >&2
      fi
    done
  done
done

if (( ${#failures[@]} > 0 )); then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "[$(date -Is)] Completed all experiment-task rollouts"
