#!/bin/bash
set -euo pipefail

if [ -f config.env ]; then
  source config.env
fi
if [ -f config.shared.env ]; then
  source config.shared.env
fi

TARGET_DRAWING_PATH="${TARGET_DRAWING_PATH:-outputs/target_drawings/episode_0.png}"
export TARGET_DRAWING_PATH

lerobot-record \
--robot.type=so101_follower \
--robot.port=/dev/follower_arm \
--robot.id=my_awesome_follower_arm \
--robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0, width: 640, height: 480, fps: 30}, top :{type: intelrealsense, serial_number_or_name : '138422075876', width: 848, height: 480, fps: 30}}" \
--display_data=true \
--dataset.repo_id="${EVAL_BLUE_DATASET_REPO_ID:-yen-0/eval_smolvla_write2_blue_world}" \
--dataset.num_episodes=10 \
--dataset.single_task="write2" \
--dataset.streaming_encoding=true \
--dataset.encoder_threads=2 \
--policy.path="${BLUE_POLICY_REPO_ID:-yen-0/smolvla-so101-digits-0707-blue-world}" \
--policy.n_action_steps=50
