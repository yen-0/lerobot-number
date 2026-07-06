#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. and The Google DeepMind team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract patterns from the end of videos in a dataset, crop them, and save/push as a new feature."""

import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging


def main():
    init_logging(console_level="INFO", file_level="DEBUG")

    # Hardcoded parameters
    src_repo_id = "k1000dai/so101-write"
    new_repo_id = "yen-0/so101-writei-patterns"
    crop_coords = [386, 60, 642, 238] # x_min y_min x_max y_max

    # Load source dataset
    logging.info(f"Loading source dataset: {src_repo_id}...")
    src_dataset = LeRobotDataset(src_repo_id)
    logging.info(f"Source dataset loaded. Total episodes: {src_dataset.num_episodes}, Total frames: {src_dataset.num_frames}")

    # Determine camera key (non-wrist camera)
    camera_keys = src_dataset.meta.camera_keys
    non_wrist_keys = [k for k in camera_keys if "wrist" not in k]
    if not non_wrist_keys:
        raise ValueError(f"No non-wrist camera found in {camera_keys}. All cameras: {camera_keys}")
    camera_key = non_wrist_keys[0]
    logging.info(f"Using camera key for pattern extraction: {camera_key}")

    x_min, y_min, x_max, y_max = crop_coords
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    logging.info(f"Crop dimensions: width={crop_width}, height={crop_height}")

    # Setup features for the new dataset, excluding system/implicit ones
    EXCLUDE_KEYS = {"index", "episode_index", "timestamp", "frame_index", "task_index"}
    new_features = {}
    for key, val in src_dataset.features.items():
        if key in EXCLUDE_KEYS:
            continue
        new_features[key] = val.copy() if hasattr(val, "copy") else dict(val)

    new_features["observation.target_drawing"] = {
        "dtype": "video",
        "shape": (crop_height, crop_width, 3),
        "names": ["height", "width", "channels"],
        "info": None,
    }

    # Create new dataset
    logging.info(f"Creating new dataset: {new_repo_id}...")
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=src_dataset.fps,
        features=new_features,
        robot_type=src_dataset.meta.robot_type,
    )

    for ep_idx in range(src_dataset.num_episodes):
        from_idx = src_dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = src_dataset.meta.episodes["dataset_to_index"][ep_idx]
        last_frame_idx = to_idx - 1

        # Extract the cropped drawing from the last frame
        last_frame_data = src_dataset[last_frame_idx]
        last_img_tensor = last_frame_data[camera_key]
        last_img_np = last_img_tensor.permute(1, 2, 0).cpu().numpy()
        if last_img_np.dtype != np.uint8:
            last_img_np = (last_img_np * 255.0).clip(0, 255).astype(np.uint8)
        last_img_pil = Image.fromarray(last_img_np)
        cropped_pil = last_img_pil.crop((x_min, y_min, x_max, y_max))
        cropped_np = np.array(cropped_pil)

        logging.info(f"Episode {ep_idx + 1}/{src_dataset.num_episodes}: frames {from_idx} to {to_idx - 1}...")

        # Write all frames of the episode
        for idx in range(from_idx, to_idx):
            frame_data = src_dataset[idx]

            new_frame = {}
            for key in new_features:
                if key == "observation.target_drawing":
                    continue
                val = frame_data[key]
                if isinstance(val, torch.Tensor):
                    if src_dataset.features[key]["dtype"] in ["video", "image"]:
                        # Convert (C, H, W) to (H, W, C)
                        val_np = val.permute(1, 2, 0).cpu().numpy()
                        if val_np.dtype == np.uint8:
                            new_frame[key] = val_np
                        else:
                            new_frame[key] = (val_np * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        new_frame[key] = val.cpu().numpy()
                else:
                    new_frame[key] = val

            # Set task instruction
            new_frame["task"] = frame_data["task"]

            # Set the cropped target drawing (repeated for all frames)
            new_frame["observation.target_drawing"] = cropped_np

            new_dataset.add_frame(new_frame)

        new_dataset.save_episode()

    new_dataset.finalize()
    logging.info(f"Dataset successfully created locally at: {new_dataset.root}")

    logging.info(f"Pushing dataset to Hugging Face Hub: {new_repo_id}...")
    new_dataset.push_to_hub()
    logging.info("Pushing completed successfully!")


if __name__ == "__main__":
    main()
