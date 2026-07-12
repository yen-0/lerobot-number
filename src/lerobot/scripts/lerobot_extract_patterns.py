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

"""Extract target drawing patterns (last frames) from each episode and save them locally."""

import logging
from pathlib import Path
import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging


def main():
    init_logging(console_level="INFO", file_level="DEBUG")

    # Hardcoded parameters
    src_repo_id = "yen-0/so101-write-5-kadokawa"
    crop_coords = [386, 60, 642, 238] # x_min y_min x_max y_max

    # Load source dataset
    logging.info(f"Loading source dataset: {src_repo_id}...")
    src_dataset = LeRobotDataset(src_repo_id)
    logging.info(f"Source dataset loaded. Total episodes: {src_dataset.num_episodes}")

    # Determine camera key (non-wrist camera)
    camera_keys = src_dataset.meta.camera_keys
    non_wrist_keys = [k for k in camera_keys if "wrist" not in k]
    if not non_wrist_keys:
        raise ValueError(f"No non-wrist camera found in {camera_keys}. All cameras: {camera_keys}")
    camera_key = non_wrist_keys[0]
    logging.info(f"Using camera key for pattern extraction: {camera_key}")

    x_min, y_min, x_max, y_max = crop_coords
    out_dir = Path("outputs/target_drawings")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Extracting target drawings to: {out_dir}")
    for ep_idx in range(src_dataset.num_episodes):
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
        
        out_path = out_dir / f"episode_{ep_idx}.png"
        cropped_pil.save(out_path)
        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            logging.info(f"Saved episode {ep_idx + 1}/{src_dataset.num_episodes} target drawing to {out_path}")

    logging.info("Target drawings extraction completed successfully!")


if __name__ == "__main__":
    main()
