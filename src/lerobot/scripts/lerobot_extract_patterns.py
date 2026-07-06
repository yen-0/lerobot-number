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

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Extract patterns from a LeRobot dataset, crop them, and create a copy with target drawing feature.")
    parser.add_argument("--repo_id", default="k1000dai/so101-writei", help="Source dataset repository ID")
    parser.add_argument("--root", default=None, help="Source dataset local root directory")
    parser.add_argument("--new_repo_id", default=None, help="Target dataset repository ID (required for process mode)")
    parser.add_argument("--new_root", default=None, help="Target dataset local root directory")
    parser.add_argument(
        "--mode", 
        choices=["calibrate", "process"], 
        required=True, 
        help="Mode: 'calibrate' to extract first episode's last frame, 'process' to crop and copy entire dataset"
    )
    parser.add_argument(
        "--crop", 
        type=int, 
        nargs=4, 
        default=[386, 60, 642, 238],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Crop box coordinates: x_min y_min x_max y_max (default: 386 60 642 238)"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push the new dataset to the Hugging Face Hub")
    return parser.parse_args()


def main():
    init_logging(console_level="INFO", file_level="DEBUG")
    args = parse_args()

    # Load source dataset
    logging.info(f"Loading source dataset: {args.repo_id}...")
    src_dataset = LeRobotDataset(args.repo_id, root=args.root)
    logging.info(f"Source dataset loaded. Total episodes: {src_dataset.num_episodes}, Total frames: {src_dataset.num_frames}")

    # Determine camera key (non-wrist camera)
    camera_keys = src_dataset.meta.camera_keys
    non_wrist_keys = [k for k in camera_keys if "wrist" not in k]
    if not non_wrist_keys:
        raise ValueError(f"No non-wrist camera found in {camera_keys}. All cameras: {camera_keys}")
    camera_key = non_wrist_keys[0]
    logging.info(f"Using camera key for pattern extraction: {camera_key}")

    # Calibration Mode
    if args.mode == "calibrate":
        # Get last frame of the first episode
        ep_idx = 0
        from_idx = src_dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = src_dataset.meta.episodes["dataset_to_index"][ep_idx]
        last_frame_idx = to_idx - 1

        logging.info(f"Calibrating using episode {ep_idx}, last frame index: {last_frame_idx}")
        frame_data = src_dataset[last_frame_idx]
        img_tensor = frame_data[camera_key]

        # Convert tensor (C, H, W) to PIL Image
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # If float, convert to uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        out_dir = Path("outputs/calibration")
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "frame_raw.png"
        img_pil.save(raw_path)
        logging.info(f"Saved raw last frame of episode 0 to: {raw_path}")

        if args.crop:
            x_min, y_min, x_max, y_max = args.crop
            cropped_pil = img_pil.crop((x_min, y_min, x_max, y_max))
            cropped_path = out_dir / "frame_cropped.png"
            cropped_pil.save(cropped_path)
            logging.info(f"Saved cropped frame to: {cropped_path} (crop box: {args.crop})")
            logging.info("Please inspect both images to verify the crop, then run process mode when satisfied.")
        else:
            logging.info("Please inspect 'frame_raw.png' to determine the crop coordinates, then rerun with --crop X_MIN Y_MIN X_MAX Y_MAX")

    # Process Mode
    elif args.mode == "process":
        if not args.new_repo_id:
            raise ValueError("--new_repo_id is required in process mode")
        if not args.crop:
            raise ValueError("--crop coordinates (X_MIN Y_MIN X_MAX Y_MAX) are required in process mode")

        x_min, y_min, x_max, y_max = args.crop
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        logging.info(f"Crop dimensions: width={crop_width}, height={crop_height}")

        # Setup features for the new dataset
        new_features = {}
        for key, val in src_dataset.features.items():
            new_features[key] = val.copy() if hasattr(val, "copy") else dict(val)

        new_features["observation.target_drawing"] = {
            "dtype": "video",
            "shape": (crop_height, crop_width, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        }

        # Create new dataset
        logging.info(f"Creating new dataset: {args.new_repo_id}...")
        new_dataset = LeRobotDataset.create(
            repo_id=args.new_repo_id,
            root=args.new_root,
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
                for key in src_dataset.features:
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

        if args.push_to_hub:
            logging.info(f"Pushing dataset to Hugging Face Hub: {args.new_repo_id}...")
            new_dataset.push_to_hub()
            logging.info("Pushing completed successfully!")


if __name__ == "__main__":
    main()
