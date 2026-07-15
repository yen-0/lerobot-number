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

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"Expected 4 crop coordinates, got: {value!r}")
    return tuple(int(part) for part in parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("EXTRACT_SOURCE_REPO_ID") or os.environ.get("DATASET_REPO_ID"),
        help="Dataset repo id to read episodes from. Defaults to EXTRACT_SOURCE_REPO_ID, then DATASET_REPO_ID.",
    )
    parser.add_argument(
        "--crop",
        default=os.environ.get("EXTRACT_CROP", "350 65 606 243"),
        help="Crop box as 'x_min y_min x_max y_max'. Defaults to EXTRACT_CROP or '350 65 606 243'.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EXTRACT_OUTPUT_DIR", str(REPO_ROOT / "target_drawings")),
        help="Directory to write episode PNGs to. Defaults to repo-root target_drawings.",
    )
    parser.add_argument(
        "--camera-key",
        default=os.environ.get("EXTRACT_CAMERA_KEY"),
        help="Optional camera key override. Defaults to the first non-wrist camera in the dataset.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=int(os.environ.get("EXTRACT_MAX_EPISODES", "0")),
        help="Optional limit on how many episodes to export. Defaults to EXTRACT_MAX_EPISODES or all episodes.",
    )
    return parser


def resolve_camera_key(src_dataset: LeRobotDataset, requested_camera_key: str | None) -> str:
    camera_keys = src_dataset.meta.camera_keys
    if requested_camera_key:
        if requested_camera_key not in camera_keys:
            raise ValueError(
                f"Requested camera key {requested_camera_key!r} not found. Available camera keys: {camera_keys}"
            )
        return requested_camera_key

    non_wrist_keys = [key for key in camera_keys if "wrist" not in key]
    if not non_wrist_keys:
        raise ValueError(f"No non-wrist camera found in {camera_keys}. All cameras: {camera_keys}")
    return non_wrist_keys[0]


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.repo_id:
        raise ValueError(
            "No dataset repo id provided. Pass --repo-id or set EXTRACT_SOURCE_REPO_ID / DATASET_REPO_ID."
        )

    init_logging(console_level="INFO", file_level="DEBUG")

    crop_coords = parse_crop(args.crop)
    out_dir = Path(args.output_dir).expanduser().resolve()

    logging.info("Repository root: %s", REPO_ROOT)
    logging.info("Loading source dataset: %s", args.repo_id)
    src_dataset = LeRobotDataset(args.repo_id)
    logging.info("Source dataset loaded. Total episodes: %s", src_dataset.num_episodes)

    camera_key = resolve_camera_key(src_dataset, args.camera_key)
    logging.info("Using camera key for pattern extraction: %s", camera_key)

    x_min, y_min, x_max, y_max = crop_coords
    out_dir.mkdir(parents=True, exist_ok=True)
    max_episodes = args.max_episodes if args.max_episodes and args.max_episodes > 0 else src_dataset.num_episodes
    max_episodes = min(max_episodes, src_dataset.num_episodes)

    logging.info("Extracting target drawings to: %s", out_dir)
    logging.info("Exporting %s episode PNGs", max_episodes)
    for ep_idx in range(max_episodes):
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
            logging.info("Saved episode %s/%s target drawing to %s", ep_idx + 1, max_episodes, out_path)

    logging.info("Target drawings extraction completed successfully!")


if __name__ == "__main__":
    main()
