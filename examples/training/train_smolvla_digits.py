#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Train SmolVLA on SO-101 digit-drawing data with MNIST auxiliary supervision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.digit_utils import (
    build_mnist_reference_bank,
    load_digit_map,
    sample_digit_references,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="k1000dai/so101-writei")
    parser.add_argument("--output_dir", default="outputs/train/smolvla_so101_digits")
    parser.add_argument("--job_name", default="smolvla_so101_digits")
    parser.add_argument("--policy.device", dest="device", default="cuda")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=5_000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--mnist_examples_per_digit", type=int, default=64)
    parser.add_argument("--mnist_cache_dir", default=None)
    parser.add_argument("--digit_map", default=None)
    parser.add_argument("--policy.repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _load_mnist_bank(cache_path: Path, examples_per_digit: int) -> dict[int, torch.Tensor]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    mnist = load_dataset("mnist", split="train")
    bank = build_mnist_reference_bank(list(mnist), examples_per_digit=examples_per_digit)
    torch.save(bank, cache_path)
    return bank


def _resolve_workdir() -> Path:
    workdir = os.environ.get("PBS_O_WORKDIR")
    if workdir:
        return Path(workdir).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_output_path(path: str) -> Path:
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    return _resolve_workdir() / output_path


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    digit_map = load_digit_map(args.digit_map)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        load_vlm_weights=True,
        freeze_vision_encoder=False,
        train_expert_only=False,
        digit_alignment_loss_weight=0.25,
        digit_classification_loss_weight=1.0,
    )

    policy = SmolVLAPolicy(config)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config,
        dataset_stats=dataset_metadata.stats,
        digit_map=digit_map,
    )
    delta_timestamps = {
        "observation.state": [t / dataset_metadata.fps for t in config.observation_delta_indices],
        ACTION: [t / dataset_metadata.fps for t in config.action_delta_indices],
    }
    delta_timestamps |= {
        key: [t / dataset_metadata.fps for t in config.observation_delta_indices]
        for key in config.image_features
    }

    dataset = StreamingLeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps, tolerance_s=1e-3)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "pin_memory": device.type != "cpu",
        "drop_last": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    mnist_cache = (
        _resolve_output_path(args.mnist_cache_dir)
        if args.mnist_cache_dir
        else output_dir / "mnist_reference_bank.pt"
    )
    digit_bank = _load_mnist_bank(mnist_cache, args.mnist_examples_per_digit)

    optimizer = config.get_optimizer_preset().build(policy.parameters())
    step = 0
    while step < args.steps:
        for raw_batch in dataloader:
            processed_batch = preprocessor(_to_device(raw_batch, device))
            digit_labels = processed_batch.get(config.digit_label_key)
            if digit_labels is None:
                raise KeyError(
                    f"The preprocessor did not produce '{config.digit_label_key}'. "
                    "Check the task text or digit mapping."
                )
            if not isinstance(digit_labels, torch.Tensor):
                digit_labels = torch.as_tensor(digit_labels, dtype=torch.long, device=device)
            digit_labels = digit_labels.to(device=device, dtype=torch.long).view(-1)
            digit_references = sample_digit_references(digit_bank, digit_labels.cpu()).to(device)
            processed_batch[config.digit_reference_image_key] = digit_references

            loss, loss_dict = policy.forward(processed_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(json.dumps({"step": step, **loss_dict}, sort_keys=True))

            if step > 0 and step % args.save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)

            step += 1
            if step >= args.steps:
                break

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    if args.push_to_hub:
        if not args.policy_repo_id:
            raise ValueError("--policy.repo_id is required when --push_to_hub is set")
        policy.push_to_hub(args.policy_repo_id)
        preprocessor.push_to_hub(args.policy_repo_id)
        postprocessor.push_to_hub(args.policy_repo_id)


if __name__ == "__main__":
    main()
