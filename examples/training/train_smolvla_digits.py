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
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.digit_utils import (
    build_mnist_reference_bank,
    load_digit_map,
    sample_digit_references,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="k1000dai/so101-write")
    parser.add_argument("--output_dir", default="outputs/train/smolvla_so101_digits")
    parser.add_argument("--job_name", default="smolvla_so101_digits")
    parser.add_argument("--policy.device", dest="device", default="cuda")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=1_000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--mnist_examples_per_digit", type=int, default=64)
    parser.add_argument("--mnist_cache_dir", default=None)
    parser.add_argument("--use-mnist", dest="use_mnist", action=argparse.BooleanOptionalAction, default=False, help="Whether to use MNIST reference images for auxiliary supervision")
    parser.add_argument("--digit_map", default=None)
    parser.add_argument("--policy.repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming", action="store_true", help="Stream the dataset from Hub instead of caching it locally")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from the latest checkpoint in output_dir if it exists")
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

    try:
        mnist_root = _resolve_output_path(".cache/mnist")
        mnist = MNIST(root=mnist_root, train=True, download=True)
        bank = build_mnist_reference_bank(
            [{"image": image, "label": label} for image, label in mnist],
            examples_per_digit=examples_per_digit,
        )
    except Exception:
        mnist = load_dataset("ylecun/mnist", split="train")
        bank = build_mnist_reference_bank(list(mnist), examples_per_digit=examples_per_digit)
    torch.save(bank, cache_path)
    return bank


def _resolve_workdir() -> Path:
    workdir = os.environ.get("PBS_O_WORKDIR")
    if workdir:
        return Path(workdir).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if os.access(cwd, os.W_OK):
        return cwd
    return HF_LEROBOT_HOME


def _resolve_output_path(path: str) -> Path:
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    return _resolve_workdir() / output_path


def log_memory_usage(stage: str) -> None:
    try:
        import psutil
        process = psutil.Process()
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)
        logging.info("[Memory Check] %s - CPU RAM Usage: %.2f MB", stage, ram_usage_mb)
    except Exception as e:
        logging.debug("Could not log CPU memory usage: %s", e)

    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logging.info("[Memory Check] %s - GPU VRAM: Allocated %.2f MB, Reserved %.2f MB, Max %.2f MB", stage, allocated, reserved, max_allocated)
    except Exception as e:
        logging.debug("Could not log GPU memory usage: %s", e)


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError(
            "--policy.repo_id is required when --push_to_hub is enabled (which is the default). "
            "Use --no-push-to-hub to disable pushing to the Hugging Face Hub."
        )
    init_logging(console_level="INFO", file_level="DEBUG")
    start_time = time.time()

    logging.info("Starting SmolVLA digit training")
    logging.info("Arguments: %s", vars(args))

    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", output_dir)

    checkpoint_dir = None
    start_step = 0
    if args.resume:
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(
                checkpoints,
                key=lambda x: int(x.name.split("-")[-1])
            )
            for ckpt in reversed(checkpoints):
                config_path = ckpt / "config.json"
                weights_path = ckpt / "model.safetensors"
                opt_path = ckpt / "optimizer.bin"
                # A checkpoint is valid if all required files exist and are not empty
                if (
                    config_path.exists() and config_path.stat().st_size > 0
                    and weights_path.exists() and weights_path.stat().st_size > 0
                    and opt_path.exists() and opt_path.stat().st_size > 0
                ):
                    checkpoint_dir = ckpt
                    start_step = int(checkpoint_dir.name.split("-")[-1])
                    logging.info("Found latest valid checkpoint to resume: %s (starting from step %d)", checkpoint_dir, start_step)
                    break
                else:
                    logging.warning("Checkpoint %s is corrupted or incomplete. Deleting and skipping.", ckpt)
                    import shutil
                    try:
                        shutil.rmtree(ckpt)
                    except Exception as e:
                        logging.error("Failed to delete corrupted checkpoint %s: %s", ckpt, e)

    device = torch.device(args.device)
    logging.info("Device: %s", device)
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    logging.info(
        "Dataset metadata loaded: repo_id=%s fps=%s episodes=%s frames=%s",
        args.dataset_repo_id,
        dataset_metadata.fps,
        dataset_metadata.total_episodes,
        dataset_metadata.total_frames,
    )
    digit_map = load_digit_map(args.digit_map)
    logging.info("Digit map entries: %d", len(digit_map))
    features = dataset_to_policy_features(dataset_metadata.features)
    # Manually inject target_drawing as a visual feature
    features["observation.target_drawing"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 178, 256)
    )
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

    if checkpoint_dir is not None:
        logging.info("Resuming policy from checkpoint: %s", checkpoint_dir)
        policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    else:
        policy = SmolVLAPolicy(config)
    policy.train()
    policy.to(device)
    logging.info("Policy initialized on %s", device)

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
        if key != "observation.target_drawing"
    }

    if args.streaming:
        dataset = StreamingLeRobotDataset(
            args.dataset_repo_id,
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-3,
        )
        logging.info(
            "Streaming dataset ready: shards=%s backend=%s tolerance_s=%s",
            dataset.num_shards,
            dataset._video_backend,
            dataset.tolerance_s,
        )
        effective_num_workers = min(args.num_workers, max(1, dataset.num_shards))
    else:
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-3,
        )
        logging.info(
            "Dataset ready: backend=%s tolerance_s=%s",
            dataset._video_backend,
            dataset.tolerance_s,
        )
        effective_num_workers = args.num_workers

    logging.info("DataLoader workers: requested=%s effective=%s", args.num_workers, effective_num_workers)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": device.type != "cpu",
        "drop_last": True,
    }
    if not args.streaming:
        dataloader_kwargs["shuffle"] = True

    if effective_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    if args.use_mnist:
        mnist_cache = (
            _resolve_output_path(args.mnist_cache_dir)
            if args.mnist_cache_dir
            else output_dir / "mnist_reference_bank.pt"
        )
        logging.info("MNIST cache path: %s", mnist_cache)
        digit_bank = _load_mnist_bank(mnist_cache, args.mnist_examples_per_digit)
        logging.info("Loaded MNIST digit bank with %d digits", len(digit_bank))
    else:
        digit_bank = None
        logging.info("MNIST dataset is disabled. No auxiliary MNIST reference images will be used.")

    optimizer = config.get_optimizer_preset().build(policy.parameters())
    logging.info("Optimizer initialized: %s", optimizer.__class__.__name__)
    if checkpoint_dir is not None and (checkpoint_dir / "optimizer.bin").exists():
        logging.info("Loading optimizer state from checkpoint: %s", checkpoint_dir / "optimizer.bin")
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.bin", map_location=device))
    step = start_step
    # Load target drawings for all episodes into RAM (Approach B)
    logging.info("Loading target drawings for all episodes into RAM...")
    target_drawings_dir = Path("outputs/target_drawings")
    from PIL import Image
    import numpy as np
    target_drawings = []
    for ep_idx in range(dataset_metadata.total_episodes):
        img_path = target_drawings_dir / f"episode_{ep_idx}.png"
        if not img_path.exists():
            raise FileNotFoundError(
                f"Missing target drawing for episode {ep_idx} at {img_path}. "
                "Please run `qsub scripts/extract_patterns.pbs` (or local equivalent) first."
            )
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # (C, H, W)
        target_drawings.append(img_tensor)
    logging.info("Loaded %d target drawings successfully.", len(target_drawings))

    last_log_time = time.time()
    dataloader_iter = iter(dataloader)
    while step < args.steps:
        if step % 10 == 0:
            log_memory_usage(f"Step {step} loop start")
        try:
            logging.info("Waiting for batch %s/%s from dataloader...", step + 1, args.steps)
            fetch_start = time.time()
            raw_batch = next(dataloader_iter)
            logging.info("Batch %s fetched in %.2fs", step + 1, time.time() - fetch_start)
            # Inject observation.target_drawing into raw_batch (Approach B)
            ep_indices = raw_batch["episode_index"].view(-1).cpu().tolist()
            batch_targets = torch.stack([target_drawings[ep_idx] for ep_idx in ep_indices])
            raw_batch["observation.target_drawing"] = batch_targets
        except StopIteration:
            logging.info("Dataloader exhausted, restarting iterator")
            dataloader_iter = iter(dataloader)
            continue
        except Exception:
            logging.exception("Dataloader fetch failed at step %s", step)
            raise

        try:
            preprocess_start = time.time()
            processed_batch = preprocessor(_to_device(raw_batch, device))
            logging.info("Batch %s preprocessed in %.2fs", step + 1, time.time() - preprocess_start)

            digit_labels = processed_batch.get(config.digit_label_key)
            if digit_labels is None:
                raise KeyError(
                    f"The preprocessor did not produce '{config.digit_label_key}'. "
                    "Check the task text or digit mapping."
                )
            if not isinstance(digit_labels, torch.Tensor):
                digit_labels = torch.as_tensor(digit_labels, dtype=torch.long, device=device)
            digit_labels = digit_labels.to(device=device, dtype=torch.long).view(-1)
            logging.info("Batch %s digit labels resolved: shape=%s", step + 1, tuple(digit_labels.shape))

            if digit_bank is not None:
                refs_start = time.time()
                digit_references = sample_digit_references(digit_bank, digit_labels.cpu()).to(device)
                logging.info("Batch %s sampled digit references in %.2fs", step + 1, time.time() - refs_start)
                processed_batch[config.digit_reference_image_key] = digit_references

            logging.info("Batch %s starting forward pass...", step + 1)
            forward_start = time.time()
            loss, loss_dict = policy.forward(processed_batch)
            logging.info("Batch %s forward pass in %.2fs", step + 1, time.time() - forward_start)
            backward_start = time.time()
            loss.backward()
            logging.info("Batch %s backward pass in %.2fs", step + 1, time.time() - backward_start)
            optim_start = time.time()
            optimizer.step()
            optimizer.zero_grad()
            logging.info("Batch %s optimizer step in %.2fs", step + 1, time.time() - optim_start)

            if step % args.log_freq == 0:
                now = time.time()
                logging.info(
                    "Step %s/%s (%.1fs elapsed, %.1fs since last log)",
                    step,
                    args.steps,
                    now - start_time,
                    now - last_log_time,
                )
                print(json.dumps({"step": step, **loss_dict}, sort_keys=True), flush=True)
                last_log_time = now

            if step > start_step and step % args.save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                tmp_checkpoint_dir = output_dir / f".checkpoint-{step}.tmp"
                logging.info("Saving checkpoint to %s", checkpoint_dir)
                log_memory_usage("Start checkpoint saving")
                if tmp_checkpoint_dir.exists():
                    import shutil
                    shutil.rmtree(tmp_checkpoint_dir)
                tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Move policy to CPU and clear CUDA cache to prevent GPU/VRAM OOM
                logging.info("Moving policy to CPU...")
                policy.to("cpu")
                torch.cuda.empty_cache()
                log_memory_usage("After moving policy to CPU")

                logging.info("Saving policy weights...")
                policy.save_pretrained(tmp_checkpoint_dir)
                log_memory_usage("After saving policy weights")

                logging.info("Saving preprocessor and postprocessor configs...")
                preprocessor.save_pretrained(tmp_checkpoint_dir)
                postprocessor.save_pretrained(tmp_checkpoint_dir)

                # Move optimizer state dict to CPU before saving to prevent OOM
                logging.info("Extracting optimizer state dict...")
                opt_state_dict = optimizer.state_dict()
                log_memory_usage("After opt_state_dict extraction")

                logging.info("Deepcopying and offloading optimizer state dict to CPU...")
                import copy
                opt_state_dict_cpu = copy.deepcopy(opt_state_dict)
                for param_id, param_state in opt_state_dict_cpu.get("state", {}).items():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.cpu()
                log_memory_usage("After converting optimizer to CPU")

                logging.info("Saving optimizer state dict to disk...")
                torch.save(opt_state_dict_cpu, tmp_checkpoint_dir / "optimizer.bin")
                del opt_state_dict, opt_state_dict_cpu
                log_memory_usage("After saving optimizer state")

                # Restore policy to original device
                logging.info("Moving policy back to original device (%s)...", device)
                policy.to(device)
                torch.cuda.empty_cache()
                log_memory_usage("After restoring policy to device")

                # Atomic rename
                logging.info("Renaming temporary checkpoint directory to %s", checkpoint_dir)
                if checkpoint_dir.exists():
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                tmp_checkpoint_dir.rename(checkpoint_dir)
                logging.info("Checkpoint saved successfully at step %d", step)

            step += 1
            if step >= args.steps:
                break
        except Exception:
            logging.exception("Training step failed at global step %s", step)
            raise

    logging.info("Saving final artifacts to %s", output_dir)
    log_memory_usage("Start final artifacts save")
    logging.info("Moving policy to CPU...")
    policy.to("cpu")
    torch.cuda.empty_cache()
    log_memory_usage("After moving policy to CPU for final save")
    logging.info("Saving policy weights...")
    policy.save_pretrained(output_dir)
    log_memory_usage("After saving policy weights for final save")
    logging.info("Saving preprocessor and postprocessor configs...")
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    log_memory_usage("Completed final artifacts save")

    if args.push_to_hub:
        if not args.policy_repo_id:
            raise ValueError("--policy.repo_id is required when --push_to_hub is enabled")
        logging.info("Pushing artifacts to the Hub: %s", args.policy_repo_id)
        policy.push_to_hub(args.policy_repo_id)
        preprocessor.push_to_hub(args.policy_repo_id)
        postprocessor.push_to_hub(args.policy_repo_id)

    logging.info("Training complete in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
