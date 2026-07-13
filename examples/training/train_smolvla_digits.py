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
import copy
import faulthandler
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
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
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="yen-0/so101-write-5-kadokawa")
    parser.add_argument("--output_dir", default="outputs/train/smolvla_so101_digits")
    parser.add_argument("--job_name", default="smolvla_so101_digits")
    parser.add_argument("--policy.device", dest="device", default="cuda")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=1_000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--mnist_examples_per_digit", type=int, default=64)
    parser.add_argument("--mnist_cache_dir", default=None)
    parser.add_argument("--use-mnist", dest="use_mnist", action=argparse.BooleanOptionalAction, default=False, help="Whether to use MNIST reference images for auxiliary supervision")
    parser.add_argument(
        "--use-target-drawing",
        dest="use_target_drawing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to inject the target drawing image as an additional visual observation branch",
    )
    parser.add_argument(
        "--policy.blue_world_filter",
        dest="blue_world_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to filter live camera observations to blue pixels on a white background",
    )
    parser.add_argument("--policy.blue_world_hue_min", dest="blue_world_hue_min", type=float, default=0.55)
    parser.add_argument("--policy.blue_world_hue_max", dest="blue_world_hue_max", type=float, default=0.75)
    parser.add_argument(
        "--policy.blue_world_saturation_min", dest="blue_world_saturation_min", type=float, default=0.2
    )
    parser.add_argument("--policy.blue_world_value_min", dest="blue_world_value_min", type=float, default=0.05)
    parser.add_argument(
        "--policy.blue_world_cleanup_passes", dest="blue_world_cleanup_passes", type=int, default=1
    )
    parser.add_argument(
        "--policy.blue_world_min_blue_neighbors",
        dest="blue_world_min_blue_neighbors",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--policy.blue_world_fill_hole_neighbors",
        dest="blue_world_fill_hole_neighbors",
        type=int,
        default=6,
    )
    parser.add_argument("--digit_map", default=None)
    parser.add_argument(
        "--target_drawings_dir",
        default=os.environ.get("TARGET_DRAWINGS_DIR", "target_drawings"),
        help="Directory containing episode_<n>.png goal images. Defaults to TARGET_DRAWINGS_DIR or target_drawings.",
    )
    parser.add_argument("--policy.repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hub_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload checkpoints/final artifacts to Hugging Face without persisting them in output_dir",
    )
    parser.add_argument("--streaming", action="store_true", help="Stream the dataset from Hub instead of caching it locally")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from the latest checkpoint in output_dir if it exists")
    parser.add_argument("--policy.gradient_checkpointing", dest="gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing to reduce VRAM usage")
    parser.add_argument("--use-amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=True, help="Whether to use Automatic Mixed Precision (AMP) for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--policy.freeze_vision_encoder", dest="freeze_vision_encoder", action=argparse.BooleanOptionalAction, default=False, help="Whether to freeze the vision encoder")
    parser.add_argument("--policy.train_expert_only", dest="train_expert_only", action=argparse.BooleanOptionalAction, default=False, help="Whether to freeze the VLM and train only the action expert and projections")
    parser.add_argument("--heartbeat_timeout", type=int, default=300, help="Seconds without a training-loop heartbeat before dumping Python stack traces")
    parser.add_argument("--diagnostic_dump_interval", type=int, default=0, help="If positive, dump all Python stack traces to stderr every N seconds")
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


def _build_export_config(config: SmolVLAConfig) -> SmolVLAConfig:
    """Build the config that will be serialized with the exported policy."""

    return copy.deepcopy(config)


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


def log_disk_usage(path: Path) -> None:
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        logging.info(
            "[Disk Check] Path: %s - Total: %.2f GB, Used: %.2f GB, Free: %.2f GB",
            path,
            total / (1024**3),
            used / (1024**3),
            free / (1024**3),
        )
    except Exception as e:
        logging.debug("Could not log disk usage: %s", e)


def log_trainable_parameters(policy: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(
        "Policy parameters: trainable=%d (%.2fM), total=%d (%.2fM), trainable_pct=%.2f%%",
        trainable_params,
        trainable_params / 1_000_000,
        total_params,
        total_params / 1_000_000,
        100.0 * trainable_params / max(total_params, 1),
    )


def log_optimizer_state(optimizer: torch.optim.Optimizer, stage: str) -> None:
    state_tensors = 0
    state_elements = 0
    for param_state in optimizer.state.values():
        for value in param_state.values():
            if isinstance(value, torch.Tensor):
                state_tensors += 1
                state_elements += value.numel()
    logging.info(
        "Optimizer state %s: entries=%d tensors=%d elements=%d (%.2fM)",
        stage,
        len(optimizer.state),
        state_tensors,
        state_elements,
        state_elements / 1_000_000,
    )


def _set_gradient_checkpointing(module: torch.nn.Module, enabled: bool) -> None:
    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    method = getattr(module, method_name, None)
    if not callable(method):
        return
    try:
        if enabled:
            method(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            method()
    except TypeError:
        method()


def apply_runtime_training_flags(policy: SmolVLAPolicy, args: argparse.Namespace) -> None:
    """Apply env/CLI finetuning flags even when the model was loaded from a checkpoint config."""
    policy.config.freeze_vision_encoder = args.freeze_vision_encoder
    policy.config.train_expert_only = args.train_expert_only
    policy.config.gradient_checkpointing = args.gradient_checkpointing

    flow_model = policy.model
    flow_model.config.freeze_vision_encoder = args.freeze_vision_encoder
    flow_model.config.train_expert_only = args.train_expert_only
    flow_model.config.gradient_checkpointing = args.gradient_checkpointing
    flow_model.config.train_state_proj = policy.config.train_state_proj

    # Reset before reapplying SmolVLA's freezing rules. This lets a resumed run either
    # freeze more of the model or intentionally unfreeze it.
    for param in policy.parameters():
        param.requires_grad = True

    flow_model.vlm_with_expert.freeze_vision_encoder = args.freeze_vision_encoder
    flow_model.vlm_with_expert.train_expert_only = args.train_expert_only
    flow_model.vlm_with_expert.set_requires_grad()
    flow_model.set_requires_grad()

    for module in (flow_model.vlm_with_expert.vlm, flow_model.vlm_with_expert.lm_expert):
        _set_gradient_checkpointing(module, args.gradient_checkpointing)

    logging.info(
        "Runtime training flags applied: freeze_vision_encoder=%s train_expert_only=%s gradient_checkpointing=%s",
        args.freeze_vision_encoder,
        args.train_expert_only,
        args.gradient_checkpointing,
    )


def _cuda_synchronize_for_diagnostics(stage: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    logging.info("%s: synchronizing CUDA stream...", stage)
    torch.cuda.synchronize()
    logging.info("%s: CUDA stream synchronized", stage)


def _install_diagnostics(dump_interval_s: int) -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception as exc:
        logging.warning("Could not enable faulthandler: %s", exc)

    if dump_interval_s > 0:
        try:
            faulthandler.dump_traceback_later(dump_interval_s, repeat=True)
            logging.info("Enabled faulthandler traceback dumps every %s seconds", dump_interval_s)
        except Exception as exc:
            logging.warning("Could not schedule faulthandler traceback dumps: %s", exc)

    def _dump_on_signal(signum, _frame):
        signame = signal.Signals(signum).name
        logging.error("Received %s. Dumping Python stack traces before exit.", signame)
        log_memory_usage(f"signal {signame}")
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        finally:
            raise SystemExit(128 + signum)

    def _dump_without_exit(signum, _frame):
        signame = signal.Signals(signum).name
        logging.error("Received %s. Dumping Python stack traces.", signame)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, _dump_on_signal)
        except Exception as exc:
            logging.debug("Could not install signal handler for %s: %s", signum, exc)

    if hasattr(signal, "SIGUSR1"):
        try:
            signal.signal(signal.SIGUSR1, _dump_without_exit)
        except Exception as exc:
            logging.debug("Could not install SIGUSR1 handler: %s", exc)


def free_checkpoint_space(output_dir: Path, required_space_gb: float = 10.0) -> None:
    try:
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / (1024**3)
        if free_gb >= required_space_gb:
            return

        logging.warning(
            "[Disk Warning] Free space is %.2f GB, which is below the required %.2f GB threshold. Starting cleanup...",
            free_gb,
            required_space_gb
        )

        # Find and sort existing checkpoints (oldest first)
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if not checkpoints:
            logging.warning("No checkpoints found to delete.")
            return

        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.name.split("-")[-1])
        )

        # Delete oldest checkpoints until we have enough space
        for ckpt in checkpoints:
            logging.warning("Deleting oldest checkpoint %s to free up space.", ckpt)
            try:
                shutil.rmtree(ckpt)
            except Exception as e:
                logging.error("Failed to delete checkpoint %s: %s", ckpt, e)
            
            # Recheck disk space
            _, _, free = shutil.disk_usage(output_dir)
            free_gb = free / (1024**3)
            if free_gb >= required_space_gb:
                logging.info("Successfully freed enough space. Current free space: %.2f GB", free_gb)
                break
    except Exception as e:
        logging.error("Error during disk space cleanup: %s", e)


import threading
import traceback
import sys

class HeartbeatMonitor(threading.Thread):
    def __init__(self, timeout=300):
        super().__init__(daemon=True)
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self.running = True

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            time.sleep(10)
            if time.time() - self.last_heartbeat > self.timeout:
                logging.error(
                    "[Diagnosis Hang Alert] No heartbeat for %.1f seconds. Dumping stack traces...",
                    time.time() - self.last_heartbeat
                )
                for thread_id, frame in sys._current_frames().items():
                    logging.error("Thread %s stack trace:", thread_id)
                    logging.error("".join(traceback.format_stack(frame)))
                # Only log once per hang event to avoid flooding
                self.heartbeat()


def main() -> None:
    import os
    if "LEROBOT_VIDEO_DECODER_CACHE_SIZE" not in os.environ:
        os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = "500"
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError(
            "--policy.repo_id is required when --push_to_hub is enabled (which is the default). "
            "Use --no-push-to-hub to disable pushing to the Hugging Face Hub."
        )
    if args.hub_only:
        if not args.push_to_hub:
            raise ValueError("--hub_only requires --push_to_hub.")
        if not args.policy_repo_id:
            raise ValueError("--hub_only requires --policy.repo_id.")
        if args.resume:
            raise ValueError("--hub_only cannot be combined with --resume because no local checkpoints are kept.")
    init_logging(console_level="INFO", file_level="DEBUG")
    _install_diagnostics(args.diagnostic_dump_interval)
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
    if args.use_target_drawing:
        # Manually inject target_drawing as a visual feature when the extra goal-image branch is enabled.
        features["observation.target_drawing"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 178, 256),
        )
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        load_vlm_weights=True,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        digit_alignment_loss_weight=0.25,
        digit_classification_loss_weight=1.0,
        blue_world_filter=args.blue_world_filter,
        blue_world_hue_min=args.blue_world_hue_min,
        blue_world_hue_max=args.blue_world_hue_max,
        blue_world_saturation_min=args.blue_world_saturation_min,
        blue_world_value_min=args.blue_world_value_min,
        blue_world_cleanup_passes=args.blue_world_cleanup_passes,
        blue_world_min_blue_neighbors=args.blue_world_min_blue_neighbors,
        blue_world_fill_hole_neighbors=args.blue_world_fill_hole_neighbors,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    export_config = _build_export_config(config)

    if checkpoint_dir is not None:
        logging.info("Resuming policy from checkpoint: %s", checkpoint_dir)
        policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    else:
        policy = SmolVLAPolicy(config)
    apply_runtime_training_flags(policy, args)
    policy.train()
    policy.to(device)
    logging.info("Policy initialized on %s", device)
    log_trainable_parameters(policy)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config,
        dataset_stats=dataset_metadata.stats,
        digit_map=digit_map,
    )
    export_preprocessor, export_postprocessor = make_smolvla_pre_post_processors(
        export_config,
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
        dataloader_kwargs["persistent_workers"] = True
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

    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after applying training flags.")
    optimizer = policy.config.get_optimizer_preset().build(trainable_params)
    logging.info("Optimizer initialized: %s", optimizer.__class__.__name__)
    if checkpoint_dir is not None and (checkpoint_dir / "optimizer.bin").exists():
        logging.info("Loading optimizer state from checkpoint: %s", checkpoint_dir / "optimizer.bin")
        try:
            optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.bin", map_location=device))
            log_optimizer_state(optimizer, "after checkpoint load")
        except ValueError as exc:
            logging.warning(
                "Skipping optimizer state from %s because it does not match the current trainable parameter set: %s",
                checkpoint_dir / "optimizer.bin",
                exc,
            )
            logging.warning("Continuing with a fresh optimizer state for the resumed model weights.")

    def _save_exportable_artifacts(save_dir: Path) -> None:
        """Write the inference-safe policy bundle used by `eval.sh`."""

        original_policy_config = policy.config
        try:
            policy.config = export_config
            policy.save_pretrained(save_dir)
        finally:
            policy.config = original_policy_config

        export_preprocessor.save_pretrained(save_dir)
        export_postprocessor.save_pretrained(save_dir)

    def upload_checkpoint_to_hub(local_checkpoint_dir: Path, step: int) -> None:
        if not args.push_to_hub or not args.policy_repo_id:
            return
        api = HfApi()
        api.create_repo(repo_id=args.policy_repo_id, repo_type="model", exist_ok=True)

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp_root = Path(tmpdir) / "checkpoints"
            step_dir = tmp_root / f"checkpoint-{step}"
            shutil.copytree(local_checkpoint_dir, step_dir)
            shutil.copytree(local_checkpoint_dir, tmp_root / "last")
            api.upload_folder(
                repo_id=args.policy_repo_id,
                repo_type="model",
                folder_path=tmp_root,
                path_in_repo="checkpoints",
                commit_message=f"Upload checkpoint at step {step}",
            )

    step = start_step
    target_drawings = None
    if args.use_target_drawing:
        # Load target drawings for all episodes into RAM (Approach B)
        logging.info("Loading target drawings for all episodes into RAM...")
        target_drawings_dir = Path(args.target_drawings_dir)
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
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (C, H, W)
            target_drawings.append(img_tensor)
        logging.info("Loaded %d target drawings successfully.", len(target_drawings))
    else:
        logging.info("Target drawing branch disabled. Training with default SmolVLA image inputs only.")

    last_log_time = time.time()
    dataloader_iter = iter(dataloader)

    # Start Heartbeat Monitor for hang detection
    monitor = HeartbeatMonitor(timeout=args.heartbeat_timeout)
    monitor.start()

    accumulated_batches = 0
    accumulated_losses = {}
    accum_steps = args.gradient_accumulation_steps

    try:
        while step < args.steps:
            monitor.heartbeat()
            if step % 10 == 0:
                log_memory_usage(f"Step {step} loop start")
                log_disk_usage(output_dir)
            if step % 100 == 0:
                torch.cuda.empty_cache()
            try:
                logging.info("Waiting for batch %s/%s from dataloader...", step + 1, args.steps)
                fetch_start = time.time()
                raw_batch = next(dataloader_iter)
                fetch_time = time.time() - fetch_start
                logging.info("Batch %s fetched in %.2fs", step + 1, fetch_time)
                if fetch_time > 10.0:
                    logging.warning(
                        "[Diagnosis Warning] Dataloader fetch took %.2fs! This could indicate slow disk I/O or network filesystem mount issues.",
                        fetch_time
                    )
                if args.use_target_drawing:
                    # Inject observation.target_drawing into raw_batch (Approach B)
                    if target_drawings is None:
                        raise RuntimeError("Target drawing branch enabled but target drawings were not loaded.")
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

                from contextlib import nullcontext
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (args.use_amp and device.type == "cuda") else nullcontext()

                logging.info(
                    "Batch %s (micro-step %d/%d) starting forward pass...",
                    step + 1,
                    accumulated_batches + 1,
                    accum_steps,
                )
                forward_start = time.time()
                with autocast_ctx:
                    loss, loss_dict = policy.forward(processed_batch)
                logging.info(
                    "Batch %s (micro-step %d/%d) forward pass completed in %.2fs",
                    step + 1,
                    accumulated_batches + 1,
                    accum_steps,
                    time.time() - forward_start,
                )

                # Scale the loss for gradient accumulation
                loss_scaled = loss / accum_steps

                log_memory_usage(f"Batch {step + 1} (micro-step {accumulated_batches + 1}/{accum_steps}) before backward")
                _cuda_synchronize_for_diagnostics(
                    f"Batch {step + 1} (micro-step {accumulated_batches + 1}/{accum_steps}) before backward",
                    device,
                )
                logging.info(
                    "Batch %s (micro-step %d/%d) starting backward pass...",
                    step + 1,
                    accumulated_batches + 1,
                    accum_steps,
                )
                backward_start = time.time()
                try:
                    loss_scaled.backward()
                    _cuda_synchronize_for_diagnostics(
                        f"Batch {step + 1} (micro-step {accumulated_batches + 1}/{accum_steps}) after backward",
                        device,
                    )
                except BaseException:
                    logging.exception(
                        "Backward failed or was interrupted at batch %s (micro-step %d/%d)",
                        step + 1,
                        accumulated_batches + 1,
                        accum_steps,
                    )
                    log_memory_usage(f"Batch {step + 1} backward failure")
                    raise
                logging.info(
                    "Batch %s (micro-step %d/%d) backward pass completed in %.2fs",
                    step + 1,
                    accumulated_batches + 1,
                    accum_steps,
                    time.time() - backward_start,
                )

                # Accumulate stats for logging
                accumulated_batches += 1
                for k, v in loss_dict.items():
                    accumulated_losses[k] = accumulated_losses.get(k, 0.0) + (v / accum_steps)

                # Only perform optimizer step and checkpointing/logging after collecting enough gradients
                if accumulated_batches >= accum_steps:
                    log_memory_usage(f"Batch {step + 1} after backward")
                    logging.info("Batch %s starting optimizer step...", step + 1)
                    optim_start = time.time()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    logging.info("Batch %s optimizer step completed in %.2fs", step + 1, time.time() - optim_start)
                    log_memory_usage(f"Batch {step + 1} after optimizer")

                    if step % args.log_freq == 0:
                        now = time.time()
                        logging.info(
                            "Step %s/%s (%.1fs elapsed, %.1fs since last log)",
                            step,
                            args.steps,
                            now - start_time,
                            now - last_log_time,
                        )
                        print(json.dumps({"step": step, **accumulated_losses}, sort_keys=True), flush=True)
                        last_log_time = now

                    # Reset accumulation states
                    accumulated_batches = 0
                    accumulated_losses = {}

                    if step > start_step and step % args.save_freq == 0:
                        checkpoint_dir = output_dir / f"checkpoint-{step}"
                        tmp_checkpoint_dir = output_dir / f".checkpoint-{step}.tmp"

                        def _do_save():
                            if args.hub_only:
                                logging.info("Saving checkpoint to a temporary staging directory for Hub upload")
                            else:
                                logging.info("Saving checkpoint to %s", checkpoint_dir)
                            log_memory_usage("Start checkpoint saving")
                            if tmp_checkpoint_dir.exists():
                                shutil.rmtree(tmp_checkpoint_dir)
                            tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                            # Move policy to CPU and clear CUDA cache to prevent GPU/VRAM OOM
                            logging.info("Moving policy to CPU...")
                            policy.to("cpu")
                            torch.cuda.empty_cache()
                            log_memory_usage("After moving policy to CPU")

                            logging.info("Saving inference-safe policy bundle...")
                            _save_exportable_artifacts(tmp_checkpoint_dir)
                            log_memory_usage("After saving policy weights")

                            # Move optimizer state dict to CPU before saving to prevent OOM
                            logging.info("Extracting optimizer state dict...")
                            opt_state_dict = optimizer.state_dict()
                            log_memory_usage("After opt_state_dict extraction")

                            logging.info("Deepcopying and offloading optimizer state dict to CPU...")
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

                            if args.hub_only:
                                logging.info("Uploading checkpoint %s to the Hub", step)
                                upload_checkpoint_to_hub(tmp_checkpoint_dir, step)
                                shutil.rmtree(tmp_checkpoint_dir)
                            else:
                                # Atomic rename
                                logging.info("Renaming temporary checkpoint directory to %s", checkpoint_dir)
                                if checkpoint_dir.exists():
                                    shutil.rmtree(checkpoint_dir)
                                tmp_checkpoint_dir.rename(checkpoint_dir)
                            logging.info("Checkpoint saved successfully at step %d", step)

                        try:
                            free_checkpoint_space(output_dir, required_space_gb=10.0)
                            _do_save()
                        except OSError as e:
                            # Clean up the temporary (youngest/latest) checkpoint to free space
                            if tmp_checkpoint_dir.exists():
                                logging.warning("[Disk Error] Write failed. Deleting temporary/incomplete checkpoint directory %s...", tmp_checkpoint_dir)
                                try:
                                    shutil.rmtree(tmp_checkpoint_dir)
                                except Exception as clean_err:
                                    logging.error("Failed to delete tmp directory: %s", clean_err)

                            # Delete the oldest checkpoint to make more space
                            logging.error("[Disk Error] Saving checkpoint failed due to write/disk space error: %s. Attempting recovery...", e)
                            free_checkpoint_space(output_dir, required_space_gb=15.0)

                            # Retry once
                            try:
                                logging.info("[Disk Recovery] Retrying checkpoint save after cleaning older checkpoints...")
                                _do_save()
                                logging.info("[Disk Recovery] Checkpoint saved successfully on retry.")
                            except Exception as retry_err:
                                logging.error("[Disk Recovery] Checkpoint saving failed even after cleanup: %s", retry_err)
                                if tmp_checkpoint_dir.exists():
                                    try:
                                        shutil.rmtree(tmp_checkpoint_dir)
                                    except Exception:
                                        pass
                                try:
                                    policy.to(device)
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass

                if accumulated_batches == 0:
                    step += 1
                    if step >= args.steps:
                        break
            except Exception:
                logging.exception("Training step failed at global step %s", step)
                raise
    finally:
        monitor.stop()

    if args.hub_only:
        logging.info("Skipping final local artifact save because --hub_only is enabled")
    else:
        logging.info("Saving final artifacts to %s", output_dir)
    log_memory_usage("Start final artifacts save")
    logging.info("Moving policy to CPU...")
    policy.to("cpu")
    torch.cuda.empty_cache()
    if not args.hub_only:
        log_memory_usage("After moving policy to CPU for final save")
        logging.info("Saving inference-safe policy bundle...")
        _save_exportable_artifacts(output_dir)
        log_memory_usage("After saving policy weights for final save")
        log_memory_usage("Completed final artifacts save")

    if args.push_to_hub:
        if not args.policy_repo_id:
            raise ValueError("--policy.repo_id is required when --push_to_hub is enabled")
        logging.info("Pushing artifacts to the Hub: %s", args.policy_repo_id)
        original_policy_config = policy.config
        try:
            policy.config = export_config
            policy.push_to_hub(args.policy_repo_id)
        finally:
            policy.config = original_policy_config
        export_preprocessor.push_to_hub(args.policy_repo_id)
        export_postprocessor.push_to_hub(args.policy_repo_id)

    logging.info("Training complete in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
