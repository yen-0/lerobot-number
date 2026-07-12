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

"""Evaluate the frozen 0707 SmolVLA teacher on MNIST without training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional fallback dependency
    load_dataset = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher.repo_id", dest="teacher_repo_id", default="yen-0/smolvla-so101-digits-0707")
    parser.add_argument("--hub_repo_id", default="yen-0/smolvla-0707-no-training-mnist-probe")
    parser.add_argument("--output_dir", default="outputs/mnist_probe_0707")
    parser.add_argument("--job_name", default="no_training_mnist_probe_0707")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher.device", dest="teacher_device", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist_root", default=".cache/mnist")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hub_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_workdir() -> Path:
    workdir = os.environ.get("PBS_O_WORKDIR")
    if workdir:
        return Path(workdir).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_output_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return _resolve_workdir() / resolved


def _resolve_device(name: str | None) -> torch.device:
    if not name or name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _image_to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
    else:
        if hasattr(image, "convert"):
            image = np.asarray(image.convert("L"))
        tensor = torch.as_tensor(image)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 3:
        tensor = tensor.mean(dim=0, keepdim=True)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() > 0 and tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


class _HfMnistDataset(Dataset):
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[int(idx)]
        return _image_to_tensor(row["image"]), int(row["label"])


def _load_mnist_dataset(root: Path, train: bool) -> Dataset:
    try:
        return MNIST(root=str(root), train=train, download=True)
    except Exception as exc:
        if load_dataset is None:
            raise
        logging.warning("Torchvision MNIST download failed; falling back to datasets load: %s", exc)
        split = "train" if train else "test"
        return _HfMnistDataset(load_dataset("ylecun/mnist", split=split))


def _subset_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return Subset(dataset, indices[:size])


def _teacher_logits(teacher: SmolVLAPolicy, images: torch.Tensor) -> torch.Tensor:
    teacher_device = torch.device(teacher.config.device or "cpu")
    images = teacher._prepare_digit_reference_images(images, device=teacher_device)
    embeddings = teacher.digit_reference_encoder(images)
    context = teacher.digit_reference_projection(embeddings)
    return teacher.digit_context_head(context)


def _evaluate(teacher: SmolVLAPolicy, loader: DataLoader, teacher_device: torch.device) -> dict[str, Any]:
    teacher.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    total_confidence = 0.0
    per_class_correct = [0 for _ in range(10)]
    per_class_total = [0 for _ in range(10)]
    confusion = [[0 for _ in range(10)] for _ in range(10)]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(teacher_device, non_blocking=True)
            labels = labels.to(teacher_device, non_blocking=True)
            logits = _teacher_logits(teacher, images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            batch_size = labels.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_confidence += float(probs.max(dim=-1).values.mean().detach().cpu()) * batch_size

            for label, pred in zip(labels.tolist(), preds.tolist(), strict=False):
                per_class_total[label] += 1
                confusion[label][pred] += 1
                if label == pred:
                    per_class_correct[label] += 1

    per_class_accuracy = {
        str(digit): (per_class_correct[digit] / per_class_total[digit] if per_class_total[digit] else 0.0)
        for digit in range(10)
    }
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "mean_confidence": total_confidence / max(total_examples, 1),
        "examples": total_examples,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def _make_repo(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)


def main() -> None:
    args = parse_args()
    teacher_device = _resolve_device(args.teacher_device or args.device)

    init_logging(console_level="INFO", file_level="DEBUG")
    logging.info("Starting MNIST probe")
    logging.info("Arguments: %s", vars(args))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.jsonl"
    api = HfApi()
    if args.push_to_hub:
        _make_repo(api, args.hub_repo_id)

    teacher = SmolVLAPolicy.from_pretrained(
        args.teacher_repo_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    teacher.to(teacher_device)
    teacher.config.device = teacher_device.type if teacher_device.index is None else str(teacher_device)
    for param in teacher.parameters():
        param.requires_grad_(False)

    mnist_root = _resolve_output_path(args.mnist_root)
    test_dataset = _load_mnist_dataset(mnist_root, train=False)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Teacher repo: %s", args.teacher_repo_id)
    logging.info("Teacher device: %s", teacher_device)
    logging.info("MNIST root: %s", mnist_root)
    logging.info("Eval examples: %d", len(test_dataset))

    metrics = _evaluate(teacher, loader, teacher_device)
    summary = {
        "job_name": args.job_name,
        "teacher_repo_id": args.teacher_repo_id,
        "examples": metrics["examples"],
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "mean_confidence": metrics["mean_confidence"],
    }

    row = {"step": 0, "phase": "eval", **metrics}
    _append_jsonl(metrics_file, row)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "metrics.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.push_to_hub:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            bundle_dir = Path(tmpdir)
            (bundle_dir / "metrics.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            (bundle_dir / "confusion_matrix.json").write_text(
                json.dumps(metrics["confusion_matrix"], indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _make_repo(api, args.hub_repo_id)
            api.upload_folder(
                repo_id=args.hub_repo_id,
                repo_type="model",
                folder_path=bundle_dir,
                path_in_repo="mnist_probe",
                commit_message="Upload MNIST probe metrics",
            )

    if args.hub_only:
        shutil.rmtree(output_dir, ignore_errors=True)

    logging.info("MNIST probe complete: accuracy=%.4f loss=%.4f", metrics["accuracy"], metrics["loss"])


if __name__ == "__main__":
    main()
