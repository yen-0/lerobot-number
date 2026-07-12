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

"""Train a small MNIST classifier with teacher distillation from the 0707 SmolVLA model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional fallback dependency
    load_dataset = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging


class MnistStudent(nn.Module):
    """Small CNN student used for MNIST classification."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        elif images.shape[1] != 1:
            raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
        return self.classifier(self.features(images))


class _HfMnistDataset(Dataset):
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[int(idx)]
        image = row["image"]
        label = int(row["label"])
        return _image_to_tensor(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher.repo_id", dest="teacher_repo_id", default="yen-0/smolvla-so101-digits-0707")
    parser.add_argument("--hub_repo_id", default="yen-0/mnist-distill-0707")
    parser.add_argument("--output_dir", default="outputs/mnist_distill")
    parser.add_argument("--job_name", default="mnist_distill")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher.device", dest="teacher_device", default=None)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--student_hidden_dim", type=int, default=64)
    parser.add_argument("--student_dropout", type=float, default=0.1)
    parser.add_argument("--save_freq", type=int, default=1_000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist_root", default=".cache/mnist")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--eval_subset", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
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


def _subset_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return Subset(dataset, indices[:size])


def _load_mnist_dataset(root: Path, train: bool) -> Dataset:
    transform = transforms.ToTensor()
    try:
        return MNIST(root=str(root), train=train, download=True, transform=transform)
    except Exception as exc:
        if load_dataset is None:
            raise
        logging.warning("Torchvision MNIST download failed; falling back to datasets load: %s", exc)
        split = "train" if train else "test"
        return _HfMnistDataset(load_dataset("ylecun/mnist", split=split))


def _prepare_teacher_images(teacher: SmolVLAPolicy, images: torch.Tensor) -> torch.Tensor:
    teacher_device = torch.device(teacher.config.device or "cpu")
    return teacher._prepare_digit_reference_images(images, device=teacher_device)


@torch.no_grad()
def _teacher_logits(teacher: SmolVLAPolicy, images: torch.Tensor) -> torch.Tensor:
    teacher_images = _prepare_teacher_images(teacher, images)
    embeddings = teacher.digit_reference_encoder(teacher_images)
    context = teacher.digit_reference_projection(embeddings)
    return teacher.digit_context_head(context)


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    distill_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
    total = ce_loss + distill_weight * kl_loss
    metrics = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, metrics


def _evaluate(
    student: MnistStudent,
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    student_device: torch.device,
    temperature: float,
    distill_weight: float,
) -> dict[str, float]:
    student.eval()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_examples = 0
    student_correct = 0
    teacher_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(student_device, non_blocking=True)
            labels = labels.to(student_device, non_blocking=True)
            student_logits = student(images)
            teacher_logits = _teacher_logits(teacher, images)
            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(student_logits.device)

            loss, metrics = _distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature,
                distill_weight=distill_weight,
            )
            batch_size = labels.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_ce += metrics["ce_loss"] * batch_size
            total_kl += metrics["kl_loss"] * batch_size
            student_correct += int((student_logits.argmax(dim=-1) == labels).sum().item())
            teacher_correct += int((teacher_logits.argmax(dim=-1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "ce_loss": total_ce / max(total_examples, 1),
        "kl_loss": total_kl / max(total_examples, 1),
        "student_accuracy": student_correct / max(total_examples, 1),
        "teacher_accuracy": teacher_correct / max(total_examples, 1),
        "examples": float(total_examples),
    }


def _save_checkpoint(
    path: Path,
    student: MnistStudent,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    step: int,
    best_eval_accuracy: float,
    args: argparse.Namespace,
    metrics: dict[str, float] | None = None,
) -> None:
    payload = {
        "step": step,
        "best_eval_accuracy": best_eval_accuracy,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "metrics": metrics or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def _make_repo(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)


def _stage_bundle(
    bundle_dir: Path,
    *,
    student: MnistStudent,
    step: int,
    best_eval_accuracy: float,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
    summary: dict[str, Any] | None,
    history: list[dict[str, Any]],
    include_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), bundle_dir / "student_state.pt")
    torch.save(optimizer.state_dict(), bundle_dir / "optimizer_state.pt")
    if include_scheduler is not None:
        torch.save(include_scheduler.state_dict(), bundle_dir / "scheduler_state.pt")
    (bundle_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    (bundle_dir / "status.json").write_text(
        json.dumps(
            {
                "step": step,
                "best_eval_accuracy": best_eval_accuracy,
                "metrics": metrics or {},
                "summary": summary or {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in history) + ("\n" if history else ""),
        encoding="utf-8",
    )


def _upload_bundle(api: HfApi, repo_id: str, bundle_dir: Path, path_in_repo: str, message: str) -> None:
    _make_repo(api, repo_id)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=bundle_dir,
        path_in_repo=path_in_repo,
        commit_message=message,
    )


def _upload_checkpoint(
    api: HfApi,
    repo_id: str,
    *,
    student: MnistStudent,
    step: int,
    best_eval_accuracy: float,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
    history: list[dict[str, Any]],
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    optimizer: torch.optim.Optimizer,
    path_in_repo: str,
) -> None:
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bundle_dir = Path(tmpdir)
        _stage_bundle(
            bundle_dir,
            student=student,
            step=step,
            best_eval_accuracy=best_eval_accuracy,
            args=args,
            metrics=metrics,
            summary=None,
            history=history,
            include_scheduler=scheduler,
            optimizer=optimizer,
        )
        _upload_bundle(api, repo_id, bundle_dir, path_in_repo, f"Upload MNIST distillation checkpoint at step {step}")


def main() -> None:
    args = parse_args()
    student_device = _resolve_device(args.device)
    teacher_device = _resolve_device(args.teacher_device or args.device)

    init_logging(console_level="INFO", file_level="DEBUG")
    logging.info("Starting MNIST distillation")
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

    teacher_config = SmolVLAConfig.from_pretrained(
        args.teacher_repo_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    teacher_config.device = teacher_device.type if teacher_device.index is None else str(teacher_device)
    teacher = SmolVLAPolicy.from_pretrained(
        args.teacher_repo_id,
        config=teacher_config,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = MnistStudent(hidden_dim=args.student_hidden_dim, dropout=args.student_dropout).to(student_device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.steps, 1))

    mnist_root = _resolve_output_path(args.mnist_root)
    train_dataset = _subset_dataset(_load_mnist_dataset(mnist_root, train=True), args.train_subset, args.seed)
    eval_dataset = _subset_dataset(_load_mnist_dataset(mnist_root, train=False), args.eval_subset, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=student_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=student_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    logging.info("MNIST root: %s", mnist_root)
    logging.info("Teacher repo: %s", args.teacher_repo_id)
    logging.info("Student device: %s", student_device)
    logging.info("Teacher device: %s", teacher_device)
    logging.info("Train examples: %d", len(train_dataset))
    logging.info("Eval examples: %d", len(eval_dataset))

    history: list[dict[str, Any]] = []

    initial_eval = _evaluate(
        student=student,
        teacher=teacher,
        loader=eval_loader,
        student_device=student_device,
        temperature=args.temperature,
        distill_weight=args.distill_weight,
    )
    logging.info("Initial eval: %s", initial_eval)
    best_eval_accuracy = initial_eval["student_accuracy"]
    initial_row = {"step": 0, "phase": "eval", **initial_eval}
    history.append(initial_row)
    _append_jsonl(metrics_file, initial_row)
    if args.push_to_hub:
        _upload_checkpoint(
            api,
            args.hub_repo_id,
            student=student,
            step=0,
            best_eval_accuracy=best_eval_accuracy,
            args=args,
            metrics=initial_eval,
            history=history,
            scheduler=scheduler,
            optimizer=optimizer,
            path_in_repo="checkpoints/checkpoint-000000",
        )

    train_iter = iter(train_loader)
    start_time = time.time()
    last_log_time = start_time

    for step in range(1, args.steps + 1):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(student_device, non_blocking=True)
        labels = labels.to(student_device, non_blocking=True)

        student.train()
        teacher.eval()

        autocast_ctx = (
            torch.amp.autocast(device_type=student_device.type, dtype=torch.bfloat16)
            if args.amp and student_device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            teacher_batch = images.to(torch.device(teacher.config.device or "cpu"))
            teacher_logits = _teacher_logits(teacher, teacher_batch)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            student_logits = student(images)
            loss, batch_metrics = _distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits.to(student_logits.device),
                labels=labels,
                temperature=args.temperature,
                distill_weight=args.distill_weight,
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_accuracy = float((student_logits.argmax(dim=-1) == labels).float().mean().detach().cpu())

        if step % args.log_freq == 0 or step == 1:
            now = time.time()
            log_row = {
                "step": step,
                "phase": "train",
                "loss": float(loss.detach().cpu()),
                "accuracy": batch_accuracy,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": now - start_time,
                "since_last_log_s": now - last_log_time,
                **batch_metrics,
            }
            logging.info("Train step %s: %s", step, log_row)
            history.append(log_row)
            _append_jsonl(metrics_file, log_row)
            last_log_time = now

        if step % args.eval_freq == 0 or step == args.steps:
            eval_metrics = _evaluate(
                student=student,
                teacher=teacher,
                loader=eval_loader,
                student_device=student_device,
                temperature=args.temperature,
                distill_weight=args.distill_weight,
            )
            logging.info("Eval step %s: %s", step, eval_metrics)
            eval_row = {"step": step, "phase": "eval", **eval_metrics}
            history.append(eval_row)
            _append_jsonl(metrics_file, eval_row)
            if eval_metrics["student_accuracy"] >= best_eval_accuracy:
                best_eval_accuracy = eval_metrics["student_accuracy"]
                if args.push_to_hub:
                    _upload_checkpoint(
                        api,
                        args.hub_repo_id,
                        student=student,
                        step=step,
                        best_eval_accuracy=best_eval_accuracy,
                        args=args,
                        metrics=eval_metrics,
                        history=history,
                        scheduler=scheduler,
                        optimizer=optimizer,
                        path_in_repo="best",
                    )

        if step % args.save_freq == 0 or step == args.steps:
            if args.push_to_hub:
                _upload_checkpoint(
                    api,
                    args.hub_repo_id,
                    student=student,
                    step=step,
                    best_eval_accuracy=best_eval_accuracy,
                    args=args,
                    metrics=batch_metrics,
                    history=history,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    path_in_repo=f"checkpoints/checkpoint-{step:06d}",
                )

    final_metrics = _evaluate(
        student=student,
        teacher=teacher,
        loader=eval_loader,
        student_device=student_device,
        temperature=args.temperature,
        distill_weight=args.distill_weight,
    )
    logging.info("Final eval: %s", final_metrics)
    final_row = {"step": args.steps, "phase": "final_eval", **final_metrics}
    history.append(final_row)
    _append_jsonl(metrics_file, final_row)

    summary = {
        "best_eval_accuracy": best_eval_accuracy,
        "final_eval": final_metrics,
        "job_name": args.job_name,
        "teacher_repo_id": args.teacher_repo_id,
        "student_hidden_dim": args.student_hidden_dim,
        "steps": args.steps,
    }
    if args.push_to_hub:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            bundle_dir = Path(tmpdir)
            _stage_bundle(
                bundle_dir,
                student=student,
                step=args.steps,
                best_eval_accuracy=best_eval_accuracy,
                args=args,
                metrics=final_metrics,
                summary=summary,
                history=history,
                include_scheduler=scheduler,
                optimizer=optimizer,
            )
            (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            _upload_bundle(
                api,
                args.hub_repo_id,
                bundle_dir,
                "final",
                "Upload final MNIST distillation artifacts",
            )

    if args.hub_only:
        shutil.rmtree(output_dir, ignore_errors=True)

    logging.info("Done. Best student eval accuracy: %.4f", best_eval_accuracy)


if __name__ == "__main__":
    main()
