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
import torch.nn.functional as F
from huggingface_hub import HfApi
from PIL import Image, ImageOps
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher.repo_id", dest="teacher_repo_id", default="yen-0/smolvla-so101-digits-0707")
    parser.add_argument("--hub_repo_id", default="yen-0/smolvla-0707-no-training-mnist-probe")
    parser.add_argument("--output_dir", default="outputs/no_training_mnist_probe_0707")
    parser.add_argument("--job_name", default="no_training_mnist_probe_0707")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher.device", dest="teacher_device", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist_root", default=".cache/mnist")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--probe_train_subset", type=int, default=5000)
    parser.add_argument("--probe_eval_subset", type=int, default=2000)
    parser.add_argument("--ablation_subset", type=int, default=2000)
    parser.add_argument("--attribution_examples", type=int, default=4)
    parser.add_argument("--ridge_l2", type=float, default=1e-2)
    parser.add_argument("--saliency_topk_fraction", type=float, default=0.1)
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
        return MNIST(root=str(root), train=train, download=True, transform=transforms.ToTensor())
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


def _resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module: torch.nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _tap_modules(teacher: SmolVLAPolicy) -> dict[str, str]:
    return {
        "encoder_conv1": "digit_reference_encoder.encoder.0",
        "encoder_conv2": "digit_reference_encoder.encoder.2",
        "encoder_conv3": "digit_reference_encoder.encoder.4",
        "encoder_pool": "digit_reference_encoder.encoder.6",
        "encoder_embed": "digit_reference_encoder.encoder.8",
        "projection": "digit_reference_projection",
    }


def _ablation_targets(teacher: SmolVLAPolicy) -> dict[str, str]:
    targets = _tap_modules(teacher)
    targets.update(
        {
            "context_head": "digit_context_head",
        }
    )
    return targets


def _pool_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 4:
        activation = F.adaptive_avg_pool2d(activation, (1, 1)).flatten(1)
    elif activation.ndim == 3:
        activation = activation.flatten(1)
    elif activation.ndim != 2:
        activation = activation.reshape(activation.shape[0], -1)
    return activation.float()


def _teacher_logits(teacher: SmolVLAPolicy, images: torch.Tensor) -> torch.Tensor:
    teacher_device = torch.device(teacher.config.device or "cpu")
    images = teacher._prepare_digit_reference_images(images, device=teacher_device)
    embeddings = teacher.digit_reference_encoder(images)
    context = teacher.digit_reference_projection(embeddings)
    return teacher.digit_context_head(context)


def _capture_taps(
    teacher: SmolVLAPolicy,
    images: torch.Tensor,
    tap_paths: dict[str, str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            activations[name] = output.detach()

        return _hook

    for tap_name, tap_path in tap_paths.items():
        module = _resolve_module(teacher, tap_path)
        hooks.append(module.register_forward_hook(_make_hook(tap_name)))

    try:
        logits = _teacher_logits(teacher, images)
    finally:
        for hook in hooks:
            hook.remove()

    return logits, activations


def _evaluate(
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    teacher_device: torch.device,
) -> dict[str, Any]:
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


def _evaluate_ablation(
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    teacher_device: torch.device,
    target_path: str | None,
    *,
    zero_input: bool = False,
) -> dict[str, Any]:
    teacher.eval()
    if zero_input:
        total_examples = 0
        total_loss = 0.0
        total_correct = 0
        total_confidence = 0.0
        per_class_correct = [0 for _ in range(10)]
        per_class_total = [0 for _ in range(10)]
        confusion = [[0 for _ in range(10)] for _ in range(10)]

        with torch.no_grad():
            for images, labels in loader:
                images = torch.zeros_like(images).to(teacher_device, non_blocking=True)
                labels = labels.to(teacher_device, non_blocking=True)
                logits = _teacher_logits(teacher, images)
                loss = F.cross_entropy(logits, labels)
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

    target_module = _resolve_module(teacher, target_path) if target_path is not None else None
    hook = None
    if target_module is not None:
        hook = target_module.register_forward_hook(lambda _m, _i, output: torch.zeros_like(output))

    try:
        metrics = _evaluate(teacher, loader, teacher_device)
    finally:
        if hook is not None:
            hook.remove()
    return metrics


def _collect_feature_bank(
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    teacher_device: torch.device,
    tap_paths: dict[str, str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    feature_rows: dict[str, list[torch.Tensor]] = {name: [] for name in tap_paths}
    labels: list[torch.Tensor] = []

    teacher.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(teacher_device, non_blocking=True)
            batch_labels = batch_labels.to(teacher_device, non_blocking=True)
            _, activations = _capture_taps(teacher, images, tap_paths)
            for tap_name, activation in activations.items():
                feature_rows[tap_name].append(_pool_activation(activation).cpu())
            labels.append(batch_labels.cpu())

    feature_bank = {name: torch.cat(rows, dim=0) for name, rows in feature_rows.items()}
    label_tensor = torch.cat(labels, dim=0)
    return feature_bank, label_tensor


def _select_attribution_examples(
    teacher: SmolVLAPolicy,
    dataset: Dataset,
    teacher_device: torch.device,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []

    teacher.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            image_tensor = _image_to_tensor(image)
            logits = _teacher_logits(teacher, image_tensor.unsqueeze(0).to(teacher_device))
            pred = int(logits.argmax(dim=-1).item())
            item = {
                "index": idx,
                "label": int(label),
                "predicted": pred,
                "confidence": float(torch.softmax(logits, dim=-1)[0, pred].item()),
                "image": image_tensor,
            }
            fallback.append(item)
            if pred == label and label not in selected:
                selected[label] = item
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        for item in fallback:
            if item["label"] not in selected:
                selected[item["label"]] = item
            if len(selected) >= limit:
                break

    ordered = list(selected.values())[:limit]
    ordered.sort(key=lambda item: item["index"])
    return ordered


def _fit_ridge_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int = 10,
    ridge_l2: float = 1e-2,
) -> dict[str, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {tuple(features.shape)}")

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    normalized = (features - mean) / std
    design = torch.cat([normalized, torch.ones(normalized.shape[0], 1)], dim=1)
    targets = F.one_hot(labels.to(torch.long), num_classes=num_classes).float()
    eye = torch.eye(design.shape[1], dtype=design.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge_l2 * eye, design.T @ targets)
    return {"mean": mean, "std": std, "weights": weights}


def _predict_ridge_probe(probe: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    normalized = (features - probe["mean"]) / probe["std"]
    design = torch.cat([normalized, torch.ones(normalized.shape[0], 1)], dim=1)
    scores = design @ probe["weights"]
    return scores.argmax(dim=-1)


def _evaluate_ridge_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    ridge_l2: float,
) -> dict[str, float]:
    probe = _fit_ridge_probe(train_features, train_labels, ridge_l2=ridge_l2)
    eval_pred = _predict_ridge_probe(probe, eval_features)
    train_pred = _predict_ridge_probe(probe, train_features)
    return {
        "train_accuracy": float((train_pred == train_labels).float().mean().item()),
        "eval_accuracy": float((eval_pred == eval_labels).float().mean().item()),
    }


def _saliency_map(
    teacher: SmolVLAPolicy,
    image: torch.Tensor,
    label: int,
    teacher_device: torch.device,
    *,
    use_predicted_label: bool = False,
    topk_fraction: float = 0.1,
) -> dict[str, Any]:
    teacher.eval()
    image = image.unsqueeze(0).to(teacher_device).clone().detach().requires_grad_(True)
    logits = _teacher_logits(teacher, image)
    probs = torch.softmax(logits, dim=-1)
    pred = int(probs.argmax(dim=-1).item())
    target = pred if use_predicted_label else label
    teacher.zero_grad(set_to_none=True)
    score = logits[0, target]
    score.backward()
    saliency = image.grad.detach().abs()[0].mean(dim=0)
    saliency = saliency / saliency.sum().clamp_min(1e-8)
    topk = max(1, int(saliency.numel() * topk_fraction))
    concentration = float(saliency.flatten().topk(topk).values.sum().item())
    flat_top = saliency.flatten().topk(5)
    top_positions = [
        {"index": int(idx), "y": int(idx // saliency.shape[1]), "x": int(idx % saliency.shape[1]), "value": float(val)}
        for idx, val in zip(flat_top.indices.tolist(), flat_top.values.tolist(), strict=False)
    ]
    return {
        "label": int(label),
        "predicted": pred,
        "target": int(target),
        "confidence": float(probs[0, pred].item()),
        "concentration_top10pct": concentration,
        "top_pixels": top_positions,
        "saliency": saliency.detach().cpu(),
    }


def _save_attribution_artifacts(
    output_dir: Path,
    example_idx: int,
    image: torch.Tensor,
    saliency: torch.Tensor,
    label: int,
    predicted: int,
) -> dict[str, str]:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    image_arr = (image.squeeze(0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    saliency_arr = saliency.clamp_min(0).cpu().numpy()
    if saliency_arr.max() > 0:
        saliency_arr = saliency_arr / saliency_arr.max()
    saliency_img = Image.fromarray((saliency_arr * 255.0).astype(np.uint8), mode="L")
    original_img = Image.fromarray(image_arr, mode="L")

    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:  # pragma: no cover - older Pillow
        resample = Image.NEAREST

    original_img = original_img.resize((224, 224), resample=resample).convert("RGB")
    saliency_img = saliency_img.resize((224, 224), resample=resample)
    heatmap_img = ImageOps.colorize(saliency_img, black="#000000", white="#ff3b30")
    overlay_img = Image.blend(original_img, heatmap_img.convert("RGB"), alpha=0.55)
    combined = Image.new("RGB", (224 * 3, 224), "white")
    combined.paste(original_img, (0, 0))
    combined.paste(heatmap_img.convert("RGB"), (224, 0))
    combined.paste(overlay_img, (224 * 2, 0))

    stem = f"attribution_{example_idx:02d}_true{label}_pred{predicted}"
    orig_path = assets_dir / f"{stem}_input.png"
    heat_path = assets_dir / f"{stem}_saliency.png"
    combo_path = assets_dir / f"{stem}_combined.png"
    original_img.save(orig_path)
    heatmap_img.save(heat_path)
    combined.save(combo_path)

    return {
        "input": str(orig_path.relative_to(output_dir)),
        "saliency": str(heat_path.relative_to(output_dir)),
        "combined": str(combo_path.relative_to(output_dir)),
    }


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _build_report(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    ablations: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
) -> str:
    best_ablation = max(ablations, key=lambda row: float(baseline["accuracy"]) - float(row["accuracy"]))
    best_probe = max(probes, key=lambda row: float(row["eval_accuracy"]))
    conclusion = (
        f"The strongest causal dependency is `{best_ablation['target']}` with an accuracy drop of "
        f"{_format_pct(float(baseline['accuracy']) - float(best_ablation['accuracy']))}. "
        f"The best linear probe is `{best_probe['tap']}` at {_format_pct(float(best_probe['eval_accuracy']))}, "
        f"so digit information is already most accessible at that tap."
    )
    report = []
    report.append(f"# MNIST No-Training Probe for {args.teacher_repo_id}")
    report.append("")
    report.append("This run freezes the teacher and measures how well the existing model already recognizes MNIST digits.")
    report.append("No weights are updated.")
    report.append("")
    report.append("## Setup")
    report.append(f"- Teacher repo: `{args.teacher_repo_id}`")
    report.append(f"- Probe repo: `{args.hub_repo_id}`")
    report.append(f"- Eval subset: `{baseline['examples']}` examples")
    report.append(f"- Probe train subset: `{args.probe_train_subset}` examples")
    report.append(f"- Probe eval subset: `{args.probe_eval_subset}` examples")
    report.append(f"- Ablation subset: `{args.ablation_subset}` examples")
    report.append(f"- Attribution examples: `{args.attribution_examples}`")
    report.append(f"- Ridge L2: `{args.ridge_l2}`")
    report.append(f"- Saliency top fraction: `{args.saliency_topk_fraction}`")
    report.append("")
    report.append("## Frozen Baseline")
    report.append(
        _render_table(
            ["Metric", "Value"],
            [
                ["Accuracy", _format_pct(float(baseline["accuracy"]))],
                ["Loss", _format_float(float(baseline["loss"]))],
                ["Mean confidence", _format_float(float(baseline["mean_confidence"]))],
            ],
        )
    )
    report.append("")
    report.append("## Causal Ablations")
    report.append(
        _render_table(
            ["Target", "Accuracy", "Drop vs baseline", "Loss"],
            [
                [
                    item["target"],
                    _format_pct(float(item["accuracy"])),
                    _format_pct(float(baseline["accuracy"]) - float(item["accuracy"])),
                    _format_float(float(item["loss"])),
                ]
                for item in sorted(ablations, key=lambda row: float(baseline["accuracy"]) - float(row["accuracy"]), reverse=True)
            ],
        )
    )
    report.append("")
    report.append("## Linear Diagnostic Probes")
    report.append(
        _render_table(
            ["Feature tap", "Train acc", "Eval acc"],
            [
                [
                    item["tap"],
                    _format_pct(float(item["train_accuracy"])),
                    _format_pct(float(item["eval_accuracy"])),
                ]
                for item in sorted(probes, key=lambda row: float(row["eval_accuracy"]), reverse=True)
            ],
        )
    )
    report.append("")
    report.append("## Attribution Examples")
    for item in attributions:
        report.append(f"### Sample {item['index']:02d}")
        report.append(
            f"- True label: `{item['label']}` | Predicted: `{item['predicted']}` | Confidence: `{_format_float(float(item['confidence']))}`"
        )
        report.append(f"- Saliency concentration top 10%: `{_format_pct(float(item['concentration_top10pct']))}`")
        report.append(f"- Top pixels: `{json.dumps(item['top_pixels'])}`")
        report.append("")
        report.append(f"![input](./{item['artifacts']['input']})")
        report.append(f"![saliency](./{item['artifacts']['saliency']})")
        report.append(f"![combined](./{item['artifacts']['combined']})")
        report.append("")
    report.append("## Readout")
    report.append(conclusion)
    report.append(
        "The strongest causal evidence comes from the ablation that drops accuracy the most. "
        "The linear probes show where digit information becomes linearly accessible, and the saliency maps show which pixels drive the frozen prediction."
    )
    return "\n".join(report) + "\n"


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
    train_dataset = _load_mnist_dataset(mnist_root, train=True)
    test_dataset = _load_mnist_dataset(mnist_root, train=False)
    baseline_dataset = _subset_dataset(test_dataset, args.ablation_subset, args.seed)
    probe_train_dataset = _subset_dataset(train_dataset, args.probe_train_subset, args.seed + 1)
    probe_eval_dataset = _subset_dataset(test_dataset, args.probe_eval_subset, args.seed + 2)

    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    probe_train_loader = DataLoader(
        probe_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    probe_eval_loader = DataLoader(
        probe_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Teacher repo: %s", args.teacher_repo_id)
    logging.info("Teacher device: %s", teacher_device)
    logging.info("MNIST root: %s", mnist_root)
    logging.info("Baseline eval examples: %d", len(baseline_dataset))
    logging.info("Probe train examples: %d", len(probe_train_dataset))
    logging.info("Probe eval examples: %d", len(probe_eval_dataset))

    baseline = _evaluate(teacher, baseline_loader, teacher_device)
    _append_jsonl(metrics_file, {"phase": "baseline", **baseline})

    ablation_targets = _ablation_targets(teacher)
    ablation_results: list[dict[str, Any]] = []
    for name, path in ablation_targets.items():
        metrics = _evaluate_ablation(teacher, baseline_loader, teacher_device, path)
        ablation_row = {"target": name, **metrics}
        ablation_results.append(ablation_row)
        _append_jsonl(metrics_file, {"phase": "ablation", **ablation_row})

    zero_input_metrics = _evaluate_ablation(teacher, baseline_loader, teacher_device, target_path=None, zero_input=True)
    zero_input_row = {"target": "zero_input", **zero_input_metrics}
    ablation_results.append(zero_input_row)
    _append_jsonl(metrics_file, {"phase": "ablation", **zero_input_row})

    tap_paths = _tap_modules(teacher)
    train_features, train_labels = _collect_feature_bank(teacher, probe_train_loader, teacher_device, tap_paths)
    eval_features, eval_labels = _collect_feature_bank(teacher, probe_eval_loader, teacher_device, tap_paths)

    probe_results: list[dict[str, Any]] = []
    for tap_name in tap_paths:
        probe_metrics = _evaluate_ridge_probe(
            train_features[tap_name],
            train_labels,
            eval_features[tap_name],
            eval_labels,
            ridge_l2=args.ridge_l2,
        )
        row = {"tap": tap_name, **probe_metrics}
        probe_results.append(row)
        _append_jsonl(metrics_file, {"phase": "linear_probe", **row})

    attribution_dataset = _subset_dataset(test_dataset, max(args.attribution_examples * 4, args.attribution_examples), args.seed + 3)
    selected_examples = _select_attribution_examples(
        teacher,
        attribution_dataset,
        teacher_device,
        limit=args.attribution_examples,
    )

    attribution_results: list[dict[str, Any]] = []
    for item in selected_examples:
        saliency = _saliency_map(
            teacher,
            item["image"],
            item["label"],
            teacher_device,
            topk_fraction=args.saliency_topk_fraction,
        )
        artifacts = _save_attribution_artifacts(
            output_dir,
            item["index"],
            item["image"],
            saliency["saliency"],
            item["label"],
            saliency["predicted"],
        )
        attr_row = {
            "index": item["index"],
            "label": item["label"],
            "predicted": saliency["predicted"],
            "confidence": saliency["confidence"],
            "concentration_top10pct": saliency["concentration_top10pct"],
            "top_pixels": saliency["top_pixels"],
            "artifacts": artifacts,
        }
        attribution_results.append(attr_row)
        _append_jsonl(metrics_file, {"phase": "attribution", **attr_row})

    summary = {
        "job_name": args.job_name,
        "teacher_repo_id": args.teacher_repo_id,
        "baseline_accuracy": baseline["accuracy"],
        "baseline_loss": baseline["loss"],
        "best_ablation_drop": max(
            baseline["accuracy"] - float(row["accuracy"]) for row in ablation_results
        ),
        "best_linear_probe_accuracy": max(float(row["eval_accuracy"]) for row in probe_results),
        "attribution_examples": len(attribution_results),
    }

    report = _build_report(args, baseline, ablation_results, probe_results, attribution_results)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "ablations.json").write_text(json.dumps(ablation_results, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "linear_probes.json").write_text(json.dumps(probe_results, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "attributions.json").write_text(json.dumps(attribution_results, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "ablations": ablation_results,
                "linear_probes": probe_results,
                "attributions": attribution_results,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.push_to_hub:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            bundle_dir = Path(tmpdir)
            for name in [
                "report.md",
                "summary.json",
                "baseline.json",
                "ablations.json",
                "linear_probes.json",
                "attributions.json",
                "analysis.json",
                "metrics.jsonl",
            ]:
                shutil.copy2(output_dir / name, bundle_dir / name)
            assets_src = output_dir / "assets"
            if assets_src.exists():
                shutil.copytree(assets_src, bundle_dir / "assets")
            _make_repo(api, args.hub_repo_id)
            api.upload_folder(
                repo_id=args.hub_repo_id,
                repo_type="model",
                folder_path=bundle_dir,
                path_in_repo="no_training_mnist_probe_0707",
                commit_message="Upload MNIST no-training probe report",
            )

    if args.hub_only and args.push_to_hub:
        shutil.rmtree(output_dir, ignore_errors=True)

    logging.info(
        "MNIST probe complete: baseline_accuracy=%.4f best_ablation_drop=%.4f best_linear_probe_accuracy=%.4f",
        baseline["accuracy"],
        summary["best_ablation_drop"],
        summary["best_linear_probe_accuracy"],
    )


if __name__ == "__main__":
    main()
