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

"""Analyze where frozen MNIST digit information lives inside SmolVLA without training."""

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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks, masked_mean
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher.repo_id", dest="teacher_repo_id", default="yen-0/smolvla-so101-digits-0707")
    parser.add_argument("--hub_repo_id", default="yen-0/smolvla-0707-blockwise-mnist-analysis")
    parser.add_argument("--output_dir", default="outputs/blockwise_mnist_analysis_0707")
    parser.add_argument("--job_name", default="blockwise_mnist_analysis_0707")
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


def _first_image_feature_key(teacher: SmolVLAPolicy) -> str:
    image_features = teacher.config.image_features
    if isinstance(image_features, dict):
        return next(iter(image_features))
    if isinstance(image_features, (list, tuple)):
        return str(image_features[0])
    return str(image_features)


def _build_context_batch(
    teacher: SmolVLAPolicy, images: torch.Tensor, teacher_device: torch.device
) -> dict[str, torch.Tensor]:
    if images.ndim == 4 and images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    tokenizer = teacher.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(
        "MNIST digit recognition.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=teacher.config.tokenizer_max_length,
    )
    batch_size = images.shape[0]
    lang_tokens = encoded["input_ids"].expand(batch_size, -1).contiguous().to(device=teacher_device)
    lang_masks = encoded["attention_mask"].expand(batch_size, -1).contiguous().to(device=teacher_device).bool()
    state = torch.zeros(batch_size, teacher.config.max_state_dim, device=teacher_device)
    return {
        _first_image_feature_key(teacher): images.to(device=teacher_device),
        OBS_LANGUAGE_TOKENS: lang_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        OBS_STATE: state,
    }


def _backbone_stage_names(teacher: SmolVLAPolicy) -> list[str]:
    num_layers = teacher.model.vlm_with_expert.num_vlm_layers
    return [
        "vision_encoder",
        "connector",
        "prefix_input",
        *[f"text_layer_{layer_idx:02d}" for layer_idx in range(num_layers)],
        "text_final_norm",
    ]


def _pool_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 4:
        activation = F.adaptive_avg_pool2d(activation, (1, 1)).flatten(1)
    elif activation.ndim == 3:
        activation = activation.mean(dim=1)
    elif activation.ndim != 2:
        activation = activation.reshape(activation.shape[0], -1)
    return activation.float()


def _run_backbone_context(
    teacher: SmolVLAPolicy,
    images: torch.Tensor,
    teacher_device: torch.device,
    *,
    zero_stage: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    activations: dict[str, torch.Tensor] = {}
    batch = _build_context_batch(teacher, images, teacher_device)
    prepared_images, img_masks = teacher.prepare_images(batch)
    state = torch.zeros(images.shape[0], teacher.config.max_state_dim, device=teacher_device)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    vlm = teacher.model.vlm_with_expert
    models = [vlm.get_vlm_model().text_model, vlm.lm_expert]
    model_layers = vlm.get_model_layers(models)
    head_dim = vlm.vlm.config.text_config.head_dim

    vision_hidden = vlm.get_vlm_model().vision_model(
        pixel_values=prepared_images[0].to(dtype=vlm.get_vlm_model().vision_model.dtype),
        patch_attention_mask=None,
    ).last_hidden_state
    if zero_stage == "vision_encoder":
        vision_hidden = torch.zeros_like(vision_hidden)
    activations["vision_encoder"] = _pool_activation(vision_hidden)

    connector_hidden = vlm.get_vlm_model().connector(vision_hidden)
    if zero_stage == "connector":
        connector_hidden = torch.zeros_like(connector_hidden)
    activations["connector"] = _pool_activation(connector_hidden)

    prefix_embs, prefix_pad_masks, prefix_att_masks = teacher.model.embed_prefix(
        prepared_images, img_masks, lang_tokens, lang_masks, state=state
    )
    if zero_stage == "prefix_input":
        prefix_embs = torch.zeros_like(prefix_embs)
    activations["prefix_input"] = masked_mean(prefix_embs, prefix_pad_masks)

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    hidden_states = prefix_embs
    for layer_idx in range(vlm.num_vlm_layers):
        att_outputs, _ = vlm.forward_attn_layer(
            model_layers,
            [hidden_states, None],
            layer_idx,
            prefix_position_ids,
            prefix_att_2d_masks,
            batch_size=hidden_states.shape[0],
            head_dim=head_dim,
            use_cache=False,
            fill_kv_cache=True,
            past_key_values=None,
        )
        layer = model_layers[0][layer_idx]
        att_output = att_outputs[0].to(dtype=layer.self_attn.o_proj.weight.dtype)
        hidden_states = hidden_states.to(dtype=layer.self_attn.o_proj.weight.dtype)
        residual = layer.self_attn.o_proj(att_output) + hidden_states
        after_residual = residual.clone()
        hidden_states = layer.post_attention_layernorm(residual)
        hidden_states = hidden_states.to(dtype=layer.mlp.gate_proj.weight.dtype)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = hidden_states + after_residual
        if zero_stage == f"text_layer_{layer_idx:02d}":
            hidden_states = torch.zeros_like(hidden_states)
        activations[f"text_layer_{layer_idx:02d}"] = masked_mean(hidden_states, prefix_pad_masks)

    final_hidden = models[0].norm(hidden_states.to(dtype=models[0].norm.weight.dtype))
    if zero_stage == "text_final_norm":
        final_hidden = torch.zeros_like(final_hidden)
    activations["text_final_norm"] = masked_mean(final_hidden, prefix_pad_masks)
    logits = teacher.digit_context_head(activations["text_final_norm"])
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
            logits, _ = _run_backbone_context(teacher, images, teacher_device)
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
) -> dict[str, Any]:
    zero_stage = target_path
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
            logits, _ = _run_backbone_context(teacher, images, teacher_device, zero_stage=zero_stage)
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


def _collect_feature_bank(
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    teacher_device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    feature_rows: dict[str, list[torch.Tensor]] = {name: [] for name in _backbone_stage_names(teacher)}
    labels: list[torch.Tensor] = []

    teacher.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(teacher_device, non_blocking=True)
            batch_labels = batch_labels.to(teacher_device, non_blocking=True)
            _, activations = _run_backbone_context(teacher, images, teacher_device)
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
            logits, _ = _run_backbone_context(teacher, image_tensor.unsqueeze(0).to(teacher_device), teacher_device)
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
    logits, _ = _run_backbone_context(teacher, image, teacher_device)
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
    backbone_probes: list[dict[str, Any]],
    backbone_ablations: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
) -> str:
    best_ablation = max(ablations, key=lambda row: float(baseline["accuracy"]) - float(row["accuracy"]))
    best_backbone_probe = max(backbone_probes, key=lambda row: float(row["eval_accuracy"]))
    best_backbone_ablation = max(backbone_ablations, key=lambda row: float(baseline["accuracy"]) - float(row["accuracy"]))
    ablation_by_target = {row["target"]: row for row in backbone_ablations}
    conclusion = (
        f"The strongest causal dependency is `{best_ablation['target']}` with an accuracy drop of "
        f"{_format_pct(float(baseline['accuracy']) - float(best_ablation['accuracy']))}. "
        f"The strongest backbone probe is `{best_backbone_probe['tap']}` at "
        f"{_format_pct(float(best_backbone_probe['eval_accuracy']))}. "
        f"The most damaging backbone ablation is `{best_backbone_ablation['target']}` with a drop of "
        f"{_format_pct(float(baseline['accuracy']) - float(best_backbone_ablation['accuracy']))}, "
        f"so the digit signal is most readable and most causally exposed in that block."
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
    report.append("## Backbone Analysis")
    report.append(
        "These probes use a constant prompt and zero state so the only changing input is the MNIST image. "
        "The table combines linear readability and causal sensitivity for each stage of the frozen backbone."
    )
    report.append(
        _render_table(
            ["Block", "Train acc", "Eval acc", "Drop vs baseline", "Loss"],
            [
                [
                    item["tap"],
                    _format_pct(float(item["train_accuracy"])),
                    _format_pct(float(item["eval_accuracy"])),
                    _format_pct(float(baseline["accuracy"]) - float(ablation_by_target[item["tap"]]["accuracy"])),
                    _format_float(float(ablation_by_target[item["tap"]]["loss"])),
                ]
                for item in backbone_probes
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
        "The causal ablations show which frozen modules the classifier truly depends on. "
        "The backbone analysis shows where digit information is linearly recoverable and where removing a stage hurts the prediction."
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
    logging.info("Starting MNIST block analysis")
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
    logging.info("Resolved output dir: %s", output_dir)
    logging.info("Resolved hub repo: %s", args.hub_repo_id)
    logging.info("Resolved upload folder: %s", "blockwise_mnist_analysis_0707")

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

    ablation_targets = {name: name for name in _backbone_stage_names(teacher)}
    ablation_results: list[dict[str, Any]] = []
    for name, path in ablation_targets.items():
        metrics = _evaluate_ablation(teacher, baseline_loader, teacher_device, path)
        ablation_row = {"target": name, **metrics}
        ablation_results.append(ablation_row)
        _append_jsonl(metrics_file, {"phase": "ablation", **ablation_row})

    backbone_train_features, backbone_train_labels = _collect_feature_bank(teacher, probe_train_loader, teacher_device)
    backbone_eval_features, backbone_eval_labels = _collect_feature_bank(teacher, probe_eval_loader, teacher_device)

    backbone_probe_results: list[dict[str, Any]] = []
    for tap_name in _backbone_stage_names(teacher):
        probe_metrics = _evaluate_ridge_probe(
            backbone_train_features[tap_name],
            backbone_train_labels,
            backbone_eval_features[tap_name],
            backbone_eval_labels,
            ridge_l2=args.ridge_l2,
        )
        row = {"tap": tap_name, **probe_metrics}
        backbone_probe_results.append(row)
        _append_jsonl(metrics_file, {"phase": "backbone_probe", **row})

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
        "best_backbone_probe_accuracy": max(float(row["eval_accuracy"]) for row in backbone_probe_results),
        "best_backbone_probe_tap": max(backbone_probe_results, key=lambda row: float(row["eval_accuracy"]))["tap"],
        "best_backbone_ablation_drop": max(
            baseline["accuracy"] - float(row["accuracy"]) for row in ablation_results
        ),
        "best_backbone_ablation_tap": max(
            ablation_results, key=lambda row: baseline["accuracy"] - float(row["accuracy"])
        )["target"],
        "attribution_examples": len(attribution_results),
    }

    report = _build_report(
        args,
        baseline,
        ablation_results,
        backbone_probe_results,
        ablation_results,
        attribution_results,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "ablations.json").write_text(json.dumps(ablation_results, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "backbone_probes.json").write_text(
        json.dumps(backbone_probe_results, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "attributions.json").write_text(json.dumps(attribution_results, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "ablations": ablation_results,
                "backbone_probes": backbone_probe_results,
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
                "backbone_probes.json",
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
                path_in_repo="blockwise_mnist_analysis_0707",
                commit_message="Upload MNIST no-training block analysis report",
            )

    if args.hub_only and args.push_to_hub:
        shutil.rmtree(output_dir, ignore_errors=True)

    logging.info(
        "MNIST block analysis complete: baseline_accuracy=%.4f best_ablation_drop=%.4f best_backbone_probe_accuracy=%.4f best_backbone_ablation_drop=%.4f",
        baseline["accuracy"],
        summary["best_ablation_drop"],
        summary["best_backbone_probe_accuracy"],
        summary["best_backbone_ablation_drop"],
    )


if __name__ == "__main__":
    main()
