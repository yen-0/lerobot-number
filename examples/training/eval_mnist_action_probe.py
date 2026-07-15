#!/usr/bin/env python

"""Probe whether frozen MNIST digit information reaches SmolVLA's action path."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_mnist_probe import (  # noqa: E402
    _append_jsonl,
    _build_context_batch,
    _evaluate_ridge_probe,
    _load_mnist_dataset,
    _make_repo,
    _pool_activation,
    _resolve_device,
    _resolve_output_path,
    _subset_dataset,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402


UPLOAD_FOLDER = "action_path_mnist_probe_0707"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher.repo_id", dest="teacher_repo_id", default="yen-0/smolvla-so101-digits-0707")
    parser.add_argument("--hub_repo_id", default="yen-0/smolvla-0707-action-path-mnist-probe")
    parser.add_argument("--output_dir", default=f"outputs/{UPLOAD_FOLDER}")
    parser.add_argument("--job_name", default=UPLOAD_FOLDER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher.device", dest="teacher_device", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist_root", default=".cache/mnist")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--probe_train_subset", type=int, default=5000)
    parser.add_argument("--probe_eval_subset", type=int, default=2000)
    parser.add_argument("--ridge_l2", type=float, default=1e-2)
    parser.add_argument("--action_timestep", type=float, default=0.5)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hub_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _action_tap_names(teacher: SmolVLAPolicy) -> list[str]:
    expert_layers = teacher.model.vlm_with_expert.num_expert_layers
    names = [
        "action_in_proj",
        "action_time_mlp_out",
        "suffix_input",
    ]
    for idx in range(expert_layers):
        names.append(f"expert_layer_{idx:02d}_attn_out")
        names.append(f"expert_layer_{idx:02d}_mlp_out")
    names.extend(["suffix_final_norm", "action_velocity"])
    return names


def _append_tap(taps: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    taps[name] = _pool_activation(value.detach()).cpu()


@torch.no_grad()
def _run_action_path(
    teacher: SmolVLAPolicy,
    images: torch.Tensor,
    teacher_device: torch.device,
    *,
    action_timestep: float,
) -> dict[str, torch.Tensor]:
    teacher.eval()
    batch = _build_context_batch(teacher, images, teacher_device)
    images_prepared, img_masks = teacher.prepare_images(batch)
    state = teacher.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    model = teacher.model
    vlm_with_expert = model.vlm_with_expert
    taps: dict[str, torch.Tensor] = {}
    handles = []

    def add_hook(name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            _append_tap(taps, name, output)

        handles.append(module.register_forward_hook(hook))

    add_hook("action_in_proj", model.action_in_proj)
    add_hook("action_time_mlp_out", model.action_time_mlp_out)
    for idx, layer in enumerate(vlm_with_expert.lm_expert.layers):
        add_hook(f"expert_layer_{idx:02d}_attn_out", layer.self_attn.o_proj)
        add_hook(f"expert_layer_{idx:02d}_mlp_out", layer.mlp)
    add_hook("action_velocity", model.action_out_proj)

    try:
        batch_size = state.shape[0]
        x_t = torch.zeros(
            batch_size,
            teacher.config.chunk_size,
            teacher.config.max_action_dim,
            dtype=torch.float32,
            device=teacher_device,
        )
        timestep = torch.full((batch_size,), float(action_timestep), dtype=torch.float32, device=teacher_device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images_prepared, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
        _append_tap(taps, "suffix_input", suffix_embs)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -teacher.config.chunk_size :]
        _append_tap(taps, "suffix_final_norm", suffix_out)
        _ = model.action_out_proj(suffix_out.to(dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    return taps


def _collect_action_feature_bank(
    teacher: SmolVLAPolicy,
    loader: DataLoader,
    teacher_device: torch.device,
    *,
    action_timestep: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    banks: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    for images, batch_labels in loader:
        taps = _run_action_path(teacher, images, teacher_device, action_timestep=action_timestep)
        for name, features in taps.items():
            banks.setdefault(name, []).append(features)
        labels.append(batch_labels.detach().cpu().long())
    return {name: torch.cat(chunks, dim=0) for name, chunks in banks.items()}, torch.cat(labels, dim=0)


def _build_report(
    args: argparse.Namespace,
    train_size: int,
    eval_size: int,
    probe_results: list[dict[str, Any]],
) -> str:
    ordered = sorted(probe_results, key=lambda row: row["eval_accuracy"], reverse=True)
    best = ordered[0]
    sanity = [row for row in probe_results if row["tap"] in {"action_in_proj", "action_time_mlp_out", "suffix_input"}]
    lines = [
        f"# MNIST Action-Path Probe for {args.teacher_repo_id}",
        "",
        "This run freezes the teacher and probes only the action-generation path.",
        "No SmolVLA weights are updated. Linear probes are external ridge classifiers trained on captured activations.",
        "",
        "## Setup",
        f"- Teacher repo: `{args.teacher_repo_id}`",
        f"- Probe repo: `{args.hub_repo_id}`",
        f"- Upload folder: `{UPLOAD_FOLDER}`",
        f"- Probe train subset: `{train_size}` examples",
        f"- Probe eval subset: `{eval_size}` examples",
        f"- Ridge L2: `{args.ridge_l2}`",
        f"- Controlled action input: zero action/noise tokens",
        f"- Controlled timestep: `{args.action_timestep}`",
        "",
        "## Linear Diagnostic Probes",
        "| Action-path tap | Train acc | Eval acc |",
        "| --- | --- | --- |",
    ]
    for row in ordered:
        lines.append(
            f"| `{row['tap']}` | {row['train_accuracy'] * 100:.2f}% | {row['eval_accuracy'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            f"The best action-path tap is `{best['tap']}` at {best['eval_accuracy'] * 100:.2f}% eval accuracy.",
            "If suffix input taps are near chance while expert/final taps are high, digit information is entering the action stream through context attention.",
            "If every action-path tap is near chance, digit information is readable in the VLM stream but not linearly present in the action stream under this controlled setup.",
            "If action velocity is high, the final denoising/action update itself carries digit-readable structure.",
            "",
            "## Sanity Taps",
        ]
    )
    if sanity:
        lines.append("| Tap | Eval acc | Interpretation |")
        lines.append("| --- | --- | --- |")
        for row in sanity:
            interpretation = "should be near chance because it sees only controlled action/time input"
            lines.append(f"| `{row['tap']}` | {row['eval_accuracy'] * 100:.2f}% | {interpretation} |")
    else:
        lines.append("No sanity taps were captured.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    teacher_device = _resolve_device(args.teacher_device or args.device)
    init_logging(console_level="INFO", file_level="DEBUG")
    logging.info("Starting MNIST action-path probe")
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
    logging.info("Resolved upload folder: %s", UPLOAD_FOLDER)

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
    train_dataset = _subset_dataset(_load_mnist_dataset(mnist_root, train=True), args.probe_train_subset, args.seed + 1)
    eval_dataset = _subset_dataset(_load_mnist_dataset(mnist_root, train=False), args.probe_eval_subset, args.seed + 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=teacher_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Teacher repo: %s", args.teacher_repo_id)
    logging.info("Teacher device: %s", teacher_device)
    logging.info("MNIST root: %s", mnist_root)
    logging.info("Probe train examples: %d", len(train_dataset))
    logging.info("Probe eval examples: %d", len(eval_dataset))

    train_features, train_labels = _collect_action_feature_bank(
        teacher, train_loader, teacher_device, action_timestep=args.action_timestep
    )
    eval_features, eval_labels = _collect_action_feature_bank(
        teacher, eval_loader, teacher_device, action_timestep=args.action_timestep
    )

    probe_results: list[dict[str, Any]] = []
    for tap_name in _action_tap_names(teacher):
        if tap_name not in train_features or tap_name not in eval_features:
            continue
        metrics = _evaluate_ridge_probe(
            train_features[tap_name],
            train_labels,
            eval_features[tap_name],
            eval_labels,
            ridge_l2=args.ridge_l2,
        )
        row = {"tap": tap_name, **metrics}
        probe_results.append(row)
        _append_jsonl(metrics_file, {"phase": "action_path_probe", **row})

    summary = {
        "best_tap": max(probe_results, key=lambda row: row["eval_accuracy"])["tap"],
        "best_eval_accuracy": max(row["eval_accuracy"] for row in probe_results),
        "num_taps": len(probe_results),
    }
    report = _build_report(args, len(train_dataset), len(eval_dataset), probe_results)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "action_path_probes.json").write_text(
        json.dumps(probe_results, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "action_path_probes": probe_results,
                "setup": {
                    "teacher_repo_id": args.teacher_repo_id,
                    "hub_repo_id": args.hub_repo_id,
                    "probe_train_subset": len(train_dataset),
                    "probe_eval_subset": len(eval_dataset),
                    "ridge_l2": args.ridge_l2,
                    "action_timestep": args.action_timestep,
                    "controlled_action_input": "zeros",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.push_to_hub:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            bundle_dir = Path(tmpdir)
            for name in ["report.md", "summary.json", "action_path_probes.json", "analysis.json", "metrics.jsonl"]:
                shutil.copy2(output_dir / name, bundle_dir / name)
            _make_repo(api, args.hub_repo_id)
            api.upload_folder(
                repo_id=args.hub_repo_id,
                repo_type="model",
                folder_path=bundle_dir,
                path_in_repo=UPLOAD_FOLDER,
                commit_message="Upload MNIST action-path probe report",
            )

    if args.hub_only and args.push_to_hub:
        shutil.rmtree(output_dir, ignore_errors=True)

    logging.info(
        "MNIST action-path probe complete: best_tap=%s best_eval_accuracy=%.4f",
        summary["best_tap"],
        summary["best_eval_accuracy"],
    )


if __name__ == "__main__":
    main()
