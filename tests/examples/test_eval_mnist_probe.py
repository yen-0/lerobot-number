#!/usr/bin/env python

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "training" / "eval_mnist_probe.py"
    spec = importlib.util.spec_from_file_location("eval_mnist_probe", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ridge_probe_fits_simple_linear_problem():
    mod = _load_module()
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    probe = mod._fit_ridge_probe(features, labels, ridge_l2=1e-4)
    preds = mod._predict_ridge_probe(probe, features)
    assert preds.tolist() == labels.tolist()


def test_pool_activation_handles_spatial_and_vector_tensors():
    mod = _load_module()
    spatial = torch.randn(3, 4, 5, 5)
    vector = torch.randn(3, 8)
    assert mod._pool_activation(spatial).shape == (3, 4)
    assert mod._pool_activation(vector).shape == (3, 8)


def test_build_report_mentions_probe_sections():
    mod = _load_module()
    args = type(
        "Args",
        (),
        {
            "teacher_repo_id": "teacher/test",
            "hub_repo_id": "hub/test",
            "probe_train_subset": 5,
            "probe_eval_subset": 3,
            "ablation_subset": 7,
            "attribution_examples": 1,
            "ridge_l2": 0.01,
            "saliency_topk_fraction": 0.1,
        },
    )()
    baseline = {"examples": 10, "accuracy": 0.4, "loss": 1.2, "mean_confidence": 0.6}
    ablations = [{"target": "text_layer_00", "accuracy": 0.3, "loss": 1.4}]
    backbone_probes = [
        {"tap": "vision_encoder", "train_accuracy": 0.2, "eval_accuracy": 0.2},
        {"tap": "text_layer_00", "train_accuracy": 0.6, "eval_accuracy": 0.5},
    ]
    attributions = [
        {
            "index": 0,
            "label": 1,
            "predicted": 1,
            "confidence": 0.95,
            "concentration_top10pct": 0.5,
            "top_pixels": [],
            "artifacts": {"input": "assets/a.png", "saliency": "assets/b.png", "combined": "assets/c.png"},
        }
    ]
    report = mod._build_report(args, baseline, ablations, backbone_probes, ablations, attributions)
    assert "Frozen Baseline" in report
    assert "Backbone Analysis" in report
    assert "Causal Ablations" in report
    assert "Attribution Examples" in report
