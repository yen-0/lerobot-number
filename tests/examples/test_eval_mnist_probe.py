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


def test_teacher_logits_uses_digit_branch():
    mod = _load_module()

    class FakeTeacher:
        def __init__(self):
            self.config = type("Cfg", (), {"device": "cpu"})()
            self._prepare_called = False
            self.digit_reference_encoder = torch.nn.Flatten()
            self.digit_reference_projection = torch.nn.Identity()
            self.digit_context_head = torch.nn.Linear(28 * 28, 10)

        def _prepare_digit_reference_images(self, images, device):
            self._prepare_called = True
            return images.to(device)

    teacher = FakeTeacher()
    logits = mod._teacher_logits(teacher, torch.randn(2, 1, 28, 28))
    assert teacher._prepare_called is True
    assert logits.shape == (2, 10)


def test_evaluate_returns_expected_metrics_shape():
    mod = _load_module()

    class FakeTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"device": "cpu"})()
            self.digit_reference_encoder = torch.nn.Flatten()
            self.digit_reference_projection = torch.nn.Identity()
            self.digit_context_head = torch.nn.Linear(28 * 28, 10)
            self._prepare_called = 0

        def _prepare_digit_reference_images(self, images, device):
            self._prepare_called += 1
            return images.to(device)

    teacher = FakeTeacher()
    images = torch.randn(6, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    loader = [(images, labels)]

    metrics = mod._evaluate(teacher, loader, torch.device("cpu"))
    assert set(metrics) == {
        "loss",
        "accuracy",
        "mean_confidence",
        "examples",
        "per_class_accuracy",
        "confusion_matrix",
    }
    assert metrics["examples"] == 6
    assert len(metrics["confusion_matrix"]) == 10
    assert teacher._prepare_called == 1
