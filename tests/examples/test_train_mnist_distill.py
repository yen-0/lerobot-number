#!/usr/bin/env python

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "training" / "train_mnist_distill.py"
    spec = importlib.util.spec_from_file_location("train_mnist_distill", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_student_forward_shapes_logits():
    mod = _load_module()
    student = mod.MnistStudent(hidden_dim=32)
    logits = student(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


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


def test_distillation_loss_returns_scalar_and_metrics():
    mod = _load_module()
    student_logits = torch.randn(3, 10, requires_grad=True)
    teacher_logits = torch.randn(3, 10)
    labels = torch.tensor([1, 2, 3])
    loss, metrics = mod._distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        temperature=2.0,
        distill_weight=0.5,
    )
    assert loss.ndim == 0
    assert set(metrics) == {"ce_loss", "kl_loss", "total_loss"}
