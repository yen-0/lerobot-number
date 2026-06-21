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

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DIGIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_DIGIT_PATTERNS = (
    re.compile(r"\b(?:digit|number|draw|write)[\s:_-]*([0-9])\b", re.IGNORECASE),
    re.compile(r"\b([0-9])\b"),
    re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE),
)


def parse_digit_label(text: str | None) -> int | None:
    """Best-effort extraction of a digit label from free-form task text."""

    if not text:
        return None
    for pattern in _DIGIT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        token = match.group(1).lower()
        if token.isdigit():
            digit = int(token)
            return digit if 0 <= digit <= 9 else None
        return _DIGIT_WORDS.get(token)
    return None


def load_digit_map(path: str | Path | None) -> dict[str, int]:
    """Load an explicit task-to-digit mapping from JSON."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        digit = int(value)
        if digit < 0 or digit > 9:
            raise ValueError(f"Digit mapping for '{key}' must be in [0, 9], got {value!r}")
        mapping[str(key)] = digit
    return mapping


def resolve_digit_label(task: Any, digit_map: dict[str, int] | None = None) -> int:
    """Resolve a per-episode digit label from the task field or an override map."""

    if digit_map is None:
        digit_map = {}

    if isinstance(task, (list, tuple)):
        if len(task) == 0:
            raise ValueError("Empty task list cannot be mapped to a digit.")
        task = task[0]

    if isinstance(task, (int, np.integer)):
        digit = int(task)
        if 0 <= digit <= 9:
            return digit
        raise ValueError(f"Digit label must be in [0, 9], got {task!r}")

    if not isinstance(task, str):
        raise TypeError(f"Expected task text or digit label, got {type(task)}")

    if task in digit_map:
        return digit_map[task]

    digit = parse_digit_label(task)
    if digit is not None:
        return digit

    lowered = task.strip().lower()
    if lowered in digit_map:
        return digit_map[lowered]

    raise ValueError(f"Could not infer a digit label from task '{task}'. Provide a digit map.")


def build_mnist_reference_bank(
    mnist_dataset: list[dict[str, Any]],
    examples_per_digit: int = 64,
    seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Build a small digit bank with tensors shaped like reference images."""

    rng = random.Random(seed)
    grouped: dict[int, list[torch.Tensor]] = {digit: [] for digit in range(10)}

    shuffled = list(mnist_dataset)
    rng.shuffle(shuffled)

    for row in shuffled:
        digit = int(row["label"])
        if digit not in grouped:
            continue
        if len(grouped[digit]) >= examples_per_digit:
            continue

        image = row["image"]
        if hasattr(image, "convert"):
            image = np.asarray(image)
        if isinstance(image, np.ndarray) and not image.flags.writeable:
            # Some datasets expose read-only image buffers; copy before tensor conversion.
            image = np.array(image, copy=True)
        image_tensor = torch.as_tensor(image, dtype=torch.float32)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(2, 0, 1)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0
        grouped[digit].append(image_tensor)

    bank: dict[int, torch.Tensor] = {}
    for digit, samples in grouped.items():
        if not samples:
            raise ValueError(f"MNIST bank is missing digit {digit}.")
        bank[digit] = torch.stack(samples, dim=0)
    return bank


def sample_digit_references(
    bank: dict[int, torch.Tensor],
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one MNIST reference image per digit label."""

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {tuple(labels.shape)}")

    references = []
    for label in labels.tolist():
        digit_bank = bank[int(label)]
        index = torch.randint(0, digit_bank.shape[0], (1,), generator=generator).item()
        references.append(digit_bank[index])
    return torch.stack(references, dim=0)
