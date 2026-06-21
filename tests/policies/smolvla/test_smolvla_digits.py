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

import warnings

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla import modeling_smolvla as smolvla_mod
from lerobot.policies.smolvla.digit_utils import parse_digit_label, resolve_digit_label, sample_digit_references
from lerobot.policies.smolvla.processor_smolvla import SmolVLADigitLabelProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def test_digit_label_helpers():
    assert parse_digit_label("draw 7") == 7
    assert parse_digit_label("write zero") == 0
    assert parse_digit_label("write1") == 1
    assert parse_digit_label("write_1") == 1
    assert resolve_digit_label("please draw three") == 3
    assert resolve_digit_label("custom task", {"custom task": 4}) == 4

    bank = {
        0: torch.zeros(2, 3, 28, 28),
        1: torch.ones(2, 3, 28, 28),
    }
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    samples = sample_digit_references(bank, labels)
    assert samples.shape == (3, 3, 28, 28)
    assert samples[0].sum().item() == 0.0
    assert samples[1].sum().item() > 0.0


def test_build_mnist_reference_bank_copies_read_only_arrays():
    read_only = np.zeros((28, 28), dtype=np.uint8)
    read_only.setflags(write=False)

    bank = smolvla_mod.build_mnist_reference_bank(
        [{"label": 0, "image": read_only}],
        examples_per_digit=1,
    )

    assert bank[0].shape == (1, 3, 28, 28)
    assert bank[0].dtype == torch.float32
    assert bank[0].max().item() == 0.0


def test_prepare_digit_reference_images_copies_read_only_arrays():
    read_only = np.zeros((28, 28, 3), dtype=np.uint8)
    read_only.setflags(write=False)

    policy = smolvla_mod.SmolVLAPolicy.__new__(smolvla_mod.SmolVLAPolicy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        images = policy._prepare_digit_reference_images([read_only], device=torch.device("cpu"))

    assert not caught
    assert images.shape == (1, 3, 28, 28)
    assert images.dtype == torch.float32
    assert images.max().item() == 0.0


def test_digit_label_processor_uses_task_text():
    step = SmolVLADigitLabelProcessorStep()
    transition = {
        "complementary_data": {
            "task": ["write 1", "write1"],
        }
    }

    processed = step.complementary_data(transition["complementary_data"])

    assert torch.equal(processed["digit_label"], torch.tensor([1, 1], dtype=torch.long))


class _FakeVLMWithExpert:
    expert_hidden_size = 8

    def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache):
        prefix = inputs_embeds[0]
        return [prefix + 1.0, None], None


class _FakeFlowMatching:
    def __init__(self, config, rtc_processor=None):
        self.vlm_with_expert = _FakeVLMWithExpert()

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        return torch.ones(actions.shape[0], actions.shape[1], actions.shape[2], device=actions.device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        batch_size = state.shape[0]
        device = state.device
        prefix = torch.cat(
            [
                state[:, None, :8],
                torch.zeros(batch_size, 2, 8, device=device, dtype=state.dtype),
            ],
            dim=1,
        )
        pad_masks = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        att_masks = torch.zeros(1, 3, dtype=torch.bool, device=device)
        return prefix, pad_masks, att_masks


def test_smolvla_digit_loss_path(monkeypatch):
    monkeypatch.setattr(smolvla_mod, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(smolvla_mod, "VLAFlowMatching", _FakeFlowMatching)

    config = SmolVLAConfig(
        digit_classification_loss_weight=1.0,
        digit_alignment_loss_weight=0.5,
        resize_imgs_with_padding=None,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    config.device = "cpu"
    policy = smolvla_mod.SmolVLAPolicy(config)

    batch = {
        OBS_STATE: torch.randn(2, 8),
        f"{OBS_IMAGES}.front": torch.rand(2, 3, 28, 28),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 10, (2, 5)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
        ACTION: torch.randn(2, 4, 2),
        config.digit_label_key: torch.tensor([1, 2]),
        config.digit_reference_image_key: torch.rand(2, 3, 28, 28),
    }

    loss, loss_dict = policy.forward(batch)

    assert loss.ndim == 0
    assert "action_loss" in loss_dict
    assert "digit_classification_loss" in loss_dict
    assert "digit_alignment_loss" in loss_dict
    assert "digit_prediction_accuracy" in loss_dict
    assert "loss" in loss_dict
    assert torch.isfinite(loss)
