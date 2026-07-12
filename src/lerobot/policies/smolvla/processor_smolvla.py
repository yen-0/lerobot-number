#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    ObservationProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.configs import FeatureType

from .configuration_smolvla import SmolVLAConfig
from .digit_utils import resolve_digit_label


@dataclass
@ProcessorStepRegistry.register(name="smolvla_digit_label_processor")
class SmolVLADigitLabelProcessorStep(ComplementaryDataProcessorStep):
    """Derive a digit label from the episode task text and store it in complementary data."""

    digit_label_key: str = "digit_label"
    task_key: str = "task"
    digit_map: dict[str, int] | None = None

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        task = complementary_data.get(self.task_key)
        if task is None:
            raise ValueError(f"Missing '{self.task_key}' in complementary data.")

        if isinstance(task, (list, tuple)):
            labels = [resolve_digit_label(item, self.digit_map) for item in task]
        else:
            labels = [resolve_digit_label(task, self.digit_map)]

        complementary_data[self.digit_label_key] = torch.tensor(labels, dtype=torch.long)
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "digit_label_key": self.digit_label_key,
            "task_key": self.task_key,
            "digit_map": self.digit_map,
        }

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="smolvla_goal_image_processor")
class SmolVLAGoalImageProcessorStep(ObservationProcessorStep):
    """Inject a fixed goal-image tensor into the observation when it is not already present.

    During training the batch already carries ``observation.target_drawing``.
    During inference, ``eval.sh`` can provide ``TARGET_DRAWING_PATH`` and this step will
    load the PNG once and reuse it for every inference call.
    """

    image_key: str = "observation.target_drawing"
    image_path: str | None = None
    image_path_env_var: str = "TARGET_DRAWING_PATH"

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_image(path_str: str) -> torch.Tensor:
        image = Image.open(path_str).convert("RGB")
        image_np = np.array(image, dtype=np.float32, copy=True) / 255.0
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

    def _resolve_image_path(self) -> Path | None:
        if self.image_path:
            return Path(self.image_path).expanduser()

        env_value = os.environ.get(self.image_path_env_var)
        if env_value:
            return Path(env_value).expanduser()
        return None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.image_key in observation:
            return observation

        image_path = self._resolve_image_path()
        if image_path is None:
            return observation
        if not image_path.exists():
            raise FileNotFoundError(
                f"Goal image path '{image_path}' does not exist. Set {self.image_path_env_var} "
                "or provide 'image_path' in the processor config."
            )

        loaded = self._load_image(str(image_path))
        observation = observation.copy()
        observation[self.image_key] = loaded
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "image_key": self.image_key,
            "image_path": self.image_path,
            "image_path_env_var": self.image_path_env_var,
        }

    def transform_features(self, features):
        return features


def _ensure_hwc_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")

    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    saturation = np.zeros_like(maxc)
    value = maxc

    nonzero = delta > 0
    saturation[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]

    r_mask = nonzero & (maxc == r)
    g_mask = nonzero & (maxc == g)
    b_mask = nonzero & (maxc == b)

    hue[r_mask] = np.mod((g[r_mask] - b[r_mask]) / delta[r_mask], 6.0)
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    return np.stack((hue, saturation, value), axis=-1)


def _blue_mask_image(
    image: torch.Tensor | np.ndarray,
    *,
    hue_min: float,
    hue_max: float,
    saturation_min: float,
    value_min: float,
) -> np.ndarray:
    image_hwc = _ensure_hwc_uint8_image(image)
    hsv = _rgb_to_hsv(image_hwc)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    if hue_min <= hue_max:
        hue_mask = (hue >= hue_min) & (hue <= hue_max)
    else:
        hue_mask = (hue >= hue_min) | (hue <= hue_max)

    blue_mask = hue_mask & (saturation >= saturation_min) & (value >= value_min)
    filtered = np.full_like(image_hwc, 255, dtype=np.uint8)
    filtered[blue_mask] = image_hwc[blue_mask]
    return filtered


@dataclass
@ProcessorStepRegistry.register(name="smolvla_blue_world_filter_processor")
class SmolVLABlueWorldProcessorStep(ObservationProcessorStep):
    """Keep only blue pixels in live camera observations and whiten everything else."""

    hue_min: float = 0.55
    hue_max: float = 0.75
    saturation_min: float = 0.2
    value_min: float = 0.05
    target_image_key: str = "observation.target_drawing"
    image_keys: list[str] | None = None

    def _filter_single_image(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        filtered = _blue_mask_image(
            image,
            hue_min=self.hue_min,
            hue_max=self.hue_max,
            saturation_min=self.saturation_min,
            value_min=self.value_min,
        )
        return torch.from_numpy(filtered.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        filtered_observation = observation.copy()
        keys_to_filter = self.image_keys or []
        for key in keys_to_filter:
            if key == self.target_image_key or key not in observation:
                continue
            value = observation[key]
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                filtered_observation[key] = torch.stack([self._filter_single_image(sample) for sample in value], dim=0)
            elif isinstance(value, np.ndarray) and value.ndim == 4:
                filtered_observation[key] = torch.stack([self._filter_single_image(sample) for sample in value], dim=0)
            else:
                filtered_observation[key] = self._filter_single_image(value)
        return filtered_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "hue_min": self.hue_min,
            "hue_max": self.hue_max,
            "saturation_min": self.saturation_min,
            "value_min": self.value_min,
            "target_image_key": self.target_image_key,
            "image_keys": self.image_keys,
        }

    def transform_features(self, features):
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    digit_map: dict[str, int] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        SmolVLAGoalImageProcessorStep(),
    ]
    if config.blue_world_filter:
        image_keys = [
            key
            for key, feature in config.input_features.items()
            if feature.type == FeatureType.VISUAL and key != "observation.target_drawing"
        ]
        input_steps.append(
            SmolVLABlueWorldProcessorStep(
                hue_min=config.blue_world_hue_min,
                hue_max=config.blue_world_hue_max,
                saturation_min=config.blue_world_saturation_min,
                value_min=config.blue_world_value_min,
                image_keys=image_keys,
            )
        )
    input_steps.extend([
        AddBatchDimensionProcessorStep(),
        SmolVLADigitLabelProcessorStep(digit_map=digit_map, digit_label_key=config.digit_label_key),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ])
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
