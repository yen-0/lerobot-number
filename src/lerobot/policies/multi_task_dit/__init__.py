#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

from .configuration_multi_task_dit import MultiTaskDiTConfig

__all__ = ["MultiTaskDiTConfig", "MultiTaskDiTPolicy", "make_multi_task_dit_pre_post_processors"]


def __getattr__(name: str):
    if name == "MultiTaskDiTPolicy":
        from .modeling_multi_task_dit import MultiTaskDiTPolicy

        return MultiTaskDiTPolicy
    if name == "make_multi_task_dit_pre_post_processors":
        from .processor_multi_task_dit import make_multi_task_dit_pre_post_processors

        return make_multi_task_dit_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
