#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for ``lerobot.datasets.video_utils`` decoding helpers."""

from fractions import Fraction

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av  # noqa: E402

from lerobot.datasets.video_utils import decode_video_frames_pyav  # noqa: E402


class _DummyFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _DummyOpenFile:
    def __init__(self, handle: _DummyFile):
        self.handle = handle

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.close()


class _DummyFrame:
    pts = 0

    def to_ndarray(self, format: str):  # noqa: A002
        assert format == "rgb24"
        return np.full((2, 2, 3), 255, dtype=np.uint8)


class _DummyStream:
    time_base = Fraction(1, 1)


class _DummyContainer:
    def __init__(self, video_file: _DummyFile, seen_handles: list[_DummyFile]):
        self.video_file = video_file
        self.seen_handles = seen_handles
        self.streams = type("Streams", (), {"video": [_DummyStream()]})()
        self.seek_calls: list[tuple[int, bool]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def seek(self, offset: int, backward: bool = False):
        self.seek_calls.append((offset, backward))

    def decode(self, stream):
        assert isinstance(self.video_file, _DummyFile)
        self.seen_handles.append(self.video_file)
        yield _DummyFrame()


def test_decode_video_frames_pyav_uses_fsspec_for_hf_urls(monkeypatch):
    seen_handles: list[_DummyFile] = []
    open_handle = _DummyFile()
    container_ref: dict[str, _DummyContainer] = {}

    def fake_fsspec_open(path, mode="rb"):
        assert path == "hf://datasets/user/repo/videos/cam.mp4"
        assert mode == "rb"
        return _DummyOpenFile(open_handle)

    def fake_av_open(video_file, *args, **kwargs):
        assert not isinstance(video_file, (str, bytes))
        container = _DummyContainer(video_file, seen_handles)
        container_ref["container"] = container
        return container

    monkeypatch.setattr("lerobot.datasets.video_utils.fsspec.open", fake_fsspec_open)
    monkeypatch.setattr("lerobot.datasets.video_utils.av.open", fake_av_open)

    frames = decode_video_frames_pyav(
        "hf://datasets/user/repo/videos/cam.mp4",
        timestamps=[0.0],
        tolerance_s=1.0,
    )

    assert isinstance(frames, torch.Tensor)
    assert frames.shape == (1, 3, 2, 2)
    assert torch.allclose(frames, torch.ones_like(frames))
    assert open_handle.closed is True
    assert seen_handles == [open_handle]
    assert container_ref["container"].seek_calls
