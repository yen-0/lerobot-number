# Changes From Upstream LeRobot

This repository is derived from the public LeRobot project and now includes a few local changes
for the SmolVLA digit workflow and streaming video decode path.

## What Changed

### 1. Hub-backed video decoding works in PyAV

Upstream `decode_video_frames_pyav()` assumed `video_path` was a local filesystem path.
In this fork, the decoder now opens the path through `fsspec` and passes PyAV a readable
file object instead of a raw `hf://datasets/...` URI.

Effect:
- Streaming dataset videos stored on the Hugging Face Hub can be decoded by PyAV.
- `ProtocolNotFoundError` on `hf://` video URLs is avoided.

### 2. Read-only image buffers are copied before tensor conversion

Some dataset/image loaders expose non-writable NumPy buffers.
In the SmolVLA digit utilities and model helpers, those buffers are now copied before converting
them with PyTorch.

Effect:
- Removes PyTorch warnings about non-writable NumPy arrays.
- Avoids undefined behavior if downstream code mutates the tensor.

### 3. Regression coverage was added

New tests were added for:
- PyAV decoding through `fsspec` for `hf://` video URLs.
- Copying read-only image arrays in the SmolVLA digit bank helper.
- Copying read-only image arrays in the SmolVLA model reference-image prep path.

## Files Changed

- `src/lerobot/datasets/video_utils.py`
- `src/lerobot/policies/smolvla/digit_utils.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `tests/datasets/test_video_utils.py`
- `tests/policies/smolvla/test_smolvla_digits.py`

## Notes

This file is intentionally short and practical: it documents the behavioral delta from upstream
LeRobot so future maintainers can tell which changes were introduced locally.
