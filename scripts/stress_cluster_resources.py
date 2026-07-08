#!/usr/bin/env python
"""Stress-test cluster disk, CPU RAM, and optional GPU memory.

This script is intentionally bounded by explicit limits. It reports free space,
write/read throughput, maximum successfully written bytes for the configured
test, and memory allocation progress without trying to fill the entire node by
default.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


MiB = 1024 * 1024
GiB = 1024 * MiB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/resource_stress")
    parser.add_argument("--disk-test-gb", type=float, default=20.0)
    parser.add_argument("--disk-chunk-mb", type=int, default=64)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--progress-every-gb", type=float, default=1.0)
    parser.add_argument("--keep-file", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--read-passes", type=int, default=1)
    parser.add_argument("--small-files", type=int, default=1000)
    parser.add_argument("--small-file-kb", type=int, default=64)
    parser.add_argument("--memory-test-gb", type=float, default=32.0)
    parser.add_argument("--memory-chunk-mb", type=int, default=512)
    parser.add_argument("--memory-hold-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-memory-test-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-chunk-mb", type=int, default=512)
    parser.add_argument("--jsonl", default=None)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str, **fields) -> None:
    payload = {"time": now(), "message": message, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def disk_usage(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": usage.total / GiB,
        "used_gb": usage.used / GiB,
        "free_gb": usage.free / GiB,
    }


def process_memory() -> dict[str, float]:
    fields: dict[str, float] = {}
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            fields["max_rss_gb"] = rss_kb / GiB
        else:
            fields["max_rss_gb"] = rss_kb * 1024 / GiB
    except Exception:
        pass

    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts and parts[0].isdigit():
                    fields[key.lower() + "_gb"] = int(parts[0]) * 1024 / GiB
    return fields


def command_context(command: list[str]) -> dict[str, str | int]:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
        return {
            "command": " ".join(command),
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-4000:],
        }
    except Exception as exc:
        return {"command": " ".join(command), "returncode": -1, "stderr": repr(exc), "stdout": ""}


def write_jsonl(path: Path | None, records: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def test_large_file(args: argparse.Namespace, output_dir: Path) -> list[dict]:
    records: list[dict] = []
    test_file = output_dir / "large_write_test.bin"
    chunk_size = args.disk_chunk_mb * MiB
    target_bytes = int(args.disk_test_gb * GiB)
    min_free_bytes = int(args.min_free_gb * GiB)
    chunk = bytes(chunk_size)
    written = 0
    progress_every_bytes = max(int(args.progress_every_gb * GiB), chunk_size)
    next_progress = progress_every_bytes

    usage_before = disk_usage(output_dir)
    log("disk large write start", path=str(test_file), target_gb=args.disk_test_gb, **usage_before)
    start = time.monotonic()
    error = None

    try:
        with test_file.open("wb", buffering=0) as file:
            while written < target_bytes:
                free_bytes = shutil.disk_usage(output_dir).free
                if free_bytes - chunk_size < min_free_bytes:
                    error = f"stopped before min_free_gb={args.min_free_gb}"
                    break
                to_write = min(chunk_size, target_bytes - written)
                file.write(chunk[:to_write])
                written += to_write
                if written >= next_progress:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    log(
                        "disk large write progress",
                        written_gb=written / GiB,
                        mb_s=written / MiB / elapsed,
                        **process_memory(),
                        **disk_usage(output_dir),
                    )
                    next_progress += progress_every_bytes
            if args.fsync:
                file.flush()
                os.fsync(file.fileno())
    except OSError as exc:
        error = repr(exc)

    elapsed = max(time.monotonic() - start, 1e-9)
    record = {
        "test": "large_write",
        "path": str(test_file),
        "written_gb": written / GiB,
        "elapsed_s": elapsed,
        "mb_s": written / MiB / elapsed,
        "error": error,
        **process_memory(),
        **disk_usage(output_dir),
    }
    records.append(record)
    log("disk large write end", **record)

    if written > 0 and test_file.exists():
        for pass_idx in range(args.read_passes):
            read_bytes = 0
            read_start = time.monotonic()
            try:
                with test_file.open("rb", buffering=0) as file:
                    while True:
                        data = file.read(chunk_size)
                        if not data:
                            break
                        read_bytes += len(data)
                read_error = None
            except OSError as exc:
                read_error = repr(exc)
            read_elapsed = max(time.monotonic() - read_start, 1e-9)
            record = {
                "test": "large_read",
                "pass": pass_idx + 1,
                "read_gb": read_bytes / GiB,
                "elapsed_s": read_elapsed,
                "mb_s": read_bytes / MiB / read_elapsed,
                "error": read_error,
            }
            records.append(record)
            log("disk large read end", **record)

    if test_file.exists() and not args.keep_file:
        test_file.unlink()
        log("disk large test file removed", path=str(test_file), **disk_usage(output_dir))

    return records


def test_small_files(args: argparse.Namespace, output_dir: Path) -> list[dict]:
    records: list[dict] = []
    small_dir = output_dir / "small_files"
    if small_dir.exists():
        shutil.rmtree(small_dir)
    small_dir.mkdir(parents=True)

    payload = bytes(args.small_file_kb * 1024)
    log("small file write start", directory=str(small_dir), files=args.small_files, kb=args.small_file_kb)
    start = time.monotonic()
    written = 0
    error = None
    try:
        for idx in range(args.small_files):
            with (small_dir / f"{idx:08d}.bin").open("wb") as file:
                file.write(payload)
            written += 1
    except OSError as exc:
        error = repr(exc)
    elapsed = max(time.monotonic() - start, 1e-9)
    record = {
        "test": "small_file_write",
        "files_written": written,
        "elapsed_s": elapsed,
        "files_s": written / elapsed,
        "mb_s": written * len(payload) / MiB / elapsed,
        "error": error,
        **process_memory(),
        **disk_usage(output_dir),
    }
    records.append(record)
    log("small file write end", **record)

    files = list(small_dir.glob("*.bin"))
    random.shuffle(files)
    start = time.monotonic()
    read_bytes = 0
    error = None
    try:
        for file_path in files:
            read_bytes += len(file_path.read_bytes())
    except OSError as exc:
        error = repr(exc)
    elapsed = max(time.monotonic() - start, 1e-9)
    record = {
        "test": "small_file_random_read",
        "files_read": len(files),
        "elapsed_s": elapsed,
        "files_s": len(files) / elapsed,
        "mb_s": read_bytes / MiB / elapsed,
        "error": error,
    }
    records.append(record)
    log("small file random read end", **record)

    if not args.keep_file:
        shutil.rmtree(small_dir)
        log("small file test directory removed", directory=str(small_dir), **disk_usage(output_dir))

    return records


def test_cpu_memory(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    target_bytes = int(args.memory_test_gb * GiB)
    chunk_bytes = args.memory_chunk_mb * MiB
    allocated: list[bytearray] = []
    allocated_bytes = 0
    error = None
    log("cpu memory test start", target_gb=args.memory_test_gb, chunk_mb=args.memory_chunk_mb)
    start = time.monotonic()
    try:
        while allocated_bytes < target_bytes:
            size = min(chunk_bytes, target_bytes - allocated_bytes)
            block = bytearray(size)
            for offset in range(0, size, 4096):
                block[offset] = 1
            allocated.append(block)
            allocated_bytes += size
            log("cpu memory allocated", allocated_gb=allocated_bytes / GiB)
    except MemoryError as exc:
        error = repr(exc)
    elapsed = max(time.monotonic() - start, 1e-9)
    record = {
        "test": "cpu_memory_allocate",
        "allocated_gb": allocated_bytes / GiB,
        "elapsed_s": elapsed,
        "gb_s": allocated_bytes / GiB / elapsed,
        "error": error,
        **process_memory(),
    }
    records.append(record)
    log("cpu memory allocation end", **record)

    if args.memory_hold_seconds > 0 and allocated:
        log("cpu memory hold start", seconds=args.memory_hold_seconds, allocated_gb=allocated_bytes / GiB)
        time.sleep(args.memory_hold_seconds)
        log("cpu memory hold end", allocated_gb=allocated_bytes / GiB)
    allocated.clear()
    return records


def test_gpu_memory(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    if args.gpu_memory_test_gb <= 0:
        return records
    try:
        import torch
    except Exception as exc:
        record = {"test": "gpu_memory_allocate", "allocated_gb": 0.0, "error": repr(exc)}
        records.append(record)
        log("gpu memory unavailable", **record)
        return records
    if not torch.cuda.is_available():
        record = {"test": "gpu_memory_allocate", "allocated_gb": 0.0, "error": "cuda unavailable"}
        records.append(record)
        log("gpu memory unavailable", **record)
        return records

    target_bytes = int(args.gpu_memory_test_gb * GiB)
    chunk_bytes = args.gpu_memory_chunk_mb * MiB
    tensors = []
    allocated_bytes = 0
    error = None
    log(
        "gpu memory test start",
        device=torch.cuda.get_device_name(0),
        target_gb=args.gpu_memory_test_gb,
        chunk_mb=args.gpu_memory_chunk_mb,
    )
    start = time.monotonic()
    try:
        while allocated_bytes < target_bytes:
            size = min(chunk_bytes, target_bytes - allocated_bytes)
            tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
            tensor.fill_(1)
            tensors.append(tensor)
            allocated_bytes += size
            torch.cuda.synchronize()
            log(
                "gpu memory allocated",
                allocated_gb=allocated_bytes / GiB,
                cuda_allocated_gb=torch.cuda.memory_allocated() / GiB,
                cuda_reserved_gb=torch.cuda.memory_reserved() / GiB,
            )
    except RuntimeError as exc:
        error = repr(exc)
    elapsed = max(time.monotonic() - start, 1e-9)
    record = {
        "test": "gpu_memory_allocate",
        "allocated_gb": allocated_bytes / GiB,
        "elapsed_s": elapsed,
        "gb_s": allocated_bytes / GiB / elapsed,
        "error": error,
    }
    records.append(record)
    log("gpu memory allocation end", **record)
    tensors.clear()
    torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl).expanduser().resolve() if args.jsonl else output_dir / "stress_results.jsonl"

    header = {
        "test": "header",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir),
        "env": {
            key: os.environ.get(key)
            for key in (
                "PBS_JOBID",
                "PBS_O_WORKDIR",
                "TMPDIR",
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_LEROBOT_HOME",
                "HF_DATASETS_CACHE",
            )
        },
        **disk_usage(output_dir),
    }
    log("stress test start", **header)
    context_records = [
        {"test": "command_context", **command_context(["df", "-h", str(output_dir)])},
        {"test": "command_context", **command_context(["df", "-i", str(output_dir)])},
        {"test": "command_context", **command_context(["bash", "-lc", f"command -v lfs >/dev/null && lfs quota -h -u $(whoami) {output_dir} || true"])},
        {"test": "command_context", **command_context(["bash", "-lc", f"command -v lfs >/dev/null && lfs quota -h -g $(id -gn) {output_dir} || true"])},
        {"test": "command_context", **command_context(["bash", "-lc", "quota -s 2>/dev/null || true"])},
        {"test": "command_context", **command_context(["mount"])},
        {"test": "command_context", **command_context(["bash", "-lc", "ulimit -a"])},
    ]
    for record in context_records:
        log("system context", **record)
    write_jsonl(jsonl_path, [header, *context_records])

    records: list[dict] = []
    records.extend(test_large_file(args, output_dir))
    records.extend(test_small_files(args, output_dir))
    records.extend(test_cpu_memory(args))
    records.extend(test_gpu_memory(args))
    write_jsonl(jsonl_path, records)

    summary = {"test": "summary", "records": len(records), "jsonl": str(jsonl_path), **disk_usage(output_dir)}
    log("stress test complete", **summary)
    write_jsonl(jsonl_path, [summary])


if __name__ == "__main__":
    main()
