#!/usr/bin/env python3
"""Measure a conservative device-memory write bandwidth on the active GPU."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def measure(size_mib: int, warmups: int, iterations: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    elements = size_mib * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
    source = torch.empty(elements, dtype=torch.float32, device="cuda").normal_()
    destination = torch.empty_like(source)
    for _ in range(warmups):
        destination.copy_(source)
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        destination.copy_(source)
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    destination_bytes = destination.numel() * destination.element_size()
    median_ms = statistics.median(samples_ms)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "size_mib": size_mib,
        "iterations": iterations,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        # Count destination bytes only.  A copy also reads the same number of
        # source bytes, so this is deliberately conservative for a write-only
        # lower-bound calculation.
        "destination_only_bandwidth_gb_s": destination_bytes / (median_ms / 1000) / 1e9,
        "aggregate_read_write_bandwidth_gb_s": 2 * destination_bytes / (median_ms / 1000) / 1e9,
        "denominator_note": "destination-only counts written bytes; aggregate counts source reads plus destination writes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    result = measure(args.size_mib, args.warmups, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

