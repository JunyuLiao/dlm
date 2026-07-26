#!/usr/bin/env python3
"""Safe H100 microbenchmark for dense, baseline sparse, and structural BLASST."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst import (
    DiffusionLambdaSchedule,
    blasst_bidirectional_flash_attn_func,
    get_kernel_stats,
    reset_kernel_stats,
)


def measure(call, warmup: int, repeats: int):
    output = None
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    assert output is not None
    ordered = sorted(samples)
    return output, {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "samples_ms": samples,
    }


def errors(output: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    difference = (output.float() - dense.float()).flatten()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(difference.norm() / dense.float().flatten().norm().clamp_min(1e-20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch != 3:
        raise ValueError("default calibrated schedule benchmark expects three noise buckets")
    torch.manual_seed(20260714)
    shape = (args.batch, args.sequence_length, args.heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask_ratios = (0.15, 0.5, 0.9)
    schedule = DiffusionLambdaSchedule()
    thresholds = torch.tensor(
        [schedule.threshold(ratio) for ratio in mask_ratios], device="cuda", dtype=torch.float32
    )

    def dense_call():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)

    dense, dense_timing = measure(dense_call, args.warmup, args.repeats)

    variants = [
        ("sparse_baseline", 128, 128, 1, None, None),
    ]
    measurements = []
    baseline_ms = None
    for name, query_tile_rows, group_rows, fallback_threshold, permutation, order in variants:
        call = lambda: blasst_bidirectional_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=thresholds,
            collect_stats=False,
            num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
            query_permutation=permutation,
            kv_tile_order=order,
            query_tile_rows=query_tile_rows,
            skip_group_rows=group_rows,
            dense_fallback_threshold=fallback_threshold,
        )
        output, timing = measure(call, args.warmup, args.repeats)
        if baseline_ms is None:
            baseline_ms = timing["median_ms"]
        reset_kernel_stats(q.device)
        blasst_bidirectional_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=thresholds,
            collect_stats=True,
            num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
            query_permutation=permutation,
            kv_tile_order=order,
            query_tile_rows=query_tile_rows,
            skip_group_rows=group_rows,
            dense_fallback_threshold=fallback_threshold,
        )
        stats = get_kernel_stats(reset=True)
        measurements.append(
            {
                "name": name,
                "query_tile_rows": query_tile_rows,
                "skip_group_rows": group_rows,
                "dense_fallback_threshold": fallback_threshold,
                "timing": timing,
                "speedup_vs_dense_flash": dense_timing["median_ms"] / timing["median_ms"],
                "speedup_vs_sparse_baseline": baseline_ms / timing["median_ms"],
                "errors_vs_dense": errors(output, dense),
                "full_parent_tile_skip_rate": stats.sparsity_ratio,
                "microgroup_skip_rate": stats.microgroup_sparsity_ratio,
                "bmm2_flop_skip_rate": stats.bmm2_flop_skip_ratio,
                "v_load_skip_rate": stats.v_load_skip_ratio,
                "dense_fallbacks": stats.dense_fallbacks,
            }
        )
    report = {
        "device": torch.cuda.get_device_name(),
        "shape": list(shape),
        "num_warps": args.num_warps,
        "pipeline_stages": args.pipeline_stages,
        "input_note": "Synthetic model-shaped QKV; this is not an end-to-end LLaDA quality benchmark.",
        "dense_flash_timing": dense_timing,
        "measurements": measurements,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
