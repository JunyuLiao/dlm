#!/usr/bin/env python3
"""Bulk H100 diagnostic for fresh versus cached physical-tile work.

This is deliberately a framework-level microbenchmark, not an optimized cache
kernel.  It establishes whether loading and composing cached ``(m,l,u)`` has a
large intrinsic advantage before any candidate-specific kernel work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def elapsed_ms(function, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20261004)
    device = torch.device("cuda")
    rows, keys, dim = 128, 64, 128
    q = torch.randn(args.tiles, rows, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(args.tiles, keys, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(args.tiles, keys, dim, device=device, dtype=torch.bfloat16)
    cached_m = torch.randn(args.tiles, rows, device=device)
    cached_l = torch.rand(args.tiles, rows, device=device) * 64 + 1
    cached_u = torch.randn(args.tiles, rows, dim, device=device, dtype=torch.bfloat16)
    cached_u_scale = cached_u.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    cached_u_int8 = (
        (cached_u.float() / cached_u_scale).round().clamp(-127, 127).to(torch.int8)
    )
    running_m = torch.randn_like(cached_m)
    running_l = torch.rand_like(cached_l) * 256 + 1
    running_u = torch.randn_like(cached_u)
    scale = dim**-0.5

    def fresh(indices: torch.Tensor):
        q_work = q.index_select(0, indices)
        k_work = k.index_select(0, indices)
        v_work = v.index_select(0, indices)
        scores = torch.bmm(q_work, k_work.transpose(1, 2)) * scale
        m = scores.amax(-1)
        probabilities = torch.exp(scores - m[..., None])
        l = probabilities.sum(-1)
        u = torch.bmm(probabilities, v_work)
        return m, l, u

    def cached(indices: torch.Tensor):
        cm = cached_m.index_select(0, indices)
        cl = cached_l.index_select(0, indices)
        cu = cached_u.index_select(0, indices).float()
        rm = running_m.index_select(0, indices)
        rl = running_l.index_select(0, indices)
        ru = running_u.index_select(0, indices).float()
        merged = torch.maximum(cm, rm)
        cs = torch.exp(cm - merged)
        rs = torch.exp(rm - merged)
        return merged, cs * cl + rs * rl, cs[..., None] * cu + rs[..., None] * ru

    def cached_int8(indices: torch.Tensor):
        cm = cached_m.index_select(0, indices)
        cl = cached_l.index_select(0, indices)
        cu = cached_u_int8.index_select(0, indices).float()
        cu = cu * cached_u_scale.index_select(0, indices)
        rm = running_m.index_select(0, indices)
        rl = running_l.index_select(0, indices)
        ru = running_u.index_select(0, indices).float()
        merged = torch.maximum(cm, rm)
        cs = torch.exp(cm - merged)
        rs = torch.exp(rm - merged)
        return merged, cs * cl + rs * rl, cs[..., None] * cu + rs[..., None] * ru

    generator = torch.Generator(device=device).manual_seed(20261004)
    rows_out: list[dict[str, object]] = []
    for reuse_ratio in (0.0, 0.14, 0.25, 0.5, 0.75, 0.9, 1.0):
        reused = round(args.tiles * reuse_ratio)
        dirty = args.tiles - reused
        for pattern in ("contiguous", "random"):
            order = (
                torch.arange(args.tiles, device=device)
                if pattern == "contiguous"
                else torch.randperm(args.tiles, generator=generator, device=device)
            )
            cached_indices = order[:reused]
            dirty_indices = order[reused:]

            def mixed():
                result = []
                if dirty:
                    result.append(fresh(dirty_indices))
                if reused:
                    result.append(cached(cached_indices))
                return result

            latency = elapsed_ms(mixed, warmup=args.warmup, repeats=args.repeats)
            rows_out.append(
                {
                    "reuse_ratio": reuse_ratio,
                    "pattern": pattern,
                    "tiles": args.tiles,
                    "dirty_tiles": dirty,
                    "reused_tiles": reused,
                    "latency_ms": latency,
                    "nanoseconds_per_tile": latency * 1e6 / args.tiles,
                }
            )

    all_indices = torch.arange(args.tiles, device=device)
    fresh_ms = elapsed_ms(
        lambda: fresh(all_indices), warmup=args.warmup, repeats=args.repeats
    )
    cached_ms = elapsed_ms(
        lambda: cached(all_indices), warmup=args.warmup, repeats=args.repeats
    )
    cached_int8_ms = elapsed_ms(
        lambda: cached_int8(all_indices), warmup=args.warmup, repeats=args.repeats
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "tiles": args.tiles,
        "geometry": {"query_rows": rows, "kv_rows": keys, "head_dim": dim},
        "fresh_all_ms": fresh_ms,
        "cached_all_ms": cached_ms,
        "cached_vs_fresh_speedup": fresh_ms / cached_ms,
        "cached_int8_all_ms": cached_int8_ms,
        "cached_int8_vs_fresh_speedup": fresh_ms / cached_int8_ms,
        "caveat": (
            "Framework-level batched diagnostic; it includes index selection and separate "
            "operator launches and is not a fused persistent attention kernel."
        ),
        "mixed": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
