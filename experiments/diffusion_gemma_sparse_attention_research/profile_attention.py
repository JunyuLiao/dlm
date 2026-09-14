"""H100 microprofile of attention components and optimistic sparsity bounds.

This is not a deployment-kernel benchmark.  It isolates eager PyTorch CUDA
operations with DiffusionGemma-like shapes to measure the relative cost of
dense QK, FP32 softmax, dense PV, BLASST routing statistics, and Sol proxies.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "results/diffusion_gemma_sparse_attention_research/profile.json"


def _measure(operation: Callable[[], torch.Tensor], warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        if isinstance(result, torch.Tensor):
            _ = result.shape
        values.append(float(start.elapsed_time(end)))
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "repeats": repeats,
    }


def profile_shape(
    *,
    name: str,
    heads: int,
    query_length: int,
    kv_length: int,
    head_dim: int,
    warmup: int,
    repeats: int,
) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    query = torch.randn((1, heads, query_length, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, heads, kv_length, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    value = torch.randn((1, heads, kv_length, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    scale = head_dim**-0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    q_tiles = (query_length + 63) // 64
    kv_tiles = (kv_length + 63) // 64
    assert query_length % 64 == kv_length % 64 == 0

    def qk():
        return torch.matmul(query, key.transpose(-2, -1)) * scale

    def softmax():
        return torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    def pv():
        return torch.matmul(probabilities, value)

    def blasst_route():
        blocks = scores.float().reshape(1, heads, q_tiles, 64, kv_tiles, 64)
        maxima = blocks.amax(-1)
        running = torch.cummax(maxima, dim=-1).values
        previous = torch.nn.functional.pad(running[..., :-1], (1, 0), value=-torch.inf)
        votes = maxima - previous < 0.0
        return votes.all(-2)

    def sol_proxy():
        pooled_query = query.reshape(1, heads, q_tiles, 64, head_dim).mean(-2)
        pooled_key = key.reshape(1, heads, kv_tiles, 64, head_dim).mean(-2)
        proxy = torch.matmul(pooled_query, pooled_key.transpose(-2, -1)) * scale
        standardized = (proxy - proxy.mean(-1, keepdim=True)) / proxy.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return standardized >= 0.0

    metrics = {
        "qk": _measure(qk, warmup, repeats),
        "softmax": _measure(softmax, warmup, repeats),
        "pv": _measure(pv, warmup, repeats),
        "blasst_route": _measure(blasst_route, warmup, repeats),
        "sol_proxy_and_threshold": _measure(sol_proxy, warmup, repeats),
    }
    dense_sum = sum(metrics[name]["median_ms"] for name in ("qk", "softmax", "pv"))
    for component in ("qk", "softmax", "pv"):
        metrics[component]["fraction_of_isolated_dense_sum"] = metrics[component]["median_ms"] / dense_sum
    bounds = {}
    for target in (0.25, 0.50, 0.75, 0.90):
        qk_ms = metrics["qk"]["median_ms"]
        softmax_ms = metrics["softmax"]["median_ms"]
        pv_ms = metrics["pv"]["median_ms"]
        blasst_ms = metrics["blasst_route"]["median_ms"]
        sol_ms = metrics["sol_proxy_and_threshold"]["median_ms"]
        dense = qk_ms + softmax_ms + pv_ms
        blasst_ideal = qk_ms + blasst_ms + (1 - target) * (softmax_ms + pv_ms)
        preqk_ideal = sol_ms + (1 - target) * dense
        bounds[f"s{int(target*100)}"] = {
            "assumed_physical_sparsity": target,
            "dense_isolated_sum_ms": dense,
            "post_qk_blasst_optimistic_ms": blasst_ideal,
            "post_qk_blasst_optimistic_attention_speedup": dense / blasst_ideal,
            "pre_qk_sol_optimistic_ms": preqk_ideal,
            "pre_qk_sol_optimistic_attention_speedup": dense / preqk_ideal,
        }
    del scores, probabilities, query, key, value
    torch.cuda.empty_cache()
    return {
        "name": name,
        "shape": {"batch": 1, "heads": heads, "query_length": query_length, "kv_length": kv_length, "head_dim": head_dim},
        "metrics": metrics,
        "optimistic_bounds": bounds,
    }


def run(output: Path = OUTPUT, warmup: int = 5, repeats: int = 20) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "protocol": {
            "device": properties.name,
            "torch": torch.__version__,
            "dtype": "bfloat16 Q/K/V and QK/PV; FP32 softmax and BLASST comparisons",
            "tile_shape": [64, 64],
            "warmup": warmup,
            "repeats": repeats,
            "timing": "CUDA events, synchronization after each repeat, median primary",
            "scope": "isolated eager operations; excludes model projections, MLP/MoE, launch fusion, HBM effects of a real sparse kernel, and end-to-end generation",
            "interpretation": "optimistic attention-only model, not measured sparse-kernel speedup",
        },
        "shapes": [],
    }
    for shape in (
        dict(name="local_1k", heads=16, query_length=256, kv_length=1024, head_dim=256),
        dict(name="global_4k", heads=16, query_length=256, kv_length=4096, head_dim=512),
        dict(name="global_16k", heads=16, query_length=256, kv_length=16384, head_dim=512),
    ):
        payload["shapes"].append(profile_shape(**shape, warmup=warmup, repeats=repeats))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    result = run(args.output, args.warmup, args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
