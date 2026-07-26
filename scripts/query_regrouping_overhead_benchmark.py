#!/usr/bin/env python3
"""H100 cost breakdown for the best offline per-head bitmap prototype."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.regrouping import unpack_keep_mask
from blasst.triton_bidirectional import DiffusionLambdaSchedule, blasst_bidirectional_flash_attn_func


def measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); call(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--mask-ratio", type=float, default=0.49)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    record = next(
        row for row in payload["records"]
        if int(row["layer_id"]) == args.layer
        and abs(float(row["remaining_mask_ratio"]) - args.mask_ratio) < 1e-6
    )
    previous = next(
        row for row in payload["records"]
        if row["request_id"] == record["request_id"]
        and int(row["layer_id"]) == args.layer
        and int(row["denoising_step"]) == int(record["denoising_step"]) - 1
    )
    keep = unpack_keep_mask(previous["keep_mask_packed"], int(previous["num_kv_tiles"])).cuda()
    heads, rows, tiles = keep.shape
    powers = torch.tensor(
        [-(1 << 63) if bit == 63 else 1 << bit for bit in range(tiles)],
        dtype=torch.int64, device="cuda",
    )
    signatures = torch.empty((heads, rows), dtype=torch.int64, device="cuda")
    sorted_signatures = torch.empty_like(signatures)
    permutation = torch.empty_like(signatures)

    def pack():
        torch.sum(keep.to(torch.int64) * powers, dim=-1, out=signatures)

    def sort():
        torch.sort(signatures, dim=-1, stable=True, out=(sorted_signatures, permutation))

    pack(); sort()
    q = torch.randn(heads, rows, 128, device="cuda", dtype=torch.bfloat16)
    grouped = torch.empty_like(q)
    restored = torch.empty_like(q)
    gather_index = permutation[..., None].expand_as(q)

    def gather():
        torch.gather(q, 1, gather_index, out=grouped)

    def scatter():
        restored.scatter_(1, gather_index, grouped)

    def full_plan_and_layout():
        pack(); sort()
        torch.gather(q, 1, permutation[..., None].expand_as(q), out=grouped)
        restored.scatter_(1, permutation[..., None].expand_as(q), grouped)

    q_model = q.permute(1, 0, 2).contiguous()[None]
    k_model, v_model = torch.randn_like(q_model), torch.randn_like(q_model)
    threshold = DiffusionLambdaSchedule().threshold(args.mask_ratio)

    def sparse_attention():
        blasst_bidirectional_flash_attn_func(
            q_model, k_model, v_model, blasst_lambda=threshold, collect_stats=False
        )

    timings = {
        "signature_pack": measure(pack, args.warmup, args.repeats),
        "per_head_sort": measure(sort, args.warmup, args.repeats),
        "gather_q": measure(gather, args.warmup, args.repeats),
        "scatter_output": measure(scatter, args.warmup, args.repeats),
        "combined": measure(full_plan_and_layout, args.warmup, args.repeats),
        "sparse_attention_kernel": measure(sparse_attention, args.warmup, args.repeats),
    }
    report = {
        "device": torch.cuda.get_device_name(), "layer": args.layer,
        "mask_ratio": args.mask_ratio, "heads": heads, "rows": rows, "tiles": tiles,
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
