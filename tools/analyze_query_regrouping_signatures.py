#!/usr/bin/env python3
"""Previous-step per-head signature ablation on adjacent diffusion states."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.regrouping import build_head_group_permutations, evaluate_permutation, unpack_keep_mask


CONFIGURATIONS = (
    ("bitmap", "bitmap", {}),
    ("topk2", "topk", {"topk": 2}),
    ("topk4", "topk", {"topk": 4}),
    ("topk8", "topk", {"topk": 8}),
    ("simhash8", "simhash", {"signature_bits": 8}),
    ("simhash16", "simhash", {"signature_bits": 16}),
    ("simhash32", "simhash", {"signature_bits": 32}),
    ("bucket4", "bucket", {"signature_bits": 16, "bucket_bits": 4}),
    ("bucket6", "bucket", {"signature_bits": 16, "bucket_bits": 6}),
    ("bucket8", "bucket", {"signature_bits": 16, "bucket_bits": 8}),
    ("minhash2", "minhash", {"num_hashes": 2}),
    ("minhash4", "minhash", {"num_hashes": 4}),
    ("minhash8", "minhash", {"num_hashes": 8}),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    records = payload["records"]
    lookup = {
        (record["request_id"], int(record["layer_id"]), int(record["denoising_step"])): record
        for record in records
    }
    rows = []
    for record in records:
        previous_record = lookup.get((record["request_id"], int(record["layer_id"]), int(record["denoising_step"]) - 1))
        if previous_record is None:
            continue
        keep = unpack_keep_mask(record["keep_mask_packed"], int(record["num_kv_tiles"]))
        previous = unpack_keep_mask(previous_record["keep_mask_packed"], int(previous_record["num_kv_tiles"]))
        baseline = evaluate_permutation(keep, torch.arange(keep.shape[1]))
        for name, strategy, options in CONFIGURATIONS:
            start = time.perf_counter()
            perm = build_head_group_permutations(
                previous, strategy, heads_per_group=1, group_size=128, **options
            )
            planning_ms = (time.perf_counter() - start) * 1000
            metric = evaluate_permutation(keep, perm)
            rows.append({
                "signature": name,
                "layer": int(record["layer_id"]),
                "mask_ratio": float(record["remaining_mask_ratio"]),
                "improvement": metric.physical_tile_sparsity - baseline.physical_tile_sparsity,
                "required_tile_reduction": 1.0 - metric.required_physical_tiles / baseline.required_physical_tiles,
                "cpu_planning_ms": planning_ms,
            })
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["signature"]].append(row)
    summary = []
    for signature, values in grouped.items():
        summary.append({
            "signature": signature,
            "records": len(values),
            "mean_improvement": statistics.mean(row["improvement"] for row in values),
            "median_improvement": statistics.median(row["improvement"] for row in values),
            "mean_required_tile_reduction": statistics.mean(row["required_tile_reduction"] for row in values),
            "median_cpu_planning_ms": statistics.median(row["cpu_planning_ms"] for row in values),
            "by_mask_ratio": {
                str(ratio): statistics.mean(
                    row["improvement"] for row in values if row["mask_ratio"] == ratio
                ) for ratio in sorted({row["mask_ratio"] for row in values})
            },
        })
    summary.sort(key=lambda row: row["mean_improvement"], reverse=True)
    report = {"records": len(rows), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
