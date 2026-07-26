#!/usr/bin/env python3
"""Focused ablation of spatial windows and head-sharing granularity."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.regrouping import (
    build_head_group_permutations,
    evaluate_permutation,
    mean_jaccard,
    unpack_keep_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    records = payload["records"]
    lookup = {
        (record["request_id"], int(record["layer_id"]), int(record["denoising_step"])): record
        for record in records
    }
    rows = []
    temporal = []
    for record in records:
        keep = unpack_keep_mask(record["keep_mask_packed"], int(record["num_kv_tiles"]))
        previous_record = lookup.get((
            record["request_id"], int(record["layer_id"]), int(record["denoising_step"]) - 1
        ))
        previous = None if previous_record is None else unpack_keep_mask(
            previous_record["keep_mask_packed"], int(previous_record["num_kv_tiles"])
        )
        if previous is None:
            continue
        identity = torch.arange(keep.shape[1])
        baseline = evaluate_permutation(keep, identity)
        temporal.append({
            "request_id": record["request_id"], "layer": int(record["layer_id"]),
            "step": int(record["denoising_step"]),
            "mask_ratio": float(record["remaining_mask_ratio"]),
            "jaccard": mean_jaccard(previous, keep),
        })
        for source, signal in (("current", keep), ("previous", previous)):
            configurations = (
                (1, 256), (1, 512), (1, 1024), (1, 4096),
                (2, 512), (2, 1024), (2, 4096),
                (4, 512), (4, 1024), (4, 4096),
                (8, 1024), (8, 4096), (32, 1024), (32, 4096),
            )
            for heads_per_group, window in configurations:
                    for strategy, options in (("bitmap", {}),):
                        method = options.pop("strategy", strategy)
                        perm = build_head_group_permutations(
                            signal, method, heads_per_group=heads_per_group,
                            rho=0.5, group_size=128, window_size=window, **options,
                        )
                        metric = evaluate_permutation(keep, perm)
                        rows.append({
                            "request_id": record["request_id"],
                            "layer": int(record["layer_id"]),
                            "step": int(record["denoising_step"]),
                            "mask_ratio": float(record["remaining_mask_ratio"]),
                            "source": source,
                            "strategy": strategy,
                            "heads_per_group": heads_per_group,
                            "window_size": window,
                            "baseline_physical_sparsity": baseline.physical_tile_sparsity,
                            "physical_tile_sparsity": metric.physical_tile_sparsity,
                            "improvement": metric.physical_tile_sparsity - baseline.physical_tile_sparsity,
                            "required_tile_reduction": 1.0 - metric.required_physical_tiles / baseline.required_physical_tiles,
                        })
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["strategy"], row["heads_per_group"], row["window_size"])].append(row)
    summary = []
    for key, values in grouped.items():
        gains = [row["improvement"] for row in values]
        work = [row["required_tile_reduction"] for row in values]
        summary.append({
            "source": key[0], "strategy": key[1],
            "heads_per_group": key[2], "window_size": key[3],
            "records": len(values),
            "mean_improvement": statistics.mean(gains),
            "median_improvement": statistics.median(gains),
            "mean_required_tile_reduction": statistics.mean(work),
            "fraction_ge_15pp": sum(value >= 0.15 for value in gains) / len(gains),
            "fraction_ge_20pct_work": sum(value >= 0.20 for value in work) / len(work),
        })
    summary.sort(key=lambda row: row["mean_improvement"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_record.csv", rows), ("summary.csv", summary), ("temporal.csv", temporal)):
        if values:
            with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader(); writer.writerows(values)
    report = {
        "records": len(records),
        "mean_temporal_jaccard": statistics.mean(row["jaccard"] for row in temporal) if temporal else None,
        "top_configurations": summary[:20],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
