#!/usr/bin/env python3
"""Sampled union-greedy upper bound for expensive per-head grouping."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.regrouping import aggregate_heads, build_permutation, evaluate_permutation, unpack_keep_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--layers", default="0,15,31")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    layers = {int(value) for value in args.layers.split(",")}
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    rows = []
    for record in payload["records"]:
        if int(record["layer_id"]) not in layers:
            continue
        keep = unpack_keep_mask(record["keep_mask_packed"], int(record["num_kv_tiles"]))
        baseline = evaluate_permutation(keep, torch.arange(keep.shape[1]))
        for name, signal, head in (
            ("shared-rho50", aggregate_heads(keep, 0.5), None),
            ("head0", keep[0], 0),
        ):
            for strategy in ("bitmap", "oracle_greedy"):
                perm = build_permutation(signal, strategy, group_size=128)
                if head is None:
                    metric = evaluate_permutation(keep, perm)
                else:
                    perms = torch.arange(keep.shape[1])[None].expand(keep.shape[0], -1).clone()
                    perms[head] = perm
                    metric = evaluate_permutation(keep, perms)
                rows.append({
                    "layer": int(record["layer_id"]),
                    "step": int(record["denoising_step"]),
                    "mask_ratio": float(record["remaining_mask_ratio"]),
                    "sharing": name,
                    "strategy": strategy,
                    "improvement": metric.physical_tile_sparsity - baseline.physical_tile_sparsity,
                    "required_tile_reduction": 1.0 - metric.required_physical_tiles / baseline.required_physical_tiles,
                })
    summary = []
    for sharing in ("shared-rho50", "head0"):
        for strategy in ("bitmap", "oracle_greedy"):
            selected = [row for row in rows if row["sharing"] == sharing and row["strategy"] == strategy]
            summary.append({
                "sharing": sharing, "strategy": strategy, "records": len(selected),
                "mean_improvement": statistics.mean(row["improvement"] for row in selected),
                "mean_required_tile_reduction": statistics.mean(row["required_tile_reduction"] for row in selected),
            })
    report = {"sample_layers": sorted(layers), "summary": summary, "per_record": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
