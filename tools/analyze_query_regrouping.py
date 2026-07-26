#!/usr/bin/env python3
"""Offline physical-tile simulator for compact BLASST row-mask dumps."""

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
    aggregate_heads,
    build_permutation,
    evaluate_permutation,
    mean_jaccard,
    unpack_keep_mask,
)


def parse_ints(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def add_result(rows, record, name, signal_source, sharing, metrics, baseline):
    result = {
        "request_id": record["request_id"],
        "step": int(record["denoising_step"]),
        "layer": int(record["layer_id"]),
        "remaining_mask_ratio": float(record["remaining_mask_ratio"]),
        "sequence_length": int(record["sequence_length"]),
        "strategy": name,
        "signal_source": signal_source,
        "head_sharing": sharing,
        **metrics.to_dict(),
    }
    result["physical_sparsity_improvement"] = (
        metrics.physical_tile_sparsity - baseline.physical_tile_sparsity
    )
    result["required_tile_reduction"] = 1.0 - (
        metrics.required_physical_tiles / max(1, baseline.required_physical_tiles)
    )
    result["estimated_v_bytes_saved"] = baseline.estimated_v_bytes - metrics.estimated_v_bytes
    result["estimated_pv_flops_saved"] = baseline.estimated_pv_flops - metrics.estimated_pv_flops
    rows.append(result)


def analyze_record(record, previous, oracle_layers, oracle_heads):
    keep = unpack_keep_mask(record["keep_mask_packed"], int(record["num_kv_tiles"]))
    rows = keep.shape[1]
    tile_size = int(record["query_tile_size"])
    metric_options = {
        "query_tile_size": tile_size,
        "kv_tile_size": int(record["kv_tile_size"]),
    }
    identity = torch.arange(rows)
    baseline = evaluate_permutation(keep, identity, **metric_options)
    results = []
    add_result(results, record, "identity", "none", "layer", baseline, baseline)

    # Current-step strategies are counterfactual upper bounds.  The per-head
    # bitmap sort is cheap to compute offline and complements sampled greedy.
    per_head = torch.stack([build_permutation(head, "bitmap", group_size=tile_size) for head in keep])
    add_result(
        results, record, "bitmap", "current", "head",
        evaluate_permutation(keep, per_head, **metric_options), baseline,
    )
    current_signals = {
        "any": aggregate_heads(keep, None),
        "rho25": aggregate_heads(keep, 0.25),
        "rho50": aggregate_heads(keep, 0.50),
        "rho75": aggregate_heads(keep, 0.75),
    }
    for aggregation, signal in current_signals.items():
        for strategy, options in (
            ("bitmap", {}),
            ("topk2", {"strategy": "topk", "topk": 2}),
            ("topk4", {"strategy": "topk", "topk": 4}),
            ("topk8", {"strategy": "topk", "topk": 8}),
            ("simhash8", {"strategy": "simhash", "signature_bits": 8}),
            ("simhash16", {"strategy": "simhash", "signature_bits": 16}),
            ("simhash32", {"strategy": "simhash", "signature_bits": 32}),
            ("bucket4", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 4}),
            ("bucket6", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 6}),
            ("bucket8", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 8}),
        ):
            method = options.pop("strategy", strategy)
            perm = build_permutation(signal, method, group_size=tile_size, **options)
            add_result(
                results, record, f"{strategy}-{aggregation}", "current", "layer",
                evaluate_permutation(keep, perm, **metric_options), baseline,
            )
        state_perm = build_permutation(
            signal, "bucket", signature_bits=16, bucket_bits=8,
            state=record["mask_state"], group_size=tile_size,
        )
        add_result(
            results, record, f"bucket8-{aggregation}-state", "current", "layer",
            evaluate_permutation(keep, state_perm, **metric_options), baseline,
        )

    if int(record["layer_id"]) in oracle_layers:
        signal = current_signals["rho50"]
        perm = build_permutation(signal, "oracle_greedy", group_size=tile_size)
        add_result(
            results, record, "oracle-greedy-rho50", "current", "layer",
            evaluate_permutation(keep, perm, **metric_options), baseline,
        )
        if oracle_heads:
            permutations = torch.arange(rows)[None].expand(keep.shape[0], -1).clone()
            for head in oracle_heads:
                if head < keep.shape[0]:
                    permutations[head] = build_permutation(
                        keep[head], "oracle_greedy", group_size=tile_size
                    )
            add_result(
                results, record, "oracle-greedy-sampled-heads", "current", "head-sampled",
                evaluate_permutation(keep, permutations, **metric_options), baseline,
            )

    temporal = None
    if previous is not None:
        previous_keep = unpack_keep_mask(previous["keep_mask_packed"], int(previous["num_kv_tiles"]))
        temporal = mean_jaccard(previous_keep, keep)
        for aggregation, rho in (("any", None), ("rho25", 0.25), ("rho50", 0.50), ("rho75", 0.75)):
            signal = aggregate_heads(previous_keep, rho)
            for strategy, options in (
                ("bitmap", {}),
                ("topk4", {"strategy": "topk", "topk": 4}),
                ("simhash16", {"strategy": "simhash", "signature_bits": 16}),
                ("bucket4", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 4}),
                ("bucket6", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 6}),
                ("bucket8", {"strategy": "bucket", "signature_bits": 16, "bucket_bits": 8}),
            ):
                method = options.pop("strategy", strategy)
                perm = build_permutation(signal, method, group_size=tile_size, **options)
                add_result(
                    results, record, f"{strategy}-{aggregation}", "previous", "layer",
                    evaluate_permutation(keep, perm, **metric_options), baseline,
                )
                state_perm = build_permutation(
                    signal, method, group_size=tile_size,
                    state=record["mask_state"], **options,
                )
                add_result(
                    results, record, f"{strategy}-{aggregation}-state", "previous", "layer",
                    evaluate_permutation(keep, state_perm, **metric_options), baseline,
                )
    return results, temporal


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["signal_source"], row["head_sharing"])].append(row)
    summary = []
    for (strategy, source, sharing), values in grouped.items():
        gains = [item["physical_sparsity_improvement"] for item in values]
        reductions = [item["required_tile_reduction"] for item in values]
        summary.append({
            "strategy": strategy,
            "signal_source": source,
            "head_sharing": sharing,
            "records": len(values),
            "mean_baseline_physical_sparsity": statistics.mean(
                item["physical_tile_sparsity"] - item["physical_sparsity_improvement"] for item in values
            ),
            "mean_physical_sparsity": statistics.mean(item["physical_tile_sparsity"] for item in values),
            "mean_physical_sparsity_improvement": statistics.mean(gains),
            "median_physical_sparsity_improvement": statistics.median(gains),
            "p10_physical_sparsity_improvement": sorted(gains)[max(0, int(0.1 * len(gains)) - 1)],
            "mean_required_tile_reduction": statistics.mean(reductions),
            "fraction_records_ge_15pp": sum(value >= 0.15 for value in gains) / len(gains),
            "fraction_records_ge_20pct_work": sum(value >= 0.20 for value in reductions) / len(reductions),
        })
    return sorted(summary, key=lambda row: row["mean_physical_sparsity_improvement"], reverse=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "outputs" / "query_regrouping" / "analysis")
    parser.add_argument("--oracle-layers", default="0,15,31")
    parser.add_argument("--oracle-heads", default="0")
    args = parser.parse_args()
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    if payload.get("format") != "blasst-row-masks-v1":
        raise ValueError("unsupported row-mask dump format")
    records = payload["records"]
    by_key = {
        (record["request_id"], int(record["layer_id"]), int(record["denoising_step"])): record
        for record in records
    }
    rows, temporal_rows = [], []
    oracle_layers, oracle_heads = parse_ints(args.oracle_layers), parse_ints(args.oracle_heads)
    for record in records:
        previous = by_key.get((
            record["request_id"], int(record["layer_id"]), int(record["denoising_step"]) - 1
        ))
        analyzed, temporal = analyze_record(record, previous, oracle_layers, oracle_heads)
        rows.extend(analyzed)
        if temporal is not None:
            temporal_rows.append({
                "request_id": record["request_id"],
                "step": int(record["denoising_step"]),
                "layer": int(record["layer_id"]),
                "remaining_mask_ratio": float(record["remaining_mask_ratio"]),
                "mean_row_head_jaccard": temporal,
            })
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_record.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "temporal_stability.csv", temporal_rows)
    report = {
        "source_dump": str(args.dump),
        "records": len(records),
        "temporal_pairs": len(temporal_rows),
        "mean_temporal_jaccard": statistics.mean(
            row["mean_row_head_jaccard"] for row in temporal_rows
        ) if temporal_rows else None,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
