#!/usr/bin/env python3
"""Phase-A feasibility analysis for Hopper cross-tile exception packing."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.exception_packing import (
    active_counts,
    exception_tiles_per_row_histogram,
    grouped_exception_totals,
    summarize_packability,
)
from blasst.regrouping import unpack_keep_mask


TAUS = (1, 2, 4, 8, 16, 32)
PACK_SIZES = (64, 128)
GROUPINGS = ("q-head", "kv-head")

ADDITIVE_FIELDS = (
    "candidate_tiles", "retained_tiles", "all_tiles", "exception_rows",
    "grouping_keys", "complete_packs", "partial_packs", "total_packs",
    "full_pack_rows", "partial_pack_rows", "padded_rows", "original_v_loads",
    "packed_v_loads", "v_bytes_before", "v_bytes_after", "extra_qk_flops",
    "baseline_qk_flops", "candidate_main_pv_flops", "packed_pv_flops",
    "pv_flops_avoided", "q_read_bytes", "tile_record_bytes",
    "expanded_descriptor_bytes", "count_offset_bytes", "pack_descriptor_bytes",
    "total_metadata_bytes",
)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def aggregate_configuration(rows, metadata_limit_bytes):
    sums = {field: sum(int(row[field]) for row in rows) for field in ADDITIVE_FIELDS}
    pack_m = int(rows[0]["pack_m"])
    exception_rows = sums["exception_rows"]
    total_packs = sums["total_packs"]
    original_loads, packed_loads = sums["original_v_loads"], sums["packed_v_loads"]
    avoided = sums["pv_flops_avoided"]
    max_metadata = max(int(row["total_metadata_bytes"]) for row in rows)
    result = {
        "tau": int(rows[0]["tau"]),
        "pack_m": pack_m,
        "grouping": rows[0]["grouping"],
        "invocations": len(rows),
        **sums,
        "zero_tile_fraction": 1.0 - ratio(sums["retained_tiles"], sums["all_tiles"]),
        "candidate_fraction_of_retained": ratio(sums["candidate_tiles"], sums["retained_tiles"]),
        "weighted_utilization": ratio(exception_rows, total_packs * pack_m),
        "complete_only_utilization": 1.0 if sums["complete_packs"] else 0.0,
        "complete_pack_row_fraction": ratio(sums["full_pack_rows"], exception_rows),
        "partial_pack_row_fraction": ratio(sums["partial_pack_rows"], exception_rows),
        "v_load_reduction": 1.0 - ratio(packed_loads, original_loads),
        "v_load_factor": ratio(original_loads, packed_loads),
        "average_rows_per_original_v_load": ratio(exception_rows, original_loads),
        "average_rows_per_packed_v_load": ratio(exception_rows, packed_loads),
        "extra_qk_fraction_of_baseline": ratio(sums["extra_qk_flops"], sums["baseline_qk_flops"]),
        "extra_qk_fraction_of_pv_avoided": ratio(sums["extra_qk_flops"], avoided) if avoided > 0 else None,
        "mean_metadata_bytes_per_invocation": statistics.mean(int(row["total_metadata_bytes"]) for row in rows),
        "max_metadata_bytes_per_invocation": max_metadata,
        "baseline_candidate_rows_processed": sums["candidate_tiles"] * 128,
        "baseline_nonvoter_rows_processed": sums["candidate_tiles"] * 128 - exception_rows,
        "voter_fraction_of_baseline_candidate_rows": ratio(exception_rows, sums["candidate_tiles"] * 128),
    }
    exact_rows = sums["candidate_tiles"] * 128
    exact_packs = exact_rows // pack_m
    result.update({
        "semantics_preserving_deferred_rows": exact_rows,
        "semantics_preserving_total_packs": exact_packs,
        "semantics_preserving_v_load_factor": ratio(sums["candidate_tiles"], exact_packs),
        "semantics_preserving_pv_flops_avoided": 0,
        "semantics_preserving_extra_qk_flops": sums["candidate_main_pv_flops"],
        "semantics_preserving_extra_qk_fraction_of_baseline": ratio(
            sums["candidate_main_pv_flops"], sums["baseline_qk_flops"]
        ),
    })
    checks = {
        "utilization_ge_70pct": result["weighted_utilization"] >= 0.70,
        "candidate_work_ge_15pct": result["candidate_fraction_of_retained"] >= 0.15,
        "v_load_reduction_ge_2x": result["v_load_factor"] >= 2.0,
        "extra_qk_le_20pct": result["extra_qk_fraction_of_baseline"] <= 0.20,
        "metadata_manageable": max_metadata <= metadata_limit_bytes,
    }
    result["gate_checks"] = checks
    result["phase_a_numeric_pass"] = all(checks.values())
    # In the existing kernel a retained physical tile updates m/l and P@V for
    # every valid row. Packing only positive voters omits non-voter output
    # contributions and therefore cannot be a drop-in replacement.
    result["same_approximation_semantics"] = result["baseline_nonvoter_rows_processed"] == 0
    result["hardware_phase_authorized"] = (
        result["phase_a_numeric_pass"] and result["same_approximation_semantics"]
    )
    return result


def write_csv(path, rows):
    if not rows:
        return
    flattened = []
    for row in rows:
        item = dict(row)
        checks = item.pop("gate_checks", None)
        if checks:
            item.update({f"gate_{name}": value for name, value in checks.items()})
        flattened.append(item)
    fields = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(flattened)


def add_marginal(counter, dimension, value, counts):
    unique, occurrences = torch.unique(counts, return_counts=True)
    for active, total in zip(unique.tolist(), occurrences.tolist()):
        counter[(dimension, str(value), int(active))] += int(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--metadata-limit-mib", type=float, default=512.0)
    args = parser.parse_args()
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    if payload.get("format") != "blasst-row-masks-v1":
        raise ValueError("unsupported row-mask dump format")
    records = payload["records"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    per_invocation = []
    config_rows = defaultdict(list)
    leftover_hist = Counter()
    row_hist = Counter()
    marginals = Counter()
    joint_hist = Counter()

    for record in records:
        keep = unpack_keep_mask(record["keep_mask_packed"], int(record["num_kv_tiles"]))
        heads, sequence_length, kv_tiles = keep.shape
        if heads % args.num_kv_heads:
            raise ValueError("--num-kv-heads must evenly divide dumped query heads")
        q_per_kv = heads // args.num_kv_heads
        counts = active_counts(keep, int(record["query_tile_size"]))
        layer = int(record["layer_id"])
        step = int(record["denoising_step"])
        mask_ratio = float(record["remaining_mask_ratio"])
        request = str(record["request_id"])
        add_marginal(marginals, "layer", layer, counts)
        add_marginal(marginals, "step", step, counts)
        add_marginal(marginals, "mask_ratio", mask_ratio, counts)
        for query_head in range(heads):
            add_marginal(marginals, "query_head", query_head, counts[query_head])
        for kv_head in range(args.num_kv_heads):
            start = kv_head * q_per_kv
            add_marginal(marginals, "kv_head", kv_head, counts[start : start + q_per_kv])
        for kv_tile in range(kv_tiles):
            add_marginal(marginals, "kv_tile", kv_tile, counts[:, :, kv_tile])
        unique, occurrences = torch.unique(counts, return_counts=True)
        for active, total in zip(unique.tolist(), occurrences.tolist()):
            joint_hist[(layer, step, mask_ratio, int(active))] += int(total)

        for tau in TAUS:
            histogram = exception_tiles_per_row_histogram(
                keep, counts, tau, int(record["query_tile_size"])
            )
            for value, total in enumerate(histogram.tolist()):
                if total:
                    row_hist[(tau, layer, step, mask_ratio, value)] += int(total)
            for grouping in GROUPINGS:
                totals, _ = grouped_exception_totals(
                    counts, tau, grouping=grouping, num_kv_heads=args.num_kv_heads
                )
                for pack_m in PACK_SIZES:
                    leftovers = totals.remainder(pack_m)
                    unique_left, left_counts = torch.unique(leftovers, return_counts=True)
                    for value, total in zip(unique_left.tolist(), left_counts.tolist()):
                        leftover_hist[(tau, pack_m, grouping, int(value))] += int(total)
                    metric = summarize_packability(
                        counts,
                        tau=tau,
                        pack_m=pack_m,
                        grouping=grouping,
                        num_kv_heads=args.num_kv_heads,
                        sequence_length=sequence_length,
                        query_tile_size=int(record["query_tile_size"]),
                        kv_tile_size=int(record["kv_tile_size"]),
                        head_dim=args.head_dim,
                        dtype_bytes=args.dtype_bytes,
                    ).to_dict()
                    row = {
                        "request_id": request, "layer": layer, "step": step,
                        "remaining_mask_ratio": mask_ratio, **metric,
                    }
                    per_invocation.append(row)
                    config_rows[(tau, pack_m, grouping)].append(row)

    metadata_limit = int(args.metadata_limit_mib * 1024 * 1024)
    configurations = [
        aggregate_configuration(config_rows[key], metadata_limit)
        for key in sorted(config_rows, key=lambda value: (value[0], value[1], value[2]))
    ]
    configurations.sort(
        key=lambda row: (
            row["hardware_phase_authorized"], row["phase_a_numeric_pass"],
            row["weighted_utilization"], row["v_load_factor"],
        ), reverse=True,
    )
    marginal_rows = [
        {"dimension": key[0], "value": key[1], "active_count": key[2], "tile_count": total}
        for key, total in sorted(marginals.items())
    ]
    joint_rows = [
        {"layer": key[0], "step": key[1], "remaining_mask_ratio": key[2],
         "active_count": key[3], "tile_count": total}
        for key, total in sorted(joint_hist.items())
    ]
    leftover_rows = [
        {"tau": key[0], "pack_m": key[1], "grouping": key[2],
         "leftover_rows": key[3], "grouping_keys": total}
        for key, total in sorted(leftover_hist.items())
    ]
    row_hist_rows = [
        {"tau": key[0], "layer": key[1], "step": key[2],
         "remaining_mask_ratio": key[3], "exception_kv_tiles_per_row": key[4],
         "row_count": total}
        for key, total in sorted(row_hist.items())
    ]
    write_csv(output / "configurations.csv", configurations)
    write_csv(output / "per_invocation.csv", per_invocation)
    write_csv(output / "active_count_marginals.csv", marginal_rows)
    write_csv(output / "active_count_by_layer_step_ratio.csv", joint_rows)
    write_csv(output / "leftover_histogram.csv", leftover_rows)
    write_csv(output / "row_exception_histogram.csv", row_hist_rows)

    numeric_passes = [row for row in configurations if row["phase_a_numeric_pass"]]
    authorized = [row for row in configurations if row["hardware_phase_authorized"]]
    report = {
        "source_dump": str(args.dump),
        "model": payload.get("model"),
        "records": len(records),
        "shape": {
            "sequence_length": int(records[0]["sequence_length"]),
            "query_heads": int(unpack_keep_mask(records[0]["keep_mask_packed"], int(records[0]["num_kv_tiles"])).shape[0]),
            "kv_heads": args.num_kv_heads,
            "query_tile_size": int(records[0]["query_tile_size"]),
            "kv_tile_size": int(records[0]["kv_tile_size"]),
            "head_dim": args.head_dim,
            "dtype_bytes": args.dtype_bytes,
        },
        "semantic_audit": {
            "existing_retained_tile_behavior": (
                "When any row votes keep, the existing kernel updates running m, l, "
                "and the P@V accumulator for all valid rows in the 128-row tile."
            ),
            "active_voter_only_replacement_equivalent": False,
            "reason": (
                "Packing only c positive voters omits P@V contributions for the 128-c "
                "negative voters that the baseline retained tile still evaluates."
            ),
        },
        "phase_a_numeric_passes": len(numeric_passes),
        "hardware_phase_authorized_configurations": len(authorized),
        "decision": (
            "proceed-to-hopper-microkernel" if authorized
            else "stop-before-hopper-microkernel"
        ),
        "configurations": configurations,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        **{key: report[key] for key in (
            "source_dump", "records", "shape", "semantic_audit",
            "phase_a_numeric_passes", "hardware_phase_authorized_configurations", "decision",
        )},
        "configurations": configurations,
    }, indent=2))


if __name__ == "__main__":
    main()
