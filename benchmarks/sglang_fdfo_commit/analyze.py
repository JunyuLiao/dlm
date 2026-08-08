#!/usr/bin/env python3
"""Analyze raw FDFO commit profiling events and render the required report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_LAYERS = 20
KV_HEADS = 4
HEAD_DIM = 128
DTYPE_BYTES = 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def pct(part: float | None, whole: float | None) -> float | None:
    return 100 * part / whole if part is not None and whole not in (None, 0) else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def expected_kv_bytes(block_size: int, batch_size: int = 1) -> int:
    return (
        MODEL_LAYERS
        * block_size
        * batch_size
        * KV_HEADS
        * HEAD_DIM
        * 2
        * DTYPE_BYTES
    )


def assign_workload(
    record: dict[str, Any], workloads: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    timestamp = record.get("started_time_ns", record.get("time_ns"))
    if timestamp is None:
        return None
    for workload in workloads:
        if workload["started_time_ns"] <= timestamp <= workload["finished_time_ns"]:
            return workload
    return None


def load_run(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    workloads = []
    for path in root.rglob("workloads.jsonl"):
        workloads.extend(read_jsonl(path))
    workloads.sort(key=lambda row: row["started_time_ns"])
    events = []
    for path in root.rglob("events-*.jsonl"):
        events.extend(read_jsonl(path))
    iterations = []
    scheduler = []
    for event in events:
        workload = assign_workload(event, workloads)
        if workload is None or workload.get("warmup"):
            continue
        enriched = {
            **event,
            "workload": workload["name"],
            "graph_mode": workload["graph_mode"],
            "context_length": workload["context_length"],
            "requested_batch_size": workload["batch_size"],
            "repetition": workload["repetition"],
        }
        if event.get("type") == "fdfo_iteration":
            # Older/raw schema-v1 traces labeled every accepted block as a
            # commit. Prompt prefill blocks also return done=True, but their
            # prefix is still before the exact client prompt boundary. Recover
            # the intended generation-only classification from per-request
            # accept lengths and prefix lengths. New traces additionally carry
            # prompt_prefill_count directly from algo-state presence.
            accepts = event.get("accept_lengths") or []
            prefixes = event.get("prefix_lens") or []
            prompt_boundary = workload["context_length"]
            if len(accepts) == event.get("batch_size") and len(prefixes) == len(accepts):
                prompt_prefill_count = sum(
                    int(accepted) > 0 and int(prefix) < prompt_boundary
                    for accepted, prefix in zip(accepts, prefixes)
                )
                final_commit_count = sum(
                    int(accepted) > 0 and int(prefix) >= prompt_boundary
                    for accepted, prefix in zip(accepts, prefixes)
                )
                batch_size = int(event["batch_size"])
                enriched.update(
                    {
                        "accepted_count": sum(int(value) > 0 for value in accepts),
                        "prompt_prefill_count": prompt_prefill_count,
                        "commit_count": final_commit_count,
                        "normal_count": batch_size
                        - prompt_prefill_count
                        - final_commit_count,
                        "commit_fraction": final_commit_count / batch_size,
                        "composition": (
                            "prompt_prefill"
                            if prompt_prefill_count
                            else "all_normal"
                            if final_commit_count == 0
                            else "all_commit"
                            if final_commit_count == batch_size
                            else "mixed"
                        ),
                    }
                )
            iterations.append(enriched)
        elif str(event.get("type", "")).startswith("scheduler_"):
            scheduler.append(enriched)
    return workloads, iterations, scheduler


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = median(row.get("total_gpu_ms") for row in rows)
    model = median(row.get("model_gpu_ms") for row in rows)
    attention = median(row.get("attention_gpu_ms") for row in rows)
    kv = median(row.get("kv_write_gpu_ms") for row in rows)
    step_gpu = median(row.get("step_gpu_ms") for row in rows)
    step_cpu = median(row.get("step_cpu_ms") for row in rows)
    logits = median(row.get("logits_gpu_ms") for row in rows)
    cpu = median(row.get("cpu_overhead_ms") for row in rows)
    wall = median(row.get("wall_ms") for row in rows)
    return {
        "samples": len(rows),
        "wall_ms": wall,
        "total_gpu_ms": total,
        "model_gpu_ms": model,
        "attention_gpu_ms": attention,
        "kv_write_gpu_ms": kv,
        "step_gpu_ms": step_gpu,
        "step_cpu_ms": step_cpu,
        "logits_gpu_ms": logits,
        "cpu_overhead_ms": cpu,
        "model_percent_total": pct(model, total),
        "attention_percent_model": pct(attention, model),
        "kv_percent_model": pct(kv, model),
        "step_percent_total": pct(step_gpu, total),
        "cpu_percent_wall": pct(cpu, wall),
        "per_request_model_ms": (
            model / rows[0]["batch_size"] if model is not None and rows else None
        ),
    }


def group_rows(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_iterations(iterations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "workload",
        "graph_mode",
        "block_size",
        "context_length",
        "requested_batch_size",
        "batch_size",
        "seq_lens",
        "prefix_lens",
        "extend_lens",
        "repetition",
        "composition",
        "accepted_count",
        "prompt_prefill_count",
        "commit_count",
        "normal_count",
        "commit_fraction",
        "cuda_graph",
        "wall_ms",
        "total_gpu_ms",
        "model_gpu_ms",
        "attention_gpu_ms",
        "kv_write_gpu_ms",
        "kv_write_bytes",
        "logits_gpu_ms",
        "step_gpu_ms",
        "step_cpu_ms",
        "cpu_overhead_ms",
    )
    return [{column: row.get(column) for column in columns} for row in iterations]


def breakdown_table(iterations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "graph_mode",
        "block_size",
        "context_length",
        "batch_size",
        "composition",
    )
    output = []
    for key, rows in sorted(group_rows(iterations, keys).items(), key=lambda item: str(item[0])):
        summary = summarize_group(rows)
        output.append(dict(zip(keys, key)) | summary)
    return output


def mixed_table(iterations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = [
        row
        for row in iterations
        if (
            (
                row["graph_mode"] == "eager"
                and row["block_size"] == 8
                and row["context_length"] == 128
                and row["batch_size"] == 4
            )
            or (
                row["block_size"] == 32
                and row["context_length"] == 1024
                and row["batch_size"] in (4, 16)
            )
        )
        and not row.get("prompt_prefill_count")
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in relevant:
        bucket = round(float(row["commit_fraction"]) * 4) / 4
        grouped[
            (
                row["graph_mode"],
                row["block_size"],
                row["context_length"],
                row["batch_size"],
                bucket,
            )
        ].append(row)
    output = []
    for (mode, block_size, context_length, batch_size, bucket), rows in sorted(
        grouped.items()
    ):
        summary = summarize_group(rows)
        output.append(
            {
                "graph_mode": mode,
                "block_size": block_size,
                "context_length": context_length,
                "batch_size": batch_size,
                "commit_fraction_bucket": bucket,
                "observed_fraction_mean": mean(row["commit_fraction"] for row in rows),
                **summary,
            }
        )
    return output


def shape_table(iterations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    selections = [
        ("block_size", lambda r: r["graph_mode"] == "eager" and r["context_length"] == 1024 and r["batch_size"] == 4),
        ("context_length", lambda r: r["graph_mode"] == "eager" and r["block_size"] == 32 and r["batch_size"] == 4),
        ("batch_size", lambda r: r["graph_mode"] == "eager" and r["block_size"] == 32 and r["context_length"] == 1024),
    ]
    for dimension, predicate in selections:
        selected = [row for row in iterations if predicate(row)]
        for (value, composition), rows in sorted(
            group_rows(selected, (dimension, "composition")).items(), key=lambda item: str(item[0])
        ):
            output.append(
                {
                    "sweep": dimension,
                    "value": value,
                    "composition": composition,
                    **summarize_group(rows),
                }
            )
    return output


def matched_commit_ratio(iterations: Sequence[dict[str, Any]]) -> tuple[float | None, int]:
    keys = ("graph_mode", "block_size", "context_length", "batch_size")
    ratios = []
    for _, rows in group_rows(iterations, keys).items():
        normal = median(row.get("model_gpu_ms") for row in rows if row["composition"] == "all_normal")
        commit = median(row.get("model_gpu_ms") for row in rows if row["composition"] == "all_commit")
        if normal and commit:
            ratios.append(commit / normal)
    return median(ratios), len(ratios)


def ratio_sweep_table(iterations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("graph_mode", "block_size", "context_length", "batch_size")
    matched = []
    for key, rows in group_rows(iterations, keys).items():
        if key[0] != "eager":
            continue
        normal = median(
            row.get("model_gpu_ms")
            for row in rows
            if row["composition"] == "all_normal"
        )
        commit = median(
            row.get("model_gpu_ms")
            for row in rows
            if row["composition"] == "all_commit"
        )
        if normal and commit:
            matched.append(
                dict(zip(keys, key))
                | {
                    "normal_model_gpu_ms": normal,
                    "commit_model_gpu_ms": commit,
                    "commit_over_normal": commit / normal,
                }
            )
    output = []
    for dimension in ("block_size", "context_length", "batch_size"):
        for (value,), rows in sorted(
            group_rows(matched, (dimension,)).items(), key=lambda item: item[0]
        ):
            output.append(
                {
                    "sweep": dimension,
                    "value": value,
                    "matched_cells": len(rows),
                    "normal_model_gpu_ms_median": median(
                        row["normal_model_gpu_ms"] for row in rows
                    ),
                    "commit_model_gpu_ms_median": median(
                        row["commit_model_gpu_ms"] for row in rows
                    ),
                    "commit_over_normal_median": median(
                        row["commit_over_normal"] for row in rows
                    ),
                }
            )
    return output


def compute_opportunity(iterations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    detailed = [
        row
        for row in iterations
        if row["graph_mode"] == "eager" and not row.get("prompt_prefill_count")
    ]
    total_gpu = sum(float(row.get("total_gpu_ms") or 0) for row in detailed)
    commit_weighted_model = sum(
        float(row.get("model_gpu_ms") or 0) * float(row["commit_fraction"])
        for row in detailed
    )
    commit_weighted_kv = sum(
        float(row.get("kv_write_gpu_ms") or 0) * float(row["commit_fraction"])
        for row in detailed
    )
    commit_weighted_step = sum(
        float(row.get("step_gpu_ms") or 0) * float(row["commit_fraction"])
        for row in detailed
    )
    commit_weighted_logits = sum(
        float(row.get("logits_gpu_ms") or 0) * float(row["commit_fraction"])
        for row in detailed
    )
    commit_weighted_last_block = 0.0
    for row in detailed:
        layer_times = [
            float(item["gpu_ms"])
            for item in row.get("block_times", [])
            if item.get("layer_id") == MODEL_LAYERS - 1 and item.get("gpu_ms") is not None
        ]
        commit_weighted_last_block += sum(layer_times) * float(row["commit_fraction"])

    write_only_removable = max(0.0, commit_weighted_model - commit_weighted_kv) + commit_weighted_step
    dependency_tail_removable = max(
        0.0,
        commit_weighted_last_block + commit_weighted_logits + commit_weighted_step - commit_weighted_kv,
    )
    step_only_removable = commit_weighted_step
    postprocess_only_removable = commit_weighted_logits + commit_weighted_step

    def ceiling(removable: float) -> dict[str, float | None]:
        fraction = removable / total_gpu if total_gpu else None
        return {
            "removable_gpu_ms": removable,
            "end_to_end_percent": 100 * fraction if fraction is not None else None,
            "upper_bound_speedup": 1 / (1 - fraction) if fraction is not None and fraction < 1 else None,
        }

    pure_commit_by_bs: dict[int, float] = {}
    for (batch_size,), rows in group_rows(
        [row for row in detailed if row["composition"] == "all_commit"], ("batch_size",)
    ).items():
        value = median(row.get("model_gpu_ms") for row in rows)
        if value is not None:
            pure_commit_by_bs[int(batch_size)] = value / int(batch_size)
    commit_batching_speedup = None
    delayed_commit_end_to_end = None
    if 1 in pure_commit_by_bs and pure_commit_by_bs:
        largest = max(pure_commit_by_bs)
        commit_batching_speedup = pure_commit_by_bs[1] / pure_commit_by_bs[largest]
        if commit_batching_speedup > 1 and total_gpu:
            target_per_request = pure_commit_by_bs[largest]
            pure_commit_rows = [
                row for row in detailed if row["composition"] == "all_commit"
            ]
            removable = sum(
                max(
                    0.0,
                    float(row.get("model_gpu_ms") or 0)
                    - int(row["batch_size"]) * target_per_request,
                )
                for row in pure_commit_rows
            )
            delayed_commit_end_to_end = ceiling(
                removable
            )

    pure_lookup: dict[tuple[Any, ...], float] = {}
    lookup_keys = ("graph_mode", "block_size", "context_length", "batch_size", "composition")
    for key, rows in group_rows(detailed, lookup_keys).items():
        value = median(row.get("model_gpu_ms") for row in rows)
        if value is not None:
            pure_lookup[key] = value
    separate_ratios = []
    for row in detailed:
        if row["composition"] != "mixed":
            continue
        prefix = (row["graph_mode"], row["block_size"], row["context_length"])
        normal = pure_lookup.get(prefix + (row["normal_count"], "all_normal"))
        commit = pure_lookup.get(prefix + (row["commit_count"], "all_commit"))
        current = row.get("model_gpu_ms")
        if normal is not None and commit is not None and current:
            separate_ratios.append(float(current) / (normal + commit))

    return {
        "profiled_total_gpu_ms": total_gpu,
        "commit_attributable_model_gpu_ms": commit_weighted_model,
        "commit_attributable_model_fraction_total_percent": pct(commit_weighted_model, total_gpu),
        "commit_attributable_kv_gpu_ms": commit_weighted_kv,
        "kv_fraction_of_commit_model_percent": pct(commit_weighted_kv, commit_weighted_model),
        "step_only": ceiling(step_only_removable),
        "token_selection_and_logits": ceiling(postprocess_only_removable),
        "dependency_respecting_tail": ceiling(dependency_tail_removable),
        "write_only_fantasy": ceiling(write_only_removable),
        "delayed_commit_batching": {
            "pure_commit_per_request_model_ms_by_batch_size": pure_commit_by_bs,
            "batch_1_to_largest_throughput_speedup": commit_batching_speedup,
            "end_to_end_ceiling": delayed_commit_end_to_end,
        },
        "separate_vs_mixed": {
            "matched_mixed_samples": len(separate_ratios),
            "mixed_over_separate_model_time_ratio_median": median(separate_ratios),
            "mixed_over_separate_model_time_ratio_min": min(separate_ratios) if separate_ratios else None,
            "mixed_over_separate_model_time_ratio_max": max(separate_ratios) if separate_ratios else None,
            "upper_bound_speedup_when_all_observed_ratios_favor_mixing": (
                1.0 if separate_ratios and max(separate_ratios) <= 1 else None
            ),
            "interpretation": "ratio below 1 means one mixed launch is faster than two measured pure sub-batch launches",
        },
        "allocation_note": "mixed-batch time is allocated to commit request slots in proportion to commit_count / batch_size",
    }


def telemetry_summary(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in root.rglob("gpu.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue
        utilization = []
        power = []
        for row in rows:
            try:
                utilization.append(float(row["utilization_gpu_percent"]))
                power.append(float(row["power_draw_watts"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not utilization:
            continue
        parts = path.relative_to(root).parts
        output.append(
            {
                "graph_mode": parts[0],
                "block_size": int(parts[1].split("_")[-1]),
                "samples": len(utilization),
                "gpu_utilization_mean_percent": statistics.fmean(utilization),
                "gpu_utilization_median_percent": statistics.median(utilization),
                "gpu_utilization_max_percent": max(utilization),
                "power_mean_watts": statistics.fmean(power) if power else None,
            }
        )
    return sorted(output, key=lambda row: (row["graph_mode"], row["block_size"]))


def validate_experiment(
    root: Path,
    workloads: Sequence[dict[str, Any]],
    iterations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    arguments_path = root / "arguments.json"
    arguments = json.loads(arguments_path.read_text()) if arguments_path.exists() else {}
    measured_workloads = [row for row in workloads if not row.get("warmup")]
    expected_eager = (
        len(arguments.get("block_sizes", []))
        * len(arguments.get("context_lengths", []))
        * len(arguments.get("batch_sizes", []))
        * int(arguments.get("repetitions", 0))
    )
    expected_captured = (
        0
        if arguments.get("skip_cuda_graph")
        else len(arguments.get("batch_sizes", []))
        * int(arguments.get("repetitions", 0))
    )
    actual_eager = sum(row.get("graph_mode") == "eager" for row in measured_workloads)
    actual_captured = sum(
        row.get("graph_mode") == "captured" for row in measured_workloads
    )
    kv_mismatches = [
        {
            "workload": row.get("workload"),
            "expected": expected_kv_bytes(row["block_size"], row["batch_size"]),
            "actual": row.get("kv_write_bytes"),
        }
        for row in iterations
        if row.get("graph_mode") == "eager"
        and row.get("kv_write_calls")
        and row.get("kv_write_bytes")
        != expected_kv_bytes(row["block_size"], row["batch_size"])
    ]
    controlled = [
        row
        for row in iterations
        if row.get("graph_mode") == "eager"
        and row.get("block_size") == 8
        and row.get("context_length") == 128
        and row.get("batch_size") == 4
        and not row.get("prompt_prefill_count")
    ]
    controlled_fractions = sorted({float(row["commit_fraction"]) for row in controlled})
    ratio, ratio_cells = matched_commit_ratio(iterations)
    checks = {
        "all_workloads_succeeded": all(
            int(row.get("failure_count", 0)) == 0 for row in measured_workloads
        ),
        "eager_cross_product_complete": actual_eager == expected_eager,
        "captured_cross_product_complete": actual_captured == expected_captured,
        "normal_commit_matched_cells_present": ratio is not None and ratio_cells > 0,
        "controlled_mixed_fractions_present": all(
            value in controlled_fractions for value in (0.0, 0.25, 0.5, 0.75)
        ),
        "kv_bytes_match_model_shape": not kv_mismatches,
        "cuda_graph_replay_observed": (
            True
            if expected_captured == 0
            else any(
                row.get("graph_mode") == "captured" and row.get("cuda_graph") is True
                for row in iterations
            )
        ),
        "bandwidth_measurement_present": (root / "hbm_bandwidth.json").exists(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_eager_workloads": expected_eager,
        "actual_eager_workloads": actual_eager,
        "expected_captured_workloads": expected_captured,
        "actual_captured_workloads": actual_captured,
        "measured_iterations": len(iterations),
        "matched_commit_normal_cells": ratio_cells,
        "controlled_fractions": controlled_fractions,
        "kv_mismatches": kv_mismatches,
    }


def graph_comparison(iterations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in iterations
        if row["block_size"] == 32 and row["context_length"] == 1024
        and not row.get("prompt_prefill_count")
    ]
    output: dict[str, Any] = {}
    for mode in ("eager", "captured"):
        rows = [row for row in selected if row["graph_mode"] == mode]
        output[mode] = {
            "samples": len(rows),
            "model_gpu_ms": median(row.get("model_gpu_ms") for row in rows),
            "wall_ms": median(row.get("wall_ms") for row in rows),
            "cpu_overhead_ms": median(row.get("cpu_overhead_ms") for row in rows),
            "cuda_graph_true_fraction": mean(1.0 if row.get("cuda_graph") else 0.0 for row in rows),
        }
    eager = output["eager"]["model_gpu_ms"]
    captured = output["captured"]["model_gpu_ms"]
    output["captured_vs_eager_model_ratio"] = captured / eager if eager and captured else None
    return output


def render_report(
    root: Path,
    iterations: Sequence[dict[str, Any]],
    breakdown: Sequence[dict[str, Any]],
    mixed: Sequence[dict[str, Any]],
    shapes: Sequence[dict[str, Any]],
    opportunity: dict[str, Any],
    bandwidth: dict[str, Any] | None,
    scheduler: Sequence[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    ratio, ratio_cells = matched_commit_ratio(iterations)
    pure_commit = summarize_group([row for row in iterations if row["composition"] == "all_commit"])
    pure_normal = summarize_group([row for row in iterations if row["composition"] == "all_normal"])
    kv_fraction = opportunity["kv_fraction_of_commit_model_percent"]
    measured_bw = bandwidth.get("destination_only_bandwidth_gb_s") if bandwidth else None
    bytes32 = expected_kv_bytes(32)
    lower_bound_ms = bytes32 / (measured_bw * 1e9) * 1000 if measured_bw else None
    observed_kv32 = median(
        row.get("kv_write_gpu_ms")
        for row in iterations
        if row["graph_mode"] == "eager" and row["block_size"] == 32
    )
    graph = graph_comparison(iterations)
    tail = opportunity["dependency_respecting_tail"]
    fantasy = opportunity["write_only_fantasy"]
    postprocess = opportunity["token_selection_and_logits"]
    separation = opportunity["separate_vs_mixed"]
    batching = opportunity["delayed_commit_batching"]
    scheduler_prepare_ms = median(
        row.get("cpu_ms") for row in scheduler if row.get("type") == "scheduler_prepare"
    )
    scheduler_result_ms = median(
        row.get("cpu_ms") for row in scheduler if row.get("type") == "scheduler_result"
    )

    telemetry = opportunity.get("telemetry", [])
    eager_telemetry = next(
        (
            row
            for row in telemetry
            if row["graph_mode"] == "eager" and row["block_size"] == 32
        ),
        {},
    )
    captured_telemetry = next(
        (
            row
            for row in telemetry
            if row["graph_mode"] == "captured" and row["block_size"] == 32
        ),
        {},
    )
    mixed_focus = [
        row
        for row in mixed
        if row["graph_mode"] == "eager"
        and row["block_size"] == 8
        and row["context_length"] == 128
        and row["batch_size"] == 4
    ]
    mixed_lines = []
    for row in mixed_focus:
        mixed_lines.append(
            f"| {row['commit_fraction_bucket']:.0%} | {row['samples']} | "
            f"{fmt(row['observed_fraction_mean'] * 100, 1)}% | {fmt(row['model_gpu_ms'])} | "
            f"{fmt(row['per_request_model_ms'])} | {fmt(row['cpu_overhead_ms'])} |"
        )
    if not mixed_lines:
        mixed_lines.append("| n/a | 0 | n/a | n/a | n/a | n/a |")

    shape_lines = []
    for row in shapes:
        if row["composition"] not in ("all_normal", "all_commit"):
            continue
        shape_lines.append(
            f"| {row['sweep']} | {row['value']} | {row['composition']} | {row['samples']} | "
            f"{fmt(row['model_gpu_ms'])} | {fmt(row['kv_write_gpu_ms'])} | "
            f"{fmt(row['step_gpu_ms'])} |"
        )
    if not shape_lines:
        shape_lines.append("| n/a | n/a | n/a | 0 | n/a | n/a | n/a |")

    ratio_lines = []
    for row in ratio_sweep_table(iterations):
        ratio_lines.append(
            f"| {row['sweep']} | {row['value']} | {row['matched_cells']} | "
            f"{fmt(row['normal_model_gpu_ms_median'])} | "
            f"{fmt(row['commit_model_gpu_ms_median'])} | "
            f"{fmt(row['commit_over_normal_median'], 4)}x |"
        )
    if not ratio_lines:
        ratio_lines.append("| n/a | n/a | 0 | n/a | n/a | n/a |")

    conclusion = (
        "not justified by the measured ceiling"
        if (tail.get("end_to_end_percent") or 0) < 5
        else "potentially justified, but only if a prototype approaches the optimistic last-layer ceiling"
    )
    return "\n".join(
        [
            "# SGLang FDFO final KV-cache commit profiling",
            "",
            "## Executive answer",
            "",
            f"1. **Commit vs normal cost:** matched pure-commit forwards are `{fmt(ratio, 4)}x` the model GPU time of pure normal forwards across {ratio_cells} matched shape cells. Both execute the same batch shape and nominal transformer FLOPs.",
            f"2. **Actual KV writing:** explicit paged-cache stores account for `{fmt_pct(kv_fraction)}` of commit-attributable model GPU time. At block 32, one request writes `{bytes32:,}` bytes; the measured write-only bandwidth lower bound is `{fmt(lower_bound_ms, 6)} ms`, versus `{fmt(observed_kv32, 6)} ms` observed across 20 writer launches.",
            f"3. **Potentially avoidable work:** skipping commit token selection plus logits has an end-to-end ceiling of `{fmt_pct(postprocess['end_to_end_percent'])}`. An optimistic last-layer-tail ceiling is `{fmt_pct(tail['end_to_end_percent'])}` or `{fmt(tail['upper_bound_speedup'], 4)}x`; it overstates removability by counting mandatory final-layer K/V projection time. The intentionally unrealistic write-only ceiling is `{fmt_pct(fantasy['end_to_end_percent'])}` or `{fmt(fantasy['upper_bound_speedup'], 4)}x`.",
            f"4. **Mixing:** the controlled composition table below reports fixed block/context/batch shapes. CUDA graph model-time ratio (captured/eager) is `{fmt(graph['captured_vs_eager_model_ratio'], 4)}`. For {separation['matched_mixed_samples']} mixed forwards with both exact pure sub-batch controls, mixed / two-separate-launch model time is `{fmt(separation['mixed_over_separate_model_time_ratio_median'], 4)}` (below 1 favors mixing).",
            f"5. **Recommendation:** a specialized commit path is **{conclusion}**. Separating commits should be judged against the optimistic last-layer ceiling, not the much larger write-only fantasy.",
            "",
            "## Exact implementation path",
            "",
            "Normal and commit requests are not separate scheduler modes. `SchedulerDllmMixin.get_new_batch_dllm()` prepares one `DLLM_EXTEND` batch; `TpModelWorker._forward_batch_generation_dllm()` calls `DllmAlgorithm.run()`; `_run_fdfo()` calls `ModelRunner.forward()` once and then `JointThreshold.step()`. A request is known to be a commit only when `step()` returns `done=True`; the scheduler then accepts its block in `process_batch_result_dllm()`.",
            "",
            "For LLaDA2.1-mini, `LLaDA2MoeAttention.forward()` computes fused Q/K/V projections, splits K and V, applies Q/K normalization and RoPE, then invokes `RadixAttention`. Because the model has `head_dim=128` and `rotary_dim=64`, its `can_fuse_set_kv` predicate is false. `FlashInferAttnBackend.forward_extend()` therefore calls `MHATokenToKVPool.set_kv_buffer()` explicitly. `_store_kv_layer()` dispatches the physical paged K/V store. Writes are consequently part of every layer's attention forward, not a scheduler-side copy after the model.",
            "",
            "A final commit uses the same `batch_size × block_size` token tensor, attention metadata, MoE blocks, final normalization, LM head, and logits processor as a normal denoising forward. The full logits are then consumed by `JointThreshold.step()` even though this terminating step changes no token. Logits and token selection are therefore avoidable once a commit-only request can be identified; the full preceding transformer is not generally avoidable because each layer's final K/V depends on hidden states produced by earlier attention and MLP layers.",
            "",
            "Authoritative installed paths: `sglang/srt/dllm/algorithm/base.py`, `sglang/srt/dllm/algorithm/joint_threshold.py`, `sglang/srt/dllm/mixin/scheduler.py`, `sglang/srt/managers/tp_worker.py`, `sglang/srt/models/llada2.py`, `sglang/srt/layers/attention/flashinfer_backend.py`, and `sglang/srt/mem_cache/memory_pool.py` (SGLang 0.5.16).",
            "",
            "## Iteration breakdown",
            "",
            "| Kind | N | Iteration GPU ms | Model ms (% iter.) | Attention ms (% model) | KV write ms (% model) | Step GPU/CPU ms | CPU overhead ms |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|",
            f"| all normal | {pure_normal['samples']} | {fmt(pure_normal['total_gpu_ms'])} | {fmt(pure_normal['model_gpu_ms'])} ({fmt_pct(pure_normal['model_percent_total'])}) | {fmt(pure_normal['attention_gpu_ms'])} ({fmt_pct(pure_normal['attention_percent_model'])}) | {fmt(pure_normal['kv_write_gpu_ms'])} ({fmt_pct(pure_normal['kv_percent_model'])}) | {fmt(pure_normal['step_gpu_ms'])}/{fmt(pure_normal['step_cpu_ms'])} | {fmt(pure_normal['cpu_overhead_ms'])} |",
            f"| all commit | {pure_commit['samples']} | {fmt(pure_commit['total_gpu_ms'])} | {fmt(pure_commit['model_gpu_ms'])} ({fmt_pct(pure_commit['model_percent_total'])}) | {fmt(pure_commit['attention_gpu_ms'])} ({fmt_pct(pure_commit['attention_percent_model'])}) | {fmt(pure_commit['kv_write_gpu_ms'])} ({fmt_pct(pure_commit['kv_percent_model'])}) | {fmt(pure_commit['step_gpu_ms'])}/{fmt(pure_commit['step_cpu_ms'])} | {fmt(pure_commit['cpu_overhead_ms'])} |",
            "",
            f"Median scheduler batch preparation is `{fmt(scheduler_prepare_ms)} ms`; median result processing is `{fmt(scheduler_result_ms)} ms`. CUDA-event time is GPU stream time. CPU overhead is synchronized iteration wall time minus total event time. `step_cpu_ms` includes host waits induced by tensor-dependent Python branches; it must not be added to GPU time.",
            "",
            "## KV-cache bytes and bandwidth",
            "",
            "Per request/block bytes are `20 layers × block_size × 4 KV heads × 128 head_dim × 2 (K,V) × 2 bytes`. Thus block sizes 8/16/32/64 write 327,680 / 655,360 / 1,310,720 / 2,621,440 bytes respectively. This is invariant between normal and commit forwards of the same shape.",
            "",
            f"The bandwidth microbenchmark measured `{fmt(measured_bw, 1)} GB/s` counting destination bytes only. Its block-32 physical lower bound is `{fmt(lower_bound_ms, 6)} ms`. The gap to observed writer time includes 20 small scatter/store kernel launches and non-ideal access, but the writer still occupies only `{fmt_pct(kv_fraction)}` of commit model time. The dominant cost is producing hidden states and K/V through the transformer, not moving 1.25 MiB into HBM.",
            "",
            "## Controlled shape sweeps",
            "",
            "| Sweep | Value | Composition | N | Model GPU ms | KV GPU ms | Step GPU ms |",
            "|:--|--:|:--|--:|--:|--:|--:|",
            *shape_lines,
            "",
            "The complete cross-product, including mixed batches and percentage fields, is in `breakdown.csv`; raw per-forward data is in `iterations.csv`.",
            "",
            "### Matched commit/normal ratios",
            "",
            "Ratios are computed inside exact graph/block/context/batch cells, then aggregated by one sweep dimension:",
            "",
            "| Sweep | Value | Matched cells | Normal model ms | Commit model ms | Commit / normal |",
            "|:--|--:|--:|--:|--:|--:|",
            *ratio_lines,
            "",
            "## Mixed commit + normal batches",
            "",
            "Block 8, 128-token context, total batch 4—the fixed-shape cell that naturally realized all requested mixed fractions:",
            "",
            "| Commit bucket | N | Observed mean | Model GPU ms | Per-request model ms | CPU overhead ms |",
            "|--:|--:|--:|--:|--:|--:|",
            *mixed_lines,
            "",
            f"All requests always extend exactly one block, so mixing causes no token padding or FLOP-shape change at fixed total batch size. It can keep a batch full, while separating it requires two smaller launches. At block 32, coarse 100 ms telemetry averaged `{fmt(eager_telemetry.get('gpu_utilization_mean_percent'), 1)}%` GPU utilization in eager mode and `{fmt(captured_telemetry.get('gpu_utilization_mean_percent'), 1)}%` with graphs; composition-specific CUDA-event model time is the higher-resolution efficiency measure. CUDA-graph eligibility is recorded per forward, and all telemetry summaries are in `opportunity.json`.",
            "",
            "## Opportunity bounds",
            "",
            f"Commit request slots account for `{fmt_pct(opportunity['commit_attributable_model_fraction_total_percent'])}` of profiled iteration GPU time (mixed batches allocated by request-slot fraction).",
            "",
            "| Hypothetical change | Maximum end-to-end removal | Upper-bound speedup |",
            "|:--|--:|--:|",
            f"| Avoid commit `step()` only | {fmt_pct(opportunity['step_only']['end_to_end_percent'])} | {fmt(opportunity['step_only']['upper_bound_speedup'], 4)}x |",
            f"| Avoid commit token selection + logits | {fmt_pct(postprocess['end_to_end_percent'])} | {fmt(postprocess['upper_bound_speedup'], 4)}x |",
            f"| Optimistic last-layer-tail bound (includes mandatory K/V projection) | {fmt_pct(tail['end_to_end_percent'])} | {fmt(tail['upper_bound_speedup'], 4)}x |",
            f"| Retain only measured KV stores (unphysical fantasy ceiling) | {fmt_pct(fantasy['end_to_end_percent'])} | {fmt(fantasy['upper_bound_speedup'], 4)}x |",
            f"| Separate mixed batches into pure sub-batches | 0.00% | {fmt(separation.get('upper_bound_speedup_when_all_observed_ratios_favor_mixing'), 4)}x |",
            f"| Delay commits from batch 1 to largest observed pure-commit batch | {fmt_pct((batching.get('end_to_end_ceiling') or {}).get('end_to_end_percent'))} | {fmt((batching.get('end_to_end_ceiling') or {}).get('upper_bound_speedup'), 4)}x |",
            "",
            f"Delaying commits from batch 1 to the largest observed pure-commit batch has a measured throughput ceiling of `{fmt(batching['batch_1_to_largest_throughput_speedup'], 4)}x` for commit work alone. Separating mixed work has a median mixed / separate model-time ratio of `{fmt(separation['mixed_over_separate_model_time_ratio_median'], 4)}` across {separation['matched_mixed_samples']} exact-composition controls. A real minimum-K/V path cannot reach the write-only ceiling because layer `l` K/V depends on the hidden state produced by all earlier attention and MLP blocks.",
            "",
            "## Reproduction and limitations",
            "",
            f"Automated validation: `{'passed' if validation['passed'] else 'FAILED'}` ({sum(validation['checks'].values())}/{len(validation['checks'])} checks). Raw root: `{root}`. Prompt-prefill blocks were excluded from commit accounting using the exact client prompt boundary; new traces also label them directly from carried algorithm state. Profiling is enabled only by `SGLANG_FDFO_PROFILE_DIR`; the startup patch wraps existing methods and returns their original outputs. CUDA events synchronize each profiled iteration and therefore measure component costs accurately but perturb throughput. End-to-end workload wall time and telemetry remain separate. Results apply to LLaDA2.1-mini, bf16 KV, FlashInfer, TP=1, one H100, and the pinned SGLang build; they should not be generalized to fused-RoPE/KV models without rerunning.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    workloads, iterations, scheduler = load_run(root)
    if not iterations:
        raise RuntimeError(f"no measured FDFO iterations found under {root}")
    breakdown = breakdown_table(iterations)
    mixed = mixed_table(iterations)
    shapes = shape_table(iterations)
    opportunity = compute_opportunity(iterations)
    opportunity["cuda_graph"] = graph_comparison(iterations)
    opportunity["telemetry"] = telemetry_summary(root)
    validation = validate_experiment(root, workloads, iterations)
    bandwidth_path = root / "hbm_bandwidth.json"
    bandwidth = json.loads(bandwidth_path.read_text()) if bandwidth_path.exists() else None

    write_csv(root / "iterations.csv", flatten_iterations(iterations))
    write_csv(root / "breakdown.csv", breakdown)
    write_csv(root / "mixed_batches.csv", mixed)
    write_csv(root / "shape_sweeps.csv", shapes)
    write_csv(root / "commit_normal_ratios.csv", ratio_sweep_table(iterations))
    write_csv(root / "scheduler.csv", scheduler)
    (root / "opportunity.json").write_text(
        json.dumps(opportunity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "report.md").write_text(
        render_report(
            root,
            iterations,
            breakdown,
            mixed,
            shapes,
            opportunity,
            bandwidth,
            scheduler,
            validation,
        ),
        encoding="utf-8",
    )
    print(root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
