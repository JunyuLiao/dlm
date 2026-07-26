#!/usr/bin/env python3
"""Context-disjoint analysis of physical attention-result reuse traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.tile_reuse import AttentionTileStatistics, compose_attention_tile_statistics


ORACLE_THRESHOLDS = (1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
RAW_PATTERN = re.compile(
    r"raw-(?P<request>.+)-step(?P<step>\d+)-layer(?P<layer>\d+)-qtile(?P<qtile>\d+)\.pt"
)


def wilson_upper(errors: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 1.0
    p = errors / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (center + radius) / denominator


def load_summaries(trace_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    plan = json.loads((trace_dir / "collection_plan.json").read_text(encoding="utf-8"))
    split_map = plan["context_splits"]
    chunks: dict[str, list[np.ndarray]] = {}
    for path in sorted(trace_dir.glob("context-*/tile-reuse-summary.npz")):
        with np.load(path) as payload:
            for name in payload.files:
                if name == "schema_version":
                    continue
                chunks.setdefault(name, []).append(payload[name])
    if not chunks:
        raise ValueError(f"no summary traces found under {trace_dir}")
    return {name: np.concatenate(values) for name, values in chunks.items()}, split_map


def phase(ratio: np.ndarray) -> np.ndarray:
    return np.where(ratio >= 0.75, 0, np.where(ratio >= 0.25, 1, 2)).astype(np.int8)


def reuse_effect(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Prefer exact full-output counterfactuals from schema v2."""

    return data.get("one_tile_full_output_error_max", data["tile_output_error_max"])


def grouped_oracle_rows(data: Mapping[str, np.ndarray], split: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    error = reuse_effect(data)
    dimensions: list[tuple[str, np.ndarray]] = [
        ("all", np.zeros(len(error), dtype=np.int8)),
        ("split", split),
        ("phase", phase(data["remaining_mask_ratio"])),
        ("layer", data["layer"]),
        ("head", data["head"]),
        ("step", data["step"]),
        ("distance_bucket", np.minimum(data["diagonal_distance"] // 4, 8)),
    ]
    for name, values in dimensions:
        for value in np.unique(values):
            selected = values == value
            for threshold in ORACLE_THRESHOLDS:
                rows.append(
                    {
                        "dimension": name,
                        "value": str(value),
                        "error_threshold": threshold,
                        "records": int(selected.sum()),
                        "oracle_reuse_fraction": float(np.mean(error[selected] <= threshold)),
                        "median_tile_error": float(np.median(error[selected])),
                        "p99_tile_error": float(np.quantile(error[selected], 0.99)),
                    }
                )
    return rows


def previous_step_field(data: Mapping[str, np.ndarray], field: str) -> np.ndarray:
    result = np.full(len(data[field]), np.nan, dtype=np.float32)
    order = np.lexsort(
        (
            data["step"],
            data["kv_tile"],
            data["query_tile"],
            data["head"],
            data["layer"],
            data["request_id"],
        )
    )
    earlier, later = order[:-1], order[1:]
    same = (
        (data["request_id"][earlier] == data["request_id"][later])
        & (data["layer"][earlier] == data["layer"][later])
        & (data["head"][earlier] == data["head"][later])
        & (data["query_tile"][earlier] == data["query_tile"][later])
        & (data["kv_tile"][earlier] == data["kv_tile"][later])
        & (data["step"][later] == data["step"][earlier] + 1)
    )
    result[later[same]] = data[field][earlier[same]].astype(np.float32)
    return result


def predictor_scores(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for statistic in ("fro", "mean", "max", "q90", "q99"):
        q = data[f"q_{statistic}"]
        k = data[f"k_{statistic}"]
        v = data[f"v_{statistic}"]
        result[f"q_only_{statistic}"] = q
        result[f"kv_only_max_{statistic}"] = np.maximum(k, v)
        result[f"kv_only_sum_{statistic}"] = k + v
        result[f"qk_max_{statistic}"] = np.maximum(q, k)
        result[f"qk_sum_{statistic}"] = q + k
        result[f"qk_product_{statistic}"] = q * k
        for beta in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            result[f"qkv_sum_b{beta:g}_{statistic}"] = q + k + beta * v
            result[f"qkv_max_b{beta:g}_{statistic}"] = np.maximum(
                np.maximum(q, k), beta * v
            )
    if "current_tile_mass_max" in data:
        for mass_statistic in ("mean", "max", "q99"):
            previous_mass = previous_step_field(
                data, f"current_tile_mass_{mass_statistic}"
            )
            result[f"previous_mass_{mass_statistic}"] = previous_mass
            for drift_statistic in ("fro", "mean", "max", "q90", "q99"):
                q = data[f"q_{drift_statistic}"]
                k = data[f"k_{drift_statistic}"]
                v = data[f"v_{drift_statistic}"]
                result[
                    f"previous_mass_{mass_statistic}_x_q_{drift_statistic}"
                ] = previous_mass * q
                result[
                    f"previous_mass_{mass_statistic}_x_kv_{drift_statistic}"
                ] = previous_mass * np.maximum(k, v)
                result[
                    f"previous_mass_{mass_statistic}_x_qk_{drift_statistic}"
                ] = previous_mass * (q + k)
                for beta in (0.1, 0.25, 0.5, 1.0, 2.0):
                    result[
                        f"previous_mass_{mass_statistic}_x_qkv_b{beta:g}_{drift_statistic}"
                    ] = previous_mass * (q + k + beta * v)
    return result


def calibrate_group_thresholds(
    score: np.ndarray,
    safe: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    calibration: np.ndarray,
    *,
    minimum_support: int = 256,
    maximum_wilson: float = 0.01,
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for group in np.unique(groups):
        indices = np.flatnonzero(
            calibration & eligible & (groups == group) & np.isfinite(score)
        )
        if indices.size < minimum_support:
            thresholds[int(group)] = -math.inf
            continue
        order = indices[np.argsort(score[indices], kind="stable")]
        errors = (~safe[order]).astype(np.int64)
        cumulative = np.cumsum(errors)
        acceptable = [
            index
            for index in range(minimum_support - 1, len(order))
            if wilson_upper(int(cumulative[index]), index + 1) <= maximum_wilson
        ]
        thresholds[int(group)] = (
            float(np.nextafter(score[order[acceptable[-1]]], math.inf))
            if acceptable
            else -math.inf
        )
    return thresholds


def apply_thresholds(
    score: np.ndarray, groups: np.ndarray, thresholds: Mapping[int, float]
) -> np.ndarray:
    selected = np.zeros(len(score), dtype=bool)
    for group, threshold in thresholds.items():
        selected |= (groups == group) & (score < threshold)
    return selected


def policy_row(
    name: str,
    split_name: str,
    selected: np.ndarray,
    scope: np.ndarray,
    safe: np.ndarray,
    error: np.ndarray,
) -> dict[str, object]:
    reused = selected & scope
    count = int(reused.sum())
    unsafe = int((reused & ~safe).sum())
    values = error[reused]
    return {
        "policy": name,
        "split": split_name,
        "records": int(scope.sum()),
        "reused_tiles": count,
        "reuse_fraction": count / max(int(scope.sum()), 1),
        "prediction_precision": 1.0 - unsafe / count if count else 1.0,
        "unsafe_reuse_rate": unsafe / count if count else 0.0,
        "unsafe_wilson95_upper": wilson_upper(unsafe, count),
        "oracle_safe_recall": int((reused & safe).sum()) / max(int((scope & safe).sum()), 1),
        "mean_reused_tile_error": float(values.mean()) if values.size else 0.0,
        "maximum_reused_tile_error": float(values.max(initial=0.0)),
    }


def evaluate_predictors(
    data: Mapping[str, np.ndarray], split: np.ndarray
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, float]]
]:
    error = reuse_effect(data)
    safe = error <= 0.01
    phases = phase(data["remaining_mask_ratio"])
    groups = phases.astype(np.int16) * 4 + (data["layer"] // 8).astype(np.int16)
    calibration = split == "calibration"
    policies: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    calibrated: dict[str, dict[str, float]] = {}
    scopes = {name: split == name for name in np.unique(split)}
    eligibility = {
        "all": np.ones(len(error), dtype=bool),
        "protect_masked_new": (
            (data["query_state_0_fraction"] + data["query_state_1_fraction"] == 0)
            & (data["kv_state_0_fraction"] + data["kv_state_1_fraction"] == 0)
        ),
    }
    for predictor, score in predictor_scores(data).items():
        finite = np.isfinite(score) & np.isfinite(error)
        correlations.append(
            {
                "predictor": predictor,
                "log_pearson_vs_tile_error": float(
                    np.corrcoef(np.log1p(score[finite]), np.log1p(error[finite]))[0, 1]
                ),
            }
        )
        for eligibility_name, eligible in eligibility.items():
            thresholds = calibrate_group_thresholds(
                score, safe, eligible, groups, calibration
            )
            calibrated[f"{predictor}:{eligibility_name}"] = {
                str(group): threshold for group, threshold in thresholds.items()
            }
            selected = eligible & apply_thresholds(score, groups, thresholds)
            for split_name, scope in scopes.items():
                policies.append(
                    policy_row(
                        f"{predictor}:{eligibility_name}",
                        split_name,
                        selected,
                        scope,
                        safe,
                        error,
                    )
                )
    return (
        policies,
        sorted(
            correlations, key=lambda row: row["log_pearson_vs_tile_error"], reverse=True
        ),
        calibrated,
    )


def state_and_region_policies(
    data: Mapping[str, np.ndarray], split: np.ndarray
) -> list[dict[str, object]]:
    error = reuse_effect(data)
    safe = error <= 0.01
    query_stable = data["query_state_3_fraction"] == 1.0
    kv_stable = data["kv_state_3_fraction"] == 1.0
    no_query_critical = data["query_state_0_fraction"] + data["query_state_1_fraction"] == 0
    no_kv_critical = data["kv_state_0_fraction"] + data["kv_state_1_fraction"] == 0
    direct = {
        "stable_query_x_stable_kv": query_stable & kv_stable,
        "stable_query_x_any_kv": query_stable,
        "any_query_x_stable_kv": kv_stable,
        "protect_masked_or_new": no_query_critical & no_kv_critical,
        "fixed_prefix_kv_tile": data["kv_tile"] == 0,
        "fixed_external_outside_local4": data["diagonal_distance"] > 4,
        "fixed_external_outside_local8": data["diagonal_distance"] > 8,
        "fixed_local_only": data["diagonal_distance"] <= 4,
        "reuse_one_step_refresh": data["step"] % 2 == 1,
    }
    rows: list[dict[str, object]] = []
    for name, selected in direct.items():
        for split_name in np.unique(split):
            rows.append(
                policy_row(name, str(split_name), selected, split == split_name, safe, error)
            )
    return rows


def query_tile_analysis(
    data: Mapping[str, np.ndarray], split: np.ndarray
) -> list[dict[str, object]]:
    key = np.rec.fromarrays(
        [data[name] for name in ("request_id", "layer", "head", "step", "query_tile")],
        names="request,layer,head,step,query",
    )
    _, indices = np.unique(key, return_index=True)
    error = data["final_output_error_max"][indices]
    safe = error <= 0.01
    query_split = split[indices]
    rows: list[dict[str, object]] = []
    for threshold in ORACLE_THRESHOLDS:
        selected = error <= threshold
        for split_name in np.unique(query_split):
            scope = query_split == split_name
            rows.append(
                {
                    "policy": f"oracle_query_tile_error_le_{threshold:g}",
                    "split": str(split_name),
                    "records": int(scope.sum()),
                    "reuse_fraction": float(np.mean(selected[scope])),
                    "median_query_tile_error": float(np.median(error[scope])),
                    "p99_query_tile_error": float(np.quantile(error[scope], 0.99)),
                }
            )
    q_score = data["q_max"][indices]
    phases = phase(data["remaining_mask_ratio"][indices])
    groups = phases.astype(np.int16) * 4 + (data["layer"][indices] // 8).astype(np.int16)
    thresholds = calibrate_group_thresholds(
        q_score, safe, np.ones(len(indices), bool), groups, query_split == "calibration"
    )
    selected = apply_thresholds(q_score, groups, thresholds)
    for split_name in np.unique(query_split):
        rows.append(
            policy_row(
                "query_tile_q_max",
                str(split_name),
                selected,
                query_split == split_name,
                safe,
                error,
            )
        )
    return rows


def raw_refresh_analysis(trace_dir: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in trace_dir.glob("context-*/raw-*.pt"):
        match = RAW_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("request"), int(match.group("layer")), int(match.group("qtile")))
        groups.setdefault(key, []).append((int(match.group("step")), path))
    rows: list[dict[str, object]] = []
    for reuse_steps in (1, 2, 3):
        errors: list[np.ndarray] = []
        query_errors: list[np.ndarray] = []
        reused = total = 0
        for paths in groups.values():
            cache_tile = cache_query = None
            age = 0
            for _, path in sorted(paths):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                current_tile = payload["tile_normalized_output"].float()
                current_query = payload["final_attention_output"].float()
                total += current_tile.shape[0] * current_tile.shape[1] * current_tile.shape[2]
                if cache_tile is None or age > reuse_steps:
                    cache_tile, cache_query, age = current_tile, current_query, 1
                    continue
                tile_error = torch.linalg.vector_norm(current_tile - cache_tile, dim=-1) / (
                    torch.linalg.vector_norm(current_tile, dim=-1) + 1e-6
                )
                query_error = torch.linalg.vector_norm(current_query - cache_query, dim=-1) / (
                    torch.linalg.vector_norm(current_query, dim=-1) + 1e-6
                )
                errors.append(tile_error.amax(-1).reshape(-1).numpy())
                query_errors.append(query_error.amax(-1).reshape(-1).numpy())
                reused += tile_error.shape[0] * tile_error.shape[1] * tile_error.shape[2]
                age += 1
        values = np.concatenate(errors) if errors else np.empty(0)
        q_values = np.concatenate(query_errors) if query_errors else np.empty(0)
        rows.append(
            {
                "policy": f"reuse_{reuse_steps}_then_refresh",
                "reuse_fraction": reused / max(total, 1),
                "tile_safe_fraction_at_1pct": float(np.mean(values <= 0.01)) if values.size else 0.0,
                "median_tile_error": float(np.median(values)) if values.size else 0.0,
                "p99_tile_error": float(np.quantile(values, 0.99)) if values.size else 0.0,
                "query_safe_fraction_at_1pct": float(np.mean(q_values <= 0.01)) if q_values.size else 0.0,
                "median_query_error": float(np.median(q_values)) if q_values.size else 0.0,
            }
        )
    return rows


def raw_cumulative_analysis(
    trace_dir: Path, data: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    """Evaluate multi-step budgets against the actual last-refreshed tile."""

    lookup: dict[tuple[str, int, int, int], int] = {}
    for index in range(len(data["step"])):
        if int(data["head"][index]) != 0:
            continue
        lookup[
            (
                str(data["request_id"][index]),
                int(data["layer"][index]),
                int(data["step"][index]),
                int(data["kv_tile"][index]),
            )
        ] = index
    groups: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in trace_dir.glob("context-*/raw-*.pt"):
        match = RAW_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("request"), int(match.group("layer")), int(match.group("qtile")))
        groups.setdefault(key, []).append((int(match.group("step")), path))

    configurations = [
        (estimator, rule, budget)
        for estimator in ("representation_drift", "oracle_one_step_output_drift")
        for rule in ("sum", "rss", "max", "decay")
        for budget in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    ]
    aggregates = {
        config: {"total": 0, "reused": 0, "step_errors": []}
        for config in configurations
    }
    for (request, layer, _), paths in groups.items():
        states: dict[
            tuple[str, str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]
        ] = {}
        for step, path in sorted(paths):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            current_m = payload["m"].float()[0, 0]
            current_l = payload["l"].float()[0, 0]
            current_u = payload["u"].float()[0, 0]
            tile_count = current_m.shape[0]
            if step == 0:
                for config in configurations:
                    states[config] = (
                        current_m.clone(),
                        current_l.clone(),
                        current_u.clone(),
                        np.zeros(tile_count, dtype=np.float32),
                    )
                continue
            indices = np.asarray(
                [lookup[(request, layer, step, tile)] for tile in range(tile_count)]
            )
            scores = {
                "representation_drift": (
                    data["q_max"][indices] + data["k_max"][indices] + data["v_max"][indices]
                ).astype(np.float32),
                "oracle_one_step_output_drift": reuse_effect(data)[indices].astype(
                    np.float32
                ),
            }
            current_aggregate = compose_attention_tile_statistics(
                [
                    AttentionTileStatistics(current_m[tile], current_l[tile], current_u[tile])
                    for tile in range(tile_count)
                ]
            ).normalized_output()
            for config in configurations:
                estimator, rule, budget = config
                cache_m, cache_l, cache_u, accumulated = states[config]
                score = scores[estimator]
                if rule == "sum":
                    proposed = accumulated + score
                elif rule == "rss":
                    proposed = np.sqrt(accumulated**2 + score**2)
                elif rule == "max":
                    proposed = np.maximum(accumulated, score)
                else:
                    proposed = 0.8 * accumulated + score
                reuse = proposed < budget
                aggregates[config]["total"] += tile_count
                aggregates[config]["reused"] += int(reuse.sum())
                if reuse.any():
                    reuse_t = torch.from_numpy(reuse)
                    mixed_m = torch.where(reuse_t[:, None], cache_m, current_m)
                    mixed_l = torch.where(reuse_t[:, None], cache_l, current_l)
                    mixed_u = torch.where(reuse_t[:, None, None], cache_u, current_u)
                    mixed = compose_attention_tile_statistics(
                        [
                            AttentionTileStatistics(mixed_m[tile], mixed_l[tile], mixed_u[tile])
                            for tile in range(tile_count)
                        ]
                    ).normalized_output()
                    actual = torch.linalg.vector_norm(mixed - current_aggregate, dim=-1) / (
                        torch.linalg.vector_norm(current_aggregate, dim=-1) + 1e-6
                    )
                    aggregates[config]["step_errors"].append(float(actual.max()))
                refreshed = torch.from_numpy(~reuse)
                cache_m[refreshed] = current_m[refreshed]
                cache_l[refreshed] = current_l[refreshed]
                cache_u[refreshed] = current_u[refreshed]
                accumulated = np.where(reuse, proposed, 0.0)
                states[config] = (cache_m, cache_l, cache_u, accumulated)

    rows: list[dict[str, object]] = []
    for (estimator, rule, budget), aggregate in aggregates.items():
        values = np.asarray(aggregate["step_errors"], dtype=np.float32)
        rows.append(
            {
                "estimator": estimator,
                "accumulation_rule": rule,
                "budget": budget,
                "decision_tiles": aggregate["total"],
                "reused_tiles": aggregate["reused"],
                "reuse_fraction": aggregate["reused"] / max(aggregate["total"], 1),
                "mixed_steps_with_reuse": len(values),
                "safe_mixed_step_fraction_at_1pct": float(np.mean(values <= 0.01))
                if values.size
                else 1.0,
                "median_mixed_attention_error": float(np.median(values)) if values.size else 0.0,
                "p99_mixed_attention_error": float(np.quantile(values, 0.99))
                if values.size
                else 0.0,
                "maximum_mixed_attention_error": float(values.max(initial=0.0)),
            }
        )
    return rows


def cache_model() -> dict[str, object]:
    sequence = 4096
    qtiles, kvtiles, layers, heads, rows, dim = 32, 64, 32, 32, 128, 128
    tiles_b1 = qtiles * kvtiles * layers * heads
    formats = {}
    for name, ml_bytes, u_bytes in (
        ("fp32_ml_bf16_u", 4, 2),
        ("bf16_ml_u", 2, 2),
        ("fp32_ml_fp8_u", 4, 1),
    ):
        per_tile = rows * (2 * ml_bytes + dim * u_bytes)
        formats[name] = {
            "bytes_per_tile": per_tile,
            "batch1_gib": tiles_b1 * per_tile / 2**30,
            "batch4_gib": 4 * tiles_b1 * per_tile / 2**30,
        }
    # One FP16 scale per query row is required to dequantize a row-wise INT8 u.
    # This is the most accurate compact format measured by
    # analyze_tile_reuse_precision.py; accounting for the scales avoids making
    # the compact-cache result look better than it is.
    int8_per_tile = rows * (2 * 2 + dim + 2)
    formats["bf16_ml_rowwise_int8_u"] = {
        "bytes_per_tile": int8_per_tile,
        "batch1_gib": tiles_b1 * int8_per_tile / 2**30,
        "batch4_gib": 4 * tiles_b1 * int8_per_tile / 2**30,
        "selected_layers_0_through_8_batch1_gib": (
            tiles_b1 * (9 / layers) * int8_per_tile / 2**30
        ),
        "selected_layers_0_through_8_batch4_gib": (
            4 * tiles_b1 * (9 / layers) * int8_per_tile / 2**30
        ),
    }
    query_bytes = rows * dim * 2
    return {
        "geometry": {
            "sequence": sequence,
            "query_tiles": qtiles,
            "kv_tiles": kvtiles,
            "layers": layers,
            "heads": heads,
            "physical_tiles_batch1": tiles_b1,
        },
        "formats": formats,
        "query_tile_output_bf16": {
            "bytes_per_query_tile": query_bytes,
            "batch1_gib": qtiles * layers * heads * query_bytes / 2**30,
            "batch4_gib": 4 * qtiles * layers * heads * query_bytes / 2**30,
        },
        "per_reused_tile": {
            "qk_flops_avoided": 2 * rows * 64 * dim,
            "pv_flops_avoided": 2 * rows * 64 * dim,
            "k_bytes_avoided_bf16": 64 * dim * 2,
            "v_bytes_avoided_bf16": 64 * dim * 2,
            "cache_bytes_read_fp32_ml_bf16_u": rows * (2 * 4 + dim * 2),
            "cache_bytes_read_bf16_ml_u": rows * (2 * 2 + dim * 2),
            "cache_bytes_read_fp32_ml_fp8_u": rows * (2 * 4 + dim),
            "cache_bytes_read_bf16_ml_rowwise_int8_u": int8_per_tile,
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, split_map = load_summaries(args.trace_dir)
    split = np.asarray([split_map[str(value)] for value in data["request_id"]])
    oracle = grouped_oracle_rows(data, split)
    predictor_policies, correlations, calibrated = evaluate_predictors(data, split)
    direct = state_and_region_policies(data, split)
    query = query_tile_analysis(data, split)
    refresh = raw_refresh_analysis(args.trace_dir)
    cumulative = raw_cumulative_analysis(args.trace_dir, data)
    write_csv(args.output_dir / "oracle_reuse.csv", oracle)
    write_csv(args.output_dir / "predictor_policies.csv", predictor_policies)
    write_csv(args.output_dir / "state_region_policies.csv", direct)
    write_csv(args.output_dir / "query_tile_reuse.csv", query)
    write_csv(args.output_dir / "fixed_refresh.csv", refresh)
    write_csv(args.output_dir / "cumulative_budgets.csv", cumulative)
    (args.output_dir / "calibrated_thresholds.json").write_text(
        json.dumps(calibrated, indent=2) + "\n", encoding="utf-8"
    )
    heldout = split == "heldout"
    error = reuse_effect(data)
    summary = {
        "trace_dir": str(args.trace_dir),
        "records": len(error),
        "contexts": len(np.unique(data["request_id"])),
        "contexts_by_split": {
            name: len(np.unique(data["request_id"][split == name])) for name in np.unique(split)
        },
        "oracle_reuse_overall": {
            str(threshold): float(np.mean(error <= threshold)) for threshold in ORACLE_THRESHOLDS
        },
        "oracle_reuse_heldout": {
            str(threshold): float(np.mean(error[heldout] <= threshold))
            for threshold in ORACLE_THRESHOLDS
        },
        "reuse_effect_quantiles": {
            str(q): float(np.quantile(error, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "predictor_correlations": correlations,
        "best_heldout_predictor_policies": sorted(
            [row for row in predictor_policies if row["split"] == "heldout"],
            key=lambda row: (row["reuse_fraction"], row["prediction_precision"]),
            reverse=True,
        )[:20],
        "query_tile_reuse": query,
        "fixed_refresh": refresh,
        "best_cumulative_budgets": sorted(
            cumulative,
            key=lambda row: (
                row["safe_mixed_step_fraction_at_1pct"] >= 0.99,
                row["reuse_fraction"],
            ),
            reverse=True,
        )[:20],
        "cache_model": cache_model(),
        "oracle_gate_pass": float(np.mean(error[heldout] <= 0.01)) >= 0.30,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
