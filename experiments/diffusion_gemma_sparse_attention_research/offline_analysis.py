"""Reanalyse preserved dense-trajectory snapshots without running the model.

The source shards were collected by ``diffusion_gemma_blasst_diagnosis``.
This module never changes a routing implementation.  It treats exact current
QK-derived quantities as labels/oracles and explicitly marks signals that are
available before the current exact QK.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import normaltest, rankdata, spearmanr


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results/diffusion_gemma_blasst_diagnosis/same_state"
SOURCE_SUMMARY = REPO / "results/diffusion_gemma_blasst_diagnosis/same_state_summary.json"
CONTROLLED = REPO / "results/diffusion_gemma_solattn_blasst_multibench_controlled"
OUTPUT = REPO / "results/diffusion_gemma_sparse_attention_research"
BETAS = {
    25: -0.6744897501960817,
    50: 0.0,
    75: 0.6744897501960817,
    90: 1.2815515655446004,
}
TARGETS = (0.25, 0.50, 0.75, 0.90)
LAMBDA_GRID = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9995, 1.0)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware ROC AUC without requiring scikit-learn."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    present = ~np.isnan(scores)
    labels, scores = labels[present], scores[present]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    present = ~np.isnan(scores)
    labels, scores = labels[present], scores[present]
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def mass_support(mass: np.ndarray, fraction: float = 0.90) -> np.ndarray:
    """Smallest stable-ranked tile set whose aggregate dense mass reaches fraction."""
    mass = np.asarray(mass, dtype=float)
    result = np.zeros(len(mass), dtype=bool)
    total = float(mass.sum())
    if not len(mass) or total <= 0:
        return result
    order = np.argsort(-mass, kind="stable")
    cutoff = int(np.searchsorted(np.cumsum(mass[order]), fraction * total, side="left"))
    result[order[: min(cutoff + 1, len(order))]] = True
    return result


def top_budget_mask(scores: np.ndarray, target_skip: float) -> np.ndarray:
    """Select a deterministic top-k set at a requested physical-tile budget."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return np.zeros(0, dtype=bool)
    keep_count = max(1, int(math.ceil((1.0 - target_skip) * len(scores))))
    order = np.argsort(-scores, kind="stable")
    keep = np.zeros(len(scores), dtype=bool)
    keep[order[:keep_count]] = True
    return keep


def online_margins(maxima: np.ndarray, valid: np.ndarray) -> np.ndarray:
    previous = np.concatenate(
        [np.full((len(maxima), 1), -np.inf), np.maximum.accumulate(maxima, axis=-1)[:, :-1]], axis=-1
    )
    with np.errstate(invalid="ignore"):
        margins = maxima - previous
    return np.where(valid, margins, -np.inf)


def physical_online_score(maxima: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Maximum current-QK BLASST margin over valid rows in a physical tile."""
    return online_margins(maxima, valid).max(axis=0)


def physical_online_mask(maxima: np.ndarray, valid: np.ndarray, lam: float) -> np.ndarray:
    threshold = math.log(lam)
    row_keep = valid & (online_margins(maxima, valid) >= threshold)
    return row_keep.any(axis=0) & valid.any(axis=0)


def normalized_tile_entropy(mass: np.ndarray, valid: np.ndarray) -> float:
    entropies: list[float] = []
    for row_mass, row_valid in zip(mass, valid):
        values = row_mass[row_valid]
        total = float(values.sum())
        if len(values) < 2 or total <= 0:
            continue
        probabilities = values / total
        entropies.append(float(-(probabilities * np.log(probabilities.clip(1e-30))).sum() / math.log(len(values))))
    return float(np.mean(entropies)) if entropies else float("nan")


def neighbor_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    if len(values) == 1:
        result[:] = values
        return result
    result[0], result[-1] = values[1], values[-2]
    if len(values) > 2:
        result[1:-1] = (values[:-2] + values[2:]) / 2
    return result


def query_union_curve(mass: np.ndarray, valid: np.ndarray, target_skip: float, group_sizes: Iterable[int]) -> dict:
    """Measure row-logical versus contiguous-query-group physical sparsity."""
    row_keep = np.zeros_like(valid, dtype=bool)
    logical_eligible = logical_skipped = 0
    for row in range(len(mass)):
        indices = np.flatnonzero(valid[row])
        if not len(indices):
            continue
        selected = top_budget_mask(mass[row, indices], target_skip)
        row_keep[row, indices[selected]] = True
        logical_eligible += len(indices)
        logical_skipped += len(indices) - int(selected.sum())
    result = {
        "logical": {
            "eligible": logical_eligible,
            "skipped": logical_skipped,
            "sparsity": _safe_ratio(logical_skipped, logical_eligible),
        }
    }
    for size in group_sizes:
        eligible_count = skipped_count = 0
        for start in range(0, len(mass), size):
            group_valid = valid[start : start + size].any(axis=0)
            group_keep = row_keep[start : start + size].any(axis=0)
            eligible_count += int(group_valid.sum())
            skipped_count += int((group_valid & ~group_keep).sum())
        result[str(size)] = {
            "eligible": eligible_count,
            "skipped": skipped_count,
            "sparsity": _safe_ratio(skipped_count, eligible_count),
        }
    return result


@dataclass
class Snapshot:
    example_id: str
    split: str
    attention_type: str
    layer: int
    call: int
    head: int
    query_start: int
    query_length: int
    kv_length: int
    denoising_step: int
    eligible: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    proxy: np.ndarray
    mass: np.ndarray
    maxima: np.ndarray
    physical_margin: np.ndarray
    tile_mass: np.ndarray
    entropy: float
    prefix_length: int
    previous_proxy: np.ndarray | None = None
    previous_mass: np.ndarray | None = None
    previous_margin: np.ndarray | None = None
    prior_mass_mean: np.ndarray | None = None


def load_snapshots(source: Path = SOURCE) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for shard_dir in sorted((source / "shards").iterdir()):
        if not shard_dir.is_dir():
            continue
        records = json.loads((shard_dir / "diagnostics.json").read_text())
        local: list[Snapshot] = []
        with np.load(shard_dir / "snapshots.npz") as arrays:
            for record in records:
                key = record["snapshot_key"]
                counts = arrays[key + "_counts"]
                valid = counts > 0
                eligible = valid.any(axis=0)
                mass = arrays[key + "_mass"].astype(np.float64)
                maxima = arrays[key + "_maxima"].astype(np.float64)
                local.append(
                    Snapshot(
                        example_id=record["example_id"],
                        split=record["split"],
                        attention_type=record["attention_type"],
                        layer=int(record["layer"]),
                        call=int(record["call"]),
                        head=int(record["head"]),
                        query_start=int(record["query_start"]),
                        query_length=int(record["query_length"]),
                        kv_length=int(record["kv_length"]),
                        denoising_step=int(record.get("metadata", {}).get("denoising_step", record["call"])),
                        eligible=eligible,
                        valid=valid,
                        counts=counts,
                        proxy=arrays[key + "_proxy"].astype(np.float64),
                        mass=mass,
                        maxima=maxima,
                        physical_margin=physical_online_score(maxima, valid),
                        tile_mass=mass.sum(axis=0),
                        entropy=normalized_tile_entropy(mass, valid),
                        prefix_length=max(0, int(record["kv_length"]) - int(record["query_length"])),
                    )
                )
        histories: dict[tuple[int, int, int, int], list[Snapshot]] = defaultdict(list)
        for item in local:
            histories[(item.layer, item.head, item.query_start, item.kv_length)].append(item)
        for sequence in histories.values():
            sequence.sort(key=lambda item: item.call)
            mass_history: list[np.ndarray] = []
            for index, item in enumerate(sequence):
                if index:
                    previous = sequence[index - 1]
                    if len(previous.proxy) == len(item.proxy):
                        item.previous_proxy = previous.proxy.copy()
                        item.previous_mass = previous.tile_mass.copy()
                        item.previous_margin = previous.physical_margin.copy()
                        item.prior_mass_mean = np.mean(mass_history, axis=0)
                mass_history.append(item.tile_mass.copy())
        snapshots.extend(local)
    return snapshots


def _region_scores(item: Snapshot) -> dict[str, np.ndarray]:
    starts = np.arange(len(item.proxy)) * 64
    centers = starts + 31.5
    length_scale = max(item.kv_length, 64)
    boundary_proximity = -np.abs(centers - item.prefix_length) / length_scale
    prefix = (starts < item.prefix_length).astype(float)
    canvas = (starts >= item.prefix_length).astype(float)
    recency = centers / length_scale
    return {
        "boundary_proximity": boundary_proximity,
        "prefix_indicator": prefix,
        "canvas_indicator": canvas,
        "absolute_recency": recency,
    }


def _signal_scores(item: Snapshot) -> dict[str, np.ndarray | None]:
    eligible_proxy = item.proxy[item.eligible]
    z = (eligible_proxy - eligible_proxy.mean()) / max(float(eligible_proxy.std()), 1e-6)
    full_z = np.full(len(item.proxy), np.nan)
    full_z[item.eligible] = z
    signals: dict[str, np.ndarray | None] = {
        "sol_proxy": full_z,
        "kv_neighbor_proxy": neighbor_mean(item.proxy),
        "previous_step_mass": item.previous_mass,
        "previous_step_proxy": item.previous_proxy,
        "previous_step_blasst_margin": item.previous_margin,
        "prior_frequency_mass": item.prior_mass_mean,
        "current_blasst_margin_oracle": item.physical_margin,
        "current_dense_mass_oracle": item.tile_mass,
    }
    signals.update(_region_scores(item))
    return signals


def distribution_analysis(snapshots: list[Snapshot]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    rows: list[dict] = []
    for item in snapshots:
        values = item.proxy[item.eligible]
        if len(values) < 2:
            continue
        z = (values - values.mean()) / max(float(values.std()), 1e-6)
        row = {
            "example_id": item.example_id,
            "split": item.split,
            "attention_type": item.attention_type,
            "layer": item.layer,
            "call": item.call,
            "denoising_step": item.denoising_step,
            "eligible_tiles": len(values),
            "skew": float(np.mean(z**3)),
            "excess_kurtosis": float(np.mean(z**4) - 3),
            "normaltest_p": float(normaltest(z).pvalue) if len(z) >= 8 else None,
        }
        for target, beta in BETAS.items():
            retained = int((z >= beta).sum())
            achieved_skip = 1.0 - retained / len(z)
            row[f"s{target}_actual"] = achieved_skip
            row[f"s{target}_error"] = achieved_skip - target / 100
        rows.append(row)
        for key in (f"{item.split}|overall", f"{item.split}|{item.attention_type}"):
            groups[key].append(row)
    summary = {}
    for key, entries in groups.items():
        result = {
            "rows": len(entries),
            "eligible_tiles": int(sum(e["eligible_tiles"] for e in entries)),
            "mean_skew": float(np.mean([e["skew"] for e in entries])),
            "median_skew": float(np.median([e["skew"] for e in entries])),
            "mean_excess_kurtosis": float(np.mean([e["excess_kurtosis"] for e in entries])),
            "normaltest_rejection_rate_p05": _safe_ratio(
                sum(e["normaltest_p"] < 0.05 for e in entries if e["normaltest_p"] is not None),
                sum(e["normaltest_p"] is not None for e in entries),
            ),
        }
        for target in BETAS:
            weight = sum(e["eligible_tiles"] for e in entries)
            actual = sum(e[f"s{target}_actual"] * e["eligible_tiles"] for e in entries) / weight
            result[f"s{target}"] = {
                "target": target / 100,
                "actual": actual,
                "error": actual - target / 100,
                "row_error_std": float(np.std([e[f"s{target}_error"] for e in entries])),
            }
        summary[key] = result
    strata: dict[str, dict] = {}
    dimensions = {
        "attention_type": lambda row: row["attention_type"],
        "layer": lambda row: str(row["layer"]),
        "head": lambda row: str(next(s.head for s in snapshots if s.example_id == row["example_id"])),
        "denoising_step": lambda row: str(row["denoising_step"]),
        "query_block": lambda row: str(next(s.query_start // 64 for s in snapshots if s.example_id == row["example_id"] and s.layer == row["layer"] and s.call == row["call"])),
    }
    final_rows = [row for row in rows if row["split"] == "final"]
    for dimension, getter in dimensions.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in final_rows:
            grouped[getter(row)].append(row)
        strata[dimension] = {}
        for value, entries in grouped.items():
            total = sum(entry["eligible_tiles"] for entry in entries)
            strata[dimension][value] = {
                "rows": len(entries),
                "eligible_tiles": total,
                "mean_skew": float(np.mean([entry["skew"] for entry in entries])),
                "mean_excess_kurtosis": float(np.mean([entry["excess_kurtosis"] for entry in entries])),
                "density_error": {
                    f"s{target}": float(
                        sum(entry[f"s{target}_error"] * entry["eligible_tiles"] for entry in entries) / total
                    )
                    for target in BETAS
                },
            }
    return {"summary": summary, "strata": strata, "rows": rows}


def signal_analysis(snapshots: list[Snapshot], split: str = "final", history_only: bool = False) -> dict:
    by_signal: dict[str, list[dict]] = defaultdict(list)
    for item in snapshots:
        if item.split != split or (history_only and item.previous_mass is None):
            continue
        mass = item.tile_mass[item.eligible]
        positives = mass_support(mass, 0.90)
        for name, full_score in _signal_scores(item).items():
            if full_score is None:
                continue
            score = np.asarray(full_score)[item.eligible]
            if len(score) != len(mass) or np.isnan(score).any():
                continue
            entry = {
                "example_id": item.example_id,
                "attention_type": item.attention_type,
                "n": len(mass),
                "mass": mass,
                "positive": positives,
                "score": score,
                "spearman": _finite_spearman(score, mass),
                "auc": binary_auc(positives, score),
                "average_precision": average_precision(positives, score),
            }
            by_signal[name].append(entry)

    output: dict[str, dict] = {}
    for signal, entries in by_signal.items():
        signal_result: dict[str, dict] = {}
        for scope in ("overall", "local", "global"):
            scoped = [e for e in entries if scope == "overall" or e["attention_type"] == scope]
            if not scoped:
                continue
            metrics = {
                "records": len(scoped),
                "prompts": len({e["example_id"] for e in scoped}),
                "tiles": int(sum(e["n"] for e in scoped)),
                "mean_spearman": float(np.mean([e["spearman"] for e in scoped if e["spearman"] is not None])),
                "mean_roc_auc": float(np.mean([e["auc"] for e in scoped if e["auc"] is not None])),
                "mean_average_precision": float(
                    np.mean([e["average_precision"] for e in scoped if e["average_precision"] is not None])
                ),
                "micro_roc_auc": binary_auc(
                    np.concatenate([e["positive"] for e in scoped]), np.concatenate([e["score"] for e in scoped])
                ),
                "micro_average_precision": average_precision(
                    np.concatenate([e["positive"] for e in scoped]), np.concatenate([e["score"] for e in scoped])
                ),
                "budgets": {},
            }
            for target in TARGETS:
                kept_mass = total_mass = positives = missed = kept = eligible = max_hits = 0
                prompt_mass: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
                for entry in scoped:
                    mask = top_budget_mask(entry["score"], target)
                    kept_mass += float(entry["mass"][mask].sum())
                    total_mass += float(entry["mass"].sum())
                    positives += int(entry["positive"].sum())
                    missed += int((entry["positive"] & ~mask).sum())
                    kept += int(mask.sum())
                    eligible += len(mask)
                    max_hits += int(mask[int(np.argmax(entry["mass"]))])
                    prompt_mass[entry["example_id"]][0] += float(entry["mass"][mask].sum())
                    prompt_mass[entry["example_id"]][1] += float(entry["mass"].sum())
                metrics["budgets"][f"s{int(target * 100)}"] = {
                    "target_sparsity": target,
                    "actual_sparsity": 1.0 - kept / eligible,
                    "retained_mass": kept_mass / total_mass,
                    "mass_support_false_negative_rate": _safe_ratio(missed, positives),
                    "highest_mass_tile_recall": max_hits / len(scoped),
                    "prompt_retained_mass_mean": float(np.mean([n / d for n, d in prompt_mass.values()])),
                    "prompt_retained_mass_std": float(np.std([n / d for n, d in prompt_mass.values()])),
                }
            signal_result[scope] = metrics
        output[signal] = signal_result
    return output


def proxy_alignment_analysis(snapshots: list[Snapshot], split: str = "final") -> dict:
    output: dict[str, dict] = {}
    for scope in ("overall", "local", "global"):
        selected = [
            item for item in snapshots
            if item.split == split and (scope == "overall" or item.attention_type == scope)
        ]
        rows = []
        for item in selected:
            proxy = item.proxy[item.eligible]
            mass = item.tile_mass[item.eligible]
            exact_max = item.maxima[:, item.eligible].max(axis=0)
            mass_positive = mass_support(mass, 0.90)
            max_positive = exact_max >= np.quantile(exact_max, 0.90)
            rows.append({
                "mass_spearman": _finite_spearman(proxy, mass),
                "max_spearman": _finite_spearman(proxy, exact_max),
                "mass_auc": binary_auc(mass_positive, proxy),
                "max_auc": binary_auc(max_positive, proxy),
                "drops_highest_mass": int(np.argmax(proxy) != np.argmax(mass) and proxy[np.argmax(mass)] < np.median(proxy)),
                "drops_highest_max": int(proxy[np.argmax(exact_max)] < np.median(proxy)),
            })
        output[scope] = {
            "records": len(rows),
            "proxy_mass_spearman_mean": float(np.mean([r["mass_spearman"] for r in rows if r["mass_spearman"] is not None])),
            "proxy_exact_max_spearman_mean": float(np.mean([r["max_spearman"] for r in rows if r["max_spearman"] is not None])),
            "mass_support_roc_auc_mean": float(np.mean([r["mass_auc"] for r in rows if r["mass_auc"] is not None])),
            "top_decile_exact_max_roc_auc_mean": float(np.mean([r["max_auc"] for r in rows if r["max_auc"] is not None])),
            "highest_mass_below_proxy_median_rate": float(np.mean([r["drops_highest_mass"] for r in rows])),
            "highest_exact_max_below_proxy_median_rate": float(np.mean([r["drops_highest_max"] for r in rows])),
        }
    return output


def entropy_lambda_analysis(snapshots: list[Snapshot], split: str = "final") -> dict:
    output: dict[str, dict] = {}
    for attention_type in ("local", "global"):
        selected = [s for s in snapshots if s.split == split and s.attention_type == attention_type and np.isfinite(s.entropy)]
        cuts = np.quantile([s.entropy for s in selected], [1 / 3, 2 / 3])
        bins = {
            "low": [s for s in selected if s.entropy <= cuts[0]],
            "middle": [s for s in selected if cuts[0] < s.entropy <= cuts[1]],
            "high": [s for s in selected if s.entropy > cuts[1]],
        }
        output[attention_type] = {"entropy_tercile_boundaries": cuts.tolist(), "bins": {}}
        for bin_name, entries in bins.items():
            result = {
                "records": len(entries),
                "entropy_mean": float(np.mean([e.entropy for e in entries])),
                "lambdas": {},
            }
            for lam in LAMBDA_GRID:
                eligible = skipped = 0
                retained_mass = total_mass = 0.0
                for item in entries:
                    keep = physical_online_mask(item.maxima, item.valid, lam)
                    eligible += int(item.eligible.sum())
                    skipped += int((item.eligible & ~keep).sum())
                    retained_mass += float(item.tile_mass[keep].sum())
                    total_mass += float(item.tile_mass[item.eligible].sum())
                result["lambdas"][f"{lam:g}"] = {
                    "physical_sparsity": _safe_ratio(skipped, eligible),
                    "retained_mass": _safe_ratio(retained_mass, total_mass),
                }
            output[attention_type]["bins"][bin_name] = result
    return output


def query_overlap_analysis(snapshots: list[Snapshot], split: str = "final") -> dict:
    aggregate: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for item in snapshots:
        if item.split != split:
            continue
        for target in (0.50, 0.75, 0.90):
            curve = query_union_curve(item.mass, item.valid, target, (1, 8, 16, 32, 64))
            for group, values in curve.items():
                key = f"{item.attention_type}|s{int(target * 100)}"
                aggregate[key][group]["eligible"] += values["eligible"]
                aggregate[key][group]["skipped"] += values["skipped"]
    output = {}
    for key, groups in aggregate.items():
        output[key] = {}
        for group, values in groups.items():
            output[key][group] = {
                "eligible": int(values["eligible"]),
                "skipped": int(values["skipped"]),
                "sparsity": _safe_ratio(values["skipped"], values["eligible"]),
            }
    return output


def _merge_kv_tiles(maxima: np.ndarray, valid: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor == 1:
        return maxima, valid
    padding = (-maxima.shape[-1]) % factor
    merged_maxima = np.pad(maxima, ((0, 0), (0, padding)), constant_values=-np.inf)
    merged_valid = np.pad(valid, ((0, 0), (0, padding)), constant_values=False)
    shape = (len(maxima), (maxima.shape[-1] + padding) // factor, factor)
    merged_maxima = np.where(merged_valid.reshape(shape), merged_maxima.reshape(shape), -np.inf).max(-1)
    merged_valid = merged_valid.reshape(shape).any(-1)
    return merged_maxima, merged_valid


def _blasst_group_counts(maxima: np.ndarray, valid: np.ndarray, q_group: int, lam: float) -> tuple[int, int]:
    row_keep = valid & (online_margins(maxima, valid) >= math.log(lam))
    eligible_count = skipped_count = 0
    for start in range(0, len(maxima), q_group):
        eligible = valid[start : start + q_group].any(0)
        keep = row_keep[start : start + q_group].any(0)
        eligible_count += int(eligible.sum())
        skipped_count += int((eligible & ~keep).sum())
    return eligible_count, skipped_count


def blasst_tile_shape_analysis(snapshots: list[Snapshot], split: str = "final") -> dict:
    """Reblock exact 64-KV maxima into larger KV tiles and query groups."""
    output: dict[str, dict] = {}
    for attention_type in ("local", "global"):
        selected = [s for s in snapshots if s.split == split and s.attention_type == attention_type]
        median_length = float(np.median([s.kv_length for s in selected]))
        length_groups = {
            "all": selected,
            "shorter_or_equal_median": [s for s in selected if s.kv_length <= median_length],
            "longer_than_median": [s for s in selected if s.kv_length > median_length],
        }
        output[attention_type] = {"median_kv_length": median_length, "length_groups": {}}
        for length_name, entries in length_groups.items():
            if not entries:
                continue
            values = {}
            for kv_size in (64, 128, 256):
                for q_size in (1, 8, 16, 32, 64):
                    for lam in (0.5, 0.9, 1.0):
                        eligible = skipped = 0
                        for item in entries:
                            maxima, valid = _merge_kv_tiles(item.maxima, item.valid, kv_size // 64)
                            e, s = _blasst_group_counts(maxima, valid, q_size, lam)
                            eligible += e
                            skipped += s
                        values[f"q{q_size}_k{kv_size}_l{lam:g}"] = {
                            "eligible_tiles": eligible,
                            "skipped_tiles": skipped,
                            "physical_sparsity": _safe_ratio(skipped, eligible),
                        }
            output[attention_type]["length_groups"][length_name] = {
                "records": len(entries),
                "min_kv_length": min(e.kv_length for e in entries),
                "max_kv_length": max(e.kv_length for e in entries),
                "configurations": values,
            }
    return output


def modeled_compute(snapshots: list[Snapshot], signal_results: dict, split: str = "final") -> dict:
    """Analytic GEMM-only bounds; these are not timings or a kernel model."""
    selected = [s for s in snapshots if s.split == split]
    qk_flops = pv_flops = 0.0
    by_type = defaultdict(lambda: [0.0, 0.0])
    for item in selected:
        head_dim = 512 if item.attention_type == "global" else 256
        valid_elements = float(item.counts.sum())
        qk_flops += valid_elements * 2 * head_dim
        pv_flops += valid_elements * 2 * head_dim
        by_type[item.attention_type][0] += valid_elements * 2 * head_dim
        by_type[item.attention_type][1] += valid_elements * 2 * head_dim
    bounds = {}
    for method in ("sol_proxy", "previous_step_mass", "current_blasst_margin_oracle"):
        if method not in signal_results or "overall" not in signal_results[method]:
            continue
        bounds[method] = {}
        for name, metrics in signal_results[method]["overall"]["budgets"].items():
            sparsity = metrics["actual_sparsity"]
            bounds[method][name] = {
                "physical_sparsity": sparsity,
                "dense_qk_fraction": 0.0 if method != "current_blasst_margin_oracle" else 1.0,
                "ideal_remaining_qk_fraction": 1.0 - sparsity if method != "current_blasst_margin_oracle" else 1.0,
                "ideal_remaining_pv_fraction": 1.0 - sparsity,
                "ideal_remaining_qk_plus_pv_fraction": (1.0 - sparsity) if method != "current_blasst_margin_oracle" else 1.0 - sparsity / 2,
            }
    return {
        "status": "analytic GEMM-only fraction; no latency, softmax, bandwidth, launch, or router-overhead measurement",
        "identity": "dense QK and dense PV have equal multiply-add counts for equal head dimensions; BLASST always pays dense current QK",
        "sampled_dense_flops": {
            "qk": qk_flops,
            "pv": pv_flops,
            "qk_plus_pv": qk_flops + pv_flops,
            "by_attention_type": {
                key: {"qk": value[0], "pv": value[1], "qk_plus_pv": sum(value)} for key, value in by_type.items()
            },
            "scope": "one sampled head and query block per preserved snapshot; exact structural valid-token counts",
        },
        "bounds": bounds,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(payload: dict, output: Path = OUTPUT) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    distribution = payload["distribution"]["summary"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for scope, marker in (("overall", "o"), ("local", "s"), ("global", "^")):
        values = distribution[f"final|{scope}"]
        x = [target / 100 for target in BETAS]
        y = [values[f"s{target}"]["actual"] for target in BETAS]
        ax.plot(x, y, marker=marker, label=scope)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    ax.set(xlabel="Gaussian target sparsity", ylabel="Measured physical-tile sparsity", xlim=(0.2, 0.93), ylim=(0.2, 0.93))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "01_sol_conditional_density_calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    rows = payload["distribution"]["rows"]
    final_rows = [r for r in rows if r["split"] == "final"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for axis, metric, title in zip(axes, ("skew", "excess_kurtosis"), ("Skewness", "Excess kurtosis")):
        data = [[r[metric] for r in final_rows if r["attention_type"] == typ] for typ in ("local", "global")]
        axis.boxplot(data, tick_labels=("local", "global"), showfliers=False)
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "02_sol_conditional_shape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    signals = payload["signals"]
    plotted = [name for name in ("sol_proxy", "previous_step_mass", "kv_neighbor_proxy", "current_blasst_margin_oracle", "current_dense_mass_oracle") if name in signals]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in plotted:
        budgets = signals[name]["overall"]["budgets"]
        x = [budgets[f"s{int(t*100)}"]["actual_sparsity"] for t in TARGETS]
        y = [budgets[f"s{int(t*100)}"]["retained_mass"] for t in TARGETS]
        ax.plot(x, y, marker="o", label=name.replace("_", " "))
    ax.set(xlabel="Actual physical-tile sparsity", ylabel="Retained dense tile mass")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "03_supervisor_mass_budget_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    overlap = payload["query_overlap"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for axis, typ in zip(axes, ("local", "global")):
        for target in (50, 75, 90):
            values = overlap[f"{typ}|s{target}"]
            groups = [1, 8, 16, 32, 64]
            axis.plot(groups, [values[str(g)]["sparsity"] for g in groups], marker="o", label=f"row target {target}%")
        axis.set_title(typ)
        axis.set_xlabel("Contiguous query-group size")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Executable group-by-KV tile sparsity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "04_query_union_bucket_effect.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    entropy = payload["entropy_lambda"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for axis, typ in zip(axes, ("local", "global")):
        for bin_name in ("low", "middle", "high"):
            values = entropy[typ]["bins"][bin_name]["lambdas"]
            x = list(LAMBDA_GRID)
            y = [values[f"{lam:g}"]["physical_sparsity"] for lam in x]
            axis.plot(x, y, marker="o", label=f"{bin_name} entropy")
        axis.set_xscale("log")
        axis.set_title(typ)
        axis.set_xlabel("lambda")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Physical-tile sparsity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "05_entropy_threshold_heterogeneity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    tile_shapes = payload["blasst_tile_shapes"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for axis, typ in zip(axes, ("local", "global")):
        configurations = tile_shapes[typ]["length_groups"]["all"]["configurations"]
        for kv_size in (64, 128, 256):
            q_sizes = (1, 8, 16, 32, 64)
            values = [configurations[f"q{q}_k{kv_size}_l1"]["physical_sparsity"] for q in q_sizes]
            axis.plot(q_sizes, values, marker="o", label=f"KV tile {kv_size}")
        axis.set_title(typ)
        axis.set_xlabel("Query group size")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("BLASST physical sparsity at lambda=1")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "09_blasst_q_k_tile_shape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output)))

    controlled = payload.get("controlled_transfer", [])
    if controlled:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
        for axis, method in zip(axes, ("sol", "blasst")):
            for benchmark in sorted({r["benchmark"] for r in controlled}):
                data = sorted([r for r in controlled if r["benchmark"] == benchmark and r["method"] == method], key=lambda r: r["target_sparsity"])
                axis.plot([r["target_sparsity"] for r in data], [r["actual_sparsity"] for r in data], marker="o", label=benchmark)
            axis.plot([0.2, 0.95], [0.2, 0.95], "k--", linewidth=1)
            axis.set_title(method)
            axis.set_xlabel("Target sparsity")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Actual physical-tile sparsity")
        axes[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        path = figure_dir / "06_multibench_threshold_transfer.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path.relative_to(output)))
    return written


def controlled_transfer_rows(path: Path = CONTROLLED / "summary.json") -> list[dict]:
    payload = json.loads(path.read_text())
    # Accommodate both the canonical list and a dict-wrapped schema.
    rows = payload if isinstance(payload, list) else payload.get(
        "rows", payload.get("conditions", payload.get("results", []))
    )
    output = []
    for row in rows:
        condition = row.get("condition", "")
        if condition == "dense":
            continue
        method = "sol" if condition.startswith("sol_") else "blasst" if condition.startswith("blasst_") else None
        if method is None:
            continue
        target = row.get("target_sparsity")
        actual = row.get("actual_sparsity", row.get("full_tile_sparsity"))
        if target is None:
            for value in (25, 50, 75, 90):
                if f"s{value}" in condition:
                    target = value / 100
                    break
        output.append({
            "benchmark": row["benchmark"],
            "condition": condition,
            "method": method,
            "target_sparsity": float(target),
            "actual_sparsity": float(actual),
            "accuracy": row.get("score", row.get("accuracy")),
            "retained_mass": row.get("retained_attention_mass", row.get("dense_path_retained_mass")),
        })
    return output


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(source)
    assert len(snapshots) == 2100, f"expected 2100 snapshots, found {len(snapshots)}"
    distribution = distribution_analysis(snapshots)
    signals = signal_analysis(snapshots)
    signals_history_matched = signal_analysis(snapshots, history_only=True)
    proxy_alignment = proxy_alignment_analysis(snapshots)
    entropy = entropy_lambda_analysis(snapshots)
    overlap = query_overlap_analysis(snapshots)
    tile_shapes = blasst_tile_shape_analysis(snapshots)
    transfer = controlled_transfer_rows()
    payload = {
        "protocol": {
            "source": str(source.relative_to(REPO)),
            "source_trajectory": "native dense",
            "samples": len({s.example_id for s in snapshots}),
            "snapshots": len(snapshots),
            "primary_split": "final (8 prompts); calibration split retained separately",
            "tile_shape": [64, 64],
            "mass_label": "exact dense probability mass summed across the sampled valid query rows",
            "positive_label": "smallest exact tile-mass-ranked support reaching 90% aggregate mass per snapshot",
            "budgeting": "deterministic top-k physical tiles per snapshot; at least one tile retained",
            "causal_labels": {
                "pre_current_qk": ["sol_proxy", "kv_neighbor_proxy", "previous_step_mass", "previous_step_proxy", "previous_step_blasst_margin", "prior_frequency_mass", "position signals"],
                "post_current_qk_oracle": ["current_blasst_margin_oracle", "current_dense_mass_oracle", "dense tile entropy"],
            },
            "limitations": [
                "one sampled head and one 64-query block per prompt, though all 16 heads are covered across prompts",
                "RULER-only same-state snapshots; multibench transfer uses aggregate controlled-run sufficient statistics",
                "no token-level scores for alternate KV tile sizes",
                "no persisted value vectors/norms or per-tile contribution vectors",
                "analytic compute fractions are not latency measurements",
            ],
        },
        "distribution": distribution,
        "signals": signals,
        "signals_history_matched": signals_history_matched,
        "proxy_alignment": proxy_alignment,
        "entropy_lambda": entropy,
        "query_overlap": overlap,
        "blasst_tile_shapes": tile_shapes,
        "controlled_transfer": transfer,
    }
    payload["modeled_compute"] = modeled_compute(snapshots, signals)
    payload["figures"] = make_plots(payload, output)
    (output / "diagnostic_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(output / "conditional_proxy_rows.csv", distribution["rows"])
    _write_csv(output / "controlled_threshold_transfer.csv", transfer)
    audit = {
        "complete": True,
        "snapshots": len(snapshots),
        "samples": len({s.example_id for s in snapshots}),
        "final_samples": len({s.example_id for s in snapshots if s.split == "final"}),
        "attention_types": sorted({s.attention_type for s in snapshots}),
        "layers": sorted({s.layer for s in snapshots}),
        "heads": sorted({s.head for s in snapshots}),
        "figures": payload["figures"],
        "dense_trajectory_only": True,
        "no_router_or_model_changes": True,
    }
    (output / "diagnostic_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = run(args.source, args.output)
    print(json.dumps({"snapshots": payload["protocol"]["snapshots"], "figures": payload["figures"]}, indent=2))


if __name__ == "__main__":
    main()
