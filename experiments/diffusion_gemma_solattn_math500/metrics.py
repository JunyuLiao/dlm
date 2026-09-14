"""Count-weighted routing and paired generation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate_calls(stats_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    calls = []
    for stats in stats_rows:
        calls.extend(item for item in stats.get("per_call", []) if isinstance(item, Mapping))

    def group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        total = sum(int(row.get("physical_total_tiles", row.get("physical_candidate_tiles", 0))) for row in rows)
        routed = sum(int(row.get("physical_candidate_tiles", 0)) for row in rows)
        skipped = sum(int(row.get("physical_skipped_tiles", int(row.get("physical_candidate_tiles", 0)) - int(row.get("physical_retained_tiles", 0)))) for row in rows)
        mass_weight = sum(int(row.get("valid_rows", 0)) for row in rows)
        mass_sum = sum(float(row.get("retained_dense_attention_mass", 0.0)) * int(row.get("valid_rows", 0)) for row in rows)
        routing_rows = sum(int(row.get("routing_rows", row.get("valid_rows", 0))) for row in rows)
        degenerate = sum(int(row.get("degenerate_rows", 0)) for row in rows)
        fallback = sum(int(row.get("fallback_rows", 0)) for row in rows)
        return {
            "attention_calls": len(rows),
            "total_tiles": total,
            "routed_tiles": routed,
            "skipped_tiles": skipped,
            "retained_tiles": total - skipped,
            "full_model_tile_sparsity": ratio(skipped, total),
            "routed_region_tile_sparsity": ratio(skipped, routed),
            "retained_dense_attention_mass": ratio(mass_sum, mass_weight),
            "routing_rows": routing_rows,
            "degenerate_rows": degenerate,
            "fallback_rows": fallback,
            "degenerate_row_rate": ratio(degenerate, routing_rows),
            "fallback_row_rate": ratio(fallback, routing_rows),
        }

    result = {"whole_model": group(calls)}
    for attention_type in ("local", "global"):
        result[attention_type] = group([
            row for row in calls if str(row.get("attention_type")) == attention_type
        ])
    regions: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total_tiles": 0, "routed_tiles": 0, "skipped_tiles": 0, "retained_tiles": 0,
    })
    for row in calls:
        for name, counts in (row.get("all_region_counts") or row.get("region_counts") or {}).items():
            regions[name]["total_tiles"] += int(counts.get("total_tiles", counts.get("candidate_tiles", 0)))
            regions[name]["routed_tiles"] += int(counts.get("candidate_tiles", 0))
            regions[name]["skipped_tiles"] += int(counts.get("skipped_tiles", 0))
            regions[name]["retained_tiles"] += int(counts.get("retained_tiles", 0))
    result["regions"] = {
        name: {
            **counts,
            "full_model_tile_sparsity": ratio(counts["skipped_tiles"], counts["total_tiles"]),
            "routed_region_tile_sparsity": ratio(counts["skipped_tiles"], counts["routed_tiles"]),
        }
        for name, counts in regions.items()
    }
    return result


def token_agreement(first: list[int], second: list[int]) -> float:
    length = max(len(first), len(second))
    if not length:
        return 1.0
    return sum(i < len(first) and i < len(second) and first[i] == second[i] for i in range(length)) / length


def bootstrap_delta(dense: np.ndarray, sparse: np.ndarray, *, seed: int = 42, repeats: int = 20_000) -> tuple[float, float, float]:
    delta = np.asarray(sparse, dtype=float) - np.asarray(dense, dtype=float)
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(axis=1)
    return float(delta.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
