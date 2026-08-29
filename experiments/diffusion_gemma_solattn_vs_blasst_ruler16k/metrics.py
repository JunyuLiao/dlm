"""Metrics used by the canonical Sol-Attn versus BLASST report."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def retained_dense_attention_mass(
    dense_scores: torch.Tensor,
    retained_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Average dense-softmax probability mass on retained positions."""

    if dense_scores.ndim != 4 or retained_mask.shape != dense_scores.shape:
        raise ValueError("scores and retained_mask must have shape [batch, heads, query, key]")
    if valid_mask is None:
        valid = torch.isfinite(dense_scores)
    else:
        valid = valid_mask.to(dense_scores.device, dtype=torch.bool)
    valid = torch.broadcast_to(valid, dense_scores.shape)
    has = valid.any(dim=-1)
    safe_scores = torch.where(has[..., None], dense_scores, torch.zeros_like(dense_scores))
    probabilities = torch.softmax(safe_scores, dim=-1, dtype=torch.float32)
    probabilities = torch.where(has[..., None], probabilities, torch.zeros_like(probabilities))
    mass = (probabilities * retained_mask.to(probabilities.device) * valid).sum(dim=-1)
    rows = has.sum()
    return float(mass[has].sum().item() / rows.item()) if int(rows.item()) else 0.0


def positional_token_id_agreement(dense_tokens: Sequence[int], sparse_tokens: Sequence[int]) -> float:
    """Agreement over the union of positions, counting missing/extra tokens."""

    dense, sparse = list(dense_tokens), list(sparse_tokens)
    denominator = max(len(dense), len(sparse))
    if denominator == 0:
        return 1.0
    matches = sum(1 for index in range(denominator) if index < len(dense) and index < len(sparse) and dense[index] == sparse[index])
    return matches / denominator


def positional_token_id_disagreements(dense_tokens: Sequence[int], sparse_tokens: Sequence[int]) -> int:
    denominator = max(len(dense_tokens), len(sparse_tokens))
    return sum(1 for index in range(denominator) if index >= len(dense_tokens) or index >= len(sparse_tokens) or dense_tokens[index] != sparse_tokens[index])


def exact_sequence_match(dense_tokens: Sequence[int], sparse_tokens: Sequence[int]) -> bool:
    return list(dense_tokens) == list(sparse_tokens)


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    repeats: int = 20_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Prompt-bootstrap confidence interval for a paired mean delta."""

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.size == 0:
        return (0.0, 0.0)
    if repeats <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("repeats must be positive and confidence must lie in (0,1)")
    seed_bytes = hashlib.sha256(f"ruler16k|{seed}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    draws = rng.integers(0, values.size, size=(int(repeats), values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _sum_metric(rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str) -> float:
    n = sum(float(row.get(numerator, 0.0)) for row in rows)
    d = sum(float(row.get(denominator, 0.0)) for row in rows)
    return n / d if d else 0.0


def aggregate_routing_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate routing counters overall and by local/global attention type."""

    values = [dict(row) for row in rows]
    # Fresh-routing prediction rows carry exact layer/head/step counters in a
    # nested ``per_call`` list.  Flatten those records for aggregation so the
    # local/global views are populated as well as the overall view.  BLASST
    # reference rows already arrive as flat per-step records and are unchanged.
    expanded: list[dict[str, Any]] = []
    for row in values:
        per_call = row.get("per_call")
        if isinstance(per_call, list) and per_call:
            expanded.extend(dict(item) for item in per_call if isinstance(item, Mapping))
    aggregation_values = expanded if expanded else values
    def aggregate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def count(row: Mapping[str, Any], direct: str, modern: str, legacy: str) -> int:
            if row.get(direct) is not None:
                return int(row.get(direct, 0))
            if row.get(modern) is not None:
                return int(row.get(modern, 0))
            return int(row.get(legacy, 0))

        eligible_values = []
        skipped_values = []
        retained_values = []
        for row in subset:
            eligible = count(row, "eligible_tiles", "physical_candidate_tiles", "logical_candidate_tiles")
            retained = count(row, "retained_tiles", "physical_retained_tiles", "logical_retained_tiles")
            skipped_raw = row.get("skipped_tiles")
            if skipped_raw is None:
                skipped_raw = row.get("physical_skipped_tiles")
            if skipped_raw is None:
                skipped_raw = row.get("logical_skipped_tiles")
            skipped = int(eligible - retained if skipped_raw is None else skipped_raw)
            eligible_values.append(eligible)
            retained_values.append(retained)
            skipped_values.append(skipped)
        result = {
            "calls": len(subset),
            "eligible_tiles": sum(eligible_values),
            "skipped_tiles": sum(skipped_values),
            "retained_tiles": sum(retained_values),
            "valid_qk_elements": sum(int(row.get("valid_qk_elements") if row.get("valid_qk_elements") is not None else row.get("valid_elements", 0)) for row in subset),
            "skipped_valid_qk_elements": sum(int(row.get("skipped_valid_qk_elements") if row.get("skipped_valid_qk_elements") is not None else row.get("skipped_valid_elements", 0)) for row in subset),
            "degenerate_rows": sum(int(row.get("degenerate_rows", 0)) for row in subset),
            "fallback_rows": sum(int(row.get("fallback_rows", 0)) for row in subset),
            "routing_rows": sum(int(
                row.get("routing_rows")
                if row.get("routing_rows") is not None
                else row.get("valid_rows", 0)
            ) for row in subset),
        }
        result["full_tile_sparsity"] = _sum_metric([result], "skipped_tiles", "eligible_tiles")
        result["full_tile_density"] = 1.0 - result["full_tile_sparsity"]
        result["valid_qk_element_sparsity"] = _sum_metric([result], "skipped_valid_qk_elements", "valid_qk_elements")
        result["degenerate_row_rate"] = _sum_metric([result], "degenerate_rows", "routing_rows")
        result["empty_row_fallback_rate"] = _sum_metric([result], "fallback_rows", "routing_rows")
        # Weighted diagnostics if per-call mass was recorded.
        masses = [
            (
                float(row["retained_dense_attention_mass"]),
                int(
                    row.get(
                        "valid_rows",
                        row.get(
                            "retained_attention_mass_rows",
                            row.get("dense_attention_mass_rows", 0),
                        ),
                    )
                ),
            )
            for row in subset
            if row.get("retained_dense_attention_mass") is not None
            and math.isfinite(float(row["retained_dense_attention_mass"]))
        ]
        result["retained_dense_attention_mass"] = (
            sum(value * max(weight, 1) for value, weight in masses) / sum(max(weight, 1) for _, weight in masses)
            if masses else None
        )
        return result

    result = {"overall": aggregate(aggregation_values)}
    for attention_type in ("global", "local"):
        result[attention_type] = aggregate([
            row for row in aggregation_values
            if str(row.get("attention_type", "")) == attention_type
        ])
    # Include an explicit prefix/canvas breakdown whenever rows provide it.
    regions: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible_tiles": 0, "skipped_tiles": 0})
    for row in aggregation_values:
        items = row.get("region_counts") if isinstance(row.get("region_counts"), Mapping) else None
        if items is None:
            items = {"prefix": row.get("prefix", {}), "canvas": row.get("canvas", {})}
        for name, item in items.items():
            if not isinstance(item, Mapping):
                continue
            eligible = int(item.get("eligible_tiles", item.get("candidate_tiles", 0)))
            retained = int(item.get("retained_tiles", item.get("retained", 0)))
            skipped_raw = item.get("skipped_tiles")
            if skipped_raw is None:
                skipped_raw = item.get("skipped")
            skipped = int(eligible - retained if skipped_raw is None else skipped_raw)
            regions[name]["eligible_tiles"] += eligible
            regions[name]["skipped_tiles"] += skipped
            regions[name]["retained_tiles"] = regions[name].get("retained_tiles", 0) + retained
    result["regions"] = dict(regions)
    return result


def paired_generation_metrics(
    dense_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
    *,
    score_fn: Any | None = None,
    bootstrap_repeats: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired accuracy, token agreement, and exact-match diagnostics."""

    dense = {str(row["sample_id"]): row for row in dense_rows}
    sparse = {str(row["sample_id"]): row for row in sparse_rows}
    ids = sorted(set(dense) & set(sparse))
    if ids and any(dense[item].get("prompt_hash") != sparse[item].get("prompt_hash") for item in ids):
        raise ValueError("paired rows do not share prompt hashes")
    dense_scores, sparse_scores = [], []
    agreements, exact = [], []
    for item in ids:
        if score_fn is not None:
            dense_scores.append(float(score_fn(dense[item])))
            sparse_scores.append(float(score_fn(sparse[item])))
        dense_tokens = dense[item].get("completion_tokens", dense[item].get("tokens", []))
        sparse_tokens = sparse[item].get("completion_tokens", sparse[item].get("tokens", []))
        agreements.append(positional_token_id_agreement(dense_tokens, sparse_tokens))
        exact.append(exact_sequence_match(dense_tokens, sparse_tokens))
    deltas = [b - a for a, b in zip(dense_scores, sparse_scores)]
    ci = paired_bootstrap_ci(deltas, repeats=bootstrap_repeats, seed=seed) if deltas else (0.0, 0.0)
    return {
        "paired_examples": len(ids),
        "dense_accuracy": float(np.mean(dense_scores)) if dense_scores else None,
        "sparse_accuracy": float(np.mean(sparse_scores)) if sparse_scores else None,
        "accuracy_delta": float(np.mean(deltas)) if deltas else None,
        "accuracy_delta_bootstrap_95ci": list(ci),
        "token_id_agreement": float(np.mean(agreements)) if agreements else None,
        "full_sequence_exact_match_rate": float(np.mean(exact)) if exact else None,
        "prompt_ids": ids,
    }


__all__ = [
    "aggregate_routing_stats",
    "exact_sequence_match",
    "paired_bootstrap_ci",
    "paired_generation_metrics",
    "positional_token_id_agreement",
    "positional_token_id_disagreements",
    "retained_dense_attention_mass",
]
