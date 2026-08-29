"""Offline BLASST lambda calibration from dense margin traces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:  # substantially faster for the multi-gigabyte calibration trace
    import orjson
except ImportError:  # pragma: no cover - optional acceleration
    orjson = None


def _loads_line(line: str | bytes) -> Any:
    """Fast JSON decode with stdlib fallback for NaN/Infinity tokens."""

    if orjson is not None:
        try:
            return orjson.loads(line)
        except Exception:
            pass
    return json.loads(line)

from dllm.attention.blasst.core import _expand_valid_mask, _prepare_attention_scores

from .config import TARGET_SPARSITIES


def candidate_lambdas() -> tuple[float, ...]:
    """Return the declared calibration grid in deterministic order."""

    values = list(np.logspace(-8.0, -1.0, 29))
    values.extend((0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999))
    return tuple(sorted({float(value) for value in values}))


def _trace_arrays(trace: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    margins = np.asarray(trace.get("margins", trace.get("row_margins")), dtype=np.float64)
    if margins.ndim == 1:
        margins = margins[None, :]
    if margins.ndim != 2:
        raise ValueError("a margin trace must have shape [query_rows, kv_tiles]")
    # A first-tile online margin is conventionally ``+inf`` (running max was
    # -inf), but that row/tile is still valid.  Only an explicitly supplied
    # validity array determines structural eligibility.
    valid = np.asarray(trace.get("valid_rows", np.ones_like(margins, dtype=bool)), dtype=bool)
    if valid.shape != margins.shape:
        valid = np.broadcast_to(valid, margins.shape)
    tile_eligible = np.asarray(trace.get("eligible_tiles", valid.any(axis=0)), dtype=bool)
    if tile_eligible.ndim != 1:
        tile_eligible = tile_eligible.reshape(-1)
    if tile_eligible.shape[0] != margins.shape[1]:
        tile_eligible = np.broadcast_to(tile_eligible, (margins.shape[1],))
    valid_elements = np.asarray(trace.get("valid_elements", valid), dtype=np.int64)
    if valid_elements.shape != margins.shape:
        valid_elements = np.broadcast_to(valid_elements, margins.shape)
    lengths = np.asarray(
        trace.get("valid_kv_length", trace.get("kv_length", margins.shape[1])),
        dtype=np.float64,
    )
    return margins, valid, tile_eligible, valid_elements


def evaluate_margin_trace(
    trace: Mapping[str, Any],
    lambda_value: float,
    *,
    metric: str = "physical",
) -> dict[str, float | int]:
    """Evaluate one BLASST threshold against a saved dense trace."""

    value = float(lambda_value)
    if not 0.0 < value < 1.0:
        raise ValueError("lambda must lie strictly between zero and one")
    if metric not in {"physical", "valid_qk"}:
        raise ValueError("metric must be physical or valid_qk")
    margins, valid, eligible, valid_elements = _trace_arrays(trace)
    valid_kv_length = float(
        np.asarray(
            trace.get("valid_kv_length", trace.get("kv_length", margins.shape[1])),
            dtype=np.float64,
        ).reshape(-1).mean()
    )
    votes = valid & (margins < math.log(value))
    # A physical tile is skippable only if every valid row votes to skip it.
    physical_skip = eligible & np.all(votes | ~valid, axis=0)
    eligible_count = int(eligible.sum())
    physical_skipped = int(physical_skip.sum())
    total_elements = int(valid_elements[valid].sum())
    skipped_elements = int(valid_elements[votes].sum())
    sparsity = (
        physical_skipped / eligible_count
        if metric == "physical" and eligible_count
        else skipped_elements / total_elements if total_elements else 0.0
    )
    return {
        "lambda": value,
        "physical_eligible_tiles": eligible_count,
        "physical_skipped_tiles": physical_skipped,
        "physical_sparsity": physical_skipped / eligible_count if eligible_count else 0.0,
        "valid_qk_elements": total_elements,
        "skipped_valid_qk_elements": skipped_elements,
        "valid_qk_element_sparsity": skipped_elements / total_elements if total_elements else 0.0,
        "valid_kv_length": valid_kv_length,
        "sparsity": sparsity,
    }


def evaluate_lambda_grid(
    traces: Iterable[Mapping[str, Any]],
    lambdas: Iterable[float] | None = None,
    *,
    metric: str = "physical",
) -> list[dict[str, Any]]:
    """Aggregate each candidate lambda over a trace collection."""

    values = tuple(candidate_lambdas() if lambdas is None else (float(x) for x in lambdas))
    return _evaluate_lambda_grid_single_pass(traces, values, metric=metric)


def _evaluate_lambda_grid_single_pass(
    traces: Iterable[Mapping[str, Any]],
    values: Sequence[float],
    *,
    metric: str,
    attention_type: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all lambdas while retaining only one parsed trace in memory.

    Calibration traces for a 16K DiffusionGemma run are multi-gigabyte JSONL.
    The original implementation materialized one measurement object per trace
    for every lambda, which could exceed host RAM during the offline fit.  This
    accumulator performs the same exact integer reductions in one pass.
    """

    if metric not in {"physical", "valid_qk"}:
        raise ValueError("metric must be physical or valid_qk")
    values = tuple(float(value) for value in values)
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("lambda must lie strictly between zero and one")
    eligible_counts = [0] * len(values)
    skipped_counts = [0] * len(values)
    element_counts = [0] * len(values)
    skipped_element_counts = [0] * len(values)
    length_sum = 0.0
    trace_count = 0
    for trace in traces:
        if attention_type is not None and str(trace.get("attention_type", "global")) != attention_type:
            continue
        margins, valid, eligible, valid_elements = _trace_arrays(trace)
        valid_kv_length = float(
            np.asarray(
                trace.get("valid_kv_length", trace.get("kv_length", margins.shape[1])),
                dtype=np.float64,
            ).reshape(-1).mean()
        )
        trace_count += 1
        length_sum += valid_kv_length
        eligible_count_value = int(eligible.sum())
        total_elements_value = int(valid_elements[valid].sum())
        for index, value in enumerate(values):
            votes = valid & (margins < math.log(value))
            physical_skipped = int((eligible & np.all(votes | ~valid, axis=0)).sum())
            skipped_elements = int(valid_elements[votes].sum())
            eligible_counts[index] += eligible_count_value
            skipped_counts[index] += physical_skipped
            element_counts[index] += total_elements_value
            skipped_element_counts[index] += skipped_elements
    result = []
    mean_length = length_sum / trace_count if trace_count else 0.0
    for index, value in enumerate(values):
        eligible = eligible_counts[index]
        skipped = skipped_counts[index]
        elements = element_counts[index]
        skipped_elements = skipped_element_counts[index]
        result.append({
            "lambda": value,
            "sparsity": skipped / eligible if metric == "physical" and eligible else (
                skipped_elements / elements if elements else 0.0
            ),
            "physical_sparsity": skipped / eligible if eligible else 0.0,
            "valid_qk_element_sparsity": skipped_elements / elements if elements else 0.0,
            "eligible_tiles": eligible,
            "skipped_tiles": skipped,
            "valid_qk_elements": elements,
            "skipped_valid_qk_elements": skipped_elements,
            "valid_kv_length": mean_length,
            "num_traces": trace_count,
        })
    return result


def evaluate_lambda_grid_jsonl(
    trace_path: str | Path,
    lambdas: Iterable[float] | None = None,
    *,
    attention_type: str | None = None,
    metric: str = "physical",
) -> list[dict[str, Any]]:
    """Stream a JSONL trace file and evaluate all lambdas without loading it."""

    values = tuple(candidate_lambdas() if lambdas is None else (float(x) for x in lambdas))

    def rows() -> Iterable[dict[str, Any]]:
        with Path(trace_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _loads_line(line)

    return _evaluate_lambda_grid_single_pass(
        rows(), values, metric=metric, attention_type=attention_type
    )


def evaluate_lambda_grid_jsonl_by_type(
    trace_path: str | Path,
    lambdas: Iterable[float] | None = None,
    *,
    metric: str = "physical",
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate local and global grids in one streaming pass over JSONL."""

    values = tuple(candidate_lambdas() if lambdas is None else (float(x) for x in lambdas))
    state = {
        name: {
            "eligible": [0] * len(values), "skipped": [0] * len(values),
            "elements": [0] * len(values), "skipped_elements": [0] * len(values),
            "length_sum": 0.0, "trace_count": 0,
        }
        for name in ("local", "global")
    }
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _loads_line(line)
            attention_type = str(row.get("attention_type", "global"))
            if attention_type not in state:
                continue
            margins, valid, eligible, valid_elements = _trace_arrays(row)
            length = float(np.asarray(
                row.get("valid_kv_length", row.get("kv_length", margins.shape[1])),
                dtype=np.float64,
            ).reshape(-1).mean())
            item = state[attention_type]
            item["trace_count"] += 1
            item["length_sum"] += length
            eligible_count = int(eligible.sum())
            total_elements = int(valid_elements[valid].sum())
            for index, value in enumerate(values):
                votes = valid & (margins < math.log(value))
                skipped = int((eligible & np.all(votes | ~valid, axis=0)).sum())
                skipped_elements = int(valid_elements[votes].sum())
                item["eligible"][index] += eligible_count
                item["skipped"][index] += skipped
                item["elements"][index] += total_elements
                item["skipped_elements"][index] += skipped_elements
    output: dict[str, list[dict[str, Any]]] = {}
    for attention_type, item in state.items():
        trace_count = int(item["trace_count"])
        mean_length = item["length_sum"] / trace_count if trace_count else 0.0
        rows: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            eligible = int(item["eligible"][index]); skipped = int(item["skipped"][index])
            elements = int(item["elements"][index]); skipped_elements = int(item["skipped_elements"][index])
            rows.append({
                "lambda": value,
                "sparsity": skipped / eligible if metric == "physical" and eligible else (
                    skipped_elements / elements if elements else 0.0
                ),
                "physical_sparsity": skipped / eligible if eligible else 0.0,
                "valid_qk_element_sparsity": skipped_elements / elements if elements else 0.0,
                "eligible_tiles": eligible, "skipped_tiles": skipped,
                "valid_qk_elements": elements, "skipped_valid_qk_elements": skipped_elements,
                "valid_kv_length": mean_length, "num_traces": trace_count,
            })
        output[attention_type] = rows
    return output


def fit_lambda_relation(
    evaluations: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, float | int | bool]:
    """Fit ``lambda * L = alpha * exp(gamma * sparsity)`` in log space."""

    rows = [row for row in evaluations if float(row.get("lambda", 0.0)) > 0 and math.isfinite(float(row.get("sparsity", math.nan)))]
    if len(rows) < 2:
        return {"alpha": math.nan, "gamma": math.nan, "r2": math.nan, "n": len(rows), "fittable": False}
    if traces:
        lengths = [float(trace.get("valid_kv_length", trace.get("kv_length", 1.0))) for trace in traces]
        length = float(np.mean(lengths)) if lengths else 1.0
    else:
        row_lengths = [float(row.get("valid_kv_length", 1.0)) for row in rows]
        length = float(np.mean(row_lengths)) if row_lengths else 1.0
    x = np.asarray([float(row["sparsity"]) for row in rows])
    row_lengths = np.asarray([
        float(row.get("valid_kv_length", length)) for row in rows
    ])
    y = np.log(np.asarray([float(row["lambda"]) for row in rows]) * row_lengths)
    if float(np.ptp(x)) < 1.0e-12:
        return {"alpha": float(np.exp(y.mean())), "gamma": 0.0, "r2": 0.0, "n": len(rows), "fittable": False}
    gamma, intercept = np.polyfit(x, y, 1)
    predicted = intercept + gamma * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "alpha": float(np.exp(intercept)),
        "gamma": float(gamma),
        "r2": 1.0 - residual / total if total else 1.0,
        "n": len(rows),
        "fittable": True,
        "mean_valid_kv_length": length,
    }


def predict_lambda(relation: Mapping[str, Any], target_sparsity: float, valid_kv_length: float) -> float:
    alpha, gamma = float(relation["alpha"]), float(relation["gamma"])
    if not math.isfinite(alpha) or not math.isfinite(gamma) or valid_kv_length <= 0:
        raise ValueError("cannot predict lambda from an unfitted relation")
    value = alpha * math.exp(gamma * float(target_sparsity)) / float(valid_kv_length)
    return float(min(max(value, 1.0e-8), 0.999999))


@dataclass
class CalibratedLambda:
    attention_type: str
    target_sparsity: float
    lambda_value: float
    observed_sparsity: float | None
    error: float | None
    unattainable: bool
    rounds: int
    relation: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_type": self.attention_type,
            "target_sparsity": self.target_sparsity,
            "lambda": self.lambda_value,
            "observed_sparsity": self.observed_sparsity,
            "absolute_error": self.error,
            "unattainable_within_calibration_range": self.unattainable,
            "rounds": self.rounds,
            "relation": self.relation,
            "provenance": self.provenance,
        }


def calibrate_attention_type(
    traces: Sequence[Mapping[str, Any]],
    attention_type: str,
    *,
    targets: Sequence[float] = TARGET_SPARSITIES,
    max_rounds: int = 3,
    metric: str = "physical",
) -> list[CalibratedLambda]:
    """Calibrate one shared local/global lambda for each target sparsity."""

    if attention_type not in {"local", "global"}:
        raise ValueError("attention_type must be local or global")
    selected = [trace for trace in traces if str(trace.get("attention_type", attention_type)) == attention_type]
    if not selected:
        raise ValueError(f"no {attention_type} calibration traces")
    grid = list(candidate_lambdas())
    evaluations = evaluate_lambda_grid(selected, grid, metric=metric)
    # Physical sparsity is monotone in lambda.  Use the mean valid KV length
    # only for the analytic relation; the grid itself remains authoritative.
    relation = fit_lambda_relation(evaluations, selected)
    mean_length = float(np.mean([float(t.get("valid_kv_length", t.get("kv_length", 1.0))) for t in selected]))
    results: list[CalibratedLambda] = []
    for target in targets:
        target = float(target)
        rounds = 0
        local_evals = list(evaluations)
        local_relation = relation
        predicted = predict_lambda(local_relation, target, mean_length) if local_relation.get("fittable") else None
        unattainable = False
        observed = None
        error = None
        chosen = None
        while rounds < max_rounds:
            sparsities = np.asarray([float(row["sparsity"]) for row in local_evals])
            lambdas = np.asarray([float(row["lambda"]) for row in local_evals])
            lo, hi = float(sparsities.min()), float(sparsities.max())
            if target < lo or target > hi:
                nearest = int(np.argmin(np.abs(sparsities - target)))
                chosen = local_evals[nearest]
                unattainable = True
                break
            if predicted is None:
                nearest = int(np.argmin(np.abs(sparsities - target)))
                chosen = local_evals[nearest]
            else:
                chosen = min(local_evals, key=lambda row: abs(float(row["lambda"]) - predicted))
            observed = float(chosen["sparsity"])
            error = abs(observed - target)
            if error <= 0.02:
                break
            if predicted is None:
                break
            # Add the predicted point and refit, up to three total rounds.
            if not any(abs(float(row["lambda"]) - predicted) < 1.0e-12 for row in local_evals):
                local_evals.extend(evaluate_lambda_grid(selected, [predicted], metric=metric))
            local_relation = fit_lambda_relation(local_evals, selected)
            predicted = predict_lambda(local_relation, target, mean_length) if local_relation.get("fittable") else predicted
            rounds += 1
        if chosen is None:
            chosen = min(local_evals, key=lambda row: abs(float(row["sparsity"]) - target))
        observed = float(chosen["sparsity"])
        error = abs(observed - target)
        results.append(CalibratedLambda(
            attention_type=attention_type,
            target_sparsity=target,
            lambda_value=float(chosen["lambda"]),
            observed_sparsity=observed,
            error=error,
            unattainable=unattainable,
            rounds=rounds,
            relation=dict(local_relation),
            provenance={
                "grid": [float(value) for value in grid],
                "metric": metric,
                "calibration_trace_count": len(selected),
                "valid_kv_length": mean_length,
            },
        ))
    return results


def fit_blasst_policy(
    traces: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[float] = TARGET_SPARSITIES,
    max_rounds: int = 3,
    metric: str = "physical",
) -> dict[str, Any]:
    """Fit independent local/global policies from one dense calibration pass."""

    local = calibrate_attention_type(traces, "local", targets=targets, max_rounds=max_rounds, metric=metric)
    global_ = calibrate_attention_type(traces, "global", targets=targets, max_rounds=max_rounds, metric=metric)
    return {
        "schema_version": 1,
        "relation": "lambda * L = alpha * exp(gamma * s)",
        "metric": metric,
        "targets": [float(value) for value in targets],
        "local": [row.to_dict() for row in local],
        "global": [row.to_dict() for row in global_],
        "lambda_local": {str(row.target_sparsity): row.lambda_value for row in local},
        "lambda_global": {str(row.target_sparsity): row.lambda_value for row in global_},
    }


def _calibrate_attention_type_jsonl(
    trace_path: str | Path,
    attention_type: str,
    *,
    targets: Sequence[float],
    max_rounds: int,
    metric: str,
    initial_evaluations: Sequence[Mapping[str, Any]] | None = None,
    rounds_by_target_override: Mapping[float, int] | None = None,
    adaptive_lambdas: Sequence[float] | None = None,
) -> list[CalibratedLambda]:
    """Fit one attention type directly from a streamed trace file."""

    grid = list(candidate_lambdas())
    evaluations = list(initial_evaluations) if initial_evaluations is not None else evaluate_lambda_grid_jsonl(
        trace_path, grid, attention_type=attention_type, metric=metric
    )
    relation = fit_lambda_relation(evaluations)
    mean_length = float(np.mean([
        float(row.get("valid_kv_length", 0.0)) for row in evaluations
    ])) if evaluations else 0.0
    local_evals = list(evaluations)
    local_relation = relation
    rounds_by_target = {
        float(target): int((rounds_by_target_override or {}).get(float(target), 0))
        for target in targets
    }
    unattainable_by_target = {float(target): False for target in targets}
    chosen_by_target: dict[float, Mapping[str, Any]] = {}
    for _round in range(max_rounds + 1):
        sparsities = np.asarray([float(row["sparsity"]) for row in local_evals])
        lo, hi = float(sparsities.min()), float(sparsities.max())
        pending: list[float] = []
        for target_value in targets:
            target = float(target_value)
            if target < lo or target > hi:
                chosen_by_target[target] = min(local_evals, key=lambda row: abs(float(row["sparsity"]) - target))
                unattainable_by_target[target] = True
                continue
            predicted = predict_lambda(local_relation, target, mean_length) if local_relation.get("fittable") else None
            chosen = min(local_evals, key=lambda row: abs(float(row["lambda"]) - predicted)) if predicted is not None else min(local_evals, key=lambda row: abs(float(row["sparsity"]) - target))
            chosen_by_target[target] = chosen
            error = abs(float(chosen["sparsity"]) - target)
            if error > 0.02 and predicted is not None and _round < max_rounds and not any(abs(float(row["lambda"]) - predicted) < 1.0e-12 for row in local_evals):
                pending.append(predicted)
        if not pending or _round >= max_rounds:
            break
        local_evals.extend(evaluate_lambda_grid_jsonl(
            trace_path, sorted(set(pending)), attention_type=attention_type, metric=metric
        ))
        local_relation = fit_lambda_relation(local_evals)
        for target_value in targets:
            rounds_by_target[float(target_value)] += 1
    results: list[CalibratedLambda] = []
    for target_value in targets:
        target = float(target_value)
        chosen = chosen_by_target[target]
        observed = float(chosen["sparsity"])
        results.append(CalibratedLambda(
            attention_type=attention_type,
            target_sparsity=target,
            lambda_value=float(chosen["lambda"]),
            observed_sparsity=observed,
            error=abs(observed - target),
            unattainable=unattainable_by_target[target],
            rounds=rounds_by_target[target],
            relation=dict(local_relation),
            provenance={
                "grid": [float(value) for value in grid],
                "metric": metric,
                "calibration_trace_count": int(evaluations[0].get("num_traces", 0)) if evaluations else 0,
                "valid_kv_length": mean_length,
                "streamed": True,
                "adaptive_lambdas": [float(value) for value in (adaptive_lambdas or ())],
            },
        ))
    return results


def fit_blasst_policy_jsonl(
    trace_path: str | Path,
    *,
    targets: Sequence[float] = TARGET_SPARSITIES,
    max_rounds: int = 3,
    metric: str = "physical",
) -> dict[str, Any]:
    """Fit independent BLASST policies from a large JSONL trace stream."""

    grids = evaluate_lambda_grid_jsonl_by_type(trace_path, metric=metric)
    rounds_by_type_target = {
        attention_type: {float(target): 0 for target in targets}
        for attention_type in ("local", "global")
    }
    adaptive_by_type: dict[str, list[float]] = {"local": [], "global": []}

    # The protocol permits up to ``max_rounds`` adaptive verification rounds.
    # We batch all pending local/global predictions for a round into one scan
    # of the multi-gigabyte trace, then refit both relations before the next
    # round.  A target only requests a point while its measured nearest point
    # is more than two percentage points away and the target is bracketed.
    for _round in range(max(0, int(max_rounds))):
        pending_by_type: dict[str, list[float]] = {"local": [], "global": []}
        pending_targets: dict[str, set[float]] = {"local": set(), "global": set()}
        for attention_type, rows in grids.items():
            if not rows:
                continue
            relation = fit_lambda_relation(rows)
            mean_length = float(np.mean([
                float(row.get("valid_kv_length", 0.0)) for row in rows
            ]))
            sparsities = np.asarray([float(row["sparsity"]) for row in rows])
            lo, hi = float(sparsities.min()), float(sparsities.max())
            for target_value in targets:
                target = float(target_value)
                if target < lo or target > hi or not relation.get("fittable"):
                    continue
                predicted = predict_lambda(relation, target, mean_length)
                chosen = min(rows, key=lambda row: abs(float(row["lambda"]) - predicted))
                error = abs(float(chosen["sparsity"]) - target)
                already_measured = any(
                    abs(float(row["lambda"]) - predicted) < 1.0e-12 for row in rows
                )
                if error > 0.02 and not already_measured:
                    pending_by_type[attention_type].append(predicted)
                    pending_targets[attention_type].add(target)
        all_pending = sorted(set(
            pending_by_type["local"] + pending_by_type["global"]
        ))
        if not all_pending:
            break
        measured = evaluate_lambda_grid_jsonl_by_type(
            trace_path, all_pending, metric=metric
        )
        for attention_type in ("local", "global"):
            existing = {
                round(float(row["lambda"]), 15) for row in grids[attention_type]
            }
            for row in measured[attention_type]:
                if round(float(row["lambda"]), 15) not in existing:
                    grids[attention_type].append(row)
            added = [
                float(value) for value in pending_by_type[attention_type]
                if any(abs(float(value) - float(row["lambda"])) < 1.0e-12
                       for row in measured[attention_type])
            ]
            adaptive_by_type[attention_type].extend(added)
            for target in pending_targets[attention_type]:
                rounds_by_type_target[attention_type][target] += 1
    local = _calibrate_attention_type_jsonl(
        trace_path, "local", targets=targets, max_rounds=0, metric=metric,
        initial_evaluations=grids["local"],
        rounds_by_target_override=rounds_by_type_target["local"],
        adaptive_lambdas=adaptive_by_type["local"],
    )
    global_ = _calibrate_attention_type_jsonl(
        trace_path, "global", targets=targets, max_rounds=0, metric=metric,
        initial_evaluations=grids["global"],
        rounds_by_target_override=rounds_by_type_target["global"],
        adaptive_lambdas=adaptive_by_type["global"],
    )
    return {
        "schema_version": 1,
        "relation": "lambda * L = alpha * exp(gamma * s)",
        "metric": metric,
        "targets": [float(value) for value in targets],
        "local": [row.to_dict() for row in local],
        "global": [row.to_dict() for row in global_],
        "lambda_local": {str(row.target_sparsity): row.lambda_value for row in local},
        "lambda_global": {str(row.target_sparsity): row.lambda_value for row in global_},
    }


class DenseMarginTraceCollector:
    """Observer callback used during one instrumented dense calibration pass.

    The callback stores only block maxima/margins and validity metadata; token
    level QK scores are released immediately.  This is sufficient to evaluate
    every lambda in :func:`candidate_lambdas` offline without letting the
    final set influence threshold selection.
    """

    def __init__(self, *, q_tile_size: int = 64, kv_tile_size: int = 64) -> None:
        self.q_tile_size = int(q_tile_size)
        self.kv_tile_size = int(kv_tile_size)
        self.traces: list[dict[str, Any]] = []

    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # ``install_blasst`` forwards observers using keyword arguments, while
        # lightweight tests and third-party bindings often use positional
        # ``(scores, valid_pair_mask, active_query_mask, ...)``.  Accept both.
        if args:
            # Registry observers receive ``(module, query, key, value,
            # attention_mask, ...)``.  A direct unit-test callback may use the
            # compact ``(scores, valid_mask, active_mask)`` form.
            if isinstance(args[0], torch.Tensor):
                if len(args) < 3:
                    raise TypeError("observer requires scores, valid_pair_mask, and active_query_mask")
                scores, valid_pair_mask, active_query_mask = args[:3]
                metadata = kwargs.get("metadata")
            else:
                if len(args) < 4:
                    raise TypeError("registry observer requires module, query, key, and value")
                module, query, key, value = args[:4]
                attention_mask = args[4] if len(args) > 4 else kwargs.get("attention_mask")
                _expanded_key, _expanded_value, scores, valid_pair_mask = _prepare_attention_scores(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling=kwargs.get("scaling"),
                    is_causal=kwargs.get("is_causal"),
                    sliding_window=kwargs.get("sliding_window"),
                )
                active_query_mask = kwargs.get(
                    "active_query_mask", kwargs.get("blasst_active_query_mask")
                )
                metadata = kwargs.get("blasst_metadata", kwargs.get("metadata"))
                if metadata is None:
                    runtime = getattr(module, "_blasst_2d_runtime", None)
                    metadata = getattr(runtime, "metadata", None)
                metadata = dict(metadata or {})
                if "attention_type" not in metadata:
                    from dllm.attention.blasst.core import _attention_type
                    metadata["attention_type"] = _attention_type(module, kwargs.get("sliding_window"))
        else:
            scores = kwargs.pop("scores")
            valid_pair_mask = kwargs.pop("valid_pair_mask")
            active_query_mask = kwargs.pop("active_query_mask", None)
            metadata = kwargs.pop("metadata", None)
        metadata = dict(metadata or {})
        if not isinstance(valid_pair_mask, torch.Tensor):
            raise TypeError("valid_pair_mask must be a tensor")
        valid = _expand_valid_mask(valid_pair_mask, scores)
        active = torch.ones(valid.shape[:3], dtype=torch.bool, device=valid.device) if active_query_mask is None else active_query_mask.to(valid.device, dtype=torch.bool)
        if active.ndim == 2:
            active = active[:, None, :]
        if active.shape[-1] != valid.shape[-2]:
            if active.shape[-1] < valid.shape[-2]:
                raise ValueError("active_query_mask is shorter than the query sequence")
            active = active[..., -valid.shape[-2]:]
        active = torch.broadcast_to(active, valid.shape[:3])
        batch, heads, q_len, kv_len = scores.shape
        q_tiles = math.ceil(q_len / self.q_tile_size)
        kv_tiles = math.ceil(kv_len / self.kv_tile_size)
        for batch_index in range(batch):
            for head in range(heads):
                for qi in range(q_tiles):
                    qs, qe = qi * self.q_tile_size, min((qi + 1) * self.q_tile_size, q_len)
                    margins = []
                    valid_rows = []
                    valid_element_counts = []
                    eligible = []
                    running = torch.full((qe - qs,), -torch.inf, dtype=torch.float32, device=scores.device)
                    for ki in range(kv_tiles):
                        ks, ke = ki * self.kv_tile_size, min((ki + 1) * self.kv_tile_size, kv_len)
                        pair = valid[batch_index, head, qs:qe, ks:ke] & active[batch_index, head, qs:qe, None]
                        has = pair.any(dim=-1)
                        maxima = scores[batch_index, head, qs:qe, ks:ke].float().masked_fill(~pair, -torch.inf).amax(dim=-1)
                        margins.append((maxima - running).cpu().numpy())
                        valid_rows.append(has.cpu().numpy())
                        valid_element_counts.append(pair.sum(dim=-1).cpu().numpy())
                        eligible.append(bool(has.any().item()))
                        running = torch.maximum(running, maxima)
                    margins_array = np.stack(margins, axis=1)
                    valid_array = np.stack(valid_rows, axis=1)
                    self.traces.append({
                        **metadata,
                        "batch": batch_index,
                        "head": head,
                        "query_block": qi,
                        "attention_type": str(metadata.get("attention_type", "global")),
                        "valid_kv_length": int(valid[batch_index, head].any(dim=0).sum().item()),
                        "margins": margins_array.tolist(),
                        "valid_rows": valid_array.tolist(),
                        "eligible_tiles": eligible,
                        "valid_elements": np.stack(valid_element_counts, axis=1).astype(np.int64).tolist(),
                    })

    def to_jsonable(self) -> list[dict[str, Any]]:
        return self.traces


__all__ = [
    "CalibratedLambda",
    "DenseMarginTraceCollector",
    "candidate_lambdas",
    "calibrate_attention_type",
    "evaluate_lambda_grid",
    "evaluate_lambda_grid_jsonl",
    "evaluate_lambda_grid_jsonl_by_type",
    "evaluate_margin_trace",
    "fit_blasst_policy",
    "fit_blasst_policy_jsonl",
    "fit_lambda_relation",
    "predict_lambda",
]
