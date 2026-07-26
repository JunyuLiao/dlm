#!/usr/bin/env python3
"""Calibrate simple proxy policies on sample-disjoint trace splits."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from analysis.proxy_relationships import joined_pairs, json_safe, split_is_calibration
from tracing.trace_schema import SCHEMA_VERSION, load_trace_directory


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    predictions: int
    errors: int
    new_max_errors: int
    false_skip_upper_bound: float
    new_max_upper_bound: float
    certified: bool


def wilson_upper_bound(errors: int, trials: int, confidence: float) -> float:
    """One-sided Wilson binomial upper confidence bound."""
    if trials <= 0:
        return 1.0
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    z = NormalDist().inv_cdf(confidence)
    observed = errors / trials
    denominator = 1.0 + z * z / trials
    center = observed + z * z / (2.0 * trials)
    radius = z * math.sqrt(observed * (1.0 - observed) / trials + z * z / (4.0 * trials * trials))
    return min(1.0, (center + radius) / denominator)


def wilson_upper_bounds(errors: np.ndarray, trials: np.ndarray, confidence: float) -> np.ndarray:
    """Vectorized one-sided Wilson bound for threshold candidate sweeps."""
    z = NormalDist().inv_cdf(confidence)
    count = trials.astype(np.float64)
    observed = errors.astype(np.float64) / count
    denominator = 1.0 + z * z / count
    center = observed + z * z / (2.0 * count)
    radius = z * np.sqrt(observed * (1.0 - observed) / count + z * z / (4.0 * count * count))
    return np.minimum(1.0, (center + radius) / denominator)


def select_certified_threshold(
    source_scores: np.ndarray,
    target_skip: np.ndarray,
    target_new_max: np.ndarray,
    budget: float,
    *,
    new_max_budget: float,
    confidence: float,
    min_predictions: int,
) -> ThresholdSelection:
    """Largest threshold whose total and new-max false-skip UCBs pass."""
    finite = np.isfinite(source_scores)
    scores = source_scores[finite]
    labels = target_skip[finite].astype(bool)
    new_max = target_new_max[finite].astype(bool)
    if not len(scores):
        return ThresholdSelection(-math.inf, 0, 0, 0, 1.0, 1.0, False)
    order = np.argsort(scores, kind="mergesort")
    scores, labels, new_max = scores[order], labels[order], new_max[order]
    errors = np.cumsum(~labels)
    new_max_errors = np.cumsum((~labels) & new_max)
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    counts = ends + 1
    false_upper = wilson_upper_bounds(errors[ends], counts, confidence)
    new_max_upper = wilson_upper_bounds(new_max_errors[ends], counts, confidence)
    eligible = (
        (counts >= min_predictions)
        & (false_upper <= budget)
        & (new_max_upper <= new_max_budget)
    )
    if not eligible.any():
        return ThresholdSelection(-math.inf, 0, 0, 0, 1.0, 1.0, False)
    selected = int(np.flatnonzero(eligible)[-1])
    end = int(ends[selected])
    return ThresholdSelection(
        float(np.nextafter(scores[end], math.inf)), int(counts[selected]), int(errors[end]),
        int(new_max_errors[end]), float(false_upper[selected]), float(new_max_upper[selected]), True,
    )


def select_threshold(source_scores: np.ndarray, target_skip: np.ndarray, budget: float) -> float:
    """Backward-compatible empirical selector used only by legacy ablations."""
    finite = np.isfinite(source_scores)
    scores, labels = source_scores[finite], target_skip[finite].astype(bool)
    if not len(scores):
        return -math.inf
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    errors = np.cumsum(~labels)
    best = -math.inf
    for end in np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]):
        count = end + 1
        if errors[end] / count <= budget:
            best = float(np.nextafter(scores[end], math.inf))
    return best


def prediction_metrics(predicted: np.ndarray, target_skip: np.ndarray, new_max: np.ndarray) -> dict[str, float | int]:
    predicted, target_skip, new_max = predicted.astype(bool), target_skip.astype(bool), new_max.astype(bool)
    correct = int((predicted & target_skip).sum())
    incorrect = int((predicted & ~target_skip).sum())
    total_predicted = int(predicted.sum())
    exact_skips = int(target_skip.sum())
    ratio = lambda a, b: a / b if b else 0.0
    return {
        "tiles": len(predicted),
        "pre_skips": total_predicted,
        "correct_pre_skips": correct,
        "incorrect_pre_skips": incorrect,
        "skip_precision": ratio(correct, total_predicted),
        "oracle_skip_recall": ratio(correct, exact_skips),
        "physical_pre_qk_sparsity": ratio(correct, len(predicted)),
        "false_skip_rate": ratio(incorrect, total_predicted),
        "false_skip_new_max": int((predicted & ~target_skip & new_max).sum()),
    }


def _split_samples(records: np.ndarray) -> set[str]:
    samples = sorted(set(str(value) for value in records["sample_id"]))
    selected = {sample for sample in samples if split_is_calibration(sample)}
    if not selected or selected == set(samples):
        selected = set(samples[: max(1, len(samples) // 2)])
    return selected


def _pair_arrays(pairs: list[tuple[np.void, np.void, str]]) -> tuple[np.ndarray, ...]:
    return (
        np.array([float(source["score"]) for source, _, _ in pairs]),
        np.array([bool(target["exact_skip"]) for _, target, _ in pairs]),
        np.array([bool(target["introduced_new_max"]) for _, target, _ in pairs]),
        np.array([str(target["sample_id"]) for _, target, _ in pairs]),
    )


def apply_safety_filter(
    pairs: list[tuple[np.void, np.void, str]],
    *,
    warmup_tiles: int,
    periodic_refresh: int,
    anchor_local: bool,
    anchor_diagonal: bool,
    anchor_sink: bool,
) -> list[tuple[np.void, np.void, str]]:
    result = []
    for pair in pairs:
        target = pair[1]
        traversal = int(target["traversal_index"])
        forced = traversal < warmup_tiles
        forced |= periodic_refresh > 0 and traversal % periodic_refresh == 0
        forced |= anchor_local and bool(target["is_local"])
        forced |= anchor_diagonal and bool(target["is_diagonal"])
        forced |= anchor_sink and bool(target["is_sink"])
        if not forced:
            result.append(pair)
    return result


def calibrate_relation(
    pairs: list[tuple[np.void, np.void, str]],
    relation: str,
    budgets: list[float],
    calibration_samples: set[str],
    *,
    confidence: float,
    min_predictions: int,
    new_max_budget_scale: float,
) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    deployable: list[dict[str, object]] = []
    transitions = sorted({
        (
            str(source["noise_bucket"]), str(target["noise_bucket"]),
            math.ceil(int(target["sequence_length"]) / 64),
        )
        for source, target, _ in pairs
    })
    candidate_groups = 0
    for source_bucket, target_bucket, num_kv_tiles in transitions:
        transition_pairs = [
            pair for pair in pairs
            if str(pair[0]["noise_bucket"]) == source_bucket
            and str(pair[1]["noise_bucket"]) == target_bucket
            and math.ceil(int(pair[1]["sequence_length"]) / 64) == num_kv_tiles
        ]
        candidate_groups += 1
        candidate_groups += len({int(pair[1]["layer"]) for pair in transition_pairs})
        candidate_groups += len({(int(pair[1]["layer"]), int(pair[1]["head"])) for pair in transition_pairs})
    # Bonferroni adjustment covers total/new-max checks across budgets and the
    # four source-policy families evaluated by this script.
    family_comparisons = max(1, candidate_groups * len(budgets) * 2 * 4)
    per_test_confidence = 1.0 - (1.0 - confidence) / family_comparisons
    for budget in budgets:
        new_max_budget = budget * new_max_budget_scale
        for source_bucket, target_bucket, num_kv_tiles in transitions:
            transition_pairs = [
                pair for pair in pairs
                if str(pair[0]["noise_bucket"]) == source_bucket
                and str(pair[1]["noise_bucket"]) == target_bucket
                and math.ceil(int(pair[1]["sequence_length"]) / 64) == num_kv_tiles
            ]
            layer_values = sorted({int(pair[1]["layer"]) for pair in transition_pairs})
            head_values = sorted({int(pair[1]["head"]) for pair in transition_pairs})
            grouped: dict[tuple[int, int], list[tuple[np.void, np.void, str]]] = {
                (-1, -1): transition_pairs
            }
            for pair in transition_pairs:
                pair_layer, pair_head = int(pair[1]["layer"]), int(pair[1]["head"])
                grouped.setdefault((pair_layer, -1), []).append(pair)
                grouped.setdefault((pair_layer, pair_head), []).append(pair)
            for (layer, head), selected in grouped.items():
                score, target, newmax, samples = _pair_arrays(selected)
                calibration = np.array([sample in calibration_samples for sample in samples])
                selection = select_certified_threshold(
                    score[calibration], target[calibration], newmax[calibration], budget,
                    new_max_budget=new_max_budget, confidence=per_test_confidence,
                    min_predictions=min_predictions,
                )
                evaluation = ~calibration
                predicted = score[evaluation] < selection.threshold
                metrics = prediction_metrics(predicted, target[evaluation], newmax[evaluation])
                evaluation_errors = int((predicted & ~target[evaluation]).sum())
                evaluation_newmax = int((predicted & ~target[evaluation] & newmax[evaluation]).sum())
                evaluation_count = int(predicted.sum())
                evaluation_upper = wilson_upper_bound(evaluation_errors, evaluation_count, per_test_confidence)
                evaluation_newmax_upper = wilson_upper_bound(
                    evaluation_newmax, evaluation_count, per_test_confidence
                )

                worst_group_fsr = 0.0
                worst_group_newmax = 0.0
                if head == -1 and evaluation_count:
                    evaluation_pairs = [pair for pair, keep in zip(selected, evaluation) if keep]
                    group_counts: dict[tuple[int, int], list[int]] = {}
                    for pair, is_predicted, is_skip, is_newmax in zip(
                        evaluation_pairs, predicted, target[evaluation], newmax[evaluation]
                    ):
                        if not is_predicted:
                            continue
                        key = (int(pair[1]["layer"]), int(pair[1]["head"]))
                        counts = group_counts.setdefault(key, [0, 0, 0])
                        counts[0] += 1
                        counts[1] += int(not is_skip)
                        counts[2] += int((not is_skip) and is_newmax)
                    for count, unsafe, unsafe_newmax in group_counts.values():
                        worst_group_fsr = max(worst_group_fsr, unsafe / count)
                        worst_group_newmax = max(worst_group_newmax, unsafe_newmax / count)
                validation_certified = (
                    selection.certified
                    and evaluation_count >= min_predictions
                    and evaluation_upper <= budget
                    and evaluation_newmax_upper <= new_max_budget
                    and worst_group_fsr <= budget
                    and worst_group_newmax <= new_max_budget
                )
                threshold_key = (
                    f"{relation}|{budget:g}|{layer}|{head}|{source_bucket}->{target_bucket}|kv{num_kv_tiles}"
                )
                thresholds[threshold_key] = selection.threshold
                row = {
                    "policy": f"{relation}_score",
                    "false_skip_budget": budget,
                    "new_max_budget": new_max_budget,
                    "family_confidence": confidence,
                    "per_test_confidence": per_test_confidence,
                    "split": "held_out_samples",
                    "layer": layer,
                    "head": head,
                    "source_noise_bucket": source_bucket,
                    "target_noise_bucket": target_bucket,
                    "num_kv_tiles": num_kv_tiles,
                    "threshold": selection.threshold,
                    "calibration_predictions": selection.predictions,
                    "calibration_errors": selection.errors,
                    "calibration_new_max_errors": selection.new_max_errors,
                    "calibration_false_skip_ucb": selection.false_skip_upper_bound,
                    "calibration_new_max_ucb": selection.new_max_upper_bound,
                    "validation_false_skip_ucb": evaluation_upper,
                    "validation_new_max_ucb": evaluation_newmax_upper,
                    "worst_layer_head_fsr": worst_group_fsr,
                    "worst_layer_head_new_max_rate": worst_group_newmax,
                    "certified": validation_certified,
                    **metrics,
                }
                rows.append(row)
                deployable.append({
                    key: row[key]
                    for key in (
                        "policy", "false_skip_budget", "new_max_budget", "family_confidence",
                        "per_test_confidence",
                        "layer", "head", "source_noise_bucket", "target_noise_bucket",
                        "num_kv_tiles",
                        "threshold", "calibration_predictions", "validation_false_skip_ucb",
                        "validation_new_max_ucb", "worst_layer_head_fsr", "certified",
                    )
                } | {
                    "relation": relation,
                    "held_out_candidate_tiles": metrics["pre_skips"],
                    "held_out_correct_pre_qk_sparsity": metrics["physical_pre_qk_sparsity"],
                    "held_out_precision": metrics["skip_precision"],
                    "held_out_oracle_skip_recall": metrics["oracle_skip_recall"],
                })
    return rows, thresholds, deployable


def binary_summary(
    pairs: list[tuple[np.void, np.void, str]], relation: str, calibration_samples: set[str]
) -> dict[str, object]:
    selected = [pair for pair in pairs if str(pair[1]["sample_id"]) not in calibration_samples]
    predicted = np.array([bool(source["exact_skip"]) for source, _, _ in selected])
    target = np.array([bool(target["exact_skip"]) for _, target, _ in selected])
    newmax = np.array([bool(target["introduced_new_max"]) for _, target, _ in selected])
    return {
        "policy": f"{relation}_binary",
        "false_skip_budget": "uncalibrated",
        "split": "held_out_samples",
        "layer": "all",
        "head": "all",
        "noise_bucket": "all",
        "threshold": "source exact skip mask",
        **prediction_metrics(predicted, target, newmax),
    }


def conservative_intersection(
    records: np.ndarray, calibration_samples: set[str]
) -> list[dict[str, object]]:
    relations = [joined_pairs(records, name) for name in ("previous_step", "previous_layer")]
    maps = []
    for pairs in relations:
        maps.append({
            (
                str(target["sample_id"]), int(target["denoising_iteration"]), int(target["layer"]),
                int(target["head"]), int(target["query_tile"]), int(target["kv_tile"]),
            ): (source, target)
            for source, target, _ in pairs
        })
    common = maps[0].keys() & maps[1].keys()
    selected = [key for key in common if key[0] not in calibration_samples]
    predicted = np.array([bool(maps[0][key][0]["exact_skip"]) and bool(maps[1][key][0]["exact_skip"]) for key in selected])
    target = np.array([bool(maps[0][key][1]["exact_skip"]) for key in selected])
    newmax = np.array([bool(maps[0][key][1]["introduced_new_max"]) for key in selected])
    return [{
        "policy": "previous_step_and_previous_layer_binary",
        "false_skip_budget": "uncalibrated",
        "split": "held_out_samples",
        "layer": "all", "head": "all", "noise_bucket": "all",
        "threshold": "both source masks skip",
        **prediction_metrics(predicted, target, newmax),
    }]


def combined_max_pairs(records: np.ndarray) -> list[tuple[dict[str, object], np.void, str]]:
    """Join previous-step and previous-layer scores using a monotonic max."""
    maps = []
    for relation in ("previous_step", "previous_layer"):
        maps.append({
            (
                str(target["sample_id"]), int(target["denoising_iteration"]), int(target["layer"]),
                int(target["head"]), int(target["query_tile"]), int(target["kv_tile"]),
            ): (source, target)
            for source, target, _ in joined_pairs(records, relation)
        })
    result = []
    for key in maps[0].keys() & maps[1].keys():
        previous_step_source, target = maps[0][key]
        previous_layer_source, _ = maps[1][key]
        result.append((
            {
                "score": max(float(previous_step_source["score"]), float(previous_layer_source["score"])),
                "noise_bucket": str(previous_step_source["noise_bucket"]),
                "exact_skip": bool(previous_step_source["exact_skip"] and previous_layer_source["exact_skip"]),
            },
            target,
            "max_previous_step_previous_layer",
        ))
    return result


def combined_source_policies(
    records: np.ndarray,
    calibration_samples: set[str],
    budgets: list[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Calibrate monotonic max-score and a two-feature logistic ablation."""
    maps = []
    for relation in ("previous_step", "previous_layer"):
        maps.append({
            (
                str(target["sample_id"]), int(target["denoising_iteration"]), int(target["layer"]),
                int(target["head"]), int(target["query_tile"]), int(target["kv_tile"]),
            ): (source, target)
            for source, target, _ in joined_pairs(records, relation)
        })
    keys = sorted(maps[0].keys() & maps[1].keys())
    if not keys:
        return [], {}
    features = np.array([
        [float(maps[0][key][0]["log_score"]), float(maps[1][key][0]["log_score"])] for key in keys
    ])
    features = np.nan_to_num(features, neginf=-32.0, posinf=32.0).clip(-32.0, 32.0)
    scores = np.exp(features.clip(-30.0, 30.0)).max(axis=1)
    target = np.array([bool(maps[0][key][1]["exact_skip"]) for key in keys])
    newmax = np.array([bool(maps[0][key][1]["introduced_new_max"]) for key in keys])
    calibration = np.array([key[0] in calibration_samples for key in keys])
    rows: list[dict[str, object]] = []
    parameters: dict[str, object] = {}
    for budget in budgets:
        threshold = select_threshold(scores[calibration], target[calibration], budget)
        predicted = scores[~calibration] < threshold
        rows.append({
            "policy": "max_previous_step_previous_layer_score",
            "false_skip_budget": budget,
            "split": "held_out_samples", "layer": "all", "head": "all", "noise_bucket": "all",
            "threshold": threshold,
            **prediction_metrics(predicted, target[~calibration], newmax[~calibration]),
        })
        parameters[f"max_score|{budget:g}"] = threshold

    # Small logistic model: two standardized source log-scores, full-batch
    # gradient descent, and a separately calibrated selective cutoff.
    x_train, y_train = features[calibration], target[calibration].astype(float)
    mean, scale = x_train.mean(0), x_train.std(0).clip(1e-6)
    x_standard = (x_train - mean) / scale
    weights, bias = np.zeros(2), 0.0
    for _ in range(400):
        logits = np.clip(x_standard @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y_train
        weights -= 0.05 * (x_standard.T @ error / len(error) + 1e-4 * weights)
        bias -= 0.05 * float(error.mean())
    evaluation_probability = 1.0 / (
        1.0 + np.exp(-np.clip(((features[~calibration] - mean) / scale) @ weights + bias, -30.0, 30.0))
    )
    train_probability = 1.0 / (1.0 + np.exp(-np.clip(x_standard @ weights + bias, -30.0, 30.0)))
    parameters["logistic"] = {
        "relations": ["previous_step", "previous_layer"],
        "weights_standardized": weights.tolist(), "bias": bias,
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
    }
    for budget in budgets:
        # select_threshold expects low score => skip, so negate probability.
        negative_cutoff = select_threshold(-train_probability, target[calibration], budget)
        probability_cutoff = -negative_cutoff
        predicted = evaluation_probability > probability_cutoff
        rows.append({
            "policy": "logistic_previous_step_previous_layer",
            "false_skip_budget": budget,
            "split": "held_out_samples", "layer": "all", "head": "all", "noise_bucket": "all",
            "threshold": probability_cutoff,
            **prediction_metrics(predicted, target[~calibration], newmax[~calibration]),
        })
        parameters[f"logistic_probability|{budget:g}"] = probability_cutoff
    return rows, parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--output-dir", default="artifacts/proxy_calibration")
    parser.add_argument("--false-skip-budgets", default="1e-2,1e-3,1e-4,1e-5")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--min-predictions", type=int, default=512)
    parser.add_argument(
        "--new-max-budget-scale", type=float, default=0.1,
        help="new-maximum false-skip budget as a fraction of the total false-skip budget",
    )
    parser.add_argument("--warmup-tiles", type=int, default=2)
    parser.add_argument("--periodic-refresh", type=int, default=8)
    parser.add_argument("--no-local-anchor", action="store_true")
    parser.add_argument("--no-diagonal-anchor", action="store_true")
    parser.add_argument("--no-sink-anchor", action="store_true")
    parser.add_argument(
        "--previous-step-only", action="store_true",
        help="calibrate only the relation implemented by the pre-QK runtime kernel",
    )
    parser.add_argument(
        "--previous-layer-only", action="store_true",
        help="calibrate only same-step scores from the immediately previous layer",
    )
    args = parser.parse_args()
    if args.previous_step_only and args.previous_layer_only:
        parser.error("choose at most one relation-only mode")
    trace_manifest = json.loads((Path(args.trace_dir) / "manifest.json").read_text(encoding="utf-8"))
    trace_schema_version = str(trace_manifest.get("schema_version", "unknown"))
    deployment_schema_eligible = trace_schema_version == SCHEMA_VERSION
    records = load_trace_directory(args.trace_dir)
    calibration_samples = _split_samples(records)
    budgets = [float(value) for value in args.false_skip_budgets.split(",")]
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    deployable_thresholds: list[dict[str, object]] = []
    leaders: dict[str, int] = {}
    combined_parameters: dict[str, object] = {}

    def finish() -> None:
        if not deployment_schema_eligible:
            for row in deployable_thresholds:
                row["certified"] = False
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        columns = sorted({key for row in rows for key in row})
        with (output / "policy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        payload = {
            "calibration_samples": sorted(calibration_samples),
            "evaluation_samples": sorted(set(str(value) for value in records["sample_id"]) - calibration_samples),
            "false_skip_budgets": budgets,
            "confidence": args.confidence,
            "minimum_predictions_per_threshold": args.min_predictions,
            "new_max_budget_scale": args.new_max_budget_scale,
            "safety_filter": {
                "warmup_tiles": args.warmup_tiles,
                "periodic_refresh": args.periodic_refresh,
                "anchor_local": not args.no_local_anchor,
                "anchor_diagonal": not args.no_diagonal_anchor,
                "anchor_sink": not args.no_sink_anchor,
            },
            "threshold_semantics": "PRE_SKIP iff source score < threshold",
            "trace_schema_version": trace_schema_version,
            "deployment_schema_eligible": deployment_schema_eligible,
            "leaders": leaders,
            "thresholds": thresholds,
            "combined_policy_parameters": combined_parameters,
            "deployable_thresholds": deployable_thresholds,
            "policies": rows,
        }
        (output / "policy_summary.json").write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        certified = [row for row in deployable_thresholds if row["certified"]]
        summary = {
            "trace_records": int(len(records)),
            "calibration_samples": len(calibration_samples),
            "evaluation_samples": len(payload["evaluation_samples"]),
            "candidate_thresholds": len(deployable_thresholds),
            "certified_thresholds": len(certified),
            "trace_schema_version": trace_schema_version,
            "deployment_schema_eligible": deployment_schema_eligible,
            "certified_by_budget": {
                str(budget): sum(
                    bool(row["certified"]) and math.isclose(float(row["false_skip_budget"]), budget)
                    for row in deployable_thresholds
                )
                for budget in budgets
            },
        }
        (output / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    if args.previous_step_only:
        relations = ("previous_step",)
    elif args.previous_layer_only:
        relations = ("previous_layer",)
    else:
        relations = ("previous_step", "previous_layer")
    for relation in relations:
        pairs = joined_pairs(records, relation)
        rows.append(binary_summary(pairs, relation, calibration_samples))
        safe_pairs = apply_safety_filter(
            pairs,
            warmup_tiles=args.warmup_tiles,
            periodic_refresh=args.periodic_refresh,
            anchor_local=not args.no_local_anchor,
            anchor_diagonal=not args.no_diagonal_anchor,
            anchor_sink=not args.no_sink_anchor,
        )
        relation_rows, relation_thresholds, relation_deployable = calibrate_relation(
            safe_pairs, relation, budgets, calibration_samples,
            confidence=args.confidence,
            min_predictions=args.min_predictions,
            new_max_budget_scale=args.new_max_budget_scale,
        )
        rows.extend(relation_rows)
        thresholds.update(relation_thresholds)
        if not deployment_schema_eligible:
            for row in relation_deployable:
                row["certified"] = False
        deployable_thresholds.extend(relation_deployable)
    if args.previous_step_only or args.previous_layer_only:
        finish()
        return
    max_pairs = combined_max_pairs(records)
    safe_max_pairs = apply_safety_filter(
        max_pairs,
        warmup_tiles=args.warmup_tiles,
        periodic_refresh=args.periodic_refresh,
        anchor_local=not args.no_local_anchor,
        anchor_diagonal=not args.no_diagonal_anchor,
        anchor_sink=not args.no_sink_anchor,
    )
    max_rows, max_thresholds, max_deployable = calibrate_relation(
        safe_max_pairs, "max_previous_step_previous_layer", budgets, calibration_samples,
        confidence=args.confidence,
        min_predictions=args.min_predictions,
        new_max_budget_scale=args.new_max_budget_scale,
    )
    rows.extend(max_rows)
    thresholds.update(max_thresholds)
    deployable_thresholds.extend(max_deployable)
    rows.extend(conservative_intersection(records, calibration_samples))
    combined_rows, combined_parameters = combined_source_policies(records, calibration_samples, budgets)
    rows.extend(combined_rows)

    # Leader selection is calibrated independently for each target
    # layer/head/noise tuple and constrained to legally earlier heads.
    earlier = joined_pairs(records, "earlier_heads")
    leader_candidates: dict[tuple[int, int, str, int], list[tuple[np.void, np.void, str]]] = {}
    for pair in earlier:
        source, target, label = pair
        group = (int(target["layer"]), int(target["head"]), str(target["noise_bucket"]), int(source["head"]))
        leader_candidates.setdefault(group, []).append(pair)
    leader_pairs = []
    target_groups = sorted({group[:3] for group in leader_candidates})
    for target_group in target_groups:
        candidates = []
        for group, pairs in leader_candidates.items():
            if group[:3] != target_group:
                continue
            calibration = [pair for pair in pairs if str(pair[1]["sample_id"]) in calibration_samples]
            predicted = sum(bool(pair[0]["exact_skip"]) for pair in calibration)
            correct = sum(bool(pair[0]["exact_skip"]) and bool(pair[1]["exact_skip"]) for pair in calibration)
            candidates.append((correct / predicted if predicted else -1.0, correct, -group[3], group[3], pairs))
        if candidates:
            chosen = max(candidates)
            leaders["|".join(map(str, target_group))] = chosen[3]
            leader_pairs.extend(chosen[4])
    rows.append(binary_summary(leader_pairs, "leader_head", calibration_samples))
    safe_leader_pairs = apply_safety_filter(
        leader_pairs,
        warmup_tiles=args.warmup_tiles, periodic_refresh=args.periodic_refresh,
        anchor_local=not args.no_local_anchor,
        anchor_diagonal=not args.no_diagonal_anchor,
        anchor_sink=not args.no_sink_anchor,
    )
    leader_rows, leader_thresholds, leader_deployable = calibrate_relation(
        safe_leader_pairs, "leader_head", budgets, calibration_samples,
        confidence=args.confidence,
        min_predictions=args.min_predictions,
        new_max_budget_scale=args.new_max_budget_scale,
    )
    rows.extend(leader_rows)
    thresholds.update(leader_thresholds)
    deployable_thresholds.extend(leader_deployable)

    # A hardware-friendly cluster uses only leader pairs whose calibration
    # skip-mask precision is at least 0.8. Followers without such a leader
    # conservatively fall back to EXACT.
    clustered_pairs = []
    for group, pairs in leader_candidates.items():
        calibration_pairs = [pair for pair in pairs if str(pair[1]["sample_id"]) in calibration_samples]
        predicted = sum(bool(pair[0]["exact_skip"]) for pair in calibration_pairs)
        correct = sum(bool(pair[0]["exact_skip"]) and bool(pair[1]["exact_skip"]) for pair in calibration_pairs)
        if predicted and correct / predicted >= 0.8 and leaders.get("|".join(map(str, group[:3]))) == group[3]:
            clustered_pairs.extend(pairs)
    rows.append(binary_summary(clustered_pairs, "clustered_head", calibration_samples))
    safe_clustered_pairs = apply_safety_filter(
        clustered_pairs,
        warmup_tiles=args.warmup_tiles, periodic_refresh=args.periodic_refresh,
        anchor_local=not args.no_local_anchor,
        anchor_diagonal=not args.no_diagonal_anchor,
        anchor_sink=not args.no_sink_anchor,
    )
    clustered_rows, clustered_thresholds, clustered_deployable = calibrate_relation(
        safe_clustered_pairs, "clustered_head", budgets, calibration_samples,
        confidence=args.confidence,
        min_predictions=args.min_predictions,
        new_max_budget_scale=args.new_max_budget_scale,
    )
    rows.extend(clustered_rows)
    thresholds.update(clustered_thresholds)
    deployable_thresholds.extend(clustered_deployable)

    finish()


if __name__ == "__main__":
    main()
