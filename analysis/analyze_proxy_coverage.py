#!/usr/bin/env python3
"""Explain binary coverage limits and optimistic continuous-score headroom."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.proxy_relationships import joined_pairs, json_safe, split_is_calibration
from proxy.calibrate_proxy_thresholds import apply_safety_filter
from tracing.trace_schema import load_trace_directory


def metrics(predicted: np.ndarray, target: np.ndarray, newmax: np.ndarray) -> dict[str, float | int]:
    predicted, target, newmax = predicted.astype(bool), target.astype(bool), newmax.astype(bool)
    correct = int((predicted & target).sum())
    false = int((predicted & ~target).sum())
    ratio = lambda a, b: a / b if b else 0.0
    return {
        "candidate_tiles": int(predicted.sum()),
        "candidate_rate": ratio(int(predicted.sum()), len(predicted)),
        "correct_pre_qk_sparsity": ratio(correct, len(predicted)),
        "precision": ratio(correct, int(predicted.sum())),
        "oracle_skip_recall": ratio(correct, int(target.sum())),
        "false_skips": false,
        "false_new_max": int((predicted & ~target & newmax).sum()),
    }


def optimistic_prefix(
    scores: np.ndarray,
    target: np.ndarray,
    newmax: np.ndarray,
    required_precision: float,
) -> dict[str, float | int]:
    """Diagnostic only: tune directly on evaluation labels to measure headroom."""
    finite_indices = np.flatnonzero(np.isfinite(scores))
    if not len(finite_indices):
        return {"required_precision": required_precision, "threshold": None, **metrics(
            np.zeros(len(target), dtype=bool), target, newmax
        )}
    order = finite_indices[np.argsort(scores[finite_indices], kind="mergesort")]
    ordered_scores, ordered_target = scores[order], target[order]
    correct = np.cumsum(ordered_target)
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    eligible = [end for end in ends if correct[end] / (end + 1) >= required_precision]
    if not eligible:
        return {"required_precision": required_precision, "threshold": None, **metrics(
            np.zeros(len(target), dtype=bool), target, newmax
        )}
    end = max(eligible)
    predicted = np.zeros(len(target), dtype=bool)
    predicted[order[: end + 1]] = True
    return {
        "required_precision": required_precision,
        "threshold": float(np.nextafter(ordered_scores[end], np.inf)),
        **metrics(predicted, target, newmax),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--output-dir", default="artifacts/proxy_analysis")
    parser.add_argument("--precisions", default="0.9,0.99,0.999,0.9999")
    parser.add_argument("--warmup-tiles", type=int, default=2)
    parser.add_argument("--periodic-refresh", type=int, default=8)
    args = parser.parse_args()
    records = load_trace_directory(args.trace_dir)
    pairs = joined_pairs(records, "previous_step")
    evaluation = [pair for pair in pairs if not split_is_calibration(str(pair[1]["sample_id"]))]
    if not evaluation:
        samples = sorted({str(pair[1]["sample_id"]) for pair in pairs})
        held_out = set(samples[len(samples) // 2 :])
        evaluation = [pair for pair in pairs if str(pair[1]["sample_id"]) in held_out]
    precisions = [float(value) for value in args.precisions.split(",")]
    transitions = sorted({(str(a["noise_bucket"]), str(b["noise_bucket"])) for a, b, _ in evaluation})
    rows: list[dict[str, object]] = []
    for source_bucket, target_bucket in transitions:
        selected = [pair for pair in evaluation if str(pair[0]["noise_bucket"]) == source_bucket and str(pair[1]["noise_bucket"]) == target_bucket]
        safe = apply_safety_filter(
            selected, warmup_tiles=args.warmup_tiles, periodic_refresh=args.periodic_refresh,
            anchor_local=True, anchor_diagonal=True, anchor_sink=True,
        )
        source_skip = np.array([bool(a["exact_skip"]) for a, _, _ in selected])
        target = np.array([bool(b["exact_skip"]) for _, b, _ in selected])
        newmax = np.array([bool(b["introduced_new_max"]) for _, b, _ in selected])
        forced_keys = {
            (str(b["sample_id"]), int(b["layer"]), int(b["head"]), int(b["query_tile"]), int(b["kv_tile"]))
            for _, b, _ in selected
        } - {
            (str(b["sample_id"]), int(b["layer"]), int(b["head"]), int(b["query_tile"]), int(b["kv_tile"]))
            for _, b, _ in safe
        }
        allowed = np.array([
            (str(b["sample_id"]), int(b["layer"]), int(b["head"]), int(b["query_tile"]), int(b["kv_tile"])) not in forced_keys
            for _, b, _ in selected
        ])
        exact_skips = int(target.sum())
        base = {
            "source_noise_bucket": source_bucket,
            "target_noise_bucket": target_bucket,
            "tiles": len(selected),
            "oracle_exact_skip_rate": exact_skips / len(selected) if selected else 0.0,
        }
        rows.append({**base, "policy": "previous_step_binary_raw", **metrics(source_skip, target, newmax)})
        rows.append({**base, "policy": "previous_step_binary_safe", **metrics(source_skip & allowed, target, newmax)})
        scores = np.array([float(a["score"]) for a, _, _ in selected])
        # Anchored tiles are made unselectable while preserving the all-tile denominator.
        scores = np.where(allowed, scores, np.inf)
        for precision in precisions:
            rows.append({
                **base,
                "policy": "continuous_score_optimistic_headroom",
                **optimistic_prefix(scores, target, newmax, precision),
            })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with (output / "coverage_headroom.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "warning": "continuous-score headroom tunes on held-out labels and is an optimistic diagnostic, not a deployable calibration",
        "rows": rows,
    }
    (output / "coverage_headroom.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
