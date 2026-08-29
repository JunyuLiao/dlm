#!/usr/bin/env python3
"""Fit BLASST Algorithm 2 thresholds by attention type and denoising phase."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "masked_only_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)
PHASE_NAMES = {0: "first", 1: "middle", 2: "late"}


def _lambda_from_dir(path: Path) -> float:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    return float(config["calibration_lambda"])


def _fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        raise ValueError("at least two calibration points are required")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance == 0:
        raise ValueError("calibration sparsities have zero variance")
    beta = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    intercept = mean_y - beta * mean_x
    residual = sum((y - (intercept + beta * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    r_squared = 1.0 - residual / total if total else 1.0
    return intercept, beta, r_squared


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--min-fit-sparsity", type=float, default=0.02)
    parser.add_argument("--max-fit-sparsity", type=float, default=0.9)
    args = parser.parse_args()
    root = args.calibration_run.resolve() / "attention_calibration"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.target_sparsity < 1.0:
        raise ValueError("target sparsity must be strictly between 0 and 1")

    lambda_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=_lambda_from_dir,
    )
    if not lambda_dirs:
        raise RuntimeError(f"no calibration exports found under {root}")

    aggregate: dict[tuple[float, str, int], dict[str, float]] = {}
    sample_points: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    reference_lengths: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    aggregate_curves: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    curve_rows: list[dict] = []
    for directory in lambda_dirs:
        lambda_value = _lambda_from_dir(directory)
        grouped: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        with (directory / "per_layer.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                attention_type = row["attention_type"]
                iteration = int(row["denoising_iteration"])
                phase = 0 if iteration < 1 else (1 if iteration < 3 else 2)
                if attention_type not in ("local", "global") or phase not in PHASE_NAMES:
                    raise RuntimeError(
                        f"unexpected calibration group: {attention_type}, phase {phase}"
                    )
                key = (attention_type, phase, row["example_id"])
                for field in COUNT_FIELDS:
                    grouped[key][field] += int(row[field])
                eligible = int(row["eligible_tiles"])
                grouped[key]["length_weighted"] += int(row["sequence_length"]) * eligible

        type_phase_totals: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for (attention_type, phase, _), counts in grouped.items():
            eligible = counts["eligible_tiles"]
            if not eligible:
                continue
            sparsity = counts["skipped_tiles"] / eligible
            effective_length = counts["length_weighted"] / eligible
            group_key = (attention_type, phase)
            reference_lengths[group_key].append((effective_length, eligible))
            if args.min_fit_sparsity <= sparsity <= args.max_fit_sparsity:
                sample_points[group_key].append(
                    (sparsity, math.log(lambda_value * effective_length))
                )
            target = type_phase_totals[group_key]
            for field in COUNT_FIELDS:
                target[field] += counts[field]
            target["length_weighted"] += counts["length_weighted"]

        for (attention_type, phase), counts in sorted(type_phase_totals.items()):
            eligible = counts["eligible_tiles"]
            curve_rows.append(
                {
                    "lambda": lambda_value,
                    "attention_type": attention_type,
                    "phase": phase,
                    "phase_name": PHASE_NAMES[phase],
                    "eligible_tiles": int(eligible),
                    "skipped_tiles": int(counts["skipped_tiles"]),
                    "physical_sparsity": counts["skipped_tiles"] / eligible,
                    "row_vote_sparsity": (
                        counts["skippable_row_votes"] / counts["valid_row_votes"]
                    ),
                    "effective_kv_length": counts["length_weighted"] / eligible,
                }
            )
            aggregate_curves[(attention_type, phase)].append(
                (counts["skipped_tiles"] / eligible, lambda_value)
            )

    fits: dict[str, dict] = {}
    selected: dict[tuple[str, int], float] = {}
    for group_key in sorted(sample_points):
        attention_type, phase = group_key
        intercept, beta, r_squared = _fit(sample_points[group_key])
        lengths = reference_lengths[group_key]
        reference_length = sum(length * weight for length, weight in lengths) / sum(
            weight for _, weight in lengths
        )
        algorithm2_threshold = (
            math.exp(intercept + beta * args.target_sparsity) / reference_length
        )
        curve = sorted(aggregate_curves[group_key])
        lower = next(
            (point for point in reversed(curve) if point[0] <= args.target_sparsity),
            None,
        )
        upper = next(
            (point for point in curve if point[0] >= args.target_sparsity),
            None,
        )
        if lower is None or upper is None:
            raise RuntimeError(
                f"target {args.target_sparsity:.1%} is outside the observed range "
                f"{curve[0][0]:.1%}-{curve[-1][0]:.1%} for {group_key}"
            )
        if lower[0] == upper[0]:
            threshold = lower[1]
        else:
            fraction = (args.target_sparsity - lower[0]) / (upper[0] - lower[0])
            threshold = math.exp(
                math.log(lower[1])
                + fraction * (math.log(upper[1]) - math.log(lower[1]))
            )
        if not 0.0 < threshold < 1.0:
            raise RuntimeError(
                f"fitted threshold {threshold} is outside (0, 1) for {group_key}"
            )
        name = f"{attention_type}_{PHASE_NAMES[phase]}"
        selected[group_key] = threshold
        fits[name] = {
            "attention_type": attention_type,
            "phase": phase,
            "phase_name": PHASE_NAMES[phase],
            "alpha": math.exp(intercept),
            "beta": beta,
            "r_squared": r_squared,
            "fit_points": len(sample_points[group_key]),
            "reference_kv_length": reference_length,
            "target_sparsity": args.target_sparsity,
            "algorithm2_global_fit_lambda": algorithm2_threshold,
            "selection_method": "piecewise Algorithm 2 interpolation on aggregate curve",
            "bracketing_points": [lower, upper],
            "selected_lambda": threshold,
        }

    policy = {
        "denoising_phase_starts": [1, 3],
        "local_phase_lambdas": [selected[("local", phase)] for phase in range(3)],
        "global_phase_lambdas": [selected[("global", phase)] for phase in range(3)],
    }
    _write_csv(output / "calibration_curve.csv", curve_rows)
    (output / "fits.json").write_text(
        json.dumps(fits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DiffusionGemma BLASST calibration",
        "",
        f"Target semantic physical sparsity: `{args.target_sparsity:.0%}`.",
        "Algorithm 2 fit: `lambda * L = alpha * exp(beta * sparsity)`; the deployed threshold uses the two bracketing calibration points to avoid global-fit bias.",
        "",
        "| Attention | Phase | Lambda | Fit R2 | Reference KV |",
        "|---|---|---:|---:|---:|",
    ]
    for name in sorted(fits):
        fit = fits[name]
        lines.append(
            f"| {fit['attention_type']} | {fit['phase_name']} | "
            f"{fit['selected_lambda']:.6g} | {fit['r_squared']:.4f} | "
            f"{fit['reference_kv_length']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Phases are iteration 0 (`first`), iterations 1-2 (`middle`), and iteration 3+ (`late`).",
            "The calibration trajectory is dense eager; all lambda candidates share exactly the same QK scores.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
