#!/usr/bin/env python3
"""Validate runtime defaults against the existing physical calibration JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst import DiffusionLambdaSchedule, calibration_matches_runtime_defaults, load_physical_calibration


TARGETS = {
    0.15: {"physical_sparsity": 0.4838, "dense_agreement": 0.9692},
    0.5: {"physical_sparsity": 0.2923, "dense_agreement": 0.9586},
    0.9: {"physical_sparsity": 0.0986, "dense_agreement": 0.9551},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=pathlib.Path,
        default=ROOT / "outputs" / "blasst_physical_calibration_4096.json",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    calibration = load_physical_calibration(args.artifact)
    schedule = DiffusionLambdaSchedule()
    checks: dict[str, bool] = {
        "runtime_lambdas_match_selected_entries_exactly": calibration_matches_runtime_defaults(
            calibration,
            high_noise_lambda=schedule.high_noise_lambda,
            mid_noise_lambda=schedule.mid_noise_lambda,
            low_noise_lambda=schedule.low_noise_lambda,
        ),
        "tile_counts_match": calibration.skipped_tiles == 3_668_691
        and calibration.total_tiles == 12_582_912,
        "overall_physical_sparsity_matches": abs(calibration.overall_physical_sparsity - 0.2916)
        <= args.absolute_tolerance,
    }
    for ratio, targets in TARGETS.items():
        checks[f"{ratio:g}_physical_sparsity_matches"] = (
            abs(calibration.physical_sparsity[ratio] - targets["physical_sparsity"])
            <= args.absolute_tolerance
        )
        checks[f"{ratio:g}_dense_agreement_matches"] = (
            abs(calibration.dense_agreement[ratio] - targets["dense_agreement"])
            <= args.absolute_tolerance
        )
        checks[f"{ratio:g}_meets_exact_0.95"] = calibration.dense_agreement[ratio] >= 0.95
    report = {
        "artifact": str(args.artifact),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selected_lambdas": calibration.lambdas,
        "physical_sparsity": calibration.physical_sparsity,
        "dense_agreement": calibration.dense_agreement,
        "overall": {
            "physical_sparsity": calibration.overall_physical_sparsity,
            "skipped_tiles": calibration.skipped_tiles,
            "total_tiles": calibration.total_tiles,
        },
        "timing_note": "artifact elapsed_ms is unfused calibration/reference-forward time, not production latency",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
