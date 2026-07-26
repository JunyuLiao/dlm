#!/usr/bin/env python3
"""Summarize causal temporal-certificate omission traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(trace_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    plan = json.loads((trace_dir / "collection_plan.json").read_text())
    chunks: dict[str, list[np.ndarray]] = {}
    for path in sorted(trace_dir.glob("context-*/tile-reuse-summary.npz")):
        with np.load(path) as payload:
            for name in payload.files:
                if name != "schema_version":
                    chunks.setdefault(name, []).append(payload[name])
    if not chunks:
        raise ValueError(f"no trace summaries found in {trace_dir}")
    return {name: np.concatenate(values) for name, values in chunks.items()}, plan


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def phase(ratio: np.ndarray) -> np.ndarray:
    return np.where(ratio >= 0.75, "high", np.where(ratio >= 0.25, "mid", "low"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, plan = load(args.trace_dir)
    budgets = [float(value) for value in plan["certificate_mass_budgets"]]
    identity_names = ("request_id", "layer", "head", "step", "query_tile")
    identity = np.rec.fromarrays(
        [data[name] for name in identity_names], names=",".join(identity_names)
    )
    _, group_first = np.unique(identity, return_index=True)
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    breakdown: dict[str, dict[str, list[dict[str, object]]]] = {}

    def metrics(budget: float, scope: np.ndarray) -> dict[str, object]:
        suffix = tag(budget)
        tile_scope = scope
        scoped_groups = group_first[scope[group_first]]
        omitted = data[f"cert_omit_b{suffix}"][tile_scope].astype(bool)
        errors = data[f"cert_group_output_error_max_b{suffix}"][scoped_groups]
        actual_mass = data[f"cert_group_actual_omitted_mass_max_b{suffix}"][scoped_groups]
        upper = data[f"cert_group_aggregate_upper_max_b{suffix}"][scoped_groups]
        interval_violation = data[f"cert_group_interval_violation_max_b{suffix}"][
            scoped_groups
        ]
        mass_violation = data[f"cert_group_mass_violation_max_b{suffix}"][scoped_groups]
        widths = data[f"cert_group_interval_width_max_b{suffix}"][scoped_groups]
        return {
            "mass_budget": budget,
            "tile_records": int(tile_scope.sum()),
            "query_groups": int(len(scoped_groups)),
            "omitted_tiles": int(omitted.sum()),
            "omission_fraction": float(omitted.mean()) if omitted.size else 0.0,
            "groups_with_any_omission": float(np.mean(errors > 0)) if errors.size else 0.0,
            "safe_group_fraction_at_1pct": float(np.mean(errors <= 0.01))
            if errors.size
            else 1.0,
            "median_group_output_error": float(np.median(errors)) if errors.size else 0.0,
            "p99_group_output_error": float(np.quantile(errors, 0.99))
            if errors.size
            else 0.0,
            "maximum_group_output_error": float(errors.max(initial=0.0)),
            "maximum_actual_omitted_mass": float(actual_mass.max(initial=0.0)),
            "maximum_aggregate_mass_upper": float(upper.max(initial=0.0)),
            "maximum_interval_width": float(widths.max(initial=0.0)),
            "maximum_interval_violation": float(interval_violation.max(initial=0.0)),
            "maximum_mass_bound_violation": float(mass_violation.max(initial=0.0)),
            "vmax_normalized_error_bound": 2.0 * budget,
        }

    all_scope = np.ones(len(data["step"]), dtype=bool)
    for budget in budgets:
        rows.append(metrics(budget, all_scope))
        suffix = tag(budget)
        for policy, mask_field, error_field in (
            (
                "temporal_upper_with_exact_current_denominator",
                f"diag_exact_den_omit_b{suffix}",
                f"diag_exact_den_output_error_max_b{suffix}",
            ),
            (
                "oracle_exact_current_mass",
                f"oracle_mass_omit_b{suffix}",
                f"oracle_mass_output_error_max_b{suffix}",
            ),
        ):
            mask = data[mask_field].astype(bool)
            errors = data[error_field][group_first]
            diagnostic_rows.append(
                {
                    "policy": policy,
                    "mass_budget": budget,
                    "omission_fraction": float(mask.mean()),
                    "safe_group_fraction_at_1pct": float(np.mean(errors <= 0.01)),
                    "p99_group_output_error": float(np.quantile(errors, 0.99)),
                    "maximum_group_output_error": float(errors.max(initial=0.0)),
                }
            )
        breakdown[suffix] = {"layer": [], "phase": []}
        for layer in np.unique(data["layer"]):
            row = metrics(budget, data["layer"] == layer)
            row["layer"] = int(layer)
            breakdown[suffix]["layer"].append(row)
        phases = phase(data["remaining_mask_ratio"])
        for name in ("high", "mid", "low"):
            row = metrics(budget, phases == name)
            row["phase"] = name
            breakdown[suffix]["phase"].append(row)

    sequence = int(plan["context_length"])
    layers = 32 if plan["layers"] is None else len(plan["layers"])
    # Native full-model memory, independent of the sampled pilot layer count.
    full_layers, heads, dim = 32, 32, 128
    physical_tiles = (sequence // 128) * (sequence // 64) * full_layers * heads
    result = {
        "trace_dir": str(args.trace_dir),
        "schema": plan["schema"],
        "sampled_layers": layers,
        "sampled_heads": len(plan["heads"]),
        "records": len(data["step"]),
        "budgets": rows,
        "bound_decomposition": diagnostic_rows,
        "breakdown": breakdown,
        "full_model_metadata_model": {
            "physical_tiles_batch1": physical_tiles,
            "previous_q_plus_k_bf16_gib_batch1": (
                2 * full_layers * sequence * heads * dim * 2 / 2**30
            ),
            "logz_lower_upper_fp32_gib_batch1": (
                physical_tiles * 128 * 2 * 4 / 2**30
            ),
            "logz_lower_upper_bf16_gib_batch1": (
                physical_tiles * 128 * 2 * 2 / 2**30
            ),
            "row_tile_bound_evaluations_per_step_batch1": physical_tiles * 128,
            "note": (
                "BF16 intervals require outward rounding/error inflation to remain certified; "
                "the reference analysis uses FP32 intervals."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
