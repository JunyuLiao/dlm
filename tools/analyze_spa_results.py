#!/usr/bin/env python3
"""Aggregate layer/step SPA oracle records into a compact gate report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
        by_layer[int(row["layer"])].append(row)

    steps = []
    for step, selected in sorted(by_step.items()):
        temporal = [
            float(row["previous_step_support_jaccard"])
            for row in selected if row["previous_step_support_jaccard"] is not None
        ]
        cross_layer = [
            float(row["previous_layer_support_jaccard"])
            for row in selected if row["previous_layer_support_jaccard"] is not None
        ]
        steps.append({
            "step": step,
            "worst_p1_retained_mass": min(float(row["p1_retained_mass"]) for row in selected),
            "worst_mean_retained_mass": min(float(row["mean_retained_mass"]) for row in selected),
            "maximum_output_relative_l2": max(float(row["output_relative_l2"]) for row in selected),
            "mean_adjacent_group_jaccard": statistics.mean(
                float(row["adjacent_group_support_jaccard"]) for row in selected
            ),
            "mean_previous_step_jaccard": statistics.mean(temporal) if temporal else None,
            "mean_previous_layer_jaccard": statistics.mean(cross_layer) if cross_layer else None,
        })

    layers = []
    for layer, selected in sorted(by_layer.items()):
        layers.append({
            "layer": layer,
            "worst_p1_retained_mass": min(float(row["p1_retained_mass"]) for row in selected),
            "worst_mean_retained_mass": min(float(row["mean_retained_mass"]) for row in selected),
            "maximum_output_relative_l2": max(float(row["output_relative_l2"]) for row in selected),
        })

    token_states = {}
    for field in ("masked_retained_mass", "revealed_retained_mass"):
        values = [row[field] for row in rows if row[field] is not None]
        token_states[field.removesuffix("_retained_mass")] = {
            "worst_p1_retained_mass": min(float(value["p1"]) for value in values),
            "worst_mean_retained_mass": min(float(value["mean"]) for value in values),
            "minimum_row_retained_mass": min(float(value["minimum"]) for value in values),
        } if values else None
    for field in ("masked_output_error", "revealed_output_error"):
        values = [row[field] for row in rows if row.get(field) is not None]
        state = field.removesuffix("_output_error")
        if values and token_states.get(state) is not None:
            token_states[state]["maximum_output_relative_l2"] = max(
                float(value["relative_l2"]) for value in values
            )
            token_states[state]["minimum_output_cosine_similarity"] = min(
                float(value["cosine_similarity"]) for value in values
            )

    individually_eligible_layers = [
        row["layer"] for row in layers
        if row["worst_p1_retained_mass"] >= 0.95
        and row["worst_mean_retained_mass"] >= 0.99
    ]
    return {
        "steps": steps,
        "layers": layers,
        "token_states": token_states,
        "individually_mass_gate_eligible_layers": individually_eligible_layers,
        "sensitive_layer_exclusion_conclusion": (
            "No sparse layer remains after exclusions."
            if not individually_eligible_layers
            else "Only the listed layers satisfy the mass gate independently."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = {
        "source": str(args.input),
        "configurations": payload["configs"],
        **aggregate(payload["oracle_records"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
