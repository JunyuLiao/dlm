#!/usr/bin/env python3
"""Causal simulation of the best previous-mass × current-drift reuse policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from analyze_tile_reuse import (  # noqa: E402
    apply_thresholds,
    calibrate_group_thresholds,
    load_summaries,
    phase,
    policy_row,
    previous_step_field,
    reuse_effect,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, split_map = load_summaries(args.trace_dir)
    split = np.asarray([split_map[str(value)] for value in data["request_id"]])
    effect = reuse_effect(data)
    safe = effect <= 0.01
    groups = phase(data["remaining_mask_ratio"]).astype(np.int16) * 4 + (
        data["layer"] // 8
    ).astype(np.int16)
    drift = data["q_max"] + data["k_max"] + 0.5 * data["v_max"]
    previous_mass = previous_step_field(data, "current_tile_mass_max")
    calibration_score = previous_mass * drift
    thresholds = calibrate_group_thresholds(
        calibration_score,
        safe,
        np.isfinite(calibration_score),
        groups,
        split == "calibration",
    )

    order = np.lexsort(
        (
            data["step"],
            data["kv_tile"],
            data["query_tile"],
            data["head"],
            data["layer"],
            data["request_id"],
        )
    )
    variants = [
        (protection, max_age, accumulation)
        for protection in ("none", "critical")
        for max_age in (0, 1, 2, 3)
        for accumulation in ("independent", "sum", "rss", "max")
    ]
    selected_by_variant = {variant: np.zeros(len(effect), dtype=bool) for variant in variants}
    state: dict[
        tuple[str, int, int, int, int],
        dict[tuple[str, int, str], tuple[float, int, float]],
    ] = {}
    for index in order:
        identity = (
            str(data["request_id"][index]),
            int(data["layer"][index]),
            int(data["head"][index]),
            int(data["query_tile"][index]),
            int(data["kv_tile"][index]),
        )
        per_variant = state.setdefault(identity, {})
        current_mass = float(data["current_tile_mass_max"][index])
        for protection, max_age, accumulation in variants:
            key = (protection, max_age, accumulation)
            previous = per_variant.get(key)
            critical = (
                data["query_state_0_fraction"][index]
                + data["query_state_1_fraction"][index]
                + data["kv_state_0_fraction"][index]
                + data["kv_state_1_fraction"][index]
                > 0
            )
            if previous is None:
                per_variant[key] = (current_mass, 0, 0.0)
                continue
            cached_mass, age, cumulative = previous
            score = cached_mass * float(drift[index])
            if accumulation == "sum":
                proposed = cumulative + score
            elif accumulation == "rss":
                proposed = float(np.sqrt(cumulative**2 + score**2))
            elif accumulation == "max":
                proposed = max(cumulative, score)
            else:
                proposed = score
            threshold = thresholds[int(groups[index])]
            reuse = (
                score < threshold
                if accumulation == "independent"
                else proposed < threshold
            )
            reuse &= not (protection == "critical" and critical)
            reuse &= max_age == 0 or age < max_age
            selected_by_variant[key][index] = reuse
            per_variant[key] = (
                (cached_mass, age + 1, proposed)
                if reuse
                else (current_mass, 0, 0.0)
            )

    rows: list[dict[str, object]] = []
    for variant, selected in selected_by_variant.items():
        protection, max_age, accumulation = variant
        name = (
            f"causal_previous_mass_x_qkv_b0.5_max:protection={protection}:"
            f"max_age={max_age or 'unlimited'}:accumulation={accumulation}"
        )
        for split_name in np.unique(split):
            rows.append(
                policy_row(name, str(split_name), selected, split == split_name, safe, effect)
            )
    heldout = sorted(
        [row for row in rows if row["split"] == "heldout"],
        key=lambda row: (
            row["unsafe_wilson95_upper"] <= 0.01,
            row["reuse_fraction"],
        ),
        reverse=True,
    )
    best_variant = ("none", 0, "sum")
    best_selected = selected_by_variant[best_variant]
    best_name = (
        "causal_previous_mass_x_qkv_b0.5_max:protection=none:"
        "max_age=unlimited:accumulation=sum"
    )
    breakdown: dict[str, list[dict[str, object]]] = {"layer": [], "phase": []}
    heldout_scope = split == "heldout"
    for layer in np.unique(data["layer"]):
        scope = heldout_scope & (data["layer"] == layer)
        breakdown["layer"].append(
            policy_row(best_name, f"layer-{int(layer)}", best_selected, scope, safe, effect)
        )
    phases = phase(data["remaining_mask_ratio"])
    for phase_index, phase_name in enumerate(("high", "mid", "low")):
        scope = heldout_scope & (phases == phase_index)
        breakdown["phase"].append(
            policy_row(best_name, phase_name, best_selected, scope, safe, effect)
        )

    group_key = np.rec.fromarrays(
        [data[name] for name in ("request_id", "layer", "head", "step", "query_tile")],
        names="request,layer,head,step,query",
    )
    transition_counts: list[int] = []
    reused_run_lengths: list[int] = []
    dirty_run_lengths: list[int] = []
    for value in np.unique(group_key[heldout_scope]):
        indices = np.flatnonzero(heldout_scope & (group_key == value))
        indices = indices[np.argsort(data["kv_tile"][indices])]
        mask = best_selected[indices]
        transition_counts.append(int((mask[1:] != mask[:-1]).sum()))
        start = 0
        for end in range(1, len(mask) + 1):
            if end == len(mask) or mask[end] != mask[start]:
                (reused_run_lengths if mask[start] else dirty_run_lengths).append(end - start)
                start = end
    fragmentation = {
        "groups": len(transition_counts),
        "mean_reuse_dirty_transitions_per_64_tiles": float(np.mean(transition_counts)),
        "p95_reuse_dirty_transitions_per_64_tiles": float(
            np.quantile(transition_counts, 0.95)
        ),
        "mean_reused_run_length": float(np.mean(reused_run_lengths)),
        "median_reused_run_length": float(np.median(reused_run_lengths)),
        "mean_dirty_run_length": float(np.mean(dirty_run_lengths)),
        "median_dirty_run_length": float(np.median(dirty_run_lengths)),
    }
    result = {
        "calibration_thresholds": {str(key): value for key, value in thresholds.items()},
        "heldout": heldout,
        "all_splits": rows,
        "best_policy_breakdown": breakdown,
        "best_policy_fragmentation": fragmentation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"heldout": heldout[:12]}, indent=2))


if __name__ == "__main__":
    main()
