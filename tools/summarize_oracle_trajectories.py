#!/usr/bin/env python3
"""Summarize complete-generation oracle-mass trajectory evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def policy_label(payload: dict) -> str:
    policy = payload["policy"]
    if policy["selection"] == "independent":
        strength = f"tau={policy['threshold']:g}"
    else:
        strength = f"B={policy['budget']:g}/{policy['budget_scope']}"
    refresh = "fixed" if payload["oracle_refresh"] == "fixed" else "current"
    return (
        f"K{policy['k']} {policy['aggregation']} {strength} "
        f"newmax={policy['new_maximum_protection']} {refresh}"
    )


def plot_label(payload: dict) -> str:
    policy = payload["policy"]
    if policy["selection"] == "independent":
        strength = f"tau{policy['threshold']:g}"
    else:
        strength = f"B{policy['budget']:g}/{policy['budget_scope']}"
    refresh = "fixed" if payload["oracle_refresh"] == "fixed" else "current"
    aggregation = "max" if policy["aggregation"] == "all_max" else policy["aggregation"]
    return (
        f"K{policy['k']} {aggregation} {strength} "
        f"NM-{policy['new_maximum_protection']} {refresh}"
    )


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    steps = [step for result in results for step in result["steps"]]
    total_tiles = sum(result["policy_stats"]["total_tiles"] for result in results)
    baseline_skipped = sum(
        result["policy_stats"]["baseline_skipped_tiles"] for result in results
    )
    removals = sum(
        result["policy_stats"]["additional_skipped_tiles"] for result in results
    )
    exact = sum(
        bool(result["metrics"]["exact_generated_sequence_agreement"])
        for result in results
    )
    prompts = {result["prompt"] for result in results}
    held_out = path.stem.endswith("-final")
    return {
        "file": path.name,
        "policy": policy_label(payload) + (" [held-out]" if held_out else ""),
        "plot_label": plot_label(payload) + (" [held-out]" if held_out else ""),
        "trajectories": len(results),
        "unique_prompts": len(prompts),
        "exact_trajectories": exact,
        "exact_trajectory_fraction": exact / len(results),
        "differing_final_tokens": sum(
            result["metrics"]["differing_generated_tokens"] for result in results
        ),
        "final_edit_distance": sum(
            result["metrics"]["final_edit_distance"] for result in results
        ),
        "additional_skipped_tiles": removals,
        "additional_physical_sparsity": removals / total_tiles,
        "fraction_of_baseline_kept_removed": removals / (total_tiles - baseline_skipped),
        "masked_top1_mean": sum(
            step["masked_token_top1_agreement"] for step in steps
        )
        / len(steps),
        "masked_top1_min": min(step["masked_token_top1_agreement"] for step in steps),
        "hidden_cosine_min": min(step["hidden_state_cosine"] for step in steps),
        "masked_kl_max": max(step["masked_token_kl"] for step in steps),
        "generated_state_agreement_min": min(
            step["generated_state_agreement"] for step in steps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.inputs]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(10, 7.5))
    for row in rows:
        x = 100.0 * row["additional_physical_sparsity"]
        y = 100.0 * row["exact_trajectory_fraction"]
        marker = "o" if y == 100.0 else "x"
        axis.scatter(
            x,
            y,
            marker=marker,
            s=65,
            label=row["plot_label"],
        )
    axis.set_xscale("symlog", linthresh=0.001)
    axis.set_xlabel("Additional physical sparsity (%)")
    axis.set_ylabel("Exact final trajectories (%)")
    axis.set_ylim(25, 105)
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "policies": len(rows),
                "csv": str(args.csv),
                "plot": str(args.plot),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
