#!/usr/bin/env python3
"""Flatten and visualize the compact tile-replacement feasibility study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import re
from statistics import fmean

import matplotlib.pyplot as plt


def flatten(value: dict, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item)
        else:
            result[name] = item
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    steps = [step for result in results for step in result["steps"]]
    total = sum(result["runtime_stats"]["physical_tiles"] for result in results)
    kept = sum(result["runtime_stats"]["baseline_kept_tiles"] for result in results)
    replaced = sum(result["runtime_stats"]["replaced_tiles"] for result in results)
    exact_final = sum(result["metrics"]["exact_final_sequence_agreement"] for result in results)
    divergent = [
        result["metrics"]["first_divergent_step"]
        for result in results
        if result["metrics"]["first_divergent_step"] >= 0
    ]
    return {
        "file": path.name,
        "policy_name": payload["policy_name"],
        **{f"policy_{key}": value for key, value in payload["policy"].items()},
        "contexts": len(results),
        "exact_final_sequences": exact_final,
        "exact_final_sequence_fraction": exact_final / len(results),
        "differing_final_tokens": sum(result["metrics"]["differing_tokens"] for result in results),
        "final_edit_distance": sum(result["metrics"]["final_edit_distance"] for result in results),
        "first_divergent_step_minimum": min(divergent) if divergent else -1,
        "physical_tiles": total,
        "baseline_kept_tiles": kept,
        "replaced_tiles": replaced,
        "physical_replacement_fraction": replaced / total,
        "kept_replacement_fraction": replaced / kept,
        "masked_top1_mean": fmean(step["masked_logits"]["top1_agreement"] for step in steps),
        "masked_logit_cosine_mean": fmean(step["masked_logits"]["cosine"] for step in steps),
        "masked_kl_maximum": max(step["masked_logits"]["kl"] for step in steps),
        "confidence_rank_mean": fmean(step["confidence_rank_correlation"] for step in steps),
        "reveal_jaccard_mean": fmean(step["reveal_position_jaccard"] for step in steps),
        "exact_reveal_steps": sum(step["exact_reveal_position_set"] for step in steps),
        "total_steps": len(steps),
        "minimum_hidden_cosine": min(step["hidden_state_cosine"] for step in steps),
        "minimum_generated_state_agreement": min(
            step["generated_state_agreement"] for step in steps
        ),
    }


def aggregate_failures(rows: list[dict]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    metrics = (
        "candidate_tiles",
        "new_maximum_candidates",
        "mean_veto_row_relative_mass_error",
        "mean_veto_row_relative_value_error",
        "compact_mean_relative_output_error",
        "compact_masked_mean_relative_output_error",
        "compact_visible_mean_relative_output_error",
        "compact_minimum_row_cosine",
        "veto_oracle_mean_relative_output_error",
    )
    dimensions = (
        ("layer",),
        ("head",),
        ("noise_phase",),
        ("layer", "noise_phase"),
        ("head", "noise_phase"),
    )
    for keys in dimensions:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        for values, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
            aggregate: dict[str, object] = {
                "grouping": "+".join(keys),
                "groups": len(members),
            }
            aggregate.update(dict(zip(keys, values)))
            for metric in metrics:
                if metric in ("candidate_tiles", "new_maximum_candidates"):
                    aggregate[metric] = sum(member[metric] for member in members)
                else:
                    aggregate[metric] = fmean(member[metric] for member in members)
            result.append(aggregate)
    return result


def slot_number(method: str) -> int | None:
    match = re.search(r"_r(1|2|4|8)$", method)
    return int(match.group(1)) if match else None


def plot_slots(data: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    families = {
        "uniform": "Uniform",
        "kmeans_k": "K-means K",
        "kmeans_joint": "Joint K/V",
        "proxy_mean_q": "Mean query",
        "proxy_mean_veto_q": "Mean veto query (oracle)",
    }
    mass = [row for row in data["mass_predictors"] if row["split"] == "test"]
    values = [
        row
        for row in data["value_predictors"]
        if row["split"] == "test" and row["scope"] == "veto_only"
    ]
    for prefix, label in families.items():
        mass_points = sorted(
            (
                slot_number(row["method"]),
                row["relative_mass_error"]["mean"],
            )
            for row in mass
            if row["method"].startswith(prefix + "_r")
        )
        value_points = sorted(
            (
                slot_number(row["method"]),
                row["relative_u_error"]["mean"],
            )
            for row in values
            if row["method"].startswith(prefix + "_r")
        )
        if mass_points:
            axes[0].plot(*zip(*mass_points), marker="o", label=label)
        if value_points:
            axes[1].plot(*zip(*value_points), marker="o", label=label)
    for axis, title, ylabel in (
        (axes[0], "Unnormalized-mass prediction", "Mean relative mass error"),
        (axes[1], "Conditional-value reconstruction", "Mean relative U error (veto rows)"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks((1, 2, 4, 8), labels=("1", "2", "4", "8"))
        axis.set_xlabel("Summary slots R")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_policy(data: dict, output: Path) -> None:
    rows = [
        row
        for row in data["policy_sweep"]
        if row["split"] == "test" and row["scope"] == "all_rows"
    ]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    styles = {
        "none": ("o", "No protection"),
        "veto_rows": ("^", "Protect veto-row maxima"),
        "all_rows": ("s", "Protect all-row maxima"),
    }
    for protection, (marker, label) in styles.items():
        selected = [row for row in rows if row["new_maximum_protection"] == protection]
        axis.scatter(
            [100 * row["physical_replacement_fraction"] for row in selected],
            [row["relative_output_error"]["mean"] for row in selected],
            marker=marker,
            alpha=0.72,
            label=label,
        )
    highlights = {
        ("exact_vetomax_lt_0.03", "none"): "trajectory: gate-scale",
        ("predicted_maxmass_lt_0.003", "all_rows"): "trajectory: strict",
    }
    for row in rows:
        tag = (row["policy"], row["new_maximum_protection"])
        if tag in highlights:
            x = 100 * row["physical_replacement_fraction"]
            y = row["relative_output_error"]["mean"]
            axis.annotate(highlights[tag], (x, y), xytext=(7, 7), textcoords="offset points")
    axis.set_yscale("symlog", linthresh=1e-5)
    axis.set_xlabel("Additional physical tile replacement (%)")
    axis.set_ylabel("Mean relative attention-output error")
    axis.set_title("Held-out attention accuracy–coverage sweep")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_matrix(data: dict, output: Path) -> None:
    wanted = [
        ("all_rows", "exact_mass__exact_u", "Exact mass + exact U"),
        ("veto_only", "exact_mass__exact_u", "Exact/exact, veto-only"),
        ("all_rows", "calibrated_maximum_logit_mass__exact_u", "Predicted mass + exact U"),
        ("all_rows", "exact_mass__oracle_centroid_r8", "Exact mass + oracle centroid R8"),
        ("all_rows", "exact_mass__proxy_mean_q_r8", "Exact mass + mean-Q R8"),
        (
            "all_rows",
            "calibrated_maximum_logit_mass__proxy_mean_q_r8",
            "Predicted mass + mean-Q R8",
        ),
    ]
    lookup = {
        (row["scope"], row["method"]): row
        for row in data["replacement_matrix"]
        if row["split"] == "test"
    }
    labels, values, maxima = [], [], []
    for scope, method, label in wanted:
        row = lookup[(scope, method)]
        labels.append(label)
        values.append(row["relative_output_error"]["mean"])
        maxima.append(row["relative_output_error"]["maximum"])
    figure, axis = plt.subplots(figsize=(10, 6))
    positions = list(range(len(labels)))
    axis.barh(positions, values, color="#4c78a8")
    axis.scatter(maxima, positions, color="#e45756", marker="|", s=150, label="Maximum row error")
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Relative attention-output error")
    axis.set_title("Mass/value bottleneck isolation on held-out QKV")
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_failures(aggregates: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    layers = sorted(
        (row for row in aggregates if row["grouping"] == "layer"),
        key=lambda row: row["layer"],
    )
    axes[0].plot(
        [row["layer"] for row in layers],
        [row["compact_mean_relative_output_error"] for row in layers],
        marker="o",
        label="Compact all-row R8",
    )
    axes[0].plot(
        [row["layer"] for row in layers],
        [row["veto_oracle_mean_relative_output_error"] for row in layers],
        marker="s",
        label="Exact replacement, veto-only",
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean relative output error")
    axes[0].set_title("Layer sensitivity")
    axes[0].legend(frameon=False)
    phases = {row["noise_phase"]: row for row in aggregates if row["grouping"] == "noise_phase"}
    order = ("high", "mid", "low")
    x = list(range(3))
    width = 0.25
    axes[1].bar(
        [value - width for value in x],
        [phases[p]["mean_veto_row_relative_mass_error"] for p in order],
        width,
        label="Mass error",
    )
    axes[1].bar(
        x,
        [phases[p]["mean_veto_row_relative_value_error"] for p in order],
        width,
        label="Value error",
    )
    axes[1].bar(
        [value + width for value in x],
        [phases[p]["compact_mean_relative_output_error"] for p in order],
        width,
        label="Output error",
    )
    axes[1].set_xticks(x, labels=order)
    axes[1].set_xlabel("Noise phase")
    axes[1].set_ylabel("Mean relative error")
    axes[1].set_title("Noise-phase failure breakdown")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_trajectories(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 6))
    for row in rows:
        x = 100 * row["physical_replacement_fraction"]
        metrics = {
            "Masked top-1": 100 * row["masked_top1_mean"],
            "Reveal Jaccard": 100 * row["reveal_jaccard_mean"],
            "Exact final": 100 * row["exact_final_sequence_fraction"],
        }
        for metric, value in metrics.items():
            axis.scatter(x, value, s=75, label=f"{metric} — {row['policy_policy']}" )
    axis.axhline(99.5, color="black", linestyle="--", linewidth=1, label="99.5% top-1 gate")
    axis.axvline(10.0, color="gray", linestyle=":", linewidth=1, label="10 pp coverage gate")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("Full-trajectory physical replacement (%)")
    axis.set_ylabel("Agreement metric (%)")
    axis.set_ylim(50, 101)
    axis.set_title("Held-out full-generation agreement")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.predictor.read_text(encoding="utf-8"))
    trajectory_rows = [trajectory_summary(path) for path in args.trajectory]
    test_failures = [row for row in data["failure_breakdown"] if row["split"] == "test"]
    failure_aggregates = aggregate_failures(test_failures)

    for name in ("coverage", "mass_predictors", "value_predictors", "replacement_matrix", "policy_sweep"):
        write_csv(args.output_dir / f"{name}.csv", [flatten(row) for row in data[name]])
    write_csv(args.output_dir / "failure_breakdown.csv", data["failure_breakdown"])
    write_csv(args.output_dir / "failure_aggregates.csv", failure_aggregates)
    write_csv(args.output_dir / "trajectory_summary.csv", trajectory_rows)

    plot_slots(data, args.output_dir / "predictor_slots.png")
    plot_policy(data, args.output_dir / "attention_accuracy_coverage.png")
    plot_matrix(data, args.output_dir / "replacement_matrix.png")
    plot_failures(failure_aggregates, args.output_dir / "layer_phase_failures.png")
    plot_trajectories(trajectory_rows, args.output_dir / "trajectory_frontier.png")

    summary = {
        "schema": "blasst-tile-replacement-analysis-v1",
        "predictor_artifact": str(args.predictor),
        "trajectory_artifacts": [str(path) for path in args.trajectory],
        "predictor_groups": len(data["breakdown"]),
        "failure_rows": len(data["failure_breakdown"]),
        "trajectory_policies": trajectory_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "files": 13}, indent=2))


if __name__ == "__main__":
    main()
