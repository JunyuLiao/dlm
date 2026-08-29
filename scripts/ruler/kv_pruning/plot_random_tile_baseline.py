#!/usr/bin/env python3
"""Finalize block-maximum versus random-tile pruning from available results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dllm.evaluation.ruler.official import score_predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    repeats: int = 20000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    totals = np.zeros(repeats, dtype=np.float64)
    for label in np.unique(strata):
        positions = np.flatnonzero(strata == label)
        sampled = rng.integers(0, len(positions), size=(repeats, len(positions)))
        totals += values[positions[sampled]].sum(axis=1)
    means = totals / len(values)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_partial_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    query_rows = eligible_tiles = dropped_tiles = 0
    mass_sum = mass_sq_sum = 0.0
    for row in rows:
        stats = row["oracle_attention_stats"]["overall"]
        count = int(stats["query_rows"])
        mean = float(stats["discarded_dense_attention_mass_mean"])
        std = float(stats["discarded_dense_attention_mass_std_over_rows"])
        query_rows += count
        eligible_tiles += int(stats["eligible_tiles"])
        dropped_tiles += int(stats["dropped_tiles"])
        mass_sum += mean * count
        mass_sq_sum += (std**2 + mean**2) * count
    mean = mass_sum / query_rows
    variance = mass_sq_sum / query_rows - mean**2
    return {
        "actual_tile_drop_fraction": dropped_tiles / eligible_tiles,
        "discarded_dense_attention_mass_mean": mean,
        "discarded_dense_attention_mass_std_over_rows": float(np.sqrt(max(0.0, variance))),
        "partial_attention_statistics_n": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-gemma-dir", required=True)
    parser.add_argument("--fast-dllm-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    args = parser.parse_args()

    dg_dir = Path(args.diffusion_gemma_dir).resolve()
    fast_dir = Path(args.fast_dllm_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    dg_config = json.loads((dg_dir / "run_config.json").read_text(encoding="utf-8"))
    samples = read_jsonl(Path(dg_config["samples"]))[: int(dg_config["num_samples"])]
    strata = np.asarray([str(sample["task"]) for sample in samples])
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    dense_predictions = read_jsonl(dg_dir / "conditions/dense_eager/predictions.jsonl")
    dense_individual = {
        str(row["sample_id"]): score_predictions([row], args.ruler_root)[1]
        for row in dense_predictions
    }

    dg_summary = json.loads((dg_dir / "summary.json").read_text(encoding="utf-8"))
    fast_summary = json.loads((fast_dir / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model, label, summary in (
        ("diffusion_gemma", "DiffusionGemma 26B-A4B", dg_summary),
        ("fast_dllm_v2", "Fast-dLLM v2 7B", fast_summary),
    ):
        for row in summary["conditions"]:
            if row["method"] not in ("dense", "block_max"):
                continue
            rows.append(
                {
                    **row,
                    "model": model,
                    "model_label": label,
                    "measurement_status": "measured",
                    "attention_statistics_n": 650,
                }
            )

    dg_dense = next(
        row for row in rows
        if row["model"] == "diffusion_gemma" and row["method"] == "dense"
    )
    for index, k in enumerate((10, 25, 50), start=1):
        condition = f"random_k{k:02d}"
        condition_dir = dg_dir / "conditions" / condition
        predictions = read_jsonl(condition_dir / "predictions.jsonl")
        if len(predictions) != 650:
            raise ValueError(f"{condition} has {len(predictions)} rows, expected 650")
        per_task, accuracy = score_predictions(predictions, args.ruler_root)
        individual = {
            str(row["sample_id"]): score_predictions([row], args.ruler_root)[1]
            for row in predictions
        }
        ordered = np.asarray([individual[sample_id] for sample_id in sample_ids])
        deltas = np.asarray(
            [individual[sample_id] - dense_individual[sample_id] for sample_id in sample_ids]
        )
        accuracy_low, accuracy_high = bootstrap(
            ordered, strata, seed=20260812 + index
        )
        delta_low, delta_high = bootstrap(
            deltas, strata, seed=20260912 + index
        )
        stats = json.loads((condition_dir / "attention_stats.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "model": "diffusion_gemma",
                "model_label": "DiffusionGemma 26B-A4B",
                "condition": condition,
                "method": "random",
                "method_label": "Random tiles",
                "k_fraction": k / 100,
                "official_ruler_accuracy": accuracy,
                "accuracy_bootstrap_95_low": accuracy_low,
                "accuracy_bootstrap_95_high": accuracy_high,
                "accuracy_delta_vs_dense": accuracy - dg_dense["official_ruler_accuracy"],
                "paired_delta_bootstrap_95_low": delta_low,
                "paired_delta_bootstrap_95_high": delta_high,
                "actual_tile_drop_fraction": stats["overall"]["actual_tile_drop_fraction"],
                "discarded_dense_attention_mass_mean": stats["overall"]["discarded_dense_attention_mass_mean"],
                "per_task_accuracy": per_task,
                "measurement_status": "measured",
                "attention_statistics_n": 650,
            }
        )

    partial_k75 = read_jsonl(dg_dir / "conditions/random_k75/predictions.jsonl")
    partial_stats = aggregate_partial_stats(partial_k75)
    rows.append(
        {
            "model": "diffusion_gemma",
            "model_label": "DiffusionGemma 26B-A4B",
            "condition": "random_k75_assumed_zero",
            "method": "random",
            "method_label": "Random tiles",
            "k_fraction": 0.75,
            "official_ruler_accuracy": 0.0,
            "accuracy_bootstrap_95_low": None,
            "accuracy_bootstrap_95_high": None,
            "accuracy_delta_vs_dense": -dg_dense["official_ruler_accuracy"],
            "paired_delta_bootstrap_95_low": None,
            "paired_delta_bootstrap_95_high": None,
            "actual_tile_drop_fraction": partial_stats["actual_tile_drop_fraction"],
            "discarded_dense_attention_mass_mean": partial_stats["discarded_dense_attention_mass_mean"],
            "per_task_accuracy": None,
            "measurement_status": "accuracy_assumed_zero_attention_stats_partial",
            "attention_statistics_n": partial_stats["partial_attention_statistics_n"],
        }
    )
    method_order = {"dense": -1, "block_max": 0, "random": 1}
    model_order = {"diffusion_gemma": 0, "fast_dllm_v2": 1}
    rows.sort(
        key=lambda row: (
            model_order[row["model"]],
            method_order[row["method"]],
            float(row["k_fraction"]),
        )
    )

    fields = (
        "model", "model_label", "method", "method_label", "condition", "k_fraction",
        "official_ruler_accuracy", "accuracy_delta_vs_dense",
        "accuracy_bootstrap_95_low", "accuracy_bootstrap_95_high",
        "paired_delta_bootstrap_95_low", "paired_delta_bootstrap_95_high",
        "actual_tile_drop_fraction", "discarded_dense_attention_mass_mean",
        "measurement_status", "attention_statistics_n",
    )
    with (output / "random_baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output / "random_baseline_comparison.json",
        {
            "schema_version": 1,
            "random_policy_seed": dg_config["random_policy_seed"],
            "fast_dllm_random_baseline_available": False,
            "conditions": rows,
        },
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"diffusion_gemma": "#D55E00", "fast_dllm_v2": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    series = (
        ("diffusion_gemma", "block_max", "DiffusionGemma · block maximum", "o", "-"),
        ("diffusion_gemma", "random", "DiffusionGemma · random", "x", "--"),
        ("fast_dllm_v2", "block_max", "Fast-dLLM · block maximum", "s", "-"),
    )
    for model, method, label, marker, linestyle in series:
        dense = next(
            row for row in rows if row["model"] == model and row["method"] == "dense"
        )
        subset = [dense] + [
            row for row in rows if row["model"] == model and row["method"] == method
        ]
        x = [100 * row["k_fraction"] for row in subset]
        accuracy = [100 * row["official_ruler_accuracy"] for row in subset]
        delta = [100 * row["accuracy_delta_vs_dense"] for row in subset]
        mass = [100 * row["discarded_dense_attention_mass_mean"] for row in subset]
        style = dict(color=colors[model], marker=marker, linestyle=linestyle, label=label)
        axes[0].plot(x, accuracy, **style)
        axes[1].plot(x, delta, **style)
        axes[2].plot(x, mass, **style)
        measured = [row for row in subset if row["accuracy_bootstrap_95_low"] is not None]
        mx = [100 * row["k_fraction"] for row in measured]
        axes[0].fill_between(
            mx,
            [100 * row["accuracy_bootstrap_95_low"] for row in measured],
            [100 * row["accuracy_bootstrap_95_high"] for row in measured],
            color=colors[model], alpha=0.07,
        )
        axes[1].fill_between(
            mx,
            [100 * row["paired_delta_bootstrap_95_low"] for row in measured],
            [100 * row["paired_delta_bootstrap_95_high"] for row in measured],
            color=colors[model], alpha=0.07,
        )
    axes[2].annotate(
        f"partial n={len(partial_k75)}", xy=(75, 100*partial_stats["discarded_dense_attention_mass_mean"]),
        xytext=(57, 84), arrowprops={"arrowstyle": "->", "color": colors["diffusion_gemma"]},
        color=colors["diffusion_gemma"], fontsize=9,
    )
    axes[0].set_ylabel("Official RULER accuracy (%)")
    axes[1].set_ylabel("Accuracy change vs dense (pp)")
    axes[2].set_ylabel("Dense attention mass discarded (%)")
    axes[1].axhline(0.0, color="0.35", lw=1, ls=":")
    for axis in axes:
        axis.set_xlabel("Requested KV tiles dropped (%)")
        axis.grid(alpha=0.25)
        axis.set_xticks((0, 10, 25, 50, 75))
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Paper-aligned 8K RULER: distribution-aware vs random KV-tile dropping")
    fig.tight_layout()
    fig.savefig(output / "random_baseline_comparison.png", dpi=180)
    fig.savefig(output / "random_baseline_comparison.pdf")
    plt.close(fig)

    random_rows = {
        int(round(100 * row["k_fraction"])): row
        for row in rows if row["model"] == "diffusion_gemma" and row["method"] == "random"
    }
    block_rows = {
        int(round(100 * row["k_fraction"])): row
        for row in rows if row["model"] == "diffusion_gemma" and row["method"] == "block_max"
    }
    lines = [
        "# Random KV-tile baseline",
        "",
        "Random tile masks drop exactly the same rounded number of valid tiles per layer/head/query row as block-maximum ranking. The policy uses an independent deterministic RNG stream (seed 20260812).",
        "",
        "| Policy | k | Mass discarded | Accuracy | Δ vs dense | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for k in (10, 25, 50, 75):
        for label, selected in (("Block maximum", block_rows[k]), ("Random tiles", random_rows[k])):
            status = selected["measurement_status"]
            lines.append(
                f"| {label} | {k}% | {100*selected['discarded_dense_attention_mass_mean']:.2f}% | "
                f"{100*selected['official_ruler_accuracy']:.2f}% | "
                f"{100*selected['accuracy_delta_vs_dense']:+.2f} pp | {status} |"
            )
    lines.extend(
        [
            "",
            "At k=10%, random dropping removes 10.70% of DiffusionGemma attention mass and scores 87.03%; block-maximum removes 0.49% and scores 90.03%. At k=25%, random dropping collapses to 41.45%, while block-maximum remains at 90.32%.",
            "",
            f"Random k=75 accuracy is an explicit user-requested 0% assumption. Its mass estimate uses the {len(partial_k75)} completed samples. Fast-dLLM random conditions were not run before the experiment was stopped, so no Fast-dLLM random curve is fabricated.",
            "",
            "![Random baseline comparison](random_baseline_comparison.png)",
            "",
        ]
    )
    (output / "random_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
