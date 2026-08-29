#!/usr/bin/env python3
"""Compare block-maximum and random KV-tile pruning across two models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = (
    ("diffusion_gemma", "DiffusionGemma 26B-A4B"),
    ("fast_dllm_v2", "Fast-dLLM v2 7B"),
)
METHODS = (
    ("block_max", "Block maximum"),
    ("random", "Random tiles"),
)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_run(path: Path, model: str, label: str) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    rows = []
    for row in summary["conditions"]:
        if row["method"] not in ("dense", *(method for method, _ in METHODS)):
            continue
        rows.append({**row, "model": model, "model_label": label})
    method_order = {"dense": -1, **{method: i for i, (method, _) in enumerate(METHODS)}}
    rows.sort(key=lambda row: (method_order[row["method"]], float(row["k_fraction"])))
    dense = [row for row in rows if row["method"] == "dense"]
    if len(dense) != 1 or float(dense[0]["k_fraction"]) != 0.0:
        raise ValueError(f"{model} does not contain exactly one dense baseline")
    expected = [0.1, 0.25, 0.5, 0.75]
    for method, _ in METHODS:
        actual = [float(row["k_fraction"]) for row in rows if row["method"] == method]
        if actual != expected:
            raise ValueError(f"{model} {method} does not contain the expected k sweep")
    return {"path": str(path.resolve()), "config": config, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-gemma-dir", required=True)
    parser.add_argument("--fast-dllm-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "diffusion_gemma": Path(args.diffusion_gemma_dir).resolve(),
        "fast_dllm_v2": Path(args.fast_dllm_dir).resolve(),
    }
    runs = {
        model: _load_run(inputs[model], model, label)
        for model, label in MODELS
    }
    rows = [row for model, _ in MODELS for row in runs[model]["rows"]]

    fields = (
        "model", "model_label", "condition", "k_fraction",
        "official_ruler_accuracy", "accuracy_delta_vs_dense",
        "accuracy_bootstrap_95_low", "accuracy_bootstrap_95_high",
        "paired_delta_bootstrap_95_low", "paired_delta_bootstrap_95_high",
        "actual_tile_drop_fraction", "discarded_dense_attention_mass_mean",
    )
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    tasks = sorted({task for row in rows for task in row["per_task_accuracy"]})
    with (output / "per_task_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "model_label", "method", "method_label", "k_fraction", *tasks),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "model_label": row["model_label"],
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "k_fraction": row["k_fraction"],
                    **row["per_task_accuracy"],
                }
            )

    _atomic_json(
        output / "comparison.json",
        {
            "schema_version": 1,
            "models": {model: runs[model]["config"] for model, _ in MODELS},
            "conditions": rows,
        },
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"diffusion_gemma": "#D55E00", "fast_dllm_v2": "#0072B2"}
    markers = {"block_max": "o", "random": "x"}
    linestyles = {"block_max": "-", "random": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    for model, label in MODELS:
        dense = next(row for row in runs[model]["rows"] if row["method"] == "dense")
        for method, method_label in METHODS:
            subset = [dense] + [
                row for row in runs[model]["rows"] if row["method"] == method
            ]
            x = [100 * row["k_fraction"] for row in subset]
            accuracy = [100 * row["official_ruler_accuracy"] for row in subset]
            accuracy_low = [100 * row["accuracy_bootstrap_95_low"] for row in subset]
            accuracy_high = [100 * row["accuracy_bootstrap_95_high"] for row in subset]
            delta = [100 * row["accuracy_delta_vs_dense"] for row in subset]
            delta_low = [100 * row["paired_delta_bootstrap_95_low"] for row in subset]
            delta_high = [100 * row["paired_delta_bootstrap_95_high"] for row in subset]
            mass = [100 * row["discarded_dense_attention_mass_mean"] for row in subset]
            series_label = f"{label} · {method_label}"
            style = dict(marker=markers[method], linestyle=linestyles[method], color=colors[model], label=series_label)
            axes[0].plot(x, accuracy, **style)
            axes[0].fill_between(x, accuracy_low, accuracy_high, color=colors[model], alpha=0.06)
            axes[1].plot(x, delta, **style)
            axes[1].fill_between(x, delta_low, delta_high, color=colors[model], alpha=0.06)
            axes[2].plot(x, mass, **style)
    axes[0].set_ylabel("Official RULER accuracy (%)")
    axes[1].set_ylabel("Accuracy change vs model's dense baseline (pp)")
    axes[2].set_ylabel("Mean dense attention mass discarded (%)")
    axes[1].axhline(0.0, color="0.35", lw=1, ls="--")
    for axis in axes:
        axis.set_xlabel("Requested bottom-k tiles dropped (%)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Paper-aligned 8K RULER: block-maximum vs random KV-tile pruning")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    fig.savefig(output / "model_comparison.pdf")
    plt.close(fig)

    by_model_method_k = {
        (row["model"], row["method"], float(row["k_fraction"])): row
        for row in rows
    }
    dg50 = by_model_method_k[("diffusion_gemma", "block_max", 0.5)]
    fast50 = by_model_method_k[("fast_dllm_v2", "block_max", 0.5)]
    dg_random50 = by_model_method_k[("diffusion_gemma", "random", 0.5)]
    fast_random50 = by_model_method_k[("fast_dllm_v2", "random", 0.5)]
    dg75 = by_model_method_k[("diffusion_gemma", "block_max", 0.75)]
    dg_dense = next(row for row in rows if row["model"] == "diffusion_gemma" and row["method"] == "dense")
    mass_ratio_50 = (
        dg50["discarded_dense_attention_mass_mean"]
        / fast50["discarded_dense_attention_mass_mean"]
    )
    vt_delta_75 = (
        dg75["per_task_accuracy"]["vt"]
        - dg_dense["per_task_accuracy"]["vt"]
    )
    lines = [
        "# KV-tile pruning: block-maximum versus random baseline",
        "",
        "Both models use 13 official RULER tasks × 50 examples, paper-style 8K total-length budgeting, q128×kv64, decoding block 256, bf16, threshold 0.9, and temperature 0. Model-specific tokenizers produce separate but protocol-matched manifests.",
        "",
        "## Main observations",
        "",
        f"- Fast-dLLM is substantially more tile-concentrated: at k=50%, it discards only {100*fast50['discarded_dense_attention_mass_mean']:.2f}% of dense attention mass, versus {100*dg50['discarded_dense_attention_mass_mean']:.2f}% for DiffusionGemma ({mass_ratio_50:.2f}× more).",
        f"- The random k=50% baseline discards {100*fast_random50['discarded_dense_attention_mass_mean']:.2f}% of Fast-dLLM mass and {100*dg_random50['discarded_dense_attention_mass_mean']:.2f}% of DiffusionGemma mass. Block-maximum ranking therefore preserves {100*(fast_random50['discarded_dense_attention_mass_mean']-fast50['discarded_dense_attention_mass_mean']):.2f} pp and {100*(dg_random50['discarded_dense_attention_mass_mean']-dg50['discarded_dense_attention_mass_mean']):.2f} pp more mass, respectively.",
        f"- The k=50% paired accuracy changes are {100*fast50['accuracy_delta_vs_dense']:+.2f} pp for Fast-dLLM and {100*dg50['accuracy_delta_vs_dense']:+.2f} pp for DiffusionGemma; both paired 95% intervals include zero.",
        f"- DiffusionGemma improves by {100*dg75['accuracy_delta_vs_dense']:+.2f} pp at k=75%, with a paired 95% interval of [{100*dg75['paired_delta_bootstrap_95_low']:+.2f}, {100*dg75['paired_delta_bootstrap_95_high']:+.2f}] pp. This aggregate gain is mainly associated with VT changing by {100*vt_delta_75:+.1f} task-score points, so it should be treated as a task-specific pruning effect rather than a general quality claim.",
        "",
        "| Model | Policy | k | Actual tile drop | Dense mass discarded | Accuracy | Δ vs dense | Paired 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['method_label']} | {100*row['k_fraction']:.0f}% | "
            f"{100*row['actual_tile_drop_fraction']:.1f}% | "
            f"{100*row['discarded_dense_attention_mass_mean']:.2f}% | "
            f"{100*row['official_ruler_accuracy']:.2f}% | "
            f"{100*row['accuracy_delta_vs_dense']:+.2f} pp | "
            f"[{100*row['paired_delta_bootstrap_95_low']:+.2f}, "
            f"{100*row['paired_delta_bootstrap_95_high']:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "Shaded accuracy intervals and paired-delta intervals use task-stratified resampling of the 650 sample-level official scores. Block-maximum uses dense QK before masking; random masks use a separate reproducible RNG stream. These are accuracy measurements rather than sparse-kernel speed measurements.",
            "",
            "![Model comparison](model_comparison.png)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
