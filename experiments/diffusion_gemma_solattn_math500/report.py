"""Canonical audit, tables, and plots for completed experiment shards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import conditions
from .metrics import aggregate_calls, bootstrap_delta, token_agreement


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_report(root: Path) -> dict[str, Any]:
    expected = conditions()
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for condition in expected:
        directory = root / "conditions" / condition.name
        path = directory / "predictions.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing predictions for {condition.name}")
        rows = sorted(_read_jsonl(path), key=lambda row: int(row["subset_index"]))
        if len(rows) != 50:
            raise RuntimeError(f"{condition.name} has {len(rows)} rows, expected 50")
        if len({row["request_id"] for row in rows}) != 50:
            raise RuntimeError(f"{condition.name} contains duplicate request IDs")
        rows_by_condition[condition.name] = rows
        configs[condition.name] = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    dense_rows = rows_by_condition["dense"]
    identity = [(row["request_id"], row["prompt_hash"], row["seed"]) for row in dense_rows]
    shared = all(
        [(row["request_id"], row["prompt_hash"], row["seed"]) for row in rows] == identity
        for rows in rows_by_condition.values()
    )
    if not shared:
        raise RuntimeError("condition prompt/seed identity audit failed")
    dense_scores = np.asarray([bool(row["symbolic_correct"]) for row in dense_rows], dtype=float)
    summaries = []
    for condition in expected:
        rows = rows_by_condition[condition.name]
        scores = np.asarray([bool(row["symbolic_correct"]) for row in rows], dtype=float)
        if condition.name == "dense":
            routing = {
                name: {
                    "total_tiles": 0, "routed_tiles": 0, "skipped_tiles": 0,
                    "retained_tiles": 0, "full_model_tile_sparsity": 0.0,
                    "routed_region_tile_sparsity": 0.0,
                    "retained_dense_attention_mass": 1.0,
                    "attention_calls": 0, "routing_rows": 0, "degenerate_rows": 0,
                    "fallback_rows": 0, "degenerate_row_rate": 0.0, "fallback_row_rate": 0.0,
                }
                for name in ("whole_model", "local", "global")
            }
            routing["regions"] = {}
        else:
            routing = aggregate_calls(row["routing_stats"] for row in rows)
        delta, low, high = bootstrap_delta(dense_scores, scores)
        agreements = [token_agreement(dense["completion_tokens"], sparse["completion_tokens"]) for dense, sparse in zip(dense_rows, rows)]
        exact = [dense["completion_tokens"] == sparse["completion_tokens"] for dense, sparse in zip(dense_rows, rows)]
        summaries.append({
            "condition": condition.name,
            "mode": condition.region or "dense",
            "target_sparsity": condition.target_sparsity,
            "analytic_beta": condition.beta,
            "symbolic_accuracy": float(scores.mean()),
            "accuracy_delta_vs_dense": delta,
            "accuracy_delta_ci95": [low, high],
            "token_id_agreement": float(np.mean(agreements)),
            "sequence_exact_match": float(np.mean(exact)),
            "mean_completion_tokens": float(np.mean([row["num_generated_tokens"] for row in rows])),
            "length_termination_rate": float(np.mean([row["termination_reason"] == "length" for row in rows])),
            "reference_mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
            "routing": routing,
        })
    audit = {
        "schema_version": 1,
        "passed": True,
        "all_nine_conditions_present": len(rows_by_condition) == 9,
        "all_conditions_have_50_examples": all(len(rows) == 50 for rows in rows_by_condition.values()),
        "identical_request_ids_prompt_hashes_and_seeds": shared,
        "correct_analytic_threshold_provenance": all(
            configs[item.name]["condition"] == item.to_dict() for item in expected
        ),
        "count_weighted_sparsity": True,
        "sparsity_formula": "sum(skipped physical tiles) / sum(all valid physical tiles)",
        "sparse_coverage": {
            row["condition"]: {
                "local_calls": row["routing"]["local"]["attention_calls"],
                "global_calls": row["routing"]["global"]["attention_calls"],
                "prefix_tiles": row["routing"]["regions"].get("prefix", {}).get("total_tiles", 0),
                "canvas_tiles": row["routing"]["regions"].get("canvas", {}).get("total_tiles", 0),
            }
            for row in summaries if row["condition"] != "dense"
        },
    }
    audit["passed"] = all((
        audit["all_nine_conditions_present"],
        audit["all_conditions_have_50_examples"],
        audit["identical_request_ids_prompt_hashes_and_seeds"],
        audit["correct_analytic_threshold_provenance"],
        all(
            value["local_calls"] > 0 and value["global_calls"] > 0
            and value["prefix_tiles"] > 0 and value["canvas_tiles"] > 0
            for value in audit["sparse_coverage"].values()
        ),
    ))
    if not audit["passed"]:
        raise RuntimeError("final audit failed")
    result = {
        "schema_version": 1,
        "benchmark": "MATH500",
        "num_samples": 50,
        "primary_metric": "symbolic pass@1",
        "conditions": summaries,
        "audit": audit,
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "audit.json", audit)
    _write_tables(root, summaries)
    _write_plots(root, summaries)
    _write_markdown(root, summaries, audit)
    return result


def _write_tables(root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "condition", "mode", "target_sparsity", "analytic_beta", "symbolic_accuracy",
        "accuracy_delta_vs_dense", "accuracy_ci95_low", "accuracy_ci95_high",
        "whole_model_sparsity", "local_sparsity", "global_sparsity",
        "routed_region_sparsity", "retained_dense_attention_mass",
        "token_id_agreement", "sequence_exact_match", "skipped_tiles", "total_tiles",
    ]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            whole = row["routing"]["whole_model"]
            writer.writerow({
                "condition": row["condition"], "mode": row["mode"],
                "target_sparsity": row["target_sparsity"], "analytic_beta": row["analytic_beta"],
                "symbolic_accuracy": row["symbolic_accuracy"], "accuracy_delta_vs_dense": row["accuracy_delta_vs_dense"],
                "accuracy_ci95_low": row["accuracy_delta_ci95"][0], "accuracy_ci95_high": row["accuracy_delta_ci95"][1],
                "whole_model_sparsity": whole["full_model_tile_sparsity"],
                "local_sparsity": row["routing"]["local"]["full_model_tile_sparsity"],
                "global_sparsity": row["routing"]["global"]["full_model_tile_sparsity"],
                "routed_region_sparsity": whole["routed_region_tile_sparsity"],
                "retained_dense_attention_mass": whole["retained_dense_attention_mass"],
                "token_id_agreement": row["token_id_agreement"], "sequence_exact_match": row["sequence_exact_match"],
                "skipped_tiles": whole["skipped_tiles"], "total_tiles": whole["total_tiles"],
            })


def _write_plots(root: Path, rows: list[dict[str, Any]]) -> None:
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    sparse = [row for row in rows if row["condition"] != "dense"]
    colors = {"prefix_only": "#2878b5", "all": "#d95319"}
    labels = {"prefix_only": "prefix only", "all": "prefix + canvas"}
    for metric, ylabel, filename in (
        ("symbolic_accuracy", "MATH500 symbolic accuracy", "accuracy_vs_sparsity.png"),
        ("token_id_agreement", "Token-ID agreement with dense", "agreement_vs_sparsity.png"),
    ):
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        for mode in ("prefix_only", "all"):
            group = [row for row in sparse if row["mode"] == mode]
            axis.plot(
                [row["routing"]["whole_model"]["full_model_tile_sparsity"] for row in group],
                [row[metric] for row in group], marker="o", label=labels[mode], color=colors[mode],
            )
        axis.set_xlabel("Measured skipped tiles / all valid tiles")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    for mode in ("prefix_only", "all"):
        group = [row for row in sparse if row["mode"] == mode]
        for scope, style in (("whole_model", "-"), ("local", "--"), ("global", ":")):
            axis.plot(
                [row["target_sparsity"] for row in group],
                [row["routing"][scope]["full_model_tile_sparsity"] for row in group],
                marker="o", linestyle=style, color=colors[mode], label=f"{labels[mode]} / {scope.replace('_', ' ')}",
            )
    axis.plot([0, 1], [0, 1], color="black", alpha=0.3, linewidth=1, label="target = achieved")
    axis.set(xlabel="Analytic target sparsity", ylabel="Count-weighted measured sparsity", xlim=(0.2, 0.95), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "target_vs_achieved_local_global.png", dpi=180)
    plt.close(fig)


def _write_markdown(root: Path, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    lines = [
        "# DiffusionGemma Sol-Attn on MATH500: prefix-only vs prefix+canvas",
        "",
        "## Results",
        "",
        "| Condition | Target | Beta | Accuracy | Delta vs dense (95% CI) | Whole sparsity | Local | Global | Routed-region sparsity | Retained mass | Token agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        routing = row["routing"]
        beta = "—" if row["analytic_beta"] is None else f"{row['analytic_beta']:+.6f}"
        ci = row["accuracy_delta_ci95"]
        lines.append(
            f"| {row['condition']} | {row['target_sparsity']:.0%} | {beta} | {row['symbolic_accuracy']:.1%} | "
            f"{row['accuracy_delta_vs_dense']:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] | "
            f"{routing['whole_model']['full_model_tile_sparsity']:.1%} | {routing['local']['full_model_tile_sparsity']:.1%} | "
            f"{routing['global']['full_model_tile_sparsity']:.1%} | {routing['whole_model']['routed_region_tile_sparsity']:.1%} | "
            f"{routing['whole_model']['retained_dense_attention_mass']:.1%} | {row['token_id_agreement']:.1%} |"
        )
    lines.extend([
        "", "## Protocol", "",
        "- 50 deterministic paired MATH500 problems from the repository's fixed subset; one seeded sample per problem. Dense uses the matched eager-attention backend.",
        "- DiffusionGemma-26B-A4B-it in BF16; temperature 0.6, top-p 0.95, 2,048-token maximum, 256-token diffusion canvas.",
        "- Sol-Attn-like routing uses post-normalization/post-RoPE Q/K, 64×64 tiles, mean-pooled proxies, row standardization, and the universal Gaussian beta. No correction is applied.",
        "- Prefix-only keeps canvas tiles dense. Prefix+canvas standardizes both regions as one candidate population.",
        "- Primary measured sparsity is the ratio of summed skipped physical tiles to summed valid physical tiles. It is never an average of per-step sparsities.",
        "- This is a logical reference-mask backend that still evaluates dense attention diagnostics; elapsed time is not sparse-kernel speedup.",
        "", f"Final audit passed: **{audit['passed']}**.", "",
    ])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
