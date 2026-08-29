#!/usr/bin/env python3
"""Validate and report the fixed local/global DiffusionGemma BLASST run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.official import score_predictions


COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
)
EXPECTED_POLICY = {
    "local_blasst_lambda": 0.95,
    "global_blasst_lambda": 0.60,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["prompt_sha256"], int(row["inference_seed"])


def aggregate_attention_types(path: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            attention_type = row["attention_type"]
            for field in COUNT_FIELDS:
                groups[attention_type][field] += int(row[field])
    result = []
    for attention_type, counts in sorted(groups.items()):
        if counts["eligible_tiles"] != counts["skipped_tiles"] + counts["retained_tiles"]:
            raise RuntimeError(f"tile count identity failed for {attention_type}")
        result.append(
            {
                "attention_type": attention_type,
                **counts,
                "physical_tile_sparsity": (
                    counts["skipped_tiles"] / counts["eligible_tiles"]
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    dense_dir = root / "dense_eager"
    sparse_dir = root / "sparse_l0p95_g0p60"
    manifest = read_json(root / "manifest/manifest.json")
    dense_config = read_json(dense_dir / "run_config.json")
    sparse_config = read_json(sparse_dir / "run_config.json")
    dense_summary = read_json(dense_dir / "summary.json")
    sparse_summary = read_json(sparse_dir / "summary.json")
    dense_rows = read_jsonl(dense_dir / "predictions.jsonl")
    sparse_rows = read_jsonl(sparse_dir / "predictions.jsonl")

    if manifest["actual_num_samples"] != 1300:
        raise RuntimeError("the manifest does not contain exactly 1,300 examples")
    if len(manifest["tasks"]) != 13 or any(
        count != 100 for count in manifest["task_counts"].values()
    ):
        raise RuntimeError("expected all 13 tasks with exactly 100 examples/task")
    if len(dense_rows) != 1300 or len(sparse_rows) != 1300:
        raise RuntimeError("dense and sparse runs must each contain 1,300 predictions")
    if [identity(row) for row in dense_rows] != [identity(row) for row in sparse_rows]:
        raise RuntimeError("dense and sparse prompts, ordering, or inference seeds differ")
    if dense_config["attention_backend"] != "eager-dense":
        raise RuntimeError("the comparison baseline is not eager-dense")
    if sparse_config["attention_backend"] != "blasst-reference":
        raise RuntimeError("the sparse run is not the eager BLASST reference")
    if sparse_config["blasst_policy"] != EXPECTED_POLICY:
        raise RuntimeError("the sparse run does not use local=0.95/global=0.60")
    for key, expected in {
        "model_adapter": "diffusion_gemma",
        "context_length": 8192,
        "precision": "bfloat16",
        "q_tile_size": 128,
        "kv_tile_size": 64,
        "block_size": 256,
        "include_masked_kv_tiles_in_physical_stats": False,
    }.items():
        if dense_config[key] != expected or sparse_config[key] != expected:
            raise RuntimeError(f"dense/sparse configuration mismatch for {key}")

    dense_by_task, dense_accuracy = score_predictions(dense_rows, args.ruler_root)
    sparse_by_task, sparse_accuracy = score_predictions(sparse_rows, args.ruler_root)
    dense_scores = {
        row["sample_id"]: score_predictions([row], args.ruler_root)[1]
        for row in dense_rows
    }
    sparse_scores = {
        row["sample_id"]: score_predictions([row], args.ruler_root)[1]
        for row in sparse_rows
    }
    by_attention_type = aggregate_attention_types(
        sparse_dir / "attention_stats/per_layer.csv"
    )
    summary_sparsity = sparse_summary["attention_sparsity"]
    for field in COUNT_FIELDS:
        aggregated = sum(row[field] for row in by_attention_type)
        if aggregated != int(summary_sparsity[field]):
            raise RuntimeError(f"layer aggregation mismatch for {field}")

    report = {
        "validation": {
            "passed": True,
            "samples": 1300,
            "tasks": 13,
            "samples_per_task": 100,
            "same_prompts_order_and_seeds": True,
            "dense_backend": "eager-dense",
            "sparse_backend": "blasst-reference",
            "q_tile_size": 128,
            "kv_tile_size": 64,
            "native_canvas_length": 256,
        },
        "policy": EXPECTED_POLICY,
        "accuracy": {
            "dense": dense_accuracy,
            "sparse": sparse_accuracy,
            "delta": sparse_accuracy - dense_accuracy,
            "dense_by_task": dense_by_task,
            "sparse_by_task": sparse_by_task,
            "paired_wins": sum(
                sparse_scores[key] > dense_scores[key] for key in dense_scores
            ),
            "paired_losses": sum(
                sparse_scores[key] < dense_scores[key] for key in dense_scores
            ),
            "paired_ties": sum(
                sparse_scores[key] == dense_scores[key] for key in dense_scores
            ),
        },
        "sparsity": {
            "overall": summary_sparsity,
            "by_attention_type": by_attention_type,
        },
        "run_summary": sparse_summary,
    }
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# DiffusionGemma fixed BLASST RULER evaluation",
        "",
        "Official 8K RULER benchmark: 13 tasks, 100 examples/task, "
        "1,300 examples per run. Dense and sparse use identical prompts, ordering, "
        "inference seeds, BF16 precision, and eager attention math.",
        "",
        "Fixed policy: local lambda `0.95`; global lambda `0.60`; "
        "Q tile `128`; KV tile `64`; native canvas `256`.",
        "",
        "| Run | Accuracy | Physical sparsity |",
        "|---|---:|---:|",
        f"| Dense eager | {dense_accuracy:.2%} | 0.00% |",
        f"| Fixed BLASST | {sparse_accuracy:.2%} | "
        f"{summary_sparsity['physical_tile_sparsity']:.2%} |",
        f"| Difference | {sparse_accuracy - dense_accuracy:+.2%} | — |",
        "",
        "## Accuracy by task",
        "",
        "| Task | Dense | Sparse | Delta |",
        "|---|---:|---:|---:|",
    ]
    for task in sorted(dense_by_task):
        lines.append(
            f"| {task} | {dense_by_task[task]:.2%} | {sparse_by_task[task]:.2%} | "
            f"{sparse_by_task[task] - dense_by_task[task]:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## Sparsity by attention type",
            "",
            "| Attention | Eligible tiles | Skipped tiles | Physical sparsity |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in by_attention_type:
        lines.append(
            f"| {row['attention_type']} | {row['eligible_tiles']} | "
            f"{row['skipped_tiles']} | {row['physical_tile_sparsity']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Paired outcomes: {report['accuracy']['paired_wins']} wins, "
            f"{report['accuracy']['paired_losses']} losses, and "
            f"{report['accuracy']['paired_ties']} ties.",
            "",
            "Physical sparsity is globally aggregated skipped/eligible tile work "
            "and excludes structurally masked tiles. This is an eager reference "
            "correctness run, not a sparse-kernel wall-clock benchmark.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
