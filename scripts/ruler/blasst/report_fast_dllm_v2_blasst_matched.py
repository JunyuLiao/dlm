#!/usr/bin/env python3
"""Report fast-dLLM-v2 on the 13-task DiffusionGemma comparison setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.official import score_predictions
from report_fast_dllm_v2_blasst_nonvt_aggressive import (
    bootstrap_ci,
    identity,
    read_json,
    read_jsonl,
    score_rows,
)


def ordered(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    if set(by_id) != {row["sample_id"] for row in manifest}:
        raise RuntimeError("prediction IDs do not match the 13-task manifest")
    result = [by_id[row["sample_id"]] for row in manifest]
    if [identity(row) for row in result] != [identity(row) for row in manifest]:
        raise RuntimeError("prediction prompts or seeds do not match the manifest")
    return result


def analyze(
    directory: Path,
    dense_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    holdout_ids: set[str],
    ruler_root: str,
) -> dict[str, Any]:
    rows = ordered(read_jsonl(directory / "predictions.jsonl"), manifest_rows)
    config = read_json(directory / "run_config.json")
    summary = read_json(directory / "summary.json")
    for key, expected in {
        "attention_backend": "blasst-reference",
        "model_adapter": "fast_dllm_v2",
        "context_length": 8192,
        "precision": "bfloat16",
        "q_tile_size": 128,
        "kv_tile_size": 64,
        "block_size": 16,
        "include_masked_kv_tiles_in_physical_stats": False,
    }.items():
        if config[key] != expected:
            raise RuntimeError(f"{directory}: expected {key}={expected!r}")
    dense_by_task, dense_accuracy, dense_scores = score_rows(dense_rows, ruler_root)
    by_task, accuracy, scores = score_rows(rows, ruler_root)
    differences = [scores[row["sample_id"]] - dense_scores[row["sample_id"]] for row in rows]

    dense_holdout = [row for row in dense_rows if row["sample_id"] in holdout_ids]
    sparse_holdout = [row for row in rows if row["sample_id"] in holdout_ids]
    _, dense_holdout_accuracy = score_predictions(dense_holdout, ruler_root)
    _, holdout_accuracy = score_predictions(sparse_holdout, ruler_root)
    sparsity = summary["attention_sparsity"]
    return {
        "run": directory.name,
        "blasst_lambda": config["blasst_lambda"],
        "blasst_policy": config["blasst_policy"],
        "accuracy": accuracy,
        "accuracy_delta": accuracy - dense_accuracy,
        "holdout_accuracy": holdout_accuracy,
        "holdout_dense_accuracy": dense_holdout_accuracy,
        "holdout_accuracy_delta": holdout_accuracy - dense_holdout_accuracy,
        "physical_tile_sparsity": sparsity["physical_tile_sparsity"],
        "valid_element_sparsity": sparsity["valid_element_sparsity"],
        "row_vote_sparsity": sparsity["row_vote_sparsity"],
        "paired_wins": sum(value > 0 for value in differences),
        "paired_losses": sum(value < 0 for value in differences),
        "paired_ties": sum(value == 0 for value in differences),
        "per_task_accuracy": by_task,
        "per_task_delta": {
            task: by_task[task] - dense_by_task[task] for task in by_task
        },
        **bootstrap_ci(
            differences,
            [row["task"] for row in rows],
            task_level_differences=[
                by_task[task] - dense_by_task[task] for task in sorted(by_task)
            ],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/kv_pruning/blockmax_ruler_paper_8k_n650/"
            "fast_dllm_v2/manifest/manifest.json"
        ),
    )
    parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    manifest = read_json(args.manifest.resolve())
    manifest_rows = read_jsonl(Path(manifest["samples"]["path"]))
    if len(manifest_rows) != 650 or len({row["task"] for row in manifest_rows}) != 13:
        raise RuntimeError("expected 13 tasks and 650 examples")

    dense_non_vt = read_jsonl(root / "runs/dense_eager/predictions.jsonl")
    dense_vt = read_jsonl(root / "vt_runs/dense_eager/predictions.jsonl")
    for path in (
        root / "runs/dense_eager/run_config.json",
        root / "vt_runs/dense_eager/run_config.json",
    ):
        config = read_json(path)
        for key, expected in {
            "attention_backend": "eager-dense",
            "model_adapter": "fast_dllm_v2",
            "context_length": 8192,
            "precision": "bfloat16",
            "q_tile_size": 128,
            "kv_tile_size": 64,
            "block_size": 16,
        }.items():
            if config[key] != expected:
                raise RuntimeError(f"{path}: expected {key}={expected!r}")
    dense_rows = ordered(dense_non_vt + dense_vt, manifest_rows)
    dense_by_task, dense_accuracy = score_predictions(dense_rows, args.ruler_root)

    screen_ids = {
        row["sample_id"] for row in read_jsonl(root / "screen_13_manifest/samples.jsonl")
    }
    holdout_ids = {row["sample_id"] for row in manifest_rows} - screen_ids
    if len(screen_ids) != 130 or len(holdout_ids) != 520:
        raise RuntimeError("expected a 10/task screen and 40/task holdout")
    dense_holdout = [row for row in dense_rows if row["sample_id"] in holdout_ids]
    _, dense_holdout_accuracy = score_predictions(dense_holdout, args.ruler_root)

    run_dirs = [
        root / "runs/lambda_0p003_13task",
        root / "runs/phase_matched_target25_13task",
    ]
    runs = [
        analyze(path, dense_rows, manifest_rows, holdout_ids, args.ruler_root)
        for path in run_dirs
    ]
    expected_policy = read_json(
        Path(
            "results/blasst/diffusion_gemma/paper_8k_n650/"
            "calibration/target_25/policy.json"
        ).resolve()
    )
    if runs[0]["blasst_policy"] != {} or runs[0]["blasst_lambda"] != 0.003:
        raise RuntimeError("the scalar run is not λ=0.003")
    if runs[1]["blasst_policy"] != expected_policy:
        raise RuntimeError("phase-matched run does not use DiffusionGemma's policy")

    stress = read_json(root / "report/report.json")
    report = {
        "model": "fast_dllm_v2_7b",
        "benchmark": "official RULER 8K, 13 tasks, 50 samples/task",
        "samples": 650,
        "holdout_samples": 520,
        "dense_accuracy": dense_accuracy,
        "dense_holdout_accuracy": dense_holdout_accuracy,
        "dense_per_task_accuracy": dense_by_task,
        "configuration": {
            "precision": "bfloat16",
            "q_tile_size": 128,
            "kv_tile_size": 64,
            "block_size": 16,
            "attention_layers": {"global": 28, "local": 0},
            "blasst_eligibility": "ordinary cached denoising queries only",
        },
        "runs": runs,
        "non_vt_aggressive_stress_test": stress,
    }
    output = root / "matched_report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = {
        "lambda_0p003_13task": "Scalar λ=0.003",
        "phase_matched_target25_13task": "DiffusionGemma global phase λs",
    }
    lines = [
        "# fast-dLLM-v2 BLASST on the matched DiffusionGemma setting",
        "",
        "Official RULER at 8K total length, all 13 tasks, 50 samples/task, "
        "BF16, Q tile 128, KV tile 64, and paired prompts/inference seeds. "
        "The block size is fast-dLLM-v2's native 16 (DiffusionGemma's native "
        "canvas is 256). All 28 fast-dLLM-v2 attention layers are global.",
        "",
        f"Dense eager baseline: **{dense_accuracy:.2%}** (40/task holdout: "
        f"**{dense_holdout_accuracy:.2%}**).",
        "",
        "| Policy | Accuracy | Delta | Physical sparsity | Holdout accuracy/delta | Wins/losses/ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {labels[row['run']]} | {row['accuracy']:.2%} | "
            f"{row['accuracy_delta']:+.2%} | {row['physical_tile_sparsity']:.2%} | "
            f"{row['holdout_accuracy']:.2%} ({row['holdout_accuracy_delta']:+.2%}) | "
            f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} |"
        )
    lines.extend(["", "## Uncertainty", ""])
    for row in runs:
        sample_ci = row["paired_sample_bootstrap_95_ci"]
        task_ci = row["task_cluster_bootstrap_95_ci"]
        lines.append(
            f"- {labels[row['run']]}: paired-sample 95% CI "
            f"[{sample_ci[0]:+.2%}, {sample_ci[1]:+.2%}]; task-cluster 95% CI "
            f"[{task_ci[0]:+.2%}, {task_ci[1]:+.2%}]."
        )
    lines.extend(
        [
            "",
            "The phase-matched policy uses the exact DiffusionGemma global "
            "thresholds 0.0760669, 0.0449779, and 0.0559884 at phase starts "
            "1 and 3. DiffusionGemma's local thresholds are inapplicable because "
            "fast-dLLM-v2 has no local-attention layers.",
            "",
            "For context, the separate 12-task aggressive stress test reaches "
            "93.93% physical sparsity at λ=0.90 and 94.07% at λ=0.95, with "
            "accuracy changes of −5.29 and −5.22 points respectively.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
