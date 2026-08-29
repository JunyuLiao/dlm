#!/usr/bin/env python3
"""Audit and report the matched fast-dLLM-v2 aggressive BLASST runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.official import score_predictions


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["prompt_sha256"], int(row["inference_seed"])


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    differences: list[float],
    tasks: list[str],
    *,
    iterations: int = 20_000,
    task_level_differences: list[float] | None = None,
) -> dict[str, list[float]]:
    rng = random.Random(20260811)
    sample_draws = []
    for _ in range(iterations):
        sample_draws.append(
            sum(rng.choice(differences) for _ in differences) / len(differences)
        )

    by_task: dict[str, list[float]] = defaultdict(list)
    for task, difference in zip(tasks, differences, strict=True):
        by_task[task].append(difference)
    task_means = task_level_differences or [
        sum(values) / len(values) for values in by_task.values()
    ]
    task_draws = []
    for _ in range(iterations):
        task_draws.append(
            sum(rng.choice(task_means) for _ in task_means) / len(task_means)
        )
    return {
        "paired_sample_bootstrap_95_ci": [
            quantile(sample_draws, 0.025),
            quantile(sample_draws, 0.975),
        ],
        "task_cluster_bootstrap_95_ci": [
            quantile(task_draws, 0.025),
            quantile(task_draws, 0.975),
        ],
    }


def score_rows(
    rows: list[dict[str, Any]], ruler_root: str
) -> tuple[dict[str, float], float, dict[str, float]]:
    by_task, accuracy = score_predictions(rows, ruler_root)
    by_id = {
        row["sample_id"]: score_predictions([row], ruler_root)[1] for row in rows
    }
    return by_task, accuracy, by_id


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "attention_backend",
        "blasst_lambda",
        "fingerprint",
        "output_dir",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def analyze_run(
    run_dir: Path,
    manifest_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    dense_config: dict[str, Any],
    holdout_ids: set[str],
    ruler_root: str,
) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "predictions.jsonl")
    config = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "summary.json")
    if [identity(row) for row in rows] != [identity(row) for row in manifest_rows]:
        raise RuntimeError(f"sample identity/order mismatch: {run_dir}")
    if comparable_config(config) != comparable_config(dense_config):
        raise RuntimeError(f"non-BLASST configuration mismatch: {run_dir}")
    if any(row["task"] == "vt" for row in rows):
        raise RuntimeError(f"VT leaked into {run_dir}")

    dense_by_task, dense_accuracy, dense_scores = score_rows(dense_rows, ruler_root)
    by_task, accuracy, scores = score_rows(rows, ruler_root)
    differences = [scores[row["sample_id"]] - dense_scores[row["sample_id"]] for row in rows]
    tasks = [row["task"] for row in rows]

    dense_holdout = [row for row in dense_rows if row["sample_id"] in holdout_ids]
    sparse_holdout = [row for row in rows if row["sample_id"] in holdout_ids]
    _, dense_holdout_accuracy, dense_holdout_scores = score_rows(dense_holdout, ruler_root)
    _, holdout_accuracy, holdout_scores = score_rows(sparse_holdout, ruler_root)
    holdout_differences = [
        holdout_scores[row["sample_id"]] - dense_holdout_scores[row["sample_id"]]
        for row in sparse_holdout
    ]

    sparsity = summary["attention_sparsity"]
    result = {
        "run": run_dir.name,
        "lambda": float(config["blasst_lambda"]),
        "accuracy": accuracy,
        "accuracy_delta": accuracy - dense_accuracy,
        "physical_tile_sparsity": sparsity["physical_tile_sparsity"],
        "valid_element_sparsity": sparsity["valid_element_sparsity"],
        "row_vote_sparsity": sparsity["row_vote_sparsity"],
        "paired_wins": sum(value > 0 for value in differences),
        "paired_losses": sum(value < 0 for value in differences),
        "paired_ties": sum(value == 0 for value in differences),
        "holdout_samples": len(sparse_holdout),
        "holdout_accuracy": holdout_accuracy,
        "holdout_dense_accuracy": dense_holdout_accuracy,
        "holdout_accuracy_delta": holdout_accuracy - dense_holdout_accuracy,
        "per_task_accuracy": by_task,
        "per_task_delta": {
            task: by_task[task] - dense_by_task[task] for task in by_task
        },
        **bootstrap_ci(
            differences,
            tasks,
            task_level_differences=[
                by_task[task] - dense_by_task[task] for task in sorted(by_task)
            ],
        ),
        "holdout_paired_sample_bootstrap_95_ci": bootstrap_ci(
            holdout_differences,
            [row["task"] for row in sparse_holdout],
        )["paired_sample_bootstrap_95_ci"],
    }
    return result


def replay_check(
    smoke_path: Path,
    full_rows: list[dict[str, Any]],
) -> dict[str, int]:
    smoke_rows = read_jsonl(smoke_path)
    full_by_id = {row["sample_id"]: row for row in full_rows}
    matched = 0
    token_equal = 0
    text_equal = 0
    for row in smoke_rows:
        full = full_by_id[row["sample_id"]]
        matched += identity(row) == identity(full)
        token_equal += row["completion_tokens"] == full["completion_tokens"]
        text_equal += row["prediction"] == full["prediction"]
    return {
        "samples": len(smoke_rows),
        "identity_equal": matched,
        "completion_tokens_equal": token_equal,
        "prediction_text_equal": text_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    manifest_rows = read_jsonl(root / "full_manifest/samples.jsonl")
    screen_ids = {
        row["sample_id"] for row in read_jsonl(root / "screen_manifest/samples.jsonl")
    }
    holdout_ids = {row["sample_id"] for row in manifest_rows} - screen_ids
    if len(manifest_rows) != 600 or len(screen_ids) != 120 or len(holdout_ids) != 480:
        raise RuntimeError("expected 600 full, 120 screen, and 480 holdout samples")

    dense_dir = root / "runs/dense_eager"
    dense_rows = read_jsonl(dense_dir / "predictions.jsonl")
    dense_config = read_json(dense_dir / "run_config.json")
    if [identity(row) for row in dense_rows] != [identity(row) for row in manifest_rows]:
        raise RuntimeError("dense sample identity/order mismatch")
    dense_by_task, dense_accuracy, _ = score_rows(dense_rows, args.ruler_root)
    dense_holdout = [row for row in dense_rows if row["sample_id"] in holdout_ids]
    _, dense_holdout_accuracy, _ = score_rows(dense_holdout, args.ruler_root)

    run_dirs = sorted(
        path
        for path in (root / "runs").glob("lambda_*")
        if (path / "summary.json").is_file()
    )
    runs = [
        analyze_run(
            run_dir,
            manifest_rows,
            dense_rows,
            dense_config,
            holdout_ids,
            args.ruler_root,
        )
        for run_dir in run_dirs
    ]
    runs.sort(key=lambda row: row["lambda"])

    replay = {
        "dense": replay_check(
            root / "smoke/dense_b16/predictions.jsonl", dense_rows
        )
    }
    lambda_095 = next((row for row in run_dirs if row.name == "lambda_0p95"), None)
    if lambda_095 is not None:
        replay["lambda_0p95"] = replay_check(
            root / "smoke/lambda_0p95_b16/predictions.jsonl",
            read_jsonl(lambda_095 / "predictions.jsonl"),
        )

    report = {
        "model": "fast_dllm_v2_7b",
        "benchmark": "RULER 8K, 12 tasks excluding VT, 50 samples/task",
        "samples": 600,
        "screen_samples": 120,
        "holdout_samples": 480,
        "dense_accuracy": dense_accuracy,
        "dense_holdout_accuracy": dense_holdout_accuracy,
        "dense_per_task_accuracy": dense_by_task,
        "matched_configuration": comparable_config(dense_config),
        "repeatability": replay,
        "runs": runs,
    }
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "run",
            "lambda",
            "accuracy",
            "accuracy_delta",
            "physical_tile_sparsity",
            "valid_element_sparsity",
            "row_vote_sparsity",
            "holdout_accuracy",
            "holdout_accuracy_delta",
            "paired_wins",
            "paired_losses",
            "paired_ties",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in runs)

    lines = [
        "# fast-dLLM-v2 aggressive BLASST comparison",
        "",
        "Matched official RULER evaluation: 8K total length, 12 non-VT tasks, "
        "50 samples/task, BF16, Q tile 128, KV tile 64, and paired prompt/inference seeds.",
        "The model-native block size is 16. All 28 attention layers are global; "
        "BLASST is applied only to ordinary cached denoising queries, matching the "
        "fast-dLLM-v2 algorithm's intended eligibility.",
        "",
        f"Dense eager baseline: **{dense_accuracy:.2%}** (480-example holdout: "
        f"**{dense_holdout_accuracy:.2%}**).",
        "",
        "| Lambda | Accuracy | Delta | Physical sparsity | Valid-element sparsity | Holdout accuracy/delta | Wins/losses/ties |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['lambda']:.2f} | {row['accuracy']:.2%} | "
            f"{row['accuracy_delta']:+.2%} | {row['physical_tile_sparsity']:.2%} | "
            f"{row['valid_element_sparsity']:.2%} | {row['holdout_accuracy']:.2%} "
            f"({row['holdout_accuracy_delta']:+.2%}) | "
            f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} |"
        )
    lines.extend(["", "## Paired uncertainty", ""])
    for row in runs:
        sample_ci = row["paired_sample_bootstrap_95_ci"]
        task_ci = row["task_cluster_bootstrap_95_ci"]
        lines.append(
            f"- Lambda {row['lambda']:.2f}: paired-sample bootstrap 95% CI "
            f"[{sample_ci[0]:+.2%}, {sample_ci[1]:+.2%}]; task-cluster bootstrap "
            f"95% CI [{task_ci[0]:+.2%}, {task_ci[1]:+.2%}]."
        )
    lines.extend(["", "## Replay audit", ""])
    for name, check in replay.items():
        lines.append(
            f"- {name}: {check['prediction_text_equal']}/{check['samples']} prediction "
            "texts and "
            f"{check['completion_tokens_equal']}/{check['samples']} token sequences "
            "reproduced exactly between the smoke and full runs."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
