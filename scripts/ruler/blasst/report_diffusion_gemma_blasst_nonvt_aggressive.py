#!/usr/bin/env python3
"""Report an aggressive local/global BLASST calibration with VT excluded."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.io import write_json
from dllm.evaluation.ruler.official import score_predictions


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["sample_id"], row["prompt_sha256"], int(row["inference_seed"])


def evaluate_runs(
    directory: Path,
    expected_rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
    ruler_root: str,
) -> list[dict[str, Any]]:
    results = []
    expected_identity = [identity(row) for row in expected_rows]
    for run_dir in sorted(path for path in directory.glob("*") if path.is_dir()):
        config_path = run_dir / "run_config.json"
        predictions_path = run_dir / "predictions.jsonl"
        summary_path = run_dir / "summary.json"
        if not (config_path.is_file() and predictions_path.is_file() and summary_path.is_file()):
            continue
        config = read_json(config_path)
        predictions = read_jsonl(predictions_path)
        if [identity(row) for row in predictions] != expected_identity:
            raise RuntimeError(f"sample identity mismatch in {run_dir}")
        if any(row["task"] == "vt" for row in predictions):
            raise RuntimeError(f"VT leaked into {run_dir}")
        if config["attention_backend"] != "blasst-reference":
            raise RuntimeError(f"{run_dir} is not a BLASST reference run")
        policy = config["blasst_policy"]
        local_lambda = float(policy["local_blasst_lambda"])
        global_lambda = float(policy["global_blasst_lambda"])
        by_task, accuracy = score_predictions(predictions, ruler_root)
        sparse_scores = {
            row["sample_id"]: score_predictions([row], ruler_root)[1]
            for row in predictions
        }
        summary = read_json(summary_path)
        results.append(
            {
                "run": run_dir.name,
                "local_lambda": local_lambda,
                "global_lambda": global_lambda,
                "samples": len(predictions),
                "accuracy": accuracy,
                "accuracy_delta": accuracy
                - sum(baseline_scores.values()) / len(baseline_scores),
                "physical_sparsity": summary["attention_sparsity"][
                    "physical_tile_sparsity"
                ],
                "paired_wins": sum(
                    sparse_scores[key] > baseline_scores[key] for key in baseline_scores
                ),
                "paired_losses": sum(
                    sparse_scores[key] < baseline_scores[key] for key in baseline_scores
                ),
                "paired_ties": sum(
                    sparse_scores[key] == baseline_scores[key] for key in baseline_scores
                ),
                "per_task_accuracy": by_task,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    parser.add_argument("--allowed-accuracy-drop", type=float, default=0.05)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    source = args.source_root.resolve()
    dense_rows = read_jsonl(source / "runs/dense_eager_calibration/predictions.jsonl")
    dense_by_id = {row["sample_id"]: row for row in dense_rows}

    report: dict[str, Any] = {
        "excluded_tasks": ["vt"],
        "allowed_accuracy_drop": args.allowed_accuracy_drop,
        "splits": {},
    }
    for split, samples_per_task in (("screen", 10), ("full", 50)):
        expected_rows = read_jsonl(root / f"{split}_manifest/samples.jsonl")
        if len(expected_rows) != 12 * samples_per_task:
            raise RuntimeError(f"unexpected {split} manifest size")
        baseline_rows = [dense_by_id[row["sample_id"]] for row in expected_rows]
        if [identity(row) for row in baseline_rows] != [identity(row) for row in expected_rows]:
            raise RuntimeError(f"dense baseline mismatch for {split}")
        baseline_by_task, baseline_accuracy = score_predictions(
            baseline_rows, args.ruler_root
        )
        baseline_scores = {
            row["sample_id"]: score_predictions([row], args.ruler_root)[1]
            for row in baseline_rows
        }
        runs = evaluate_runs(
            root / split, expected_rows, baseline_scores, args.ruler_root
        )
        first_degraded = next(
            (
                row
                for row in runs
                if row["accuracy"] <= baseline_accuracy - args.allowed_accuracy_drop
            ),
            None,
        )
        limiting_run = max(
            runs,
            key=lambda row: (
                min(row["local_lambda"], row["global_lambda"]),
                row["local_lambda"] + row["global_lambda"],
            ),
            default=None,
        )
        report["splits"][split] = {
            "samples": len(expected_rows),
            "samples_per_task": samples_per_task,
            "baseline_accuracy": baseline_accuracy,
            "degradation_threshold": baseline_accuracy - args.allowed_accuracy_drop,
            "baseline_by_task": baseline_by_task,
            "runs": runs,
            "first_degraded_run": first_degraded,
            "limiting_valid_lambda_run": limiting_run,
        }

    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    flat_rows = []
    for split, split_report in report["splits"].items():
        for row in split_report["runs"]:
            flat_rows.append(
                {
                    "split": split,
                    **{key: value for key, value in row.items() if key != "per_task_accuracy"},
                }
            )
    if flat_rows:
        with (output / "calibration_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)

    lines = [
        "# DiffusionGemma aggressive BLASST calibration without VT",
        "",
        "Accuracy is official RULER accuracy over 12 equally represented tasks; VT is excluded.",
        f"The degradation boundary is an absolute {args.allowed_accuracy_drop:.0%} "
        "(five percentage points) below the paired eager-dense baseline.",
        "",
    ]
    for split, split_report in report["splits"].items():
        lines.extend(
            [
                f"## {split.title()} ({split_report['samples']} examples)",
                "",
                f"Dense baseline: {split_report['baseline_accuracy']:.2%}; boundary: {split_report['degradation_threshold']:.2%}.",
                "",
                "| Run | Local lambda | Global lambda | Accuracy | Delta | Physical sparsity | Wins/losses/ties |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in split_report["runs"]:
            lines.append(
                f"| {row['run']} | {row['local_lambda']:.6g} | {row['global_lambda']:.6g} | "
                f"{row['accuracy']:.2%} | {row['accuracy_delta']:+.2%} | "
                f"{row['physical_sparsity']:.2%} | "
                f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} |"
            )
        boundary = split_report["first_degraded_run"]
        if boundary is None:
            limiting = split_report["limiting_valid_lambda_run"]
            lines.extend(
                [
                    "",
                    "No evaluated run reached the degradation boundary. "
                    f"The limiting valid-lambda run used local `{limiting['local_lambda']}` "
                    f"and global `{limiting['global_lambda']}`, scoring "
                    f"{limiting['accuracy']:.2%} ({limiting['accuracy_delta']:+.2%}) "
                    f"at {limiting['physical_sparsity']:.2%} physical sparsity.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    f"First boundary crossing: local lambda `{boundary['local_lambda']}`, "
                    f"global lambda `{boundary['global_lambda']}` "
                    f"({boundary['accuracy']:.2%}, {boundary['accuracy_delta']:+.2%}).",
                    "",
                ]
            )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
