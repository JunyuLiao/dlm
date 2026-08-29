#!/usr/bin/env python3
"""Paired GSM8K/HumanEval sanity metrics with sub-blocks disabled."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import torch


V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    generate_one,
    load_model,
    normalized_token_difference,
    set_seed,
)
from scripts.sweep_blasst_controlled import _build_manifest  # noqa: E402
from sparse_attention import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DStats,
    install_blasst_2d,
)


SAFE_IMPORTS = {
    "bisect",
    "collections",
    "copy",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--gsm8k-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument(
        "--lambdas",
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,0.5",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/blasst/fast_dllm_v2/controlled_sweeps"),
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Re-audit existing task artifacts and refresh the report without inference.",
    )
    return parser.parse_args()


def _canonical_number(text: str) -> str:
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".")
    try:
        normalized = format(Decimal(value).normalize(), "f")
    except InvalidOperation:
        return value.lower()
    return "0" if normalized in ("", "-0") else normalized


def _gsm_prediction(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    candidates = boxed or re.findall(
        r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?",
        text,
    )
    return _canonical_number(candidates[-1]) if candidates else None


def _strip_code(text: str, entry_point: str, official_prompt: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    candidate = fenced[-1] if fenced else text
    candidate = candidate.strip()
    if f"def {entry_point}" not in candidate:
        candidate = official_prompt + candidate
    else:
        starts = [
            index
            for marker in ("from ", "import ", f"def {entry_point}")
            if (index := candidate.find(marker)) >= 0
        ]
        if starts:
            candidate = candidate[min(starts) :]
    return candidate.strip() + "\n"


def _static_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"syntax_error: {error.msg} line {error.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in SAFE_IMPORTS:
                    return False, f"forbidden_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS:
                return False, f"forbidden_import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"forbidden_call: {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden_dunder_attribute: {node.attr}"
    return True, "safe"


def _resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _execute_humaneval(
    completion: str,
    label: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    candidate = _strip_code(
        completion,
        str(label["entry_point"]),
        str(label.get("prompt", "")),
    )
    safe, reason = _static_safety(candidate)
    if not safe:
        return {
            "passed": False,
            "status": reason,
            "candidate": candidate,
            "stdout": "",
            "stderr": "",
        }
    program = (
        candidate
        + "\n"
        + str(label["test"])
        + "\n"
        + f"check({label['entry_point']})\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="blasst-humaneval-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-"],
                input=program,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=directory,
                env={"PYTHONHASHSEED": "0"},
                preexec_fn=_resource_limits,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        return {
            "passed": False,
            "status": "timeout",
            "candidate": candidate,
            "stdout": (error.stdout or "")[-2000:],
            "stderr": (error.stderr or "")[-2000:],
        }
    return {
        "passed": completed.returncode == 0,
        "status": "passed" if completed.returncode == 0 else "failed_tests",
        "returncode": completed.returncode,
        "candidate": candidate,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def _wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_counts(stats: Blasst2DStats) -> dict[str, Any]:
    summary = stats.summary()
    query_lengths = sorted(
        {int(row["query_length"]) for row in stats.per_step.values()}
    )
    if query_lengths and query_lengths != [16]:
        raise AssertionError(f"residual task query lengths: {query_lengths}")
    return {**summary, "query_lengths": query_lengths}


def _append_report(
    output_dir: Path,
    accuracy: list[dict[str, Any]],
    elapsed: float,
) -> None:
    lines = [
        "## Fixed-manifest task sanity checks",
        "",
        "These are paired sanity metrics on exactly eight GSM8K and eight HumanEval "
        "examples. HumanEval pass@1 executes generated code after static rejection of "
        "filesystem/process/network imports and inside a resource-limited subprocess. "
        "The sample is too small for publication-quality accuracy estimates.",
        "",
        "| Mode | λ | Benchmark | Correct | Accuracy | 95% Wilson interval | Dense outcome agreement |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in accuracy:
        lines.append(
            f"| {row['mode']} | {row['lambda']} | {row['benchmark']} | "
            f"{row['correct']}/{row['samples']} | {row['accuracy']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | "
            f"{row['outcome_agreement_with_dense']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Task-evaluation wall time: {elapsed:.1f} seconds. Exact prompts, labels, "
            "seeds, generations, extracted answers, candidate programs, and execution "
            "statuses are stored in the manifest and task prediction artifacts. "
            "Agreement is diagnostic; only labeled exact match/pass@1 is accuracy. "
            "With eight examples per benchmark, apparent improvements or regressions "
            "inside the overlapping Wilson intervals are not statistically resolved.",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    original = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "\n## Fixed-manifest task sanity checks\n"
    original = original.split(marker, 1)[0].rstrip()
    report_path.write_text(
        original + "\n\n" + "\n".join(lines),
        encoding="utf-8",
    )


def _audit_task_outputs(
    predictions: list[dict[str, Any]],
    task_sparsity: dict[str, Any],
    lambdas: list[float],
) -> dict[str, Any]:
    violations: list[str] = []
    modes: list[tuple[str, float | None]] = [("dense", None)] + [
        ("blasst", value) for value in lambdas
    ]
    if len(predictions) != len(modes) * 16:
        violations.append("prediction row count mismatch")
    for mode, lambda_value in modes:
        subset = [
            row
            for row in predictions
            if row["mode"] == mode and row["lambda"] == lambda_value
        ]
        if len(subset) != 16:
            violations.append(f"mode sample count mismatch: {mode}/{lambda_value}")
        for benchmark in ("gsm8k", "humaneval"):
            if sum(row["benchmark"] == benchmark for row in subset) != 8:
                violations.append(
                    f"benchmark count mismatch: {mode}/{lambda_value}/{benchmark}"
                )
    expected_lambda_keys = {str(value) for value in lambdas}
    if set(task_sparsity) != expected_lambda_keys:
        violations.append("task sparsity threshold keys mismatch")
    for lambda_value, summary in task_sparsity.items():
        if summary["query_lengths"] != [16]:
            violations.append(f"residual task query length at λ={lambda_value}")
        if int(summary["eligible_tiles"]) != (
            int(summary["skipped_tiles"]) + int(summary["retained_tiles"])
        ):
            violations.append(f"task count identity failed at λ={lambda_value}")
        for key, value in summary.items():
            if isinstance(value, float) and not math.isfinite(value):
                violations.append(
                    f"non-finite task statistic {key} at λ={lambda_value}"
                )
    human_rows = [
        row for row in predictions if row["benchmark"] == "humaneval"
    ]
    budget_values = [
        bool(row["reached_generation_budget"])
        for row in predictions
        if "reached_generation_budget" in row
    ]
    return {
        "passed": not violations,
        "violations": violations,
        "prediction_rows": len(predictions),
        "modes": len(modes),
        "examples_per_mode": 16,
        "examples_per_benchmark_per_mode": 8,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "observed_sparse_query_lengths": sorted(
            {
                query_length
                for summary in task_sparsity.values()
                for query_length in summary["query_lengths"]
            }
        ),
        "humaneval_timeouts": sum(
            row["execution"]["status"] == "timeout" for row in human_rows
        ),
        "humaneval_static_safety_rejections": sum(
            str(row["execution"]["status"]).startswith("forbidden")
            for row in human_rows
        ),
        "generation_budget_hits": (
            sum(budget_values) if len(budget_values) == len(predictions) else None
        ),
        "generation_budget_field_recorded": len(budget_values) == len(predictions),
    }


def _finalize_existing(
    output_dir: Path,
    lambdas: list[float],
) -> None:
    task_config = json.loads(
        (output_dir / "task_run_config.json").read_text(encoding="utf-8")
    )
    with (output_dir / "task_accuracy.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        accuracy = list(csv.DictReader(handle))
    for row in accuracy:
        for key in (
            "accuracy",
            "ci95_low",
            "ci95_high",
            "outcome_agreement_with_dense",
            "sequence_exact_match_with_dense",
        ):
            row[key] = float(row[key])
        for key in ("correct", "samples"):
            row[key] = int(row[key])
    predictions = json.loads(
        (output_dir / "task_predictions.json").read_text(encoding="utf-8")
    )
    if predictions and "reached_generation_budget" not in predictions[0]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            task_config["model_path"],
            trust_remote_code=True,
        )
        block_size = int(task_config["block_size"])
        max_new_tokens = int(task_config["max_new_tokens"])
        for row in predictions:
            prompt_tokens = tokenizer(str(row["prompt"]))["input_ids"]
            prompt_remainder = len(prompt_tokens) % block_size
            completion_capacity = (
                max_new_tokens - prompt_remainder
                if prompt_remainder
                else max_new_tokens
            )
            row["completion_token_count"] = len(row["completion_tokens"])
            row["completion_capacity"] = completion_capacity
            row["reached_generation_budget"] = (
                len(row["completion_tokens"]) >= completion_capacity
            )
        (output_dir / "task_predictions.json").write_text(
            json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    task_sparsity = json.loads(
        (output_dir / "task_sparsity_diagnostic.json").read_text(encoding="utf-8")
    )
    task_checks = _audit_task_outputs(predictions, task_sparsity, lambdas)
    (output_dir / "task_correctness_checks.json").write_text(
        json.dumps(task_checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not task_checks["passed"]:
        raise AssertionError(json.dumps(task_checks, indent=2))
    _append_report(
        output_dir,
        accuracy,
        float(task_config["observed_wall_seconds"]),
    )
    print(json.dumps(task_checks, indent=2, sort_keys=True))


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    lambdas = [
        float(value.strip()) for value in args.lambdas.split(",") if value.strip()
    ]
    if args.block_size != 16:
        raise ValueError("task evaluation requires block_size=16")
    if args.max_new_tokens % args.block_size:
        raise ValueError("max-new-tokens must be divisible by block_size")
    output_dir = Path(args.output_dir)
    if args.finalize_existing:
        _finalize_existing(output_dir, lambdas)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
    examples, manifest = _build_manifest(
        tokenizer,
        output_dir,
        args.gsm8k_path,
        args.humaneval_path,
        args.seed,
    )
    by_id = {
        str(example["example_id"]): example for example in manifest["examples"]
    }
    config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=lambdas[0],
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
    )
    runtime = install_blasst_2d(
        model,
        replace(config, enable_blasst_2d=False),
        mask_token_id=getattr(model.config, "mask_token_id", 151665),
        pad_token_id=tokenizer.pad_token_id,
        dual_cache_only=False,
        ordinary_cache_queries_only=True,
    )

    predictions: list[dict[str, Any]] = []
    task_sparsity: dict[str, Any] = {}
    dense_by_id: dict[str, dict[str, Any]] = {}
    modes: list[tuple[str, float | None]] = [("dense", None)] + [
        ("blasst", value) for value in lambdas
    ]
    for mode, lambda_value in modes:
        runtime.stats = Blasst2DStats(record_layers=False, record_heads=False)
        runtime.forward_call_index = 0
        runtime.config = (
            replace(config, enable_blasst_2d=False)
            if lambda_value is None
            else replace(
                config,
                enable_blasst_2d=True,
                blasst_lambda=lambda_value,
            )
        )
        for example in examples:
            example_id = str(example["example_id"])
            set_seed(int(example["seed"]))
            generated = generate_one(
                model,
                tokenizer,
                str(example["task_prompt"]),
                block_size=16,
                max_new_tokens=args.max_new_tokens,
                threshold=args.denoising_threshold,
            )
            completion = str(generated["completion"])
            prompt_remainder = len(generated["prompt_tokens"]) % args.block_size
            completion_capacity = args.max_new_tokens - prompt_remainder
            if prompt_remainder == 0:
                completion_capacity = args.max_new_tokens
            label = by_id[example_id]["label"]
            if example["benchmark"] == "gsm8k":
                prediction = _gsm_prediction(completion)
                reference = _canonical_number(str(label["final_answer"]))
                correct = prediction == reference
                outcome = prediction
                execution = None
            else:
                execution_label = {
                    **label,
                    "prompt": example["sparsity_prompt"],
                }
                execution = _execute_humaneval(
                    completion,
                    execution_label,
                    args.execution_timeout,
                )
                prediction = None
                reference = "passes official tests"
                correct = bool(execution["passed"])
                outcome = execution["status"]
            row = {
                "mode": mode,
                "lambda": lambda_value,
                "benchmark": example["benchmark"],
                "example_id": example_id,
                "dataset_index": example["dataset_index"],
                "seed": example["seed"],
                "prompt": example["task_prompt"],
                "label": label,
                "completion": completion,
                "completion_tokens": generated["completion_tokens"],
                "completion_token_count": len(generated["completion_tokens"]),
                "completion_capacity": completion_capacity,
                "reached_generation_budget": (
                    len(generated["completion_tokens"]) >= completion_capacity
                ),
                "predicted_answer": prediction,
                "reference": reference,
                "correct": int(correct),
                "execution": execution,
                "outcome": outcome,
            }
            if mode == "dense":
                dense_by_id[example_id] = row
                row["outcome_agrees_with_dense"] = 1
                row["sequence_exact_match_with_dense"] = 1
                row["normalized_token_difference_from_dense"] = 0.0
            else:
                dense = dense_by_id[example_id]
                row["outcome_agrees_with_dense"] = int(
                    outcome == dense["outcome"]
                )
                row["sequence_exact_match_with_dense"] = int(
                    generated["completion_tokens"] == dense["completion_tokens"]
                )
                row["normalized_token_difference_from_dense"] = (
                    normalized_token_difference(
                        generated["completion_tokens"],
                        dense["completion_tokens"],
                    )
                )
            predictions.append(row)
        if lambda_value is not None:
            task_sparsity[str(lambda_value)] = _summary_counts(runtime.stats)
        print(
            f"completed task mode {mode} λ={lambda_value}",
            flush=True,
        )

    accuracy: list[dict[str, Any]] = []
    for mode, lambda_value in modes:
        for benchmark in ("gsm8k", "humaneval"):
            subset = [
                row
                for row in predictions
                if row["mode"] == mode
                and row["lambda"] == lambda_value
                and row["benchmark"] == benchmark
            ]
            correct = sum(int(row["correct"]) for row in subset)
            low, high = _wilson(correct, len(subset))
            accuracy.append(
                {
                    "mode": mode,
                    "lambda": "dense" if lambda_value is None else lambda_value,
                    "benchmark": benchmark,
                    "metric": (
                        "extracted_answer_exact_match"
                        if benchmark == "gsm8k"
                        else "execution_pass_at_1"
                    ),
                    "samples": len(subset),
                    "correct": correct,
                    "accuracy": correct / len(subset),
                    "ci95_low": low,
                    "ci95_high": high,
                    "outcome_agreement_with_dense": sum(
                        int(row["outcome_agrees_with_dense"]) for row in subset
                    )
                    / len(subset),
                    "sequence_exact_match_with_dense": sum(
                        int(row["sequence_exact_match_with_dense"])
                        for row in subset
                    )
                    / len(subset),
                }
            )

    elapsed = time.monotonic() - started
    _write_csv(output_dir / "task_accuracy.csv", accuracy)
    (output_dir / "task_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "task_sparsity_diagnostic.json").write_text(
        json.dumps(task_sparsity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_checks = _audit_task_outputs(predictions, task_sparsity, lambdas)
    (output_dir / "task_correctness_checks.json").write_text(
        json.dumps(task_checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not task_checks["passed"]:
        raise AssertionError(json.dumps(task_checks, indent=2))
    task_config = {
        **vars(args),
        "lambdas": lambdas,
        "manifest_path": str(output_dir / "evaluation_manifest.json"),
        "manifest_examples": 16,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "ordinary_cache_queries_only": True,
        "same_examples_and_seeds_across_modes": True,
        "humaneval_execution": (
            "static unsafe-import/call rejection, python -I, CPU/address-space/"
            "file-size/open-file limits, wall timeout"
        ),
        "observed_wall_seconds": elapsed,
    }
    (output_dir / "task_run_config.json").write_text(
        json.dumps(task_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_report(output_dir, accuracy, elapsed)
    print(
        json.dumps(
            {
                "accuracy": accuracy,
                "elapsed_seconds": elapsed,
                "prediction_rows": len(predictions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
