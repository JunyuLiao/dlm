#!/usr/bin/env python3
"""Lightweight paired GSM8K accuracy evaluation for reference 2D-BLASST."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    generate_one,
    load_model,
    normalized_token_difference,
    set_seed,
)
from sparse_attention import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DStats,
    install_blasst_2d,
)


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-url", default=GSM8K_URL)
    parser.add_argument("--lambdas", default="1e-3,1e-2,1e-1,0.5")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default=str(V2_ROOT.parent / "results/blasst/fast_dllm_v2/context_sweep"),
    )
    return parser.parse_args()


def _download_or_validate(path: Path, url: str) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if url == GSM8K_URL and digest != GSM8K_SHA256:
        raise ValueError(
            f"official GSM8K checksum mismatch: got {digest}, "
            f"expected {GSM8K_SHA256}"
        )
    return digest


def _load_subset(
    path: Path,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    examples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if count <= 0 or count > len(examples):
        raise ValueError(f"num-samples must be in [1, {len(examples)}]")
    indices = sorted(random.Random(seed).sample(range(len(examples)), count))
    return [
        {
            "dataset_index": index,
            "question": examples[index]["question"],
            "answer": examples[index]["answer"],
        }
        for index in indices
    ], len(examples)


def _reference_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if match is None:
        raise ValueError(f"GSM8K answer lacks #### delimiter: {answer!r}")
    return _canonical_number(match.group(1))


def _canonical_number(text: str) -> str:
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".")
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return value.lower()
    normalized = format(decimal.normalize(), "f")
    return "0" if normalized in ("-0", "") else normalized


def _predicted_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    candidates = boxed or re.findall(
        r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?",
        text,
    )
    return _canonical_number(candidates[-1]) if candidates else None


def _prompt(tokenizer, question: str) -> str:
    content = (
        f"Question: {question}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return content


def _wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
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


def _update_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "## Lightweight actual task accuracy",
        "",
        f"A deterministic {args.num_samples}-example subset of the official GSM8K "
        "test split was scored by extracted final-answer exact match. This is real "
        "labeled-task accuracy, but the subset is a smoke test—not a statistically "
        "precise or directly publishable full-benchmark score.",
        "",
        "| Mode | λ | Correct | Accuracy | 95% Wilson interval | "
        "Boxed-answer rate | Extracted-answer agreement with dense |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['lambda']} | "
            f"{row['correct']}/{row['samples']} | {row['accuracy']:.3f} | "
            f"[{row['accuracy_ci95_low']:.3f}, {row['accuracy_ci95_high']:.3f}] | "
            f"{row['boxed_answer_rate']:.3f} | "
            f"{row['answer_agreement_with_dense']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Across modes, the boxed-answer rate ranges from "
            f"{min(float(row['boxed_answer_rate']) for row in rows):.1%} to "
            f"{max(float(row['boxed_answer_rate']) for row in rows):.1%}, and the "
            f"generation-budget hit rate ranges from "
            f"{min(float(row['generation_budget_hit_rate']) for row in rows):.1%} to "
            f"{max(float(row['generation_budget_hit_rate']) for row in rows):.1%}.",
            "",
            "The exact prompts, generations, extracted answers, labels, sample indices, "
            "and per-mode BLASST counts are in `task_predictions.json` and "
            "`task_sparsity.json`. Agreement with dense remains a diagnostic; only the "
            "GSM8K exact-match column is task accuracy.",
            "",
        ]
    )
    section = "\n".join(lines)
    report_path = output_dir / "report.md"
    if report_path.exists():
        original = report_path.read_text(encoding="utf-8")
        marker = "\n## Lightweight actual task accuracy\n"
        original = original.split(marker, 1)[0].rstrip()
        report_path.write_text(original + "\n\n" + section, encoding="utf-8")
    else:
        (output_dir / "task_report.md").write_text(
            "# BLASST task evaluation\n\n" + section,
            encoding="utf-8",
        )


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    lambdas = [
        float(value.strip()) for value in args.lambdas.split(",") if value.strip()
    ]
    if any(not 0.0 < value <= 1.0 for value in lambdas):
        raise ValueError("every lambda must be in (0, 1]")
    if args.max_new_tokens % args.block_size:
        raise ValueError("max-new-tokens must be a block-size multiple")
    output_dir = Path(args.output_dir)
    from dllm.attention.blasst import BLASST_MASK_SEMANTICS, validate_blasst_output_directory
    validate_blasst_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path
        else output_dir / "gsm8k_test.jsonl"
    )
    digest = _download_or_validate(dataset_path, args.dataset_url)
    examples, dataset_size = _load_subset(
        dataset_path,
        args.num_samples,
        args.seed,
    )
    (output_dir / "gsm8k_samples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
    sparse_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=lambdas[0],
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
    )
    runtime = install_blasst_2d(
        model,
        replace(sparse_config, enable_blasst_2d=False),
        mask_token_id=getattr(model.config, "mask_token_id", 151665),
        pad_token_id=tokenizer.pad_token_id,
        dual_cache_only=True,
    )

    predictions: list[dict[str, Any]] = []
    sparsity: dict[str, Any] = {}
    modes: list[tuple[str, float | None]] = [("dense", None)] + [
        ("blasst", value) for value in lambdas
    ]
    dense_by_index: dict[int, dict[str, Any]] = {}
    for mode, lambda_value in modes:
        runtime.stats = Blasst2DStats()
        runtime.forward_call_index = 0
        runtime.config = (
            replace(sparse_config, enable_blasst_2d=False)
            if lambda_value is None
            else replace(
                sparse_config,
                enable_blasst_2d=True,
                blasst_lambda=lambda_value,
            )
        )
        for local_index, example in enumerate(examples):
            sample_seed = args.seed + int(example["dataset_index"])
            set_seed(sample_seed)
            prompt = _prompt(tokenizer, str(example["question"]))
            generated = generate_one(
                model,
                tokenizer,
                prompt,
                block_size=args.block_size,
                max_new_tokens=args.max_new_tokens,
                threshold=args.denoising_threshold,
            )
            prediction = _predicted_answer(generated["completion"])
            reference = _reference_answer(str(example["answer"]))
            prompt_remainder = len(generated["prompt_tokens"]) % args.block_size
            completion_capacity = args.max_new_tokens - prompt_remainder
            if prompt_remainder == 0:
                completion_capacity = args.max_new_tokens
            row = {
                "mode": mode,
                "lambda": lambda_value,
                "sample": local_index,
                "dataset_index": int(example["dataset_index"]),
                "seed": sample_seed,
                "question": example["question"],
                "reference_reasoning": example["answer"],
                "reference_answer": reference,
                "prompt": prompt,
                "completion": generated["completion"],
                "completion_tokens": generated["completion_tokens"],
                "completion_token_count": len(generated["completion_tokens"]),
                "completion_capacity": completion_capacity,
                "reached_generation_budget": (
                    len(generated["completion_tokens"]) >= completion_capacity
                ),
                "boxed_answer_present": bool(
                    re.search(r"\\boxed\s*\{[^{}]+\}", generated["completion"])
                ),
                "predicted_answer": prediction,
                "correct": int(prediction == reference),
            }
            if mode == "dense":
                dense_by_index[int(example["dataset_index"])] = row
                row["answer_agrees_with_dense"] = 1
                row["sequence_exact_match_with_dense"] = 1
                row["normalized_token_difference_from_dense"] = 0.0
            else:
                dense = dense_by_index[int(example["dataset_index"])]
                row["answer_agrees_with_dense"] = int(
                    prediction == dense["predicted_answer"]
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
            sparsity[str(lambda_value)] = runtime.stats.summary()

    accuracy_rows: list[dict[str, Any]] = []
    for mode, lambda_value in modes:
        subset = [
            row
            for row in predictions
            if row["mode"] == mode and row["lambda"] == lambda_value
        ]
        correct = sum(int(row["correct"]) for row in subset)
        low, high = _wilson(correct, len(subset))
        accuracy_rows.append(
            {
                "mode": mode,
                "lambda": "dense" if lambda_value is None else lambda_value,
                "samples": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "boxed_answer_rate": sum(
                    int(row["boxed_answer_present"]) for row in subset
                )
                / len(subset),
                "generation_budget_hit_rate": sum(
                    int(row["reached_generation_budget"]) for row in subset
                )
                / len(subset),
                "answer_agreement_with_dense": sum(
                    int(row["answer_agrees_with_dense"]) for row in subset
                )
                / len(subset),
                "sequence_exact_match_with_dense": sum(
                    int(row["sequence_exact_match_with_dense"]) for row in subset
                )
                / len(subset),
                "mean_normalized_token_difference_from_dense": sum(
                    float(row["normalized_token_difference_from_dense"])
                    for row in subset
                )
                / len(subset),
            }
        )

    _write_csv(output_dir / "task_accuracy.csv", accuracy_rows)
    (output_dir / "task_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "task_sparsity.json").write_text(
        json.dumps(sparsity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion_checks = {
        "passed": all(row["boxed_answer_present"] for row in predictions)
        and not any(row["reached_generation_budget"] for row in predictions),
        "total_generations_checked": len(predictions),
        "all_modes_boxed_answer_rate": sum(
            int(row["boxed_answer_present"]) for row in predictions
        )
        / len(predictions),
        "generation_budget_hit_rate": sum(
            int(row["reached_generation_budget"]) for row in predictions
        )
        / len(predictions),
        "max_completion_tokens_observed": max(
            int(row["completion_token_count"]) for row in predictions
        ),
        "max_new_tokens": args.max_new_tokens,
    }
    (output_dir / "task_completion_checks.json").write_text(
        json.dumps(completion_checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_config = {
        "blasst_mask_semantics": BLASST_MASK_SEMANTICS,
        **vars(args),
        "lambdas": lambdas,
        "dataset_path": str(dataset_path),
        "dataset_sha256": digest,
        "dataset_total_test_examples": dataset_size,
        "selected_indices": [row["dataset_index"] for row in examples],
        "selection": "uniform random sample without replacement using seed",
        "metric": "exact match of canonicalized final numeric answer",
        "prompt_semantics": (
            "repository GSM8K instruction plus checkpoint chat template"
        ),
        "paired_control": (
            "same examples and per-example RNG seeds in dense and every lambda mode"
        ),
        "scope": (
            "lightweight subset smoke evaluation; not a replacement for the full "
            "1319-example lm-evaluation-harness score"
        ),
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "observed_wall_seconds_through_export": time.monotonic() - started_at,
    }
    (output_dir / "task_run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_report(output_dir, accuracy_rows, args)
    print(json.dumps(accuracy_rows, indent=2))


if __name__ == "__main__":
    main()
