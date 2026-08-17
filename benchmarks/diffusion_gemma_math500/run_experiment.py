#!/usr/bin/env python3
"""Resumable DiffusionGemma BLASST evaluation on NeMo-Skills MATH500."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from dllm.attention.blasst import Blasst2DConfig, Blasst2DStats, install_blasst
from dllm.models import GenerationRequest, create_adapter


COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "masked_only_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)
PAPER_URL = "https://arxiv.org/pdf/2512.12087"
PAPER_SHA256 = "fc43af806cb7b115910c5d452faed450fd23008ea30eb293956b087dcb9a9ee7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-skills-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", default="google/diffusiongemma-26B-A4B-it"
    )
    parser.add_argument(
        "--revision", default="f7f5b7f5fa82ffc52addd066915886d497f5517b"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--samples-per-problem", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--local-lambda", type=float, default=0.9)
    parser.add_argument("--global-lambda", type=float, default=0.6)
    parser.add_argument(
        "--dense-baseline",
        action="store_true",
        help="disable BLASST and run native dense attention",
    )
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_nemo_skills(root: Path):
    sys.path.insert(0, str(root.resolve()))
    from nemo_skills.evaluation.math_grader import extract_answer, math_equal
    from nemo_skills.prompt.utils import get_prompt

    return extract_answer, math_equal, get_prompt


def git_revision(root: Path) -> str:
    head = root / ".git" / "HEAD"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        value = (root / ".git" / value[5:]).read_text(encoding="utf-8").strip()
    return value


def empty_counts() -> dict[str, int]:
    return {name: 0 for name in COUNT_FIELDS}


def add_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for name in COUNT_FIELDS:
        target[name] += int(source[name])


def sample_attention_counts(stats: Blasst2DStats) -> dict[str, dict[str, int]]:
    grouped = {"local": empty_counts(), "global": empty_counts()}
    for row in stats.per_step.values():
        attention_type = str(row.get("attention_type"))
        if attention_type not in grouped:
            raise RuntimeError(f"unexpected attention type in BLASST stats: {attention_type}")
        add_counts(grouped[attention_type], row)
    # The individual rows are persisted with each prediction. Keeping them in
    # memory for all 5,000 generations would make resume needlessly expensive.
    stats.per_step.clear()
    stats.per_layer.clear()
    stats.per_head.clear()
    stats.traces.clear()
    for attention_type, counts in grouped.items():
        if counts["eligible_tiles"] == 0:
            raise RuntimeError(f"no eligible {attention_type} BLASST tiles were observed")
    return grouped


def counts_with_ratios(counts: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(counts)
    result["physical_tile_sparsity"] = (
        counts["skipped_tiles"] / counts["eligible_tiles"]
        if counts["eligible_tiles"]
        else 0.0
    )
    result["row_vote_sparsity"] = (
        counts["skippable_row_votes"] / counts["valid_row_votes"]
        if counts["valid_row_votes"]
        else 0.0
    )
    result["valid_element_sparsity"] = (
        counts["skipped_valid_elements"] / counts["valid_elements"]
        if counts["valid_elements"]
        else 0.0
    )
    return result


def aggregate_attention(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = {"local": empty_counts(), "global": empty_counts()}
    for row in rows:
        for attention_type in grouped:
            add_counts(grouped[attention_type], row["attention_counts"][attention_type])
    return {name: counts_with_ratios(counts) for name, counts in grouped.items()}


def nemo_majority(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Match NeMo-Skills BaseMetrics._compute_majority_at_k tie semantics."""
    valid = [
        (row["predicted_answer"], bool(row["symbolic_correct"]))
        for row in predictions
        if row["predicted_answer"] is not None
    ]
    if not valid:
        return {
            "answer": None,
            "score": 0.0,
            "selected_score": 0.0,
            "votes": 0,
            "tie_count": 0,
        }
    counts = Counter(valid)
    votes = counts.most_common(1)[0][1]
    tied = sorted(pair for pair, count in counts.items() if count == votes)
    return {
        "answer": tied[0][0],
        "score": sum(float(correct) for _, correct in tied) / len(tied),
        # Keep a concrete, reproducible selected-answer score in addition to
        # NeMo-Skills' fractional credit when multiple modes tie.
        "selected_score": float(tied[0][1]),
        "votes": votes,
        "tie_count": len(tied),
    }


def summarize(
    rows: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    by_problem: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(int(row["problem_index"]), []).append(row)
    aggregate_rows = []
    for index, problem in enumerate(problems):
        predictions = sorted(by_problem[index], key=lambda row: int(row["sample_index"]))
        if len(predictions) != int(config["samples_per_problem"]):
            raise RuntimeError(f"problem {index} has {len(predictions)} predictions")
        majority = nemo_majority(predictions)
        aggregate_rows.append(
            {
                "problem_index": index,
                "unique_id": problem["unique_id"],
                "expected_answer": problem["expected_answer"],
                "majority_answer": majority["answer"],
                "majority_score": majority["score"],
                "majority_selected_score": majority["selected_score"],
                "majority_votes": majority["votes"],
                "majority_tie_count": majority["tie_count"],
                "any_symbolic_correct": any(row["symbolic_correct"] for row in predictions),
                "individual_symbolic_correct": sum(
                    bool(row["symbolic_correct"]) for row in predictions
                ),
            }
        )
    aggregate_path = output_dir / "self_consistency.jsonl"
    aggregate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in aggregate_rows),
        encoding="utf-8",
    )

    prompt_lengths = [
        int(sorted(by_problem[index], key=lambda row: int(row["sample_index"]))[0]["prompt_length"])
        for index in range(len(problems))
    ]
    attention = aggregate_attention(rows)
    total = len(rows)
    summary = {
        "schema_version": 1,
        "run_fingerprint": config["run_fingerprint"],
        "benchmark": "math-500",
        "num_problems": len(problems),
        "samples_per_problem": config["samples_per_problem"],
        "num_generations": total,
        "accuracy": {
            "primary_metric": "nemo_skills_majority_at_10",
            "nemo_skills_majority_at_10": sum(
                float(row["majority_score"]) for row in aggregate_rows
            )
            / len(aggregate_rows),
            "deterministic_majority_at_10": sum(
                float(row["majority_selected_score"]) for row in aggregate_rows
            )
            / len(aggregate_rows),
            "pass_at_1_avg_of_10": sum(bool(row["symbolic_correct"]) for row in rows)
            / total,
            "pass_at_10": sum(bool(row["any_symbolic_correct"]) for row in aggregate_rows)
            / len(aggregate_rows),
            "no_extracted_answer_rate": sum(row["predicted_answer"] is None for row in rows)
            / total,
            "majority_tie_problem_rate": sum(row["majority_tie_count"] > 1 for row in aggregate_rows)
            / len(aggregate_rows),
        },
        "physical_sparsity": attention,
        "prompt_length_tokens": {
            "definition": (
                "full DiffusionGemma chat-template input, including generation prefix"
                + (
                    " and thinking system/control turn"
                    if config["thinking"]
                    else ""
                )
            ),
            "minimum": min(prompt_lengths),
            "maximum": max(prompt_lengths),
            "mean": statistics.fmean(prompt_lengths),
            "median": statistics.median(prompt_lengths),
            "system_prompt": None,
            "few_shot_examples": 0,
            "thinking_control_token_enabled": bool(config["thinking"]),
        },
        "generation": {
            "total_model_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "mean_model_elapsed_seconds": statistics.fmean(
                float(row["elapsed_seconds"]) for row in rows
            ),
            "mean_completion_tokens": statistics.fmean(
                int(row["num_generated_tokens"]) for row in rows
            ),
            "length_termination_count": sum(
                row["termination_reason"] == "length" for row in rows
            ),
            "length_termination_rate": sum(
                row["termination_reason"] == "length" for row in rows
            )
            / total,
        },
        "peak_cuda_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "provenance": config["provenance"],
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary, config)
    return summary


def write_report(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    accuracy = summary["accuracy"]
    local = summary["physical_sparsity"]["local"]
    global_ = summary["physical_sparsity"]["global"]
    prompt = summary["prompt_length_tokens"]
    dense_baseline = bool(config.get("dense_baseline", False))
    attention_protocol = (
        "- Attention: native dense baseline; BLASST installation and masking disabled."
        if dense_baseline
        else (
            f"- BLASST policy: local lambda `{config['local_lambda']}`, global lambda "
            f"`{config['global_lambda']}`; Q tile `{config['q_tile_size']}`, KV tile "
            f"`{config['kv_tile_size']}`."
        )
    )
    sparsity_note = (
        "- Physical sparsity is 0% by definition for this native dense baseline."
        if dense_baseline
        else (
            "- Physical sparsity is `skipped eligible QxKV tiles / eligible QxKV tiles`, "
            "aggregated separately over native sliding/local and full/global decoder layers. "
            "This reference backend applies the sparse mask but still materializes dense QK "
            "scores, so it is not a kernel speed benchmark."
        )
    )
    report_title = (
        "DiffusionGemma dense baseline on MATH500"
        if dense_baseline
        else "DiffusionGemma BLASST on MATH500"
    )
    text = f"""# {report_title}

## Result

| Metric | Value |
|---|---:|
| NeMo-Skills majority@10 accuracy | {accuracy['nemo_skills_majority_at_10']:.2%} |
| Deterministic majority@10 accuracy | {accuracy['deterministic_majority_at_10']:.2%} |
| pass@1 (average of 10) | {accuracy['pass_at_1_avg_of_10']:.2%} |
| pass@10 | {accuracy['pass_at_10']:.2%} |
| Local-layer physical sparsity | {local['physical_tile_sparsity']:.2%} |
| Global-layer physical sparsity | {global_['physical_tile_sparsity']:.2%} |
| Prompt length (mean; min-max) | {prompt['mean']:.2f}; {prompt['minimum']}-{prompt['maximum']} tokens |

## Protocol

- Model: `{config['model_path']}` at `{config['revision']}` in BF16.
{attention_protocol}
- NeMo-Skills `generic/math` zero-shot prompt; no natural-language system prompt and no few-shot examples. DiffusionGemma thinking control is `{'enabled' if config['thinking'] else 'disabled'}`.
- 10 independent generations/problem, temperature `{config['temperature']}`, top-p `{config['top_p']}`, maximum `{config['max_new_tokens']}` new tokens.
- The primary score follows NeMo-Skills symbolic grading and exact-answer majority/self-consistency tie semantics.
- NeMo-Skills averages correctness across tied modes; deterministic majority uses the lexicographically first `(answer, correctness)` pair for the {accuracy['majority_tie_problem_rate']:.2%} of problems with a tied mode.
- {summary['generation']['length_termination_count']} of {summary['num_generations']} generations ({summary['generation']['length_termination_rate']:.2%}) reached the maximum-token cap; all others ended with EOS.
{sparsity_note}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_problem <= 0 or args.max_new_tokens <= 0:
        raise ValueError("samples-per-problem and max-new-tokens must be positive")
    extract_answer, math_equal, get_prompt = load_nemo_skills(args.nemo_skills_root)
    data_path = args.nemo_skills_root / "nemo_skills/dataset/math-500/test.jsonl"
    prompt_path = args.nemo_skills_root / "nemo_skills/prompt/config/generic/math.yaml"
    problems = read_jsonl(data_path)
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        raise RuntimeError("MATH500 dataset is empty; run NeMo-Skills prepare_data first")
    prompt_builder = get_prompt(str(prompt_path))

    base_config = {
        "schema_version": 1,
        "model_path": args.model_path,
        "revision": args.revision,
        "device": args.device,
        "precision": args.precision,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "samples_per_problem": args.samples_per_problem,
        "max_new_tokens": args.max_new_tokens,
        "base_seed": args.base_seed,
        "local_lambda": args.local_lambda,
        "global_lambda": args.global_lambda,
        "q_tile_size": args.q_tile_size,
        "kv_tile_size": args.kv_tile_size,
        "thinking": args.thinking,
        "num_problems": len(problems),
        "provenance": {
            "blasst_paper_url": PAPER_URL,
            "blasst_paper_pdf_sha256": PAPER_SHA256,
            "nemo_skills_commit": git_revision(args.nemo_skills_root),
            "math500_sha256": sha256_file(data_path),
            "nemo_prompt_sha256": sha256_file(prompt_path),
        },
    }
    # Preserve fingerprints of sparse runs created before this flag existed.
    if args.dense_baseline:
        base_config["dense_baseline"] = True
    fingerprint = stable_hash(base_config)
    config = {**base_config, "run_fingerprint": fingerprint}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "run_config.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config.get("run_fingerprint") != fingerprint:
            raise RuntimeError("output directory belongs to a different run fingerprint")
    else:
        write_json(run_config_path, config)

    predictions_path = args.output_dir / "predictions.jsonl"
    existing_rows = read_jsonl(predictions_path)
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in existing_rows:
        if row.get("run_fingerprint") != fingerprint:
            raise RuntimeError("prediction fingerprint does not match run configuration")
        key = (int(row["problem_index"]), int(row["sample_index"]))
        if key in completed:
            raise RuntimeError(f"duplicate prediction key: {key}")
        completed[key] = row

    adapter = create_adapter(
        "diffusion_gemma",
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    config.update(adapter.runtime_metadata())
    write_json(run_config_path, config)

    stats = None
    binding = None
    if not args.dense_baseline:
        stats = Blasst2DStats(record_layers=False, record_heads=False)
        blasst_config = Blasst2DConfig(
            enable_blasst_2d=True,
            blasst_lambda=args.global_lambda,
            local_blasst_lambda=args.local_lambda,
            global_blasst_lambda=args.global_lambda,
            q_tile_size=args.q_tile_size,
            kv_tile_size=args.kv_tile_size,
            collect_blasst_stats=True,
            collect_blasst_layer_stats=False,
            collect_blasst_head_stats=False,
            include_masked_kv_tiles_in_physical_stats=False,
            apply_blasst_mask=True,
        )
        binding = install_blasst(
            adapter.model,
            blasst_config,
            stats,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            attention_class_names=adapter.attention_class_names,
            module_selector=adapter.is_blasst_attention_module,
            query_ids_extractor=adapter.blasst_query_ids,
            filter_special_query_ids=adapter.blasst_filter_special_query_ids,
            call_selector=adapter.blasst_call_is_eligible,
            dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
            integration=adapter.attention_integration,
        )
    total_expected = len(problems) * args.samples_per_problem
    try:
        for problem_index, problem in enumerate(problems):
            messages = prompt_builder.fill(problem)
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise RuntimeError("generic/math unexpectedly produced a non-user prompt")
            prompt = str(messages[0]["content"])
            encoded_prompt = adapter.encode_prompt(prompt, {"thinking": args.thinking})
            for sample_index in range(args.samples_per_problem):
                key = (problem_index, sample_index)
                if key in completed:
                    continue
                seed = args.base_seed + problem_index * args.samples_per_problem + sample_index
                if binding is not None and stats is not None:
                    binding.runtime.metadata_context = {
                        "benchmark": "math-500",
                        "example_id": f"{problem_index}:{sample_index}",
                        "inference_seed": seed,
                    }
                    stats.per_step.clear()
                started = time.time()
                result = adapter.generate(
                    GenerationRequest(
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        block_size=256,
                        temperature=args.temperature,
                        seed=seed,
                        extra={
                            "thinking": args.thinking,
                            "top_p": args.top_p,
                        },
                    )
                )
                attention_counts = (
                    sample_attention_counts(stats)
                    if stats is not None
                    else {"local": empty_counts(), "global": empty_counts()}
                )
                predicted_answer = extract_answer(result.text)
                symbolic_correct = bool(
                    math_equal(problem["expected_answer"], predicted_answer)
                )
                row = {
                    "schema_version": 1,
                    "run_fingerprint": fingerprint,
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "unique_id": problem["unique_id"],
                    "subject": problem.get("subject"),
                    "level": problem.get("level"),
                    "problem": problem["problem"],
                    "expected_answer": problem["expected_answer"],
                    "prompt": prompt,
                    "prompt_length": len(result.prompt_tokens),
                    "prompt_length_prevalidated": len(encoded_prompt),
                    "seed": seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "generation": result.text,
                    "completion_tokens": result.completion_tokens,
                    "num_generated_tokens": len(result.completion_tokens),
                    "predicted_answer": predicted_answer,
                    "symbolic_correct": symbolic_correct,
                    "elapsed_seconds": result.elapsed_seconds,
                    "wall_end_time": time.time(),
                    "wall_start_time": started,
                    "termination_reason": result.termination_reason,
                    "model_evaluations": result.model_evaluations,
                    "generation_metadata": result.metadata,
                    "attention_counts": attention_counts,
                }
                if row["prompt_length"] != row["prompt_length_prevalidated"]:
                    raise RuntimeError("prompt token count changed between validation and generation")
                append_jsonl(predictions_path, row)
                completed[key] = row
                print(
                    f"completed {len(completed)}/{total_expected} "
                    f"problem={problem_index} sample={sample_index} "
                    f"correct={symbolic_correct} tokens={len(result.completion_tokens)} "
                    f"seconds={result.elapsed_seconds:.2f}",
                    flush=True,
                )
    finally:
        if binding is not None:
            binding.close()

    ordered = [
        completed[(problem_index, sample_index)]
        for problem_index in range(len(problems))
        for sample_index in range(args.samples_per_problem)
    ]
    summary = summarize(ordered, problems, config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
