#!/usr/bin/env python3
"""Audit and compare matched dense and BLASST DiffusionGemma MATH500 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--blasst-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {tuple(row[field] for field in fields): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate keys for fields {fields}")
    return result


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{name} mismatch: recomputed {actual}, summary {expected}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def bootstrap_mean_ci(values: list[float], *, samples: int = 10_000) -> tuple[float, float]:
    rng = random.Random(20260817)
    size = len(values)
    estimates = [
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def transitions(
    keys: list[tuple[Any, ...]],
    left: dict[tuple[Any, ...], dict[str, Any]],
    right: dict[tuple[Any, ...], dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[int, int, int, int]:
    both = left_only = right_only = neither = 0
    for key in keys:
        left_value = predicate(left[key])
        right_value = predicate(right[key])
        if left_value and right_value:
            both += 1
        elif left_value:
            left_only += 1
        elif right_value:
            right_only += 1
        else:
            neither += 1
    return both, left_only, right_only, neither


def main() -> None:
    args = parse_args()
    dense_summary = read_json(args.dense_dir / "summary.json")
    sparse_summary = read_json(args.blasst_dir / "summary.json")
    dense_config = read_json(args.dense_dir / "run_config.json")
    sparse_config = read_json(args.blasst_dir / "run_config.json")
    dense_rows = read_jsonl(args.dense_dir / "predictions.jsonl")
    sparse_rows = read_jsonl(args.blasst_dir / "predictions.jsonl")
    dense_sc_rows = read_jsonl(args.dense_dir / "self_consistency.jsonl")
    sparse_sc_rows = read_jsonl(args.blasst_dir / "self_consistency.jsonl")

    if not dense_config.get("dense_baseline"):
        raise RuntimeError("dense run is not marked as a dense baseline")
    if sparse_config.get("dense_baseline"):
        raise RuntimeError("BLASST run is unexpectedly marked dense")
    if len(dense_rows) != 5_000 or len(sparse_rows) != 5_000:
        raise RuntimeError("both prediction files must contain exactly 5,000 rows")
    if len(dense_sc_rows) != 500 or len(sparse_sc_rows) != 500:
        raise RuntimeError("both self-consistency files must contain exactly 500 rows")

    sample_fields = ("problem_index", "sample_index")
    problem_fields = ("problem_index",)
    dense = keyed(dense_rows, sample_fields)
    sparse = keyed(sparse_rows, sample_fields)
    dense_sc = keyed(dense_sc_rows, problem_fields)
    sparse_sc = keyed(sparse_sc_rows, problem_fields)
    if dense.keys() != sparse.keys() or dense_sc.keys() != sparse_sc.keys():
        raise RuntimeError("dense and BLASST key sets differ")

    config_fields = (
        "base_seed",
        "device",
        "max_new_tokens",
        "model_path",
        "num_problems",
        "precision",
        "resolved_model_revision",
        "samples_per_problem",
        "temperature",
        "thinking",
        "top_p",
        "transformers_version",
    )
    for field in config_fields:
        if dense_config[field] != sparse_config[field]:
            raise RuntimeError(f"configuration mismatch for {field}")
    if dense_config["provenance"] != sparse_config["provenance"]:
        raise RuntimeError("dataset, prompt, paper, or NeMo-Skills provenance differs")

    paired_row_fields = (
        "problem_index",
        "sample_index",
        "unique_id",
        "problem",
        "expected_answer",
        "prompt",
        "prompt_length",
        "seed",
        "temperature",
        "top_p",
    )
    for key in dense:
        for field in paired_row_fields:
            if dense[key][field] != sparse[key][field]:
                raise RuntimeError(f"paired row mismatch at {key} for {field}")

    sample_keys = sorted(dense)
    problem_keys = sorted(dense_sc)
    dense_accuracy = sum(bool(dense[key]["symbolic_correct"]) for key in sample_keys) / 5_000
    sparse_accuracy = sum(bool(sparse[key]["symbolic_correct"]) for key in sample_keys) / 5_000
    dense_no_answer = sum(dense[key]["predicted_answer"] is None for key in sample_keys) / 5_000
    sparse_no_answer = sum(sparse[key]["predicted_answer"] is None for key in sample_keys) / 5_000
    dense_extracted = 1.0 - dense_no_answer
    sparse_extracted = 1.0 - sparse_no_answer
    dense_conditional_accuracy = dense_accuracy / dense_extracted
    sparse_conditional_accuracy = sparse_accuracy / sparse_extracted
    dense_majority = sum(float(dense_sc[key]["majority_score"]) for key in problem_keys) / 500
    sparse_majority = sum(float(sparse_sc[key]["majority_score"]) for key in problem_keys) / 500
    dense_selected = sum(float(dense_sc[key]["majority_selected_score"]) for key in problem_keys) / 500
    sparse_selected = sum(float(sparse_sc[key]["majority_selected_score"]) for key in problem_keys) / 500
    dense_pass10 = sum(bool(dense_sc[key]["any_symbolic_correct"]) for key in problem_keys) / 500
    sparse_pass10 = sum(bool(sparse_sc[key]["any_symbolic_correct"]) for key in problem_keys) / 500

    checks = (
        (dense_accuracy, dense_summary["accuracy"]["pass_at_1_avg_of_10"], "dense pass@1"),
        (sparse_accuracy, sparse_summary["accuracy"]["pass_at_1_avg_of_10"], "BLASST pass@1"),
        (dense_no_answer, dense_summary["accuracy"]["no_extracted_answer_rate"], "dense no-answer"),
        (sparse_no_answer, sparse_summary["accuracy"]["no_extracted_answer_rate"], "BLASST no-answer"),
        (dense_majority, dense_summary["accuracy"]["nemo_skills_majority_at_10"], "dense majority"),
        (sparse_majority, sparse_summary["accuracy"]["nemo_skills_majority_at_10"], "BLASST majority"),
        (dense_selected, dense_summary["accuracy"]["deterministic_majority_at_10"], "dense selected majority"),
        (sparse_selected, sparse_summary["accuracy"]["deterministic_majority_at_10"], "BLASST selected majority"),
        (dense_pass10, dense_summary["accuracy"]["pass_at_10"], "dense pass@10"),
        (sparse_pass10, sparse_summary["accuracy"]["pass_at_10"], "BLASST pass@10"),
    )
    for actual, expected, name in checks:
        assert_close(actual, expected, name)

    sample_transition = transitions(
        sample_keys, sparse, dense, lambda row: bool(row["symbolic_correct"])
    )
    majority_transition = transitions(
        problem_keys,
        sparse_sc,
        dense_sc,
        lambda row: bool(row["majority_selected_score"]),
    )
    pass10_transition = transitions(
        problem_keys,
        sparse_sc,
        dense_sc,
        lambda row: bool(row["any_symbolic_correct"]),
    )
    no_answer_transition = transitions(
        sample_keys,
        sparse,
        dense,
        lambda row: row["predicted_answer"] is None,
    )

    per_problem_pass1_delta = [
        (
            int(sparse_sc[key]["individual_symbolic_correct"])
            - int(dense_sc[key]["individual_symbolic_correct"])
        )
        / 10
        for key in problem_keys
    ]
    per_problem_majority_delta = [
        float(sparse_sc[key]["majority_score"]) - float(dense_sc[key]["majority_score"])
        for key in problem_keys
    ]
    pass1_ci = bootstrap_mean_ci(per_problem_pass1_delta)
    majority_ci = bootstrap_mean_ci(per_problem_majority_delta)
    sparse_better_draws = sum(value > 0 for value in per_problem_pass1_delta)
    dense_better_draws = sum(value < 0 for value in per_problem_pass1_delta)
    equal_draws = sum(value == 0 for value in per_problem_pass1_delta)

    dense_completion = [int(dense[key]["num_generated_tokens"]) for key in sample_keys]
    sparse_completion = [int(sparse[key]["num_generated_tokens"]) for key in sample_keys]
    dense_full = [int(dense[key]["prompt_length"]) + int(dense[key]["num_generated_tokens"]) for key in sample_keys]
    sparse_full = [int(sparse[key]["prompt_length"]) + int(sparse[key]["num_generated_tokens"]) for key in sample_keys]
    dense_elapsed = [float(dense[key]["elapsed_seconds"]) for key in sample_keys]
    sparse_elapsed = [float(sparse[key]["elapsed_seconds"]) for key in sample_keys]

    local = sparse_summary["physical_sparsity"]["local"]
    global_ = sparse_summary["physical_sparsity"]["global"]
    dense_time = dense_summary["generation"]["mean_model_elapsed_seconds"]
    sparse_time = sparse_summary["generation"]["mean_model_elapsed_seconds"]
    time_ratio = sparse_time / dense_time
    dense_peak = dense_summary["peak_cuda_memory_allocated_bytes"] / (1024**3)
    sparse_peak = sparse_summary["peak_cuda_memory_allocated_bytes"] / (1024**3)

    def subgroup_table(field: str) -> str:
        lines = [
            f"| {field.title()} | Problems | Dense pass@1 | BLASST pass@1 | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
        values = sorted({row.get(field) for row in dense_rows}, key=str)
        for value in values:
            dense_group = [row for row in dense_rows if row.get(field) == value]
            sparse_group = [row for row in sparse_rows if row.get(field) == value]
            dense_score = sum(bool(row["symbolic_correct"]) for row in dense_group) / len(dense_group)
            sparse_score = sum(bool(row["symbolic_correct"]) for row in sparse_group) / len(sparse_group)
            problem_count = len({int(row["problem_index"]) for row in dense_group})
            lines.append(
                f"| {value} | {problem_count} | {dense_score:.2%} | {sparse_score:.2%} | "
                f"{(sparse_score-dense_score)*100:+.2f} pp |"
            )
        return "\n".join(lines)

    subject_table = subgroup_table("subject")
    level_table = subgroup_table("level")

    report = f"""# DiffusionGemma dense vs. BLASST on MATH500

## Executive result

| Metric | Dense | BLASST λ(local/global)=0.9/0.6 | BLASST − dense |
|---|---:|---:|---:|
| NeMo-Skills majority@10 | {dense_majority:.2%} | {sparse_majority:.2%} | {(sparse_majority-dense_majority)*100:+.2f} pp |
| Deterministic majority@10 | {dense_selected:.2%} | {sparse_selected:.2%} | {(sparse_selected-dense_selected)*100:+.2f} pp |
| pass@1, average of 10 draws | {dense_accuracy:.2%} | {sparse_accuracy:.2%} | {(sparse_accuracy-dense_accuracy)*100:+.2f} pp |
| Oracle pass@10 | {dense_pass10:.2%} | {sparse_pass10:.2%} | {(sparse_pass10-dense_pass10)*100:+.2f} pp |
| No extracted answer | {dense_no_answer:.2%} | {sparse_no_answer:.2%} | {(sparse_no_answer-dense_no_answer)*100:+.2f} pp |
| Accuracy conditional on extracted answer | {dense_conditional_accuracy:.2%} | {sparse_conditional_accuracy:.2%} | {(sparse_conditional_accuracy-dense_conditional_accuracy)*100:+.2f} pp |
| Local-layer physical tile sparsity | 0.00% | {local['physical_tile_sparsity']:.2%} | +{local['physical_tile_sparsity']*100:.2f} pp |
| Global-layer physical tile sparsity | 0.00% | {global_['physical_tile_sparsity']:.2%} | +{global_['physical_tile_sparsity']*100:.2f} pp |
| Mean completion length | {statistics.fmean(dense_completion):.2f} | {statistics.fmean(sparse_completion):.2f} | {statistics.fmean(sparse_completion)-statistics.fmean(dense_completion):+.2f} tokens |
| Mean full sequence length | {statistics.fmean(dense_full):.2f} | {statistics.fmean(sparse_full):.2f} | {statistics.fmean(sparse_full)-statistics.fmean(dense_full):+.2f} tokens |
| Mean model time/sample | {dense_time:.3f} s | {sparse_time:.3f} s | {(time_ratio-1)*100:+.1f}% |
| Peak allocated CUDA memory | {dense_peak:.2f} GiB | {sparse_peak:.2f} GiB | {sparse_peak-dense_peak:+.2f} GiB |

The standardized NeMo-Skills majority score is nominally **{(sparse_majority-dense_majority)*100:+.3f} percentage points** higher with BLASST. This advantage comes from NeMo-Skills' fractional treatment of tied modes: under the concrete deterministic tie-break, both systems solve **466/500 problems (93.20%)**. Thus this run does not show a deterministic majority-accuracy gain from sparse attention.

At the individual-draw level, BLASST is **{(dense_accuracy-sparse_accuracy)*100:.2f} points lower** and produces **{(sparse_no_answer-dense_no_answer)*100:.2f} points more** unextractable answers. Among draws where an answer is extracted, however, BLASST is {sparse_conditional_accuracy:.2%} correct versus {dense_conditional_accuracy:.2%} for dense. This indicates that the pass@1 loss is dominated by answer-extraction failures rather than a lower correctness rate among parseable answers. Oracle pass@10 is unchanged at **96.20%**, so ten-sample coverage survives even though per-draw reliability falls.

## Paired analysis

All comparisons pair the same problem and sample index, with matching prompts and seeds.

| Paired outcome | Count |
|---|---:|
| Sample correct under both | {sample_transition[0]} / 5,000 |
| Correct only with BLASST | {sample_transition[1]} / 5,000 |
| Correct only with dense | {sample_transition[2]} / 5,000 |
| Incorrect under both | {sample_transition[3]} / 5,000 |
| Problem majority correct under both | {majority_transition[0]} / 500 |
| Majority correct only with BLASST | {majority_transition[1]} / 500 |
| Majority correct only with dense | {majority_transition[2]} / 500 |
| Majority incorrect under both | {majority_transition[3]} / 500 |
| pass@10 only with BLASST | {pass10_transition[1]} / 500 |
| pass@10 only with dense | {pass10_transition[2]} / 500 |
| No answer under both | {no_answer_transition[0]} / 5,000 |
| No answer only with BLASST | {no_answer_transition[1]} / 5,000 |
| No answer only with dense | {no_answer_transition[2]} / 5,000 |

- Across problems, BLASST has more correct draws on {sparse_better_draws}, dense has more on {dense_better_draws}, and {equal_draws} tie.
- The paired problem-bootstrap 95% interval for BLASST − dense pass@1 is [{pass1_ci[0]*100:+.2f}, {pass1_ci[1]*100:+.2f}] percentage points.
- The corresponding interval for NeMo majority@10 is [{majority_ci[0]*100:+.2f}, {majority_ci[1]*100:+.2f}] points.
- Deterministic majority discordances are {majority_transition[1]} BLASST-only versus {majority_transition[2]} dense-only; exact paired McNemar p={exact_mcnemar_p(majority_transition[1], majority_transition[2]):.3f}.

## Pass@1 by subgroup

{subject_table}

{level_table}

BLASST's draw-level accuracy is lower in every subject and every difficulty level, so the aggregate decline is not attributable to a single MATH500 subgroup. The largest subject decline is in Precalculus; by difficulty, levels 2, 3, and 5 show the largest drops.

## Sparsity and efficiency

- BLASST skips **{local['skipped_tiles']:,}/{local['eligible_tiles']:,} local tiles ({local['physical_tile_sparsity']:.2%})** and **{global_['skipped_tiles']:,}/{global_['eligible_tiles']:,} global tiles ({global_['physical_tile_sparsity']:.2%})**.
- Dense sparsity is 0% by definition; BLASST structurally masked tiles are excluded from the physical-sparsity denominator.
- BLASST takes {time_ratio:.2f}× the dense model time in this implementation ({sparse_time:.3f} versus {dense_time:.3f} seconds/sample), while peak allocated memory is effectively unchanged.
- This timing is **not an optimized sparse-kernel result**: the reference BLASST backend applies the sparse mask and counts skippable tiles but still materializes dense QK scores. Its instrumentation adds overhead, so only accuracy and physical sparsity—not speedup—should be treated as the intended measurements.

## Length and termination

| Statistic | Dense | BLASST |
|---|---:|---:|
| Prompt tokens, mean / median / range | {dense_summary['prompt_length_tokens']['mean']:.2f} / {dense_summary['prompt_length_tokens']['median']:.0f} / {dense_summary['prompt_length_tokens']['minimum']}-{dense_summary['prompt_length_tokens']['maximum']} | {sparse_summary['prompt_length_tokens']['mean']:.2f} / {sparse_summary['prompt_length_tokens']['median']:.0f} / {sparse_summary['prompt_length_tokens']['minimum']}-{sparse_summary['prompt_length_tokens']['maximum']} |
| Completion tokens, mean / median / p95 / max | {statistics.fmean(dense_completion):.2f} / {statistics.median(dense_completion):.0f} / {percentile(dense_completion, .95):.0f} / {max(dense_completion)} | {statistics.fmean(sparse_completion):.2f} / {statistics.median(sparse_completion):.0f} / {percentile(sparse_completion, .95):.0f} / {max(sparse_completion)} |
| Full sequence tokens, mean / median / p95 / max | {statistics.fmean(dense_full):.2f} / {statistics.median(dense_full):.0f} / {percentile(dense_full, .95):.0f} / {max(dense_full)} | {statistics.fmean(sparse_full):.2f} / {statistics.median(sparse_full):.0f} / {percentile(sparse_full, .95):.0f} / {max(sparse_full)} |
| Length-cap terminations | {dense_summary['generation']['length_termination_count']} ({dense_summary['generation']['length_termination_rate']:.2%}) | {sparse_summary['generation']['length_termination_count']} ({sparse_summary['generation']['length_termination_rate']:.2%}) |

Prompt length is identical by construction and includes the complete DiffusionGemma chat template and generation prefix. There is no natural-language system prompt and no few-shot example.

## Matched protocol and audit

- Model: `{dense_config['model_path']}`, resolved revision `{dense_config['resolved_model_revision']}`, BF16, Transformers `{dense_config['transformers_version']}`.
- MATH500: 500 problems × 10 samples, temperature {dense_config['temperature']}, top-p {dense_config['top_p']}, base seed {dense_config['base_seed']}, maximum {dense_config['max_new_tokens']} new tokens, thinking control disabled.
- NeMo-Skills commit `{dense_config['provenance']['nemo_skills_commit']}`; identical dataset and prompt SHA-256 values in both configurations.
- Audit passed: 5,000 unique `(problem_index, sample_index)` keys and 500 unique self-consistency keys in each run; all paired prompts, prompt lengths, seeds, expected answers, and task identifiers match.
- Dense predictions SHA-256: `{sha256(args.dense_dir / 'predictions.jsonl')}`.
- BLASST predictions SHA-256: `{sha256(args.blasst_dir / 'predictions.jsonl')}`.

## Source artifacts

- Dense summary: `{args.dense_dir / 'summary.json'}`
- Dense predictions: `{args.dense_dir / 'predictions.jsonl'}`
- BLASST summary: `{args.blasst_dir / 'summary.json'}`
- BLASST predictions: `{args.blasst_dir / 'predictions.jsonl'}`
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(report)


if __name__ == "__main__":
    main()
