#!/usr/bin/env python3
"""Evaluate dense-first bottom-k block-maximum KV pruning on MATH-500.

The model first materializes dense QK scores for every decoder attention call.
For every layer, head, and query row, valid KV tiles are ranked by their
maximum logit; the lowest requested fraction is masked before softmax is
renormalized.  This is a quality/oracle experiment, not a sparse-kernel speed
benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import GenerationRequest, create_adapter
from scripts.ruler.kv_pruning.quantile_kv_pruning_experiment import (
    OracleStats,
    OracleTilePruner,
)


DEFAULT_K_PERCENT = (0, 25, 50, 75, 90, 95)
PAPER_URL = "https://arxiv.org/pdf/2512.12087"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-gym-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument(
        "--revision", default="f7f5b7f5fa82ffc52addd066915886d497f5517b"
    )
    parser.add_argument("--k-percent", type=int, nargs="+", default=DEFAULT_K_PERCENT)
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--samples-per-problem", type=int, default=10)
    parser.add_argument("--subset-seed", type=int, default=20260816)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument(
        "--q-tile-size-metadata",
        type=int,
        default=128,
        help="paper-style query tile metadata; decisions are per query row",
    )
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def condition_name(k_percent: int) -> str:
    return "dense_eager" if k_percent == 0 else f"block_max_k{k_percent:02d}"


def validate_args(args: argparse.Namespace) -> None:
    if sorted(set(args.k_percent)) != sorted(args.k_percent):
        raise ValueError("k-percent values must be unique and sorted")
    if any(value < 0 or value >= 100 for value in args.k_percent):
        raise ValueError("k-percent values must be in [0, 100)")
    if list(args.k_percent) != list(DEFAULT_K_PERCENT):
        raise ValueError(f"this protocol requires k-percent={list(DEFAULT_K_PERCENT)}")
    if args.num_problems <= 0 or args.samples_per_problem <= 0:
        raise ValueError("num-problems and samples-per-problem must be positive")
    if args.num_problems != 50 or args.samples_per_problem != 10:
        raise ValueError("this protocol requires exactly 50 problems and 10 samples/problem")
    if args.max_new_tokens <= 0 or args.kv_tile_size <= 0:
        raise ValueError("max-new-tokens and kv-tile-size must be positive")
    if not 0.0 < args.temperature or not 0.0 < args.top_p <= 1.0:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")


def load_protocol(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    data_path = (
        args.nemo_gym_root
        / "benchmarks/math-500/data/math-500_benchmark.jsonl"
    )
    prompt_path = args.nemo_gym_root / "benchmarks/prompts/generic/math.yaml"
    all_problems = read_jsonl(data_path)
    if len(all_problems) != 500:
        raise RuntimeError(f"expected the 500-row MATH-500 test set, found {len(all_problems)}")

    # The official Gym YAML contains one literal user template. Keep the exact
    # text while avoiding a runtime dependency on the full Gym server stack.
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("user: |-") + 1
    except ValueError as error:
        raise RuntimeError("unexpected NeMo-Gym generic/math prompt format") from error
    prompt_lines = []
    for line in lines[start:]:
        if line and not line.startswith("  "):
            break
        prompt_lines.append(line[2:] if line.startswith("  ") else "")
    prompt_template = "\n".join(prompt_lines).rstrip()
    if "{question}" not in prompt_template or "\\boxed{{}}" not in prompt_template:
        raise RuntimeError("NeMo-Gym prompt lost its question/boxed-answer contract")

    selected_indices = sorted(
        random.Random(args.subset_seed).sample(range(len(all_problems)), args.num_problems)
    )
    problems = []
    for subset_index, dataset_index in enumerate(selected_indices):
        row = dict(all_problems[dataset_index])
        row["subset_index"] = subset_index
        row["dataset_index"] = dataset_index
        problems.append(row)
    provenance = {
        "blasst_paper_url": PAPER_URL,
        "nemo_gym_commit": git_revision(args.nemo_gym_root),
        "math500_path": str(data_path.resolve()),
        "math500_sha256": sha256_file(data_path),
        "gym_prompt_path": str(prompt_path.resolve()),
        "gym_prompt_sha256": sha256_file(prompt_path),
    }
    return problems, prompt_template, provenance


def write_subset(path: Path, problems: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in problems)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError("existing subset manifest differs from the deterministic sample")
    path.write_text(payload, encoding="utf-8")


def build_math_verifier():
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def gym_verify(verifier: Any, expected_answer: str, generation: str) -> tuple[float, str | None]:
    """Match NeMo-Gym math_with_judge's symbolic-only verification path."""
    expected = expected_answer.strip()
    if expected.startswith("\\(") and expected.endswith("\\)"):
        expected = expected[2:-2].strip()
    if expected.startswith("$") and expected.endswith("$") and len(expected) > 1:
        expected = expected[1:-1].strip()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            score, extracted = verifier([f"\\boxed{{{expected}}}"], [generation])
        answer: str | None = None
        if extracted is not None:
            gold, predictions = extracted
            for prediction in predictions:
                if any(grader.verify(target, prediction) for target in gold):
                    answer = str(prediction)
                    break
            if answer is None and predictions:
                answer = str(predictions[0])
        return float(score), answer
    except (Exception, TimeoutException):
        return 0.0, None


def aggregate_oracle_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    combined = OracleStats()
    found = False
    for row in rows:
        value = row.get("oracle_attention_stats")
        if value is not None:
            combined.add(OracleStats.from_summary(value))
            found = True
    return combined.summary() if found else OracleStats().summary()


def gym_metrics(rows: list[dict[str, Any]], samples_per_problem: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["subset_index"]), []).append(row)
    problem_rows = []
    for problem_index in sorted(grouped):
        rollouts = sorted(grouped[problem_index], key=lambda row: int(row["sample_index"]))
        if len(rollouts) != samples_per_problem:
            raise RuntimeError(
                f"subset problem {problem_index} has {len(rollouts)} rollouts, "
                f"expected {samples_per_problem}"
            )
        valid = [
            (row["extracted_answer"], float(row["library_reward"]))
            for row in rollouts
            if row["extracted_answer"] is not None
        ]
        if valid:
            counts = Counter(valid)
            maximum = counts.most_common(1)[0][1]
            tied = [(answer, score) for (answer, score), count in counts.items() if count == maximum]
            majority_score = statistics.fmean(score for _, score in tied)
            majority_answers = sorted(str(answer) for answer, _ in tied)
        else:
            maximum = 0
            tied = []
            majority_score = 0.0
            majority_answers = []
        problem_rows.append(
            {
                "subset_index": problem_index,
                "dataset_index": int(rollouts[0]["dataset_index"]),
                "unique_id": rollouts[0]["unique_id"],
                "subject": rollouts[0].get("subject"),
                "level": rollouts[0].get("level"),
                "majority_at_10": majority_score,
                "majority_answers": majority_answers,
                "majority_votes": maximum,
                "majority_tie_count": len(tied),
                "pass_at_10": float(any(row["library_reward"] > 0.5 for row in rollouts)),
                "pass_at_1_avg_of_10": statistics.fmean(
                    float(row["library_reward"]) for row in rollouts
                ),
            }
        )
    metrics = {
        "majority_at_10_symbolic_accuracy": statistics.fmean(
            row["majority_at_10"] for row in problem_rows
        ),
        "pass_at_1_avg_of_10_symbolic_accuracy": statistics.fmean(
            row["pass_at_1_avg_of_10"] for row in problem_rows
        ),
        "pass_at_10_symbolic_accuracy": statistics.fmean(
            row["pass_at_10"] for row in problem_rows
        ),
        "no_extracted_answer_rate": statistics.fmean(
            row["extracted_answer"] is None for row in rows
        ),
        "majority_tie_problem_rate": statistics.fmean(
            row["majority_tie_count"] > 1 for row in problem_rows
        ),
    }
    return metrics, problem_rows


def bootstrap_mean(values: np.ndarray, *, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(
    output_dir: Path,
    config: dict[str, Any],
    problems: list[dict[str, Any]],
    *,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    condition_summaries = []
    per_problem_by_k: dict[int, list[dict[str, Any]]] = {}
    for index, k_percent in enumerate(config["k_percent"]):
        name = condition_name(int(k_percent))
        rows = read_jsonl(output_dir / "conditions" / name / "predictions.jsonl")
        expected = len(problems) * int(config["samples_per_problem"])
        if len(rows) != expected:
            raise RuntimeError(f"{name} has {len(rows)} predictions; expected {expected}")
        metrics, problem_rows = gym_metrics(rows, int(config["samples_per_problem"]))
        per_problem_by_k[int(k_percent)] = problem_rows
        majority = np.asarray([row["majority_at_10"] for row in problem_rows])
        low, high = bootstrap_mean(
            majority, seed=int(config["subset_seed"]) + index, repeats=bootstrap_repeats
        )
        stats = aggregate_oracle_stats(rows)
        condition_summaries.append(
            {
                "condition": name,
                "k_percent": int(k_percent),
                **metrics,
                "majority_at_10_bootstrap_95_low": low,
                "majority_at_10_bootstrap_95_high": high,
                "oracle_attention_stats": stats,
                "mean_generation_seconds": statistics.fmean(
                    float(row["elapsed_seconds"]) for row in rows
                ),
                "mean_completion_tokens": statistics.fmean(
                    int(row["num_generated_tokens"]) for row in rows
                ),
            }
        )

    dense = condition_summaries[0]
    dense_values = np.asarray(
        [row["majority_at_10"] for row in per_problem_by_k[0]], dtype=np.float64
    )
    for index, row in enumerate(condition_summaries):
        k_percent = int(row["k_percent"])
        row["majority_at_10_delta_vs_dense"] = (
            row["majority_at_10_symbolic_accuracy"]
            - dense["majority_at_10_symbolic_accuracy"]
        )
        delta = np.asarray(
            [value["majority_at_10"] for value in per_problem_by_k[k_percent]],
            dtype=np.float64,
        ) - dense_values
        low, high = bootstrap_mean(
            delta,
            seed=int(config["subset_seed"]) + 100 + index,
            repeats=bootstrap_repeats,
        )
        row["paired_delta_bootstrap_95_low"] = low
        row["paired_delta_bootstrap_95_high"] = high
        condition_path = output_dir / "conditions" / row["condition"]
        atomic_json(condition_path / "summary.json", row)
        (condition_path / "self_consistency.jsonl").write_text(
            "".join(
                json.dumps(value, sort_keys=True) + "\n"
                for value in per_problem_by_k[k_percent]
            ),
            encoding="utf-8",
        )

    summary = {
        "schema_version": 1,
        "run_fingerprint": config["run_fingerprint"],
        "primary_metric": "NeMo-Gym majority@10 symbolic accuracy",
        "num_problems": len(problems),
        "samples_per_problem": config["samples_per_problem"],
        "conditions": condition_summaries,
    }
    atomic_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "accuracy_tradeoff.csv", condition_summaries)
    write_report(output_dir / "report.md", config, condition_summaries)
    plot_results(output_dir, condition_summaries)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "k_percent",
        "majority_at_10_symbolic_accuracy",
        "majority_at_10_delta_vs_dense",
        "majority_at_10_bootstrap_95_low",
        "majority_at_10_bootstrap_95_high",
        "paired_delta_bootstrap_95_low",
        "paired_delta_bootstrap_95_high",
        "pass_at_1_avg_of_10_symbolic_accuracy",
        "pass_at_10_symbolic_accuracy",
        "no_extracted_answer_rate",
        "actual_tile_drop_fraction",
        "local_actual_tile_drop_fraction",
        "global_actual_tile_drop_fraction",
        "discarded_dense_attention_mass_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row["oracle_attention_stats"]
            overall = stats["overall"]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "actual_tile_drop_fraction": overall["actual_tile_drop_fraction"],
                    "local_actual_tile_drop_fraction": stats["by_attention_type"]
                    .get("local", {})
                    .get("actual_tile_drop_fraction", 0.0),
                    "global_actual_tile_drop_fraction": stats["by_attention_type"]
                    .get("global", {})
                    .get("actual_tile_drop_fraction", 0.0),
                    "discarded_dense_attention_mass_mean": overall[
                        "discarded_dense_attention_mass_mean"
                    ],
                }
            )


def write_report(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    table = []
    for row in rows:
        stats = row["oracle_attention_stats"]
        overall = stats["overall"]["actual_tile_drop_fraction"]
        local = stats["by_attention_type"].get("local", {}).get(
            "actual_tile_drop_fraction", 0.0
        )
        global_ = stats["by_attention_type"].get("global", {}).get(
            "actual_tile_drop_fraction", 0.0
        )
        table.append(
            f"| {row['k_percent']}% | {overall:.2%} | {local:.2%} | {global_:.2%} | "
            f"{row['majority_at_10_symbolic_accuracy']:.2%} | "
            f"{row['majority_at_10_delta_vs_dense']:+.2%} | "
            f"{row['pass_at_1_avg_of_10_symbolic_accuracy']:.2%} | "
            f"{row['pass_at_10_symbolic_accuracy']:.2%} |"
        )
    text = f"""# DiffusionGemma block-maximum tile dropping on MATH-500

## Results

| Requested drop k | Actual tile drop | Local | Global | Majority@10 | Delta vs dense | pass@1 avg-of-10 | pass@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Protocol

- Model: `{config['model_path']}` at revision `{config['revision']}` in BF16.
- Data: deterministic random sample of 50 problems from NeMo-Gym MATH-500 (seed `{config['subset_seed']}`); the same problems and rollout seeds are paired across all k values.
- Sampling: 10 rollouts/problem, temperature `{config['temperature']}`, top-p `{config['top_p']}`, maximum `{config['max_new_tokens']}` generated tokens, thinking control disabled.
- Prompt and symbolic verifier: NeMo-Gym `generic/math` boxed-answer prompt and `math_with_judge`'s symbolic-only `math-verify` path (`should_use_judge: false`).
- Primary metric: NeMo-Gym majority@10/self-consistency symbolic accuracy. pass@10 is also included as an oracle upper bound and is not the primary score.
- Pruning: for every DiffusionGemma decoder-canvas attention layer/head/query row, compute dense QK, rank valid `{config['kv_tile_size']}`-token KV tiles by block maximum, drop `floor(N*k+0.5)` (capped at `N-1`), and renormalize softmax over retained tiles.
- The Q tile size is recorded as `{config['q_tile_size_metadata']}` for paper comparability, but the requested policy makes a separate decision for each query row. Local and global layers are both pruned.
- This is a dense-first oracle implementation: sparsity/accuracy are valid, but elapsed time is not a sparse-kernel speed measurement.
"""
    path.write_text(text, encoding="utf-8")


def plot_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    k = np.asarray([row["k_percent"] for row in rows], dtype=float)
    majority = 100 * np.asarray(
        [row["majority_at_10_symbolic_accuracy"] for row in rows]
    )
    low = 100 * np.asarray([row["majority_at_10_bootstrap_95_low"] for row in rows])
    high = 100 * np.asarray([row["majority_at_10_bootstrap_95_high"] for row in rows])
    pass1 = 100 * np.asarray(
        [row["pass_at_1_avg_of_10_symbolic_accuracy"] for row in rows]
    )
    actual = 100 * np.asarray(
        [row["oracle_attention_stats"]["overall"]["actual_tile_drop_fraction"] for row in rows]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.errorbar(
        k,
        majority,
        yerr=np.vstack((majority - low, high - majority)),
        marker="o",
        linewidth=2.2,
        capsize=3,
        label="Majority@10 (95% bootstrap CI)",
        color="#1864AB",
    )
    ax.plot(k, pass1, marker="s", linewidth=1.8, label="pass@1 (avg. of 10)", color="#E67700")
    for x, y, value in zip(k, majority, actual):
        ax.annotate(f"{value:.0f}% actual", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Requested lowest-ranked KV tiles dropped, k (%)")
    ax.set_ylabel("MATH-500 accuracy (%)")
    ax.set_xticks(k)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    ax.set_title("DiffusionGemma: block-maximum tile dropping (50 problems)")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"accuracy_vs_k.{suffix}", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    problems, prompt_template, provenance = load_protocol(args)
    write_subset(output_dir / "math500_subset.jsonl", problems)

    base_config = {
        "schema_version": 1,
        "experiment": "dense-first per-query bottom-k KV-tile pruning by block maximum",
        "model_path": args.model_path,
        "revision": args.revision,
        "device": args.device,
        "precision": args.precision,
        "k_percent": list(args.k_percent),
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "subset_seed": args.subset_seed,
        "base_seed": args.base_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "kv_tile_size": args.kv_tile_size,
        "q_tile_size_metadata": args.q_tile_size_metadata,
        "thinking": args.thinking,
        "ranking_scope": "independent valid KV tiles for every layer/head/query row",
        "rounding": "floor(N*k + 0.5), capped at N-1",
        "attention_scope": "DiffusionGemma decoder-canvas denoising attention; local and global",
        "oracle_dense_qk_first": True,
        "gym_symbolic_only": True,
        "provenance": provenance,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    fingerprint = stable_hash(base_config)
    config = {**base_config, "run_fingerprint": fingerprint}
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != fingerprint:
            raise RuntimeError("output directory belongs to a different run fingerprint")
        config = prior
    else:
        atomic_json(config_path, config)

    if args.report_only:
        summary = report(
            output_dir,
            config,
            problems,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    verifier = build_math_verifier()
    adapter = create_adapter(
        "diffusion_gemma",
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    config.update(adapter.runtime_metadata())
    atomic_json(config_path, config)

    blasst_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=0.003,
        q_tile_size=args.q_tile_size_metadata,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=False,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
        apply_blasst_mask=False,
    )
    binding = install_blasst(
        adapter.model,
        blasst_config,
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
    expected_per_condition = len(problems) * args.samples_per_problem
    try:
        for condition_index, k_percent in enumerate(args.k_percent, start=1):
            name = condition_name(k_percent)
            condition_dir = output_dir / "conditions" / name
            predictions_path = condition_dir / "predictions.jsonl"
            existing = read_jsonl(predictions_path)
            completed: dict[tuple[int, int], dict[str, Any]] = {}
            for row in existing:
                if row.get("run_fingerprint") != fingerprint:
                    raise RuntimeError(f"{name} contains a mismatched run fingerprint")
                key = (int(row["subset_index"]), int(row["sample_index"]))
                if key in completed:
                    raise RuntimeError(f"duplicate {name} prediction key: {key}")
                completed[key] = row
            if len(completed) == expected_per_condition:
                print(f"resume {condition_index}/{len(args.k_percent)}: {name} complete", flush=True)
                continue

            pruner = OracleTilePruner(
                method="dense" if k_percent == 0 else "block_max",
                k_fraction=k_percent / 100.0,
                kv_tile_size=args.kv_tile_size,
            )
            binding.runtime.attention_override = pruner
            for problem in problems:
                subset_index = int(problem["subset_index"])
                dataset_index = int(problem["dataset_index"])
                prompt = prompt_template.format(question=problem["question"])
                for sample_index in range(args.samples_per_problem):
                    key = (subset_index, sample_index)
                    if key in completed:
                        continue
                    seed = args.base_seed + dataset_index * args.samples_per_problem + sample_index
                    pruner.stats = OracleStats()
                    pruner.set_sample_seed(seed)
                    binding.runtime.metadata_context = {
                        "benchmark": "math-500",
                        "example_id": f"{problem['unique_id']}:{sample_index}",
                        "inference_seed": seed,
                    }
                    started = time.time()
                    result = adapter.generate(
                        GenerationRequest(
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            block_size=256,
                            temperature=args.temperature,
                            seed=seed,
                            extra={"thinking": args.thinking, "top_p": args.top_p},
                        )
                    )
                    reward, extracted_answer = gym_verify(
                        verifier, problem["expected_answer"], result.text
                    )
                    row = {
                        "schema_version": 1,
                        "run_fingerprint": fingerprint,
                        "condition": name,
                        "k_percent": k_percent,
                        "subset_index": subset_index,
                        "dataset_index": dataset_index,
                        "sample_index": sample_index,
                        "unique_id": problem["unique_id"],
                        "subject": problem.get("subject"),
                        "level": problem.get("level"),
                        "question": problem["question"],
                        "expected_answer": problem["expected_answer"],
                        "prompt": prompt,
                        "seed": seed,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "generation": result.text,
                        "completion_tokens": result.completion_tokens,
                        "num_generated_tokens": len(result.completion_tokens),
                        "termination_reason": result.termination_reason,
                        "elapsed_seconds": result.elapsed_seconds,
                        "wall_start_time": started,
                        "wall_end_time": time.time(),
                        "model_evaluations": result.model_evaluations,
                        "generation_metadata": result.metadata,
                        "library_reward": reward,
                        "extracted_answer": extracted_answer,
                        "oracle_attention_stats": (
                            pruner.stats.summary() if k_percent > 0 else None
                        ),
                    }
                    append_jsonl(predictions_path, row)
                    completed[key] = row
                    print(
                        f"{name} {len(completed)}/{expected_per_condition} "
                        f"({condition_index}/{len(args.k_percent)}): "
                        f"problem={subset_index} sample={sample_index} "
                        f"reward={reward:.0f} tokens={len(result.completion_tokens)} "
                        f"seconds={result.elapsed_seconds:.2f}",
                        flush=True,
                    )
    finally:
        binding.runtime.attention_override = None
        binding.close()

    summary = report(
        output_dir,
        config,
        problems,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
