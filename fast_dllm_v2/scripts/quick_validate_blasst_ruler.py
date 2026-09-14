#!/usr/bin/env python3
"""Quick count-level reproduction of the legacy 8K RULER BLASST result."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from scripts import eval_blasst_ruler_legacy as legacy
from scripts.blasst_common import generate_one, load_model, set_seed
from scripts.sweep_blasst_controlled import _aggregate
from sparse_attention import Blasst2DConfig, Blasst2DStats, install_blasst_2d
from sparse_attention import BLASST_MASK_SEMANTICS, validate_blasst_output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--ruler-root", default="/tmp/nvidia-ruler")
    parser.add_argument(
        "--source-results", default="results/blasst/fast_dllm_v2/ruler"
    )
    parser.add_argument(
        "--output-dir", default="results/blasst/fast_dllm_v2/quick_validation_b32_n20"
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--blasst-lambda", type=float, default=0.003)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    source = Path(args.source_results).resolve()
    output = Path(args.output_dir).resolve()
    validate_blasst_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", {**vars(args), "blasst_mask_semantics": BLASST_MASK_SEMANTICS})

    samples = legacy._read_jsonl(
        source / "ruler_cache/ruler_accuracy_samples_8k.jsonl"
    )[: args.num_samples]
    block_samples = legacy._read_jsonl(
        source / "ruler_cache/ruler_block_samples_8k.jsonl"
    )[: args.num_samples]
    if [row["sample_id"] for row in samples] != [
        row["sample_id"] for row in block_samples
    ]:
        raise AssertionError("sparsity and accuracy subsets do not match")
    task_counts = Counter(str(row["task"]) for row in samples)
    if set(task_counts.values()) != {args.num_samples // len(legacy.PAPER_TASKS)}:
        raise AssertionError(f"sample subset is not balanced: {task_counts}")

    ruler_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.ruler_root,
        text=True,
    ).strip()
    if ruler_commit != legacy.RULER_COMMIT:
        raise AssertionError(
            f"RULER commit {ruler_commit} != pinned {legacy.RULER_COMMIT}"
        )
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
    mask_token_id = int(getattr(model.config, "mask_token_id", 151665))
    config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=args.blasst_lambda,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
        include_masked_kv_tiles_in_physical_stats=False,
    )
    runtime = install_blasst_2d(
        model,
        replace(config, enable_blasst_2d=False),
        mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ordinary_cache_queries_only=True,
    )

    # Reuse the archived deterministic dense continuations, but recompute every
    # attention score and BLASST decision for the selected samples.
    archived_states = legacy._load_state_cache(
        source / "ruler_cache/ruler_sweep_states.json"
    )
    selected_states = {
        str(row["sample_id"]): archived_states[str(row["sample_id"])]
        for row in block_samples
    }
    state_path = output / "controlled_states.json"
    legacy._save_state_cache(state_path, selected_states)
    sweep_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(block_samples, start=1):
        rows, _ = legacy._observe_sample(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            config=config,
            sample=sample,
            experiments=("block_size",),
            block_sizes=[args.block_size],
            ratios=list(legacy.MASK_RATIOS),
            lambdas=[args.blasst_lambda],
            state_cache=selected_states,
            state_cache_path=state_path,
            mask_token_id=mask_token_id,
        )
        sweep_rows.extend(rows)
        print(f"controlled sparsity {index}/{len(block_samples)}", flush=True)
    write_jsonl(output / "controlled_sparsity_per_state.jsonl", sweep_rows)
    controlled = _aggregate(sweep_rows, ())[0]

    scorers, scorer_hash = legacy._load_official_scorers(Path(args.ruler_root))
    accuracy_path = output / "accuracy_per_example.jsonl"
    accuracy_rows = (
        legacy._read_jsonl(accuracy_path) if accuracy_path.exists() else []
    )
    completed = {str(row["sample_id"]) for row in accuracy_rows}
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        task = str(sample["task"])
        generation_budget = legacy._task_generation_budget(
            task, Path(args.ruler_root), args.block_size
        )
        runtime.sweep_lambdas = ()
        runtime.config = replace(config, enable_blasst_2d=False)
        runtime.stats = Blasst2DStats(record_layers=False, record_heads=False)
        set_seed(int(sample["inference_seed"]))
        dense = generate_one(
            model,
            tokenizer,
            str(sample["prompt"]),
            block_size=args.block_size,
            max_new_tokens=generation_budget,
            threshold=args.threshold,
        )

        runtime.config = config
        runtime.stats = Blasst2DStats(record_layers=False, record_heads=False)
        set_seed(int(sample["inference_seed"]))
        sparse = generate_one(
            model,
            tokenizer,
            str(sample["prompt"]),
            block_size=args.block_size,
            max_new_tokens=generation_budget,
            threshold=args.threshold,
        )
        counts = runtime.stats.summary()
        query_lengths = sorted(
            {int(row["query_length"]) for row in runtime.stats.per_step.values()}
        )
        if query_lengths != [args.block_size]:
            raise AssertionError(f"unexpected sparse query lengths {query_lengths}")
        references = [str(value) for value in sample["outputs"]]
        scorer = scorers[str(sample["task_base"])]
        row = {
            "sample_id": sample_id,
            "task": task,
            "task_base": str(sample["task_base"]),
            "actual_prompt_length": int(sample["actual_prompt_length"]),
            "inference_seed": int(sample["inference_seed"]),
            "outputs": references,
            "generation_budget": generation_budget,
            "dense_prediction": str(dense["completion"]),
            "sparse_prediction": str(sparse["completion"]),
            "dense_accuracy": legacy._score_one(
                scorer, str(dense["completion"]), references
            ),
            "sparse_accuracy": legacy._score_one(
                scorer, str(sparse["completion"]), references
            ),
            "query_lengths": query_lengths,
            **counts,
        }
        legacy._append_jsonl(accuracy_path, [row])
        accuracy_rows.append(row)
        print(f"paired accuracy {index}/{len(samples)}: {sample_id}", flush=True)

    if len(accuracy_rows) != args.num_samples:
        raise AssertionError(f"expected {args.num_samples} accuracy rows")
    accuracy_counts = _aggregate(accuracy_rows, ())[0]
    per_task: dict[str, Any] = {}
    dense_weighted = 0.0
    sparse_weighted = 0.0
    for task in legacy.PAPER_TASKS:
        subset = [row for row in accuracy_rows if row["task"] == task]
        dense_score = legacy._score_subset(scorers, subset, "dense_prediction")
        sparse_score = legacy._score_subset(scorers, subset, "sparse_prediction")
        dense_weighted += dense_score * len(subset)
        sparse_weighted += sparse_score * len(subset)
        per_task[task] = {
            "samples": len(subset),
            "dense_accuracy": dense_score,
            "sparse_accuracy": sparse_score,
        }
    dense_accuracy = dense_weighted / len(accuracy_rows)
    sparse_accuracy = sparse_weighted / len(accuracy_rows)

    archived_ids = {str(row["sample_id"]) for row in block_samples}
    archived_raw = [
        row
        for row in legacy._read_jsonl(source / "ruler_sweep_raw.jsonl")
        if row["experiment"] == "block_size"
        and int(row["sweep_value"]) == args.block_size
        and float(row["lambda"]) == args.blasst_lambda
        and str(row["sample_id"]) in archived_ids
    ]
    archived_controlled = _aggregate(archived_raw, ())[0]
    result = {
        "blasst_mask_semantics": BLASST_MASK_SEMANTICS,
        "configuration": {
            "model_path": str(Path(args.model_path).resolve()),
            "context_length": 8192,
            "samples": args.num_samples,
            "tasks": dict(task_counts),
            "block_size": args.block_size,
            "q_tile_size": args.q_tile_size,
            "kv_tile_size": args.kv_tile_size,
            "blasst_lambda": args.blasst_lambda,
            "mask_ratios": list(legacy.MASK_RATIOS),
            "precision": args.precision,
            "include_fully_masked_or_empty_tiles": False,
            "empty_or_padding_note": (
                "single-sample contiguous ordinary KV caches contain no padded or "
                "empty cache entries; semantic-invalid tiles remain excluded"
            ),
        },
        "controlled_sparsity": controlled,
        "archived_same_20_sample_controlled_sparsity": archived_controlled,
        "controlled_count_reproduction": {
            "skipped_tiles_equal": controlled["skipped_tiles"]
            == archived_controlled["skipped_tiles"],
            "eligible_tiles_equal": controlled["eligible_tiles"]
            == archived_controlled["eligible_tiles"],
        },
        "accuracy_generation": {
            "dense_accuracy": dense_accuracy,
            "sparse_accuracy": sparse_accuracy,
            "accuracy_delta": sparse_accuracy - dense_accuracy,
            "physical_tile_sparsity": accuracy_counts[
                "physical_tile_sparsity"
            ],
            "counts": accuracy_counts,
            "per_task": per_task,
        },
        "provenance": {
            "ruler_commit": ruler_commit,
            "official_scorer_sha256": scorer_hash,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "wall_seconds": time.monotonic() - started,
        },
    }
    write_json(output / "summary.json", result)
    (output / "report.md").write_text(
        "# Fast-dLLM v2 BLASST quick validation\n\n"
        f"- Controlled physical sparsity: {controlled['physical_tile_sparsity']:.4%}\n"
        f"- Archived same-20 prediction: {archived_controlled['physical_tile_sparsity']:.4%}\n"
        f"- Dense RULER accuracy: {dense_accuracy:.2%}\n"
        f"- Sparse RULER accuracy: {sparse_accuracy:.2%}\n"
        f"- Accuracy delta: {sparse_accuracy - dense_accuracy:+.2%}\n"
        f"- Generation-path physical sparsity: {accuracy_counts['physical_tile_sparsity']:.4%}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
