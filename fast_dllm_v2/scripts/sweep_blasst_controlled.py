#!/usr/bin/env python3
"""Two controlled 2D-BLASST sparsity sweeps with no sub-block path."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import load_model, set_seed  # noqa: E402
from sparse_attention import Blasst2DConfig, install_blasst_2d  # noqa: E402


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    "master/data/HumanEval.jsonl.gz"
)
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"

COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--gsm8k-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument(
        "--extension-corpus",
        default=str(V2_ROOT / "data/alpaca/test/test_252.json"),
    )
    parser.add_argument(
        "--contexts",
        default="512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--block-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--block-sweep-context", type=int, default=8192)
    parser.add_argument(
        "--lambdas",
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,0.5",
    )
    parser.add_argument("--mask-ratios", default="0.90,0.70,0.50,0.30,0.15")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/blasst/fast_dllm_v2/controlled_sweeps"),
    )
    return parser.parse_args()


def _values(text: str, cast) -> list[Any]:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _download_or_validate(
    requested_path: str,
    default_path: Path,
    url: str,
    expected_sha256: str,
) -> tuple[Path, str]:
    path = Path(requested_path) if requested_path else default_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"dataset checksum mismatch for {path}: {digest} != {expected_sha256}"
        )
    return path, digest


def _apply_chat(tokenizer, content: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return content


def _build_manifest(
    tokenizer,
    output_dir: Path,
    gsm8k_path: str,
    humaneval_path: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_dir = output_dir / "source_data"
    gsm_path, gsm_digest = _download_or_validate(
        gsm8k_path,
        source_dir / "gsm8k_test.jsonl",
        GSM8K_URL,
        GSM8K_SHA256,
    )
    human_path, human_digest = _download_or_validate(
        humaneval_path,
        source_dir / "HumanEval.jsonl.gz",
        HUMANEVAL_URL,
        HUMANEVAL_SHA256,
    )
    gsm_all = [
        json.loads(line)
        for line in gsm_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with gzip.open(human_path, "rt", encoding="utf-8") as handle:
        human_all = [json.loads(line) for line in handle if line.strip()]

    gsm_indices = sorted(random.Random(seed).sample(range(len(gsm_all)), 8))
    human_indices = sorted(random.Random(seed + 1).sample(range(len(human_all)), 8))
    examples: list[dict[str, Any]] = []
    for local_index, dataset_index in enumerate(gsm_indices):
        item = gsm_all[dataset_index]
        task_content = (
            f"Question: {item['question']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        examples.append(
            {
                "benchmark": "gsm8k",
                "example_id": f"gsm8k_test_{dataset_index}",
                "dataset_index": dataset_index,
                "seed": seed + local_index,
                "task_prompt": _apply_chat(tokenizer, task_content),
                "sparsity_prompt": item["question"],
                "label": {
                    "answer": item["answer"],
                    "final_answer": item["answer"].split("####")[-1].strip(),
                },
            }
        )
    for local_index, dataset_index in enumerate(human_indices):
        item = human_all[dataset_index]
        task_content = (
            "Complete the following Python function. Return only valid Python code "
            "containing the complete function, without Markdown fences.\n\n"
            f"{item['prompt']}"
        )
        examples.append(
            {
                "benchmark": "humaneval",
                "example_id": item["task_id"],
                "dataset_index": dataset_index,
                "seed": seed + 100 + local_index,
                "task_prompt": _apply_chat(tokenizer, task_content),
                "sparsity_prompt": item["prompt"],
                "label": {
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                },
            }
        )
    manifest = {
        "version": 1,
        "selection_seed": seed,
        "selection": {
            "gsm8k": "8 uniform samples without replacement using seed",
            "humaneval": "8 uniform samples without replacement using seed+1",
        },
        "dataset_sources": {
            "gsm8k": {
                "url": GSM8K_URL,
                "path": str(gsm_path),
                "sha256": gsm_digest,
                "total_examples": len(gsm_all),
            },
            "humaneval": {
                "url": HUMANEVAL_URL,
                "path": str(human_path),
                "sha256": human_digest,
                "total_examples": len(human_all),
            },
        },
        "examples": examples,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return examples, manifest


def _extension_tokens(tokenizer, path: str) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    instances = payload.get("instances", payload)
    text = "\n\n".join(
        f"{item['input']}\n{item['output']}" for item in instances
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("extension corpus tokenized to zero tokens")
    return ids


def _cycle(values: list[int], count: int, offset: int = 0) -> list[int]:
    if not values:
        raise ValueError("cannot cycle an empty token sequence")
    return [values[(offset + index) % len(values)] for index in range(count)]


def _sample_state(
    tokenizer,
    example: Mapping[str, Any],
    extension: list[int],
    max_prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prompt_ids = tokenizer(
        str(example["sparsity_prompt"]),
        add_special_tokens=False,
    )["input_ids"]
    if not prompt_ids:
        prompt_ids = extension[:1]
    dataset_index = int(example["dataset_index"])
    rotated_extension = _cycle(extension, max_prefix, dataset_index * 997)
    prefix_ids = (prompt_ids + rotated_extension)[:max_prefix]
    if len(prefix_ids) < max_prefix:
        prefix_ids += _cycle(
            extension,
            max_prefix - len(prefix_ids),
            dataset_index * 997 + len(prefix_ids),
        )
    if example["benchmark"] == "gsm8k":
        response_text = str(example["label"]["answer"])
    else:
        response_text = (
            str(example["sparsity_prompt"])
            + str(example["label"]["canonical_solution"])
        )
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]
    current64 = _cycle(response_ids or prompt_ids, 64, dataset_index * 31)
    construction = {
        "prompt_tokens_at_prefix_start": min(len(prompt_ids), max_prefix),
        "extension": (
            "prompt tokens followed by a deterministic dataset-index rotation "
            "of the local Alpaca test corpus"
        ),
        "current_state": (
            "64 deterministic response/reference tokens; each query uses its suffix"
        ),
        "padding_tokens": 0,
    }
    return (
        torch.tensor(prefix_ids, dtype=torch.long),
        torch.tensor(current64, dtype=torch.long),
        construction,
    )


class PrefixCacheView:
    def __init__(self, cache: Any, length: int) -> None:
        self.cache = cache
        self.length = int(length)

    def __len__(self) -> int:
        return len(self.cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.length

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.cache[layer_idx]
        return key[:, :, : self.length], value[:, :, : self.length]


def _masked_queries(
    clean64: torch.Tensor,
    query_length: int,
    ratios: list[float],
    mask_token_id: int,
    seed: int,
) -> list[tuple[int, float, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    order64 = torch.randperm(64, generator=generator)
    query_start = 64 - query_length
    clean = clean64[query_start:].clone()
    suffix_order = [
        position - query_start
        for position in order64.tolist()
        if position >= query_start
    ]
    states = []
    for step, ratio in enumerate(ratios):
        selected = suffix_order[: round(query_length * ratio)]
        noisy = clean.clone()
        if selected:
            noisy[torch.tensor(selected, dtype=torch.long)] = mask_token_id
        states.append((step, ratio, noisy))
    return states


def _with_ratios(counts: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(counts)
    eligible = int(row["eligible_tiles"])
    votes = int(row["valid_row_votes"])
    elements = int(row["valid_elements"])
    row_sparsity = (
        int(row["skippable_row_votes"]) / votes if votes else 0.0
    )
    physical = int(row["skipped_tiles"]) / eligible if eligible else 0.0
    row["row_vote_sparsity"] = row_sparsity
    row["physical_tile_sparsity"] = physical
    row["valid_element_sparsity"] = (
        int(row["skipped_valid_elements"]) / elements if elements else 0.0
    )
    row["physical_to_row_sparsity_ratio"] = (
        physical / row_sparsity if row_sparsity else 0.0
    )
    return row


def _aggregate(
    rows: Iterable[Mapping[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        target = grouped.setdefault(
            key,
            {name: row[name] for name in keys}
            | {name: 0 for name in COUNT_FIELDS},
        )
        for field in COUNT_FIELDS:
            target[field] += int(row[field])
    return [_with_ratios(row) for _, row in sorted(grouped.items())]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats_rows(runtime, lambdas: list[float]) -> list[dict[str, Any]]:
    rows = []
    for lambda_value in lambdas:
        stats = runtime.sweep_stats[lambda_value]
        if stats.per_layer or stats.per_head:
            raise AssertionError("memory-saving sweep unexpectedly recorded layer/head rows")
        for row in stats._rows(stats.per_step):
            rows.append({"lambda": lambda_value, **row})
    return rows


def _rename_axis(
    rows: list[dict[str, Any]],
    axis_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            axis_name: int(row["sweep_value"]),
            **{
                key: value
                for key, value in row.items()
                if key not in ("sweep_value", "experiment")
            },
        }
        for row in rows
    ]


def _check(
    raw_rows: list[dict[str, Any]],
    context_summary: list[dict[str, Any]],
    block_summary: list[dict[str, Any]],
    contexts: list[int],
    block_sizes: list[int],
    lambdas: list[float],
    dense_regression: Mapping[str, Any],
    finite_outputs: int,
) -> dict[str, Any]:
    violations: list[str] = []
    for row in raw_rows + context_summary + block_summary:
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"count identity failed: {row}")
        for metric in (
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
        ):
            if not math.isfinite(float(row[metric])):
                violations.append(f"non-finite {metric}: {row}")

    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    stratum_fields = (
        "experiment",
        "sweep_value",
        "benchmark",
        "example_id",
        "denoising_step",
        "mask_ratio",
        "masked_tokens",
        "query_length",
        "valid_kv_length",
    )
    for row in raw_rows:
        strata.setdefault(
            tuple(row[field] for field in stratum_fields), []
        ).append(row)
    for key, group in strata.items():
        ordered = sorted(group, key=lambda row: float(row["lambda"]))
        if [float(row["lambda"]) for row in ordered] != sorted(lambdas):
            violations.append(f"incomplete lambda grid at {key}")
            continue
        for lower, upper in zip(ordered, ordered[1:]):
            for field in ("skipped_tiles", "skippable_row_votes"):
                if int(upper[field]) < int(lower[field]):
                    violations.append(f"non-monotonic {field} at {key}")
            for field in (
                "eligible_tiles",
                "valid_row_votes",
                "valid_elements",
                "structurally_masked_tiles",
            ):
                if int(upper[field]) != int(lower[field]):
                    violations.append(f"lambda-dependent {field} at {key}")

    for row in raw_rows:
        experiment = row["experiment"]
        sweep_value = int(row["sweep_value"])
        query_length = int(row["query_length"])
        valid_length = int(row["valid_kv_length"])
        if experiment == "context_length":
            if query_length != 16:
                violations.append(f"residual non-16 context query: {row}")
            if valid_length != sweep_value:
                violations.append(f"context valid-KV mismatch: {row}")
        elif experiment == "block_size":
            if query_length != sweep_value:
                violations.append(f"block/query mismatch: {row}")
            if valid_length != 8192:
                violations.append(f"block-sweep valid-KV mismatch: {row}")
        else:
            violations.append(f"unknown experiment {experiment}")
        if query_length == 8 and not (
            experiment == "block_size" and sweep_value == 8
        ):
            violations.append(f"residual sub-block q=8 call: {row}")

    # At context=8192/block=16 both experiments deliberately use the same
    # prefix view, query tokens, masks, and dense scores.
    for lambda_value in lambdas:
        left = next(
            row
            for row in context_summary
            if int(row["context_length"]) == 8192
            and float(row["lambda"]) == lambda_value
        )
        right = next(
            row
            for row in block_summary
            if int(row["block_size"]) == 16
            and float(row["lambda"]) == lambda_value
        )
        for field in COUNT_FIELDS:
            if int(left[field]) != int(right[field]):
                violations.append(
                    f"cross-experiment state mismatch λ={lambda_value}, {field}"
                )

    observed_contexts = sorted(
        {int(row["context_length"]) for row in context_summary}
    )
    observed_blocks = sorted({int(row["block_size"]) for row in block_summary})
    if observed_contexts != contexts:
        violations.append(f"context grid mismatch {observed_contexts}")
    if observed_blocks != block_sizes:
        violations.append(f"block grid mismatch {observed_blocks}")
    if not dense_regression["exact_match"] or not dense_regression["all_finite"]:
        violations.append("dense-disabled regression failed")
    result = {
        "passed": not violations,
        "violations": violations,
        "raw_strata_rows": len(raw_rows),
        "exact_valid_kv_recorded_per_attention_call": True,
        "finite_model_outputs_checked": finite_outputs,
        "dense_disabled_regression": dict(dense_regression),
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "query_length_gate": (
            "context sweep q=16; block sweep q equals configured block size; "
            "q=8 occurs only at block_size=8"
        ),
        "aggregation": "ratios of summed integer counts, never means of percentages",
        "structural_masking_excluded": True,
        "monotonicity_checked_within_identical_score_strata": True,
        "cross_experiment_8192_block16_identical": True,
    }
    if violations:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _plot_curves(
    rows: list[dict[str, Any]],
    x_field: str,
    metric: str,
    lambdas: list[float],
    title: str,
    x_label: str,
    output_base: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/blasst-matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(8.3, 5.3))
    for lambda_value in lambdas:
        subset = sorted(
            [
                row
                for row in rows
                if float(row["lambda"]) == lambda_value
            ],
            key=lambda row: int(row[x_field]),
        )
        axis.plot(
            [int(row[x_field]) for row in subset],
            [float(row[metric]) for row in subset],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"λ={lambda_value:g}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        sorted({int(row[x_field]) for row in rows}),
        [str(value) for value in sorted({int(row[x_field]) for row in rows})],
    )
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(x_label)
    axis.set_ylabel("Sparsity")
    axis.set_title(title)
    axis.grid(True, which="major", alpha=0.25)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "repository_commit": commit,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }


def _report(
    output_dir: Path,
    context_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    context_benchmark: list[dict[str, Any]],
    block_benchmark: list[dict[str, Any]],
    lambdas: list[float],
    elapsed: float,
) -> None:
    context_lookup = {
        (int(row["context_length"]), float(row["lambda"])): row
        for row in context_rows
    }
    block_lookup = {
        (int(row["block_size"]), float(row["lambda"])): row
        for row in block_rows
    }
    context_changes = []
    block_changes = []
    severe = []
    for value in lambdas:
        low = context_lookup[(512, value)]
        high = context_lookup[(16384, value)]
        context_changes.append(
            (
                value,
                float(high["row_vote_sparsity"])
                - float(low["row_vote_sparsity"]),
                float(high["physical_tile_sparsity"])
                - float(low["physical_tile_sparsity"]),
            )
        )
        one = block_lookup[(1, value)]
        sixty_four = block_lookup[(64, value)]
        block_changes.append(
            (
                value,
                float(sixty_four["physical_tile_sparsity"])
                - float(one["physical_tile_sparsity"]),
            )
        )
    for row in block_rows:
        gap = float(row["row_vote_sparsity"]) - float(
            row["physical_tile_sparsity"]
        )
        if gap >= 0.20:
            severe.append((int(row["block_size"]), float(row["lambda"]), gap))

    benchmark_lines = []
    for benchmark in ("gsm8k", "humaneval"):
        for value in (lambdas[0], lambdas[-1]):
            low = next(
                row
                for row in context_benchmark
                if row["benchmark"] == benchmark
                and int(row["context_length"]) == 512
                and float(row["lambda"]) == value
            )
            high = next(
                row
                for row in context_benchmark
                if row["benchmark"] == benchmark
                and int(row["context_length"]) == 16384
                and float(row["lambda"]) == value
            )
            benchmark_lines.append(
                f"{benchmark}, λ={value:g}: physical "
                f"{float(low['physical_tile_sparsity']):.1%}→"
                f"{float(high['physical_tile_sparsity']):.1%}"
            )

    previous_path = (
        REPO_ROOT / "results/blasst/fast_dllm_v2/context_sweep/context_lambda_summary.csv"
    )
    proxy_differences = []
    for value in lambdas:
        mixed_proxy = (
            float(block_lookup[(8, value)]["physical_tile_sparsity"])
            + float(block_lookup[(32, value)]["physical_tile_sparsity"])
        ) / 2.0
        fixed16 = float(block_lookup[(16, value)]["physical_tile_sparsity"])
        proxy_differences.append((value, fixed16 - mixed_proxy))
    previous_text = (
        "The prior path mixed q=32 and residual q=8 calls. On the current matched "
        "states, an equal-count mean of the b=8 and b=32 physical counts differs "
        "from fixed b=16 by "
        + "; ".join(
            f"λ={value:g}: {difference:+.1%}"
            for value, difference in proxy_differences
        )
        + ". Thus the aggregate change is small on this controlled proxy, especially "
        "at larger λ, although disabling sub-blocks removes large q-specific ambiguity. "
        + (
            "The raw prior artifact is available, but its different examples and "
            "prefix construction prevent attributing its cross-run difference solely "
            "to sub-block removal."
            if previous_path.exists()
            else "The raw prior artifact was unavailable for a cross-run comparison."
        )
    )
    lines = [
        "# Controlled 2D-BLASST context and diffusion-block sweeps",
        "",
        "This is dense-QK observation. It measures BLASST decisions and theoretical "
        "skipped softmax/PV work; it does not measure or claim runtime speedup.",
        "",
        "## Controls",
        "",
        "- Fixed manifest: 8 GSM8K + 8 HumanEval examples, five deterministic "
        "nested mask states per example.",
        "- Sub-block splitting and dual block cache are disabled. Ordinary read-only "
        "KV prefix views are retained.",
        "- Prefix K/V states are strict-causal and shared across both experiments. "
        "This makes arbitrary prefix views safe and isolates current diffusion-block "
        "unanimity from prefix re-encoding.",
        "- Q/KV physical attention tiles remain 128/64. Diffusion block size is a "
        "separate variable.",
        f"- Sweep wall time through export: {elapsed:.1f} seconds.",
        "",
        "## Context-length sweep",
        "",
        "| Context | λ | Eligible | Skipped | Row | Physical | Valid element | Physical/row |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in context_rows:
        lines.append(
            f"| {int(row['context_length'])} | {float(row['lambda']):g} | "
            f"{int(row['eligible_tiles'])} | {int(row['skipped_tiles'])} | "
            f"{float(row['row_vote_sparsity']):.4f} | "
            f"{float(row['physical_tile_sparsity']):.4f} | "
            f"{float(row['valid_element_sparsity']):.4f} | "
            f"{float(row['physical_to_row_sparsity_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diffusion-block-size sweep at 8192 valid KV tokens",
            "",
            "| Block/Q | λ | Eligible | Skipped | Row | Physical | Valid element | Physical/row |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in block_rows:
        lines.append(
            f"| {int(row['block_size'])} | {float(row['lambda']):g} | "
            f"{int(row['eligible_tiles'])} | {int(row['skipped_tiles'])} | "
            f"{float(row['row_vote_sparsity']):.4f} | "
            f"{float(row['physical_tile_sparsity']):.4f} | "
            f"{float(row['valid_element_sparsity']):.4f} | "
            f"{float(row['physical_to_row_sparsity_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. **Context effect.** Across λ, 512→16384 changes "
            + "; ".join(
                f"λ={value:g}: row {row_delta:+.1%}, physical {physical_delta:+.1%}"
                for value, row_delta, physical_delta in context_changes
            )
            + ".",
            "",
            "2. **Block-size unanimity effect.** At fixed 8192 context, block "
            "1→64 changes physical sparsity by "
            + "; ".join(
                f"λ={value:g}: {delta:+.1%}" for value, delta in block_changes
            )
            + ". Larger blocks require more query rows to agree.",
            "",
            "3. **Severe row/physical gaps.** Using an absolute ≥20 percentage-point "
            "gap, observed cells are "
            + (
                ", ".join(
                    f"b={block}, λ={value:g} ({gap:.1%})"
                    for block, value, gap in severe
                )
                if severe
                else "none"
            )
            + ".",
            "",
            "4. **Benchmark consistency.** "
            + "; ".join(benchmark_lines)
            + ". Full per-benchmark counts are in the CSVs.",
            "",
            "5. **Change after disabling sub-blocks.** " + previous_text,
            "",
            "Task accuracy sanity checks are produced by "
            "`eval_blasst_mixed_tasks.py` and appended below after that paired run.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    contexts = _values(args.contexts, int)
    block_sizes = _values(args.block_sizes, int)
    lambdas = _values(args.lambdas, float)
    ratios = _values(args.mask_ratios, float)
    if args.block_size != 16:
        raise ValueError("experiment 1 requires block_size=16")
    if contexts != [512, 1024, 2048, 4096, 8192, 16384]:
        raise ValueError("context grid must match the controlled specification")
    if block_sizes != [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("block-size grid must match the controlled specification")
    if args.block_sweep_context != 8192:
        raise ValueError("block-size sweep context must be 8192")
    if any(not 0.0 < value <= 1.0 for value in lambdas):
        raise ValueError("all lambdas must be in (0, 1]")

    output_dir = Path(args.output_dir)
    from dllm.attention.blasst import BLASST_MASK_SEMANTICS, validate_blasst_output_directory
    validate_blasst_output_directory(output_dir)
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
    if len(examples) != 16:
        raise AssertionError("manifest must contain exactly 16 examples")
    extension = _extension_tokens(tokenizer, args.extension_corpus)
    mask_token_id = int(getattr(model.config, "mask_token_id", 151665))

    sweep_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=lambdas[0],
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
    )
    probe_ids = tokenizer(
        str(examples[0]["sparsity_prompt"]),
        return_tensors="pt",
    )["input_ids"][:, :16].to(model.device)
    if probe_ids.shape[1] < 16:
        probe_ids = torch.cat(
            (
                probe_ids,
                torch.full(
                    (1, 16 - probe_ids.shape[1]),
                    mask_token_id,
                    device=model.device,
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
    with torch.no_grad():
        dense_before = model.model(
            input_ids=probe_ids,
            use_cache=False,
            use_block_cache=False,
            block_size=16,
        ).last_hidden_state.detach()
    runtime = install_blasst_2d(
        model,
        replace(sweep_config, enable_blasst_2d=False),
        mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        dual_cache_only=False,
        ordinary_cache_queries_only=True,
        sweep_lambdas=lambdas,
    )
    with torch.no_grad():
        dense_after = model.model(
            input_ids=probe_ids,
            use_cache=False,
            use_block_cache=False,
            block_size=16,
        ).last_hidden_state.detach()
    dense_regression = {
        "exact_match": bool(torch.equal(dense_before, dense_after)),
        "all_finite": bool(
            torch.isfinite(dense_before).all()
            and torch.isfinite(dense_after).all()
        ),
        "max_abs_error": float((dense_before - dense_after).abs().max()),
    }
    if not dense_regression["exact_match"]:
        raise AssertionError(f"dense-disabled regression failed: {dense_regression}")

    max_prefix = max(contexts) - args.block_size
    finite_outputs = 0
    construction_by_example: dict[str, Any] = {}
    for example_index, example in enumerate(examples):
        prefix_ids, clean64, construction = _sample_state(
            tokenizer,
            example,
            extension,
            max_prefix,
        )
        construction_by_example[str(example["example_id"])] = construction
        prefix_ids = prefix_ids.to(model.device)
        clean64 = clean64.to(model.device)
        runtime.config = replace(sweep_config, enable_blasst_2d=False)
        with torch.no_grad():
            cache_output = model.model(
                input_ids=prefix_ids[None],
                use_cache=True,
                update_past_key_values=True,
                use_block_cache=False,
                block_size=1,
            )
        full_cache = cache_output.past_key_values
        if full_cache.get_seq_length() != max_prefix:
            raise AssertionError("causal prefix cache length mismatch")
        del cache_output
        runtime.config = sweep_config

        states16 = _masked_queries(
            clean64,
            16,
            ratios,
            mask_token_id,
            int(example["seed"]),
        )
        for context in contexts:
            prefix = PrefixCacheView(full_cache, context - 16)
            for step, requested_ratio, noisy in states16:
                masked = int((noisy == mask_token_id).sum())
                metadata = {
                    "experiment": "context_length",
                    "sweep_value": context,
                    "benchmark": example["benchmark"],
                    "example_id": example["example_id"],
                    "denoising_step": step,
                    "mask_ratio": masked / 16,
                    "requested_mask_ratio": requested_ratio,
                    "masked_tokens": masked,
                }
                with torch.no_grad():
                    output = model.model(
                        input_ids=noisy[None],
                        past_key_values=prefix,
                        use_cache=True,
                        update_past_key_values=False,
                        use_block_cache=False,
                        block_size=16,
                        blasst_active_query_mask=torch.ones(
                            (1, 16), dtype=torch.bool, device=model.device
                        ),
                        blasst_metadata=metadata,
                    )
                if not torch.isfinite(output.last_hidden_state).all():
                    raise FloatingPointError(
                        f"non-finite context output {example['example_id']}"
                    )
                finite_outputs += 1
                del output

        for block_size in block_sizes:
            prefix = PrefixCacheView(
                full_cache,
                args.block_sweep_context - block_size,
            )
            states = _masked_queries(
                clean64,
                block_size,
                ratios,
                mask_token_id,
                int(example["seed"]),
            )
            for step, requested_ratio, noisy in states:
                masked = int((noisy == mask_token_id).sum())
                metadata = {
                    "experiment": "block_size",
                    "sweep_value": block_size,
                    "benchmark": example["benchmark"],
                    "example_id": example["example_id"],
                    "denoising_step": step,
                    "mask_ratio": masked / block_size,
                    "requested_mask_ratio": requested_ratio,
                    "masked_tokens": masked,
                }
                with torch.no_grad():
                    output = model.model(
                        input_ids=noisy[None],
                        past_key_values=prefix,
                        use_cache=True,
                        update_past_key_values=False,
                        use_block_cache=False,
                        block_size=block_size,
                        blasst_active_query_mask=torch.ones(
                            (1, block_size),
                            dtype=torch.bool,
                            device=model.device,
                        ),
                        blasst_metadata=metadata,
                    )
                if not torch.isfinite(output.last_hidden_state).all():
                    raise FloatingPointError(
                        f"non-finite block output {example['example_id']}"
                    )
                finite_outputs += 1
                del output
        del full_cache, prefix_ids, clean64
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"completed sparsity sample {example_index + 1}/16: "
            f"{example['example_id']}",
            flush=True,
        )

    raw_rows = _stats_rows(runtime, lambdas)
    context_raw = [
        row for row in raw_rows if row["experiment"] == "context_length"
    ]
    block_raw = [row for row in raw_rows if row["experiment"] == "block_size"]
    context_summary = _rename_axis(
        _aggregate(context_raw, ("sweep_value", "lambda")),
        "context_length",
    )
    block_summary = _rename_axis(
        _aggregate(block_raw, ("sweep_value", "lambda")),
        "block_size",
    )
    context_examples = _rename_axis(
        _aggregate(
            context_raw,
            ("sweep_value", "lambda", "benchmark", "example_id"),
        ),
        "context_length",
    )
    block_examples = _rename_axis(
        _aggregate(
            block_raw,
            ("sweep_value", "lambda", "benchmark", "example_id"),
        ),
        "block_size",
    )
    context_benchmark = _rename_axis(
        _aggregate(
            context_raw,
            ("sweep_value", "lambda", "benchmark"),
        ),
        "context_length",
    )
    block_benchmark = _rename_axis(
        _aggregate(
            block_raw,
            ("sweep_value", "lambda", "benchmark"),
        ),
        "block_size",
    )

    checks = _check(
        raw_rows,
        context_summary,
        block_summary,
        contexts,
        block_sizes,
        lambdas,
        dense_regression,
        finite_outputs,
    )
    _write_csv(output_dir / "context_length_sweep.csv", context_summary)
    _write_csv(output_dir / "block_size_sweep.csv", block_summary)
    _write_csv(
        output_dir / "context_length_per_example.csv",
        context_examples,
    )
    _write_csv(
        output_dir / "block_size_per_example.csv",
        block_examples,
    )
    _write_csv(
        output_dir / "context_length_per_benchmark.csv",
        context_benchmark,
    )
    _write_csv(
        output_dir / "block_size_per_benchmark.csv",
        block_benchmark,
    )
    (output_dir / "correctness_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plot_curves(
        context_summary,
        "context_length",
        "row_vote_sparsity",
        lambdas,
        "Row-vote sparsity vs valid KV context",
        "Actual valid KV context length",
        output_dir / "context_row_sparsity",
    )
    _plot_curves(
        context_summary,
        "context_length",
        "physical_tile_sparsity",
        lambdas,
        "Physical tile sparsity vs valid KV context",
        "Actual valid KV context length",
        output_dir / "context_physical_sparsity",
    )
    _plot_curves(
        block_summary,
        "block_size",
        "row_vote_sparsity",
        lambdas,
        "Row-vote sparsity vs diffusion block size",
        "Diffusion block size / effective query length",
        output_dir / "block_row_sparsity",
    )
    _plot_curves(
        block_summary,
        "block_size",
        "physical_tile_sparsity",
        lambdas,
        "Physical tile sparsity vs diffusion block size",
        "Diffusion block size / effective query length",
        output_dir / "block_physical_sparsity",
    )
    elapsed = time.monotonic() - started
    run_config = {
        "blasst_mask_semantics": BLASST_MASK_SEMANTICS,
        **vars(args),
        **_environment(),
        "contexts": contexts,
        "block_sizes": block_sizes,
        "lambdas": lambdas,
        "mask_ratios": ratios,
        "examples": 16,
        "examples_per_benchmark": 8,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "ordinary_kv_cache": True,
        "prefix_cache_semantics": (
            "strict-causal block_size=1 prefix K/V shared by all context and "
            "block-size points; read-only prefix views make shorter lengths exact"
        ),
        "context_extension": (
            "sample prompt followed by deterministic dataset-index rotation of "
            "local Alpaca test-corpus tokens; no padding"
        ),
        "current_state_control": (
            "one deterministic 64-token response/reference state per example; "
            "each block-size query is its suffix and uses shared nested mask ranks"
        ),
        "lambda_control": (
            "all eight lambdas threshold the same FP32 local/running maxima "
            "within every attention call"
        ),
        "aggregation": "integer count sums before division",
        "physical_attention_tiles": {
            "query": args.q_tile_size,
            "key_value": args.kv_tile_size,
        },
        "manifest_sha256": hashlib.sha256(
            (output_dir / "evaluation_manifest.json").read_bytes()
        ).hexdigest(),
        "sample_construction": construction_by_example,
        "observed_wall_seconds": elapsed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _report(
        output_dir,
        context_summary,
        block_summary,
        context_benchmark,
        block_benchmark,
        lambdas,
        elapsed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "context_rows": len(context_summary),
                "block_rows": len(block_summary),
                "checks": checks,
                "elapsed_seconds": elapsed,
                "manifest_examples": len(manifest["examples"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
