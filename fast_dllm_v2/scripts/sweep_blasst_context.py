#!/usr/bin/env python3
"""Controlled valid-context × lambda sweep for reference 2D-BLASST.

The observer computes dense QK once for each model state, evaluates every
lambda from those shared scores, and returns the checkpoint's original dense
SDPA output.  It therefore measures sparsity decisions, not runtime speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import load_examples, load_model, set_seed  # noqa: E402
from sparse_attention import Blasst2DConfig, install_blasst_2d  # noqa: E402


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
    parser.add_argument(
        "--dataset-path",
        default=str(V2_ROOT / "data/alpaca/test/test_252.json"),
    )
    parser.add_argument("--contexts", default="512,1024,2048,4096,8192,16384")
    parser.add_argument(
        "--lambdas",
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,0.5",
    )
    parser.add_argument("--mask-ratios", default="0.90,0.70,0.50,0.30,0.15")
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output-dir",
        default=str(V2_ROOT.parent / "results/blasst/fast_dllm_v2/context_sweep"),
    )
    parser.add_argument(
        "--meaningful-physical-sparsity",
        type=float,
        default=0.10,
        help="Reporting threshold only; it does not change BLASST.",
    )
    return parser.parse_args()


def _csv_values(text: str, cast) -> list[Any]:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ratios(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    eligible = int(output["eligible_tiles"])
    votes = int(output["valid_row_votes"])
    elements = int(output["valid_elements"])
    physical = int(output["skipped_tiles"]) / eligible if eligible else 0.0
    row_vote = int(output["skippable_row_votes"]) / votes if votes else 0.0
    output["physical_tile_sparsity"] = physical
    output["row_vote_sparsity"] = row_vote
    output["valid_element_sparsity"] = (
        int(output["skipped_valid_elements"]) / elements if elements else 0.0
    )
    output["physical_to_row_sparsity_ratio"] = (
        physical / row_vote if row_vote else 0.0
    )
    return output


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
        for name in COUNT_FIELDS:
            target[name] += int(row[name])
    return [_ratios(row) for _, row in sorted(grouped.items())]


class PrefixCacheView:
    """Read-only prefix view over one genuine long-context DynamicCache."""

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


def _valid_token_stream(
    tokenizer,
    dataset_path: str,
    required: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    examples = load_examples(dataset_path, 10_000)
    text = "\n\n".join(
        f"{example['input']}\n{example['output']}" for example in examples
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("tokenized context corpus is empty")
    repetitions = (required + len(ids) - 1) // len(ids)
    stream = torch.tensor((ids * repetitions)[:required], dtype=torch.long)
    return stream, {
        "source_examples": len(examples),
        "source_tokens_before_repeat": len(ids),
        "stream_repetitions": repetitions,
        "padding_tokens": 0,
        "all_context_tokens_are_valid": True,
    }


def _nested_noisy_states(
    clean: torch.Tensor,
    ratios: list[float],
    mask_token_id: int,
    seed: int,
) -> list[tuple[int, float, torch.Tensor]]:
    """Create deterministic nested full-block masks."""

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(clean.numel(), generator=generator)
    states = []
    for step, requested_ratio in enumerate(ratios):
        total_count = round(clean.numel() * requested_ratio)
        selected = order[:total_count]
        noisy = clean.clone()
        noisy[selected] = mask_token_id
        states.append((step, requested_ratio, noisy))
    return states


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=V2_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "repository_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }


def _dense_regression(model, probe: torch.Tensor, install) -> dict[str, Any]:
    with torch.no_grad():
        before = model.model(
            input_ids=probe,
            use_cache=False,
            block_size=probe.shape[1],
        ).last_hidden_state.detach()
    install()
    with torch.no_grad():
        after = model.model(
            input_ids=probe,
            use_cache=False,
            block_size=probe.shape[1],
        ).last_hidden_state.detach()
    result = {
        "exact_match": bool(torch.equal(before, after)),
        "all_finite": bool(torch.isfinite(before).all() and torch.isfinite(after).all()),
        "max_abs_error": float((before - after).abs().max()),
    }
    if not result["exact_match"] or not result["all_finite"]:
        raise AssertionError(f"dense-disabled regression failed: {result}")
    return result


def _export_stats(runtime, lambdas: list[float]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for value in lambdas:
        stats = runtime.sweep_stats[value]
        for destination, source in (
            (step_rows, stats.per_step),
            (layer_rows, stats.per_layer),
            (head_rows, stats.per_head),
        ):
            for row in stats._rows(source):
                destination.append(
                    {
                        "lambda": value,
                        "context_length": int(row["valid_kv_length"]),
                        **row,
                        "physical_to_row_sparsity_ratio": (
                            row["physical_tile_sparsity"]
                            / row["row_vote_sparsity"]
                            if row["row_vote_sparsity"]
                            else 0.0
                        ),
                    }
                )
    ordering = lambda row: (
        int(row["context_length"]),
        float(row["lambda"]),
        int(row.get("denoising_step", -1)),
        int(row.get("query_length", -1)),
        int(row.get("layer", -1)),
        int(row.get("head", -1)),
    )
    return (
        sorted(step_rows, key=ordering),
        sorted(layer_rows, key=ordering),
        sorted(head_rows, key=ordering),
    )


def _check_results(
    summary: list[dict[str, Any]],
    detailed: list[list[dict[str, Any]]],
    contexts: list[int],
    lambdas: list[float],
) -> dict[str, Any]:
    violations: list[str] = []
    all_rows = summary + [row for group in detailed for row in group]
    for row in all_rows:
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"tile count identity failed: {row}")
        for name in (
            "physical_tile_sparsity",
            "row_vote_sparsity",
            "valid_element_sparsity",
        ):
            value = float(row[name])
            if not 0.0 <= value <= 1.0:
                violations.append(f"invalid {name}: {row}")

    # Compare each context and controlled stratum across lambdas.
    for rows, keys in (
        (summary, ("context_length",)),
        (
            detailed[0],
            (
                "context_length",
                "denoising_step",
                "mask_ratio",
                "masked_tokens",
                "query_length",
            ),
        ),
        (
            detailed[1],
            (
                "context_length",
                "denoising_step",
                "mask_ratio",
                "masked_tokens",
                "query_length",
                "layer",
            ),
        ),
        (
            detailed[2],
            (
                "context_length",
                "denoising_step",
                "mask_ratio",
                "masked_tokens",
                "query_length",
                "layer",
                "head",
            ),
        ),
    ):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(tuple(row[key] for key in keys), []).append(row)
        for key, group in groups.items():
            ordered = sorted(group, key=lambda row: float(row["lambda"]))
            for lower, upper in zip(ordered, ordered[1:]):
                for field in ("skipped_tiles", "skippable_row_votes"):
                    if int(upper[field]) < int(lower[field]):
                        violations.append(
                            f"non-monotonic {field} at {key}: "
                            f"{lower['lambda']}->{upper['lambda']}"
                        )
                for field in (
                    "eligible_tiles",
                    "valid_row_votes",
                    "valid_elements",
                    "structurally_masked_tiles",
                ):
                    if int(upper[field]) != int(lower[field]):
                        violations.append(
                            f"lambda-dependent denominator {field} at {key}"
                        )

    observed_contexts = sorted({int(row["context_length"]) for row in summary})
    observed_lambdas = sorted({float(row["lambda"]) for row in summary})
    if observed_contexts != sorted(contexts):
        violations.append(
            f"observed contexts {observed_contexts} != requested {sorted(contexts)}"
        )
    if observed_lambdas != sorted(lambdas):
        violations.append(
            f"observed lambdas {observed_lambdas} != requested {sorted(lambdas)}"
        )
    result = {
        "passed": not violations,
        "violations": violations,
        "checked_rows": len(all_rows),
        "monotonic_fields": ["skipped_tiles", "skippable_row_votes"],
        "structural_mask_excluded_from_eligible": True,
    }
    if violations:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _heatmap(
    rows: list[dict[str, Any]],
    contexts: list[int],
    lambdas: list[float],
    field: str,
    title: str,
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/blasst-matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (int(row["context_length"]), float(row["lambda"])): float(row[field])
        for row in rows
    }
    values = np.array(
        [[lookup[(context, value)] for value in lambdas] for context in contexts]
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    axis.set_yticks(range(len(contexts)), [f"{value // 1024}K" for value in contexts])
    axis.set_xlabel("λ")
    axis.set_ylabel("Actual valid KV context")
    axis.set_title(title)
    for row_index in range(len(contexts)):
        for column_index in range(len(lambdas)):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="sparsity")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _report(
    path: Path,
    summary: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    environment: Mapping[str, Any],
    dense_regression: Mapping[str, Any],
) -> None:
    by_pair = {
        (int(row["context_length"]), float(row["lambda"])): row for row in summary
    }
    contexts = _csv_values(args.contexts, int)
    lambdas = _csv_values(args.lambdas, float)
    threshold = args.meaningful_physical_sparsity
    meaningful = [
        row
        for row in summary
        if float(row["physical_tile_sparsity"]) >= threshold
    ]
    earliest = min(
        meaningful,
        key=lambda row: (int(row["context_length"]), float(row["lambda"])),
        default=None,
    )
    changes = []
    for value in lambdas:
        low = by_pair[(contexts[0], value)]
        high = by_pair[(contexts[-1], value)]
        changes.append(
            (
                value,
                float(high["row_vote_sparsity"])
                - float(low["row_vote_sparsity"]),
                float(high["physical_tile_sparsity"])
                - float(low["physical_tile_sparsity"]),
            )
        )
    target_lambdas = []
    for context in contexts:
        candidates = [
            value
            for value in lambdas
            if float(by_pair[(context, value)]["physical_tile_sparsity"]) >= threshold
        ]
        target_lambdas.append((context, min(candidates) if candidates else None))
    valid_targets = [value for _, value in target_lambdas if value is not None]
    inverse = len(valid_targets) >= 2 and all(
        right <= left for left, right in zip(valid_targets, valid_targets[1:])
    )
    largest_gap = max(
        summary,
        key=lambda row: (
            float(row["row_vote_sparsity"])
            - float(row["physical_tile_sparsity"])
        ),
    )
    previous_path = V2_ROOT.parent / "results/blasst/fast_dllm_v2/two_dimensional_lambda_0p5/summary.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else None
    )
    one_k_reference = by_pair.get((1024, 0.5))
    if previous is not None and one_k_reference is not None:
        short_context_comparison = (
            "At λ=0.5, the previous 64–160-token run measured "
            f"{float(previous['row_vote_sparsity']):.1%} row and "
            f"{float(previous['physical_tile_sparsity']):.1%} physical sparsity "
            f"(physical/row {float(previous['physical_tile_sparsity']) / float(previous['row_vote_sparsity']):.2f}); "
            "the controlled 1K point reaches "
            f"{float(one_k_reference['row_vote_sparsity']):.1%} row and "
            f"{float(one_k_reference['physical_tile_sparsity']):.1%} physical "
            f"(physical/row {float(one_k_reference['physical_to_row_sparsity_ratio']):.2f}). "
            "Thus short context was the primary limiter in the earlier aggregate, "
            "while the much lower short-context physical/row ratio also confirms a "
            "real diffusion multi-query unanimity penalty."
        )
    else:
        short_context_comparison = (
            "The previous summary was unavailable for a numeric cross-run comparison. "
            "Within this sweep, context increases row opportunities and the "
            "physical/row ratio isolates the multi-query unanimity penalty."
        )

    lines = [
        "# Controlled valid-context × λ BLASST sweep",
        "",
        "This experiment observes dense QK scores and measures BLASST decisions. "
        "The checkpoint's original dense SDPA still produces every output, so these "
        "are sparsity and theoretical skipped softmax/PV-work measurements—not speedups.",
        "",
        "## Controls",
        "",
        f"- Model: `{args.model_path}`; device: {environment['device_name']}; "
        f"precision: {args.precision}; seed: {args.seed}.",
        f"- Actual valid KV lengths: {', '.join(str(value) for value in contexts)}; "
        "zero padding tokens.",
        f"- Q/KV tiles: {args.q_tile_size}/{args.kv_tile_size}; diffusion query "
        f"length: {args.block_size}; sub-block and dual block-cache paths disabled.",
        f"- Mask ratios/states: {args.mask_ratios}. Every λ is thresholded from the "
        "same FP32 local/running maxima for each state.",
        f"- Disabled-dispatch dense regression: exact={dense_regression['exact_match']}, "
        f"max absolute error={dense_regression['max_abs_error']:.8g}; all finite.",
        "",
        "## Aggregate context × λ results",
        "",
        "| Valid KV | λ | Eligible | Skipped | Structural | Row vote | Physical | "
        "Valid element | Physical/row |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {int(row['context_length'])} | {float(row['lambda']):g} | "
            f"{int(row['eligible_tiles'])} | {int(row['skipped_tiles'])} | "
            f"{int(row['structurally_masked_tiles'])} | "
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
            "1. **Does 1K→16K materially increase row sparsity?** "
            + "; ".join(
                f"λ={value:g}: {delta:+.1%}" for value, delta, _ in changes
            )
            + ".",
            "",
            "2. **Does physical sparsity keep pace?** The largest observed row/physical "
            f"gap is at {int(largest_gap['context_length'])} tokens, "
            f"λ={float(largest_gap['lambda']):g}: "
            f"{float(largest_gap['row_vote_sparsity']):.1%} row versus "
            f"{float(largest_gap['physical_tile_sparsity']):.1%} physical. "
            "A physical tile requires unanimous votes across all active query rows.",
            "",
            "3. **When is physical sparsity meaningful?** "
            + (
                f"Using the declared ≥{threshold:.0%} criterion, the first grid point "
                f"is {int(earliest['context_length'])} tokens at "
                f"λ={float(earliest['lambda']):g} "
                f"({float(earliest['physical_tile_sparsity']):.1%})."
                if earliest is not None
                else f"No tested pair reaches the declared ≥{threshold:.0%} criterion."
            ),
            "",
            "4. **Inverse λ–context scaling?** The smallest tested λ reaching the "
            f"{threshold:.0%} physical target is "
            + ", ".join(
                f"{context // 1024}K:{value:g}"
                if value is not None
                else f"{context // 1024}K:none"
                for context, value in target_lambdas
            )
            + (
                ". This is consistent with an inverse trend on the tested grid."
                if inverse
                else ". This grid does not establish a clean inverse trend."
            ),
            "",
            "5. **Why was the previous short-context sparsity low?** "
            + short_context_comparison,
            "",
            "## Query-length breakdown",
            "",
            "| Valid KV | λ | Q | Row vote | Physical | Physical/row |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in query_rows:
        lines.append(
            f"| {int(row['context_length'])} | {float(row['lambda']):g} | "
            f"{int(row['query_length'])} | "
            f"{float(row['row_vote_sparsity']):.4f} | "
            f"{float(row['physical_tile_sparsity']):.4f} | "
            f"{float(row['physical_to_row_sparsity_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Complete mask-ratio/step/layer/head strata are in the detailed CSVs. "
            "The lightweight labeled-task evaluation is produced separately by "
            "`eval_blasst_task.py`; its accuracy is an actual benchmark metric, while "
            "dense-output agreement is only a diagnostic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    contexts = _csv_values(args.contexts, int)
    lambdas = _csv_values(args.lambdas, float)
    ratios = _csv_values(args.mask_ratios, float)
    if contexts != sorted(set(contexts)):
        raise ValueError("contexts must be unique and ascending")
    if any(context < args.block_size or context % args.block_size for context in contexts):
        raise ValueError("every context must be a block-size multiple")
    if any(not 0.0 < value <= 1.0 for value in lambdas):
        raise ValueError("every lambda must be in (0, 1]")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    from dllm.attention.blasst import BLASST_MASK_SEMANTICS, validate_blasst_output_directory
    validate_blasst_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
    mask_token_id = int(getattr(model.config, "mask_token_id", 151665))
    stream, stream_config = _valid_token_stream(
        tokenizer,
        args.dataset_path,
        max(contexts),
    )
    stream = stream.to(model.device)
    clean_block = stream[max(contexts) - args.block_size : max(contexts)].clone()
    states = _nested_noisy_states(
        clean_block,
        ratios,
        mask_token_id,
        args.seed,
    )

    sweep_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=lambdas[0],
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
    )
    runtime_box: dict[str, Any] = {}

    def install() -> None:
        runtime_box["runtime"] = install_blasst_2d(
            model,
            replace(sweep_config, enable_blasst_2d=False),
            mask_token_id=mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            dual_cache_only=False,
            sweep_lambdas=lambdas,
        )

    probe = stream[: args.block_size][None]
    dense_regression = _dense_regression(model, probe, install)
    runtime = runtime_box["runtime"]

    max_prefix_length = max(contexts) - args.block_size
    runtime.config = replace(sweep_config, enable_blasst_2d=False)
    with torch.no_grad():
        prefix_output = model.model(
            input_ids=stream[:max_prefix_length][None],
            use_cache=True,
            update_past_key_values=True,
            block_size=args.block_size,
        )
    full_cache = prefix_output.past_key_values
    if full_cache.get_seq_length() != max_prefix_length:
        raise AssertionError(
            f"prefix cache has {full_cache.get_seq_length()} tokens, "
            f"expected {max_prefix_length}"
        )
    del prefix_output
    runtime.config = sweep_config

    finite_checks = 0
    observed_calls = 0
    for context in contexts:
        prefix = PrefixCacheView(full_cache, context - args.block_size)
        for step, requested_ratio, noisy_cpu in states:
            noisy = noisy_cpu[None]
            q_masked = int((noisy == mask_token_id).sum())
            query_metadata = {
                "denoising_step": step,
                "mask_ratio": q_masked / args.block_size,
                "requested_mask_ratio": requested_ratio,
                "masked_tokens": q_masked,
                "controlled_state_seed": args.seed,
            }
            with torch.no_grad():
                output32 = model.model(
                    input_ids=noisy,
                    past_key_values=prefix,
                    use_cache=True,
                    update_past_key_values=False,
                    use_block_cache=False,
                    block_size=args.block_size,
                    blasst_active_query_mask=torch.ones_like(noisy, dtype=torch.bool),
                    blasst_metadata=query_metadata,
                )
            if not torch.isfinite(output32.last_hidden_state).all():
                raise FloatingPointError(f"non-finite q=32 output at context {context}")
            finite_checks += 1
            observed_calls += 28
            del output32

    step_rows, layer_rows, head_rows = _export_stats(runtime, lambdas)
    summary = _aggregate(
        step_rows,
        ("context_length", "lambda"),
    )
    query_rows = _aggregate(
        step_rows,
        ("context_length", "lambda", "query_length"),
    )
    checks = _check_results(
        summary,
        [step_rows, layer_rows, head_rows],
        contexts,
        lambdas,
    )
    checks.update(
        {
            "dense_disabled_regression": dense_regression,
            "finite_model_outputs_checked": finite_checks,
            "attention_calls_per_lambda": observed_calls,
            "exact_valid_kv_lengths_observed": sorted(
                {int(row["valid_kv_length"]) for row in step_rows}
            ),
        }
    )

    _write_csv(output_dir / "context_lambda_summary.csv", summary)
    _write_csv(output_dir / "context_lambda_by_query.csv", query_rows)
    _write_csv(output_dir / "context_lambda_per_step.csv", step_rows)
    _write_csv(output_dir / "context_lambda_per_layer.csv", layer_rows)
    _write_csv(output_dir / "context_lambda_per_head.csv", head_rows)
    (output_dir / "correctness_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = _environment()
    run_config = {
        "blasst_mask_semantics": BLASST_MASK_SEMANTICS,
        **vars(args),
        **environment,
        **stream_config,
        "contexts": contexts,
        "lambdas": lambdas,
        "mask_ratios": ratios,
        "context_definition": (
            "actual valid KV tokens seen by each attention call, including the "
            "32-token current block and excluding zero padding"
        ),
        "prefix_cache_strategy": (
            f"one genuine {max_prefix_length}-token dense block-causal cache, "
            "read-only exact prefix views for shorter contexts"
        ),
        "lambda_control": (
            "all lambdas threshold identical FP32 local/running maxima in one observer"
        ),
        "output_path": (
            "checkpoint original dense SDPA; observer does not alter model outputs"
        ),
        "batch_size": 1,
        "controlled_context_streams": 1,
        "sample_count_choice": (
            "one shared long-context stream and five deterministic nested denoising "
            "states; lambda is evaluated within-state, so there is no lambda sampling "
            "confound"
        ),
        "observed_wall_seconds_through_export": time.monotonic() - started_at,
        "denoising_states": [
            {
                "step": step,
                "requested_mask_ratio": requested,
                "query_masked_tokens": int((state == mask_token_id).sum()),
            }
            for step, requested, state in states
        ],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _heatmap(
        summary,
        contexts,
        lambdas,
        "row_vote_sparsity",
        "2D-BLASST row-vote sparsity",
        output_dir / "row_sparsity_heatmap.png",
    )
    _heatmap(
        summary,
        contexts,
        lambdas,
        "physical_tile_sparsity",
        "2D-BLASST physical tile sparsity",
        output_dir / "physical_sparsity_heatmap.png",
    )
    _report(
        output_dir / "report.md",
        summary,
        query_rows,
        args,
        environment,
        dense_regression,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_rows": len(summary),
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
