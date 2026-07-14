#!/usr/bin/env python3
"""Calibrate BLASST on native-context, masked LLaDA denoising inputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst import (
    PhysicalTileScoreCollector,
    collect_blasst_stats,
    install_blasst,
    lambda_for_physical_sparsity,
    select_largest_eligible_lambda,
)
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def calibration_text() -> str:
    """Use heterogeneous repository prose rather than a repeated sentence."""
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "blasst.md",
        ROOT / "docs" / "llada_blasst_kernel.md",
        ROOT / "data" / "prompts_heterogeneous.jsonl",
    ]
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def build_masked_batch(tokenizer, context_length: int, mask_ratios: list[float], mask_id: int, num_contexts: int):
    tokens = tokenizer(calibration_text(), add_special_tokens=False)["input_ids"]
    if len(tokens) < context_length:
        raise ValueError(f"calibration corpus has {len(tokens)} tokens; need {context_length}")
    max_offset = len(tokens) - context_length
    clean, corrupted, masks, metadata = [], [], [], []
    for context_index in range(num_contexts):
        offset = round(context_index * max_offset / max(1, num_contexts - 1))
        row = torch.tensor(tokens[offset : offset + context_length])
        for ratio_index, ratio in enumerate(mask_ratios):
            generator = torch.Generator().manual_seed(20260701 + context_index * 100 + ratio_index)
            mask = torch.rand(context_length, generator=generator).lt(ratio)
            changed = row.clone()
            changed[mask] = mask_id
            clean.append(row)
            corrupted.append(changed)
            masks.append(mask)
            metadata.append({"context_index": context_index, "mask_ratio": ratio})
    return torch.stack(clean), torch.stack(corrupted), torch.stack(masks), metadata


def forward(model, inputs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=inputs).logits
    torch.cuda.synchronize()
    return logits, (time.perf_counter() - start) * 1000


def masked_metrics(predictions, labels, masks, metadata, reference_predictions=None):
    rows = []
    for row in range(labels.shape[0]):
        selected = masks[row]
        item = {
            **metadata[row],
            "masked_tokens": int(selected.sum()),
            "recovery_accuracy": (predictions[row, selected] == labels[row, selected]).float().mean().item(),
        }
        if reference_predictions is not None:
            item["agreement_with_lambda_zero"] = (
                predictions[row, selected] == reference_predictions[row, selected]
            ).float().mean().item()
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mask-ratios", default="0.15,0.5,0.9")
    parser.add_argument("--num-contexts", type=int, default=2)
    parser.add_argument(
        "--target-physical-sparsities",
        default="0.05,0.1,0.2,0.3,0.4,0.5",
        help="comma-separated physical tile sparsities used to derive lambda quantiles",
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument(
        "--max-relative-disagreement",
        type=float,
        default=0.05,
        help="maximum masked-token disagreement with lambda=0 allowed at every noise level",
    )
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "blasst_calibration_4096.json")
    args = parser.parse_args()
    if not 0.0 <= args.max_relative_disagreement <= 1.0:
        raise ValueError("max relative disagreement must be in [0, 1]")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > config.max_sequence_length:
        raise ValueError(
            f"context length {args.context_length} exceeds model maximum {config.max_sequence_length}"
        )
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ratios = [float(value) for value in args.mask_ratios.split(",")]
    labels, inputs, masks, metadata = build_masked_batch(
        tokenizer, args.context_length, ratios, choose_mask_token_id(tokenizer, None), args.num_contexts
    )
    labels, inputs, masks = labels.cuda(), inputs.cuda(), masks.cuda()

    targets = [float(value) for value in args.target_physical_sparsities.split(",")]
    if targets != sorted(set(targets)) or any(not 0.0 <= value <= 1.0 for value in targets):
        raise ValueError("target physical sparsities must be unique, sorted values in [0, 1]")

    # A single lambda-zero pass both establishes the prediction reference and
    # records R_j = max_i exp(local_max_ij - running_max_i) for every physical
    # attention tile. Threshold candidates then come directly from quantiles
    # of this physical score distribution, conditioned on denoising state.
    collector = PhysicalTileScoreCollector()
    collect_blasst_stats(reset=True)
    with install_blasst(
        model,
        blasst_lambda=0.0,
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
        physical_score_callback=collector,
    ):
        reference_logits, reference_elapsed_ms = forward(model, inputs)
    reference_stats = collect_blasst_stats(reset=True)
    reference_predictions = reference_logits.argmax(dim=-1)
    del reference_logits

    score_groups = collector.bucketed_scores([item["mask_ratio"] for item in metadata])
    candidate_schedules = []
    for target in targets:
        candidate_schedules.append(
            {
                "target_physical_sparsity": target,
                "lambda_by_mask_ratio": {
                    ratio: lambda_for_physical_sparsity(score_groups[(ratio,)], target) for ratio in ratios
                },
            }
        )

    def evaluate(lambda_by_ratio, *, target_physical_sparsity):
        threshold = torch.tensor(
            [lambda_by_ratio[item["mask_ratio"]] for item in metadata],
            dtype=torch.float32,
            device=inputs.device,
        )
        collect_blasst_stats(reset=True)
        with install_blasst(
            model,
            blasst_lambda=threshold,
            q_block_size=args.q_block_size,
            kv_block_size=args.kv_block_size,
        ):
            logits, elapsed_ms = forward(model, inputs)
        stats = collect_blasst_stats(reset=True)
        predictions = logits.argmax(dim=-1)
        del logits
        per_sample = masked_metrics(predictions, labels, masks, metadata, reference_predictions)
        per_ratio = []
        for ratio in ratios:
            selected_indices = [index for index, item in enumerate(metadata) if item["mask_ratio"] == ratio]
            selected = [per_sample[index] for index in selected_indices]
            total = sum(row["masked_tokens"] for row in selected)
            skipped_tiles = sum(stats.skipped_blocks_per_sequence[index] for index in selected_indices)
            total_tiles = sum(stats.total_blocks_per_sequence[index] for index in selected_indices)
            per_ratio.append(
                {
                    "mask_ratio": ratio,
                    "lambda": lambda_by_ratio[ratio],
                    "target_physical_sparsity": target_physical_sparsity,
                    "achieved_physical_sparsity": skipped_tiles / total_tiles if total_tiles else 0.0,
                    "masked_tokens": total,
                    "recovery_accuracy": sum(
                        row["recovery_accuracy"] * row["masked_tokens"] for row in selected
                    ) / total,
                    "agreement_with_lambda_zero": sum(
                        row["agreement_with_lambda_zero"] * row["masked_tokens"] for row in selected
                    ) / total,
                }
            )
        mean_accuracy = sum(row["recovery_accuracy"] for row in per_ratio) / len(per_ratio)
        return {
            "target_physical_sparsity": target_physical_sparsity,
            "lambda_by_mask_ratio": {str(key): value for key, value in lambda_by_ratio.items()},
            "elapsed_ms": elapsed_ms,
            "physical_block_sparsity": stats.sparsity_ratio,
            "skipped_physical_blocks": stats.skipped_blocks,
            "total_physical_blocks": stats.total_blocks,
            "row_block_sparsity": (
                stats.skipped_row_blocks / stats.total_row_blocks if stats.total_row_blocks else 0.0
            ),
            "mean_recovery_accuracy": mean_accuracy,
            "per_mask_ratio": per_ratio,
            "per_sample": per_sample,
        }

    baseline_samples = masked_metrics(
        reference_predictions, labels, masks, metadata, reference_predictions
    )
    baseline = {
        "target_physical_sparsity": 0.0,
        "lambda_by_mask_ratio": {str(ratio): 0.0 for ratio in ratios},
        "elapsed_ms": reference_elapsed_ms,
        "physical_block_sparsity": reference_stats.sparsity_ratio,
        "skipped_physical_blocks": reference_stats.skipped_blocks,
        "total_physical_blocks": reference_stats.total_blocks,
        "per_sample": baseline_samples,
    }
    measurements = [baseline]
    for candidate in candidate_schedules:
        measurements.append(
            evaluate(
                candidate["lambda_by_mask_ratio"],
                target_physical_sparsity=candidate["target_physical_sparsity"],
            )
        )

    selected_lambda_by_ratio = {}
    for ratio in ratios:
        candidates = []
        for measurement in measurements[1:]:
            metric = next(row for row in measurement["per_mask_ratio"] if row["mask_ratio"] == ratio)
            candidates.append((metric["lambda"], metric["agreement_with_lambda_zero"]))
        # Deliberately exact: 0.9499999938 fails a 0.95 requirement.
        selected_lambda_by_ratio[ratio] = select_largest_eligible_lambda(
            candidates, minimum_agreement=1.0 - args.max_relative_disagreement
        )

    selected_measurement = evaluate(selected_lambda_by_ratio, target_physical_sparsity="accuracy-constrained")
    result = {
        "model": args.model,
        "context_length": args.context_length,
        "mask_ratios": ratios,
        "num_contexts": args.num_contexts,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "kv_traversal": "reverse (matching FlashAttention 2.8)",
        "calibration_score": "R_j = max_i exp(local_max_ij - running_max_i)",
        "calibration_bucket": "remaining mask ratio",
        "target_physical_sparsities": targets,
        "primary_accuracy_metric": "top-1 agreement with lambda=0 on masked positions",
        "selection_rule": f"largest quantile-derived lambda per denoising bucket with relative disagreement <= {args.max_relative_disagreement}",
        "selected_lambda_by_mask_ratio": {
            str(key): value for key, value in selected_lambda_by_ratio.items()
        },
        "selected_measurement": selected_measurement,
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
