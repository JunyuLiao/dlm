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

from blasst import collect_blasst_stats, install_blasst
from llada_block_step_probe import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def calibration_text() -> str:
    """Use heterogeneous repository prose rather than a repeated sentence."""
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "experiment_plan.md",
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
        "--lambdas",
        default="0,0.001,0.003,0.01,0.03,0.1,0.3,1.0",
        help="lambda=0 control followed by a readable approximately logarithmic sweep",
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

    measurements, reference_predictions = [], None
    for threshold in [float(value) for value in args.lambdas.split(",")]:
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
        if threshold == 0.0:
            reference_predictions = predictions.clone()
        assert reference_predictions is not None, "lambda sweep must start with 0"
        per_sample = masked_metrics(predictions, labels, masks, metadata, reference_predictions)
        per_ratio = []
        for ratio in ratios:
            selected = [row for row in per_sample if row["mask_ratio"] == ratio]
            total = sum(row["masked_tokens"] for row in selected)
            per_ratio.append(
                {
                    "mask_ratio": ratio,
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
        measurements.append(
            {
                "lambda": threshold,
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
        )

    eligible = [
        row
        for row in measurements
        if all(
            metric["agreement_with_lambda_zero"] >= 1.0 - args.max_relative_disagreement
            for metric in row["per_mask_ratio"]
        )
    ]
    selected = max(eligible, key=lambda row: row["physical_block_sparsity"])
    result = {
        "model": args.model,
        "context_length": args.context_length,
        "mask_ratios": ratios,
        "num_contexts": args.num_contexts,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "kv_traversal": "reverse (matching FlashAttention 2.8)",
        "primary_accuracy_metric": "top-1 agreement with lambda=0 on masked positions",
        "selection_rule": f"maximum physical sparsity with relative disagreement <= {args.max_relative_disagreement} at every noise level",
        "selected_lambda": selected["lambda"],
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
