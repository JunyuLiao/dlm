#!/usr/bin/env python3
"""Benchmark the fused bidirectional BLASST kernel inside LLaDA."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import (
    DiffusionLambdaSchedule,
    get_kernel_stats,
    install_bidirectional_blasst_kernel,
    reset_kernel_stats,
)
from llada_blasst_calibrate import build_masked_batch
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def timed_forward(model, inputs: torch.Tensor, repeats: int) -> tuple[torch.Tensor, float]:
    times, logits = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=inputs).logits
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    assert logits is not None
    return logits, sum(times) / len(times)


def agreement_by_noise(predictions, reference, masks, metadata):
    result = []
    for ratio in sorted({item["mask_ratio"] for item in metadata}):
        rows = [idx for idx, item in enumerate(metadata) if item["mask_ratio"] == ratio]
        matches, total = 0, 0
        for row in rows:
            selected = masks[row]
            matches += int((predictions[row, selected] == reference[row, selected]).sum())
            total += int(selected.sum())
        result.append({"mask_ratio": ratio, "agreement": matches / total, "masked_tokens": total})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--mask-ratios", default="0.15,0.5,0.9")
    parser.add_argument("--lambdas", default="0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "blasst_llada_kernel_4096.json")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > config.max_sequence_length:
        raise ValueError("context length exceeds LLaDA's native maximum")
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
    _, inputs, masks, metadata = build_masked_batch(
        tokenizer,
        args.context_length,
        ratios,
        choose_mask_token_id(tokenizer, None),
        args.num_contexts,
    )
    inputs, masks = inputs.cuda(), masks.cuda()

    # The model-bound flash_attn_func is the installed compiled dense baseline.
    for _ in range(args.warmup):
        timed_forward(model, inputs, 1)
    dense_logits, dense_ms = timed_forward(model, inputs, args.repeats)
    dense_predictions = dense_logits.argmax(-1)
    del dense_logits

    # Lambda zero uses identical fused-kernel arithmetic without pruning.
    reset_kernel_stats(inputs.device)
    with install_bidirectional_blasst_kernel(model, blasst_lambda=0.0, collect_stats=False):
        for _ in range(args.warmup):
            timed_forward(model, inputs, 1)
        zero_logits, zero_ms = timed_forward(model, inputs, args.repeats)
    zero_predictions = zero_logits.argmax(-1)
    del zero_logits

    rows = []
    for threshold in [float(value) for value in args.lambdas.split(",")]:
        # Timings exclude optional counter atomics.
        with install_bidirectional_blasst_kernel(
            model,
            blasst_lambda=threshold,
            collect_stats=False,
            num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
        ):
            for _ in range(args.warmup):
                timed_forward(model, inputs, 1)
            logits, elapsed_ms = timed_forward(model, inputs, args.repeats)
        predictions = logits.argmax(-1)
        del logits
        # One separate instrumented forward obtains exact physical tile counts.
        reset_kernel_stats(inputs.device)
        with install_bidirectional_blasst_kernel(
            model,
            blasst_lambda=threshold,
            collect_stats=True,
            num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
        ):
            timed_forward(model, inputs, 1)
        stats = get_kernel_stats(reset=True)
        rows.append(
            {
                "lambda": threshold,
                "latency_ms": elapsed_ms,
                "speedup_vs_dense_flash": dense_ms / elapsed_ms,
                "skipped_2d_tiles_per_forward": stats.skipped_tiles,
                "total_2d_tiles_per_forward": stats.total_tiles,
                "physical_tile_sparsity": stats.sparsity_ratio,
                "agreement_with_kernel_lambda_zero": agreement_by_noise(
                    predictions, zero_predictions, masks, metadata
                ),
                "agreement_with_compiled_dense": agreement_by_noise(
                    predictions, dense_predictions, masks, metadata
                ),
            }
        )

    # Diffusion-serving case: each request selects lambda from its own current
    # remaining-mask ratio, without splitting a heterogeneous batch.
    schedule = DiffusionLambdaSchedule()
    scheduled_lambdas = torch.tensor(
        [schedule.threshold(item["mask_ratio"]) for item in metadata],
        dtype=torch.float32,
        device=inputs.device,
    )
    with install_bidirectional_blasst_kernel(
        model,
        blasst_lambda=scheduled_lambdas,
        collect_stats=False,
        num_warps=args.num_warps,
        pipeline_stages=args.pipeline_stages,
    ):
        for _ in range(args.warmup):
            timed_forward(model, inputs, 1)
        scheduled_logits, scheduled_ms = timed_forward(model, inputs, args.repeats)
    scheduled_predictions = scheduled_logits.argmax(-1)
    del scheduled_logits
    reset_kernel_stats(inputs.device)
    with install_bidirectional_blasst_kernel(
        model,
        blasst_lambda=scheduled_lambdas,
        collect_stats=True,
        num_warps=args.num_warps,
        pipeline_stages=args.pipeline_stages,
    ):
        timed_forward(model, inputs, 1)
    scheduled_stats = get_kernel_stats(reset=True)

    result = {
        "model": args.model,
        "context_length": args.context_length,
        "mask_ratios": ratios,
        "num_contexts": args.num_contexts,
        "kernel": "BF16 bidirectional 2D BLASST, Q tile 128, KV tile 64, reverse traversal, base-2 softmax",
        "launch": {"num_warps": args.num_warps, "pipeline_stages": args.pipeline_stages},
        "dense_flash_ms": dense_ms,
        "kernel_lambda_zero_ms": zero_ms,
        "kernel_zero_agreement_with_compiled_dense": agreement_by_noise(
            zero_predictions, dense_predictions, masks, metadata
        ),
        "measurements": rows,
        "diffusion_schedule": {
            "per_sequence_lambdas": scheduled_lambdas.cpu().tolist(),
            "latency_ms": scheduled_ms,
            "speedup_vs_dense_flash": dense_ms / scheduled_ms,
            "skipped_2d_tiles_per_forward": scheduled_stats.skipped_tiles,
            "total_2d_tiles_per_forward": scheduled_stats.total_tiles,
            "physical_tile_sparsity": scheduled_stats.sparsity_ratio,
            "agreement_with_kernel_lambda_zero": agreement_by_noise(
                scheduled_predictions, zero_predictions, masks, metadata
            ),
            "agreement_with_compiled_dense": agreement_by_noise(
                scheduled_predictions, dense_predictions, masks, metadata
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
