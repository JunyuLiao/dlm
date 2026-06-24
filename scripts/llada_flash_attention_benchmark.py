#!/usr/bin/env python3
"""Compare LLaDA execution time with flash_attention off vs on.

Uses the same batched block forward path as llada_block_step_probe.py so the
baseline matches the normal H100 experiment runner.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llada_block_step_probe import (
    PromptRecord,
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
    probe_one_batch,
)
from llada_length_heterogeneity_probe import make_prompt_for_target


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    flash_attention: bool
    flash_layers_ready: int
    forward_calls: int
    forward_ms_mean: float
    forward_ms_max: float
    total_ms: float


def build_prompts(tokenizer: AutoTokenizer, batch_size: int, target_tokens: int) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    for request_id in range(batch_size):
        text, actual = make_prompt_for_target(
            tokenizer,
            group_name="flash_benchmark",
            request_id=request_id,
            trial=0,
            target_tokens=target_tokens,
        )
        prompts.append(PromptRecord(request_id, "medium", text))
        if request_id == 0:
            print(f"[flash benchmark] target_prompt_tokens={target_tokens} actual_prompt_tokens={actual}")
    return prompts


def count_flash_ready_layers(model: AutoModelForCausalLM) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "flash_attn_func") and getattr(module, "flash_attn_func", None) is not None
    )


def load_model(
    model_id: str,
    flash_attention: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.flash_attention = flash_attention
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    return model.to(device).eval()


def unload_model(model: AutoModelForCausalLM) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def run_benchmark(
    label: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[PromptRecord],
    args: argparse.Namespace,
    device: torch.device,
    mask_token_id: int,
) -> BenchmarkResult:
    flash_layers_ready = count_flash_ready_layers(model)

    for _ in range(args.warmup_batches):
        probe_one_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            max_steps=args.max_steps,
            confidence_threshold=args.confidence_threshold,
            acceptance_policy=args.acceptance_policy,
            mask_token_id=mask_token_id,
            trial=0,
            batch_id=0,
            run_id="warmup",
            device=device,
        )

    cuda_sync()
    start = time.perf_counter()
    _rows, batch_rows, forward_calls = probe_one_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_steps=args.max_steps,
        confidence_threshold=args.confidence_threshold,
        acceptance_policy=args.acceptance_policy,
        mask_token_id=mask_token_id,
        trial=0,
        batch_id=0,
        run_id=label,
        device=device,
    )
    cuda_sync()
    total_ms = (time.perf_counter() - start) * 1000.0

    forward_ms_total = sum(batch_row.latency_ms for batch_row in batch_rows)
    forward_ms = forward_ms_total / forward_calls if forward_calls else 0.0
    forward_ms_max = max(
        (batch_row.latency_ms / max(batch_row.batch_block_steps, 1) for batch_row in batch_rows),
        default=0.0,
    )

    return BenchmarkResult(
        label=label,
        flash_attention=bool(model.config.flash_attention),
        flash_layers_ready=flash_layers_ready,
        forward_calls=forward_calls,
        forward_ms_mean=forward_ms,
        forward_ms_max=forward_ms_max,
        total_ms=total_ms,
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"[flash benchmark] {result.label}: flash_attention={result.flash_attention} "
        f"flash_layers_ready={result.flash_layers_ready}/{32} "
        f"forward_calls={result.forward_calls} "
        f"forward_ms_mean={result.forward_ms_mean:.2f} "
        f"forward_ms_max={result.forward_ms_max:.2f} "
        f"total_ms={result.total_ms:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-prompt-tokens", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["confidence_cutoff", "topk", "threshold"], default="confidence_cutoff")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup-batches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[flash benchmark] ERROR: CUDA is required")
        sys.exit(1)

    try:
        import flash_attn  # type: ignore

        print(f"[flash benchmark] flash_attn version: {getattr(flash_attn, '__version__', 'unknown')}")
    except ImportError:
        print("[flash benchmark] WARN: flash_attn not installed; flash_attention=True will fall back to SDPA")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_token_id = choose_mask_token_id(tokenizer, args.mask_token_id)
    prompts = build_prompts(tokenizer, args.batch_size, args.target_prompt_tokens)

    print(
        "[flash benchmark] config "
        f"batch_size={args.batch_size} block_size={args.block_size} "
        f"num_blocks={args.num_blocks} max_steps={args.max_steps} "
        f"acceptance_policy={args.acceptance_policy}"
    )

    cases = [
        ("baseline_sdpa", False),
        ("flash_attention", True),
    ]
    results: list[BenchmarkResult] = []

    for label, flash_attention in cases:
        print(f"\n[flash benchmark] loading {label} (flash_attention={flash_attention})")
        model = load_model(args.model, flash_attention=flash_attention, dtype=dtype, device=device)
        result = run_benchmark(label, model, tokenizer, prompts, args, device, mask_token_id)
        print_result(result)
        results.append(result)
        unload_model(model)

    baseline, flash = results
    speedup = baseline.total_ms / flash.total_ms if flash.total_ms > 0 else float("inf")
    forward_speedup = baseline.forward_ms_mean / flash.forward_ms_mean if flash.forward_ms_mean > 0 else float("inf")

    print("\n[flash benchmark] comparison")
    print(f"  baseline total_ms={baseline.total_ms:.2f}")
    print(f"  flash    total_ms={flash.total_ms:.2f}")
    print(f"  total speedup={speedup:.3f}x")
    print(f"  baseline forward_ms_mean={baseline.forward_ms_mean:.2f}")
    print(f"  flash    forward_ms_mean={flash.forward_ms_mean:.2f}")
    print(f"  forward speedup={forward_speedup:.3f}x")
    if flash.flash_layers_ready == 0:
        print("  note: flash_attention=True but flash_attn_func not bound; compare may show no gain.")
    elif speedup < 1.0:
        print("  note: flash path was slower on this run; try more warmup or larger batch/seq.")


if __name__ == "__main__":
    main()
