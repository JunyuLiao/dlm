#!/usr/bin/env python3
"""Compare dense FlashAttention and BLASST on fixed-length LLaDA inputs."""

import argparse
import json
import pathlib
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from blasst import collect_blasst_stats, install_blasst
from llada_eval_utils import (
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def timed_forward(model, input_ids, repeats):
    elapsed, logits = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
    return logits, sum(elapsed) / len(elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--blasst-lambda", type=float, default=0.001)
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--calibration-lambdas",
        default=None,
        help="Comma-separated lambda sweep; reuses one model load and dense baseline",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for the LLaDA comparison")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > config.max_sequence_length:
        raise ValueError("context length exceeds the model's native maximum")
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = tokenizer("BLASST sparse attention evaluation. ", return_tensors="pt").input_ids
    input_ids = seed.repeat(1, (args.context_length + seed.shape[1] - 1) // seed.shape[1])[:, : args.context_length].cuda()
    for _ in range(args.warmup):
        timed_forward(model, input_ids, 1)
    dense_logits, dense_ms = timed_forward(model, input_ids, args.repeats)
    baseline = {
        "context_length": args.context_length,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "dense_ms": dense_ms,
    }
    lambdas = (
        [float(value) for value in args.calibration_lambdas.split(",")]
        if args.calibration_lambdas
        else [args.blasst_lambda]
    )
    measurements = []
    dense_tokens = dense_logits.argmax(-1)
    for threshold in lambdas:
        collect_blasst_stats(reset=True)
        with install_blasst(
            model,
            blasst_lambda=threshold,
            q_block_size=args.q_block_size,
            kv_block_size=args.kv_block_size,
        ):
            sparse_logits, sparse_ms = timed_forward(model, input_ids, args.repeats)
        stats = collect_blasst_stats(reset=True)
        measurements.append(
            {
                "blasst_lambda": threshold,
                "log_lambda": __import__("math").log(threshold) if threshold else float("-inf"),
                "sparse_ms": sparse_ms,
                "speedup": dense_ms / sparse_ms,
                "accuracy_top1_agreement": (dense_tokens == sparse_logits.argmax(-1)).float().mean().item(),
                "sparsity_ratio": stats.sparsity_ratio,
                "skipped_blocks_per_forward": stats.skipped_blocks // args.repeats,
                "total_blocks_per_forward": stats.total_blocks // args.repeats,
                "row_block_sparsity": stats.skipped_row_blocks / stats.total_row_blocks,
            }
        )
    print(json.dumps({**baseline, "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
