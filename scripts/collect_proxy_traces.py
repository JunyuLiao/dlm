#!/usr/bin/env python3
"""Collect exact multi-step LLaDA physical-tile traces (slow reference path)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule, install_blasst
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)
from tracing import IncrementalTraceCollector, TraceContext


def corpus() -> str:
    paths = [ROOT / "README.md", ROOT / "docs" / "blasst.md", ROOT / "data" / "prompts_heterogeneous.jsonl"]
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def config_heads(config: object) -> tuple[int, int]:
    query = next(
        int(getattr(config, name))
        for name in ("n_heads", "num_attention_heads", "n_head")
        if hasattr(config, name)
    )
    kv = next(
        (int(getattr(config, name)) for name in ("num_key_value_heads", "n_kv_heads") if hasattr(config, name)),
        query,
    )
    return query, kv


def cyclic_contexts(tokens: list[int], context_length: int, num_contexts: int) -> torch.Tensor:
    """Build deterministic rotated windows, repeating a short corpus safely."""
    if not tokens:
        raise ValueError("trace corpus tokenized to an empty sequence")
    if context_length <= 0 or num_contexts <= 0:
        raise ValueError("context_length and num_contexts must be positive")
    rows = []
    for context_index in range(num_contexts):
        offset = (context_index * len(tokens)) // num_contexts
        rows.append(torch.tensor([tokens[(offset + position) % len(tokens)] for position in range(context_length)]))
    return torch.stack(rows)


def write_json_atomic(path: pathlib.Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-ratios", default="0.9,0.5,0.15")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--shard-records", type=int, default=100000)
    parser.add_argument("--device-buffer-tiles", type=int, default=256)
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--bitpack-masks", action="store_true")
    parser.add_argument("--quantize-log-scores", type=int, choices=(4, 8))
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "artifacts" / "proxy_traces")
    parser.add_argument("--resume", action="store_true", help="resume units recorded in progress.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA trace collection requires CUDA")
    ratios = [float(value) for value in args.mask_ratios.split(",")]
    if ratios != sorted(ratios, reverse=True):
        raise ValueError("mask ratios must follow the denoising trajectory from high to low")
    if args.batch_size <= 0 or args.batch_size > args.num_contexts:
        raise ValueError("batch size must be in [1, num_contexts]")
    progress_path = args.output_dir / "progress.json"
    existing_shards = list(args.output_dir.glob("trace-*.npz")) if args.output_dir.exists() else []
    if existing_shards and not progress_path.exists():
        raise ValueError("trace shards exist without progress.json; choose a fresh output directory")
    if progress_path.exists() and not args.resume:
        raise ValueError("trace collection already exists; pass --resume or choose a fresh output directory")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": []}
    completed = set(progress["completed"])

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > config.max_sequence_length:
        raise ValueError("context length exceeds model maximum")
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokens = tokenizer(corpus(), add_special_tokens=False)["input_ids"]
    clean = cyclic_contexts(tokens, args.context_length, args.num_contexts)
    mask_id = choose_mask_token_id(tokenizer, None)
    query_heads, kv_heads = config_heads(config)
    schedule = DiffusionLambdaSchedule()

    priorities = []
    for context_index in range(args.num_contexts):
        priorities.append(
            torch.rand(args.context_length, generator=torch.Generator().manual_seed(args.seed + context_index))
        )
    priority = torch.stack(priorities)
    sample_ids = [f"context-{index}" for index in range(args.num_contexts)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": args.model,
        "context_length": args.context_length,
        "num_contexts": args.num_contexts,
        "batch_size": args.batch_size,
        "mask_ratios": ratios,
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "corpus_tokens": len(tokens),
        "corpus_repeated": len(tokens) < args.context_length,
        "corpus_sha256": hashlib.sha256(corpus().encode()).hexdigest(),
        "spatial_sampling_shared_across_relations": True,
    }
    plan_path = args.output_dir / "collection_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("resume configuration differs from collection_plan.json")
    write_json_atomic(plan_path, plan)
    write_json_atomic(progress_path, {"completed": sorted(completed), "planned_units": len(ratios) * math.ceil(args.num_contexts / args.batch_size)})

    for iteration, ratio in enumerate(ratios):
        for start in range(0, args.num_contexts, args.batch_size):
            indices = list(range(start, min(args.num_contexts, start + args.batch_size)))
            unit = f"iteration-{iteration}:contexts-{indices[0]}-{indices[-1]}"
            if unit in completed:
                print(f"already complete: {unit}")
                continue
            masked = clean[indices].clone()
            masked[priority[indices] < ratio] = mask_id
            selected_ids = [sample_ids[index] for index in indices]
            context = TraceContext(
                sample_ids=selected_ids,
                request_ids=selected_ids,
                seeds=[args.seed + index for index in indices],
                sequence_length=args.context_length,
                diffusion_block=0,
                denoising_iteration=iteration,
                remaining_mask_ratios=[ratio] * len(indices),
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
            with IncrementalTraceCollector(
                args.output_dir,
                context,
                sample_fraction=args.sample_fraction,
                shard_records=args.shard_records,
                q_block_size=args.q_block_size,
                kv_block_size=args.kv_block_size,
                device_buffer_tiles=args.device_buffer_tiles,
                bitpack_masks=args.bitpack_masks,
                quantize_log_scores=args.quantize_log_scores,
            ) as collector:
                with install_blasst(
                    model,
                    blasst_lambda=schedule.threshold(ratio),
                    q_block_size=args.q_block_size,
                    kv_block_size=args.kv_block_size,
                    tile_trace_callback=collector,
                    tile_trace_filter=collector.wants_tile,
                ):
                    with torch.inference_mode():
                        model(input_ids=masked.cuda())
            completed.add(unit)
            write_json_atomic(
                progress_path,
                {"completed": sorted(completed), "planned_units": len(ratios) * math.ceil(args.num_contexts / args.batch_size)},
            )
            print(f"collected {unit} remaining_mask_ratio={ratio}")


if __name__ == "__main__":
    main()
