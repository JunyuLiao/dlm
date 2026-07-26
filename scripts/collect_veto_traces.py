#!/usr/bin/env python3
"""Collect sampled rich physical-tile traces for veto-row pruning studies."""

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
from collect_proxy_traces import config_heads, corpus, cyclic_contexts, write_json_atomic
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)
from tracing import VetoTraceCollector, VetoTraceContext


def token_states(
    priority: torch.Tensor,
    ratio: float,
    previous_ratio: float | None,
    *,
    prefix_length: int = 0,
) -> torch.Tensor:
    """Return masked/newly/previously-visible/prefix state codes."""
    masked = priority < ratio
    states = torch.full(priority.shape, 2, dtype=torch.uint8)
    states[masked] = 0
    if previous_ratio is not None:
        states[(~masked) & (priority < previous_ratio)] = 1
    if prefix_length:
        states[:, :prefix_length] = 4
    return states


def split_for_context(index: int, total: int) -> str:
    calibration_end = total // 3
    heldout_end = 2 * total // 3
    if index < calibration_end:
        return "calibration"
    if index < heldout_end:
        return "heldout"
    return "final_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-ratios", default="0.90,0.89,0.50,0.49,0.15,0.14")
    parser.add_argument("--prefix-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--sample-fraction", type=float, default=0.005)
    parser.add_argument(
        "--query-tiles-per-context",
        type=int,
        default=1,
        help="trace every KV tile for this many deterministic Q tiles per context",
    )
    parser.add_argument("--stable-query-delta", type=float, default=0.05)
    parser.add_argument("--shard-records", type=int, default=25000)
    parser.add_argument("--device-buffer-tiles", type=int, default=32)
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=ROOT / "artifacts" / "veto_traces"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("veto-row trace collection requires CUDA")
    ratios = [float(value) for value in args.mask_ratios.split(",")]
    if ratios != sorted(ratios, reverse=True):
        raise ValueError("mask ratios must be ordered from high to low")
    if args.batch_size <= 0 or args.batch_size > args.num_contexts:
        raise ValueError("batch size must be in [1, num_contexts]")
    if not 0 <= args.prefix_length <= args.context_length:
        raise ValueError("prefix length must be within the context")

    progress_path = args.output_dir / "progress.json"
    existing = list(args.output_dir.glob("veto-trace-*.npz")) if args.output_dir.exists() else []
    if existing and not progress_path.exists():
        raise ValueError("trace shards exist without progress.json; use a fresh directory")
    if progress_path.exists() and not args.resume:
        raise ValueError("trace collection exists; pass --resume or use a fresh directory")
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"completed": []}
    )
    completed = set(progress["completed"])

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > config.max_sequence_length:
        raise ValueError("context length exceeds model maximum")
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = corpus()
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    clean = cyclic_contexts(tokens, args.context_length, args.num_contexts)
    mask_id = choose_mask_token_id(tokenizer, None)
    query_heads, _ = config_heads(config)
    schedule = DiffusionLambdaSchedule()

    priority = torch.stack(
        [
            torch.rand(
                args.context_length,
                generator=torch.Generator().manual_seed(args.seed + index),
            )
            for index in range(args.num_contexts)
        ]
    )
    sample_ids = [f"context-{index:03d}" for index in range(args.num_contexts)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": args.model,
        "context_length": args.context_length,
        "num_contexts": args.num_contexts,
        "batch_size": args.batch_size,
        "mask_ratios": ratios,
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "query_tiles_per_context": args.query_tiles_per_context,
        "stable_query_delta": args.stable_query_delta,
        "prefix_length": args.prefix_length,
        "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "context_splits": {
            sample_ids[index]: split_for_context(index, args.num_contexts)
            for index in range(args.num_contexts)
        },
    }
    plan_path = args.output_dir / "collection_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("resume configuration differs from collection_plan.json")
    write_json_atomic(plan_path, plan)
    planned_units = len(ratios) * math.ceil(args.num_contexts / args.batch_size)
    write_json_atomic(
        progress_path,
        {"completed": sorted(completed), "planned_units": planned_units},
    )

    # Context batches are outermost so the query cache follows an actual
    # denoising trajectory.  Completed earlier ratios are replayed on resume to
    # reconstruct the cache before a later traced state.
    for start in range(0, args.num_contexts, args.batch_size):
        indices = list(range(start, min(args.num_contexts, start + args.batch_size)))
        selected_ids = [sample_ids[index] for index in indices]
        query_history: dict[int, torch.Tensor] = {}
        previous_ratio: float | None = None
        for step, ratio in enumerate(ratios):
            unit = f"contexts-{indices[0]}-{indices[-1]}:step-{step}:ratio-{ratio:.4f}"
            states = token_states(
                priority[indices], ratio, previous_ratio, prefix_length=args.prefix_length
            )
            masked = clean[indices].clone()
            masked[states == 0] = mask_id
            threshold = float(schedule.threshold(ratio))
            if unit in completed:
                with install_blasst(
                    model,
                    blasst_lambda=threshold,
                    q_block_size=args.q_block_size,
                    kv_block_size=args.kv_block_size,
                    query_history=query_history,
                ):
                    with torch.inference_mode():
                        model(input_ids=masked.cuda())
                print(f"replayed completed {unit}")
                previous_ratio = ratio
                continue
            context = VetoTraceContext(
                sample_ids=selected_ids,
                request_ids=selected_ids,
                seeds=[args.seed + index for index in indices],
                sequence_length=args.context_length,
                diffusion_block=0,
                denoising_step=step,
                remaining_mask_ratios=[ratio] * len(indices),
                thresholds=[threshold] * len(indices),
                token_states=states,
                query_heads=query_heads,
            )
            with VetoTraceCollector(
                args.output_dir,
                context,
                sample_fraction=args.sample_fraction,
                query_tiles_per_sample=args.query_tiles_per_context,
                q_block_size=args.q_block_size,
                kv_block_size=args.kv_block_size,
                shard_records=args.shard_records,
                device_buffer_tiles=args.device_buffer_tiles,
                stable_query_delta=args.stable_query_delta,
            ) as collector:
                with install_blasst(
                    model,
                    blasst_lambda=threshold,
                    q_block_size=args.q_block_size,
                    kv_block_size=args.kv_block_size,
                    rich_tile_trace_callback=collector,
                    tile_trace_filter=collector.wants_tile,
                    query_history=query_history,
                ):
                    with torch.inference_mode():
                        model(input_ids=masked.cuda())
            completed.add(unit)
            write_json_atomic(
                progress_path,
                {"completed": sorted(completed), "planned_units": planned_units},
            )
            print(f"collected {unit}")
            previous_ratio = ratio


if __name__ == "__main__":
    main()
