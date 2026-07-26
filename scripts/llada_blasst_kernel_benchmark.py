#!/usr/bin/env python3
"""Benchmark the fused bidirectional BLASST kernel inside LLaDA."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

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
from blasst.regrouping import pack_keep_mask
from llada_blasst_calibrate import build_masked_batch
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def timed_forward(model, inputs: torch.Tensor, repeats: int) -> tuple[torch.Tensor, dict[str, float]]:
    times, logits = [], None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            logits = model(input_ids=inputs).logits
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    assert logits is not None
    ordered = sorted(times)
    return logits, {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def stats_payload(stats) -> dict[str, float | int]:
    return {
        "skipped_parent_tiles": stats.skipped_tiles,
        "total_parent_tiles": stats.total_tiles,
        "full_parent_tile_skip_rate": stats.sparsity_ratio,
        "skipped_microgroups": stats.skipped_microgroups,
        "total_microgroups": stats.total_microgroups,
        "microgroup_skip_rate": stats.microgroup_sparsity_ratio,
        "skipped_bmm2_groups": stats.skipped_bmm2_groups,
        "total_bmm2_groups": stats.total_bmm2_groups,
        "bmm2_flop_skip_rate": stats.bmm2_flop_skip_ratio,
        "skipped_v_loads": stats.skipped_v_loads,
        "total_v_loads": stats.total_v_loads,
        "v_load_skip_rate": stats.v_load_skip_ratio,
        "dense_fallbacks": stats.dense_fallbacks,
        "partial_tiles": stats.partial_tiles,
    }


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


def build_nested_masked_batch(tokenizer, context_length, ratios, mask_id, num_contexts):
    """Build request trajectories whose masked positions monotonically reveal."""
    clean, _, _, _ = build_masked_batch(
        tokenizer, context_length, [ratios[0]], mask_id, num_contexts
    )
    clean = clean.reshape(num_contexts, context_length)
    corrupted, masks, metadata = [], [], []
    descending = sorted(set(ratios), reverse=True)
    step_by_ratio = {ratio: step for step, ratio in enumerate(descending)}
    for context_index, row in enumerate(clean):
        generator = torch.Generator().manual_seed(20260719 + context_index)
        ranks = torch.rand(context_length, generator=generator)
        for ratio in ratios:
            mask = ranks < ratio
            changed = row.clone()
            changed[mask] = mask_id
            corrupted.append(changed)
            masks.append(mask)
            metadata.append({
                "context_index": context_index,
                "request_id": f"context-{context_index}",
                "denoising_step": step_by_ratio[ratio],
                "mask_ratio": ratio,
            })
    return torch.stack(corrupted), torch.stack(masks), metadata


def build_adjacent_masked_batch(tokenizer, context_length, pairs, mask_id, num_contexts):
    """Build independent two-step request trajectories around noise regimes."""
    clean, _, _, _ = build_masked_batch(tokenizer, context_length, [pairs[0][0]], mask_id, num_contexts)
    clean = clean.reshape(num_contexts, context_length)
    corrupted, masks, metadata = [], [], []
    for context_index, row in enumerate(clean):
        for pair_index, pair in enumerate(pairs):
            generator = torch.Generator().manual_seed(20260729 + context_index * 100 + pair_index)
            ranks = torch.rand(context_length, generator=generator)
            for step, ratio in enumerate(pair):
                mask = ranks < ratio
                changed = row.clone(); changed[mask] = mask_id
                corrupted.append(changed); masks.append(mask)
                metadata.append({
                    "context_index": context_index,
                    "request_id": f"context-{context_index}-regime-{pair_index}",
                    "denoising_step": step,
                    "mask_ratio": ratio,
                })
    return torch.stack(corrupted), torch.stack(masks), metadata


def dump_blasst_row_masks(model, inputs, masks, metadata, path, num_warps, pipeline_stages):
    schedule = DiffusionLambdaSchedule()
    thresholds = torch.tensor(
        [schedule.threshold(item["mask_ratio"]) for item in metadata],
        dtype=torch.float32,
        device=inputs.device,
    )
    captured = {}

    def collect(layer, keep_mask):
        captured[layer] = keep_mask.detach().cpu()

    with install_bidirectional_blasst_kernel(
        model,
        blasst_lambda=thresholds,
        collect_stats=False,
        num_warps=num_warps,
        pipeline_stages=pipeline_stages,
        row_mask_callback=collect,
    ):
        with torch.inference_mode():
            model(input_ids=inputs).logits
    records = []
    for layer, layer_masks in sorted(captured.items()):
        for batch_index, item in enumerate(metadata):
            records.append({
                **item,
                "layer_id": layer,
                "sequence_length": inputs.shape[1],
                "remaining_mask_ratio": item["mask_ratio"],
                "query_tile_size": 128,
                "kv_tile_size": 64,
                "num_kv_tiles": layer_masks.shape[-1],
                "threshold": float(thresholds[batch_index]),
                "keep_mask_packed": pack_keep_mask(layer_masks[batch_index]),
                "mask_state": masks[batch_index].cpu(),
                "original_row_order": torch.arange(inputs.shape[1], dtype=torch.int32),
            })
    payload = {
        "format": "blasst-row-masks-v1",
        "model": getattr(model.config, "_name_or_path", "unknown"),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_mib = path.stat().st_size / (1024 * 1024)
    print(json.dumps({"row_mask_dump": str(path), "records": len(records), "size_mib": size_mib}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--mask-ratios", default="0.15,0.5,0.9")
    parser.add_argument("--lambdas", default="0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--skip-lambda-sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "blasst_llada_kernel_4096.json")
    parser.add_argument(
        "--dump-blasst-row-masks",
        type=pathlib.Path,
        nargs="?",
        const=ROOT / "outputs" / "query_regrouping" / "row_masks.pt",
        help="opt-in compact row-mask dump path; uses nested masks for temporal analysis",
    )
    parser.add_argument("--dump-only", action="store_true", help="exit after the row-mask trace forward")
    parser.add_argument(
        "--dump-temporal-pairs",
        default="",
        help="optional comma-separated adjacent trajectories, e.g. 0.90:0.89,0.50:0.49",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model latency benchmarking")

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
    if args.dump_blasst_row_masks is not None:
        if args.dump_temporal_pairs:
            pairs = [tuple(float(value) for value in item.split(":")) for item in args.dump_temporal_pairs.split(",")]
            if any(len(pair) != 2 or pair[1] > pair[0] for pair in pairs):
                raise ValueError("temporal pairs must be reveal trajectories high:low")
            inputs, masks, metadata = build_adjacent_masked_batch(
                tokenizer, args.context_length, pairs, choose_mask_token_id(tokenizer, None), args.num_contexts
            )
        else:
            inputs, masks, metadata = build_nested_masked_batch(
                tokenizer, args.context_length, ratios, choose_mask_token_id(tokenizer, None), args.num_contexts
            )
    else:
        _, inputs, masks, metadata = build_masked_batch(
            tokenizer,
            args.context_length,
            ratios,
            choose_mask_token_id(tokenizer, None),
            args.num_contexts,
        )
    inputs, masks = inputs.cuda(), masks.cuda()

    if args.dump_blasst_row_masks is not None:
        dump_blasst_row_masks(
            model, inputs, masks, metadata, args.dump_blasst_row_masks,
            args.num_warps, args.pipeline_stages,
        )
        if args.dump_only:
            return

    # The model-bound flash_attn_func is the installed compiled dense baseline.
    for _ in range(args.warmup):
        timed_forward(model, inputs, 1)
    dense_logits, dense_timing = timed_forward(model, inputs, args.repeats)
    dense_ms = dense_timing["median_ms"]
    dense_predictions = dense_logits.argmax(-1)
    del dense_logits

    # Lambda zero uses identical fused-kernel arithmetic without pruning.
    reset_kernel_stats(inputs.device)
    with install_bidirectional_blasst_kernel(model, blasst_lambda=0.0, collect_stats=False):
        for _ in range(args.warmup):
            timed_forward(model, inputs, 1)
        zero_logits, zero_timing = timed_forward(model, inputs, args.repeats)
    zero_ms = zero_timing["median_ms"]
    zero_predictions = zero_logits.argmax(-1)
    del zero_logits

    rows = []
    thresholds = [] if args.skip_lambda_sweep else [float(value) for value in args.lambdas.split(",")]
    for threshold in thresholds:
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
            logits, timing = timed_forward(model, inputs, args.repeats)
        elapsed_ms = timing["median_ms"]
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
                "timing": timing,
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
        scheduled_logits, scheduled_timing = timed_forward(model, inputs, args.repeats)
    scheduled_ms = scheduled_timing["median_ms"]
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

    # First-step production fallback plans: no current-step QK oracle. Query
    # grouping uses only current mask state; KV priority uses visible fraction.
    torch.cuda.synchronize()
    result = {
        "model": args.model,
        "context_length": args.context_length,
        "mask_ratios": ratios,
        "num_contexts": args.num_contexts,
        "kernel": "BF16 bidirectional 2D BLASST, Q tile 128, KV tile 64, reverse traversal, base-2 softmax",
        "launch": {"num_warps": args.num_warps, "pipeline_stages": args.pipeline_stages},
        "dense_flash_ms": dense_ms,
        "dense_flash_timing": dense_timing,
        "kernel_lambda_zero_ms": zero_ms,
        "kernel_lambda_zero_timing": zero_timing,
        "kernel_zero_agreement_with_compiled_dense": agreement_by_noise(
            zero_predictions, dense_predictions, masks, metadata
        ),
        "measurements": rows,
        "diffusion_schedule": {
            "per_sequence_lambdas": scheduled_lambdas.cpu().tolist(),
            "latency_ms": scheduled_ms,
            "timing": scheduled_timing,
            "speedup_vs_dense_flash": dense_ms / scheduled_ms,
            "skipped_2d_tiles_per_forward": scheduled_stats.skipped_tiles,
            "total_2d_tiles_per_forward": scheduled_stats.total_tiles,
            "physical_tile_sparsity": scheduled_stats.sparsity_ratio,
            "stats": stats_payload(scheduled_stats),
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
