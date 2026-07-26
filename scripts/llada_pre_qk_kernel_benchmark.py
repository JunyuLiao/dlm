#!/usr/bin/env python3
"""Three-way LLaDA benchmark: dense, sparse BLASST, and pre-QK BLASST.

The previous-step bootstrap is excluded from target-step latency because it is
the ordinary preceding diffusion step, not extra serving work.  Counter atomics
are collected in separate forwards and never included in timed measurements.
"""

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

from blasst import (  # noqa: E402
    DiffusionLambdaSchedule,
    PreQKKernelConfig,
    get_kernel_stats,
    install_bidirectional_blasst_kernel,
    install_pre_qk_blasst_kernel,
    reset_kernel_stats,
)
from proxy.proxy_policy import ThresholdTable, load_calibrated_thresholds  # noqa: E402
from llada_blasst_calibrate import build_masked_batch  # noqa: E402
from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def timed_forward(model: torch.nn.Module, inputs: torch.Tensor, repeats: int):
    times: list[float] = []
    logits = None
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
        "samples_ms": times,
    }


def agreement(predictions: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected = mask.bool()
    return {
        "masked_top1_agreement": float((predictions[selected] == reference[selected]).float().mean()),
        "all_token_top1_agreement": float((predictions == reference).float().mean()),
        "masked_tokens": int(selected.sum()),
        "all_tokens": predictions.numel(),
    }


def make_nested_states(tokenizer, context_length: int, contexts: int, mask_id: int, source: float, target: float):
    if source < target:
        raise ValueError("source mask ratio must be >= target mask ratio for a denoising transition")
    clean, _, _, _ = build_masked_batch(tokenizer, context_length, [target], mask_id, contexts)
    previous_rows, target_rows, target_masks = [], [], []
    source_count = round(source * context_length)
    target_count = round(target * context_length)
    for row_index, row in enumerate(clean):
        order = torch.randperm(context_length, generator=torch.Generator().manual_seed(20260716 + row_index))
        source_mask = torch.zeros(context_length, dtype=torch.bool)
        target_mask = torch.zeros_like(source_mask)
        source_mask[order[:source_count]] = True
        target_mask[order[:target_count]] = True
        previous, current = row.clone(), row.clone()
        previous[source_mask], current[target_mask] = mask_id, mask_id
        previous_rows.append(previous)
        target_rows.append(current)
        target_masks.append(target_mask)
    return torch.stack(previous_rows), torch.stack(target_rows), torch.stack(target_masks)


def stats_dict(stats) -> dict[str, float | int]:
    return {
        "skipped_tiles_after_or_before_qk": stats.skipped_tiles,
        "total_tiles": stats.total_tiles,
        "downstream_tile_skip_rate": stats.sparsity_ratio,
        "pre_qk_skipped_tiles": stats.pre_qk_skipped_tiles,
        "proxy_metadata_tiles": stats.proxy_metadata_tiles,
        "qk_avoidance_rate": stats.pre_qk_skip_ratio,
        "skipped_v_loads": stats.skipped_v_loads,
        "v_load_skip_rate": stats.v_load_skip_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--source-mask-ratio", type=float, default=0.9)
    parser.add_argument("--target-mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--thresholds", type=pathlib.Path,
        default=ROOT / "artifacts/proxy_calibration_v3/policy_summary.json",
    )
    parser.add_argument("--false-skip-budget", type=float, default=1e-3)
    parser.add_argument(
        "--manual-proxy-threshold", type=float,
        help="uncertified global score threshold for kernel ablation only",
    )
    parser.add_argument("--warmup-tiles", type=int, default=2)
    parser.add_argument("--periodic-refresh", type=int, default=8)
    parser.add_argument("--no-local-anchor", action="store_true")
    parser.add_argument("--no-sink-anchor", action="store_true")
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "outputs/llada_pre_qk_kernel_h100.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model/kernel latency benchmarking")
    if args.repeats < 3:
        raise ValueError("use at least three repeats for a stable median")

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
    previous, current, target_mask = make_nested_states(
        tokenizer, args.context_length, args.num_contexts,
        choose_mask_token_id(tokenizer, None), args.source_mask_ratio, args.target_mask_ratio,
    )
    previous, current, target_mask = previous.cuda(), current.cuda(), target_mask.cuda()
    configured_heads = getattr(config, "n_heads", None) or getattr(config, "num_attention_heads", None)
    if configured_heads is None:
        raise ValueError("model config does not expose its attention-head count")
    heads = int(configured_heads)
    schedule = DiffusionLambdaSchedule()
    sparse_lambda = schedule.threshold(args.target_mask_ratio)

    if args.manual_proxy_threshold is not None:
        if args.manual_proxy_threshold <= 0:
            raise ValueError("manual proxy threshold must be positive")
        from blasst.pre_qk import noise_bucket
        source_bucket, target_bucket = noise_bucket(args.source_mask_ratio), noise_bucket(args.target_mask_ratio)
        thresholds = ThresholdTable({
            (-1, -1, source_bucket, target_bucket, math.ceil(args.context_length / 64)):
                args.manual_proxy_threshold
        })
        threshold_provenance = "manual_uncertified"
    else:
        thresholds = ThresholdTable({})
        threshold_provenance = "safe_disabled_no_threshold_file"
        if args.thresholds.exists():
            threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
            if threshold_payload.get("deployment_schema_eligible", False):
                calibrated_safety = threshold_payload.get("safety_filter", {})
                requested_safety = {
                    "warmup_tiles": args.warmup_tiles,
                    "periodic_refresh": args.periodic_refresh,
                    "anchor_local": not args.no_local_anchor,
                    "anchor_sink": not args.no_sink_anchor,
                }
                for key, value in requested_safety.items():
                    if calibrated_safety.get(key) != value:
                        raise ValueError(f"runtime safety setting {key} must match threshold calibration")
                if calibrated_safety.get("anchor_diagonal", False) and args.no_local_anchor:
                    raise ValueError("calibration protected diagonal tiles but runtime local protection is disabled")
                if args.local_radius != 1:
                    raise ValueError("certified trace calibration currently requires local_radius=1")
                thresholds = load_calibrated_thresholds(
                    args.thresholds, relation="previous_step", false_skip_budget=args.false_skip_budget
                )
                threshold_provenance = (
                    str(args.thresholds) if thresholds.values
                    else "safe_disabled_no_certified_threshold_for_budget"
                )
            else:
                threshold_provenance = "safe_disabled_legacy_or_ineligible_calibration"

    # Compiled dense FlashAttention.
    for _ in range(args.warmup):
        timed_forward(model, current, 1)
    dense_logits, dense_timing = timed_forward(model, current, args.repeats)
    dense_predictions = dense_logits.argmax(-1)
    del dense_logits

    # Existing sparse baseline: no proxy metadata load/store or early gate.
    with install_bidirectional_blasst_kernel(
        model, blasst_lambda=sparse_lambda, collect_stats=False,
        num_warps=args.num_warps, pipeline_stages=args.pipeline_stages,
    ):
        for _ in range(args.warmup):
            timed_forward(model, current, 1)
        sparse_logits, sparse_timing = timed_forward(model, current, args.repeats)
    sparse_predictions = sparse_logits.argmax(-1)
    del sparse_logits
    reset_kernel_stats(current.device)
    with install_bidirectional_blasst_kernel(
        model, blasst_lambda=sparse_lambda, collect_stats=True,
        num_warps=args.num_warps, pipeline_stages=args.pipeline_stages,
    ):
        timed_forward(model, current, 1)
    sparse_stats = get_kernel_stats(reset=True)

    def run_specialized(enable_pre_skipping: bool, collect_stats: bool = False):
        feature = PreQKKernelConfig(
            use_pre_qk_kernel=True,
            enable_pre_skipping=enable_pre_skipping,
            warmup_tiles=args.warmup_tiles,
            periodic_refresh=args.periodic_refresh,
            anchor_local=not args.no_local_anchor,
            anchor_sink=not args.no_sink_anchor,
            local_radius=args.local_radius,
        )
        with install_pre_qk_blasst_kernel(
            model, batch=current.shape[0], sequence_length=current.shape[1], heads=heads,
            device=current.device, config=feature, thresholds=thresholds, schedule=schedule,
            collect_stats=collect_stats, num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
        ) as controller:
            # Source step: ordinary exact-QK BLASST plus metadata production.
            controller.set_remaining_mask_ratio(args.source_mask_ratio)
            timed_forward(model, previous, 1)
            controller.commit_step()
            controller.set_remaining_mask_ratio(args.target_mask_ratio)
            if collect_stats:
                reset_kernel_stats(current.device)
                timed_forward(model, current, 1)
                return None, None, get_kernel_stats(reset=True)
            for _ in range(args.warmup):
                timed_forward(model, current, 1)
            logits, timing = timed_forward(model, current, args.repeats)
            return logits, timing, None

    kernel_only_logits, kernel_only_timing, _ = run_specialized(False)
    assert kernel_only_logits is not None and kernel_only_timing is not None
    kernel_only_predictions = kernel_only_logits.argmax(-1)
    del kernel_only_logits

    pre_logits, pre_timing, _ = run_specialized(True)
    assert pre_logits is not None and pre_timing is not None
    pre_predictions = pre_logits.argmax(-1)
    del pre_logits
    _, _, pre_stats = run_specialized(True, collect_stats=True)
    assert pre_stats is not None

    dense_ms = dense_timing["median_ms"]
    sparse_ms = sparse_timing["median_ms"]
    kernel_only_ms = kernel_only_timing["median_ms"]
    pre_ms = pre_timing["median_ms"]
    result = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(current.device),
        "context_length": args.context_length,
        "batch": args.num_contexts,
        "transition": {"source_mask_ratio": args.source_mask_ratio, "target_mask_ratio": args.target_mask_ratio},
        "threshold_provenance": threshold_provenance,
        "active_certified_or_manual_thresholds": len(thresholds.values),
        "false_skip_budget": args.false_skip_budget if args.manual_proxy_threshold is None else None,
        "exact_blasst_lambda": sparse_lambda,
        "safety": {
            "warmup_tiles": args.warmup_tiles, "periodic_refresh": args.periodic_refresh,
            "anchor_local": not args.no_local_anchor, "anchor_sink": not args.no_sink_anchor,
            "local_radius": args.local_radius, "non_cascading_unknown_sentinel": "+inf",
        },
        "dense_baseline": {"timing": dense_timing},
        "sparse_baseline_no_pre_qk": {
            "timing": sparse_timing,
            "speedup_vs_dense": dense_ms / sparse_ms,
            "stats": stats_dict(sparse_stats),
            "agreement_vs_dense": agreement(sparse_predictions, dense_predictions, target_mask),
        },
        "pre_qk_kernel_gate_disabled": {
            "timing": kernel_only_timing,
            "speedup_vs_dense": dense_ms / kernel_only_ms,
            "speedup_vs_sparse": sparse_ms / kernel_only_ms,
            "agreement_vs_dense": agreement(kernel_only_predictions, dense_predictions, target_mask),
            "agreement_vs_sparse": agreement(kernel_only_predictions, sparse_predictions, target_mask),
        },
        "pre_qk_enabled": {
            "timing": pre_timing,
            "speedup_vs_dense": dense_ms / pre_ms,
            "speedup_vs_sparse": sparse_ms / pre_ms,
            "speedup_vs_same_kernel_gate_disabled": kernel_only_ms / pre_ms,
            "stats": stats_dict(pre_stats),
            "agreement_vs_dense": agreement(pre_predictions, dense_predictions, target_mask),
            "agreement_vs_sparse": agreement(pre_predictions, sparse_predictions, target_mask),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
