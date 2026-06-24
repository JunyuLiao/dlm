#!/usr/bin/env python3
"""Compare padded heterogeneous batching vs flash-decoding-style micro-batches.

Baseline (normal probe path):
  - One padded batch with mixed prompt lengths (short + long requests).
  - flash_attention=False, use_cache=False (LLaDA MDM default).

Flash-decoding pipeline:
  - Split the heterogeneous batch into bs=1 micro-batches at each request's true length.
  - No cross-request padding, so short prompts avoid attending over pad tokens.
  - flash_attention=True so attention uses flash_attn kernels; inside flash-attn,
    num_splits=0 lets the library pick split-KV parallelism along sequence length.

Note: LLaDA-8B asserts KV cache is unsupported for MDM, so this is not the
autoregressive flash-decoding path (q_len=1 + KV cache). modeling_llada.py calls
flash_attn_func for full bidirectional attention; it does not pass num_splits and
is not flash_attn_with_kvcache. On H100, that path is often close to torch SDPA.
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
    append_mask_block,
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    get_num_transfer_tokens,
    patch_llada_transformers_compat,
)
from llada_length_heterogeneity_probe import (
    LengthPromptRecord,
    build_length_prompt_batch,
    length_targets,
    probe_length_batch,
)


@dataclass(frozen=True)
class RequestStats:
    request_id: int
    prompt_tokens: int
    seq_len_with_block: int
    denoise_steps: int
    forward_calls: int
    total_ms: float
    forward_ms_mean: float


@dataclass(frozen=True)
class RunStats:
    label: str
    flash_attention: bool
    batch_size: int
    padded_seq_len: int
    pad_waste_tokens_per_forward: int
    forward_calls: int
    forward_ms_mean: float
    total_ms: float
    flash_attn_calls: int
    sdpa_calls: int
    attn_token_positions: int
    request_stats: tuple[RequestStats, ...] = ()


@dataclass
class AttentionPathCounts:
    flash_attn_calls: int = 0
    sdpa_calls: int = 0


def count_flash_ready_layers(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "flash_attn_func") and getattr(module, "flash_attn_func", None) is not None
    )


def install_attention_counters(model: torch.nn.Module) -> AttentionPathCounts:
    counts = AttentionPathCounts()
    originals: list[tuple[object, object]] = []

    for module in model.modules():
        if not hasattr(module, "_scaled_dot_product_attention"):
            continue
        original = module._scaled_dot_product_attention

        def wrapped(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            _original: object = original,
            _counts: AttentionPathCounts = counts,
            _module: object = module,
        ) -> torch.Tensor:
            flash_fn = getattr(_module, "flash_attn_func", None)
            if flash_fn is not None and attn_mask is None:
                _counts.flash_attn_calls += 1
            else:
                _counts.sdpa_calls += 1
            return _original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        module._scaled_dot_product_attention = wrapped  # type: ignore[method-assign]
        originals.append((module, original))

    counts._originals = originals  # type: ignore[attr-defined]
    return counts


def restore_attention_counters(counts: AttentionPathCounts) -> None:
    for module, original in getattr(counts, "_originals", []):
        module._scaled_dot_product_attention = original  # type: ignore[method-assign]


def load_model(model_id: str, flash_attention: bool, dtype: torch.dtype, device: torch.device) -> AutoModelForCausalLM:
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


def pad_waste_per_forward(prompt_lengths: list[int], padded_seq_len: int) -> int:
    return sum(padded_seq_len - length for length in prompt_lengths)


def attn_cost_token_positions(batch_size: int, seq_len: int) -> int:
    """Rough attention cost proxy: batch_size * seq_len^2 per forward."""
    return batch_size * seq_len * seq_len


def accept_block_tokens(
    token_steps_row: torch.Tensor,
    block_logits: torch.Tensor,
    block_start: int,
    block_end: int,
    step: int,
    confidence_threshold: float,
    acceptance_policy: str,
    num_transfer_tokens: torch.Tensor,
    request_id: int,
    input_ids_row: torch.Tensor,
) -> int:
    masked_positions = token_steps_row.eq(0)
    if not bool(masked_positions.any()):
        return 0

    probs = torch.softmax(block_logits, dim=-1)
    confidence, predicted = torch.max(probs, dim=-1)
    force_accept = torch.zeros_like(masked_positions)
    threshold_accept = torch.zeros_like(masked_positions)
    masked_conf = confidence.masked_fill(~masked_positions, -float("inf"))
    if acceptance_policy == "topk":
        k = int(num_transfer_tokens[request_id, step - 1].item())
        k = min(k, int(masked_positions.sum().item()))
        if k > 0:
            _, select_index = torch.topk(masked_conf, k=k)
            threshold_accept[select_index] = True
        accept = threshold_accept
    elif acceptance_policy == "confidence_cutoff":
        sorted_conf, sorted_index = torch.sort(masked_conf, descending=True)
        keep = sorted_conf.ge(confidence_threshold)
        if bool(keep.any()):
            threshold_accept[sorted_index[keep]] = True
        accept = threshold_accept.clone()
        if not bool(accept.any()):
            force_accept[torch.argmax(masked_conf)] = True
            accept = force_accept
    else:
        threshold_accept = masked_positions & confidence.ge(confidence_threshold)
        accept = threshold_accept.clone()
        if not bool(accept.any()):
            force_accept[torch.argmax(masked_conf)] = True
            accept = force_accept
    if not bool(accept.any()):
        return 0

    device = input_ids_row.device
    absolute_positions = torch.arange(block_start, block_end, device=device)[accept]
    input_ids_row[absolute_positions] = predicted[accept]
    token_steps_row[accept] = step
    return int(accept.sum().item())


@torch.no_grad()
def probe_flash_decoding_microbatches(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[LengthPromptRecord],
    block_size: int,
    num_blocks: int,
    max_steps: int,
    confidence_threshold: float,
    acceptance_policy: str,
    mask_token_id: int,
    device: torch.device,
) -> tuple[int, float, int, int, int, int, list[RequestStats]]:
    """Run one bs=1 micro-batch per request at native length (no cross-request padding)."""
    counts = install_attention_counters(model)
    forward_calls = 0
    forward_ms_total = 0.0
    max_padded = 0
    attn_token_positions = 0
    request_stats: list[RequestStats] = []

    try:
        for prompt in prompts:
            input_ids = torch.tensor(
                [tokenizer(prompt.prompt, add_special_tokens=True)["input_ids"]],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            max_padded = max(max_padded, int(input_ids.shape[1]))
            request_forward_calls = 0
            request_ms = 0.0
            denoise_steps = 0

            for _block_index in range(num_blocks):
                input_ids, attention_mask, (block_start, block_end) = append_mask_block(
                    input_ids,
                    attention_mask,
                    block_size,
                    mask_token_id,
                )
                token_steps = torch.zeros((1, block_size), dtype=torch.long, device=device)
                block_mask_index = torch.ones((1, block_size), dtype=torch.bool, device=device)
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, max_steps)

                for step in range(1, max_steps + 1):
                    if not bool(token_steps.eq(0).any()):
                        break

                    seq_len = int(input_ids.shape[1])
                    attn_token_positions += attn_cost_token_positions(1, seq_len)

                    cuda_sync()
                    start = time.perf_counter()
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    cuda_sync()
                    step_ms = (time.perf_counter() - start) * 1000.0
                    forward_calls += 1
                    request_forward_calls += 1
                    forward_ms_total += step_ms
                    request_ms += step_ms
                    denoise_steps = step

                    accept_block_tokens(
                        token_steps[0],
                        logits[0, block_start:block_end],
                        block_start,
                        block_end,
                        step,
                        confidence_threshold,
                        acceptance_policy,
                        num_transfer_tokens,
                        0,
                        input_ids[0],
                    )

            request_stats.append(
                RequestStats(
                    request_id=prompt.request_id,
                    prompt_tokens=prompt.actual_prompt_tokens,
                    seq_len_with_block=int(input_ids.shape[1]),
                    denoise_steps=denoise_steps,
                    forward_calls=request_forward_calls,
                    total_ms=request_ms,
                    forward_ms_mean=request_ms / request_forward_calls if request_forward_calls else 0.0,
                )
            )
    finally:
        restore_attention_counters(counts)

    forward_ms_mean = forward_ms_total / forward_calls if forward_calls else 0.0
    return (
        forward_calls,
        forward_ms_mean,
        max_padded,
        counts.flash_attn_calls,
        counts.sdpa_calls,
        attn_token_positions,
        request_stats,
    )


@torch.no_grad()
def run_padded_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    targets: list[int],
    args: argparse.Namespace,
    device: torch.device,
    mask_token_id: int,
) -> RunStats:
    counts = install_attention_counters(model)
    attn_token_positions = 0
    cuda_sync()
    start = time.perf_counter()
    try:
        _rows, batch_step_rows, _step_rows, forward_calls = probe_length_batch(
            model=model,
            tokenizer=tokenizer,
            group_name=args.length_group,
            targets=targets,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            max_steps=args.max_steps,
            confidence_threshold=args.confidence_threshold,
            acceptance_policy=args.acceptance_policy,
            mask_token_id=mask_token_id,
            trial=0,
            batch_id=0,
            run_id="padded_baseline",
            device=device,
        )
        for row in batch_step_rows:
            attn_token_positions += attn_cost_token_positions(
                len(targets),
                row.padded_sequence_length,
            )
    finally:
        restore_attention_counters(counts)
    cuda_sync()
    total_ms = (time.perf_counter() - start) * 1000.0

    _input_ids, _attention_mask, prompts = build_length_prompt_batch(
        tokenizer, args.length_group, targets, trial=0, device=device
    )
    prompt_lengths = [record.actual_prompt_tokens for record in prompts]
    padded_seq_len = int(_input_ids.shape[1])
    forward_ms_mean = (
        sum(row.step_latency_ms for row in batch_step_rows) / forward_calls if forward_calls else 0.0
    )

    return RunStats(
        label="padded_baseline",
        flash_attention=bool(model.config.flash_attention),
        batch_size=len(targets),
        padded_seq_len=padded_seq_len,
        pad_waste_tokens_per_forward=pad_waste_per_forward(prompt_lengths, padded_seq_len),
        forward_calls=forward_calls,
        forward_ms_mean=forward_ms_mean,
        total_ms=total_ms,
        flash_attn_calls=counts.flash_attn_calls,
        sdpa_calls=counts.sdpa_calls,
        attn_token_positions=attn_token_positions,
    )


@torch.no_grad()
def run_flash_decoding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    targets: list[int],
    args: argparse.Namespace,
    device: torch.device,
    mask_token_id: int,
) -> RunStats:
    _input_ids, _attention_mask, prompts = build_length_prompt_batch(
        tokenizer, args.length_group, targets, trial=0, device=device
    )
    prompt_lengths = [record.actual_prompt_tokens for record in prompts]
    padded_seq_len = int(_input_ids.shape[1])

    cuda_sync()
    start = time.perf_counter()
    forward_calls, forward_ms_mean, _max_native, flash_calls, sdpa_calls, attn_cost, request_stats = (
        probe_flash_decoding_microbatches(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            max_steps=args.max_steps,
            confidence_threshold=args.confidence_threshold,
            acceptance_policy=args.acceptance_policy,
            mask_token_id=mask_token_id,
            device=device,
        )
    )
    cuda_sync()
    total_ms = (time.perf_counter() - start) * 1000.0

    return RunStats(
        label="flash_decoding_microbatch",
        flash_attention=bool(model.config.flash_attention),
        batch_size=len(targets),
        padded_seq_len=padded_seq_len,
        pad_waste_tokens_per_forward=0,
        forward_calls=forward_calls,
        forward_ms_mean=forward_ms_mean,
        total_ms=total_ms,
        flash_attn_calls=flash_calls,
        sdpa_calls=sdpa_calls,
        attn_token_positions=attn_cost,
        request_stats=tuple(request_stats),
    )


@torch.no_grad()
def time_long_request_only(
    flash_model: AutoModelForCausalLM,
    baseline_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    targets: list[int],
    group_name: str,
    args: argparse.Namespace,
    mask_token_id: int,
    device: torch.device,
) -> None:
    padded_input, padded_mask, prompts = build_length_prompt_batch(
        tokenizer, group_name, targets, trial=0, device=device
    )
    padded_input, padded_mask, _ = append_mask_block(
        padded_input, padded_mask, args.block_size, mask_token_id
    )

    long_id = max(range(len(prompts)), key=lambda i: prompts[i].actual_prompt_tokens)
    long_prompt = prompts[long_id]
    native_ids = torch.tensor(
        [tokenizer(long_prompt.prompt, add_special_tokens=True)["input_ids"]],
        dtype=torch.long,
        device=device,
    )
    native_mask = torch.ones_like(native_ids)
    native_ids, native_mask, _ = append_mask_block(native_ids, native_mask, args.block_size, mask_token_id)

    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(5):
        flash_model(input_ids=native_ids, attention_mask=native_mask)
    cuda_sync()
    flash_fwd_ms = (time.perf_counter() - t0) / 5 * 1000

    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(5):
        baseline_model(
            input_ids=padded_input[long_id : long_id + 1],
            attention_mask=padded_mask[long_id : long_id + 1],
        )
    cuda_sync()
    padded_fwd_ms = (time.perf_counter() - t0) / 5 * 1000

    print("\n[flash decoding] long-request-only single-forward (apples-to-apples)")
    print(f"  request_id={long_id} prompt_tokens={long_prompt.actual_prompt_tokens} seq_with_block={native_ids.shape[1]}")
    print(f"  flash_attn forward_ms={flash_fwd_ms:.2f}")
    print(f"  padded bs=1 SDPA forward_ms={padded_fwd_ms:.2f}")
    print(f"  per-forward speedup={padded_fwd_ms / flash_fwd_ms if flash_fwd_ms else float('inf'):.3f}x")


def print_request_stats(stats: RunStats) -> None:
    if not stats.request_stats:
        return
    print(f"[flash decoding] per-request breakdown ({stats.label}):")
    for row in stats.request_stats:
        print(
            f"  req={row.request_id} prompt_tokens={row.prompt_tokens} "
            f"seq_with_block={row.seq_len_with_block} denoise_steps={row.denoise_steps} "
            f"forward_calls={row.forward_calls} total_ms={row.total_ms:.2f} "
            f"forward_ms_mean={row.forward_ms_mean:.2f}"
        )


def print_stats(stats: RunStats, prompt_lengths: list[int]) -> None:
    print(
        f"[flash decoding] {stats.label}: flash_attention={stats.flash_attention} "
        f"batch_size={stats.batch_size} prompt_lengths={prompt_lengths} "
        f"padded_seq_len={stats.padded_seq_len} "
        f"pad_waste_tokens_per_forward={stats.pad_waste_tokens_per_forward} "
        f"forward_calls={stats.forward_calls} forward_ms_mean={stats.forward_ms_mean:.2f} "
        f"total_ms={stats.total_ms:.2f} attn_token_positions={stats.attn_token_positions} "
        f"flash_attn_calls={stats.flash_attn_calls} sdpa_calls={stats.sdpa_calls}"
    )
    print_request_stats(stats)


def print_extreme_analysis(
    baseline: RunStats,
    flash: RunStats,
    batch_forward_ms_mean: float,
) -> None:
    if not flash.request_stats:
        return
    short = [r for r in flash.request_stats if r.prompt_tokens < max(r.prompt_tokens for r in flash.request_stats)]
    long = max(flash.request_stats, key=lambda r: r.prompt_tokens)
    short_ms = sum(r.total_ms for r in short)
    short_forwards = sum(r.forward_calls for r in short)
    print("\n[flash decoding] extreme-case analysis")
    print(
        f"  padded baseline: {baseline.forward_calls} batched forwards "
        f"@ {batch_forward_ms_mean:.1f} ms -> {baseline.total_ms:.1f} ms total"
    )
    print(
        f"  short requests (native length): {len(short)} reqs, {short_forwards} forwards, "
        f"{short_ms:.1f} ms ({100 * short_ms / flash.total_ms:.0f}% of flash total)"
    )
    print(
        f"  long request (native length): {long.forward_calls} forwards, {long.total_ms:.1f} ms, "
        f"denoise_steps={long.denoise_steps} (same RoPE positions as padded batch)"
    )
    if short:
        print(
            f"  short denoise_steps native={ [r.denoise_steps for r in short] } vs padded baseline=3 each; "
            "mask block moves from position ~max_prompt to ~prompt_len (RoPE shift), "
            "so acceptance slows and dominates wall time."
        )
    print(
        "  flash_attn_func is full-sequence bidirectional attention, not KV-cache flash decoding; "
        "long-request per-forward gain on H100 is small (~5%)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-prompt-tokens", type=int, default=2000)
    parser.add_argument(
        "--length-group",
        choices=["moderate", "extreme"],
        default="extreme",
        help="extreme = one very long + several short prompts",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["confidence_cutoff", "topk", "threshold"], default="confidence_cutoff")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[flash decoding] ERROR: CUDA is required")
        sys.exit(1)

    try:
        import flash_attn  # type: ignore

        print(f"[flash decoding] flash_attn version: {getattr(flash_attn, '__version__', 'unknown')}")
    except ImportError:
        print("[flash decoding] WARN: flash_attn not installed; flash path will fall back to SDPA")

    targets = length_targets(args.batch_size, args.total_prompt_tokens)[args.length_group]
    print(
        f"[flash decoding] heterogeneous targets ({args.length_group}): {targets} "
        f"sum={sum(targets)} max={max(targets)} min={min(targets)}"
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_token_id = choose_mask_token_id(tokenizer, args.mask_token_id)

    print("\n[flash decoding] loading padded baseline model (flash_attention=False)")
    baseline_model = load_model(args.model, flash_attention=False, dtype=dtype, device=device)
    baseline = run_padded_baseline(baseline_model, tokenizer, targets, args, device, mask_token_id)
    print_stats(baseline, [t for t in targets])

    print("\n[flash decoding] loading flash-decoding model (flash_attention=True)")
    flash_model = load_model(args.model, flash_attention=True, dtype=dtype, device=device)
    ready = count_flash_ready_layers(flash_model)
    print(f"[flash decoding] flash_attn_func ready layers: {ready}/32")

    time_long_request_only(
        flash_model, baseline_model, tokenizer, targets, args.length_group, args, mask_token_id, device
    )

    flash = run_flash_decoding(flash_model, tokenizer, targets, args, device, mask_token_id)
    print_stats(flash, [t for t in targets])
    unload_model(baseline_model)
    unload_model(flash_model)

    speedup = baseline.total_ms / flash.total_ms if flash.total_ms > 0 else float("inf")
    print("\n[flash decoding] comparison")
    print(f"  padded baseline total_ms={baseline.total_ms:.2f}")
    print(f"  flash microbatch total_ms={flash.total_ms:.2f}")
    print(f"  speedup={speedup:.3f}x")
    print(f"  padded attn_token_positions={baseline.attn_token_positions}")
    print(f"  flash attn_token_positions={flash.attn_token_positions}")
    attn_ratio = baseline.attn_token_positions / flash.attn_token_positions if flash.attn_token_positions else float("inf")
    print(f"  attn_cost_ratio padded/flash={attn_ratio:.3f}x")
    print(f"  padded pad_waste_tokens_per_forward={baseline.pad_waste_tokens_per_forward}")
    print_extreme_analysis(baseline, flash, baseline.forward_ms_mean)
    print(
        "  note: total-time comparison mixes batched parallelism with per-request semantics; "
        "see long-request-only and per-request breakdown above."
    )


if __name__ == "__main__":
    main()
