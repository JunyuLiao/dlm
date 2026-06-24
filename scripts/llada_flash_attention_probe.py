#!/usr/bin/env python3
"""Probe whether LLaDA can run with flash_attention=True.

The remote modeling_llada.py enables flash_attn only when:
  1) config.flash_attention is True
  2) the flash_attn package is installed
  3) _scaled_dot_product_attention is called with attn_mask=None

In practice the model forward usually builds a bidirectional attention bias
(and padding mask), so attn_mask is rarely None even with flash_attention=True.
This script loads the model, counts attention backend usage, and reports whether
flash_attn kernels are actually exercised.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llada_block_step_probe import (
    append_mask_block,
    build_prompt_batch,
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)
from llada_length_heterogeneity_probe import make_prompt_for_target


@dataclass
class AttentionPathCounts:
    flash_attn_calls: int = 0
    sdpa_calls: int = 0


def count_flash_ready_layers(model: AutoModelForCausalLM) -> tuple[int, int]:
    total = 0
    ready = 0
    for module in model.modules():
        if hasattr(module, "flash_attn_func"):
            total += 1
            if getattr(module, "flash_attn_func", None) is not None:
                ready += 1
    return ready, total


def install_attention_counters(model: AutoModelForCausalLM) -> AttentionPathCounts:
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


def build_prompts(tokenizer: AutoTokenizer, batch_size: int, target_tokens: int) -> list:
    from llada_block_step_probe import PromptRecord

    prompts: list[PromptRecord] = []
    for request_id in range(batch_size):
        text, _actual = make_prompt_for_target(
            tokenizer,
            group_name="flash_probe",
            request_id=request_id,
            trial=0,
            target_tokens=target_tokens,
        )
        prompts.append(PromptRecord(request_id, "medium", text))
    return prompts


@torch.no_grad()
def run_forward_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list,
    device: torch.device,
    block_size: int,
    mask_token_id: int,
    padded: bool,
) -> tuple[float, AttentionPathCounts]:
    if padded:
        input_ids, attention_mask, _ = build_prompt_batch(tokenizer, prompts, device)
    else:
        encoded = [tokenizer(item.prompt, add_special_tokens=True)["input_ids"] for item in prompts]
        if len({len(ids) for ids in encoded}) != 1:
            raise ValueError("unpadded case requires equal-length prompts")
        input_ids = torch.tensor(encoded, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    input_ids, attention_mask, _ = append_mask_block(input_ids, attention_mask, block_size, mask_token_id)

    counts = install_attention_counters(model)
    try:
        cuda_sync()
        start = time.perf_counter()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        cuda_sync()
        latency_ms = (time.perf_counter() - start) * 1000.0
    finally:
        restore_attention_counters(counts)
    return latency_ms, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-prompt-tokens", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--mask-token-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[flash probe] ERROR: CUDA is required")
        sys.exit(1)

    try:
        import flash_attn  # type: ignore

        flash_attn_version = getattr(flash_attn, "__version__", "unknown")
        flash_attn_installed = True
    except ImportError:
        flash_attn_installed = False
        flash_attn_version = None

    print("[flash probe] flash_attn package:", "installed" if flash_attn_installed else "MISSING")
    if flash_attn_installed:
        print(f"[flash probe] flash_attn version: {flash_attn_version}")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"[flash probe] checkpoint config flash_attention={config.flash_attention}")
    config.flash_attention = True
    print(f"[flash probe] overriding flash_attention={config.flash_attention}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.to(device).eval()

    ready, total = count_flash_ready_layers(model)
    print(f"[flash probe] layers with flash_attn_func ready: {ready}/{total}")
    if config.flash_attention and ready == 0:
        print(
            "[flash probe] RESULT: flash_attention=True but no layer bound flash_attn_func. "
            "Install flash-attn (pip install flash-attn) and retry."
        )

    mask_token_id = choose_mask_token_id(tokenizer, args.mask_token_id)
    prompts = build_prompts(tokenizer, args.batch_size, args.target_prompt_tokens)

    cases = [
        ("unpadded_equal_length", False),
        ("padded_batch", True),
    ]

    print(
        f"[flash probe] running batch_size={args.batch_size} "
        f"target_prompt_tokens={args.target_prompt_tokens} block_size={args.block_size}"
    )
    for label, padded in cases:
        latency_ms, counts = run_forward_case(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            block_size=args.block_size,
            mask_token_id=mask_token_id,
            padded=padded,
        )
        print(
            f"[flash probe] case={label} latency_ms={latency_ms:.2f} "
            f"flash_attn_calls={counts.flash_attn_calls} sdpa_calls={counts.sdpa_calls}"
        )

    print("\n[flash probe] interpretation:")
    print("  - flash_attn_calls>0 means modeling_llada used flash_attn_func kernels.")
    print("  - padded_batch almost always stays on sdpa because attention_bias/mask is set.")
    print("  - unpadded equal-length inputs can hit flash_attn when attn_mask stays None.")
    if not flash_attn_installed:
        print("  - Install in ljy_dlm (torch must be visible during build):")
        print("      conda activate ljy_dlm")
        print("      pip install ninja")
        print("      pip install flash-attn --no-build-isolation")
        print("    Plain `pip install flash-attn` fails: setup.py imports torch in an isolated env without it.")
    elif ready == 0:
        print("  - flash-attn import failed at layer init; check CUDA/PyTorch compatibility.")
    else:
        print("  - Config override is doable; practical speedup depends on hitting flash_attn_calls>0.")


if __name__ == "__main__":
    main()
