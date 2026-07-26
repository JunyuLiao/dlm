#!/usr/bin/env python3
"""Collect compact full-trajectory traces for attention-result reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing.tile_reuse_collector import (
    ReuseTraceStep,
    SCHEMA_VERSION,
    TileReuseTraceCollector,
    install_tile_reuse_tracer,
)
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


PROMPTS = (
    "Explain why physical attention tiles matter in GPU kernels.",
    "Derive the intuition behind diffusion language model denoising.",
    "Compare memory bandwidth and arithmetic intensity on modern accelerators.",
    "Write a concise argument for validating optimizations on held-out prompts.",
    "Describe how online softmax combines independently normalized blocks.",
    "Explain why rare attention rows can control a whole physical tile.",
    "Discuss when caching is slower than recomputation on a GPU.",
    "Summarize the risks of using stale neural-network activations.",
)


def split_for_context(index: int) -> str:
    return ("development", "calibration", "heldout", "final_benchmark")[index % 4]


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return result


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--heads", type=parse_ints, default=(0, 7, 15, 31))
    parser.add_argument("--layers", type=parse_ints)
    parser.add_argument("--raw-layers", type=parse_ints, default=(0, 15, 31))
    parser.add_argument("--raw-heads", type=parse_ints, default=(0,))
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--topk-logits", type=int, default=16)
    parser.add_argument("--stable-drift", type=float, default=0.05)
    parser.add_argument(
        "--certificate-mass-budgets",
        type=parse_floats,
        default=(0.001, 0.005, 0.01, 0.025),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("tile-reuse tracing requires CUDA")
    if args.context_length % 128 or args.context_length % 64:
        raise ValueError("context length must be divisible by 128 and 64")
    if args.num_contexts > len(PROMPTS):
        raise ValueError(f"at most {len(PROMPTS)} built-in prompts are available")

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
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
    mask_id = choose_mask_token_id(tokenizer, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema": SCHEMA_VERSION,
        "model": args.model,
        "context_length": args.context_length,
        "num_contexts": args.num_contexts,
        "steps": args.steps,
        "heads": args.heads,
        "layers": args.layers,
        "raw_layers": args.raw_layers,
        "raw_heads": args.raw_heads,
        "seed": args.seed,
        "topk_logits": args.topk_logits,
        "stable_drift": args.stable_drift,
        "certificate_mass_budgets": args.certificate_mass_budgets,
        "context_splits": {
            f"context-{index:03d}": split_for_context(index)
            for index in range(args.num_contexts)
        },
    }
    write_json(args.output_dir / "collection_plan.json", plan)

    for context_index, prompt in enumerate(PROMPTS[: args.num_contexts]):
        request_id = f"context-{context_index:03d}"
        context_dir = args.output_dir / request_id
        summary_path = context_dir / "tile-reuse-summary.npz"
        if summary_path.exists():
            print(f"skipping completed {request_id}", flush=True)
            continue
        context_dir.mkdir(parents=True, exist_ok=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(prompt_ids) >= args.context_length:
            raise ValueError("prompt is too long for requested context")
        state = torch.full(
            (1, args.context_length), mask_id, dtype=torch.long, device="cuda"
        )
        state[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device="cuda")
        previous_state = state.clone()
        generated_region = torch.arange(args.context_length, device="cuda") >= len(prompt_ids)
        generated_length = int(generated_region.sum())
        query_tiles = args.context_length // 128
        query_tile = int.from_bytes(
            hashlib.blake2b(request_id.encode(), digest_size=4).digest(), "little"
        ) % query_tiles
        collector = TileReuseTraceCollector(
            context_dir,
            heads=args.heads,
            layers=args.layers,
            raw_layers=args.raw_layers,
            raw_heads=args.raw_heads,
            stable_drift=args.stable_drift,
            certificate_mass_budgets=args.certificate_mass_budgets,
        )
        step_records: list[dict[str, object]] = []
        with install_tile_reuse_tracer(model, collector):
            for step in range(args.steps):
                token_states = torch.full_like(state, 2, dtype=torch.uint8)
                token_states[state == mask_id] = 0
                token_states[(state != mask_id) & (previous_state == mask_id)] = 1
                token_states[:, : len(prompt_ids)] = 4
                remaining = int((state[0] == mask_id).logical_and(generated_region).sum())
                ratio = remaining / generated_length
                collector.set_step(
                    ReuseTraceStep((request_id,), step, ratio, query_tile, token_states)
                )
                with torch.inference_mode():
                    logits = model(input_ids=state).logits
                masked = (state[0] == mask_id) & generated_region
                probabilities = logits[0, masked].float().softmax(-1)
                confidence, token = probabilities.max(-1)
                target_remaining = round(
                    generated_length * (1.0 - (step + 1) / args.steps)
                )
                reveal = max(1, remaining - target_remaining)
                selected = confidence.topk(min(reveal, confidence.numel())).indices
                positions = masked.nonzero().flatten()[selected]
                top_probability, top_token = probabilities.topk(
                    min(args.topk_logits, probabilities.shape[-1]), dim=-1
                )
                torch.save(
                    {
                        "positions": masked.nonzero().flatten().cpu(),
                        "top_token": top_token.to(torch.int32).cpu(),
                        "top_probability": top_probability.to(torch.float16).cpu(),
                    },
                    context_dir / f"masked-logits-topk-step{step:02d}.pt",
                )
                next_state = state.clone()
                next_state[0, positions] = token[selected]
                step_records.append(
                    {
                        "step": step,
                        "remaining_mask_ratio": ratio,
                        "remaining_masked": remaining,
                        "revealed": int(positions.numel()),
                        "revealed_positions": positions.cpu().tolist(),
                        "revealed_token_ids": token[selected].cpu().tolist(),
                    }
                )
                previous_state, state = state, next_state
                print(
                    f"{request_id} step={step} ratio={ratio:.4f} revealed={positions.numel()}",
                    flush=True,
                )
        collector.flush()
        write_json(
            context_dir / "trajectory.json",
            {
                "request_id": request_id,
                "split": split_for_context(context_index),
                "prompt": prompt,
                "prompt_length": len(prompt_ids),
                "query_tile": query_tile,
                "steps": step_records,
                "final_token_ids": state[0].cpu().tolist(),
                "final_text": tokenizer.decode(
                    state[0, len(prompt_ids) :], skip_special_tokens=True
                ),
            },
        )
        print(f"completed {request_id}", flush=True)


if __name__ == "__main__":
    main()
