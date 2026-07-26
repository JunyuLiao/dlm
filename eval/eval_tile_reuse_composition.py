#!/usr/bin/env python3
"""Model-level numerical validation of fresh tiled-softmax composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import install_blasst
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.flash_attention = True
    config.output_hidden_states = True
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
    clean = torch.randint(
        0, model.config.vocab_size, (1, args.context_length), device="cuda"
    )
    masked = torch.rand(args.context_length, device="cuda") < args.mask_ratio
    input_ids = clean.clone()
    input_ids[0, masked] = mask_id

    with torch.inference_mode():
        dense = model(input_ids=input_ids, output_hidden_states=True)
    with install_blasst(model, blasst_lambda=0.0, q_block_size=128, kv_block_size=64):
        with torch.inference_mode():
            tiled = model(input_ids=input_ids, output_hidden_states=True)
    dense_logits = dense.logits[0, masked].float()
    tiled_logits = tiled.logits[0, masked].float()
    dense_hidden = dense.hidden_states[-1].float()
    tiled_hidden = tiled.hidden_states[-1].float()
    logit_difference = tiled_logits - dense_logits
    hidden_difference = tiled_hidden - dense_hidden
    result = {
        "model": args.model,
        "context_length": args.context_length,
        "mask_ratio": args.mask_ratio,
        "masked_tokens": int(masked.sum()),
        "maximum_logit_absolute_error": float(logit_difference.abs().max()),
        "mean_logit_absolute_error": float(logit_difference.abs().mean()),
        "masked_logit_relative_error": float(
            torch.linalg.vector_norm(logit_difference)
            / (torch.linalg.vector_norm(dense_logits) + 1e-6)
        ),
        "masked_logit_cosine": float(
            torch.nn.functional.cosine_similarity(
                tiled_logits.flatten(), dense_logits.flatten(), dim=0
            )
        ),
        "masked_top1_agreement": float(
            (tiled_logits.argmax(-1) == dense_logits.argmax(-1)).float().mean()
        ),
        "maximum_hidden_absolute_error": float(hidden_difference.abs().max()),
        "mean_hidden_absolute_error": float(hidden_difference.abs().mean()),
        "hidden_relative_error": float(
            torch.linalg.vector_norm(hidden_difference)
            / (torch.linalg.vector_norm(dense_hidden) + 1e-6)
        ),
        "hidden_cosine": float(
            torch.nn.functional.cosine_similarity(
                tiled_hidden.flatten(), dense_hidden.flatten(), dim=0
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
