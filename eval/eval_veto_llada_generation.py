#!/usr/bin/env python3
"""Complete denoising-trajectory evaluation for post-QK veto pruning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule, install_blasst
from blasst.veto_pruning import KVetoPolicy
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--prompt", default="Explain why physical attention tiles matter in GPU kernels."
    )
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--restriction",
        choices=("any", "visible_only", "nonmasked_only", "stable_only"),
        default="any",
    )
    parser.add_argument(
        "--new-max-protection", choices=("all", "critical", "masked", "none"), default="none"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA generation evaluation requires CUDA")

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
    prompt_ids = tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    total_length = len(prompt_ids) + args.generation_length
    initial = torch.full((1, total_length), mask_id, dtype=torch.long, device="cuda")
    initial[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device="cuda")
    generated_region = torch.arange(total_length, device="cuda") >= len(prompt_ids)
    schedule = DiffusionLambdaSchedule()

    def denoise(forward):
        state = initial.clone()
        previous = state.clone()
        step_metrics = []
        for step in range(args.steps):
            masked_before = state.eq(mask_id)
            ratio = float((masked_before[0] & generated_region).sum() / args.generation_length)
            logits, stats = forward(state, previous, ratio)
            masked = masked_before[0] & generated_region
            if not bool(masked.any()):
                break
            probabilities = logits[0, masked].softmax(-1)
            confidence, token = probabilities.max(-1)
            target_remaining = round(
                args.generation_length * (1.0 - (step + 1) / args.steps)
            )
            reveal = max(1, int(masked.sum()) - target_remaining)
            selected = confidence.topk(min(reveal, confidence.numel())).indices
            positions = masked.nonzero().flatten()[selected]
            previous = state.clone()
            state[0, positions] = token[selected]
            step_metrics.append(
                {
                    "step": step,
                    "remaining_mask_ratio": ratio,
                    "revealed": int(positions.numel()),
                    **stats,
                }
            )
        return state, step_metrics

    with torch.inference_mode():
        dense, dense_steps = denoise(
            lambda state, previous, ratio: (model(input_ids=state).logits, {})
        )

    threshold = [schedule.high_noise_lambda]
    with install_blasst(model, blasst_lambda=lambda layer: threshold[0]):
        def exact_forward(state, previous, ratio):
            del previous
            threshold[0] = schedule.threshold(ratio)
            return model(input_ids=state).logits, {}

        with torch.inference_mode():
            exact, exact_steps = denoise(exact_forward)

    layer_log_threshold = {layer: math.log(schedule.high_noise_lambda) for layer in range(32)}
    policy = KVetoPolicy(
        args.k,
        log_threshold=layer_log_threshold,
        restriction=args.restriction,
        new_max_protection=args.new_max_protection,
    )
    token_state_holder = [torch.empty(0, device="cuda", dtype=torch.uint8)]

    def current_token_states():
        return token_state_holder[0]

    with install_blasst(
        model,
        blasst_lambda=lambda layer: threshold[0],
        row_token_state=current_token_states,
        veto_decision_callback=policy,
    ):
        def candidate_forward(state, previous, ratio):
            threshold[0] = schedule.threshold(ratio)
            for layer in layer_log_threshold:
                layer_log_threshold[layer] = math.log(threshold[0])
            states = torch.full_like(state, 2, dtype=torch.uint8)
            states[state == mask_id] = 0
            states[(state != mask_id) & (previous == mask_id)] = 1
            states[:, : len(prompt_ids)] = 4
            token_state_holder[0] = states
            before = policy.stats.additional_skipped_tiles
            logits = model(input_ids=state).logits
            return logits, {
                "additional_skipped_tiles": policy.stats.additional_skipped_tiles - before
            }

        with torch.inference_mode():
            candidate, candidate_steps = denoise(candidate_forward)

    region = generated_region[None]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "generation_length": args.generation_length,
        "policy": {
            "family": "k_veto",
            "k": args.k,
            "restriction": args.restriction,
            "new_max_protection": args.new_max_protection,
        },
        "metrics": {
            "candidate_token_agreement_vs_sparse": float(
                (candidate[region] == exact[region]).float().mean()
            ),
            "candidate_token_agreement_vs_dense": float(
                (candidate[region] == dense[region]).float().mean()
            ),
            "sparse_token_agreement_vs_dense": float(
                (exact[region] == dense[region]).float().mean()
            ),
            "candidate_exact_generated_sequence_vs_sparse": bool(
                torch.equal(candidate[region], exact[region])
            ),
        },
        "policy_stats": policy.stats.to_dict(),
        "dense_text": tokenizer.decode(dense[0, len(prompt_ids) :], skip_special_tokens=True),
        "sparse_text": tokenizer.decode(exact[0, len(prompt_ids) :], skip_special_tokens=True),
        "candidate_text": tokenizer.decode(
            candidate[0, len(prompt_ids) :], skip_special_tokens=True
        ),
        "dense_steps": dense_steps,
        "sparse_steps": exact_steps,
        "candidate_steps": candidate_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
