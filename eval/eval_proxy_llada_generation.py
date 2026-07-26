#!/usr/bin/env python3
"""Small complete-block LLaDA generation comparison for proxy-BLASST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule
from llada_eval_utils import choose_mask_token_id, disable_use_cache, ensure_all_tied_weights_keys, patch_llada_transformers_compat
from proxy import (
    BinaryReusePolicy, DisabledPolicy, MaxScorePolicy, MetadataStore, SafetyPolicy,
    ScoreThresholdPolicy, install_proxy_blasst_reference, load_calibrated_thresholds,
)


def bucket(ratio: float) -> str:
    return "high" if ratio >= 0.75 else "mid" if ratio >= 0.25 else "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", default="Explain why physical attention tiles matter in GPU kernels.")
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mode", choices=("oracle", "realistic"), default="oracle")
    parser.add_argument("--policy", choices=("binary", "score", "max_score"), default="binary")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--false-skip-budget", type=float, default=1e-3)
    parser.add_argument("--warmup-tiles", type=int, default=2)
    parser.add_argument("--periodic-refresh", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/proxy_llada_generation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA generation evaluation requires CUDA")
    if args.policy == "binary":
        inner_policy = BinaryReusePolicy("previous_step")
        calibrated_threshold_count = 0
    else:
        if args.calibration is None:
            raise ValueError("--calibration is required for continuous-score policies")
        relation = "previous_step" if args.policy == "score" else "max_previous_step_previous_layer"
        table = load_calibrated_thresholds(
            args.calibration, relation=relation, false_skip_budget=args.false_skip_budget
        )
        calibrated_threshold_count = len(table.values)
        inner_policy = (
            ScoreThresholdPolicy("previous_step", table)
            if args.policy == "score"
            else MaxScorePolicy(("previous_step", "previous_layer"), table)
        )
    policy = SafetyPolicy(
        inner_policy, warmup_tiles=args.warmup_tiles, periodic_refresh=args.periodic_refresh
    )
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    total_length = len(prompt_ids) + args.generation_length
    initial = torch.full((1, total_length), choose_mask_token_id(tokenizer, None), dtype=torch.long, device="cuda")
    initial[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device="cuda")
    generated_region = torch.arange(total_length, device="cuda") >= len(prompt_ids)
    schedule = DiffusionLambdaSchedule()

    def denoise(inputs: torch.Tensor, forward) -> torch.Tensor:
        state = inputs.clone()
        initial_masks = args.generation_length
        for step in range(args.steps):
            logits = forward(state, step)
            masked = state[0].eq(choose_mask_token_id(tokenizer, None)) & generated_region
            if not bool(masked.any()):
                break
            probabilities = logits[0, masked].softmax(-1)
            confidence, token = probabilities.max(-1)
            target_remaining = round(initial_masks * (1.0 - (step + 1) / args.steps))
            reveal = max(1, int(masked.sum()) - target_remaining)
            selected = confidence.topk(min(reveal, confidence.numel())).indices
            positions = masked.nonzero().flatten()[selected]
            state[0, positions] = token[selected]
        return state

    with torch.inference_mode():
        dense = denoise(initial, lambda state, step: model(input_ids=state).logits)

    exact_store = MetadataStore()
    threshold = [schedule.high_noise_lambda]
    with install_proxy_blasst_reference(
        model, blasst_lambda=lambda: threshold[0], policy=DisabledPolicy(),
        output_store=exact_store, sample_ids=["generation"],
    ) as exact_controller:
        def exact_forward(state: torch.Tensor, step: int) -> torch.Tensor:
            ratio = float(state[0].eq(choose_mask_token_id(tokenizer, None)).sum() / args.generation_length)
            threshold[0] = schedule.threshold(ratio)
            exact_controller.set_step(step, bucket(ratio))
            return model(input_ids=state).logits

        with torch.inference_mode():
            exact = denoise(initial, exact_forward)

    proxy_store = MetadataStore()
    source_store = exact_store if args.mode == "oracle" else proxy_store
    proxy_stats = []
    with install_proxy_blasst_reference(
        model, blasst_lambda=lambda: threshold[0], policy=policy,
        source_store=source_store, output_store=proxy_store, oracle_targets=exact_store,
        sample_ids=["generation"],
    ) as proxy_controller:
        def proxy_forward(state: torch.Tensor, step: int) -> torch.Tensor:
            ratio = float(state[0].eq(choose_mask_token_id(tokenizer, None)).sum() / args.generation_length)
            threshold[0] = schedule.threshold(ratio)
            proxy_controller.set_step(step, bucket(ratio))
            logits = model(input_ids=state).logits
            if proxy_controller.stats is not None:
                proxy_stats.append({"step": step, "remaining_mask_ratio": ratio, **proxy_controller.stats.summary()})
            return logits

        with torch.inference_mode():
            proxy = denoise(initial, proxy_forward)

    region = generated_region[None]
    result = {
        "model": args.model,
        "mode": args.mode,
        "policy": args.policy,
        "false_skip_budget": args.false_skip_budget if args.policy != "binary" else None,
        "calibrated_threshold_count": calibrated_threshold_count,
        "prompt": args.prompt,
        "steps": args.steps,
        "generation_length": args.generation_length,
        "proxy_token_agreement_vs_exact_blasst": float((proxy[region] == exact[region]).float().mean()),
        "proxy_token_agreement_vs_dense": float((proxy[region] == dense[region]).float().mean()),
        "exact_blasst_token_agreement_vs_dense": float((exact[region] == dense[region]).float().mean()),
        "dense_text": tokenizer.decode(dense[0, len(prompt_ids):], skip_special_tokens=True),
        "exact_blasst_text": tokenizer.decode(exact[0, len(prompt_ids):], skip_special_tokens=True),
        "proxy_text": tokenizer.decode(proxy[0, len(prompt_ids):], skip_special_tokens=True),
        "proxy_stats": proxy_stats,
        "quality_note": "Inspect text and task-specific scoring on a larger prompt set; token agreement is the primary regression metric here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
