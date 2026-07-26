#!/usr/bin/env python3
"""Full-forward and complete-trajectory LLaDA tile-replacement evaluator."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule  # noqa: E402
from blasst.tile_replacement_runtime import (  # noqa: E402
    ReplacementRuntimeConfig,
    TileReplacementRuntime,
)
from collect_proxy_traces import corpus, cyclic_contexts  # noqa: E402
from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def edit_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    a, b = left.tolist(), right.tolist()
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def distribution_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    left, right = candidate.float(), reference.float()
    return {
        "cosine": float(F.cosine_similarity(left, right, dim=-1).mean()),
        "kl": max(
            0.0,
            float(F.kl_div(left.log_softmax(-1), right.softmax(-1), reduction="batchmean")),
        ),
        "top1_agreement": float((left.argmax(-1) == right.argmax(-1)).float().mean()),
    }


def rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 1.0
    left_rank = left.argsort().argsort().float()
    right_rank = right.argsort().argsort().float()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    return float(
        (left_rank * right_rank).sum()
        / (
            torch.linalg.vector_norm(left_rank)
            * torch.linalg.vector_norm(right_rank)
        ).clamp_min(1e-12)
    )


def confidence_and_token(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = logits.float().softmax(-1)
    return probability.max(-1)


@contextmanager
def install_runtime(
    model: torch.nn.Module,
    runtime: TileReplacementRuntime,
    threshold: list[float],
    masked_rows: list[torch.Tensor],
):
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    originals = [(module, module.flash_attn_func) for module in modules]
    for layer, (module, _) in enumerate(originals):
        def call(q, k, v, *, _layer=layer, **kwargs):
            return runtime.attention(
                q,
                k,
                v,
                blasst_threshold=threshold[0],
                layer=_layer,
                row_masked=masked_rows[0],
                softmax_scale=kwargs.get("softmax_scale"),
                causal=bool(kwargs.get("causal", False)),
            )

        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original


def policy_name(config: ReplacementRuntimeConfig) -> str:
    return (
        f"{config.scope}__{config.mass_method}__{config.value_method}_r{config.summary_slots}"
        f"__{config.policy}_{config.qualification:g}__newmax_{config.new_maximum_protection}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--mode", choices=("screening", "trajectory"), default="trajectory")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--num-contexts", type=int, default=12)
    parser.add_argument("--calibration-contexts", type=int, default=4)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--context-indices", default="")
    parser.add_argument("--screening-ratios", default="0.9,0.5,0.15")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--scope", choices=("veto_only", "all_rows"), default="all_rows")
    parser.add_argument("--mass-method", choices=("exact", "calibrated_maximum"), default="calibrated_maximum")
    parser.add_argument("--value-method", choices=("exact", "proxy_mean_q"), default="proxy_mean_q")
    parser.add_argument("--summary-slots", type=int, default=8)
    parser.add_argument(
        "--policy",
        choices=(
            "none",
            "all_candidates",
            "exact_allmax",
            "exact_vetomax",
            "predicted_maxmass",
            "predicted_error",
            "cumulative_exact_mass",
            "cumulative_predicted_error",
        ),
        default="all_candidates",
    )
    parser.add_argument("--qualification", type=float, default=0.03)
    parser.add_argument(
        "--new-max-protection",
        choices=("none", "veto_rows", "all_rows"),
        default="none",
    )
    parser.add_argument("--protect-layers", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full tile-replacement evaluation requires CUDA")
    if args.context_length % 128 or args.context_length % 64:
        raise ValueError("context length must be divisible by 128 and 64")
    if args.generation_length >= args.context_length:
        raise ValueError("generation length must leave a non-empty prefix")

    runtime_config = ReplacementRuntimeConfig(
        k=args.k,
        scope=args.scope,
        mass_method=args.mass_method,
        value_method=args.value_method,
        summary_slots=args.summary_slots,
        policy=args.policy,
        qualification=args.qualification,
        new_maximum_protection=args.new_max_protection,
        protected_layers=tuple(
            int(item) for item in args.protect_layers.split(",") if item.strip()
        ),
    )
    baseline_config = ReplacementRuntimeConfig(k=args.k, policy="none")
    torch.manual_seed(args.seed)
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
    contexts = cyclic_contexts(
        tokenizer(corpus(), add_special_tokens=False)["input_ids"],
        args.context_length,
        args.num_contexts,
    )
    mask_id = choose_mask_token_id(tokenizer, None)
    prefix_length = args.context_length - args.generation_length
    generated = torch.arange(args.context_length, device="cuda") >= prefix_length
    default_indices = (
        range(args.calibration_contexts)
        if args.split == "calibration"
        else range(args.calibration_contexts, args.num_contexts)
    )
    indices = (
        [int(item) for item in args.context_indices.split(",") if item.strip()]
        if args.context_indices
        else list(default_indices)
    )
    schedule = DiffusionLambdaSchedule()
    results = []

    def run_model(state, runtime, step, ratio):
        runtime.set_step(step, ratio)
        threshold = [float(schedule.threshold(ratio))]
        holder = [state.eq(mask_id)]
        with install_runtime(model, runtime, threshold, holder), torch.inference_mode():
            return model(input_ids=state, output_hidden_states=True)

    def reveal(state, logits, step):
        masked = state[0].eq(mask_id) & generated
        confidence, token = confidence_and_token(logits[0, masked])
        target = round(args.generation_length * (1.0 - (step + 1) / args.steps))
        count = max(1, int(masked.sum()) - target)
        selected = confidence.topk(min(count, confidence.numel())).indices
        positions = masked.nonzero().flatten()[selected]
        result = state.clone()
        result[0, positions] = token[selected]
        return result, positions, confidence

    for context_index in indices:
        clean = contexts[context_index : context_index + 1].cuda()
        if args.mode == "screening":
            priority = torch.rand(
                args.generation_length,
                generator=torch.Generator().manual_seed(args.seed + context_index),
            ).cuda()
            for ratio in (float(item) for item in args.screening_ratios.split(",")):
                state = clean.clone()
                suffix_mask = priority < ratio
                state[0, prefix_length:][suffix_mask] = mask_id
                baseline_runtime = TileReplacementRuntime(baseline_config)
                candidate_runtime = TileReplacementRuntime(runtime_config)
                baseline = run_model(state, baseline_runtime, 0, ratio)
                candidate = run_model(state, candidate_runtime, 0, ratio)
                selected = state[0].eq(mask_id) & generated
                left, right = candidate.logits[0, selected], baseline.logits[0, selected]
                left_conf, _ = confidence_and_token(left)
                right_conf, _ = confidence_and_token(right)
                reveal_count = max(1, round(int(selected.sum()) / args.steps))
                left_positions = selected.nonzero().flatten()[left_conf.topk(reveal_count).indices]
                right_positions = selected.nonzero().flatten()[right_conf.topk(reveal_count).indices]
                results.append(
                    {
                        "context_index": context_index,
                        "split": args.split,
                        "remaining_mask_ratio": ratio,
                        "masked_logits": distribution_metrics(left, right),
                        "confidence_rank_correlation": rank_correlation(left_conf, right_conf),
                        "reveal_position_agreement": float(
                            torch.isin(left_positions, right_positions).float().mean()
                        ),
                        "exact_reveal_position_set": bool(
                            torch.equal(left_positions.sort().values, right_positions.sort().values)
                        ),
                        "hidden_state_cosine": float(
                            F.cosine_similarity(
                                candidate.hidden_states[-1].float(),
                                baseline.hidden_states[-1].float(),
                                dim=-1,
                            ).mean()
                        ),
                        "runtime_stats": candidate_runtime.stats.to_dict(),
                    }
                )
                del baseline, candidate
        else:
            baseline_state = clean.clone()
            candidate_state = clean.clone()
            baseline_state[:, prefix_length:] = mask_id
            candidate_state[:, prefix_length:] = mask_id
            baseline_runtime = TileReplacementRuntime(baseline_config)
            candidate_runtime = TileReplacementRuntime(runtime_config)
            steps = []
            for step in range(args.steps):
                baseline_masked = baseline_state[0].eq(mask_id) & generated
                ratio = float(baseline_masked.sum() / args.generation_length)
                baseline = run_model(baseline_state, baseline_runtime, step, ratio)
                candidate = run_model(candidate_state, candidate_runtime, step, ratio)
                left = candidate.logits[0, baseline_masked]
                right = baseline.logits[0, baseline_masked]
                left_conf, _ = confidence_and_token(left)
                right_conf, _ = confidence_and_token(right)
                baseline_state, baseline_positions, _ = reveal(
                    baseline_state, baseline.logits, step
                )
                candidate_state, candidate_positions, _ = reveal(
                    candidate_state, candidate.logits, step
                )
                intersection = torch.isin(candidate_positions, baseline_positions).sum()
                union = len(set(candidate_positions.tolist()) | set(baseline_positions.tolist()))
                steps.append(
                    {
                        "step": step,
                        "remaining_mask_ratio": ratio,
                        "masked_logits": distribution_metrics(left, right),
                        "confidence_rank_correlation": rank_correlation(left_conf, right_conf),
                        "reveal_position_jaccard": float(intersection / max(union, 1)),
                        "exact_reveal_position_set": bool(
                            torch.equal(
                                candidate_positions.sort().values,
                                baseline_positions.sort().values,
                            )
                        ),
                        "generated_state_agreement": float(
                            (
                                candidate_state[:, generated]
                                == baseline_state[:, generated]
                            ).float().mean()
                        ),
                        "hidden_state_cosine": float(
                            F.cosine_similarity(
                                candidate.hidden_states[-1].float(),
                                baseline.hidden_states[-1].float(),
                                dim=-1,
                            ).mean()
                        ),
                    }
                )
                del baseline, candidate
            reference = baseline_state[0, generated]
            candidate_tokens = candidate_state[0, generated]
            different = int((candidate_tokens != reference).sum())
            results.append(
                {
                    "context_index": context_index,
                    "split": args.split,
                    "metrics": {
                        "exact_final_sequence_agreement": different == 0,
                        "differing_tokens": different,
                        "first_divergent_step": next(
                            (
                                row["step"]
                                for row in steps
                                if row["generated_state_agreement"] < 1.0
                            ),
                            -1,
                        ),
                        "final_edit_distance": edit_distance(candidate_tokens, reference),
                        "minimum_hidden_cosine": min(
                            row["hidden_state_cosine"] for row in steps
                        ),
                        "mean_masked_top1_agreement": sum(
                            row["masked_logits"]["top1_agreement"] for row in steps
                        )
                        / len(steps),
                        "mean_confidence_rank_correlation": sum(
                            row["confidence_rank_correlation"] for row in steps
                        )
                        / len(steps),
                        "mean_reveal_position_jaccard": sum(
                            row["reveal_position_jaccard"] for row in steps
                        )
                        / len(steps),
                    },
                    "runtime_stats": candidate_runtime.stats.to_dict(),
                    "steps": steps,
                    "reference_text": tokenizer.decode(reference, skip_special_tokens=True),
                    "candidate_text": tokenizer.decode(candidate_tokens, skip_special_tokens=True),
                }
            )
        print(f"completed context={context_index} mode={args.mode}", flush=True)

    payload = {
        "schema": "blasst-tile-replacement-llada-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "mode": args.mode,
        "context_length": args.context_length,
        "generation_length": args.generation_length,
        "steps": args.steps,
        "policy_name": policy_name(runtime_config),
        "policy": runtime_config.__dict__,
        "split": args.split,
        "context_indices": indices,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": args.mode,
                "contexts": len(results),
                "exact": sum(
                    bool(row.get("metrics", {}).get("exact_final_sequence_agreement"))
                    for row in results
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
