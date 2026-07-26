#!/usr/bin/env python3
"""Real LLaDA one-step quality evaluation for post-QK veto pruning."""

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
from blasst.veto_pruning import KVetoPolicy, ScoreThresholdPolicy
from collect_proxy_traces import corpus, cyclic_contexts
from collect_veto_traces import token_states
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def phase_index(ratio: float) -> int:
    return 0 if ratio >= 0.75 else 1 if ratio >= 0.25 else 2


def load_layer_thresholds(path: Path, policy: str, ratio: float) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if policy not in payload:
        raise ValueError(f"policy {policy!r} is absent from {path}")
    groups = payload[policy]
    phase = phase_index(ratio)
    return {layer: float(groups[str(phase * 4 + layer // 8)]) for layer in range(32)}


def build_score_policy(
    name: str,
    thresholds: dict[int, float],
    *,
    new_max_protection: str,
):
    options: dict[str, object] = {}
    if name.startswith("quantile_q"):
        options.update(
            score="quantile",
            aggregator="quantile",
            quantile=int(name.removeprefix("quantile_q")) / 100.0,
        )
    elif name.startswith("online_mass_v_max"):
        options.update(score="online_mass_v_max")
    elif name.startswith("online_mass_v_mean"):
        options.update(score="online_mass_v_mean")
    elif name == "online_variance_bound":
        options.update(score="online_variance_bound")
    elif name == "masked_global_budget":
        options.update(score="masked_total")
    elif name.startswith("online_mass_weighted_sum_b"):
        options.update(
            score="online_mass",
            aggregator="weighted_sum",
            beta=float(name.rsplit("b", 1)[1]),
        )
    elif name == "online_mass_masked_max":
        options.update(score="online_mass", row_scope="masked")
    elif name == "online_mass_stable_relaxed_max":
        options.update(score="online_mass", row_scope="nonstable")
    elif name == "online_mass_q95":
        options.update(score="online_mass", aggregator="quantile", quantile=0.95)
    elif name == "online_mass_q99":
        options.update(score="online_mass", aggregator="quantile", quantile=0.99)
    elif name == "online_mass_max":
        options.update(score="online_mass")
    else:
        raise ValueError(f"policy {name!r} is not online-computable by the reference path")
    return ScoreThresholdPolicy(
        thresholds,
        new_max_protection=new_max_protection,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )


def masked_distribution_metrics(
    candidate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, chunk: int = 128
) -> tuple[float, float]:
    selected_candidate = candidate[mask].float()
    selected_reference = reference[mask].float()
    cosine_sum = 0.0
    kl_sum = 0.0
    for start in range(0, selected_candidate.shape[0], chunk):
        left = selected_candidate[start : start + chunk]
        right = selected_reference[start : start + chunk]
        cosine_sum += float(torch.nn.functional.cosine_similarity(left, right, dim=-1).sum())
        kl_sum += float(
            torch.nn.functional.kl_div(
                left.log_softmax(-1), right.softmax(-1), reduction="sum"
            )
        )
    count = max(selected_candidate.shape[0], 1)
    return cosine_sum / count, max(0.0, kl_sum / count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--previous-mask-ratio", type=float, default=0.50)
    parser.add_argument("--current-mask-ratio", type=float, default=0.49)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--policy", choices=("k_veto", "score"), default="k_veto")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--restriction",
        choices=("any", "visible_only", "nonmasked_only", "stable_only"),
        default="any",
    )
    parser.add_argument(
        "--new-max-protection", choices=("all", "critical", "masked", "none"), default="all"
    )
    parser.add_argument("--score-policy", default="online_mass_max")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA veto evaluation requires CUDA")
    if args.previous_mask_ratio < args.current_mask_ratio:
        raise ValueError("previous ratio must be at least the current ratio")

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
    tokens = tokenizer(corpus(), add_special_tokens=False)["input_ids"]
    clean = cyclic_contexts(tokens, args.context_length, args.num_contexts)
    priorities = torch.stack(
        [
            torch.rand(
                args.context_length,
                generator=torch.Generator().manual_seed(args.seed + index),
            )
            for index in range(args.num_contexts)
        ]
    )
    states = token_states(
        priorities, args.current_mask_ratio, args.previous_mask_ratio
    )
    current = clean.clone()
    mask_id = choose_mask_token_id(tokenizer, None)
    current[states == 0] = mask_id
    current = current.cuda()
    states = states.cuda()
    masked = states == 0
    schedule = DiffusionLambdaSchedule()
    threshold = float(schedule.threshold(args.current_mask_ratio))

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            result = model(input_ids=current, output_hidden_states=True)
        return result.logits, result.hidden_states[-1]

    dense_logits, dense_hidden = forward()
    with install_blasst(model, blasst_lambda=threshold):
        sparse_logits, sparse_hidden = forward()

    if args.policy == "k_veto":
        policy = KVetoPolicy(
            args.k,
            log_threshold=math.log(threshold),
            restriction=args.restriction,
            new_max_protection=args.new_max_protection,
        )
        policy_description = {
            "family": "k_veto",
            "k": args.k,
            "restriction": args.restriction,
        }
    else:
        if args.thresholds is None:
            raise ValueError("--thresholds is required for score policies")
        layer_thresholds = load_layer_thresholds(
            args.thresholds, args.score_policy, args.current_mask_ratio
        )
        policy = build_score_policy(
            args.score_policy,
            layer_thresholds,
            new_max_protection=args.new_max_protection,
        )
        policy_description = {
            "family": "score",
            "score_policy": args.score_policy,
            "layer_thresholds": layer_thresholds,
        }

    with install_blasst(
        model,
        blasst_lambda=threshold,
        row_token_state=states,
        veto_decision_callback=policy,
    ):
        candidate_logits, candidate_hidden = forward()

    masked_cosine, masked_kl = masked_distribution_metrics(
        candidate_logits, sparse_logits, masked
    )
    per_context = []
    for index in range(args.num_contexts):
        selected = masked[index]
        per_context.append(
            {
                "context": index,
                "masked_tokens": int(selected.sum()),
                "masked_top1_agreement_vs_sparse": float(
                    (
                        candidate_logits[index, selected].argmax(-1)
                        == sparse_logits[index, selected].argmax(-1)
                    ).float().mean()
                ),
                "masked_top1_agreement_vs_dense": float(
                    (
                        candidate_logits[index, selected].argmax(-1)
                        == dense_logits[index, selected].argmax(-1)
                    ).float().mean()
                ),
            }
        )

    result = {
        "model": args.model,
        "context_length": args.context_length,
        "num_contexts": args.num_contexts,
        "previous_mask_ratio": args.previous_mask_ratio,
        "current_mask_ratio": args.current_mask_ratio,
        "blasst_lambda": threshold,
        "new_max_protection": args.new_max_protection,
        "policy": policy_description,
        "metrics": {
            "masked_top1_agreement_vs_sparse": float(
                (candidate_logits[masked].argmax(-1) == sparse_logits[masked].argmax(-1))
                .float()
                .mean()
            ),
            "masked_top1_agreement_vs_dense": float(
                (candidate_logits[masked].argmax(-1) == dense_logits[masked].argmax(-1))
                .float()
                .mean()
            ),
            "sparse_masked_top1_agreement_vs_dense": float(
                (sparse_logits[masked].argmax(-1) == dense_logits[masked].argmax(-1))
                .float()
                .mean()
            ),
            "all_token_top1_agreement_vs_sparse": float(
                (candidate_logits.argmax(-1) == sparse_logits.argmax(-1)).float().mean()
            ),
            "masked_logit_cosine_vs_sparse": masked_cosine,
            "masked_logit_kl_vs_sparse": masked_kl,
            "hidden_state_cosine_vs_sparse": float(
                torch.nn.functional.cosine_similarity(
                    candidate_hidden.float().flatten(), sparse_hidden.float().flatten(), dim=0
                )
            ),
            "hidden_state_cosine_vs_dense": float(
                torch.nn.functional.cosine_similarity(
                    candidate_hidden.float().flatten(), dense_hidden.float().flatten(), dim=0
                )
            ),
            "worst_context_masked_agreement_vs_sparse": min(
                row["masked_top1_agreement_vs_sparse"] for row in per_context
            ),
        },
        "policy_stats": policy.stats.to_dict(),
        "per_context": per_context,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
