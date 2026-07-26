#!/usr/bin/env python3
"""Reference one-step counterfactual evaluation (CPU smoke or supplied scale)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proxy.proxy_blasst_reference import proxy_blasst_reference
from proxy.proxy_policy import BinaryReusePolicy, DisabledPolicy, MetadataStore, SafetyPolicy


def metrics(candidate: torch.Tensor, reference: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    candidate_float, reference_float = candidate.float(), reference.float()
    relative = float((candidate_float - reference_float).norm() / reference_float.norm().clamp_min(1e-20))
    cosine = float(torch.nn.functional.cosine_similarity(candidate_float.flatten(), reference_float.flatten(), dim=0))
    projection = torch.randn(reference.shape[-1], 31, generator=torch.Generator().manual_seed(4))
    candidate_logits, reference_logits = candidate_float @ projection, reference_float @ projection
    log_p = candidate_logits.log_softmax(-1)
    q = reference_logits.softmax(-1)
    kl = max(0.0, float(torch.nn.functional.kl_div(log_p, q, reduction="batchmean")))
    return {
        "attention_output_relative_error_vs_exact_blasst": relative,
        "hidden_state_cosine_similarity": cosine,
        "final_logit_kl_divergence": kl,
        "top1_agreement_vs_exact_blasst": float((candidate_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
        "top1_agreement_vs_dense": float((candidate_logits.argmax(-1) == (dense.float() @ projection).argmax(-1)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--q-block-size", type=int, default=16)
    parser.add_argument("--kv-block-size", type=int, default=8)
    parser.add_argument("--mode", choices=("oracle", "realistic"), default="oracle")
    parser.add_argument("--output", type=Path, default=Path("artifacts/proxy_one_step.json"))
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(19)
    base_q = torch.randn(1, args.length, args.heads, args.head_dim, generator=generator)
    k = torch.randn(base_q.shape, generator=generator)
    v = torch.randn(base_q.shape, generator=generator)
    previous_q = base_q + torch.randn(base_q.shape, generator=generator) * 0.03
    exact_sources = MetadataStore()
    proxy_blasst_reference(
        previous_q, k, v,
        blasst_lambda=args.lambda_value,
        policy=DisabledPolicy(),
        output_store=exact_sources,
        denoising_iteration=0,
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
    )
    exact_targets = MetadataStore()
    exact = proxy_blasst_reference(
        base_q, k, v,
        blasst_lambda=args.lambda_value,
        policy=DisabledPolicy(),
        output_store=exact_targets,
        denoising_iteration=1,
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
    )
    realistic = exact_sources if args.mode == "realistic" else MetadataStore()
    policy = SafetyPolicy(
        BinaryReusePolicy("previous_step"),
        warmup_tiles=1,
        periodic_refresh=4,
        anchor_local=True,
        anchor_diagonal=True,
        anchor_sink=True,
    )
    source = exact_sources
    output_store = realistic
    proxy = proxy_blasst_reference(
        base_q, k, v,
        blasst_lambda=args.lambda_value,
        policy=policy,
        source_store=source,
        output_store=output_store,
        oracle_targets=exact_targets,
        denoising_iteration=1,
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
    )
    dense = torch.nn.functional.scaled_dot_product_attention(
        base_q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    result = {
        "mode": args.mode,
        "policy": "safe previous-step binary reuse",
        "metrics": metrics(proxy.output, exact.output, dense),
        "proxy_stats": proxy.stats.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
