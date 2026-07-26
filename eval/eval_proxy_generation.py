#!/usr/bin/env python3
"""Small iterative counterfactual proxy evaluation across denoising steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proxy.proxy_blasst_reference import proxy_blasst_reference
from proxy.proxy_policy import BinaryReusePolicy, MetadataStore, SafetyPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/proxy_generation.json"))
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(1, args.length, 4, 16, generator=generator)
    k = torch.randn(q.shape, generator=generator)
    v = torch.randn(q.shape, generator=generator)
    exact_store, online_store = MetadataStore(), MetadataStore()
    policy = SafetyPolicy(BinaryReusePolicy("previous_step"), warmup_tiles=1, periodic_refresh=4)
    step_rows = []
    exact_state, proxy_state = q.clone(), q.clone()
    for step in range(args.steps):
        exact = proxy_blasst_reference(
            exact_state, k, v, blasst_lambda=args.lambda_value,
            output_store=exact_store, denoising_iteration=step, q_block_size=16, kv_block_size=8,
        )
        proxy = proxy_blasst_reference(
            proxy_state, k, v, blasst_lambda=args.lambda_value, policy=policy,
            source_store=online_store, output_store=online_store, oracle_targets=exact_store,
            denoising_iteration=step, q_block_size=16, kv_block_size=8,
        )
        cosine = float(torch.nn.functional.cosine_similarity(exact.output.flatten(), proxy.output.flatten(), dim=0))
        step_rows.append({"step": step, "hidden_cosine": cosine, **proxy.stats.summary()})
        exact_state = 0.8 * exact_state + 0.2 * exact.output
        proxy_state = 0.8 * proxy_state + 0.2 * proxy.output
    result = {
        "evaluation": "small reference block-generation trajectory (not a quality benchmark)",
        "oracle_source": False,
        "steps": step_rows,
        "final_state_cosine": float(torch.nn.functional.cosine_similarity(exact_state.flatten(), proxy_state.flatten(), dim=0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
