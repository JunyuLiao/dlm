#!/usr/bin/env python3
"""Complete LLaDA trajectory evaluation for exact post-pass mass pruning.

This is intentionally a dense correctness harness, not a kernel benchmark.  It
requires CUDA and local/model-hub access to LLaDA-8B-Instruct.
"""

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
from blasst.oracle_mass_pruning import (  # noqa: E402
    OracleMassPruner,
    OraclePolicyConfig,
    oracle_mass_pruned_attention,
)
from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


DEFAULT_PROMPTS = (
    "Explain why physical attention tiles matter in GPU kernels.",
    "Write a short proof that the square root of two is irrational.",
    "Describe a careful experiment for comparing two approximation methods.",
)


def edit_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    a, b = left.tolist(), right.tolist()
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def distribution_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    candidate = candidate.float()
    reference = reference.float()
    cosine = float(F.cosine_similarity(candidate, reference, dim=-1).mean())
    kl = float(
        F.kl_div(candidate.log_softmax(-1), reference.softmax(-1), reduction="batchmean")
    )
    return cosine, max(0.0, kl)


@contextmanager
def install_oracle(
    model: torch.nn.Module,
    pruner: OracleMassPruner,
    threshold: list[float],
    masked_rows: list[torch.Tensor],
    cache_namespace: list[object],
):
    replaced: list[tuple[torch.nn.Module, object]] = []
    for layer, module in enumerate(item for item in model.modules() if hasattr(item, "flash_attn_func")):
        def call(q, k, v, *, _layer=layer, **kwargs):
            if float(kwargs.get("dropout_p", 0.0)) != 0.0:
                raise ValueError("oracle evaluation only supports inference without dropout")
            return oracle_mass_pruned_attention(
                q,
                k,
                v,
                blasst_threshold=threshold[0],
                pruner=pruner,
                layer=_layer,
                row_masked=masked_rows[0],
                softmax_scale=kwargs.get("softmax_scale"),
                causal=bool(kwargs.get("causal", False)),
                cache_namespace=cache_namespace[0],
            )

        replaced.append((module, module.flash_attn_func))
        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in replaced:
            module.flash_attn_func = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--seeds", default="20260721,20260722")
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--aggregation", default="all_max")
    parser.add_argument("--selection", choices=("independent", "cumulative"), default="cumulative")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--budget", type=float, default=1e-3)
    parser.add_argument("--budget-scope", choices=("all", "masked", "mixed"), default="all")
    parser.add_argument("--order", choices=("score", "traversal", "masked_mass"), default="score")
    parser.add_argument("--new-max-protection", choices=("none", "all", "masked"), default="none")
    parser.add_argument(
        "--protect-layers",
        default="",
        help="comma-separated transformer layers that may not receive extra pruning",
    )
    parser.add_argument("--fixed-oracle-mask", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full oracle LLaDA trajectory evaluation requires CUDA")
    if args.selection == "independent" and args.threshold is None:
        raise ValueError("--threshold is required for independent selection")

    prompts = list(DEFAULT_PROMPTS)
    if args.prompts is not None:
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
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
    mask_id = choose_mask_token_id(tokenizer, None)
    schedule = DiffusionLambdaSchedule()
    policy_config = OraclePolicyConfig(
        k=args.k,
        aggregation=args.aggregation,
        selection=args.selection,
        threshold=args.threshold,
        budget=args.budget,
        budget_scope=args.budget_scope,
        order=args.order,
        new_maximum_protection=args.new_max_protection,
        protected_layers=tuple(
            int(value) for value in args.protect_layers.split(",") if value.strip()
        ),
    )
    results = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        for seed in seeds:
            torch.manual_seed(seed)
            total = len(prompt_ids) + args.generation_length
            initial = torch.full((1, total), mask_id, dtype=torch.long, device="cuda")
            initial[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device="cuda")
            generated = torch.arange(total, device="cuda") >= len(prompt_ids)
            dense_state = initial.clone()
            sparse_state = initial.clone()
            oracle_state = initial.clone()
            pruner = OracleMassPruner(policy_config, reuse_mask=args.fixed_oracle_mask)
            # Use the identical post-pass implementation for the ordinary
            # sparse reference. A zero independent threshold selects no extra
            # tiles, isolating pruning from floating-point reduction-order
            # differences between the online and post-pass implementations.
            baseline_pruner = OracleMassPruner(
                OraclePolicyConfig(
                    k=args.k,
                    aggregation="all_max",
                    selection="independent",
                    threshold=0.0,
                )
            )
            threshold = [schedule.high_noise_lambda]
            masked_holder = [oracle_state.eq(mask_id)]
            baseline_masked_holder = [sparse_state.eq(mask_id)]
            namespace = [(prompt_index, seed)]
            baseline_namespace = [("baseline", prompt_index, seed)]
            step_rows = []

            def reveal(state: torch.Tensor, logits: torch.Tensor, step: int) -> tuple[torch.Tensor, list[int]]:
                masked = state[0].eq(mask_id) & generated
                if not bool(masked.any()):
                    return state, []
                probabilities = logits[0, masked].float().softmax(-1)
                confidence, tokens = probabilities.max(-1)
                target = round(args.generation_length * (1.0 - (step + 1) / args.steps))
                count = max(1, int(masked.sum()) - target)
                chosen = confidence.topk(min(count, confidence.numel())).indices
                positions = masked.nonzero().flatten()[chosen]
                result = state.clone()
                result[0, positions] = tokens[chosen]
                return result, positions.tolist()

            for step in range(args.steps):
                sparse_mask = sparse_state.eq(mask_id)
                ratio = float((sparse_mask[0] & generated).sum() / args.generation_length)
                threshold[0] = schedule.threshold(ratio)
                with torch.inference_mode():
                    dense_result = model(input_ids=dense_state, output_hidden_states=True)
                baseline_masked_holder[0] = sparse_state.eq(mask_id)
                with torch.inference_mode(), install_oracle(
                    model,
                    baseline_pruner,
                    threshold,
                    baseline_masked_holder,
                    baseline_namespace,
                ):
                    sparse_result = model(input_ids=sparse_state, output_hidden_states=True)
                masked_holder[0] = oracle_state.eq(mask_id)
                before = pruner.stats.additional_skipped_tiles
                before_by_layer = dict(pruner.stats.by_layer)
                before_by_layer_head = dict(pruner.stats.by_layer_head)
                with torch.inference_mode(), install_oracle(
                    model, pruner, threshold, masked_holder, namespace
                ):
                    oracle_result = model(input_ids=oracle_state, output_hidden_states=True)
                sparse_masked_positions = sparse_mask[0] & generated
                logit_cosine, logit_kl = distribution_metrics(
                    oracle_result.logits[0, sparse_masked_positions],
                    sparse_result.logits[0, sparse_masked_positions],
                )
                masked_top1 = float(
                    (
                        oracle_result.logits[0, sparse_masked_positions].argmax(-1)
                        == sparse_result.logits[0, sparse_masked_positions].argmax(-1)
                    ).float().mean()
                )
                all_top1 = float(
                    (
                        oracle_result.logits.argmax(-1)
                        == sparse_result.logits.argmax(-1)
                    ).float().mean()
                )
                hidden_cosine = float(
                    F.cosine_similarity(
                        oracle_result.hidden_states[-1].float(),
                        sparse_result.hidden_states[-1].float(),
                        dim=-1,
                    ).mean()
                )
                sparse_dense_hidden_cosine = float(
                    F.cosine_similarity(
                        sparse_result.hidden_states[-1].float(),
                        dense_result.hidden_states[-1].float(),
                        dim=-1,
                    ).mean()
                )
                dense_state, dense_positions = reveal(dense_state, dense_result.logits, step)
                sparse_state, sparse_positions = reveal(sparse_state, sparse_result.logits, step)
                oracle_state, oracle_positions = reveal(oracle_state, oracle_result.logits, step)
                agreement = float((sparse_state[:, generated] == oracle_state[:, generated]).float().mean())
                step_rows.append(
                    {
                        "step": step,
                        "mask_ratio": ratio,
                        "masked_token_logit_cosine": logit_cosine,
                        "masked_token_kl": logit_kl,
                        "masked_token_top1_agreement": masked_top1,
                        "all_token_top1_agreement": all_top1,
                        "hidden_state_cosine": hidden_cosine,
                        "sparse_vs_dense_hidden_state_cosine": sparse_dense_hidden_cosine,
                        "generated_state_agreement": agreement,
                        "oracle_vs_dense_generated_state_agreement": float(
                            (oracle_state[:, generated] == dense_state[:, generated]).float().mean()
                        ),
                        "sparse_vs_dense_generated_state_agreement": float(
                            (sparse_state[:, generated] == dense_state[:, generated]).float().mean()
                        ),
                        "dense_revealed_positions": dense_positions,
                        "sparse_revealed_positions": sparse_positions,
                        "oracle_revealed_positions": oracle_positions,
                        "additional_skipped_tiles": pruner.stats.additional_skipped_tiles - before,
                        "additional_skipped_tiles_by_layer": {
                            str(layer): count - before_by_layer.get(layer, 0)
                            for layer, count in pruner.stats.by_layer.items()
                            if count - before_by_layer.get(layer, 0)
                        },
                        "additional_skipped_tiles_by_layer_head": {
                            key: count - before_by_layer_head.get(key, 0)
                            for key, count in pruner.stats.by_layer_head.items()
                            if count - before_by_layer_head.get(key, 0)
                        },
                    }
                )
            sparse_generated = sparse_state[0, generated]
            oracle_generated = oracle_state[0, generated]
            dense_generated = dense_state[0, generated]
            difference = int((sparse_generated != oracle_generated).sum())
            first_divergence = next(
                (row["step"] for row in step_rows if row["generated_state_agreement"] < 1.0), -1
            )
            results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "seed": seed,
                    "metrics": {
                        "exact_generated_sequence_agreement": difference == 0,
                        "differing_generated_tokens": difference,
                        "differing_generated_token_fraction": difference / args.generation_length,
                        "first_divergent_step": first_divergence,
                        "final_edit_distance": edit_distance(oracle_generated, sparse_generated),
                        "oracle_vs_dense_differing_tokens": int(
                            (oracle_generated != dense_generated).sum()
                        ),
                        "sparse_vs_dense_differing_tokens": int(
                            (sparse_generated != dense_generated).sum()
                        ),
                        "oracle_vs_dense_edit_distance": edit_distance(
                            oracle_generated, dense_generated
                        ),
                    },
                    "policy_stats": pruner.stats.to_dict(),
                    "steps": step_rows,
                    "sparse_text": tokenizer.decode(sparse_generated, skip_special_tokens=True),
                    "oracle_text": tokenizer.decode(oracle_generated, skip_special_tokens=True),
                    "dense_text": tokenizer.decode(dense_generated, skip_special_tokens=True),
                }
            )
    payload = {
        "schema": "blasst-oracle-mass-llada-trajectory-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability()),
        "prompts_file": str(args.prompts) if args.prompts is not None else None,
        "generation_length": args.generation_length,
        "steps": args.steps,
        "policy": policy_config.__dict__,
        "oracle_refresh": "fixed" if args.fixed_oracle_mask else "current_step",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output": str(args.output),
                "trajectories": len(results),
                "exact_trajectories": sum(
                    bool(result["metrics"]["exact_generated_sequence_agreement"])
                    for result in results
                ),
                "differing_tokens": [
                    result["metrics"]["differing_generated_tokens"] for result in results
                ],
                "additional_skipped_tiles": [
                    result["policy_stats"]["additional_skipped_tiles"] for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
