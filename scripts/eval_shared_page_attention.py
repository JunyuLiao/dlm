#!/usr/bin/env python3
"""Oracle and complete-trajectory evaluation for Shared-Page Attention."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)
from spa.analyzer import (  # noqa: E402
    SPAExactController,
    SPAOracleAnalyzer,
    config_name,
    install_exact_spa,
    install_spa_oracle_analyzer,
    tensor_error,
)
from spa.reference import SPAConfig  # noqa: E402


PROMPTS = (
    "Explain why physical attention tiles matter in GPU kernels.",
    "Derive the intuition behind diffusion language model denoising.",
    "Compare memory bandwidth and arithmetic intensity on modern accelerators.",
    "Write a concise argument for validating optimizations on held-out prompts.",
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def add_config(configs: dict[str, SPAConfig], config: SPAConfig) -> None:
    configs.setdefault(config_name(config), config)


def oracle_grid(args: argparse.Namespace) -> list[SPAConfig]:
    if args.sink_grid:
        return [
            SPAConfig(group_size=64, support="hybrid", density=0.6, sink_pages=sinks)
            for sinks in (0, 1, 2)
        ]
    if args.screening_grid:
        configs: dict[str, SPAConfig] = {}
        for group in (32, 64, 128):
            for density in (0.5, 0.6, 0.7):
                add_config(configs, SPAConfig(group_size=group, density=density))
        add_config(configs, SPAConfig(
            group_size=64, support="columns", density=0.7
        ))
        add_config(configs, SPAConfig(
            group_size=64, density=0.7, selector="oracle-tail",
            tail_beta=0.5, tail_quantile=0.95,
        ))
        add_config(configs, SPAConfig(
            group_size=64, density=0.7, selector="oracle-coverage",
            coverage_objective="p1",
        ))
        add_config(configs, SPAConfig(
            group_size=64, support="hybrid", density=0.6, sink_pages=1
        ))
        return list(configs.values())
    if not args.full_ablation:
        return [
            SPAConfig(
                group_size=args.spa_group_size,
                support=args.spa_support,
                density=args.spa_density,
                selector=args.spa_selector,
                sink_pages=args.spa_sink_pages,
                tail_beta=args.spa_tail_beta,
                tail_quantile=args.spa_tail_quantile,
                coverage_objective=args.spa_coverage_objective,
                mandatory_local=args.spa_mandatory_local,
                mandatory_first=args.spa_mandatory_first,
                mandatory_last=args.spa_mandatory_last,
            )
        ]
    configs: dict[str, SPAConfig] = {}
    # Core group/density Pareto surface.
    for group in (32, 64, 128):
        for density in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            add_config(configs, SPAConfig(group_size=group, density=density))
    # Format, selector, mandatory support, sink, and broader-sharing ablations
    # at the primary hardware shape and middle density.
    for support in ("columns", "pages", "hybrid"):
        sinks = (0, 1, 2) if support == "hybrid" else (0,)
        for sink_pages in sinks:
            add_config(configs, SPAConfig(support=support, density=0.5, sink_pages=sink_pages))
    for beta in (0.25, 0.5, 1.0, 2.0):
        for quantile in (0.90, 0.95, 1.0):
            add_config(configs, SPAConfig(
                density=0.5, selector="oracle-tail", tail_beta=beta,
                tail_quantile=quantile,
            ))
    for objective in ("minimum", "p1", "p5", "mean-min"):
        add_config(configs, SPAConfig(
            density=0.5, selector="oracle-coverage", coverage_objective=objective
        ))
    add_config(configs, SPAConfig(density=0.5, mandatory_local=True))
    add_config(configs, SPAConfig(density=0.5, mandatory_first=True))
    add_config(configs, SPAConfig(density=0.5, mandatory_last=True))
    add_config(configs, SPAConfig(density=0.5, share_heads_experimental=True))
    return list(configs.values())


def load_model(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    return model.eval().cuda(), AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )


def initial_state(tokenizer, prompt: str, context_length: int, mask_id: int) -> tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if len(prompt_ids) >= context_length:
        raise ValueError("prompt is too long for the requested context")
    state = torch.full((1, context_length), mask_id, dtype=torch.long, device="cuda")
    state[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device="cuda")
    return state, len(prompt_ids)


def reveal_step(
    state: torch.Tensor,
    logits: torch.Tensor,
    *,
    mask_id: int,
    prompt_length: int,
    initial_masked: int,
    step: int,
    steps: int,
) -> tuple[torch.Tensor, int]:
    masked = state[0].eq(mask_id)
    masked[:prompt_length] = False
    remaining = int(masked.sum())
    target = round(initial_masked * (1.0 - (step + 1) / steps))
    reveal = min(remaining, max(1, remaining - target))
    probabilities = logits[0, masked].float().softmax(-1)
    confidence, token = probabilities.max(-1)
    chosen = confidence.topk(reveal).indices
    positions = masked.nonzero().flatten()[chosen]
    result = state.clone()
    result[0, positions] = token[chosen]
    return result, reveal


def summarize_oracle(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row["config"])].append(row)
    result = []
    for name, rows in grouped.items():
        density = max(float(row["selected_key_density"]) for row in rows)
        worst_p1 = min(float(row["p1_retained_mass"]) for row in rows)
        worst_mean = min(float(row["mean_retained_mass"]) for row in rows)
        worst_l2 = max(float(row["output_relative_l2"]) for row in rows)
        gate = density <= 0.70 + 1e-8 and worst_p1 >= 0.95 and worst_mean >= 0.99
        result.append({
            "config": name,
            "layers_and_steps": len(rows),
            "maximum_selected_key_density": density,
            "logical_attention_work_reduction": 1.0 - density,
            "worst_layer_step_p1_retained_mass": worst_p1,
            "worst_layer_step_mean_retained_mass": worst_mean,
            "worst_layer_step_output_relative_l2": worst_l2,
            "mass_gate_pass": gate,
            "note": "Output-error acceptability remains a reported screening dimension, not a relaxed mass gate.",
        })
    return sorted(
        result,
        key=lambda row: (
            not bool(row["mass_gate_pass"]),
            -float(row["worst_layer_step_p1_retained_mass"]),
            float(row["maximum_selected_key_density"]),
        ),
    )


def distribution_metrics(candidate: torch.Tensor, reference: torch.Tensor, selected: torch.Tensor) -> dict[str, float]:
    left = candidate[selected].float()
    right = reference[selected].float()
    if not left.shape[0]:
        return {"tokens": 0, "top1_agreement": 1.0, "logit_kl": 0.0, "relative_l2": 0.0}
    difference = left - right
    return {
        "tokens": int(left.shape[0]),
        "top1_agreement": float((left.argmax(-1) == right.argmax(-1)).float().mean()),
        "logit_kl": float(F.kl_div(left.log_softmax(-1), right.softmax(-1), reduction="batchmean")),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(right).clamp_min(1e-30)
        ),
    }


def refresh_steps(args: argparse.Namespace) -> set[int] | None:
    if args.spa_refresh == "current":
        return None
    if args.spa_refresh == "first":
        return {args.spa_dense_early_steps}
    return set(range(args.spa_dense_early_steps, args.steps, args.spa_refresh_interval))


def run_oracle(args, model, tokenizer, mask_id, configs):
    all_records: list[dict[str, object]] = []
    trajectories = []
    for seed in args.seeds:
        for context_index, prompt in enumerate(PROMPTS[: args.num_contexts]):
            torch.manual_seed(seed)
            state, prompt_length = initial_state(tokenizer, prompt, args.context_length, mask_id)
            initial_masked = args.context_length - prompt_length
            analyzer = SPAOracleAnalyzer(configs)
            steps = []
            with install_spa_oracle_analyzer(model, analyzer):
                for step in range(args.steps):
                    masked = state.eq(mask_id)
                    remaining = int(masked[:, prompt_length:].sum())
                    analyzer.set_step(
                        step=step,
                        remaining_mask_ratio=remaining / initial_masked,
                        masked_rows=masked,
                        request_ids=(f"seed-{seed}-context-{context_index}",),
                    )
                    with torch.inference_mode():
                        logits = model(input_ids=state).logits
                    analyzer.finish_step()
                    state, revealed = reveal_step(
                        state, logits, mask_id=mask_id, prompt_length=prompt_length,
                        initial_masked=initial_masked, step=step, steps=args.steps,
                    )
                    steps.append({"step": step, "remaining": remaining, "revealed": revealed})
                    del logits
            for row in analyzer.records:
                row["seed"] = seed
                row["context_index"] = context_index
            all_records.extend(analyzer.records)
            trajectories.append({
                "seed": seed,
                "context_index": context_index,
                "prompt": prompt,
                "steps": steps,
                "final_text": tokenizer.decode(state[0, prompt_length:], skip_special_tokens=True),
            })
            gc.collect()
            torch.cuda.empty_cache()
    return {"oracle_records": all_records, "oracle_summary": summarize_oracle(all_records), "trajectories": trajectories}


def forward_spa(model, state, controller, step, dense_early_steps):
    controller.set_step(step)
    if step < dense_early_steps:
        with torch.inference_mode():
            return model(input_ids=state, output_hidden_states=True)
    with install_exact_spa(model, controller):
        with torch.inference_mode():
            return model(input_ids=state, output_hidden_states=True)


def run_quality(args, model, tokenizer, mask_id, config):
    results = []
    for seed in args.seeds:
        for context_index, prompt in enumerate(PROMPTS[: args.num_contexts]):
            torch.manual_seed(seed)
            initial, prompt_length = initial_state(tokenizer, prompt, args.context_length, mask_id)
            initial_masked = args.context_length - prompt_length
            teacher_state = initial.clone()
            controller = SPAExactController(config, refresh_steps=refresh_steps(args))
            teacher_steps = []
            for step in range(args.steps):
                masked = teacher_state.eq(mask_id)
                with torch.inference_mode():
                    dense = model(input_ids=teacher_state, output_hidden_states=True)
                candidate = forward_spa(model, teacher_state, controller, step, args.spa_dense_early_steps)
                teacher_steps.append({
                    "step": step,
                    "remaining_mask_ratio": float(masked[:, prompt_length:].float().mean()),
                    "masked_logits": distribution_metrics(candidate.logits, dense.logits, masked),
                    "revealed_logits": distribution_metrics(candidate.logits, dense.logits, ~masked),
                    "hidden_states": [
                        {"hidden_state": layer, **tensor_error(left, right)}
                        for layer, (left, right) in enumerate(
                            zip(candidate.hidden_states, dense.hidden_states, strict=True)
                        )
                    ],
                })
                teacher_state, _ = reveal_step(
                    teacher_state, dense.logits, mask_id=mask_id, prompt_length=prompt_length,
                    initial_masked=initial_masked, step=step, steps=args.steps,
                )
                del dense, candidate

            def denoise(method: str):
                state = initial.clone()
                local_controller = SPAExactController(config, refresh_steps=refresh_steps(args))
                for step in range(args.steps):
                    if method == "dense":
                        with torch.inference_mode():
                            output = model(input_ids=state)
                    else:
                        output = forward_spa(
                            model, state, local_controller, step, args.spa_dense_early_steps
                        )
                    state, _ = reveal_step(
                        state, output.logits, mask_id=mask_id,
                        prompt_length=prompt_length, initial_masked=initial_masked,
                        step=step, steps=args.steps,
                    )
                    del output
                return state, local_controller

            dense_final, _ = denoise("dense")
            spa_final, free_controller = denoise("spa")
            generated = torch.arange(args.context_length, device="cuda") >= prompt_length
            results.append({
                "seed": seed,
                "context_index": context_index,
                "prompt": prompt,
                "teacher_forced_steps": teacher_steps,
                "teacher_attention_records": controller.attention_records,
                "free_running_attention_records": free_controller.attention_records,
                "final_token_agreement": float(
                    (dense_final[0, generated] == spa_final[0, generated]).float().mean()
                ),
                "exact_final_sequence_agreement": bool(torch.equal(dense_final, spa_final)),
                "dense_final_text": tokenizer.decode(dense_final[0, prompt_length:], skip_special_tokens=True),
                "spa_final_text": tokenizer.decode(spa_final[0, prompt_length:], skip_special_tokens=True),
            })
            gc.collect()
            torch.cuda.empty_cache()
    worst_step = min(
        step["masked_logits"]["top1_agreement"]
        for result in results for step in result["teacher_forced_steps"]
    )
    return {
        "quality_results": results,
        "quality_summary": {
            "worst_teacher_forced_masked_top1_agreement": worst_step,
            "mean_final_token_agreement": sum(row["final_token_agreement"] for row in results) / len(results),
            "quality_gate_top1_pass": worst_step >= 0.99,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--mode", choices=("oracle", "quality"), default="oracle")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seeds", type=parse_ints, default=[20260721])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-ablation", action="store_true")
    parser.add_argument(
        "--screening-grid", action="store_true",
        help="compact target-length gate sweep across group, format, and selector",
    )
    parser.add_argument(
        "--sink-grid", action="store_true",
        help="compare zero, one, and two head-shared sink pages at 60% ordinary pages",
    )
    parser.add_argument("--enable-shared-page-attention", action="store_true")
    parser.add_argument("--spa-group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--spa-support", choices=("columns", "pages", "hybrid"), default="pages")
    parser.add_argument("--spa-density", type=float, default=0.5)
    parser.add_argument(
        "--spa-selector",
        choices=("oracle-mean", "oracle-tail", "oracle-coverage"),
        default="oracle-mean",
    )
    parser.add_argument("--spa-sink-pages", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--spa-tail-beta", type=float, default=1.0)
    parser.add_argument("--spa-tail-quantile", type=float, default=0.95)
    parser.add_argument(
        "--spa-coverage-objective",
        choices=("minimum", "p1", "p5", "mean-min"), default="p1",
    )
    parser.add_argument("--spa-mandatory-local", action="store_true")
    parser.add_argument("--spa-mandatory-first", action="store_true")
    parser.add_argument("--spa-mandatory-last", action="store_true")
    parser.add_argument("--spa-refresh", choices=("current", "first", "periodic"), default="current")
    parser.add_argument("--spa-refresh-interval", type=int, default=2)
    parser.add_argument("--spa-dense-early-steps", type=int, default=0)
    args = parser.parse_args()
    if not args.enable_shared_page_attention:
        raise ValueError("SPA experiments require explicit --enable-shared-page-attention opt-in")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaDA SPA evaluation")
    if args.num_contexts < 1 or args.num_contexts > len(PROMPTS):
        raise ValueError(f"num-contexts must be in [1, {len(PROMPTS)}]")
    if args.steps < 1 or args.spa_refresh_interval < 1:
        raise ValueError("steps and refresh interval must be positive")
    configs = oracle_grid(args)
    if args.mode == "quality" and len(configs) != 1:
        raise ValueError("quality mode accepts one SPA configuration, not an ablation grid")

    model, tokenizer = load_model(args.model)
    mask_id = choose_mask_token_id(tokenizer, None)
    payload = {
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "mode": args.mode,
        "context_length": args.context_length,
        "steps": args.steps,
        "seeds": args.seeds,
        "num_contexts": args.num_contexts,
        "configs": [{"name": config_name(config), **config.__dict__} for config in configs],
        "methodology": {
            "oracle": "Current-step support uses exact normalized QK softmax mass.",
            "sparse_reference": "Quality mode recomputes scores and softmax over retained keys only.",
            "isolation": "Supports are per request and head unless explicitly marked sharedheads ablation.",
        },
    }
    payload.update(
        run_oracle(args, model, tokenizer, mask_id, configs)
        if args.mode == "oracle"
        else run_quality(args, model, tokenizer, mask_id, configs[0])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = payload.get("oracle_summary", payload.get("quality_summary"))
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
