#!/usr/bin/env python3
"""Masked-token LLaDA evaluation of the counterfactual reference path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule
from llada_eval_utils import (
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)
from collect_proxy_traces import cyclic_contexts
from proxy import (
    BinaryReusePolicy,
    DisabledPolicy,
    MaxScorePolicy,
    MetadataStore,
    SafetyPolicy,
    ScoreThresholdPolicy,
    install_proxy_blasst_reference,
    load_calibrated_thresholds,
)
from proxy.proxy_policy import ProxyContext


def bucket(ratio: float) -> str:
    return "high" if ratio >= 0.75 else "mid" if ratio >= 0.25 else "low"


def oracle_policy_diagnostics(
    policy: object,
    sources: MetadataStore,
    targets: MetadataStore,
    *,
    context_length: int,
    iteration: int,
) -> dict[str, float | int]:
    num_kv_tiles = math.ceil(context_length / 64)
    predicted = correct = false = false_newmax = exact_skips = 0
    raw_binary = safe_binary = raw_binary_correct = safe_binary_correct = 0
    anchor_suppressed = 0
    safety = policy if isinstance(policy, SafetyPolicy) else None
    for key, target in targets.records.items():
        sample_id, step, layer, head, query_tile, kv_tile = key
        if step != iteration:
            continue
        # Use the target's recorded bucket rather than assuming the caller's ratio.
        context = ProxyContext(
            sample_id, layer, head, head, step, target.noise_bucket or "mid", query_tile, kv_tile,
            num_kv_tiles - 1 - kv_tile, num_kv_tiles,
        )
        source = sources.resolve(context, "previous_step")
        exact_skips += int(target.skipped)
        binary = source is not None and source.skipped
        raw_binary += int(binary)
        raw_binary_correct += int(binary and target.skipped)
        forced = safety.forces_exact(context) if safety is not None else False
        safe_binary += int(binary and not forced)
        safe_binary_correct += int(binary and not forced and target.skipped)
        anchor_suppressed += int(binary and forced)
        decision = policy.decide(context, sources)  # type: ignore[attr-defined]
        if decision.value == "pre_skip":
            predicted += 1
            correct += int(target.skipped)
            false += int(not target.skipped)
            false_newmax += int((not target.skipped) and target.introduced_new_max)
    total = sum(1 for key in targets.records if key[1] == iteration)
    ratio = lambda a, b: a / b if b else 0.0
    return {
        "total_tiles": total,
        "oracle_exact_skip_tiles": exact_skips,
        "oracle_exact_skip_rate": ratio(exact_skips, total),
        "policy_candidates": predicted,
        "policy_candidate_rate": ratio(predicted, total),
        "policy_precision_with_exact_sources": ratio(correct, predicted),
        "policy_oracle_skip_recall": ratio(correct, exact_skips),
        "policy_false_skips": false,
        "policy_false_new_max": false_newmax,
        "raw_previous_binary_candidates": raw_binary,
        "raw_previous_binary_precision": ratio(raw_binary_correct, raw_binary),
        "raw_previous_binary_recall": ratio(raw_binary_correct, exact_skips),
        "safe_previous_binary_candidates": safe_binary,
        "safe_previous_binary_precision": ratio(safe_binary_correct, safe_binary),
        "anchor_suppressed_binary_candidates": anchor_suppressed,
        "target_skips_not_recalled_by_previous_binary": exact_skips - raw_binary_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--previous-mask-ratio", type=float, default=0.9)
    parser.add_argument("--current-mask-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260815, help="held-out seed; differs from trace collection")
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument("--mode", choices=("oracle", "realistic"), default="oracle")
    parser.add_argument("--policy", choices=("binary", "score", "max_score"), default="binary")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--false-skip-budget", type=float, default=1e-3)
    parser.add_argument("--warmup-tiles", type=int, default=2)
    parser.add_argument("--periodic-refresh", type=int, default=8)
    parser.add_argument("--no-local-anchor", action="store_true")
    parser.add_argument("--no-diagonal-anchor", action="store_true")
    parser.add_argument("--no-sink-anchor", action="store_true")
    parser.add_argument("--allow-zero-coverage", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/proxy_llada_one_step.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA evaluation requires CUDA")
    if args.previous_mask_ratio < args.current_mask_ratio:
        raise ValueError("previous mask ratio must be at least the current ratio")
    if args.policy == "binary":
        inner_policy = BinaryReusePolicy("previous_step")
        calibrated_threshold_count = 0
    else:
        if args.calibration is None:
            raise ValueError("--calibration is required for continuous-score policies")
        calibration_relation = (
            "previous_step" if args.policy == "score" else "max_previous_step_previous_layer"
        )
        threshold_table = load_calibrated_thresholds(
            args.calibration,
            relation=calibration_relation,
            false_skip_budget=args.false_skip_budget,
        )
        calibrated_threshold_count = len(threshold_table.values)
        if calibrated_threshold_count == 0 and not args.allow_zero_coverage:
            raise ValueError(
                f"no certified {calibration_relation} thresholds at budget {args.false_skip_budget:g}; "
                "collect more data or use a looser budget"
            )
        inner_policy = (
            ScoreThresholdPolicy("previous_step", threshold_table)
            if args.policy == "score"
            else MaxScorePolicy(("previous_step", "previous_layer"), threshold_table)
        )
    policy = SafetyPolicy(
        inner_policy,
        warmup_tiles=args.warmup_tiles,
        periodic_refresh=args.periodic_refresh,
        anchor_local=not args.no_local_anchor,
        anchor_diagonal=not args.no_diagonal_anchor,
        anchor_sink=not args.no_sink_anchor,
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
    text = (ROOT / "README.md").read_text(encoding="utf-8") + (ROOT / "docs" / "blasst.md").read_text(encoding="utf-8")
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(tokens) < args.context_length:
        repeats = (args.context_length + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeats)[: args.context_length]
    clean = cyclic_contexts(tokens, args.context_length, args.num_contexts)
    priority = torch.stack([
        torch.rand(args.context_length, generator=torch.Generator().manual_seed(args.seed + index))
        for index in range(args.num_contexts)
    ])
    mask_id = choose_mask_token_id(tokenizer, None)

    def state(ratio: float) -> torch.Tensor:
        result = clean.clone()
        result[priority < ratio] = mask_id
        return result.cuda()

    previous_inputs, current_inputs = state(args.previous_mask_ratio), state(args.current_mask_ratio)
    current_mask = (priority < args.current_mask_ratio).cuda()
    sample_ids = [f"llada-eval-{index}" for index in range(args.num_contexts)]
    schedule = DiffusionLambdaSchedule()

    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            result = model(input_ids=inputs, output_hidden_states=True)
        return result.logits, result.hidden_states[-1]

    dense_logits, dense_hidden = forward(current_inputs)
    exact_sources = MetadataStore()
    with install_proxy_blasst_reference(
        model,
        blasst_lambda=schedule.threshold(args.previous_mask_ratio),
        policy=DisabledPolicy(),
        output_store=exact_sources,
        sample_ids=sample_ids,
    ) as controller:
        controller.set_step(0, bucket(args.previous_mask_ratio))
        forward(previous_inputs)
    exact_targets = MetadataStore()
    with install_proxy_blasst_reference(
        model,
        blasst_lambda=schedule.threshold(args.current_mask_ratio),
        policy=DisabledPolicy(),
        output_store=exact_targets,
        sample_ids=sample_ids,
    ) as controller:
        controller.set_step(1, bucket(args.current_mask_ratio))
        exact_logits, exact_hidden = forward(current_inputs)

    combined_exact_sources = MetadataStore(records={**exact_sources.records, **exact_targets.records})
    oracle_diagnostics = oracle_policy_diagnostics(
        policy, combined_exact_sources, exact_targets,
        context_length=args.context_length, iteration=1,
    )
    if args.mode == "oracle":
        sources, emitted = combined_exact_sources, MetadataStore()
    else:
        sources = emitted = MetadataStore()
        with install_proxy_blasst_reference(
            model,
            blasst_lambda=schedule.threshold(args.previous_mask_ratio),
            policy=policy,
            source_store=sources,
            output_store=emitted,
            sample_ids=sample_ids,
        ) as controller:
            controller.set_step(0, bucket(args.previous_mask_ratio))
            forward(previous_inputs)
    with install_proxy_blasst_reference(
        model,
        blasst_lambda=schedule.threshold(args.current_mask_ratio),
        policy=policy,
        source_store=sources,
        output_store=emitted,
        oracle_targets=exact_targets,
        sample_ids=sample_ids,
    ) as controller:
        controller.set_step(1, bucket(args.current_mask_ratio))
        proxy_logits, proxy_hidden = forward(current_inputs)
        proxy_stats = controller.stats.summary() if controller.stats is not None else {}

    selected_proxy = proxy_logits[current_mask].float()
    selected_exact = exact_logits[current_mask].float()
    selected_dense = dense_logits[current_mask].float()
    per_sample = []
    for sample in range(args.num_contexts):
        selected = current_mask[sample]
        per_sample.append({
            "sample_id": sample_ids[sample],
            "masked_tokens": int(selected.sum()),
            "agreement_vs_exact_blasst": float((
                proxy_logits[sample, selected].argmax(-1) == exact_logits[sample, selected].argmax(-1)
            ).float().mean()),
            "agreement_vs_dense": float((
                proxy_logits[sample, selected].argmax(-1) == dense_logits[sample, selected].argmax(-1)
            ).float().mean()),
        })
    result = {
        "model": args.model,
        "mode": args.mode,
        "policy": args.policy,
        "false_skip_budget": args.false_skip_budget if args.policy != "binary" else None,
        "calibrated_threshold_count": calibrated_threshold_count,
        "previous_mask_ratio": args.previous_mask_ratio,
        "current_mask_ratio": args.current_mask_ratio,
        "num_contexts": args.num_contexts,
        "masked_tokens": int(current_mask.sum()),
        "masked_top1_agreement_vs_exact_blasst": float((selected_proxy.argmax(-1) == selected_exact.argmax(-1)).float().mean()),
        "masked_top1_agreement_vs_dense": float((selected_proxy.argmax(-1) == selected_dense.argmax(-1)).float().mean()),
        "exact_blasst_agreement_vs_dense": float((selected_exact.argmax(-1) == selected_dense.argmax(-1)).float().mean()),
        "final_logit_kl_vs_exact_blasst": max(0.0, float(torch.nn.functional.kl_div(
            selected_proxy.log_softmax(-1), selected_exact.softmax(-1), reduction="batchmean"
        ))),
        "hidden_state_cosine_vs_exact_blasst": float(torch.nn.functional.cosine_similarity(
            proxy_hidden.float().flatten(), exact_hidden.float().flatten(), dim=0
        )),
        "hidden_state_cosine_vs_dense": float(torch.nn.functional.cosine_similarity(
            proxy_hidden.float().flatten(), dense_hidden.float().flatten(), dim=0
        )),
        "proxy_stats": proxy_stats,
        "oracle_source_diagnostics": oracle_diagnostics,
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
