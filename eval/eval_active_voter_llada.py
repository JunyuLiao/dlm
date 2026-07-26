#!/usr/bin/env python3
"""Quality evaluation for the opt-in Active-Voter BLASST approximation.

The Triton row-masked path used here is a quality-only diagnostic: it masks the
online-softmax recurrence but still executes dense tile dot products.  It makes
no packed-kernel or speedup claim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import sys
from typing import Iterator

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import (  # noqa: E402
    DiffusionLambdaSchedule,
    blasst_bidirectional_flash_attn_func,
    get_kernel_stats,
    install_bidirectional_blasst_kernel,
    reset_kernel_stats,
)
from collect_proxy_traces import corpus, cyclic_contexts  # noqa: E402
from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


def parse_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one value is required")
    return result


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    left = candidate.float().flatten()
    right = reference.float().flatten()
    difference = left - right
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(right).clamp_min(1e-12)
        ),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
    }


def distribution_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    selected: torch.Tensor,
    *,
    chunk: int = 128,
) -> dict[str, float]:
    candidate = candidate[selected]
    reference = reference[selected]
    count = int(candidate.shape[0])
    top1_equal = 0
    top5_equal = 0
    cosine_sum = 0.0
    kl_sum = 0.0
    absolute_sum = 0.0
    absolute_max = 0.0
    difference_square = 0.0
    reference_square = 0.0
    for start in range(0, count, chunk):
        left = candidate[start : start + chunk].float()
        right = reference[start : start + chunk].float()
        difference = left - right
        top1_equal += int((left.argmax(-1) == right.argmax(-1)).sum())
        left_top5 = left.topk(5, dim=-1).indices
        right_top1 = right.argmax(-1, keepdim=True)
        top5_equal += int((left_top5 == right_top1).any(-1).sum())
        cosine_sum += float(torch.nn.functional.cosine_similarity(left, right, dim=-1).sum())
        kl_sum += float(
            torch.nn.functional.kl_div(
                left.log_softmax(-1), right.softmax(-1), reduction="sum"
            )
        )
        absolute_sum += float(difference.abs().sum())
        absolute_max = max(absolute_max, float(difference.abs().max()))
        difference_square += float(difference.square().sum())
        reference_square += float(right.square().sum())
    denominator = max(count, 1)
    elements = max(count * candidate.shape[-1], 1)
    return {
        "tokens": count,
        "top1_agreement": top1_equal / denominator,
        "reference_top1_in_candidate_top5": top5_equal / denominator,
        "mean_logit_cosine_similarity": cosine_sum / denominator,
        "mean_logit_kl_divergence": max(0.0, kl_sum / denominator),
        "maximum_absolute_logit_error": absolute_max,
        "mean_absolute_logit_error": absolute_sum / elements,
        "relative_logit_l2_error": math.sqrt(difference_square / max(reference_square, 1e-30)),
    }


def hidden_metrics(
    candidate: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]
) -> list[dict[str, float]]:
    return [
        {"hidden_state": layer, **tensor_error(left, right)}
        for layer, (left, right) in enumerate(zip(candidate, reference, strict=True))
    ]


def per_head_error(
    candidate: torch.Tensor, reference: torch.Tensor, selected_rows: torch.Tensor
) -> dict[str, list[float]]:
    # Inputs are [B, S, H, D]; reductions preserve H.
    selected = selected_rows[:, :, None, None]
    difference = torch.where(selected, candidate.float() - reference.float(), 0.0)
    reference_selected = torch.where(selected, reference.float(), 0.0)
    count = selected_rows.sum().clamp_min(1) * candidate.shape[-1]
    numerator = difference.square().sum((0, 1, 3))
    denominator = reference_selected.square().sum((0, 1, 3)).clamp_min(1e-30)
    dot = (torch.where(selected, candidate.float(), 0.0) * reference_selected).sum((0, 1, 3))
    left_norm = torch.where(selected, candidate.float(), 0.0).square().sum((0, 1, 3)).sqrt()
    right_norm = denominator.sqrt()
    return {
        "maximum_absolute_error": difference.abs().amax((0, 1, 3)).cpu().tolist(),
        "mean_absolute_error": (difference.abs().sum((0, 1, 3)) / count).cpu().tolist(),
        "relative_l2_error": (numerator / denominator).sqrt().cpu().tolist(),
        "cosine_similarity": (dot / (left_norm * right_norm).clamp_min(1e-30)).cpu().tolist(),
    }


@contextmanager
def attention_diagnostics(
    model: torch.nn.Module,
    *,
    tile_lambda: float,
    row_lambda: float,
    max_active_rows: int,
    masked_rows: torch.Tensor,
    records: list[dict[str, object]],
) -> Iterator[None]:
    """Return AV outputs while comparing attention outputs on identical Q/K/V."""
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    originals = [(module, module.flash_attn_func) for module in modules]
    for layer, (module, dense_function) in enumerate(originals):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer: int = layer,
            _dense=dense_function,
            **kwargs: object,
        ) -> torch.Tensor:
            dense = _dense(q, k, v, **kwargs)
            baseline = blasst_bidirectional_flash_attn_func(
                q, k, v, **kwargs, blasst_lambda=tile_lambda, collect_stats=False
            )
            candidate = blasst_bidirectional_flash_attn_func(
                q,
                k,
                v,
                **kwargs,
                blasst_lambda=tile_lambda,
                row_lambda=row_lambda,
                row_mask_variant="full",
                max_active_rows=max_active_rows,
                collect_stats=False,
            )
            for state_name, rows in (
                ("masked", masked_rows),
                ("revealed", ~masked_rows),
            ):
                if not bool(rows.any()):
                    continue
                for comparison, left, right in (
                    ("active_vs_blasst", candidate, baseline),
                    ("active_vs_dense", candidate, dense),
                    ("blasst_vs_dense", baseline, dense),
                ):
                    metrics = per_head_error(left, right, rows)
                    for head in range(q.shape[2]):
                        records.append(
                            {
                                "layer": _layer,
                                "head": head,
                                "token_state": state_name,
                                "comparison": comparison,
                                **{name: values[head] for name, values in metrics.items()},
                            }
                        )
            return candidate

        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original


def kernel_stats_dict() -> dict[str, float | int]:
    stats = get_kernel_stats()
    return {
        "retained_tiles": stats.total_tiles - stats.skipped_tiles,
        "candidate_tiles": stats.active_voter_candidate_tiles,
        "candidate_row_work": stats.active_voter_candidate_row_work,
        "selected_rows": stats.active_voter_selected_rows,
        "removed_rows": stats.active_voter_removed_rows,
        "retained_row_work": stats.active_voter_retained_row_work,
        "candidate_work_fraction": stats.active_voter_candidate_work_fraction,
        "selected_fraction_within_candidates": stats.active_voter_selected_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--ratios", default="0.90,0.50,0.15")
    parser.add_argument("--margins", default="0,0.5,1,2,4,6,8")
    parser.add_argument("--max-active-rows", type=int, default=32)
    parser.add_argument("--row-variant", choices=("full", "output"), default="full")
    parser.add_argument(
        "--active-layers",
        default="all",
        help="comma-separated zero-based layers; 'all' enables every layer",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trajectory-steps", type=int, default=16)
    parser.add_argument("--run-trajectories", action="store_true")
    parser.add_argument("--paired-trajectory", action="store_true")
    parser.add_argument("--attention-diagnostics", action="store_true")
    parser.add_argument(
        "--blasst-row-masking",
        action="store_true",
        help="explicit opt-in for Active-Voter BLASST evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.blasst_row_masking:
        raise ValueError("Active-Voter evaluation requires --blasst-row-masking")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ratios = parse_floats(args.ratios)
    margins = parse_floats(args.margins)
    active_layers = (
        None
        if args.active_layers == "all"
        else {int(value) for value in args.active_layers.split(",") if value.strip()}
    )
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
    clean = cyclic_contexts(
        tokenizer(corpus(), add_special_tokens=False)["input_ids"],
        args.context_length,
        1,
    ).cuda()
    mask_id = choose_mask_token_id(tokenizer, None)
    priority = torch.rand(
        (1, args.context_length),
        generator=torch.Generator().manual_seed(args.seed),
        device="cpu",
    ).cuda()
    schedule = DiffusionLambdaSchedule()

    def state_for_ratio(ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        masked = priority < ratio
        state = clean.clone()
        state[masked] = mask_id
        return state, masked

    def forward_dense(state: torch.Tensor):
        with torch.inference_mode():
            return model(input_ids=state, output_hidden_states=True)

    def forward_blasst(state: torch.Tensor, threshold: float):
        with install_bidirectional_blasst_kernel(
            model, blasst_lambda=threshold, collect_stats=False
        ):
            with torch.inference_mode():
                return model(input_ids=state, output_hidden_states=True)

    def forward_active(state: torch.Tensor, threshold: float, margin: float):
        row_threshold = threshold * math.exp(-margin)
        reset_kernel_stats(state.device)
        with install_bidirectional_blasst_kernel(
            model,
            blasst_lambda=threshold,
            row_mask_variant=args.row_variant,
            row_lambda=row_threshold,
            max_active_rows=args.max_active_rows,
            active_voter_layers=active_layers,
            collect_stats=True,
        ):
            with torch.inference_mode():
                output = model(input_ids=state, output_hidden_states=True)
        return output, kernel_stats_dict()

    screening: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    for ratio in ratios:
        state, masked = state_for_ratio(ratio)
        threshold = float(schedule.threshold(ratio))
        dense = forward_dense(state)
        baseline = forward_blasst(state, threshold)
        dense_masked = distribution_metrics(baseline.logits, dense.logits, masked)
        for margin in margins:
            candidate, work = forward_active(state, threshold, margin)
            screening.append(
                {
                    "remaining_mask_ratio": ratio,
                    "tile_lambda": threshold,
                    "row_margin": margin,
                    "row_lambda": threshold * math.exp(-margin),
                    "active_vs_blasst_masked_logits": distribution_metrics(
                        candidate.logits, baseline.logits, masked
                    ),
                    "active_vs_dense_masked_logits": distribution_metrics(
                        candidate.logits, dense.logits, masked
                    ),
                    "blasst_vs_dense_masked_logits": dense_masked,
                    "active_vs_blasst_hidden_states": hidden_metrics(
                        candidate.hidden_states, baseline.hidden_states
                    ),
                    "active_vs_dense_hidden_states": hidden_metrics(
                        candidate.hidden_states, dense.hidden_states
                    ),
                    "work": work,
                }
            )
            del candidate
        del dense, baseline
        gc.collect()
        torch.cuda.empty_cache()

    by_margin: dict[float, list[dict[str, object]]] = {
        margin: [row for row in screening if row["row_margin"] == margin]
        for margin in margins
    }
    eligible = [
        margin
        for margin, rows in by_margin.items()
        if min(
            row["active_vs_blasst_masked_logits"]["top1_agreement"]  # type: ignore[index]
            for row in rows
        )
        >= 0.99
    ]
    chosen_margin = min(eligible) if eligible else max(
        margins,
        key=lambda margin: min(
            row["active_vs_blasst_masked_logits"]["top1_agreement"]  # type: ignore[index]
            for row in by_margin[margin]
        ),
    )

    if args.attention_diagnostics:
        for ratio in ratios:
            state, masked = state_for_ratio(ratio)
            threshold = float(schedule.threshold(ratio))
            before = len(attention_rows)
            with attention_diagnostics(
                model,
                tile_lambda=threshold,
                row_lambda=threshold * math.exp(-chosen_margin),
                max_active_rows=args.max_active_rows,
                masked_rows=masked,
                records=attention_rows,
            ):
                with torch.inference_mode():
                    model(input_ids=state)
            for row in attention_rows[before:]:
                row["remaining_mask_ratio"] = ratio
                row["row_margin"] = chosen_margin

    trajectories: dict[str, object] = {}
    if args.run_trajectories:
        methods = ["dense", "blasst", "active_raw", "active_chosen"]

        def denoise(method: str) -> dict[str, object]:
            state, _ = state_for_ratio(0.90)
            initial_mask = state.eq(mask_id)
            steps: list[dict[str, object]] = []
            for step in range(args.trajectory_steps):
                masked = state.eq(mask_id)
                remaining = int(masked.sum())
                if not remaining:
                    break
                ratio = remaining / args.context_length
                threshold = float(schedule.threshold(ratio))
                if method == "dense":
                    output = forward_dense(state)
                    work = None
                elif method == "blasst":
                    output = forward_blasst(state, threshold)
                    work = None
                else:
                    margin = 0.0 if method == "active_raw" else chosen_margin
                    output, work = forward_active(state, threshold, margin)
                logits = output.logits[masked].float()
                probabilities = logits.softmax(-1)
                confidence, token = probabilities.max(-1)
                target = round(int(initial_mask.sum()) * (1.0 - (step + 1) / args.trajectory_steps))
                reveal = max(1, remaining - target)
                selected = confidence.topk(min(reveal, remaining)).indices
                positions = masked.nonzero()[selected]
                state[positions[:, 0], positions[:, 1]] = token[selected]
                steps.append(
                    {
                        "step": step,
                        "remaining_mask_ratio": ratio,
                        "revealed": int(selected.numel()),
                        "work": work,
                    }
                )
                del output, logits, probabilities
            return {"tokens": state.cpu(), "steps": steps}

        raw = {method: denoise(method) for method in methods}
        reference_tokens = raw["blasst"]["tokens"]
        dense_tokens = raw["dense"]["tokens"]
        initially_masked = (priority < 0.90).cpu()
        for method, result in raw.items():
            tokens = result.pop("tokens")
            result["final_token_agreement_vs_blasst"] = float(
                (tokens == reference_tokens).float().mean()
            )
            result["final_token_agreement_vs_dense"] = float(
                (tokens == dense_tokens).float().mean()
            )
            result["exact_sequence_vs_blasst"] = bool(torch.equal(tokens, reference_tokens))
            result["initially_masked_token_agreement_vs_blasst"] = float(
                (tokens[initially_masked] == reference_tokens[initially_masked]).float().mean()
            )
            result["initially_masked_token_agreement_vs_dense"] = float(
                (tokens[initially_masked] == dense_tokens[initially_masked]).float().mean()
            )
            result["decoded_prefix"] = tokenizer.decode(tokens[0, :256], skip_special_tokens=True)
        trajectories = raw

    paired_steps: list[dict[str, object]] = []
    if args.paired_trajectory:
        state, initial_mask = state_for_ratio(0.90)
        initial_count = int(initial_mask.sum())
        for step in range(args.trajectory_steps):
            masked = state.eq(mask_id)
            remaining = int(masked.sum())
            if not remaining:
                break
            ratio = remaining / args.context_length
            threshold = float(schedule.threshold(ratio))
            dense = forward_dense(state)
            baseline = forward_blasst(state, threshold)
            raw, raw_work = forward_active(state, threshold, 0.0)
            chosen, chosen_work = forward_active(state, threshold, chosen_margin)
            row: dict[str, object] = {
                "step": step,
                "remaining_mask_ratio": ratio,
                "raw_work": raw_work,
                "chosen_work": chosen_work,
            }
            for name, output in (("active_raw", raw), ("active_chosen", chosen)):
                row[f"{name}_vs_blasst_masked_logits"] = distribution_metrics(
                    output.logits, baseline.logits, masked
                )
                row[f"{name}_vs_dense_masked_logits"] = distribution_metrics(
                    output.logits, dense.logits, masked
                )
                row[f"{name}_vs_blasst_hidden_states"] = hidden_metrics(
                    output.hidden_states, baseline.hidden_states
                )
            row["blasst_vs_dense_masked_logits"] = distribution_metrics(
                baseline.logits, dense.logits, masked
            )
            logits = baseline.logits[masked].float()
            confidence, token = logits.softmax(-1).max(-1)
            target = round(initial_count * (1.0 - (step + 1) / args.trajectory_steps))
            reveal = max(1, remaining - target)
            selected = confidence.topk(min(reveal, remaining)).indices
            positions = masked.nonzero()[selected]
            state[positions[:, 0], positions[:, 1]] = token[selected]
            row["revealed"] = int(selected.numel())
            paired_steps.append(row)
            del dense, baseline, raw, chosen, logits

    payload = {
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "context_length": args.context_length,
        "seed": args.seed,
        "max_active_rows": args.max_active_rows,
        "row_variant": args.row_variant,
        "active_layers": "all" if active_layers is None else sorted(active_layers),
        "screened_margins": margins,
        "quality_eligible_margins_at_sampled_states": eligible,
        "chosen_margin": chosen_margin,
        "screening": screening,
        "attention_output_by_layer_head": attention_rows,
        "trajectories": trajectories,
        "teacher_forced_blasst_trajectory": paired_steps,
        "diagnostic_kernel_note": (
            "The row-masked Triton path evaluates quality semantics but still executes dense tile dots."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "eligible": eligible,
        "chosen_margin": chosen_margin,
        "screening_rows": len(screening),
        "attention_rows": len(attention_rows),
        "trajectory_methods": list(trajectories),
        "teacher_forced_steps": len(paired_steps),
    }, indent=2))


if __name__ == "__main__":
    main()
