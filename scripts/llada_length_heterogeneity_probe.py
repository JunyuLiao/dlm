#!/usr/bin/env python3
"""Probe LLaDA serving behavior under prompt-length heterogeneity.

This runner constructs synthetic batches whose total prompt-token budget is
fixed, but whose per-request prompt lengths range from equal to extremely
heterogeneous. It records per-request/block denoising steps and per-step shared
batch latency attribution for each request.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_block_step_probe import (
    append_mask_block,
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    get_num_transfer_tokens,
    patch_llada_transformers_compat,
)


@dataclass(frozen=True)
class LengthPromptRecord:
    group_name: str
    request_id: int
    target_prompt_tokens: int
    actual_prompt_tokens: int
    prompt: str


@dataclass(frozen=True)
class LengthRequestBlockRow:
    run_id: str
    group_name: str
    batch_size: int
    block_size: int
    trial: int
    batch_id: int
    request_id: int
    block_index: int
    target_prompt_tokens: int
    actual_prompt_tokens: int
    total_batch_target_prompt_tokens: int
    total_batch_actual_prompt_tokens: int
    batch_max_prompt_tokens: int
    padding_tokens_for_request: int
    num_blocks: int
    max_steps_per_block: int
    confidence_threshold: float
    acceptance_policy: str
    steps_used: int
    steps_executed: int
    request_latency_ms: float
    batch_block_latency_ms: float
    useful_token_steps: int
    executed_token_steps: int
    token_step_min: int
    token_step_mean: float
    token_step_max: int
    mean_confidence: float
    min_confidence: float
    finalized_by_threshold: int
    finalized_by_force: int
    hit_max_steps: bool
    block_internal_waste_ratio: float
    num_tokens: int
    tokens_generated: int


@dataclass(frozen=True)
class LengthBatchStepRow:
    run_id: str
    group_name: str
    batch_size: int
    block_size: int
    trial: int
    batch_id: int
    block_index: int
    step: int
    step_latency_ms: float
    active_requests_before_step: int
    active_requests_after_step: int
    finished_requests_after_step: int
    accepted_tokens_this_step: int
    unfinished_tokens_before_step: int
    unfinished_tokens_after_step: int
    padded_sequence_length: int
    max_logical_sequence_length_before_step: int


@dataclass(frozen=True)
class LengthRequestStepRow:
    run_id: str
    group_name: str
    batch_size: int
    block_size: int
    trial: int
    batch_id: int
    request_id: int
    block_index: int
    step: int
    target_prompt_tokens: int
    actual_prompt_tokens: int
    padding_tokens_for_request: int
    request_active_before_step: bool
    request_active_after_step: bool
    request_finished_after_step: bool
    accepted_tokens_this_step: int
    unfinished_tokens_before_step: int
    unfinished_tokens_after_step: int
    shared_batch_step_latency_ms: float
    dense_request_step_latency_ms: float
    active_request_step_latency_ms: float
    padded_sequence_length: int
    logical_sequence_length_before_step: int


def length_targets(batch_size: int, total_prompt_tokens: int) -> dict[str, list[int]]:
    if total_prompt_tokens < batch_size:
        raise ValueError("total_prompt_tokens must be at least batch_size")

    equal_base = total_prompt_tokens // batch_size
    equal = [equal_base] * batch_size
    equal[-1] += total_prompt_tokens - sum(equal)

    if batch_size == 4:
        moderate_template = [300, 400, 600, 700]
        scale = total_prompt_tokens / sum(moderate_template)
        moderate = [max(1, round(value * scale)) for value in moderate_template]
    elif batch_size == 8:
        moderate_template = [250, 300, 400, 450, 550, 650, 700, 700]
        scale = total_prompt_tokens / sum(moderate_template)
        moderate = [max(1, round(value * scale)) for value in moderate_template]
    else:
        weights = [0.55 + (0.9 * i / max(1, batch_size - 1)) for i in range(batch_size)]
        scale = total_prompt_tokens / sum(weights)
        moderate = [max(1, round(w * scale)) for w in weights]
    moderate[-1] += total_prompt_tokens - sum(moderate)

    short = min(100, max(1, total_prompt_tokens // (batch_size * 4)))
    extreme = [short] * (batch_size - 1)
    extreme.append(total_prompt_tokens - sum(extreme))

    return {
        "equal": equal,
        "moderate": moderate,
        "extreme": extreme,
    }


NEUTRAL_PROMPT_UNITS = [
    (
        "This neutral benchmark passage discusses sequence length, attention masks, "
        "batched forward passes, token confidence, and request completion. "
    ),
    (
        "The serving trace records padding, prompt tokens, denoising iterations, "
        "shared step latency, and block-level completion behavior. "
    ),
    (
        "A diffusion language model repeatedly refines masked positions while the "
        "batch shape is determined by the longest visible sequence. "
    ),
    (
        "The measurement focuses on runtime mechanics rather than domain knowledge, "
        "using plain context about scheduling, latency, and token finalization. "
    ),
    (
        "Each request contains neutral technical prose so length heterogeneity can be "
        "studied without changing the intended difficulty category. "
    ),
    (
        "The batch contains requests with controlled prompt lengths, fixed block size, "
        "and comparable wording about inference system behavior. "
    ),
]


def make_prompt_for_target(
    tokenizer: AutoTokenizer,
    group_name: str,
    request_id: int,
    trial: int,
    target_tokens: int,
) -> tuple[str, int]:
    prefix = (
        f"Request group {group_name}, trial {trial}, request {request_id}. "
        "Read the following neutral serving-context passage and continue with a concise technical answer. "
    )
    offset = (trial * 3 + request_id) % len(NEUTRAL_PROMPT_UNITS)
    units = NEUTRAL_PROMPT_UNITS[offset:] + NEUTRAL_PROMPT_UNITS[:offset]
    unit = "".join(units)
    text = prefix + unit * max(8, target_tokens // 8)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(token_ids) < target_tokens + 64:
        text += unit * max(8, target_tokens // 16)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    special_overhead = len(tokenizer("", add_special_tokens=True)["input_ids"])
    body_target = max(1, target_tokens - special_overhead)
    prompt = tokenizer.decode(token_ids[:body_target], skip_special_tokens=True)
    actual = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    return prompt, actual


def build_length_prompt_batch(
    tokenizer: AutoTokenizer,
    group_name: str,
    targets: list[int],
    trial: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[LengthPromptRecord]]:
    records: list[LengthPromptRecord] = []
    encoded: list[list[int]] = []
    for request_id, target in enumerate(targets):
        prompt, actual = make_prompt_for_target(tokenizer, group_name, request_id, trial, target)
        records.append(
            LengthPromptRecord(
                group_name=group_name,
                request_id=request_id,
                target_prompt_tokens=target,
                actual_prompt_tokens=actual,
                prompt=prompt,
            )
        )
        encoded.append(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    max_prompt = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = [ids + [pad_id] * (max_prompt - len(ids)) for ids in encoded]
    masks = [[1] * len(ids) + [0] * (max_prompt - len(ids)) for ids in encoded]
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        records,
    )


def write_dataclass_rows(rows: list[object], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))  # type: ignore[arg-type]
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))  # type: ignore[arg-type]


def write_length_plot_summary(
    request_block_rows: list[LengthRequestBlockRow],
    batch_step_rows: list[LengthBatchStepRow],
    path: Path,
) -> list[dict[str, object]]:
    block_steps: dict[tuple[str, int, int], list[int]] = {}
    seen_blocks: set[tuple[int, int]] = set()
    for row in request_block_rows:
        block_key = (row.batch_id, row.block_index)
        if block_key in seen_blocks:
            continue
        seen_blocks.add(block_key)
        key = (row.group_name, row.batch_size, row.block_size)
        block_steps.setdefault(key, []).append(row.steps_executed)

    step_latencies: dict[tuple[str, int, int], list[float]] = {}
    for row in batch_step_rows:
        key = (row.group_name, row.batch_size, row.block_size)
        step_latencies.setdefault(key, []).append(row.step_latency_ms)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(set(block_steps) | set(step_latencies)):
        group_name, batch_size, block_size = key
        steps = block_steps.get(key, [])
        latencies = step_latencies.get(key, [])
        summary_rows.append(
            {
                "group_name": group_name,
                "batch_size": batch_size,
                "block_size": block_size,
                "mean_block_steps": mean(steps) if steps else 0.0,
                "num_block_samples": len(steps),
                "mean_step_latency_ms": mean(latencies) if latencies else 0.0,
                "num_step_samples": len(latencies),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "group_name",
            "batch_size",
            "block_size",
            "mean_block_steps",
            "num_block_samples",
            "mean_step_latency_ms",
            "num_step_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def plot_length_group_figures(summary_rows: list[dict[str, object]], out_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped length heterogeneity PNG plots.")
        return False

    group_order = ["equal", "moderate", "extreme"]
    series = sorted({(int(row["batch_size"]), int(row["block_size"])) for row in summary_rows})
    if not summary_rows or not series:
        return False

    def metric_for(group_name: str, batch_size: int, block_size: int, metric: str) -> float:
        for row in summary_rows:
            if (
                row["group_name"] == group_name
                and int(row["batch_size"]) == batch_size
                and int(row["block_size"]) == block_size
            ):
                return float(row[metric])
        return 0.0

    def grouped_bar(metric: str, ylabel: str, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        xs = list(range(len(group_order)))
        width = min(0.12, 0.75 / max(1, len(series)))
        offsets = [(index - (len(series) - 1) / 2) * width for index in range(len(series))]
        for offset, (batch_size, block_size) in zip(offsets, series):
            ys = [metric_for(group, batch_size, block_size, metric) for group in group_order]
            ax.bar(
                [x + offset for x in xs],
                ys,
                width=width,
                label=f"bs={batch_size}, block={block_size}",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(group_order)
        ax.set_xlabel("Prompt length heterogeneity group")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    grouped_bar(
        metric="mean_block_steps",
        ylabel="Mean denoising steps for entire block",
        title="Block Completion Steps by Length Heterogeneity",
        filename="length_group_block_steps.png",
    )
    grouped_bar(
        metric="mean_step_latency_ms",
        ylabel="Mean execution time per denoising step (ms)",
        title="Per-Step Batch Forward Latency by Length Heterogeneity",
        filename="length_group_step_latency.png",
    )
    return True


@torch.inference_mode()
def probe_length_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    group_name: str,
    targets: list[int],
    block_size: int,
    num_blocks: int,
    max_steps: int,
    confidence_threshold: float,
    acceptance_policy: str,
    mask_token_id: int | None,
    trial: int,
    batch_id: int,
    run_id: str,
    device: torch.device,
) -> tuple[list[LengthRequestBlockRow], list[LengthBatchStepRow], list[LengthRequestStepRow], int]:
    mask_token_id = choose_mask_token_id(tokenizer, mask_token_id)
    input_ids, attention_mask, prompts = build_length_prompt_batch(tokenizer, group_name, targets, trial, device)
    batch_size = len(prompts)
    prompt_lengths = [record.actual_prompt_tokens for record in prompts]
    batch_max_prompt_tokens = max(prompt_lengths)
    total_target_prompt_tokens = sum(record.target_prompt_tokens for record in prompts)
    total_actual_prompt_tokens = sum(prompt_lengths)

    request_block_rows: list[LengthRequestBlockRow] = []
    batch_step_rows: list[LengthBatchStepRow] = []
    request_step_rows: list[LengthRequestStepRow] = []
    forward_calls = 0

    for block_index in range(num_blocks):
        input_ids, attention_mask, (block_start, block_end) = append_mask_block(
            input_ids,
            attention_mask,
            block_size,
            mask_token_id,
        )
        token_steps = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        token_confidences = torch.zeros((batch_size, block_size), dtype=torch.float32, device=device)
        finalized_by_threshold = torch.zeros((batch_size, block_size), dtype=torch.bool, device=device)
        finalized_by_force = torch.zeros((batch_size, block_size), dtype=torch.bool, device=device)
        step_latencies_ms: list[float] = []
        block_mask_index = torch.ones((batch_size, block_size), dtype=torch.bool, device=device)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, max_steps)

        for step in range(1, max_steps + 1):
            if not bool(token_steps.eq(0).any()):
                break

            active_before = [bool(token_steps[request_id].eq(0).any()) for request_id in range(batch_size)]
            unfinished_before = [int(token_steps[request_id].eq(0).sum().item()) for request_id in range(batch_size)]

            cuda_sync()
            start = time.perf_counter()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            cuda_sync()
            forward_calls += 1
            step_latency_ms = (time.perf_counter() - start) * 1000.0
            step_latencies_ms.append(step_latency_ms)

            accepted_this_step = [0] * batch_size
            for request_id in range(batch_size):
                masked_positions = token_steps[request_id].eq(0)
                if not bool(masked_positions.any()):
                    continue

                block_logits = logits[request_id, block_start:block_end]
                probs = torch.softmax(block_logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)
                force_accept = torch.zeros_like(masked_positions)
                threshold_accept = torch.zeros_like(masked_positions)
                masked_conf = confidence.masked_fill(~masked_positions, -float("inf"))

                if acceptance_policy == "topk":
                    k = int(num_transfer_tokens[request_id, step - 1].item())
                    k = min(k, int(masked_positions.sum().item()))
                    if k > 0:
                        _, select_index = torch.topk(masked_conf, k=k)
                        threshold_accept[select_index] = True
                    accept = threshold_accept
                elif acceptance_policy == "confidence_cutoff":
                    sorted_conf, sorted_index = torch.sort(masked_conf, descending=True)
                    keep = sorted_conf.ge(confidence_threshold)
                    if bool(keep.any()):
                        threshold_accept[sorted_index[keep]] = True
                    accept = threshold_accept.clone()
                    if not bool(accept.any()):
                        force_accept[torch.argmax(masked_conf)] = True
                        accept = force_accept
                else:
                    threshold_accept = masked_positions & confidence.ge(confidence_threshold)
                    accept = threshold_accept.clone()
                    if not bool(accept.any()):
                        force_accept[torch.argmax(masked_conf)] = True
                        accept = force_accept

                if not bool(accept.any()):
                    continue

                absolute_positions = torch.arange(block_start, block_end, device=device)[accept]
                input_ids[request_id, absolute_positions] = predicted[accept]
                token_steps[request_id, accept] = step
                token_confidences[request_id, accept] = confidence[accept].float()
                finalized_by_threshold[request_id, threshold_accept] = True
                finalized_by_force[request_id, force_accept] = True
                accepted_this_step[request_id] = int(accept.sum().item())

            active_after = [bool(token_steps[request_id].eq(0).any()) for request_id in range(batch_size)]
            unfinished_after = [int(token_steps[request_id].eq(0).sum().item()) for request_id in range(batch_size)]
            logical_lengths = [
                record.actual_prompt_tokens + (block_index * block_size) + (block_size - unfinished_before[request_id])
                for request_id, record in enumerate(prompts)
            ]
            step_number = len(step_latencies_ms)
            batch_step_rows.append(
                LengthBatchStepRow(
                    run_id=run_id,
                    group_name=group_name,
                    batch_size=batch_size,
                    block_size=block_size,
                    trial=trial,
                    batch_id=batch_id,
                    block_index=block_index,
                    step=step_number,
                    step_latency_ms=step_latency_ms,
                    active_requests_before_step=sum(active_before),
                    active_requests_after_step=sum(active_after),
                    finished_requests_after_step=sum(1 for before, after in zip(active_before, active_after) if before and not after),
                    accepted_tokens_this_step=sum(accepted_this_step),
                    unfinished_tokens_before_step=sum(unfinished_before),
                    unfinished_tokens_after_step=sum(unfinished_after),
                    padded_sequence_length=input_ids.shape[1],
                    max_logical_sequence_length_before_step=max(logical_lengths),
                )
            )

            for request_id, record in enumerate(prompts):
                request_step_rows.append(
                    LengthRequestStepRow(
                        run_id=run_id,
                        group_name=group_name,
                        batch_size=batch_size,
                        block_size=block_size,
                        trial=trial,
                        batch_id=batch_id,
                        request_id=request_id,
                        block_index=block_index,
                        step=step_number,
                        target_prompt_tokens=record.target_prompt_tokens,
                        actual_prompt_tokens=record.actual_prompt_tokens,
                        padding_tokens_for_request=batch_max_prompt_tokens - record.actual_prompt_tokens,
                        request_active_before_step=active_before[request_id],
                        request_active_after_step=active_after[request_id],
                        request_finished_after_step=active_before[request_id] and not active_after[request_id],
                        accepted_tokens_this_step=accepted_this_step[request_id],
                        unfinished_tokens_before_step=unfinished_before[request_id],
                        unfinished_tokens_after_step=unfinished_after[request_id],
                        shared_batch_step_latency_ms=step_latency_ms,
                        dense_request_step_latency_ms=step_latency_ms,
                        active_request_step_latency_ms=step_latency_ms if active_before[request_id] else 0.0,
                        padded_sequence_length=input_ids.shape[1],
                        logical_sequence_length_before_step=logical_lengths[request_id],
                    )
                )

        unfinished = token_steps.eq(0)
        hit_max_steps = unfinished.any(dim=1)
        token_steps = token_steps.masked_fill(unfinished, max_steps)
        token_confidences = token_confidences.masked_fill(unfinished, 0.0)
        request_steps = [int(max(row.tolist())) for row in token_steps]
        batch_finish_steps = max(request_steps)
        batch_block_latency_ms = sum(step_latencies_ms[:batch_finish_steps])

        for request_id, record in enumerate(prompts):
            steps = [int(x) for x in token_steps[request_id].tolist()]
            confidences = [float(x) for x in token_confidences[request_id].tolist()]
            steps_used = max(steps)
            useful_token_steps = sum(steps)
            request_latency_ms = sum(step_latencies_ms[:steps_used])
            request_block_rows.append(
                LengthRequestBlockRow(
                    run_id=run_id,
                    group_name=group_name,
                    batch_size=batch_size,
                    block_size=block_size,
                    trial=trial,
                    batch_id=batch_id,
                    request_id=request_id,
                    block_index=block_index,
                    target_prompt_tokens=record.target_prompt_tokens,
                    actual_prompt_tokens=record.actual_prompt_tokens,
                    total_batch_target_prompt_tokens=total_target_prompt_tokens,
                    total_batch_actual_prompt_tokens=total_actual_prompt_tokens,
                    batch_max_prompt_tokens=batch_max_prompt_tokens,
                    padding_tokens_for_request=batch_max_prompt_tokens - record.actual_prompt_tokens,
                    num_blocks=num_blocks,
                    max_steps_per_block=max_steps,
                    confidence_threshold=confidence_threshold,
                    acceptance_policy=acceptance_policy,
                    steps_used=steps_used,
                    steps_executed=batch_finish_steps,
                    request_latency_ms=request_latency_ms,
                    batch_block_latency_ms=batch_block_latency_ms,
                    useful_token_steps=useful_token_steps,
                    executed_token_steps=batch_finish_steps * block_size,
                    token_step_min=min(steps),
                    token_step_mean=mean(steps),
                    token_step_max=max(steps),
                    mean_confidence=mean(confidences),
                    min_confidence=min(confidences),
                    finalized_by_threshold=int(finalized_by_threshold[request_id].sum().item()),
                    finalized_by_force=int(finalized_by_force[request_id].sum().item()),
                    hit_max_steps=bool(hit_max_steps[request_id].item()),
                    block_internal_waste_ratio=(steps_used * block_size) / useful_token_steps,
                    num_tokens=block_size,
                    tokens_generated=block_size,
                )
            )

    return request_block_rows, batch_step_rows, request_step_rows, forward_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/length_heterogeneity/manual"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "manual"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--target-tokens-per-request", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["confidence_cutoff", "topk", "threshold"], default="confidence_cutoff")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["none", "auto"], default="none")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG summary plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    if args.device_map == "none":
        model = model.to(device)
    model.eval()

    request_block_rows: list[LengthRequestBlockRow] = []
    batch_step_rows: list[LengthBatchStepRow] = []
    request_step_rows: list[LengthRequestStepRow] = []
    total_forward_calls = 0
    batch_id = 0

    print(
        "[llada length probe] config "
        f"run_id={args.run_id} batch_sizes={args.batch_sizes} block_sizes={args.block_sizes} "
        f"target_tokens_per_request={args.target_tokens_per_request} num_blocks={args.num_blocks} "
        f"max_steps_per_block={args.max_steps} confidence_threshold={args.confidence_threshold} "
        f"acceptance_policy={args.acceptance_policy} mask_token_id={args.mask_token_id}"
    )

    for batch_size in args.batch_sizes:
        total_prompt_tokens = args.target_tokens_per_request * batch_size
        groups = length_targets(batch_size, total_prompt_tokens)
        for block_size in args.block_sizes:
            for group_name, targets in groups.items():
                for trial in range(args.trials):
                    block_rows, step_rows, req_step_rows, forward_calls = probe_length_batch(
                        model=model,
                        tokenizer=tokenizer,
                        group_name=group_name,
                        targets=targets,
                        block_size=block_size,
                        num_blocks=args.num_blocks,
                        max_steps=args.max_steps,
                        confidence_threshold=args.confidence_threshold,
                        acceptance_policy=args.acceptance_policy,
                        mask_token_id=args.mask_token_id,
                        trial=trial,
                        batch_id=batch_id,
                        run_id=args.run_id,
                        device=device,
                    )
                    request_block_rows.extend(block_rows)
                    batch_step_rows.extend(step_rows)
                    request_step_rows.extend(req_step_rows)
                    total_forward_calls += forward_calls
                    print(
                        f"done batch_size={batch_size} block_size={block_size} group={group_name} "
                        f"targets={targets} batch_id={batch_id} trial={trial} forward_calls={forward_calls}"
                    )
                    batch_id += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    request_block_out = args.out_dir / "length_request_block_rows.csv"
    batch_step_out = args.out_dir / "length_batch_step_rows.csv"
    request_step_out = args.out_dir / "length_request_step_rows.csv"
    plot_summary_out = args.out_dir / "length_group_summary.csv"
    write_dataclass_rows(request_block_rows, request_block_out)
    write_dataclass_rows(batch_step_rows, batch_step_out)
    write_dataclass_rows(request_step_rows, request_step_out)
    summary_rows = write_length_plot_summary(request_block_rows, batch_step_rows, plot_summary_out)
    plotted = False
    if not args.no_plots:
        plotted = plot_length_group_figures(summary_rows, args.out_dir)

    steps_used = [row.steps_used for row in request_block_rows]
    print(f"wrote {request_block_out}")
    print(f"wrote {batch_step_out}")
    print(f"wrote {request_step_out}")
    print(f"wrote {plot_summary_out}")
    if plotted:
        print(f"wrote {args.out_dir / 'length_group_block_steps.png'}")
        print(f"wrote {args.out_dir / 'length_group_step_latency.png'}")
    print(f"[llada length probe] total real model forward calls={total_forward_calls}")
    if steps_used:
        sorted_steps = sorted(steps_used)
        p50 = sorted_steps[len(sorted_steps) // 2]
        p90 = sorted_steps[min(len(sorted_steps) - 1, int(len(sorted_steps) * 0.9))]
        print(
            "[llada length probe] "
            f"steps_used mean={mean(steps_used):.3f} p50={p50} p90={p90} max={max(steps_used)}"
        )
    print(
        "[llada length probe] note: request-step latency is shared batch forward latency "
        "attributed to active requests, not isolated per-request kernel time."
    )


if __name__ == "__main__":
    main()
