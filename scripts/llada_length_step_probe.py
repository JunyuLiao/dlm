#!/usr/bin/env python3
"""Measure LLaDA denoising steps as request length changes.

Experiment 1 runs every request alone (batch size 1) and records every
denoising iteration plus per-block and per-request totals. Experiment 2 reuses
the same requests in three batches of eight, ordered by length, so the batches
have different maximum prompt lengths. In experiment 2, ``denoising_steps`` is
the number of synchronized iterations executed by the whole batch/block.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_block_step_probe import (
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    get_num_transfer_tokens,
    patch_llada_transformers_compat,
)


DEFAULT_LENGTHS = [
    50, 75, 100, 125, 150, 200, 250, 300,
    400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1600, 1750, 1900, 2000,
]

PROMPT_SEEDS = [
    "Explain how masked diffusion language models generate text, including confidence-based token acceptance, batching, and stopping conditions. ",
    "Analyze a distributed inference service. Discuss scheduling, memory use, synchronization, tail latency, and practical ways to measure each bottleneck. ",
    "Write a careful technical review of an algorithm, with assumptions, examples, edge cases, complexity analysis, and a concise conclusion. ",
    "Compare several approaches to solving a scientific problem. Show the reasoning, identify uncertainty, and propose experiments that distinguish the alternatives. ",
]


@dataclass(frozen=True)
class LengthRequest:
    request_id: int
    target_tokens: int
    input_ids: list[int]
    preview: str


@dataclass(frozen=True)
class StepRow:
    experiment: str
    batch_id: int
    batch_size: int
    block_index: int
    step: int
    min_prompt_tokens: int
    max_prompt_tokens: int
    active_requests_before_step: int
    step_latency_ms: float


@dataclass(frozen=True)
class BlockRow:
    experiment: str
    batch_id: int
    batch_size: int
    block_index: int
    block_size: int
    request_ids: str
    request_lengths: str
    min_prompt_tokens: int
    max_prompt_tokens: int
    sequence_tokens: int
    max_steps: int
    confidence_threshold: float
    acceptance_policy: str
    denoising_steps: int
    total_time_ms: float
    mean_time_per_step_ms: float
    min_time_per_step_ms: float
    max_time_per_step_ms: float
    hit_max_steps: bool


@dataclass(frozen=True)
class MembershipRow:
    experiment: str
    batch_id: int
    request_id: int
    prompt_tokens: int
    batch_max_prompt_tokens: int


def parse_lengths(values: list[str] | None) -> list[int]:
    if values is None:
        return DEFAULT_LENGTHS.copy()
    lengths: list[int] = []
    for value in values:
        lengths.extend(int(part) for part in value.replace(",", " ").split())
    return lengths


def make_requests(tokenizer: AutoTokenizer, lengths: list[int]) -> list[LengthRequest]:
    requests: list[LengthRequest] = []
    for request_id, target_tokens in enumerate(sorted(lengths)):
        seed_ids = tokenizer(PROMPT_SEEDS[request_id % len(PROMPT_SEEDS)], add_special_tokens=False)["input_ids"]
        if target_tokens <= 0 or not seed_ids:
            raise ValueError(f"cannot construct a request with {target_tokens} tokens")
        # TokenizersBackend in recent transformers versions does not expose
        # build_inputs_with_special_tokens(). These are already valid token
        # IDs, and LLaDA accepts raw prompt IDs, so cycling and slicing them is
        # both backend-independent and guarantees the requested exact length.
        input_ids = list(islice(cycle(seed_ids), target_tokens))
        preview = tokenizer.decode(input_ids[: min(48, len(input_ids))], skip_special_tokens=True).replace("\n", " ")
        requests.append(LengthRequest(request_id, target_tokens, input_ids, preview))
    return requests


def build_batch(
    requests: list[LengthRequest],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(request.input_ids) for request in requests)
    # Left-pad so the prompt end and every appended generation block have the
    # same relative positions as in the corresponding batch-size-1 run. With
    # right padding, shorter prompts have a masked positional gap before the
    # block; the attention mask hides pad keys but does not renumber RoPE.
    rows = [[pad_token_id] * (max_length - len(request.input_ids)) + request.input_ids for request in requests]
    masks = [[0] * (max_length - len(request.input_ids)) + [1] * len(request.input_ids) for request in requests]
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def run_batch(
    model: AutoModelForCausalLM,
    requests: list[LengthRequest],
    pad_token_id: int,
    mask_token_id: int,
    block_size: int,
    num_blocks: int,
    max_steps: int,
    confidence_threshold: float,
    acceptance_policy: str,
    experiment: str,
    batch_id: int,
    device: torch.device,
) -> tuple[list[StepRow], list[BlockRow]]:
    input_ids, attention_mask = build_batch(requests, pad_token_id, device)
    batch_size = len(requests)
    prompt_lengths = [len(request.input_ids) for request in requests]
    step_rows: list[StepRow] = []
    block_rows: list[BlockRow] = []

    for block_index in range(num_blocks):
        block_start = input_ids.shape[1]
        block_end = block_start + block_size
        mask_block = torch.full((batch_size, block_size), mask_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, mask_block], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(mask_block)], dim=1)
        token_steps = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        transfer = get_num_transfer_tokens(torch.ones_like(token_steps, dtype=torch.bool), max_steps)
        block_step_times: list[float] = []

        for step in range(1, max_steps + 1):
            unfinished_requests = token_steps.eq(0).any(dim=1)
            if not bool(unfinished_requests.any()):
                break

            cuda_sync()
            started = time.perf_counter()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            for request_index in range(batch_size):
                masked = token_steps[request_index].eq(0)
                if not bool(masked.any()):
                    continue
                block_logits = logits[request_index, block_start:block_end]
                confidence, predicted = torch.max(torch.softmax(block_logits, dim=-1), dim=-1)
                masked_confidence = confidence.masked_fill(~masked, -float("inf"))
                accepted = torch.zeros_like(masked)

                if acceptance_policy == "topk":
                    k = min(int(transfer[request_index, step - 1].item()), int(masked.sum().item()))
                    if k:
                        accepted[torch.topk(masked_confidence, k=k).indices] = True
                else:
                    accepted = masked & confidence.ge(confidence_threshold)
                    if not bool(accepted.any()):
                        accepted[torch.argmax(masked_confidence)] = True

                absolute_positions = torch.arange(block_start, block_end, device=device)[accepted]
                input_ids[request_index, absolute_positions] = predicted[accepted]
                token_steps[request_index, accepted] = step

            cuda_sync()
            latency_ms = (time.perf_counter() - started) * 1000.0
            # A full-vocabulary logits tensor is large for the 2k-token bs=8
            # cases. Release it before evaluating the next model forward so
            # two iterations' logits cannot overlap in memory.
            del logits, block_logits, confidence, predicted, masked_confidence, accepted
            block_step_times.append(latency_ms)
            step_rows.append(
                StepRow(
                    experiment=experiment,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    block_index=block_index,
                    step=step,
                    min_prompt_tokens=min(prompt_lengths),
                    max_prompt_tokens=max(prompt_lengths),
                    active_requests_before_step=int(unfinished_requests.sum().item()),
                    step_latency_ms=latency_ms,
                )
            )

        hit_max_steps = bool(token_steps.eq(0).any())
        block_rows.append(
            BlockRow(
                experiment=experiment,
                batch_id=batch_id,
                batch_size=batch_size,
                block_index=block_index,
                block_size=block_size,
                request_ids="|".join(str(request.request_id) for request in requests),
                request_lengths="|".join(str(len(request.input_ids)) for request in requests),
                min_prompt_tokens=min(prompt_lengths),
                max_prompt_tokens=max(prompt_lengths),
                sequence_tokens=max(prompt_lengths) + (block_index + 1) * block_size,
                max_steps=max_steps,
                confidence_threshold=confidence_threshold,
                acceptance_policy=acceptance_policy,
                denoising_steps=len(block_step_times),
                total_time_ms=sum(block_step_times),
                mean_time_per_step_ms=mean(block_step_times),
                min_time_per_step_ms=min(block_step_times),
                max_time_per_step_ms=max(block_step_times),
                hit_max_steps=hit_max_steps,
            )
        )
        if hit_max_steps:
            raise RuntimeError(
                f"batch {batch_id}, block {block_index} did not finish in {max_steps} steps; "
                "increase --max-steps before using the measurements"
            )
    return step_rows, block_rows


def write_dataclasses(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def write_request_catalog(path: Path, requests: list[LengthRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_id", "target_tokens", "prompt_tokens", "preview"])
        writer.writeheader()
        for request in requests:
            writer.writerow(
                {
                    "request_id": request.request_id,
                    "target_tokens": request.target_tokens,
                    "prompt_tokens": len(request.input_ids),
                    "preview": request.preview,
                }
            )


def write_exp1_request_summary(path: Path, requests: list[LengthRequest], blocks: list[BlockRow]) -> None:
    by_batch = {request.request_id: [row for row in blocks if row.batch_id == request.request_id] for request in requests}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id", "prompt_tokens", "num_blocks",
    ]
    num_blocks = max(row.block_index for row in blocks) + 1
    for block_index in range(num_blocks):
        fieldnames.extend([f"block_{block_index}_steps", f"block_{block_index}_time_ms"])
    fieldnames.extend(["total_denoising_steps", "total_time_ms", "mean_time_per_step_ms"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for request in requests:
            rows = sorted(by_batch[request.request_id], key=lambda row: row.block_index)
            total_steps = sum(row.denoising_steps for row in rows)
            total_time = sum(row.total_time_ms for row in rows)
            output: dict[str, object] = {
                "request_id": request.request_id,
                "prompt_tokens": len(request.input_ids),
                "num_blocks": len(rows),
                "total_denoising_steps": total_steps,
                "total_time_ms": total_time,
                "mean_time_per_step_ms": total_time / total_steps,
            }
            for row in rows:
                output[f"block_{row.block_index}_steps"] = row.denoising_steps
                output[f"block_{row.block_index}_time_ms"] = row.total_time_ms
            writer.writerow(output)


def write_exp2_batch_summary(
    path: Path,
    batches: list[list[LengthRequest]],
    blocks: list[BlockRow],
) -> None:
    by_batch = {batch_id: sorted(
        [row for row in blocks if row.batch_id == batch_id],
        key=lambda row: row.block_index,
    ) for batch_id in range(len(batches))}
    num_blocks = max(row.block_index for row in blocks) + 1
    fieldnames = ["batch_id", "batch_size", "request_ids", "request_lengths", "max_prompt_tokens"]
    for block_index in range(num_blocks):
        fieldnames.extend([f"block_{block_index}_steps", f"block_{block_index}_time_ms"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch_id, batch in enumerate(batches):
            output: dict[str, object] = {
                "batch_id": batch_id,
                "batch_size": len(batch),
                "request_ids": "|".join(str(request.request_id) for request in batch),
                "request_lengths": "|".join(str(len(request.input_ids)) for request in batch),
                "max_prompt_tokens": max(len(request.input_ids) for request in batch),
            }
            for row in by_batch[batch_id]:
                output[f"block_{row.block_index}_steps"] = row.denoising_steps
                output[f"block_{row.block_index}_time_ms"] = row.total_time_ms
            writer.writerow(output)


def validate_args(args: argparse.Namespace, lengths: list[int]) -> None:
    if len(lengths) != 24:
        raise ValueError("exactly 24 --lengths are required: three disjoint batches of 8 reuse experiment-1 requests")
    if len(set(lengths)) != len(lengths):
        raise ValueError("--lengths must be unique")
    if min(lengths) < 1 or args.block_size < 1 or args.num_blocks < 1 or args.max_steps < 1:
        raise ValueError("lengths, block size, number of blocks, and max steps must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/llada_length_steps"))
    parser.add_argument("--lengths", nargs="+", help="24 prompt-token lengths, comma- or space-separated")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["confidence_cutoff", "threshold", "topk"], default="confidence_cutoff")
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["none", "auto"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=2)
    return parser.parse_args()


@torch.inference_mode()
def warm_up(
    model: AutoModelForCausalLM,
    request: LengthRequest,
    pad_token_id: int,
    mask_token_id: int,
    block_size: int,
    steps: int,
    device: torch.device,
) -> None:
    if steps <= 0:
        return
    input_ids, attention_mask = build_batch([request], pad_token_id, device)
    masks = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([input_ids, masks], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(masks)], dim=1)
    for _ in range(steps):
        model(input_ids=input_ids, attention_mask=attention_mask)
    cuda_sync()


def main() -> None:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    validate_args(args, lengths)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs: dict[str, object] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    if args.device_map == "none":
        model = model.to(device)
    model.eval()

    requests = make_requests(tokenizer, lengths)
    required_context = max(lengths) + args.num_blocks * args.block_size
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None and required_context > int(model_context):
        raise ValueError(f"experiment needs context {required_context}, but model supports {model_context}")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    mask_token_id = choose_mask_token_id(tokenizer, args.mask_token_id)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_request_catalog(args.outdir / "requests.csv", requests)
    print(f"[length probe] warming up with {args.warmup_steps} forwards")
    warm_up(model, requests[0], int(pad_token_id), mask_token_id, args.block_size, args.warmup_steps, device)

    exp1_steps: list[StepRow] = []
    exp1_blocks: list[BlockRow] = []
    for request in requests:
        steps, blocks = run_batch(
            model, [request], int(pad_token_id), mask_token_id, args.block_size,
            args.num_blocks, args.max_steps, args.confidence_threshold,
            args.acceptance_policy, "exp1_bs1", request.request_id, device,
        )
        exp1_steps.extend(steps)
        exp1_blocks.extend(blocks)
        print(
            f"[length probe] exp1 request={request.request_id} tokens={len(request.input_ids)} "
            f"block_steps={[row.denoising_steps for row in blocks]} "
            f"total_time_ms={sum(row.total_time_ms for row in blocks):.3f}"
        )

    exp2_steps: list[StepRow] = []
    exp2_blocks: list[BlockRow] = []
    memberships: list[MembershipRow] = []
    exp2_batches: list[list[LengthRequest]] = []
    sorted_requests = sorted(requests, key=lambda request: len(request.input_ids))
    original_by_id = {request.request_id: request for request in requests}
    for batch_id in range(3):
        batch = sorted_requests[batch_id * 8 : (batch_id + 1) * 8]
        if any(original_by_id[request.request_id] is not request for request in batch):
            raise RuntimeError("experiment 2 must reuse the exact request objects from experiment 1")
        exp2_batches.append(batch)
        batch_max = max(len(request.input_ids) for request in batch)
        for request in batch:
            memberships.append(MembershipRow("exp2_bs8", batch_id, request.request_id, len(request.input_ids), batch_max))
        steps, blocks = run_batch(
            model, batch, int(pad_token_id), mask_token_id, args.block_size,
            args.num_blocks, args.max_steps, args.confidence_threshold,
            args.acceptance_policy, "exp2_bs8", batch_id, device,
        )
        exp2_steps.extend(steps)
        exp2_blocks.extend(blocks)
        print(
            f"[length probe] exp2 batch={batch_id} "
            f"request_lengths={[len(request.input_ids) for request in batch]} max_tokens={batch_max} "
            f"block_steps={[row.denoising_steps for row in blocks]}"
        )

    reused_ids = [request.request_id for batch in exp2_batches for request in batch]
    if sorted(reused_ids) != sorted(request.request_id for request in requests):
        raise RuntimeError("experiment 2 must reuse every experiment-1 request exactly once")

    write_dataclasses(args.outdir / "exp1_step_latency.csv", exp1_steps)
    write_dataclasses(args.outdir / "exp1_block_summary.csv", exp1_blocks)
    write_exp1_request_summary(args.outdir / "exp1_request_summary.csv", requests, exp1_blocks)
    write_dataclasses(args.outdir / "exp2_batch_membership.csv", memberships)
    write_dataclasses(args.outdir / "exp2_step_latency.csv", exp2_steps)
    write_dataclasses(args.outdir / "exp2_block_summary.csv", exp2_blocks)
    write_exp2_batch_summary(args.outdir / "exp2_batch_summary.csv", exp2_batches, exp2_blocks)
    print(f"[length probe] wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
