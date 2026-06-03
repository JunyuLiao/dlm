#!/usr/bin/env python3
"""Run real batched LLaDA block probes on H100 and record dynamic steps.

This is the real-data path, not the simulator. It batches prompts, appends masked
blocks sequentially, runs real model forward passes, dynamically finalizes tokens
by confidence, and writes per-request/block plus per-batch/block CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_tied_weight_keys(value: object) -> dict[str, None]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple, set)):
        return {str(item): None for item in value}
    return {str(value): None}


@contextmanager
def patch_llada_transformers_compat() -> object:
    """Shim LLaDA remote-code models for newer transformers during loading."""

    original_getattr = torch.nn.Module.__getattr__
    original_getattribute = torch.nn.Module.__getattribute__

    def patched_getattr(self: torch.nn.Module, name: str) -> object:
        if name == "all_tied_weights_keys":
            tied = self.__dict__.get("_tied_weights_keys", None)
            if tied is None:
                tied = getattr(type(self), "_tied_weights_keys", None)
            return normalize_tied_weight_keys(tied)
        return original_getattr(self, name)

    def patched_getattribute(self: torch.nn.Module, name: str) -> object:
        attr = original_getattribute(self, name)
        if name != "tie_weights" or not callable(attr):
            return attr

        def tie_weights_compat(*args: object, **kwargs: object) -> object:
            try:
                return attr(*args, **kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                return attr()

        return tie_weights_compat

    torch.nn.Module.__getattr__ = patched_getattr
    torch.nn.Module.__getattribute__ = patched_getattribute  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.Module.__getattr__ = original_getattr
        torch.nn.Module.__getattribute__ = original_getattribute  # type: ignore[method-assign]


def disable_use_cache(model: torch.nn.Module) -> None:
    """Disable KV cache for masked-diffusion block probing."""

    model.config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False
    print("[llada probe] set config.use_cache=False")


def ensure_all_tied_weights_keys(model: torch.nn.Module) -> None:
    if not hasattr(model, "all_tied_weights_keys"):
        tied = getattr(model, "_tied_weights_keys", None)
        model.all_tied_weights_keys = normalize_tied_weight_keys(tied)


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    difficulty: str
    prompt: str


@dataclass(frozen=True)
class ProbeRow:
    run_id: str
    batch_size: int
    block_size: int
    scheduler: str
    trial: int
    batch_id: int
    request_id: int
    block_index: int
    prompt_id: int
    prompt_difficulty: str
    prompt_len: int
    num_blocks: int
    max_steps_per_block: int
    confidence_threshold: float
    steps_used: int
    steps_needed: int
    steps_executed: int
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
    latency_ms: float
    num_tokens: int
    tokens_generated: int


@dataclass(frozen=True)
class BatchBlockRow:
    run_id: str
    batch_size: int
    batch_id: int
    block_index: int
    active_requests: int
    batch_block_steps: int
    max_request_steps_used: int
    min_request_steps_used: int
    mean_request_steps_used: float
    latency_ms: float
    waste_token_steps: int


def infer_difficulty(prompt: str) -> str:
    lower = prompt.lower()
    if any(word in lower for word in ["证明", "prove", "复杂", "step by step", "逻辑"]):
        return "extreme"
    if any(word in lower for word in ["sql", "algorithm", "算法", "代码", "python", "cuda"]):
        return "hard"
    if any(word in lower for word in ["翻译", "translate", "email", "两句话"]):
        return "easy"
    return "medium"


def read_prompts(path: Path) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                item = json.loads(line)
                prompt = str(item["prompt"])
                difficulty = str(item.get("difficulty") or infer_difficulty(prompt))
            else:
                prompt = line
                difficulty = infer_difficulty(prompt)
            prompts.append(PromptRecord(index, difficulty, prompt))
    return prompts


def choose_mixed_batch(prompts: list[PromptRecord], batch_size: int, batch_id: int) -> list[PromptRecord]:
    by_level = {level: [p for p in prompts if p.difficulty == level] for level in ["easy", "medium", "hard", "extreme"]}
    batch: list[PromptRecord] = []
    cursor = batch_id
    while len(batch) < batch_size:
        for level in ["easy", "medium", "hard", "extreme"]:
            candidates = by_level[level]
            if not candidates:
                continue
            batch.append(candidates[cursor % len(candidates)])
            if len(batch) == batch_size:
                break
        cursor += 1
    return batch


def choose_mask_token_id(tokenizer: AutoTokenizer, explicit_mask_token_id: int | None) -> int:
    env_mask_token_id = os.environ.get("MASK_TOKEN_ID")
    if env_mask_token_id:
        return int(env_mask_token_id)
    if explicit_mask_token_id is not None:
        return explicit_mask_token_id
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return int(token_id)
    return 126336


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_prompt_batch(
    tokenizer: AutoTokenizer,
    prompts: list[PromptRecord],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = [tokenizer(item.prompt, add_special_tokens=True)["input_ids"] for item in prompts]
    prompt_lengths = [len(ids) for ids in encoded]
    max_prompt = max(prompt_lengths)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = [ids + [pad_id] * (max_prompt - len(ids)) for ids in encoded]
    masks = [[1] * len(ids) + [0] * (max_prompt - len(ids)) for ids in encoded]
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        prompt_lengths,
    )


def append_mask_block(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    batch_size = input_ids.shape[0]
    block_start = input_ids.shape[1]
    block_end = block_start + block_size
    mask_block = torch.full((batch_size, block_size), mask_token_id, dtype=torch.long, device=input_ids.device)
    mask_attention = torch.ones_like(mask_block, device=input_ids.device)
    return torch.cat([input_ids, mask_block], dim=1), torch.cat([attention_mask, mask_attention], dim=1), (block_start, block_end)


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Match official LLaDA linear transfer schedule for top-k unmasking."""

    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        num_transfer_tokens[row, : int(remainder[row].item())] += 1
    return num_transfer_tokens


def make_probe_rows(
    prompts: list[PromptRecord],
    prompt_lengths: list[int],
    token_steps: torch.Tensor,
    token_confidences: torch.Tensor,
    finalized_by_threshold: torch.Tensor,
    finalized_by_force: torch.Tensor,
    hit_max_steps: torch.Tensor,
    step_latencies_ms: list[float],
    run_id: str,
    batch_size: int,
    block_size: int,
    trial: int,
    batch_id: int,
    block_index: int,
    num_blocks: int,
    max_steps_per_block: int,
    confidence_threshold: float,
) -> tuple[list[ProbeRow], BatchBlockRow]:
    request_steps = [int(max(row.tolist())) for row in token_steps]
    batch_finish_steps = max(request_steps)
    full_batch_latency = sum(step_latencies_ms[:batch_finish_steps])
    rows: list[ProbeRow] = []

    for request_id, prompt in enumerate(prompts):
        steps = [int(x) for x in token_steps[request_id].tolist()]
        confidences = [float(x) for x in token_confidences[request_id].tolist()]
        steps_used = max(steps)
        useful_token_steps = sum(steps)
        common = dict(
            run_id=run_id,
            batch_size=batch_size,
            block_size=block_size,
            trial=trial,
            batch_id=batch_id,
            request_id=request_id,
            block_index=block_index,
            prompt_id=prompt.prompt_id,
            prompt_difficulty=prompt.difficulty,
            prompt_len=prompt_lengths[request_id],
            num_blocks=num_blocks,
            max_steps_per_block=max_steps_per_block,
            confidence_threshold=confidence_threshold,
            steps_used=steps_used,
            steps_needed=steps_used,
            useful_token_steps=useful_token_steps,
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
        rows.append(
            ProbeRow(
                scheduler="real_llada",
                steps_executed=batch_finish_steps,
                executed_token_steps=batch_finish_steps * block_size,
                latency_ms=full_batch_latency,
                **common,
            )
        )

    useful_batch_token_steps = int(token_steps.sum().item())
    batch_row = BatchBlockRow(
        run_id=run_id,
        batch_size=batch_size,
        batch_id=batch_id,
        block_index=block_index,
        active_requests=len(prompts),
        batch_block_steps=batch_finish_steps,
        max_request_steps_used=max(request_steps),
        min_request_steps_used=min(request_steps),
        mean_request_steps_used=mean(request_steps),
        latency_ms=full_batch_latency,
        waste_token_steps=batch_finish_steps * block_size * len(prompts) - useful_batch_token_steps,
    )
    return rows, batch_row


@torch.inference_mode()
def probe_one_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[PromptRecord],
    batch_size: int,
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
) -> tuple[list[ProbeRow], list[BatchBlockRow], int]:
    mask_token_id = choose_mask_token_id(tokenizer, mask_token_id)
    input_ids, attention_mask, prompt_lengths = build_prompt_batch(tokenizer, prompts, device)
    rows: list[ProbeRow] = []
    batch_rows: list[BatchBlockRow] = []
    forward_calls = 0

    for block_index in range(num_blocks):
        input_ids, attention_mask, (block_start, block_end) = append_mask_block(input_ids, attention_mask, block_size, mask_token_id)
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

            cuda_sync()
            start = time.perf_counter()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            cuda_sync()
            forward_calls += 1
            step_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            for request_id in range(batch_size):
                masked_positions = token_steps[request_id].eq(0)
                if not bool(masked_positions.any()):
                    continue
                block_logits = logits[request_id, block_start:block_end]
                probs = torch.softmax(block_logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)
                force_accept = torch.zeros_like(masked_positions)
                if acceptance_policy == "topk":
                    k = int(num_transfer_tokens[request_id, step - 1].item())
                    k = min(k, int(masked_positions.sum().item()))
                    threshold_accept = torch.zeros_like(masked_positions)
                    if k > 0:
                        masked_conf = confidence.masked_fill(~masked_positions, -float("inf"))
                        _, select_index = torch.topk(masked_conf, k=k)
                        threshold_accept[select_index] = True
                    accept = threshold_accept
                else:
                    threshold_accept = masked_positions & confidence.ge(confidence_threshold)
                    accept = threshold_accept.clone()
                    if not bool(accept.any()):
                        masked_conf = confidence.masked_fill(~masked_positions, -1.0)
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

        unfinished = token_steps.eq(0)
        hit_max_steps = unfinished.any(dim=1)
        token_steps = token_steps.masked_fill(unfinished, max_steps)
        token_confidences = token_confidences.masked_fill(unfinished, 0.0)
        block_rows, batch_row = make_probe_rows(
            prompts=prompts,
            prompt_lengths=prompt_lengths,
            token_steps=token_steps,
            token_confidences=token_confidences,
            finalized_by_threshold=finalized_by_threshold,
            finalized_by_force=finalized_by_force,
            hit_max_steps=hit_max_steps,
            step_latencies_ms=step_latencies_ms,
            run_id=run_id,
            batch_size=batch_size,
            block_size=block_size,
            trial=trial,
            batch_id=batch_id,
            block_index=block_index,
            num_blocks=num_blocks,
            max_steps_per_block=max_steps,
            confidence_threshold=confidence_threshold,
        )
        rows.extend(block_rows)
        batch_rows.append(batch_row)
    return rows, batch_rows, forward_calls


def write_block_step_rows(rows: list[ProbeRow], path: Path) -> None:
    fieldnames = [
        "run_id",
        "batch_size",
        "batch_id",
        "request_id",
        "prompt_id",
        "difficulty",
        "prompt_len",
        "block_index",
        "num_blocks",
        "block_size",
        "max_steps_per_block",
        "confidence_threshold",
        "steps_used",
        "latency_ms",
        "num_tokens",
        "mean_confidence",
        "min_confidence",
        "finalized_by_threshold",
        "finalized_by_force",
        "hit_max_steps",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "batch_size": row.batch_size,
                    "batch_id": row.batch_id,
                    "request_id": row.request_id,
                    "prompt_id": row.prompt_id,
                    "difficulty": row.prompt_difficulty,
                    "prompt_len": row.prompt_len,
                    "block_index": row.block_index,
                    "num_blocks": row.num_blocks,
                    "block_size": row.block_size,
                    "max_steps_per_block": row.max_steps_per_block,
                    "confidence_threshold": row.confidence_threshold,
                    "steps_used": row.steps_used,
                    "latency_ms": row.latency_ms,
                    "num_tokens": row.num_tokens,
                    "mean_confidence": row.mean_confidence,
                    "min_confidence": row.min_confidence,
                    "finalized_by_threshold": row.finalized_by_threshold,
                    "finalized_by_force": row.finalized_by_force,
                    "hit_max_steps": row.hit_max_steps,
                }
            )


def write_batch_block_rows(rows: list[BatchBlockRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_rows(rows: list[ProbeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_heterogeneous.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/h100_llada/per_request_rows.csv"))
    parser.add_argument("--block-steps-out", type=Path, help="Compact CSV with one row per request/prompt block")
    parser.add_argument("--batch-block-out", type=Path, help="CSV with one row per real batch/block latency")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "manual"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["topk", "threshold"], default="topk")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["none", "auto"], default="none")
    return parser.parse_args()


def print_run_checks(
    args: argparse.Namespace,
    block_rows: list[ProbeRow],
    batch_rows: list[BatchBlockRow],
    total_forward_calls: int,
) -> None:
    real_rows = block_rows
    steps_used = [row.steps_used for row in real_rows]
    expected_rows = sum(row.active_requests for row in batch_rows)
    unique_block_index = sorted({row.block_index for row in real_rows})
    num_requests = expected_rows // args.num_blocks if args.num_blocks else 0
    num_batches = len(batch_rows) // args.num_blocks if args.num_blocks else 0
    print(f"[llada probe] num_batches={num_batches}")
    print(f"[llada probe] batch_size={args.batch_sizes}")
    print(f"[llada probe] num_requests={num_requests}")
    print(f"[llada probe] num_blocks={args.num_blocks}")
    print(f"[llada probe] actual rows in block_steps.csv={len(real_rows)}")
    print(f"[llada probe] expected rows = num_requests * num_blocks = {expected_rows}")
    print(f"[llada probe] unique block_index={unique_block_index}")
    print(f"[llada probe] num unique steps_used={len(set(steps_used))}")
    if steps_used:
        sorted_steps = sorted(steps_used)
        p50 = sorted_steps[len(sorted_steps) // 2]
        p90 = sorted_steps[min(len(sorted_steps) - 1, int(len(sorted_steps) * 0.9))]
        print(f"[llada probe] steps_used mean={mean(steps_used):.3f} p50={p50} p90={p90} max={max(steps_used)}")
    print(f"[llada probe] total real model forward calls={total_forward_calls}")
    if any(row.active_requests != row.batch_size for row in batch_rows):
        print("[llada probe][WARN] BATCH_SIZE may be ignored: active_requests != batch_size in batch_block_latency.csv")
    if len(set(steps_used)) <= 1:
        print("[llada probe][WARN] all steps_used are identical")
    if unique_block_index == [0]:
        print("[llada probe][WARN] block_steps.csv has only block_index=0")
    if len(real_rows) != expected_rows:
        print("[llada probe][WARN] rows != num_requests * NUM_BLOCKS")


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

    prompts = read_prompts(args.prompts)
    rows: list[ProbeRow] = []
    batch_block_rows: list[BatchBlockRow] = []
    total_forward_calls = 0
    batch_id = 0
    print(
        "[llada probe] config "
        f"run_id={args.run_id} batch_sizes={args.batch_sizes} block_sizes={args.block_sizes} "
        f"num_blocks={args.num_blocks} max_steps_per_block={args.max_steps} "
        f"confidence_threshold={args.confidence_threshold} acceptance_policy={args.acceptance_policy} mask_token_id={args.mask_token_id}"
    )
    for block_size in args.block_sizes:
        for batch_size in args.batch_sizes:
            if len(prompts) < batch_size:
                raise ValueError(f"prompts 数量至少要有 batch_size={batch_size} 条")
            for trial in range(args.trials):
                batch_prompts = choose_mixed_batch(prompts, batch_size, batch_id)
                probe_rows, batch_rows, forward_calls = probe_one_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    batch_size=batch_size,
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
                rows.extend(probe_rows)
                batch_block_rows.extend(batch_rows)
                total_forward_calls += forward_calls
                print(
                    f"done block_size={block_size} batch_size={batch_size} "
                    f"batch_id={batch_id} trial={trial} num_blocks={args.num_blocks} forward_calls={forward_calls}"
                )
                batch_id += 1

    write_rows(rows, args.out)
    block_steps_out = args.block_steps_out or (args.out.parent / "block_steps.csv")
    batch_block_out = args.batch_block_out or (args.out.parent / "batch_block_latency.csv")
    write_block_step_rows(rows, block_steps_out)
    write_batch_block_rows(batch_block_rows, batch_block_out)
    print(f"wrote {args.out}")
    print(f"wrote {block_steps_out}")
    print(f"wrote {batch_block_out}")
    print_run_checks(args, rows, batch_block_rows, total_forward_calls)


if __name__ == "__main__":
    main()
