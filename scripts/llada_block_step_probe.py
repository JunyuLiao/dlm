#!/usr/bin/env python3
"""用真实 masked-diffusion LM 在 H100 上探测 block 内 token step 分布。

默认读取带 difficulty 的 prompt JSONL：easy/medium/hard/extreme。脚本会在 prompt
后按 block 逐段追加 masked tokens；第 k 个 block 会以上一个 block 已经生成的
内容作为前文。每个 block 内逐步接受 confidence 足够高的 token，记录每个 token
在哪一步完成，输出与 `dlm_block_sampling_benchmark.py --mode plot-csv` 兼容的 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    difficulty: str
    prompt: str


@dataclass(frozen=True)
class ProbeRow:
    batch_size: int
    block_size: int
    scheduler: str
    trial: int
    request_id: int
    block_index: int
    prompt_id: int
    prompt_difficulty: str
    steps_needed: int
    steps_executed: int
    useful_token_steps: int
    executed_token_steps: int
    token_step_min: int
    token_step_mean: float
    token_step_max: int
    block_internal_waste_ratio: float
    latency_ms: float
    tokens_generated: int


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


def choose_mixed_batch(prompts: list[PromptRecord], batch_size: int, trial: int) -> list[PromptRecord]:
    by_level = {level: [p for p in prompts if p.difficulty == level] for level in ["easy", "medium", "hard", "extreme"]}
    batch: list[PromptRecord] = []
    cursor = trial
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


def choose_mask_token_id(tokenizer: AutoTokenizer) -> int:
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer 没有 mask token；请使用 masked diffusion LM，例如 LLaDA。")
    return int(token_id)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_prompt_batch(
    tokenizer: AutoTokenizer,
    prompts: list[PromptRecord],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts and keep an attention mask so padding is not context."""

    encoded = [tokenizer(item.prompt, add_special_tokens=True)["input_ids"] for item in prompts]
    max_prompt = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = [ids + [pad_id] * (max_prompt - len(ids)) for ids in encoded]
    masks = [[1] * len(ids) + [0] * (max_prompt - len(ids)) for ids in encoded]
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def append_mask_block(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Append one masked block to every request and return its shared span."""

    batch_size = input_ids.shape[0]
    block_start = input_ids.shape[1]
    block_end = block_start + block_size
    mask_block = torch.full(
        (batch_size, block_size),
        mask_token_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    mask_attention = torch.ones_like(mask_block, device=input_ids.device)
    return torch.cat([input_ids, mask_block], dim=1), torch.cat([attention_mask, mask_attention], dim=1), (block_start, block_end)


def make_probe_rows(
    prompts: list[PromptRecord],
    token_steps: torch.Tensor,
    step_latencies_ms: list[float],
    batch_size: int,
    block_size: int,
    trial: int,
    block_index: int,
) -> list[ProbeRow]:
    """Convert one finished block's token_steps into sync/dynamic_oracle CSV rows."""

    batch_finish_steps = int(token_steps.max().item())
    full_batch_latency = sum(step_latencies_ms[:batch_finish_steps])
    rows: list[ProbeRow] = []
    for request_id, prompt in enumerate(prompts):
        steps = [int(x) for x in token_steps[request_id].tolist()]
        steps_needed = max(steps)
        useful_token_steps = sum(steps)
        block_internal_waste = (steps_needed * block_size) / useful_token_steps
        common = dict(
            batch_size=batch_size,
            block_size=block_size,
            trial=trial,
            request_id=request_id,
            block_index=block_index,
            prompt_id=prompt.prompt_id,
            prompt_difficulty=prompt.difficulty,
            steps_needed=steps_needed,
            useful_token_steps=useful_token_steps,
            token_step_min=min(steps),
            token_step_mean=mean(steps),
            token_step_max=max(steps),
            block_internal_waste_ratio=block_internal_waste,
            tokens_generated=block_size,
        )
        rows.append(
            ProbeRow(
                scheduler="sync",
                steps_executed=batch_finish_steps,
                executed_token_steps=batch_finish_steps * block_size,
                latency_ms=full_batch_latency,
                **common,
            )
        )
        rows.append(
            ProbeRow(
                scheduler="dynamic_oracle",
                steps_executed=steps_needed,
                executed_token_steps=useful_token_steps,
                latency_ms=sum(step_latencies_ms[:steps_needed]),
                **common,
            )
        )
    return rows


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
    trial: int,
    device: torch.device,
) -> list[ProbeRow]:
    """Probe multiple blocks; each later block is conditioned on previous blocks."""

    mask_token_id = choose_mask_token_id(tokenizer)
    input_ids, attention_mask = build_prompt_batch(tokenizer, prompts, device)
    rows: list[ProbeRow] = []

    for block_index in range(num_blocks):
        input_ids, attention_mask, (block_start, block_end) = append_mask_block(
            input_ids,
            attention_mask,
            block_size,
            mask_token_id,
        )
        token_steps = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        step_latencies_ms: list[float] = []

        for step in range(1, max_steps + 1):
            still_masked = token_steps.eq(0)
            if not bool(still_masked.any()):
                break

            cuda_sync()
            start = time.perf_counter()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            cuda_sync()
            step_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            for request_id in range(batch_size):
                masked_positions = token_steps[request_id].eq(0)
                if not bool(masked_positions.any()):
                    continue
                block_logits = logits[request_id, block_start:block_end]
                probs = torch.softmax(block_logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)
                accept = masked_positions & confidence.ge(confidence_threshold)
                if not bool(accept.any()):
                    masked_conf = confidence.masked_fill(~masked_positions, -1.0)
                    accept[torch.argmax(masked_conf)] = True
                absolute_positions = torch.arange(block_start, block_end, device=device)[accept]
                input_ids[request_id, absolute_positions] = predicted[accept]
                token_steps[request_id, accept] = step

        token_steps = token_steps.masked_fill(token_steps.eq(0), max_steps)
        rows.extend(
            make_probe_rows(
                prompts=prompts,
                token_steps=token_steps,
                step_latencies_ms=step_latencies_ms,
                batch_size=batch_size,
                block_size=block_size,
                trial=trial,
                block_index=block_index,
            )
        )
    return rows

def write_rows(rows: list[ProbeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_heterogeneous.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/h100_llada/per_request_rows.csv"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    prompts = read_prompts(args.prompts)
    rows: list[ProbeRow] = []
    for block_size in args.block_sizes:
        for batch_size in args.batch_sizes:
            if len(prompts) < batch_size:
                raise ValueError(f"prompts 数量至少要有 batch_size={batch_size} 条")
            for trial in range(args.trials):
                batch_prompts = choose_mixed_batch(prompts, batch_size, trial)
                rows.extend(
                    probe_one_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch_prompts,
                        batch_size=batch_size,
                        block_size=block_size,
                        num_blocks=args.num_blocks,
                        max_steps=args.max_steps,
                        confidence_threshold=args.confidence_threshold,
                        trial=trial,
                        device=device,
                    )
                )
                print(f"done block_size={block_size} batch={batch_size} trial={trial} num_blocks={args.num_blocks}")
    write_rows(rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
