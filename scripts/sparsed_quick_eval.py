#!/usr/bin/env python3
"""Small dense-vs-SparseD valuation harness for LLaDA.

This intentionally is not a benchmark-suite reproduction.  It uses the official
SparseD model/generation implementation as a backend and measures a few prompts
twice: original dense attention and SparseD attention with identical decoding
settings.  Dense-output agreement is a fidelity proxy, not task accuracy.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measurement:
    prompt_id: str
    mode: str
    repeat: int
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    tokens_per_second: float
    peak_memory_gib: float
    output_text: str
    output_token_ids: str
    reference_match: bool | None


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def reference_match(text: str, references: Sequence[str]) -> bool | None:
    if not references:
        return None
    normalized = normalize_text(text)
    return any(normalize_text(answer) in normalized for answer in references)


def token_agreement(left: Sequence[int], right: Sequence[int]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    matches = sum(a == b for a, b in zip(left, right))
    return matches / denominator


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def read_prompts(path: Path, prompt_ids: set[str] | None, max_prompts: int | None) -> list[PromptCase]:
    """Read JSONL, a JSON list, or SparseD's {questions: {name: text}} file."""

    def make_case(item: dict[str, Any], fallback_id: str) -> PromptCase:
        answers = item.get("references", item.get("answers", item.get("reference", item.get("answer", ()))))
        if isinstance(answers, str):
            answers = (answers,)
        return PromptCase(
            prompt_id=str(item.get("id", item.get("prompt_id", fallback_id))),
            prompt=str(item["prompt"]),
            references=tuple(str(answer) for answer in answers),
        )

    if path.suffix == ".jsonl":
        raw_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cases = [make_case(item, str(index)) for index, item in enumerate(raw_items)]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("questions"), dict):
            cases = [PromptCase(str(key), str(value)) for key, value in payload["questions"].items()]
        elif isinstance(payload, list):
            cases = [make_case(item, str(index)) for index, item in enumerate(payload)]
        else:
            raise ValueError("prompt JSON must be a list or contain a 'questions' object")
    if prompt_ids is not None:
        cases = [case for case in cases if case.prompt_id in prompt_ids]
    if max_prompts is not None:
        cases = cases[:max_prompts]
    if not cases:
        raise ValueError(f"no prompts selected from {path}")
    return cases


def import_sparsed(repo: Path) -> tuple[type[Any], Callable[..., Any]]:
    if not (repo / "models" / "LLaDA" / "generate.py").is_file():
        raise FileNotFoundError(
            f"SparseD backend not found at {repo}. Clone https://github.com/INV-WZQ/SparseD there "
            "or pass --sparsed-repo."
        )
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("models.LLaDA")
        return module.LLaDAModelLM, module.generate
    except Exception as exc:
        raise RuntimeError(
            "could not import the official SparseD backend; use its pinned environment "
            "(PyTorch 2.6 / transformers 4.46.2)"
        ) from exc


def build_input(tokenizer: Any, prompt: str, device: Any) -> Any:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(rendered, return_tensors="pt").input_ids.to(device)


def cuda_sync(torch: Any) -> None:
    torch.cuda.synchronize()


def generate_once(
    *,
    torch: Any,
    generate: Callable[..., Any],
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    args: argparse.Namespace,
    mode: str,
    measure: bool,
) -> tuple[list[int], str, float, float]:
    # Keep stochastic remasking paired across dense/sparse modes as well.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sparse_params = None
    if mode == "sparse":
        sparse_params = {
            "skip": args.skip,
            "select": args.select,
            "block_size": args.sparse_block_size,
            "new_generation": args.gen_length,
            "whole_steps": args.steps,
        }
    if measure:
        torch.cuda.reset_peak_memory_stats()
        cuda_sync(torch)
        started = time.perf_counter()
    output = generate(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking=args.remasking,
        mask_id=args.mask_token_id,
        SparseD_param=sparse_params,
    )
    if measure:
        cuda_sync(torch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        elapsed_ms = peak_gib = math.nan
    generated = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, elapsed_ms, peak_gib


def summarize(measurements: Sequence[Measurement]) -> dict[str, Any]:
    result: dict[str, Any] = {"modes": {}}
    by_mode: dict[str, list[Measurement]] = {}
    for row in measurements:
        by_mode.setdefault(row.mode, []).append(row)
    for mode, rows in by_mode.items():
        latencies = [row.latency_ms for row in rows]
        result["modes"][mode] = {
            "samples": len(rows),
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_p50": statistics.median(latencies),
            "latency_ms_p95": percentile(latencies, 0.95),
            "tokens_per_second_mean": statistics.fmean(row.tokens_per_second for row in rows),
            "peak_memory_gib_max": max(row.peak_memory_gib for row in rows),
        }
    dense = result["modes"].get("dense")
    sparse = result["modes"].get("sparse")
    if dense and sparse:
        result["efficiency"] = {
            "speedup_dense_over_sparse": dense["latency_ms_mean"] / sparse["latency_ms_mean"],
            "latency_reduction_fraction": 1.0 - sparse["latency_ms_mean"] / dense["latency_ms_mean"],
            "peak_memory_ratio_sparse_over_dense": sparse["peak_memory_gib_max"] / dense["peak_memory_gib_max"],
        }
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparsed-repo", type=Path, default=Path(os.environ.get("SPARSED_REPO", ROOT / "reference/SparseD")))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/prompts_heterogeneous.jsonl")
    parser.add_argument("--prompt-ids", help="Comma-separated IDs; supports SparseD prompts such as 4k,8k")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/sparsed_quick_eval")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--skip", type=float, default=0.2)
    parser.add_argument("--select", type=float, default=0.5)
    parser.add_argument("--sparse-block-size", type=int, default=32)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--mask-token-id", type=int, default=126336)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.gen_length <= 0 or args.block_length <= 0:
        raise ValueError("steps and generation lengths must be positive")
    if args.gen_length % args.block_length or args.steps % (args.gen_length // args.block_length):
        raise ValueError("gen-length must divide by block-length, and steps by the resulting block count")
    if not 0.0 <= args.skip <= 1.0 or not 0.0 < args.select <= 1.0:
        raise ValueError("skip must be in [0,1] and select in (0,1]")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install requirements-h100.txt and the official SparseD requirements first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: SparseD's FlexAttention path is GPU-only")

    prompt_ids = set(args.prompt_ids.split(",")) if args.prompt_ids else None
    cases = read_prompts(args.prompts, prompt_ids, args.max_prompts)
    model_class, generate = import_sparsed(args.sparsed_repo.resolve())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = model_class.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows: list[Measurement] = []
    comparisons: list[dict[str, Any]] = []
    for case in cases:
        input_ids = build_input(tokenizer, case.prompt, model.device)
        max_sequence_length = getattr(model.config, "max_sequence_length", None)
        if max_sequence_length and input_ids.shape[1] + args.gen_length > max_sequence_length:
            raise ValueError(
                f"prompt {case.prompt_id!r} plus generation is {input_ids.shape[1] + args.gen_length} "
                f"tokens, above model maximum {max_sequence_length}"
            )
        outputs: dict[str, tuple[list[int], str]] = {}
        mode_latencies: dict[str, list[float]] = {}
        for mode in ("dense", "sparse"):
            for _ in range(args.warmup):
                generate_once(
                    torch=torch, generate=generate, model=model, tokenizer=tokenizer,
                    input_ids=input_ids, args=args, mode=mode, measure=False,
                )
            for repeat in range(args.repeats):
                tokens, text, latency_ms, peak_gib = generate_once(
                    torch=torch, generate=generate, model=model, tokenizer=tokenizer,
                    input_ids=input_ids, args=args, mode=mode, measure=True,
                )
                outputs.setdefault(mode, (tokens, text))
                mode_latencies.setdefault(mode, []).append(latency_ms)
                rows.append(Measurement(
                    prompt_id=case.prompt_id, mode=mode, repeat=repeat,
                    prompt_tokens=int(input_ids.shape[1]), generated_tokens=len(tokens),
                    latency_ms=latency_ms, tokens_per_second=len(tokens) / (latency_ms / 1000.0),
                    peak_memory_gib=peak_gib, output_text=text,
                    output_token_ids=json.dumps(tokens),
                    reference_match=reference_match(text, case.references),
                ))
        dense_tokens, dense_text = outputs["dense"]
        sparse_tokens, sparse_text = outputs["sparse"]
        comparisons.append({
            "prompt_id": case.prompt_id,
            "prompt_tokens": int(input_ids.shape[1]),
            "dense_latency_ms_mean": statistics.fmean(mode_latencies["dense"]),
            "sparse_latency_ms_mean": statistics.fmean(mode_latencies["sparse"]),
            "speedup_dense_over_sparse": statistics.fmean(mode_latencies["dense"]) / statistics.fmean(mode_latencies["sparse"]),
            "token_agreement_with_dense": token_agreement(dense_tokens, sparse_tokens),
            "text_exact_match_with_dense": normalize_text(dense_text) == normalize_text(sparse_text),
            "text_similarity_with_dense": SequenceMatcher(None, normalize_text(dense_text), normalize_text(sparse_text)).ratio(),
            "dense_reference_match": reference_match(dense_text, case.references),
            "sparse_reference_match": reference_match(sparse_text, case.references),
        })
        print(f"[{case.prompt_id}] tokens={input_ids.shape[1]} speedup={comparisons[-1]['speedup_dense_over_sparse']:.3f}x agreement={comparisons[-1]['token_agreement_with_dense']:.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "measurements.csv", [asdict(row) for row in rows])
    write_csv(args.outdir / "comparisons.csv", comparisons)
    summary = summarize(rows)
    quality: dict[str, Any] = {
        "token_agreement_with_dense_mean": statistics.fmean(row["token_agreement_with_dense"] for row in comparisons),
        "text_exact_match_rate": statistics.fmean(float(row["text_exact_match_with_dense"]) for row in comparisons),
        "text_similarity_with_dense_mean": statistics.fmean(row["text_similarity_with_dense"] for row in comparisons),
    }
    scored = [row for row in comparisons if row["dense_reference_match"] is not None]
    if scored:
        dense_accuracy = statistics.fmean(float(row["dense_reference_match"]) for row in scored)
        sparse_accuracy = statistics.fmean(float(row["sparse_reference_match"]) for row in scored)
        quality["reference_scored_prompts"] = len(scored)
        quality["dense_reference_match_rate"] = dense_accuracy
        quality["sparse_reference_match_rate"] = sparse_accuracy
        quality["reference_match_rate_delta_sparse_minus_dense"] = sparse_accuracy - dense_accuracy
    summary.update({
        "scope": "quick feasibility test; dense agreement is not benchmark accuracy",
        "model": args.model,
        "prompts": str(args.prompts),
        "prompt_count": len(cases),
        "decoding": {key: getattr(args, key) for key in ("steps", "gen_length", "block_length", "remasking")},
        "sparsed": {"skip": args.skip, "select": args.select, "block_size": args.sparse_block_size},
        "quality": quality,
    })
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
