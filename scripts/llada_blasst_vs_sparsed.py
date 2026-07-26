#!/usr/bin/env python3
"""Reproduce official SparseD and compare it with BLASST on identical LLaDA runs.

SparseD is executed from the pinned INV-WZQ/SparseD artifact.  All methods use
the authors' denoising semantics, with the final vocabulary projection limited
to the generated suffix.  Prompt-position logits are never consumed by LLaDA
generation; avoiding them keeps the official 64k sweep within an 80 GiB GPU.

The default profile matches the authors' long-context latency sweep: batch 1,
LLaDA-1.5, 128 generated tokens, 128 denoising steps, block length 32,
temperature 0, low-confidence remasking, SparseD skip=0.2/select=0.3/block=128,
and the official 4k/8k/16k/32k/64k prompts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from llada_eval_utils import (  # noqa: E402
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


SPARSED_REPOSITORY = "https://github.com/INV-WZQ/SparseD.git"
SPARSED_COMMIT = "52155b4aa78368695f96744cade06d2e0865d085"
DEFAULT_ARTIFACT = ROOT / "reference" / "SparseD"
DEFAULT_OUTPUT = ROOT / "outputs" / "llada_official_sparsed_vs_blasst.json"
PAPER_PROMPTS = ("4k", "8k", "16k", "32k", "64k")
EXPECTED_ANSWERS = {
    "short_context": "5",
    "4k": "yes",
    "8k": "yes",
    "16k": "yes",
    "32k": "yes",
    "64k": "yes",
}


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one value is required")
    if len(values) != len(set(values)):
        raise ValueError("values must be unique")
    return values


def normalize_answer(value: str) -> str:
    value = value.lower().replace("\\boxed", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def answer_metrics(text: str, expected: str | None) -> dict[str, Any]:
    normalized = normalize_answer(text)
    expected_normalized = normalize_answer(expected or "")
    return {
        "decoded_text": text,
        "normalized_text": normalized,
        "expected_answer": expected,
        "exact_expected_answer": bool(expected_normalized) and normalized == expected_normalized,
        "expected_answer_prefix": bool(expected_normalized) and (
            normalized == expected_normalized
            or normalized.startswith(expected_normalized + " ")
        ),
    }


def length_scaled_schedule_values(
    total_sequence_length: int,
    *,
    calibration_length: int = 4096,
) -> dict[str, float]:
    """Apply BLASST's lambda proportional to 1/L rule to calibrated defaults."""
    if total_sequence_length <= 0 or calibration_length <= 0:
        raise ValueError("sequence and calibration lengths must be positive")
    scale = calibration_length / total_sequence_length
    return {
        "high_noise_lambda": min(1.0, 0.04858582466840744 * scale),
        "mid_noise_lambda": min(1.0, 0.4334796965122223 * scale),
        "low_noise_lambda": min(1.0, 1.0 * scale),
        "high_noise_boundary": 0.75,
        "low_noise_boundary": 0.25,
    }


def artifact_revision(path: pathlib.Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_artifact(path: pathlib.Path, allow_revision_mismatch: bool) -> str:
    required = (
        path / "llada_generation.py",
        path / "prompts.json",
        path / "models" / "LLaDA" / "modeling_llada.py",
        path / "models" / "LLaDA" / "generate.py",
        path / "models" / "LLaDA" / "SparseD_utils.py",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "official SparseD artifact is missing; run "
            f"`git clone --depth 1 {SPARSED_REPOSITORY} {path}`; missing: {missing}"
        )
    revision = artifact_revision(path)
    if revision is None:
        revision = "unknown_unversioned_artifact"
    elif revision != SPARSED_COMMIT and not allow_revision_mismatch:
        raise ValueError(
            f"SparseD revision {revision} differs from validated revision {SPARSED_COMMIT}; "
            "pass --allow-artifact-revision-mismatch to run intentionally"
        )
    return revision


def import_official_sparsed(path: pathlib.Path) -> type:
    sys.path.insert(0, str(path))
    module = importlib.import_module("models.LLaDA")
    return module.LLaDAModelLM


@contextmanager
def flash_sdpa_only() -> Iterator[None]:
    """Force the paper's dense FlashAttention baseline for supported SDPA calls."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exact helper semantics from the official SparseD LLaDA generator."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    result = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for row in range(mask_num.size(0)):
        result[row, : remainder[row]] += 1
    return result


@contextmanager
def generation_suffix_projection(
    model: torch.nn.Module, prompt_length: int
) -> Iterator[None]:
    """Project vocabulary logits only for positions generation can update.

    The official LLaDA model applies ``ln_f`` immediately before its vocabulary
    projection.  Slicing that module's output leaves every transformer layer and
    every attention query unchanged while reducing the 64k BF16 logits tensor
    from roughly 16.6 GiB to about 31 MiB for 128 generated tokens.
    """
    if prompt_length < 0:
        raise ValueError("prompt length must be non-negative")
    try:
        final_norm = model.model.transformer.ln_f
    except AttributeError as exc:
        raise TypeError("expected an official SparseD LLaDA model") from exc

    def slice_suffix(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output[:, prompt_length:, :]

    handle = final_norm.register_forward_hook(slice_suffix)
    try:
        yield
    finally:
        handle.remove()


def clear_sparsed_runtime_state(model: torch.nn.Module) -> None:
    """Release per-shape SparseD masks without changing model parameters."""
    for module in model.modules():
        for attribute in ("fine_mask", "block_mask", "last"):
            if hasattr(module, attribute):
                setattr(module, attribute, None)


def release_method_memory(model: torch.nn.Module) -> None:
    """Prevent one prompt length's temporary allocations affecting the next."""
    clear_sparsed_runtime_state(model)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def generate_memory_efficient(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    sparsed_parameters: dict[str, Any] | None = None,
    controller: Any | None = None,
) -> torch.Tensor:
    """Official LLaDA denoising with a generation-only vocabulary projection."""
    if prompt.shape[0] != 1:
        raise ValueError("official SparseD LLaDA generation supports batch size 1")
    if prompt.eq(mask_id).any():
        raise ValueError("the fixed prompt must not contain the LLaDA mask token")
    prompt_length = int(prompt.shape[1])
    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt.clone()
    if gen_length % block_length:
        raise ValueError("generation length must be divisible by block length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by the number of generation blocks")
    steps_per_block = steps // num_blocks

    generated = x[:, prompt_length:]
    with generation_suffix_projection(model, prompt_length):
        for block in range(num_blocks):
            block_start = block * block_length
            block_end = (block + 1) * block_length
            block_mask = generated[:, block_start:block_end].eq(mask_id)
            transfers = get_num_transfer_tokens(block_mask, steps_per_block)
            for step in range(steps_per_block):
                mask_index = generated.eq(mask_id)
                if controller is not None:
                    controller.set_remaining_mask_ratio(mask_index.float().mean(dim=1))
                if sparsed_parameters is not None:
                    sparsed_parameters["now_step"] = step + block * steps_per_block
                    logits = model(x, SparseD_param=sparsed_parameters).logits
                else:
                    logits = model(x).logits
                if logits.shape[:2] != generated.shape:
                    raise RuntimeError(
                        "generation-only projection returned an unexpected shape: "
                        f"{tuple(logits.shape)}"
                    )
                noisy_logits = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(noisy_logits, dim=-1)
                if remasking == "low_confidence":
                    probabilities = F.softmax(logits, dim=-1)
                    confidence = torch.gather(
                        probabilities, -1, x0.unsqueeze(-1)
                    ).squeeze(-1)
                elif remasking == "random":
                    confidence = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError("remasking must be low_confidence or random")
                confidence[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, generated)
                confidence = torch.where(mask_index, confidence, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for row in range(confidence.shape[0]):
                    count = int(transfers[row, step])
                    _, selected = torch.topk(confidence[row], k=count)
                    transfer_index[row, selected] = True
                generated[transfer_index] = x0[transfer_index]
    return x


def timed_generation(
    run: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    for _ in range(warmup):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        run()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    outputs: list[torch.Tensor] = []
    for _ in range(repeats):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        outputs.append(output.detach().clone())
    ordered = sorted(samples)
    stable = all(torch.equal(outputs[0], output) for output in outputs[1:])
    return outputs[-1], {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "samples_ms": samples,
        "deterministic_across_repeats": stable,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def token_agreement(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference generations must have identical shapes")
    return float(candidate.eq(reference).float().mean())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compare official SparseD and fused BLASST with the paper's LLaDA settings."
    )
    result.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
    result.add_argument("--allow-artifact-revision-mismatch", action="store_true")
    result.add_argument("--model", default="GSAI-ML/LLaDA-1.5")
    result.add_argument("--prompts", default=",".join(PAPER_PROMPTS))
    result.add_argument("--seq-len", type=int, default=128)
    result.add_argument("--steps", type=int, default=128)
    result.add_argument("--block-length", type=int, default=32)
    result.add_argument("--sampling-alg", default="low_confidence")
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--mask-id", type=int, default=126336)
    result.add_argument("--sparsed-skip", type=float, default=0.2)
    result.add_argument("--sparsed-select", type=float, default=0.3)
    result.add_argument("--sparsed-block-size", type=int, default=128)
    result.add_argument("--blasst-calibration-length", type=int, default=4096)
    result.add_argument("--num-warps", type=int, default=4)
    result.add_argument("--pipeline-stages", type=int, default=2)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260716)
    result.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    result.add_argument("--collect-blasst-stats", action="store_true")
    return result


def validate_args(args: argparse.Namespace, available_prompts: Sequence[str]) -> list[str]:
    prompts = parse_csv(args.prompts)
    unknown = sorted(set(prompts) - set(available_prompts))
    if unknown:
        raise ValueError(f"unknown official SparseD prompts: {unknown}")
    for name in ("seq_len", "steps", "block_length", "warmup", "repeats"):
        if getattr(args, name) < (0 if name == "warmup" else 1):
            raise ValueError(f"--{name.replace('_', '-')} has an invalid value")
    if args.seq_len % args.block_length:
        raise ValueError("--seq-len must be divisible by --block-length")
    if args.steps % (args.seq_len // args.block_length):
        raise ValueError("--steps must be divisible by seq_len/block_length")
    if not 0.0 <= args.sparsed_skip <= 1.0:
        raise ValueError("--sparsed-skip must be in [0, 1]")
    if not 0.0 < args.sparsed_select <= 1.0:
        raise ValueError("--sparsed-select must be in (0, 1]")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    return prompts


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official SparseD/BLASST comparison")
    revision = validate_artifact(args.artifact, args.allow_artifact_revision_mismatch)
    prompts_payload = json.loads((args.artifact / "prompts.json").read_text(encoding="utf-8"))[
        "questions"
    ]
    prompt_names = validate_args(args, prompts_payload)
    LLaDAModelLM = import_official_sparsed(args.artifact)
    from transformers import AutoTokenizer
    from blasst import (
        DiffusionLambdaSchedule,
        get_kernel_stats,
        install_diffusion_blasst_kernel,
        reset_kernel_stats,
    )

    with patch_llada_transformers_compat():
        model = LLaDAModelLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    report: dict[str, Any] = {
        "schema_version": "official-sparsed-vs-blasst-v2",
        "status": "running",
        "artifact": {
            "repository": SPARSED_REPOSITORY,
            "revision": revision,
            "validated_revision": SPARSED_COMMIT,
            "path": str(args.artifact),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": sys.version,
        },
        "paper_settings": {
            "model": args.model,
            "batch_size": 1,
            "prompt_names": prompt_names,
            "generated_tokens": args.seq_len,
            "denoising_steps": args.steps,
            "generation_block_length": args.block_length,
            "temperature": args.temperature,
            "remasking": args.sampling_alg,
            "mask_id": args.mask_id,
            "sparsed_skip": args.sparsed_skip,
            "sparsed_select": args.sparsed_select,
            "sparsed_block_size": args.sparsed_block_size,
            "dense_backend": "PyTorch SDPA forced to FLASH_ATTENTION",
            "vocabulary_projection": f"generated {args.seq_len}-token suffix only",
            "warmup_generations_per_method_and_shape": args.warmup,
            "timed_generations_per_method_and_shape": args.repeats,
        },
        "blasst_settings": {
            "kernel": "fused bidirectional 128x64 BLASST",
            "lambda_source": "LLaDA 4096-token calibrated noise schedule scaled by calibration_length/L",
            "calibration_length": args.blasst_calibration_length,
            "num_warps": args.num_warps,
            "pipeline_stages": args.pipeline_stages,
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for prompt_name in prompt_names:
        messages = [{"role": "user", "content": prompts_payload[prompt_name]}]
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt = tokenizer(
            rendered, return_tensors="pt", padding=True, padding_side="left"
        ).input_ids.to("cuda")
        total_length = int(prompt.shape[1] + args.seq_len)
        expected_answer = EXPECTED_ANSWERS.get(prompt_name)
        generation_kwargs = {
            "steps": args.steps,
            "gen_length": args.seq_len,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "remasking": args.sampling_alg,
            "mask_id": args.mask_id,
        }

        release_method_memory(model)

        def run_dense() -> torch.Tensor:
            with flash_sdpa_only():
                return generate_memory_efficient(model, prompt, **generation_kwargs)

        dense_output, dense_timing = timed_generation(
            run_dense, warmup=args.warmup, repeats=args.repeats, seed=args.seed
        )
        dense_generated = dense_output[:, prompt.shape[1] :]
        dense_text = tokenizer.batch_decode(dense_generated, skip_special_tokens=True)[0]
        release_method_memory(model)

        def run_sparsed() -> torch.Tensor:
            parameters = {
                "skip": args.sparsed_skip,
                "select": args.sparsed_select,
                "block_size": args.sparsed_block_size,
                "new_generation": args.seq_len,
                "whole_steps": args.steps,
            }
            with flash_sdpa_only():
                return generate_memory_efficient(
                    model,
                    prompt,
                    sparsed_parameters=parameters,
                    **generation_kwargs,
                )

        sparsed_output, sparsed_timing = timed_generation(
            run_sparsed, warmup=args.warmup, repeats=args.repeats, seed=args.seed
        )
        sparsed_generated = sparsed_output[:, prompt.shape[1] :]
        sparsed_text = tokenizer.batch_decode(sparsed_generated, skip_special_tokens=True)[0]
        release_method_memory(model)

        schedule_values = length_scaled_schedule_values(
            total_length, calibration_length=args.blasst_calibration_length
        )
        schedule = DiffusionLambdaSchedule.from_dict(schedule_values)
        with install_diffusion_blasst_kernel(
            model,
            schedule=schedule,
            collect_stats=False,
            num_warps=args.num_warps,
            pipeline_stages=args.pipeline_stages,
        ) as controller:
            def run_blasst() -> torch.Tensor:
                return generate_memory_efficient(
                    model,
                    prompt,
                    controller=controller,
                    steps=args.steps,
                    gen_length=args.seq_len,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.sampling_alg,
                    mask_id=args.mask_id,
                )

            blasst_output, blasst_timing = timed_generation(
                run_blasst, warmup=args.warmup, repeats=args.repeats, seed=args.seed
            )
        release_method_memory(model)
        blasst_generated = blasst_output[:, prompt.shape[1] :]
        blasst_text = tokenizer.batch_decode(blasst_generated, skip_special_tokens=True)[0]

        blasst_stats = None
        if args.collect_blasst_stats:
            reset_kernel_stats(prompt.device)
            with install_diffusion_blasst_kernel(
                model,
                schedule=schedule,
                collect_stats=True,
                num_warps=args.num_warps,
                pipeline_stages=args.pipeline_stages,
            ) as stats_controller:
                generate_memory_efficient(
                    model,
                    prompt,
                    controller=stats_controller,
                    steps=args.steps,
                    gen_length=args.seq_len,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.sampling_alg,
                    mask_id=args.mask_id,
                )
                torch.cuda.synchronize()
            release_method_memory(model)
            stats = get_kernel_stats(reset=True)
            blasst_stats = {
                "skipped_tiles": stats.skipped_tiles,
                "total_tiles": stats.total_tiles,
                "physical_tile_sparsity": stats.sparsity_ratio,
                "bmm2_flop_skip_rate": stats.bmm2_flop_skip_ratio,
                "v_load_skip_rate": stats.v_load_skip_ratio,
            }

        dense_ms = float(dense_timing["median_ms"])
        sparsed_ms = float(sparsed_timing["median_ms"])
        blasst_ms = float(blasst_timing["median_ms"])
        row = {
            "prompt": prompt_name,
            "prompt_tokens": int(prompt.shape[1]),
            "total_sequence_length": total_length,
            "expected_answer": expected_answer,
            "dense": {
                "timing": dense_timing,
                "speedup_vs_dense": 1.0,
                "accuracy": answer_metrics(dense_text, expected_answer),
            },
            "sparsed_official": {
                "timing": sparsed_timing,
                "speedup_vs_dense": dense_ms / sparsed_ms,
                "accuracy": {
                    **answer_metrics(sparsed_text, expected_answer),
                    "generated_token_agreement_vs_dense": token_agreement(
                        sparsed_generated, dense_generated
                    ),
                    "exact_generated_tokens_vs_dense": bool(
                        torch.equal(sparsed_generated, dense_generated)
                    ),
                },
            },
            "blasst": {
                "timing": blasst_timing,
                "speedup_vs_dense": dense_ms / blasst_ms,
                "speedup_vs_sparsed": sparsed_ms / blasst_ms,
                "length_scaled_lambda_schedule": schedule_values,
                "accuracy": {
                    **answer_metrics(blasst_text, expected_answer),
                    "generated_token_agreement_vs_dense": token_agreement(
                        blasst_generated, dense_generated
                    ),
                    "exact_generated_tokens_vs_dense": bool(
                        torch.equal(blasst_generated, dense_generated)
                    ),
                },
                "stats": blasst_stats,
            },
        }
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)

    results = report["results"]
    report["summary"] = {
        "prompt_count": len(results),
        "dense_expected_answer_accuracy": sum(
            int(row["dense"]["accuracy"]["expected_answer_prefix"]) for row in results
        ) / len(results),
        "sparsed_expected_answer_accuracy": sum(
            int(row["sparsed_official"]["accuracy"]["expected_answer_prefix"])
            for row in results
        ) / len(results),
        "blasst_expected_answer_accuracy": sum(
            int(row["blasst"]["accuracy"]["expected_answer_prefix"]) for row in results
        ) / len(results),
        "note": (
            "Accuracy here scores the official example prompts only. Reproducing paper Table 1 "
            "requires the authors' external MMLU/GSM8K/HumanEval/RULER evaluation harness."
        ),
    }
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
