#!/usr/bin/env python3
"""Run observation-first DiffusionGemma attention analysis on native generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from dllm.models import GenerationRequest, create_adapter

from .analyze import analyze_result_directory
from .hooks import install_attention_observer
from .proxy import AttentionObservationConfig, AttentionObserver


DEFAULT_PROMPTS = (
    "Compute 17 * 23. Give only the final integer.",
    "A box contains 12 red and 8 blue balls. What fraction are blue? Give the reduced fraction.",
    "Explain in one sentence why the sky appears blue during the day.",
    "Continue the pattern 2, 6, 12, 20, 30 with the next two numbers.",
)
DEFAULT_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"


def _indices(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(sorted({int(item) for item in value.split(",")}))


def _load_prompts(path: Path | None, limit: int, *, balanced_by_task: bool) -> list[dict[str, Any]]:
    if path is None:
        return [
            {"request_id": f"smoke-{index}", "prompt": prompt}
            for index, prompt in enumerate(DEFAULT_PROMPTS[:limit])
        ]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            prompt = value if isinstance(value, str) else value.get("prompt", value.get("input"))
            if not isinstance(prompt, str):
                raise ValueError(f"prompt row {index} has no string prompt/input")
            request_id = str(value.get("request_id", value.get("id", index))) if isinstance(value, dict) else str(index)
            row = dict(value) if isinstance(value, dict) else {}
            row.update(request_id=request_id, prompt=prompt)
            rows.append(row)
    if not rows:
        raise ValueError("no prompts were loaded")
    if balanced_by_task:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("task", "unclassified")), []).append(row)
        selected = []
        offset = 0
        task_names = sorted(grouped)
        while len(selected) < limit:
            added = False
            for task in task_names:
                if offset < len(grouped[task]):
                    selected.append(grouped[task][offset])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            offset += 1
        return selected
    return rows[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/attention/analysis/main")
    )
    parser.add_argument("--model-path", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--balanced-by-task",
        action="store_true",
        help="round-robin cached manifest tasks while preserving each row's prompt and inference seed",
    )
    parser.add_argument(
        "--use-manifest-generation-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--q-block-size", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--physical-q-tile-size", type=int, default=128)
    parser.add_argument("--physical-kv-tile-size", type=int, default=64)
    parser.add_argument("--proxy", choices=("mean", "max", "both"), default="both")
    parser.add_argument(
        "--layers",
        default="0,5,12,17,24,29",
        help="comma-separated explicit layer sample; default spans local/global and depth",
    )
    parser.add_argument("--heads", default=None, help="comma-separated explicit head sample; default records all")
    parser.add_argument("--save-tile-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analyze-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analyze-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase0-parity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase0-prompts", type=int, default=2)
    parser.add_argument("--replay-mask", choices=("none", "fresh", "previous"), default="none")
    parser.add_argument("--refresh-interval", type=int, default=1)
    parser.add_argument("--uncertainty-band", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--skip-offline-analysis", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.steps <= 0 or args.max_new_tokens <= 0:
        parser.error("num-samples, steps, and max-new-tokens must be positive")
    if not 0 <= args.phase0_prompts <= args.num_samples:
        parser.error("phase0-prompts must be between zero and num-samples")
    if args.refresh_interval <= 0 or args.uncertainty_band < 0:
        parser.error("refresh-interval must be positive and uncertainty-band non-negative")
    if args.replay_mask != "none":
        parser.error("live replay is phase-gated; run observation first and use the generated decisions.json")
    return args


class LogitCapture:
    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []
        self.handle = None

    def __call__(self, module, args, kwargs, output) -> None:
        del module, args, kwargs
        if isinstance(output, torch.Tensor):
            # This hook is installed on lm_head. Reproduce the immediately
            # following model soft-cap so the captured values are the exact
            # final logits returned by DiffusionGemma.forward.
            logits = output.float() / self.softcap
            logits = torch.tanh(logits) * self.softcap
            self.values.append(logits.detach().to(device="cpu", copy=True))

    def install(self, model) -> "LogitCapture":
        self.softcap = float(model.final_logit_softcapping)
        self.handle = model.lm_head.register_forward_hook(self, with_kwargs=True)
        return self

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _request(args: argparse.Namespace, prompt_row: dict[str, Any], seed: int) -> GenerationRequest:
    extra: dict[str, Any] = {"thinking": args.thinking}
    if args.temperature:
        extra["top_p"] = args.top_p
    return GenerationRequest(
        prompt=prompt_row["prompt"],
        max_new_tokens=(
            int(prompt_row.get("tokens_to_generate", args.max_new_tokens))
            if args.use_manifest_generation_length
            else args.max_new_tokens
        ),
        block_size=256,
        steps=args.steps,
        temperature=args.temperature,
        seed=seed,
        extra=extra,
    )


def _generate_with_logits(adapter, request) -> tuple[Any, list[torch.Tensor]]:
    capture = LogitCapture().install(adapter.model)
    try:
        result = adapter.generate(request)
    finally:
        capture.close()
    return result, capture.values


def _compare_logits(baseline: list[torch.Tensor], observed: list[torch.Tensor]) -> dict[str, Any]:
    if not baseline and not observed:
        return {
            "forward_count_baseline": 0,
            "forward_count_instrumented": 0,
            "shape_match": False,
            "max_output_logit_difference": float("inf"),
            "logits_bitwise_identical": False,
            "validation_error": "no logit-bearing forwards were captured",
        }
    if len(baseline) != len(observed):
        return {
            "forward_count_baseline": len(baseline),
            "forward_count_instrumented": len(observed),
            "shape_match": False,
            "max_output_logit_difference": float("inf"),
            "logits_bitwise_identical": False,
        }
    maximum = 0.0
    identical = True
    shape_match = True
    for left, right in zip(baseline, observed):
        if left.shape != right.shape:
            shape_match = False
            maximum = float("inf")
            identical = False
            break
        identical &= torch.equal(left, right)
        maximum = max(maximum, float((left.float() - right.float()).abs().max().item()))
    return {
        "forward_count_baseline": len(baseline),
        "forward_count_instrumented": len(observed),
        "shape_match": shape_match,
        "max_output_logit_difference": maximum,
        "logits_bitwise_identical": bool(identical),
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.prompts_jsonl, args.num_samples, balanced_by_task=args.balanced_by_task)
    adapter = create_adapter(
        "diffusion_gemma",
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    observation_config = AttentionObservationConfig(
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
        physical_q_tile_size=args.physical_q_tile_size,
        physical_kv_tile_size=args.physical_kv_tile_size,
        layers=_indices(args.layers),
        heads=_indices(args.heads),
        save_tile_stats=args.save_tile_stats,
        analyze_temporal=args.analyze_temporal,
        analyze_prefix=args.analyze_prefix,
    )
    observer = AttentionObserver(observation_config)
    generations = []
    parity = []
    started = time.time()

    # Phase 0: native run, then the same seeded run through the read-only
    # observer wrapper.  Per-forward logits are compared, not just tokens.
    parity_count = args.phase0_prompts if args.phase0_parity else 0
    for index, prompt_row in enumerate(prompts[:parity_count]):
        seed = int(prompt_row.get("inference_seed", args.base_seed + index))
        request = _request(args, prompt_row, seed)
        baseline_result, baseline_logits = _generate_with_logits(adapter, request)
        observer.begin_request(prompt_row["request_id"], num_denoising_steps=args.steps)
        with install_attention_observer(adapter.model, observer):
            observed_result, observed_logits = _generate_with_logits(adapter, request)
        comparison = _compare_logits(baseline_logits, observed_logits)
        comparison.update(
            request_id=prompt_row["request_id"],
            seed=seed,
            generated_tokens_identical=baseline_result.completion_tokens == observed_result.completion_tokens,
            generated_text_identical=baseline_result.text == observed_result.text,
        )
        parity.append(comparison)
        generations.append({
            "request_id": prompt_row["request_id"],
            "prompt": prompt_row["prompt"],
            "task": prompt_row.get("task"),
            "source_sample_id": prompt_row.get("sample_id"),
            "expected_outputs": prompt_row.get("outputs"),
            "seed": seed,
            "completion_tokens": observed_result.completion_tokens,
            "text": observed_result.text,
            "model_evaluations": observed_result.model_evaluations,
            "metadata": observed_result.metadata,
        })
        observer.export_shard(args.output_dir / "raw", f"{index:04d}")
        del baseline_logits, observed_logits

    if parity:
        parity_summary = {
            "prompts": parity,
            "maximum_output_logit_difference": max(row["max_output_logit_difference"] for row in parity),
            "all_logits_bitwise_identical": all(row["logits_bitwise_identical"] for row in parity),
            "all_generated_tokens_identical": all(row["generated_tokens_identical"] for row in parity),
        }
    else:
        parity_summary = {"prompts": [], "not_run": True}
    (args.output_dir / "phase0_parity.json").write_text(
        json.dumps(parity_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Remaining prompts need only the observed native-dense run.
    if parity_count < len(prompts):
        with install_attention_observer(adapter.model, observer):
            for index, prompt_row in enumerate(prompts[parity_count:], start=parity_count):
                seed = int(prompt_row.get("inference_seed", args.base_seed + index))
                observer.begin_request(prompt_row["request_id"], num_denoising_steps=args.steps)
                result = adapter.generate(_request(args, prompt_row, seed))
                generations.append({
                    "request_id": prompt_row["request_id"],
                    "prompt": prompt_row["prompt"],
                    "task": prompt_row.get("task"),
                    "source_sample_id": prompt_row.get("sample_id"),
                    "expected_outputs": prompt_row.get("outputs"),
                    "seed": seed,
                    "completion_tokens": result.completion_tokens,
                    "text": result.text,
                    "model_evaluations": result.model_evaluations,
                    "metadata": result.metadata,
                })
                observer.export_shard(args.output_dir / "raw", f"{index:04d}")

    (args.output_dir / "generations.json").write_text(
        json.dumps(generations, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    hardware = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else args.device
    )
    observer.export(
        args.output_dir,
        {
            **adapter.runtime_metadata(),
            "model": args.model_path,
            "model_revision": args.revision,
            "git_commit": _git_commit(),
            "hardware": hardware,
            "dtype": args.precision,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dataset": str(args.prompts_jsonl) if args.prompts_jsonl else "built-in smoke prompts",
            "dataset_selection": {
                "balanced_by_task": args.balanced_by_task,
                "preserve_manifest_inference_seed": True,
                "use_manifest_generation_length": args.use_manifest_generation_length,
                "request_ids": [row["request_id"] for row in prompts],
                "tasks": [row.get("task") for row in prompts],
            },
            "num_samples": len(prompts),
            "decoding_configuration": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p if args.temperature else None,
                "thinking": args.thinking,
                "base_seed": args.base_seed,
            },
            "denoising_configuration": {"requested_max_denoising_steps": args.steps},
            "proxy_selection": args.proxy,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "phase0_parity": parity_summary,
        },
    )
    if not args.skip_offline_analysis:
        analyze_result_directory(args.output_dir)


if __name__ == "__main__":
    main()
