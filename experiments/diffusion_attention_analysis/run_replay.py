#!/usr/bin/env python3
"""Run Phase 8 fresh/previous-mask correctness emulation after its gate passes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from dllm.evaluation.ruler.official import score_predictions, verify_checkout
from dllm.models import create_adapter

from .hooks import install_attention_replay
from .replay import ReplayAttention, ReplayConfig
from .run_analysis import DEFAULT_REVISION, LogitCapture, _indices, _load_prompts, _request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation-dir", type=Path, default=Path("results/attention/analysis/main")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/attention/analysis/main/replay")
    )
    parser.add_argument("--model-path", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, default=Path("/tmp/NVIDIA-RULER"))
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--balanced-by-task", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-manifest-generation-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q-block-size", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--physical-q-tile-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.28)
    parser.add_argument(
        "--threshold-mode",
        choices=("gaussian", "empirical_quantile"),
        default="gaussian",
    )
    parser.add_argument("--target-density", type=float, default=0.5)
    parser.add_argument(
        "--proxies",
        choices=("mean", "max", "both"),
        default="both",
        help="proxy variants to execute",
    )
    parser.add_argument(
        "--eager-dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the eager dense implementation regression variant",
    )
    parser.add_argument("--layers", default=None, help="comma-separated layers; default applies replay to every decoder layer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.steps <= 0:
        parser.error("num-samples and steps must be positive")
    decisions_path = args.observation_dir / "decisions.json"
    if not decisions_path.exists():
        parser.error(f"missing Phase 3 decisions: {decisions_path}")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not decisions.get("previous_step_replay", {}).get("go", False):
        parser.error("Phase 3 previous-step replay gate did not pass")
    return args


class GenerationCapture:
    def __init__(self) -> None:
        self.logits = LogitCapture()
        self.hidden: list[torch.Tensor] = []
        self.hidden_handle = None

    def install(self, model) -> "GenerationCapture":
        self.logits.install(model)

        def capture_hidden(module, args, kwargs, output) -> None:
            del module, args, kwargs
            value = getattr(output, "last_hidden_state", None)
            if value is not None:
                self.hidden.append(value.detach().to(device="cpu", copy=True))

        self.hidden_handle = model.model.decoder.register_forward_hook(capture_hidden, with_kwargs=True)
        return self

    def close(self) -> None:
        self.logits.close()
        if self.hidden_handle is not None:
            self.hidden_handle.remove()
            self.hidden_handle = None


def _generate_capture(adapter, request):
    capture = GenerationCapture().install(adapter.model)
    try:
        result = adapter.generate(request)
    finally:
        capture.close()
    return result, capture


def _tensor_comparison(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, Any]:
    if not reference or not candidate:
        return {
            "reference_forwards": len(reference),
            "candidate_forwards": len(candidate),
            "forward_count_match": len(reference) == len(candidate),
            "max_absolute_difference": float("inf"),
            "relative_l2_error": float("inf"),
            "bitwise_identical": False,
        }
    maximum = 0.0
    difference_sq = 0.0
    reference_sq = 0.0
    identical = True
    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            return {
                "reference_forwards": len(reference),
                "candidate_forwards": len(candidate),
                "forward_count_match": True,
                "shape_match": False,
                "max_absolute_difference": float("inf"),
                "relative_l2_error": float("inf"),
                "bitwise_identical": False,
            }
        difference = left.float() - right.float()
        maximum = max(maximum, float(difference.abs().max().item()))
        difference_sq += float(difference.square().sum().item())
        reference_sq += float(left.float().square().sum().item())
        identical &= torch.equal(left, right)
    return {
        "reference_forwards": len(reference),
        "candidate_forwards": len(candidate),
        "forward_count_match": len(reference) == len(candidate),
        "shape_match": True,
        "max_absolute_difference": maximum,
        "relative_l2_error": math.sqrt(difference_sq / max(reference_sq, 1.0e-24)),
        "bitwise_identical": bool(identical),
        "compared_forwards": min(len(reference), len(candidate)),
        "final_max_absolute_difference": (
            float((reference[-1].float() - candidate[-1].float()).abs().max().item())
            if reference[-1].shape == candidate[-1].shape else float("inf")
        ),
    }


def _official_accuracy(rows: list[dict[str, Any]], ruler_root: Path) -> dict[str, Any]:
    score_rows = [
        {
            "prediction": row["text"],
            "outputs": row["outputs"],
            "task": row["task"],
            "task_base": row["task_base"],
        }
        for row in rows
    ]
    per_task, overall = score_predictions(score_rows, ruler_root)
    return {"official_ruler_accuracy": overall, "per_task_accuracy": per_task}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = verify_checkout(args.ruler_root)
    prompts = _load_prompts(args.prompts_jsonl, args.num_samples, balanced_by_task=args.balanced_by_task)
    adapter = create_adapter(
        "diffusion_gemma",
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    layers = _indices(args.layers)
    replay_common = {
        "beta": args.beta,
        "threshold_mode": args.threshold_mode,
        "target_density": args.target_density,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "physical_q_tile_size": args.physical_q_tile_size,
        "layers": layers,
    }
    selected_proxies = ("mean", "max") if args.proxies == "both" else (args.proxies,)
    variants = {}
    if args.eager_dense:
        variants["eager_dense"] = ReplayAttention(ReplayConfig(mode="dense_eager", layers=layers))
    for proxy in selected_proxies:
        variants[f"fresh_{proxy}"] = ReplayAttention(
            ReplayConfig(mode="fresh", proxy=proxy, **replay_common)
        )
        variants[f"previous_{proxy}"] = ReplayAttention(
            ReplayConfig(mode="previous", proxy=proxy, **replay_common)
        )
    per_request = []
    prediction_rows: dict[str, list[dict[str, Any]]] = {"native_dense": []}
    prediction_rows.update({name: [] for name in variants})
    try:
        for index, prompt in enumerate(prompts):
            seed = int(prompt.get("inference_seed", args.base_seed + index))
            request = _request(args, prompt, seed)
            baseline_result, baseline_capture = _generate_capture(adapter, request)
            common = {
                "request_id": prompt["request_id"],
                "sample_id": prompt.get("sample_id"),
                "task": prompt.get("task"),
                "task_base": prompt.get("task_base"),
                "outputs": prompt.get("outputs"),
                "seed": seed,
            }
            prediction_rows["native_dense"].append({**common, "text": baseline_result.text})
            request_row = {
                **common,
                "native_dense": {
                    "text": baseline_result.text,
                    "completion_tokens": baseline_result.completion_tokens,
                    "model_evaluations": baseline_result.model_evaluations,
                },
                "variants": {},
            }
            for name, replay in variants.items():
                replay.begin_request(str(prompt["request_id"]))
                with install_attention_replay(adapter.model, replay):
                    result, capture = _generate_capture(adapter, request)
                request_row["variants"][name] = {
                    "text": result.text,
                    "completion_tokens": result.completion_tokens,
                    "model_evaluations": result.model_evaluations,
                    "tokens_identical": result.completion_tokens == baseline_result.completion_tokens,
                    "text_identical": result.text == baseline_result.text,
                    "logits": _tensor_comparison(baseline_capture.logits.values, capture.logits.values),
                    "hidden_states": _tensor_comparison(baseline_capture.hidden, capture.hidden),
                }
                prediction_rows[name].append({**common, "text": result.text})
                del capture
            per_request.append(request_row)
            for proxy in selected_proxies:
                fresh = request_row["variants"][f"fresh_{proxy}"]
                previous = request_row["variants"][f"previous_{proxy}"]
                previous["tokens_identical_to_fresh"] = (
                    previous["completion_tokens"] == fresh["completion_tokens"]
                )
                previous["text_identical_to_fresh"] = previous["text"] == fresh["text"]
            del baseline_capture
            print(f"completed replay sample {index + 1}/{len(prompts)}: {prompt.get('sample_id', prompt['request_id'])}", flush=True)
    finally:
        pass

    summary = {
        "schema_version": 1,
        "phase_gate": "previous_step_replay GO in observation decisions.json",
        "model": args.model_path,
        "revision": args.revision,
        "num_samples": len(prompts),
        "beta": args.beta,
        "threshold_mode": args.threshold_mode,
        "target_density": args.target_density,
        "proxies": list(selected_proxies),
        "eager_dense": args.eager_dense,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "physical_q_tile_size": args.physical_q_tile_size,
        "layers": list(layers),
        "ruler": provenance,
        "variants": {},
    }
    for name, rows in prediction_rows.items():
        comparisons = [row["variants"][name] for row in per_request] if name != "native_dense" else []
        summary["variants"][name] = {
            **_official_accuracy(rows, args.ruler_root),
            "token_agreement": (
                sum(value["tokens_identical"] for value in comparisons) / len(comparisons)
                if comparisons else 1.0
            ),
            "text_agreement": (
                sum(value["text_identical"] for value in comparisons) / len(comparisons)
                if comparisons else 1.0
            ),
            "max_logit_difference": max((value["logits"]["max_absolute_difference"] for value in comparisons), default=0.0),
            "mean_logit_relative_l2_error": (
                sum(value["logits"]["relative_l2_error"] for value in comparisons) / len(comparisons)
                if comparisons else 0.0
            ),
            "max_hidden_difference": max((value["hidden_states"]["max_absolute_difference"] for value in comparisons), default=0.0),
            "mean_hidden_relative_l2_error": (
                sum(value["hidden_states"]["relative_l2_error"] for value in comparisons) / len(comparisons)
                if comparisons else 0.0
            ),
            "attention": variants[name].stats.summary() if name in variants else None,
            "token_agreement_with_fresh": (
                sum(
                    row["variants"][name].get("tokens_identical_to_fresh", False)
                    for row in per_request
                ) / len(per_request)
                if name.startswith("previous_") else None
            ),
            "text_agreement_with_fresh": (
                sum(
                    row["variants"][name].get("text_identical_to_fresh", False)
                    for row in per_request
                ) / len(per_request)
                if name.startswith("previous_") else None
            ),
        }
    (args.output_dir / "per_request.json").write_text(
        json.dumps(per_request, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
