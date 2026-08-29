"""Paired 50-problem symbolic evaluation of ten fresh-routing conditions."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import GenerationRequest, create_adapter

from .collect import DEFAULT_DIFFUSION_GEMMA_REVISION
from .datasets import load_math500
from .routing import FreshRoutingAttention, FreshRoutingConfig, routing_condition_name


DENSITIES = (0.25, 0.50, 0.75)


def build_math_verifier():
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def symbolic_verify(verifier: Any, expected_answer: str, generation: str) -> tuple[float, str | None]:
    expected = expected_answer.strip()
    if expected.startswith("\\(") and expected.endswith("\\)"):
        expected = expected[2:-2].strip()
    if expected.startswith("$") and expected.endswith("$"):
        expected = expected[1:-1].strip()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            score, extracted = verifier([f"\\boxed{{{expected}}}"], [generation])
        answer = None
        if extracted is not None:
            gold, predictions = extracted
            for prediction in predictions:
                if any(grader.verify(target, prediction) for target in gold):
                    answer = str(prediction)
                    break
            if answer is None and predictions:
                answer = str(predictions[0])
        return float(score), answer
    except (Exception, TimeoutException):
        return 0.0, None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def condition_grid() -> list[tuple[str, float | None]]:
    return [("dense", None)] + [
        (mode, density) for mode in ("gaussian", "profiled", "oracle") for density in DENSITIES
    ]


def paired_bootstrap_delta(
    dense: np.ndarray, sparse: np.ndarray, *, seed: int, repeats: int
) -> tuple[float, float, float]:
    differences = np.asarray(sparse, dtype=float) - np.asarray(dense, dtype=float)
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(repeats, len(differences)))].mean(axis=1)
    return float(differences.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_math500(root: Path, *, bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / "conditions").glob("*/predictions.jsonl")):
        condition_rows[path.parent.name] = sorted(_read_jsonl(path), key=lambda row: int(row["subset_index"]))
    dense_rows = condition_rows.get("dense", [])
    if len(dense_rows) != 50:
        raise RuntimeError("the dense condition must contain exactly 50 problems")
    dense = np.asarray([row["symbolic_correct"] for row in dense_rows], dtype=float)
    summaries = []
    for name, rows in sorted(condition_rows.items()):
        if len(rows) != 50:
            raise RuntimeError(f"condition {name} has {len(rows)} problems, expected 50")
        scores = np.asarray([row["symbolic_correct"] for row in rows], dtype=float)
        delta, low, high = paired_bootstrap_delta(dense, scores, seed=seed, repeats=bootstrap_repeats)
        n01 = int(np.sum((dense == 0) & (scores == 1)))
        n10 = int(np.sum((dense == 1) & (scores == 0)))
        stats_rows = [row["attention_stats"] for row in rows if row.get("attention_stats")]
        def total_ratio(numerator: str, denominator: str) -> float | None:
            if not stats_rows:
                return None
            bottom = sum(row[denominator] for row in stats_rows)
            return sum(row[numerator] for row in stats_rows) / bottom if bottom else None
        summaries.append({
            "condition": name,
            "symbolic_pass_at_1": float(scores.mean()),
            "paired_delta_vs_dense": delta,
            "paired_delta_ci95_lower": low,
            "paired_delta_ci95_upper": high,
            "mcnemar_dense_wrong_sparse_right": n01,
            "mcnemar_dense_right_sparse_wrong": n10,
            "logical_density": total_ratio("logical_retained_tiles", "logical_candidate_tiles") if stats_rows else 1.0,
            "physical_density": total_ratio("physical_retained_tiles", "physical_candidate_tiles") if stats_rows else 1.0,
            "retained_dense_attention_mass": float(np.mean([row["retained_dense_attention_mass"] for row in stats_rows])) if stats_rows else 1.0,
            "attention_output_relative_error": float(np.mean([row["attention_output_relative_error"] for row in stats_rows])) if stats_rows else 0.0,
            "accuracy_safe_at_minus_5pp": bool(low > -0.05),
        })
    result = {
        "schema_version": 1,
        "equivalence_margin": -0.05,
        "conditions": summaries,
        "all_conditions_share_problem_ids_and_seeds": all(
            [(row["request_id"], row["seed"]) for row in rows]
            == [(row["request_id"], row["seed"]) for row in dense_rows]
            for rows in condition_rows.values()
        ),
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def run_math500(args: Any) -> dict[str, Any]:
    if args.revision not in (None, DEFAULT_DIFFUSION_GEMMA_REVISION):
        raise ValueError("the canonical Math500 sweep requires the pinned DiffusionGemma revision")
    if args.precision != "bfloat16":
        raise ValueError("the canonical Math500 sweep requires BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    problems = load_math500(args.nemo_gym_root, 50, args.split_seed)
    threshold_path = args.threshold_model or args.output_dir / "fit" / "threshold_models.json"
    threshold_model = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    adapter = create_adapter(
        "diffusion_gemma",
        args.model_path or "google/diffusiongemma-26B-A4B-it",
        device=args.device,
        precision=args.precision,
        revision=args.revision or DEFAULT_DIFFUSION_GEMMA_REVISION,
    ).load()
    verifier = build_math_verifier()
    output = args.output_dir / "math500"
    output.mkdir(parents=True, exist_ok=True)
    subset_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in problems)
    subset_path = output / "math500_subset.jsonl"
    if subset_path.exists() and subset_path.read_text(encoding="utf-8") != subset_payload:
        raise RuntimeError("existing Math500 subset differs from the canonical fixed subset")
    subset_path.write_text(subset_payload, encoding="utf-8")
    for mode, density in condition_grid():
        name = "dense" if mode == "dense" else routing_condition_name(mode, float(density))
        predictions = output / "conditions" / name / "predictions.jsonl"
        completed = {str(row["request_id"]): row for row in _read_jsonl(predictions)}
        binding = None
        router = None
        if mode != "dense":
            router = FreshRoutingAttention(FreshRoutingConfig(
                mode=mode,
                target_density=float(density),
                threshold_model=threshold_model if mode == "profiled" else None,
                adapter="diffusion_gemma",
                corpus="math500",
            ))
            binding = install_blasst(
                adapter.model,
                Blasst2DConfig(enable_blasst_2d=True, apply_blasst_mask=False, collect_blasst_stats=False),
                mask_token_id=adapter.mask_token_id,
                pad_token_id=adapter.pad_token_id,
                attention_class_names=adapter.attention_class_names,
                module_selector=adapter.is_blasst_attention_module,
                query_ids_extractor=adapter.blasst_query_ids,
                filter_special_query_ids=adapter.blasst_filter_special_query_ids,
                call_selector=adapter.blasst_call_is_eligible,
                # FreshRoutingAttention derives the concatenated encoder-prefix
                # boundary as KV length - query length. Do not install the
                # legacy DiffusionGemma policy that exposes encoder KV to older
                # BLASST routing experiments.
                dense_kv_prefix_extractor=None,
                integration=adapter.attention_integration,
            )
            binding.runtime.attention_override = router
        try:
            for index, problem in enumerate(problems):
                request_id = str(problem["request_id"])
                if request_id in completed:
                    continue
                seed = args.base_seed + int(problem["dataset_index"])
                if binding is not None:
                    binding.runtime.forward_call_index = 0
                    binding.runtime.current_denoising_iteration = -1
                    binding.runtime.metadata_context = {"request_id": request_id, "inference_seed": seed}
                    router.stats = type(router.stats)()
                result = adapter.generate(GenerationRequest(
                    prompt=problem["prompt"], max_new_tokens=2048, block_size=256,
                    temperature=0.6, seed=seed,
                    extra={"thinking": False, "top_p": 0.95},
                ))
                score, extracted = symbolic_verify(verifier, problem["expected_answer"], result.text)
                row = {
                    "schema_version": 1,
                    "condition": name,
                    "request_id": request_id,
                    "subset_index": int(problem["subset_index"]),
                    "dataset_index": int(problem["dataset_index"]),
                    "seed": seed,
                    "prompt": problem["prompt"],
                    "expected_answer": problem["expected_answer"],
                    "generation": result.text,
                    "completion_tokens": result.completion_tokens,
                    "symbolic_correct": bool(score > 0.5),
                    "extracted_answer": extracted,
                    "attention_stats": router.stats.summary() if router is not None else None,
                }
                _append_jsonl(predictions, row)
        finally:
            if binding is not None:
                binding.close()
    return summarize_math500(output, bootstrap_repeats=args.bootstrap_repeats, seed=args.split_seed)
