"""Resumable fresh logical tile-routing sweep on one fixed RULER manifest."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dllm.evaluation.ruler.io import read_jsonl, write_json
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from dllm.models import create_adapter


DENSITIES = (0.75, 0.50, 0.25, 0.10)


def condition_grid(*, include_prefix: bool = True, include_profiled: bool = False) -> list[dict[str, Any]]:
    conditions = [{"name": "dense", "backend": "dense"}]
    for region in (("all", "prefix_only") if include_prefix else ("all",)):
        suffix = "" if region == "all" else "_prefix_only"
        modes = ["gaussian", "oracle", "random"]
        if include_profiled:
            modes.insert(1, "profiled")
        for mode in modes:
            for density in DENSITIES:
                conditions.append({
                    "name": f"{mode}_rho{int(round(density * 100)):02d}{suffix}",
                    "backend": "fresh-routing",
                    "mode": mode,
                    "density": density,
                    "region": region,
                })
    return conditions


def _routing_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    values = [row.get("routing_stats") for row in rows if row.get("routing_stats")]
    if not values:
        return None
    def ratio(numerator: str, denominator: str) -> float | None:
        n = sum(float(item.get(numerator, 0.0)) for item in values)
        d = sum(float(item.get(denominator, 0.0)) for item in values)
        return n / d if d else None
    region_counts: dict[str, dict[str, int]] = {}
    grouped_counts = {name: {} for name in ("by_attention_type", "by_layer", "by_head", "by_step")}
    retained_histogram: dict[str, int] = {}
    overlap_intersection = overlap_union = overlap_comparisons = 0
    for item in values:
        for region, counts in item.get("region_counts", {}).items():
            target = region_counts.setdefault(region, {"candidate_tiles": 0, "retained_tiles": 0})
            target["candidate_tiles"] += int(counts["candidate_tiles"])
            target["retained_tiles"] += int(counts["retained_tiles"])
        for group_name, target_groups in grouped_counts.items():
            for key, counts in item.get(group_name, {}).items():
                target = target_groups.setdefault(key, {})
                for name, value in counts.items():
                    target[name] = target.get(name, 0) + int(value)
        for retained, count in item.get("retained_tiles_per_row_histogram", {}).items():
            retained_histogram[retained] = retained_histogram.get(retained, 0) + int(count)
        overlap_intersection += int(item.get("mask_overlap_intersection", 0))
        overlap_union += int(item.get("mask_overlap_union", 0))
        overlap_comparisons += int(item.get("mask_overlap_comparisons", 0))
    routing_rows = sum(int(item.get("routing_rows", 0)) for item in values)
    fallback_rows = sum(int(item.get("fallback_rows", 0)) for item in values)
    degenerate_rows = sum(int(item.get("degenerate_rows", 0)) for item in values)
    return {
        "samples": len(values),
        "calls": sum(int(item["calls"]) for item in values),
        "logical_density": ratio("logical_retained_tiles", "logical_candidate_tiles"),
        "physical_density": ratio("physical_retained_tiles", "physical_candidate_tiles"),
        "retained_dense_attention_mass": (
            sum(float(item["retained_dense_attention_mass"]) for item in values if item.get("retained_dense_attention_mass") is not None)
            / sum(item.get("retained_dense_attention_mass") is not None for item in values)
            if any(item.get("retained_dense_attention_mass") is not None for item in values) else None
        ),
        "attention_output_relative_error": (
            sum(float(item["attention_output_relative_error"]) for item in values if item.get("attention_output_relative_error") is not None)
            / sum(item.get("attention_output_relative_error") is not None for item in values)
            if any(item.get("attention_output_relative_error") is not None for item in values) else None
        ),
        "proxy_mask_seconds": sum(float(item.get("proxy_mask_seconds", 0.0)) for item in values),
        "sparse_attention_seconds": sum(float(item.get("sparse_attention_seconds", 0.0)) for item in values),
        "prefill_attention_calls": sum(int(item.get("prefill_attention_calls", 0)) for item in values),
        "denoising_attention_calls": sum(int(item.get("denoising_attention_calls", 0)) for item in values),
        "prefill_proxy_mask_seconds": sum(float(item.get("prefill_proxy_mask_seconds", 0.0)) for item in values),
        "denoising_proxy_mask_seconds": sum(float(item.get("denoising_proxy_mask_seconds", 0.0)) for item in values),
        "prefill_sparse_attention_seconds": sum(float(item.get("prefill_sparse_attention_seconds", 0.0)) for item in values),
        "denoising_sparse_attention_seconds": sum(float(item.get("denoising_sparse_attention_seconds", 0.0)) for item in values),
        "fallback_rows": fallback_rows,
        "routing_rows": routing_rows,
        "fallback_row_fraction": fallback_rows / routing_rows if routing_rows else None,
        "degenerate_rows": degenerate_rows,
        "degenerate_row_fraction": degenerate_rows / routing_rows if routing_rows else None,
        "region_counts": region_counts,
        "retained_tiles_per_row_histogram": retained_histogram,
        "consecutive_mask_jaccard": overlap_intersection / overlap_union if overlap_union else None,
        "mask_overlap_comparisons": overlap_comparisons,
        **grouped_counts,
    }


def _dense_correct_subset(dense_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], ruler_root: str | Path) -> dict[str, Any]:
    from dllm.evaluation.ruler.official import score_predictions
    dense_correct: set[str] = set()
    for row in dense_rows:
        _, score = score_predictions([row], ruler_root)
        if float(score) >= 1.0:
            dense_correct.add(str(row["sample_id"]))
    restricted = [row for row in candidate_rows if str(row["sample_id"]) in dense_correct]
    _, accuracy = score_predictions(restricted, ruler_root) if restricted else ({}, 0.0)
    return {"dense_correct_examples": len(dense_correct), "accuracy_on_dense_correct_subset": accuracy}


def run_ruler_sweep(args: Any) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    threshold_model = None
    if args.threshold_model:
        threshold_model = json.loads(Path(args.threshold_model).read_text(encoding="utf-8"))
    conditions = condition_grid(
        include_prefix=not args.no_prefix_ablation,
        include_profiled=threshold_model is not None,
    )
    if args.conditions:
        available = {item["name"] for item in conditions}
        unknown = set(args.conditions) - available
        if unknown:
            raise ValueError("unknown routing condition(s): " + ", ".join(sorted(unknown)))
        conditions = [item for item in conditions if item["name"] in set(args.conditions)]
    adapter = create_adapter(
        args.adapter,
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    summaries = []
    for condition in conditions:
        condition_dir = output / condition["name"]
        config = RulerRunConfig(
            model_adapter=args.adapter,
            model_path=args.model_path,
            manifest_path=str(args.manifest_path),
            ruler_root=str(args.ruler_root),
            output_dir=str(condition_dir),
            num_samples=args.num_samples,
            context_length=args.context_length,
            attention_backend=condition["backend"],
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            temperature=args.temperature,
            device=args.device,
            precision=args.precision,
            revision=args.revision,
            generation_extra=json.loads(args.generation_extra) if args.generation_extra else {},
            routing_mode=condition.get("mode"),
            routing_density=condition.get("density"),
            routing_region=condition.get("region", "all"),
            routing_threshold_model=threshold_model if condition.get("mode") == "profiled" else None,
            routing_random_seed=args.random_seed,
            routing_execution=args.routing_execution,
            routing_q_block_size=args.q_block_size,
            routing_kv_block_size=args.kv_block_size,
            performance_warmups=args.performance_warmups,
            performance_repeats=args.performance_repeats,
            progress_every=args.progress_every,
            quiet=args.quiet,
        )
        result = run_evaluation(config, loaded_adapter=adapter)
        predictions = read_jsonl(condition_dir / "predictions.jsonl")
        dense_rows = read_jsonl(output / "dense" / "predictions.jsonl") if condition["name"] != "dense" else predictions
        result["condition"] = condition
        result["routing_aggregate"] = _routing_aggregate(predictions)
        if condition["name"] != "dense":
            result["dense_correct_subset"] = _dense_correct_subset(dense_rows, predictions, args.ruler_root)
        write_json(condition_dir / "summary.json", result)
        summaries.append(result)
    final = {
        "schema_version": 1,
        "manifest_path": str(args.manifest_path),
        "adapter": args.adapter,
        "context_length": args.context_length,
        "num_samples": args.num_samples,
        "routing_execution": args.routing_execution,
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "performance_warmups": args.performance_warmups,
        "performance_repeats": args.performance_repeats,
        "output_dir": str(output),
        "conditions": summaries,
    }
    write_json(output / "sweep_summary.json", final)
    return final
