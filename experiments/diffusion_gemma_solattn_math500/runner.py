"""Resumable CUDA execution for the Sol-Attn MATH500 comparison."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import GenerationRequest, create_adapter
from experiments.diffusion_attention_threshold_modeling.math500 import (
    build_math_verifier,
    symbolic_verify,
)
from experiments.diffusion_attention_threshold_modeling.routing import (
    FreshRoutingAttention,
    FreshRoutingConfig,
)

from .config import Condition, condition_map, conditions
from .dataset import read_manifest


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _install_router(adapter: Any, condition: Condition, *, density_override: float | None = None):
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="gaussian",
        target_density=condition.retained_density if density_override is None else density_override,
        q_block_size=64,
        kv_block_size=64,
        adapter="diffusion_gemma",
        corpus="math500",
        region=str(condition.region),
        combined_region_population=True,
        execution="logical",
    ))
    binding = install_blasst(
        adapter.model,
        Blasst2DConfig(
            enable_blasst_2d=True,
            apply_blasst_mask=False,
            collect_blasst_stats=False,
            q_tile_size=64,
            kv_tile_size=64,
        ),
        mask_token_id=adapter.mask_token_id,
        pad_token_id=adapter.pad_token_id,
        attention_class_names=adapter.attention_class_names,
        module_selector=adapter.is_blasst_attention_module,
        query_ids_extractor=adapter.blasst_query_ids,
        filter_special_query_ids=adapter.blasst_filter_special_query_ids,
        call_selector=adapter.blasst_call_is_eligible,
        dense_kv_prefix_extractor=None,
        integration=adapter.attention_integration,
    )
    binding.runtime.attention_override = router
    return router, binding


def _install_eager_dense(adapter: Any):
    """Install the same eager backend used by routing, without any mask."""
    return install_blasst(
        adapter.model,
        Blasst2DConfig(
            enable_blasst_2d=True,
            apply_blasst_mask=False,
            collect_blasst_stats=False,
            q_tile_size=64,
            kv_tile_size=64,
        ),
        mask_token_id=adapter.mask_token_id,
        pad_token_id=adapter.pad_token_id,
        attention_class_names=adapter.attention_class_names,
        module_selector=adapter.is_blasst_attention_module,
        query_ids_extractor=adapter.blasst_query_ids,
        filter_special_query_ids=adapter.blasst_filter_special_query_ids,
        call_selector=adapter.blasst_call_is_eligible,
        dense_kv_prefix_extractor=None,
        integration=adapter.attention_integration,
    )


def _request(sample: dict[str, Any], max_new_tokens: int | None = None) -> GenerationRequest:
    return GenerationRequest(
        prompt=str(sample["prompt"]),
        max_new_tokens=max_new_tokens or int(sample["generation_budget"]),
        block_size=256,
        temperature=0.6,
        seed=int(sample["inference_seed"]),
        extra={"thinking": False, "top_p": 0.95},
    )


def _reset_router(router: FreshRoutingAttention, binding: Any, sample: dict[str, Any]) -> None:
    router.stats = type(router.stats)()
    router._previous_retained_tiles.clear()
    binding.runtime.forward_call_index = 0
    binding.runtime.current_denoising_iteration = -1
    binding.runtime.metadata_context = {
        "benchmark": "math500",
        "request_id": str(sample["request_id"]),
        "inference_seed": int(sample["inference_seed"]),
    }


def cuda_smoke(
    manifest_path: Path,
    output_dir: Path,
    *,
    model_path: str,
    revision: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    samples = read_manifest(manifest_path)[:2]
    adapter = create_adapter("diffusion_gemma", model_path, device=device, precision="bfloat16", revision=revision).load()
    records: list[dict[str, Any]] = []
    for sample in samples:
        eager_binding = _install_eager_dense(adapter)
        try:
            eager_binding.runtime.forward_call_index = 0
            eager_binding.runtime.current_denoising_iteration = -1
            dense = adapter.generate(_request(sample, max_new_tokens))
        finally:
            eager_binding.close()
        record = {
            "request_id": sample["request_id"],
            "dense_tokens": dense.completion_tokens,
            "modes": {},
        }
        for name in ("sol_prefix_s25", "sol_prefix_canvas_s25"):
            condition = condition_map()[name]
            router, binding = _install_router(adapter, condition, density_override=1.0)
            try:
                _reset_router(router, binding, sample)
                routed = adapter.generate(_request(sample, max_new_tokens))
                summary = router.stats.summary()
            finally:
                binding.close()
            attention_types = set(summary["by_attention_type"])
            regions = set(summary["all_region_counts"])
            record["modes"][condition.region] = {
                "tokens_identical": routed.completion_tokens == dense.completion_tokens,
                "finite_text": isinstance(routed.text, str),
                "attention_types": sorted(attention_types),
                "regions": sorted(regions),
                "total_tiles": summary["physical_total_tiles"],
                "skipped_tiles": summary["physical_skipped_tiles"],
            }
        records.append(record)
    passed = all(
        mode["tokens_identical"]
        and mode["finite_text"]
        and set(mode["attention_types"]) == {"local", "global"}
        and {"prefix", "canvas"}.issubset(mode["regions"])
        and mode["total_tiles"] > 0
        and mode["skipped_tiles"] == 0
        for record in records for mode in record["modes"].values()
    )
    result = {"schema_version": 1, "passed": passed, "max_new_tokens": max_new_tokens, "samples": records}
    _write_json(output_dir / "smoke" / "audit.json", result)
    if not passed:
        raise RuntimeError("CUDA smoke audit failed")
    return result


def run_sweep(
    manifest_path: Path,
    output_dir: Path,
    *,
    model_path: str,
    revision: str,
    selected_conditions: Iterable[str] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    samples = read_manifest(manifest_path)
    selected = list(selected_conditions or [item.name for item in conditions()])
    unknown = set(selected) - set(condition_map())
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    smoke_path = output_dir / "smoke" / "audit.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("a passing two-example CUDA smoke is required before the sweep")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    adapter = create_adapter("diffusion_gemma", model_path, device=device, precision="bfloat16", revision=revision).load()
    verifier = build_math_verifier()
    statuses = []
    for name in selected:
        condition = condition_map()[name]
        condition_dir = output_dir / "conditions" / name
        predictions_path = condition_dir / "predictions.jsonl"
        run_config = {
            "schema_version": 1,
            "condition": condition.to_dict(),
            "model_path": model_path,
            "revision": revision,
            "precision": "bfloat16",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "num_samples": 50,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 2048,
            "generation_block_size": 256,
            "attention_backend": "matched_eager_reference",
            "q_tile_size": 64,
            "kv_tile_size": 64,
            "sparsity_definition": "sum(skipped physical tiles) / sum(all valid physical tiles)",
        }
        fingerprint = _stable_hash(run_config)
        run_config["run_fingerprint"] = fingerprint
        config_path = condition_dir / "run_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")).get("run_fingerprint") != fingerprint:
                raise RuntimeError(f"{name} output has a different run fingerprint")
        else:
            _write_json(config_path, run_config)
        existing = _read_jsonl(predictions_path)
        completed = {str(row["request_id"]): row for row in existing}
        if len(completed) != len(existing) or any(row.get("run_fingerprint") != fingerprint for row in existing):
            raise RuntimeError(f"{name} prediction resume audit failed")
        router = None
        binding = None
        if condition.region is not None:
            router, binding = _install_router(adapter, condition)
        else:
            binding = _install_eager_dense(adapter)
        try:
            for sample in samples:
                request_id = str(sample["request_id"])
                if request_id in completed:
                    continue
                if router is not None:
                    _reset_router(router, binding, sample)
                else:
                    binding.runtime.forward_call_index = 0
                    binding.runtime.current_denoising_iteration = -1
                    binding.runtime.metadata_context = {
                        "benchmark": "math500",
                        "request_id": request_id,
                        "inference_seed": int(sample["inference_seed"]),
                    }
                wall_start = time.time()
                result = adapter.generate(_request(sample))
                score, extracted = symbolic_verify(verifier, str(sample["expected_answer"]), result.text)
                row = {
                    "schema_version": 1,
                    "run_fingerprint": fingerprint,
                    "condition": name,
                    "request_id": request_id,
                    "subset_index": int(sample["subset_index"]),
                    "dataset_index": int(sample["dataset_index"]),
                    "prompt_hash": sample["prompt_hash"],
                    "seed": int(sample["inference_seed"]),
                    "prompt_tokens": len(result.prompt_tokens),
                    "expected_answer": sample["expected_answer"],
                    "generation": result.text,
                    "completion_tokens": result.completion_tokens,
                    "num_generated_tokens": len(result.completion_tokens),
                    "termination_reason": result.termination_reason,
                    "symbolic_correct": bool(score > 0.5),
                    "extracted_answer": extracted,
                    "elapsed_seconds": float(result.elapsed_seconds),
                    "wall_seconds": time.time() - wall_start,
                    "routing_stats": router.stats.summary() if router is not None else None,
                }
                _append_jsonl(predictions_path, row)
                completed[request_id] = row
                progress = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} condition={name} completed={len(completed)}/50 request={request_id} correct={row['symbolic_correct']} tokens={row['num_generated_tokens']}"
                with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
                    handle.write(progress + "\n")
                print(progress, flush=True)
        finally:
            if binding is not None:
                binding.close()
        statuses.append({"condition": name, "completed": len(completed), "complete": len(completed) == 50})
    status = {"schema_version": 1, "conditions": statuses, "complete": all(row["complete"] for row in statuses)}
    _write_json(output_dir / "run_status.json", status)
    return status
