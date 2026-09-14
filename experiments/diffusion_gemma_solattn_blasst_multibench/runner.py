"""Resumable paired generation and isolated LiveCodeBench scoring."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from dllm.attention.blasst import Blasst2DConfig, Blasst2DStats, install_blasst
from dllm.models import GenerationRequest, create_adapter
from experiments.diffusion_attention_threshold_modeling.routing import FreshRoutingAttention, FreshRoutingConfig
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import aggregate_routing_stats

from .config import Condition, REVISION, condition_map
from .dataset import RULER_ROOT, code_from_prediction, score


POLICY = Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k/calibration/blasst_policy.json")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _policy_for_target(target: float) -> dict[str, dict[str, Any]]:
    policy = json.loads(POLICY.read_text())
    if policy.get("relation") != "lambda * L = alpha * exp(gamma * s)":
        raise RuntimeError("unexpected BLASST calibration policy")
    output: dict[str, dict[str, Any]] = {}
    for attention_type in ("local", "global"):
        row = next(item for item in policy[attention_type] if abs(float(item["target_sparsity"]) - target) < 1e-12)
        relation = row["relation"]
        output[attention_type] = {
            "alpha": float(relation["alpha"]), "gamma": float(relation["gamma"]),
            "target_sparsity": float(target), "unattainable": bool(row.get("unattainable_within_calibration_range", False)),
            "calibrated_lambda_at_mean_length": float(row["lambda"]),
            "calibration_mean_valid_kv_length": float(relation["mean_valid_kv_length"]),
        }
    return output


def _install_dense(adapter: Any):
    return install_blasst(adapter.model, Blasst2DConfig(enable_blasst_2d=True, apply_blasst_mask=False),
        mask_token_id=adapter.mask_token_id, pad_token_id=adapter.pad_token_id,
        attention_class_names=adapter.attention_class_names, module_selector=adapter.is_blasst_attention_module,
        query_ids_extractor=adapter.blasst_query_ids, filter_special_query_ids=adapter.blasst_filter_special_query_ids,
        call_selector=adapter.blasst_call_is_eligible, dense_kv_prefix_extractor=None,
        integration=adapter.attention_integration)


def _install_sol(adapter: Any, condition: Condition, *, density: float | None = None):
    router = FreshRoutingAttention(FreshRoutingConfig(mode="gaussian", target_density=(1.0 - condition.target_sparsity) if density is None else density,
        q_block_size=64, kv_block_size=64, adapter="diffusion_gemma", corpus="ruler16k", region="all",
        combined_region_population=True, execution="logical"))
    binding = _install_dense(adapter)
    binding.runtime.attention_override = router
    return router, binding


def _install_blasst(adapter: Any, condition: Condition):
    length_policy = _policy_for_target(condition.target_sparsity)
    stats = Blasst2DStats(record_layers=True, record_heads=False)
    binding = install_blasst(adapter.model, Blasst2DConfig(enable_blasst_2d=True, apply_blasst_mask=True,
        q_tile_size=64, kv_tile_size=64, collect_blasst_stats=True, collect_blasst_layer_stats=True,
        collect_blasst_head_stats=False, length_aware_policy=length_policy), stats,
        mask_token_id=adapter.mask_token_id, pad_token_id=adapter.pad_token_id,
        attention_class_names=adapter.attention_class_names, module_selector=adapter.is_blasst_attention_module,
        query_ids_extractor=adapter.blasst_query_ids, filter_special_query_ids=adapter.blasst_filter_special_query_ids,
        call_selector=adapter.blasst_call_is_eligible, dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
        integration=adapter.attention_integration)
    return stats, binding, length_policy


def _request(row: dict[str, Any]) -> GenerationRequest:
    return GenerationRequest(prompt=row["prompt"], max_new_tokens=int(row["generation_budget"]), block_size=256,
        temperature=0.0, seed=int(row["seed"]), extra={"thinking": False})


def _set_context(binding: Any, row: dict[str, Any]) -> None:
    binding.runtime.forward_call_index = 0
    binding.runtime.current_denoising_iteration = -1
    binding.runtime.metadata_context = {"benchmark": row["benchmark"], "example_id": row["id"], "inference_seed": int(row["seed"])}


def smoke(manifest_path: Path, output_dir: Path, *, model_path: str, revision: str = REVISION) -> dict[str, Any]:
    """Two prompt exact-parity smoke for unpruned Sol and finite BLASST output."""
    rows = _rows(manifest_path)[:2]
    adapter = create_adapter("diffusion_gemma", model_path, device="cuda", precision="bfloat16", revision=revision).load()
    records = []
    for row in rows:
        dense_binding = _install_dense(adapter)
        try:
            _set_context(dense_binding, row); dense = adapter.generate(_request({**row, "generation_budget": min(128, row["generation_budget"])}))
        finally: dense_binding.close()
        sol = condition_map()["sol_gaussian_s25"]
        router, binding = _install_sol(adapter, sol, density=1.0)
        try:
            _set_context(binding, row); routed = adapter.generate(_request({**row, "generation_budget": min(128, row["generation_budget"])})); stats = router.stats.summary()
        finally: binding.close()
        bstats, bbinding, _ = _install_blasst(adapter, condition_map()["blasst_length_aware_s25"])
        try:
            _set_context(bbinding, row); blasst = adapter.generate(_request({**row, "generation_budget": min(128, row["generation_budget"])})); bsummary = bstats.summary()
        finally: bbinding.close()
        records.append({"id": row["id"], "sol_dense_parity": routed.completion_tokens == dense.completion_tokens,
            "sol_skipped_tiles": stats["physical_skipped_tiles"], "sol_types": sorted(stats["by_attention_type"]),
            "blasst_finite": isinstance(blasst.text, str), "blasst_tiles": bsummary["eligible_tiles"]})
    passed = all(item["sol_dense_parity"] and item["sol_skipped_tiles"] == 0 and set(item["sol_types"]) == {"local", "global"} and item["blasst_finite"] and item["blasst_tiles"] > 0 for item in records)
    result = {"passed": passed, "records": records}
    _write(output_dir / "smoke" / "audit.json", result)
    if not passed: raise RuntimeError("CUDA smoke failed")
    return result


def run(manifest_path: Path, output_dir: Path, *, model_path: str, revision: str = REVISION, selected: Iterable[str] | None = None) -> dict[str, Any]:
    rows = _rows(manifest_path); selected = list(selected or condition_map())
    if not (output_dir / "smoke" / "audit.json").exists() or not json.loads((output_dir / "smoke" / "audit.json").read_text()).get("passed"):
        raise RuntimeError("passing smoke required")
    adapter = create_adapter("diffusion_gemma", model_path, device="cuda", precision="bfloat16", revision=revision).load()
    summaries = []
    for name in selected:
        condition = condition_map()[name]; directory = output_dir / "conditions" / name; predictions = directory / "predictions.jsonl"
        policy = _policy_for_target(condition.target_sparsity) if condition.method == "blasst" else None
        config = {"condition": condition.to_dict(), "model_path": model_path, "revision": revision, "manifest": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "attention_backend": "matched_eager_reference" if condition.method != "blasst" else "blasst_physical_tile_reference",
            "tile_size": 64, "sol_region": "prefix_plus_canvas" if condition.method == "sol" else None,
            "blasst_length_aware_policy": policy, "temperature": 0.0, "top_p": None, "sparsity": "sum(skipped eligible physical tiles) / sum(eligible physical tiles)"}
        fingerprint = _fingerprint(config); config["fingerprint"] = fingerprint
        if (directory / "run_config.json").exists() and json.loads((directory / "run_config.json").read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError(f"incompatible resume directory: {directory}")
        _write(directory / "run_config.json", config)
        existing = _rows(predictions) if predictions.exists() else []; done = {row["id"]: row for row in existing}
        if len(done) != len(existing) or any(row.get("fingerprint") != fingerprint for row in existing): raise RuntimeError("prediction resume audit failed")
        for index, row in enumerate(rows, 1):
            if row["id"] in done: continue
            router = stats = binding = None
            if condition.method == "dense": binding = _install_dense(adapter)
            elif condition.method == "sol": router, binding = _install_sol(adapter, condition)
            else: stats, binding, _ = _install_blasst(adapter, condition)
            try:
                _set_context(binding, row); started = time.time(); generated = adapter.generate(_request(row))
                routing = router.stats.summary() if router is not None else None
                blasst_steps = list(stats.per_step.values()) if stats is not None else None
                blasst_summary = stats.summary() if stats is not None else None
            finally: binding.close()
            correct = score(row, generated.text)
            output = {"fingerprint": fingerprint, "condition": name, **row, "prediction": generated.text,
                "completion_tokens": generated.completion_tokens, "termination_reason": generated.termination_reason,
                "correct": correct, "elapsed_seconds": float(generated.elapsed_seconds), "wall_seconds": time.time() - started,
                "routing_stats": routing, "blasst_steps": blasst_steps, "blasst_summary": blasst_summary,
                "code": code_from_prediction(generated.text) if row["benchmark"] == "livecodebench_v6" else None}
            _append(predictions, output); done[row["id"]] = output
            with (directory / "progress.log").open("a") as handle: handle.write(f"{time.strftime('%FT%TZ', time.gmtime())} {index}/{len(rows)} {row['id']}\n")
        summaries.append({"condition": name, "completed": len(done)})
    _write(output_dir / "run_status.json", {"completed": summaries, "expected_per_condition": len(rows)})
    return {"conditions": summaries}


def grade_livecodebench(output_dir: Path, *, docker_image: str = "dlm-experiment:h100") -> dict[str, Any]:
    """Grade code in a network-disabled, read-only Docker container.

    The official LiveCodeBench package is installed into a throwaway container
    layer; no model output is executed on the host and the container receives
    only the condition-specific evaluation JSONL mount.
    """
    results = {}
    for condition in condition_map():
        directory = output_dir / "conditions" / condition; predictions = _rows(directory / "predictions.jsonl")
        code_rows = [row for row in predictions if row["benchmark"] == "livecodebench_v6"]
        if len(code_rows) != 10: raise RuntimeError(f"{condition}: missing LiveCodeBench rows")
        source = directory / "livecodebench_eval.jsonl"
        if not (directory / "livecodebench_eval_results-saved.json").exists():
            with source.open("w") as handle:
                for row in code_rows:
                    lcb = dict(row["lcb"]); lcb.update(task_id=lcb["question_id"], question_id=lcb["question_id"], completion=row["code"])
                    handle.write(json.dumps(lcb, default=str) + "\n")
            command = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g", "--pids-limit", "128", "--memory", "3g",
                "-v", f"{directory.resolve()}:/work:rw", "-w", "/work", docker_image, "bash", "-lc",
                "python -m pip install --no-cache-dir git+https://github.com/wasiahmad/livecodebench.git && python - <<'PY'\nfrom livecodebench.evaluate import evaluate\nevaluate(custom_output_file='livecodebench_eval.jsonl', release_version='release_v6', k_list=[1], language='python', num_process_evaluate=2, timeout=6)\nPY"]
            subprocess.run(command, check=True, timeout=1800)
            result_path = directory / "livecodebench_eval_results.json"
            if not result_path.exists(): raise RuntimeError(f"{condition}: evaluator did not write results")
            result_path.replace(directory / "livecodebench_eval_results-saved.json")
        grades = json.loads((directory / "livecodebench_eval_results-saved.json").read_text())["eval"]
        by_id = {str(key): bool(value["graded_list"][0]) for key, value in grades.items()}
        updated = []
        for row in predictions:
            if row["benchmark"] == "livecodebench_v6": row["correct"] = by_id[str(row["lcb"]["question_id"])]
            updated.append(row)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in updated))
        results[condition] = {"graded": len(by_id), "accuracy": sum(by_id.values()) / len(by_id)}
    _write(output_dir / "livecodebench_grading.json", results)
    return results
