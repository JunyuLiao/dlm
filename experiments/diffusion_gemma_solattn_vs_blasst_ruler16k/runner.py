"""Resumable execution helpers for the canonical nine-condition sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dllm.evaluation.ruler.io import (
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from dllm.attention.blasst import (
    BLASST_MASK_SEMANTICS, require_blasst_mask_semantics, validate_blasst_output_directory,
)

from .calibration import DenseMarginTraceCollector, fit_blasst_policy_jsonl
from .config import (
    DEFAULT_MODEL_REVISION,
    ExperimentConfig,
    TARGET_SPARSITIES,
    canonical_conditions,
)


def _study_rows(manifest_path: str | Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry = manifest[split]
    rows_path = Path(entry["path"])
    if not rows_path.is_absolute():
        rows_path = path.parent / rows_path
    if sha256_file(rows_path) != entry["sha256"]:
        raise ValueError(f"{split} manifest hash mismatch")
    return manifest, read_jsonl(rows_path)


def _runner_manifest(
    study_manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    path: Path,
    *,
    adapter: str,
) -> Path:
    """Write the small adapter manifest expected by ``dllm``'s RULER runner."""

    samples = []
    for row in rows:
        item = dict(row)
        item.update(
            actual_prompt_length=int(item.get("token_count", item.get("actual_prompt_length", 0))),
            tokens_to_generate=int(item.get("generation_budget", item.get("tokens_to_generate", 1))),
            inference_seed=int(item.get("seed", item.get("inference_seed", 0))),
            prompt_sha256=str(item.get("prompt_hash", item.get("prompt_sha256", ""))),
            task_base=str(item.get("task_base", "niah" if str(item.get("task", "")).startswith("niah") else str(item.get("task", "")))),
            sample_id=str(item["sample_id"]),
        )
        samples.append(item)
    samples_path = path.with_name(path.stem + "_samples.jsonl")
    write_jsonl(samples_path, samples)
    payload = {
        "schema_version": 1,
        "ruler": study_manifest["ruler"],
        "model_adapter": adapter,
        "prompt_configuration": study_manifest.get("prompt_configuration", {"thinking": False}),
        "context_length": int(study_manifest["context_length"]),
        "requested_num_samples": len(samples),
        "actual_num_samples": len(samples),
        "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
    }
    write_json(path, payload)
    return path


def _policy_entry(policy: Mapping[str, Any], attention_type: str, target: float) -> dict[str, Any]:
    """Return the exact fitted policy row for one target/type, if present."""

    rows = policy.get(attention_type)
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, Mapping) and abs(float(row.get("target_sparsity", -1.0)) - float(target)) < 1.0e-12:
            return dict(row)
    return {}


def _threshold_provenance(
    condition: ExperimentConfig,
    threshold_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Make threshold origin and attainability auditable in every shard."""

    if condition.method == "dense":
        return {"kind": "dense", "target_sparsity": 0.0}
    if condition.method == "sol_gaussian":
        return {
            "kind": "analytic_gaussian",
            "target_sparsity": float(condition.target_sparsity),
            "beta": float(condition.beta_used),
            "formula": "Phi^-1(target skipped-tile sparsity)",
            "calibration_used": False,
        }
    if threshold_policy is None:
        return {
            "kind": "calibrated_blasst",
            "target_sparsity": float(condition.target_sparsity),
            "calibration_used": False,
            "error": "missing policy",
        }
    local = _policy_entry(threshold_policy, "local", condition.target_sparsity)
    global_ = _policy_entry(threshold_policy, "global", condition.target_sparsity)
    return {
        "kind": "calibrated_blasst",
        "target_sparsity": float(condition.target_sparsity),
        "lambda_local": float(condition.lambda_local),
        "lambda_global": float(condition.lambda_global),
        "metric": threshold_policy.get("metric"),
        "relation": threshold_policy.get("relation"),
        "policy_schema_version": threshold_policy.get("schema_version"),
        "policy_path": threshold_policy.get("policy_path"),
        "policy_sha256": threshold_policy.get("policy_sha256"),
        "local_fit": local,
        "global_fit": global_,
        "unattainable_within_calibration_range": bool(
            local.get("unattainable_within_calibration_range", False)
            or global_.get("unattainable_within_calibration_range", False)
        ),
        "calibration_used": True,
    }


def _write_policy(path: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Write a policy with a stable content digest that excludes the digest field."""

    output = dict(policy)
    if Path(path).exists():
        require_blasst_mask_semantics(json.loads(Path(path).read_text()), path)
    output["blasst_mask_semantics"] = BLASST_MASK_SEMANTICS
    output.pop("policy_sha256", None)
    output["policy_sha256"] = sha256_json(output)
    write_json(path, output)
    return output


def run_one_condition(
    *,
    study_manifest_path: str | Path,
    adapter: str,
    model_path: str,
    model_revision: str | None,
    output_dir: str | Path,
    condition: ExperimentConfig,
    split: str = "final",
    ruler_root: str | Path,
    threshold_policy: Mapping[str, Any] | None = None,
    device: str = "cuda",
    precision: str = "bfloat16",
    temperature: float = 0.0,
    generation_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    study_manifest, rows = _study_rows(study_manifest_path, split)
    if adapter == "diffusion_gemma" and model_revision is None:
        model_revision = DEFAULT_MODEL_REVISION
    output = Path(output_dir).resolve()
    if condition.method == "blasst_calibrated":
        validate_blasst_output_directory(output)
        # Legacy physical-tile calibration used the same decision rule and
        # is reusable. Legacy row-element calibration is not interchangeable.
        if threshold_policy is not None and threshold_policy.get("metric") != "physical":
            require_blasst_mask_semantics(threshold_policy, "BLASST threshold policy")
    output.mkdir(parents=True, exist_ok=True)
    runner_manifest = _runner_manifest(study_manifest, rows, output / "runner_manifest.json", adapter=adapter)
    if condition.method == "dense":
        backend = "dense"
        routing_kwargs: dict[str, Any] = {}
    elif condition.method == "sol_gaussian":
        backend = "fresh-routing"
        routing_kwargs = {
            "routing_mode": "gaussian",
            "routing_density": condition.retained_density,
            "routing_region": "all",
            "routing_q_block_size": 64,
            "routing_kv_block_size": 64,
            "routing_combined_region_population": True,
        }
    else:
        backend = "blasst-reference"
        if threshold_policy is None:
            raise ValueError("BLASST condition requires calibrated threshold policy")
        local = threshold_policy["lambda_local"].get(str(condition.target_sparsity), threshold_policy["lambda_local"].get(condition.target_sparsity))
        global_ = threshold_policy["lambda_global"].get(str(condition.target_sparsity), threshold_policy["lambda_global"].get(condition.target_sparsity))
        if local is None or global_ is None:
            raise KeyError(f"threshold policy has no target {condition.target_sparsity}")
        routing_kwargs = {
            "blasst_policy": {"local_blasst_lambda": float(local), "global_blasst_lambda": float(global_)},
            "collect_attention_stats": True,
            "stats_level": "head",
            "q_tile_size": 64,
            "kv_tile_size": 64,
        }
    config = RulerRunConfig(
        model_adapter=adapter,
        model_path=model_path,
        revision=model_revision,
        manifest_path=str(runner_manifest),
        ruler_root=str(ruler_root),
        output_dir=str(output),
        num_samples=len(rows),
        context_length=int(study_manifest["context_length"]),
        attention_backend=backend,
        temperature=float(temperature),
        device=device,
        precision=precision,
        generation_extra=dict(generation_extra or {}),
        **routing_kwargs,
    )
    result = run_evaluation(config)
    provenance = _threshold_provenance(condition, threshold_policy)
    condition_record = condition.to_dict()
    condition_record["threshold_provenance"] = provenance
    result["condition"] = condition_record
    result["threshold_provenance"] = provenance
    result["study_split"] = split
    write_json(output / "summary.json", result)
    return result


def run_sweep(
    *,
    study_manifest_path: str | Path,
    adapter: str,
    model_path: str,
    model_revision: str | None,
    output_dir: str | Path,
    ruler_root: str | Path,
    threshold_policy: Mapping[str, Any],
    conditions: list[ExperimentConfig] | None = None,
    device: str = "cuda",
    precision: str = "bfloat16",
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    policy_path = output / "blasst_policy.json"
    if policy_path.exists():
        require_blasst_mask_semantics(json.loads(policy_path.read_text()), policy_path)
    output.mkdir(parents=True, exist_ok=True)
    conditions = conditions or canonical_conditions(
        local_lambdas={float(key): value for key, value in threshold_policy["lambda_local"].items()},
        global_lambdas={float(key): value for key, value in threshold_policy["lambda_global"].items()},
    )
    results = []
    for condition in conditions:
        results.append(run_one_condition(
            study_manifest_path=study_manifest_path,
            adapter=adapter,
            model_path=model_path,
            model_revision=model_revision,
            output_dir=output / condition.name,
            condition=condition,
            ruler_root=ruler_root,
            threshold_policy=threshold_policy,
            device=device,
            precision=precision,
        ))
    summary = {
        "schema_version": 1,
        "study": "diffusion_gemma_solattn_vs_blasst_ruler16k",
        "manifest_path": str(study_manifest_path),
        "adapter": adapter,
        "conditions": results,
    }
    write_json(output / "summary.json", summary)
    return summary


def calibrate_from_traces(trace_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise ValueError(f"no calibration traces found at {trace_path}")
    policy = fit_blasst_policy_jsonl(trace_path)
    policy["trace_path"] = str(Path(trace_path).resolve())
    output_path = Path(output_path).resolve()
    policy["policy_path"] = str(output_path)
    policy = _write_policy(output_path, policy)
    return policy


def run_calibration_pass(
    *,
    study_manifest_path: str | Path,
    adapter: str,
    model_path: str,
    model_revision: str | None,
    output_dir: str | Path,
    ruler_root: str | Path,
    device: str = "cuda",
    precision: str = "bfloat16",
) -> dict[str, Any]:
    """Run one dense calibration pass and fit/export the BLASST policy."""

    study_manifest, rows = _study_rows(study_manifest_path, "calibration")
    if adapter == "diffusion_gemma" and model_revision is None:
        model_revision = DEFAULT_MODEL_REVISION
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    runner_manifest = _runner_manifest(study_manifest, rows, output / "runner_manifest.json", adapter=adapter)
    trace_path = output / "margin_traces.jsonl"
    # A completed dense pass may have serialized its trace before being
    # interrupted during policy fitting.  Resume directly from that trace
    # rather than loading the 26B checkpoint and replaying calibration.
    if trace_path.exists() and trace_path.stat().st_size > 0:
        policy = fit_blasst_policy_jsonl(trace_path)
        policy["trace_path"] = str(trace_path)
        prior_summary = output / "summary.json"
        if prior_summary.exists():
            try:
                policy["calibration_summary"] = json.loads(prior_summary.read_text(encoding="utf-8"))
            except Exception:
                pass
        policy_path = output / "blasst_policy.json"
        policy["policy_path"] = str(policy_path.resolve())
        return _write_policy(policy_path, policy)
    collector = DenseMarginTraceCollector(q_tile_size=64, kv_tile_size=64)
    config = RulerRunConfig(
        model_adapter=adapter,
        model_path=model_path,
        revision=model_revision,
        manifest_path=str(runner_manifest),
        ruler_root=str(ruler_root),
        output_dir=str(output),
        num_samples=len(rows),
        context_length=int(study_manifest["context_length"]),
        attention_backend="eager-dense",
        collect_attention_stats=False,
        device=device,
        precision=precision,
    )
    summary = run_evaluation(config, attention_observer=collector)
    prior_traces = read_jsonl(trace_path)
    traces = prior_traces + collector.to_jsonable()
    # A crashed/restarted pass may leave a trace shard beside an incomplete
    # prediction shard.  Preserve only one copy of each deterministic trace.
    unique: dict[str, dict[str, Any]] = {}
    for trace in traces:
        identity = json.dumps(
            {
                key: trace.get(key)
                for key in (
                    "example_id", "batch", "head", "query_block", "attention_type",
                    "valid_kv_length", "margins", "valid_rows", "eligible_tiles",
                )
            },
            sort_keys=True,
            default=str,
        )
        unique.setdefault(identity, trace)
    traces = list(unique.values())
    write_jsonl(trace_path, traces)
    # Stream the multi-gigabyte trace file for fitting; do not materialize all
    # margins in host memory after serialization.
    policy = fit_blasst_policy_jsonl(trace_path)
    policy["trace_path"] = str(trace_path)
    policy["calibration_summary"] = summary
    policy_path = output / "blasst_policy.json"
    policy["policy_path"] = str(policy_path.resolve())
    policy = _write_policy(policy_path, policy)
    return policy


__all__ = [
    "calibrate_from_traces",
    "run_calibration_pass",
    "run_one_condition",
    "run_sweep",
]
