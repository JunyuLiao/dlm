from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from dllm.attention.blasst import Blasst2DConfig, Blasst2DStats, install_blasst
from dllm.models import GenerationRequest, create_adapter

from .io import append_jsonl, read_jsonl, sha256_file, sha256_json, write_json
from .official import RULER_COMMIT, score_predictions


@dataclass(frozen=True)
class RulerRunConfig:
    model_adapter: str
    model_path: str
    manifest_path: str
    ruler_root: str
    output_dir: str
    num_samples: int
    context_length: int
    attention_backend: str = "dense"
    blasst_lambda: float = 0.003
    q_tile_size: int = 128
    kv_tile_size: int = 64
    block_size: int = 16
    max_new_tokens: int | None = None
    steps: int | None = None
    threshold: float = 0.9
    temperature: float = 0.0
    device: str = "cuda"
    precision: str = "bfloat16"
    revision: str | None = None
    collect_attention_stats: bool = False
    stats_level: str = "summary"
    resume: bool = True
    generation_extra: dict[str, Any] = field(default_factory=dict)
    verify_dense_after_blasst: bool = False


def _load_manifest(config: RulerRunConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(config.manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("ruler", {}).get("commit") != RULER_COMMIT:
        raise ValueError("manifest was not produced from the pinned RULER commit")
    if int(manifest["context_length"]) != config.context_length:
        raise ValueError("manifest context length does not match this run")
    sample_entry = manifest["samples"]
    samples_path = Path(sample_entry["path"])
    if not samples_path.is_absolute():
        samples_path = path.parent / samples_path
    if sha256_file(samples_path) != sample_entry["sha256"]:
        raise ValueError("RULER sample manifest hash mismatch")
    rows = read_jsonl(samples_path)
    manifest_requested = int(manifest["requested_num_samples"])
    manifest_actual = int(manifest["actual_num_samples"])
    if manifest_requested != config.num_samples or manifest_actual != config.num_samples:
        raise ValueError(
            "RULER manifest count does not match --num-samples: "
            f"requested={manifest_requested}, actual={manifest_actual}, "
            f"run={config.num_samples}"
        )
    if len(rows) != config.num_samples:
        raise ValueError(
            f"manifest has {len(rows)} records, but exactly {config.num_samples} are required"
        )
    return manifest, rows


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    environment = {
        "repository_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import transformers

        environment["transformers"] = transformers.__version__
    except ImportError:
        environment["transformers"] = "not-installed"
    return environment


def run_evaluation(config: RulerRunConfig) -> dict[str, Any]:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if config.attention_backend not in ("dense", "blasst-reference"):
        raise ValueError("attention_backend must be dense or blasst-reference")
    if config.stats_level not in ("summary", "step", "layer", "head"):
        raise ValueError("stats_level must be summary, step, layer, or head")
    manifest, samples = _load_manifest(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        **asdict(config),
        "output_dir": None,
        "manifest_sha256": sha256_file(config.manifest_path),
    }
    fingerprint = sha256_json(fingerprint_payload)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        prior = json.loads(run_config_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("output directory belongs to a different RULER run")
    write_json(
        run_config_path,
        {**asdict(config), **_environment(), "fingerprint": fingerprint},
    )

    predictions_path = output_dir / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if config.resume else []
    if not config.resume and predictions_path.exists():
        predictions_path.unlink()
    completed = {str(row["sample_id"]): row for row in existing}

    adapter = create_adapter(
        config.model_adapter,
        config.model_path,
        device=config.device,
        precision=config.precision,
        revision=config.revision,
    ).load()
    if config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    current_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    write_json(run_config_path, {**current_run_config, **adapter.runtime_metadata()})
    manifest_adapter = manifest.get("model_adapter")
    if manifest_adapter and manifest_adapter != config.model_adapter:
        raise ValueError(
            f"manifest adapter {manifest_adapter} does not match {config.model_adapter}"
        )
    expected_prompt_configuration = manifest.get("prompt_configuration")
    runtime_prompt_configuration = adapter.prompt_configuration(
        config.generation_extra
    )
    if (
        expected_prompt_configuration is not None
        and expected_prompt_configuration != runtime_prompt_configuration
    ):
        raise ValueError(
            "manifest prompt configuration does not match generation overrides: "
            f"manifest={expected_prompt_configuration}, runtime={runtime_prompt_configuration}"
        )
    for sample in samples:
        actual = len(
            adapter.encode_prompt(str(sample["prompt"]), config.generation_extra)
        )
        if actual != int(sample["actual_prompt_length"]):
            raise ValueError(
                f"tokenizer/prompt mismatch for {sample['sample_id']}: "
                f"manifest={sample['actual_prompt_length']}, runtime={actual}"
            )
    stats = Blasst2DStats(
        record_layers=config.stats_level in ("layer", "head"),
        record_heads=config.stats_level == "head",
    )
    binding = None
    if config.attention_backend == "blasst-reference":
        binding = install_blasst(
            adapter.model,
            Blasst2DConfig(
                enable_blasst_2d=True,
                blasst_lambda=config.blasst_lambda,
                q_tile_size=config.q_tile_size,
                kv_tile_size=config.kv_tile_size,
                collect_blasst_stats=config.collect_attention_stats,
                collect_blasst_layer_stats=config.stats_level in ("layer", "head"),
                collect_blasst_head_stats=config.stats_level == "head",
            ),
            stats,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            attention_class_names=adapter.attention_class_names,
            module_selector=adapter.is_blasst_attention_module,
            query_ids_extractor=adapter.blasst_query_ids,
            filter_special_query_ids=adapter.blasst_filter_special_query_ids,
            dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
            integration=adapter.attention_integration,
        )
    try:
        for index, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            if binding is not None:
                binding.runtime.metadata_context = {
                    "benchmark": str(sample["task"]),
                    "example_id": sample_id,
                    "inference_seed": int(sample["inference_seed"]),
                }
            budget = config.max_new_tokens or int(sample["tokens_to_generate"])
            result = adapter.generate(
                GenerationRequest(
                    prompt=str(sample["prompt"]),
                    max_new_tokens=budget,
                    block_size=config.block_size,
                    steps=config.steps,
                    threshold=config.threshold,
                    temperature=config.temperature,
                    seed=int(sample["inference_seed"]),
                    extra=config.generation_extra,
                )
            )
            if not math.isfinite(result.elapsed_seconds):
                raise RuntimeError(f"non-finite runtime for RULER sample {sample_id}")
            row = {
                **sample,
                "prediction": result.text,
                "completion_tokens": result.completion_tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "model_evaluations": result.model_evaluations,
                "termination_reason": result.termination_reason,
                "generation_metadata": result.metadata,
            }
            append_jsonl(predictions_path, row)
            completed[sample_id] = row
            print(f"completed RULER sample {index}/{len(samples)}: {sample_id}", flush=True)
    finally:
        if binding is not None:
            binding.close()

    post_cleanup_dense_verified = False
    if binding is not None and config.verify_dense_after_blasst:
        sample = samples[0]
        budget = config.max_new_tokens or int(sample["tokens_to_generate"])
        cleanup_result = adapter.generate(
            GenerationRequest(
                prompt=str(sample["prompt"]),
                max_new_tokens=budget,
                block_size=config.block_size,
                steps=config.steps,
                threshold=config.threshold,
                temperature=config.temperature,
                seed=int(sample["inference_seed"]),
                extra=config.generation_extra,
            )
        )
        if not cleanup_result.completion_tokens:
            raise RuntimeError("post-BLASST dense cleanup check produced no tokens")
        if not math.isfinite(cleanup_result.elapsed_seconds):
            raise RuntimeError("post-BLASST dense cleanup check had non-finite runtime")
        post_cleanup_dense_verified = True

    ordered = [completed[str(sample["sample_id"])] for sample in samples]
    if len(ordered) != config.num_samples:
        raise AssertionError(
            f"evaluated {len(ordered)} records, expected exactly {config.num_samples}"
        )
    per_task, overall = score_predictions(ordered, config.ruler_root)
    summary = {
        "schema_version": 1,
        "requested_num_samples": config.num_samples,
        "actual_num_samples": len(ordered),
        "context_length": config.context_length,
        "model_adapter": config.model_adapter,
        "attention_backend": config.attention_backend,
        "official_ruler_accuracy": overall,
        "per_task_accuracy": per_task,
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in ordered),
        "manifest_requested_num_samples": manifest["requested_num_samples"],
        "all_completions_nonempty": all(
            bool(row["completion_tokens"]) for row in ordered
        ),
        "post_blasst_dense_verified": post_cleanup_dense_verified,
    }
    if config.device.startswith("cuda") and torch.cuda.is_available():
        summary["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        parameter_devices = sorted({str(parameter.device) for parameter in adapter.model.parameters()})
        summary["model_parameter_devices"] = parameter_devices
        summary["full_checkpoint_on_cuda"] = bool(parameter_devices) and all(
            value.startswith("cuda") for value in parameter_devices
        )
    if config.collect_attention_stats:
        summary["attention_sparsity"] = stats.summary()
        stats.export(
            output_dir / "attention_stats",
            binding.runtime.config if binding is not None else Blasst2DConfig(),
            {"run_fingerprint": fingerprint},
        )
    write_json(output_dir / "summary.json", summary)
    return summary
