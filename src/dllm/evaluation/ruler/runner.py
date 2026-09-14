from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from dllm.attention.blasst import Blasst2DConfig, Blasst2DStats, install_blasst
from dllm.attention.blasst import BLASST_MASK_SEMANTICS, validate_blasst_output_directory
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
    include_masked_kv_tiles_in_physical_stats: bool = False
    blasst_policy: dict[str, Any] = field(default_factory=dict)
    apply_blasst_mask: bool | None = None
    blasst_calibration_lambdas: tuple[float, ...] = ()
    stats_level: str = "summary"
    resume: bool = True
    generation_extra: dict[str, Any] = field(default_factory=dict)
    verify_dense_after_blasst: bool = False
    routing_mode: str | None = None
    routing_density: float | None = None
    routing_region: str = "all"
    routing_threshold_model: dict[str, Any] | None = None
    routing_random_seed: int = 20260825
    routing_execution: str = "logical"
    routing_q_block_size: int = 64
    routing_kv_block_size: int = 64
    routing_combined_region_population: bool = False
    performance_warmups: int = 0
    performance_repeats: int = 1
    progress_every: int = 0
    quiet: bool = False


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


def run_evaluation(
    config: RulerRunConfig,
    *,
    loaded_adapter: Any | None = None,
    attention_observer: Any | None = None,
    attention_override: Any | None = None,
    before_prediction_commit: Any | None = None,
) -> dict[str, Any]:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if config.attention_backend not in ("dense", "eager-dense", "blasst-reference", "fresh-routing"):
        raise ValueError(
            "attention_backend must be dense, eager-dense, or blasst-reference"
        )
    if config.stats_level not in ("summary", "step", "layer", "head"):
        raise ValueError("stats_level must be summary, step, layer, or head")
    if config.performance_warmups < 0 or config.performance_repeats <= 0:
        raise ValueError("performance_warmups must be nonnegative and performance_repeats positive")
    manifest, samples = _load_manifest(config)
    output_dir = Path(config.output_dir).resolve()
    semantics = {}
    if config.attention_backend == "blasst-reference" or config.blasst_calibration_lambdas:
        validate_blasst_output_directory(output_dir)
        semantics = {"blasst_mask_semantics": BLASST_MASK_SEMANTICS}
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "progress.log"
    def log_progress(message: str) -> None:
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    if config.progress_every < 0:
        raise ValueError("progress_every must be nonnegative")
    log_progress(f"run_start adapter={config.model_adapter} backend={config.attention_backend} samples={config.num_samples}")
    fingerprint_payload = {
        **asdict(config),
        **semantics,
        "output_dir": None,
        "manifest_sha256": sha256_file(config.manifest_path),
    }
    fingerprint = sha256_json(fingerprint_payload)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        prior = json.loads(run_config_path.read_text(encoding="utf-8"))
        compatible_fingerprints = {fingerprint}
        # Routing execution and logical block-size fields were added after
        # the first canonical Fast-dLLM conditions started. Preserve exact
        # resumability for those artifacts when the new fields are at their
        # historical defaults; any non-default value still receives a new
        # fingerprint and a separate output directory.
        if (
            config.routing_execution == "logical"
            and config.routing_q_block_size == 64
            and config.routing_kv_block_size == 64
        ):
            legacy_payload = dict(fingerprint_payload)
            for name in (
                "routing_execution", "routing_q_block_size", "routing_kv_block_size",
                "performance_warmups", "performance_repeats",
                "progress_every", "quiet",
            ):
                legacy_payload.pop(name, None)
            compatible_fingerprints.add(sha256_json(legacy_payload))
        if prior.get("fingerprint") not in compatible_fingerprints:
            raise ValueError("output directory belongs to a different RULER run")
    write_json(
        run_config_path,
        {**asdict(config), **_environment(), **semantics, "fingerprint": fingerprint},
    )

    predictions_path = output_dir / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if config.resume else []
    if not config.resume and predictions_path.exists():
        predictions_path.unlink()
    completed = {str(row["sample_id"]): row for row in existing}

    adapter = loaded_adapter
    if adapter is None:
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
    # Attention statistics are persisted independently of prediction shards.
    # On resume, seed the accumulator from the previous export so a partial
    # run cannot silently overwrite routing totals for already-completed
    # examples.
    if config.resume and config.collect_attention_stats and config.attention_backend == "blasst-reference":
        previous_stats_path = Path(config.output_dir).resolve() / "attention_stats" / "summary.json"
        if previous_stats_path.exists():
            previous = Blasst2DStats.load_export(previous_stats_path.parent)
            previous.record_layers = stats.record_layers
            previous.record_heads = stats.record_heads
            stats = previous
    binding = None
    routing = None
    if config.routing_mode is not None:
        if config.attention_backend != "fresh-routing":
            raise ValueError("routing_mode requires attention_backend=fresh-routing")
        if config.routing_density is None:
            raise ValueError("routing_density is required for fresh routing")
        from experiments.diffusion_attention_threshold_modeling.routing import (
            FreshRoutingAttention,
            FreshRoutingConfig,
        )
        routing = FreshRoutingAttention(FreshRoutingConfig(
            mode=config.routing_mode,
            target_density=float(config.routing_density),
            threshold_model=config.routing_threshold_model,
            adapter=config.model_adapter,
            corpus="ruler16k" if config.context_length == 16384 else "ruler8k",
            region=config.routing_region,
            random_seed=config.routing_random_seed,
            execution=config.routing_execution,
            q_block_size=config.routing_q_block_size,
            kv_block_size=config.routing_kv_block_size,
            combined_region_population=config.routing_combined_region_population,
        ))
    if config.attention_backend in ("eager-dense", "blasst-reference", "fresh-routing"):
        allowed_policy = {
            "local_blasst_lambda",
            "global_blasst_lambda",
            "denoising_phase_starts",
            "local_phase_lambdas",
            "global_phase_lambdas",
        }
        unknown_policy = set(config.blasst_policy) - allowed_policy
        if unknown_policy:
            raise ValueError(
                "unsupported BLASST policy field(s): "
                + ", ".join(sorted(unknown_policy))
            )
        policy = dict(config.blasst_policy)
        for name in (
            "denoising_phase_starts",
            "local_phase_lambdas",
            "global_phase_lambdas",
        ):
            if name in policy:
                policy[name] = tuple(policy[name])
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
                include_masked_kv_tiles_in_physical_stats=(
                    config.include_masked_kv_tiles_in_physical_stats
                ),
                apply_blasst_mask=(
                    config.attention_backend == "blasst-reference"
                    if config.apply_blasst_mask is None
                    else bool(config.apply_blasst_mask)
                ),
                **policy,
            ),
            stats,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            attention_class_names=adapter.attention_class_names,
            module_selector=adapter.is_blasst_attention_module,
            query_ids_extractor=adapter.blasst_query_ids,
            filter_special_query_ids=adapter.blasst_filter_special_query_ids,
            call_selector=adapter.blasst_call_is_eligible,
            dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
            sweep_lambdas=config.blasst_calibration_lambdas,
            attention_observer=attention_observer,
                integration=adapter.attention_integration,
            )
        if routing is not None:
            binding.runtime.attention_override = routing
        elif attention_override is not None:
            binding.runtime.attention_override = attention_override

    def persist_attention_stats() -> None:
        """Checkpoint BLASST routing totals after each completed sample.

        Predictions are appended before the next sample starts. Persisting the
        independent attention shard at the same boundary prevents a crash or
        pre-emption between those two operations from leaving a completed
        prediction with no corresponding routing statistics on resume.
        """

        if (
            binding is not None
            and config.collect_attention_stats
            and config.attention_backend == "blasst-reference"
        ):
            stats.export(
                output_dir / "attention_stats",
                binding.runtime.config,
                {"run_fingerprint": fingerprint},
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
            def reset_routing_state() -> None:
                if routing is not None:
                    routing.stats = type(routing.stats)()
                    routing._previous_retained_tiles.clear()

            budget = config.max_new_tokens or int(sample["tokens_to_generate"])
            request = GenerationRequest(
                prompt=str(sample["prompt"]),
                max_new_tokens=budget,
                block_size=config.block_size,
                steps=config.steps,
                threshold=config.threshold,
                temperature=config.temperature,
                seed=int(sample["inference_seed"]),
                extra=config.generation_extra,
            )
            for _ in range(config.performance_warmups):
                reset_routing_state()
                adapter.generate(request)
                if config.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
            repeated_results = []
            external_elapsed = []
            routing_repetitions = []
            for _ in range(config.performance_repeats):
                reset_routing_state()
                if config.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                repeated_results.append(adapter.generate(request))
                if config.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                external_elapsed.append(time.perf_counter() - started)
                if routing is not None:
                    routing_repetitions.append(routing.stats.summary())
            result = repeated_results[-1]
            if not math.isfinite(result.elapsed_seconds):
                raise RuntimeError(f"non-finite runtime for RULER sample {sample_id}")
            row = {
                **sample,
                **semantics,
                "prediction": result.text,
                "completion_tokens": result.completion_tokens,
                "elapsed_seconds": float(statistics.median(external_elapsed)),
                "adapter_elapsed_seconds_repetitions": [
                    float(item.elapsed_seconds) for item in repeated_results
                ],
                "synchronized_elapsed_seconds_repetitions": external_elapsed,
                "model_evaluations": result.model_evaluations,
                "termination_reason": result.termination_reason,
                "generation_metadata": result.metadata,
            }
            if routing is not None:
                row["routing_stats"] = routing_repetitions[-1]
            if before_prediction_commit is not None:
                before_prediction_commit()
            append_jsonl(predictions_path, row)
            completed[sample_id] = row
            persist_attention_stats()
            log_progress(
                f"completed sample={index}/{len(samples)} id={sample_id} elapsed={row['elapsed_seconds']:.6f}"
            )
            if (
                config.progress_every
                and index % config.progress_every == 0
                and not config.quiet
            ):
                print(f"progress {index}/{len(samples)} condition={output_dir.name}", flush=True)
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
        **semantics,
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
        "performance_protocol": {
            "warmups_per_sample": config.performance_warmups,
            "measured_repeats_per_sample": config.performance_repeats,
            "cuda_synchronized": bool(config.device.startswith("cuda") and torch.cuda.is_available()),
            "sample_latency_reducer": "median",
        },
    }
    if routing is not None:
        summary["routing"] = {
            "mode": config.routing_mode,
            "target_density": config.routing_density,
            "region": config.routing_region,
            "random_seed": config.routing_random_seed,
            "execution": config.routing_execution,
        }
        # Fresh-routing keeps its detailed per-call records inside each
        # prediction row. Also emit one canonical per-condition shard so
        # reports can be regenerated without replaying generation outputs.
        from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import (
            aggregate_routing_stats,
        )

        routing_rows = [
            row.get("routing_stats")
            for row in ordered
            if isinstance(row.get("routing_stats"), dict)
        ]
        summary["routing_aggregate"] = aggregate_routing_stats(routing_rows)
        write_json(
            output_dir / "routing_stats.json",
            {
                "schema_version": 1,
                "per_example": routing_rows,
                "aggregate": summary["routing_aggregate"],
            },
        )
    if config.device.startswith("cuda") and torch.cuda.is_available():
        summary["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        parameter_devices = sorted({str(parameter.device) for parameter in adapter.model.parameters()})
        summary["model_parameter_devices"] = parameter_devices
        summary["full_checkpoint_on_cuda"] = bool(parameter_devices) and all(
            value.startswith("cuda") for value in parameter_devices
        )
    if config.collect_attention_stats:
        if config.attention_backend == "blasst-reference":
            summary["attention_sparsity"] = stats.summary()
            stats.export(
                output_dir / "attention_stats",
                binding.runtime.config if binding is not None else Blasst2DConfig(),
                {"run_fingerprint": fingerprint},
            )
        if binding is not None and binding.runtime.sweep_stats:
            calibration_summary = {}
            for value, sweep_stats in binding.runtime.sweep_stats.items():
                name = f"lambda_{value:g}".replace(".", "p")
                calibration_summary[str(value)] = sweep_stats.summary()
                sweep_stats.export(
                    output_dir / "attention_calibration" / name,
                    binding.runtime.config,
                    {"run_fingerprint": fingerprint, "calibration_lambda": value},
                )
            summary["attention_calibration"] = calibration_summary
    log_progress(
        f"run_complete adapter={config.model_adapter} backend={config.attention_backend} samples={len(ordered)} accuracy={overall:.6f}"
    )
    write_json(output_dir / "summary.json", summary)
    return summary
