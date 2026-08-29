"""Native-dense collection runner for both study model adapters."""

from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import GenerationRequest, create_adapter

from .datasets import load_math500, load_ruler, stratified_split
from .convergence import write_convergence
from .proxy import CollectorConfig, PromptShardCollector


DEFAULT_MODELS = {
    "diffusion_gemma": "google/diffusiongemma-26B-A4B-it",
    "fast_dllm_v2": "Efficient-Large-Model/Fast_dLLM_v2_7B",
}


def _distribution_prefix_extractor(adapter_name: str, adapter: Any):
    """Select the deployable dense-prefix policy for distribution collection.

    DiffusionGemma concatenates a read-only encoder KV cache before its decoder
    canvas.  Its legacy adapter intentionally exposes that cache to historical
    BLASST routing experiments, but this study keeps prompt/encoder KV dense.
    With no runtime extractor, the collector uses ``KV length - query length``,
    which is exactly that encoder-cache boundary (and naturally zero when a
    local cache contains only canvas KV). Fast-dLLM-v2 needs its adapter's
    block-cache-aware boundary.
    """
    return None if adapter_name == "diffusion_gemma" else adapter.blasst_dense_kv_prefix
DEFAULT_DIFFUSION_GEMMA_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"


class TensorCapture:
    """Fingerprint the generation-relevant tail of each lm_head result."""

    def __init__(self, model: torch.nn.Module, max_sequence_rows: int) -> None:
        self.values: list[dict[str, Any]] = []
        self.max_sequence_rows = max_sequence_rows
        self.handle = model.lm_head.register_forward_hook(self._capture)

    def _capture(self, module, args, output) -> None:
        del module, args
        if isinstance(output, torch.Tensor):
            fingerprinted = output[..., -self.max_sequence_rows :, :]
            value = fingerprinted.detach().contiguous().view(torch.uint8).cpu().numpy()
            self.values.append({
                "shape": tuple(output.shape),
                "fingerprinted_shape": tuple(fingerprinted.shape),
                "dtype": str(output.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            })

    def close(self) -> None:
        self.handle.remove()


def _run_capture(adapter: Any, request: GenerationRequest) -> tuple[Any, list[dict[str, Any]]]:
    capture = TensorCapture(adapter.model, request.block_size)
    try:
        result = adapter.generate(request)
    finally:
        capture.close()
    return result, capture.values


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parity_row_passed(row: dict[str, Any]) -> bool:
    return bool(
        row.get("bitwise_identical", False)
        and row.get("completion_tokens_identical", False)
        and row.get("native_noop_parity", {}).get("bitwise_identical", False)
        and row.get("native_noop_parity", {}).get("completion_tokens_identical", False)
        and row.get("attention_output_guard", {}).get("passed", False)
        and row.get("coverage", {}).get("complete", False)
    )


def compare_captures(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    shape_match = len(left) == len(right) and all(
        a["shape"] == b["shape"] and a["dtype"] == b["dtype"] for a, b in zip(left, right)
    )
    identical = shape_match and bool(left) and all(a["sha256"] == b["sha256"] for a, b in zip(left, right))
    return {
        "forward_count_baseline": len(left),
        "forward_count_observed": len(right),
        "shape_match": shape_match,
        "bitwise_identical": identical,
        "comparison": (
            "SHA-256 over every byte in the generation-relevant trailing "
            "block of every lm_head output tensor"
        ),
    }


def coverage_check(path: Path, modules: list[torch.nn.Module]) -> dict[str, Any]:
    """Require every observed denoising call to cover every layer and head."""
    with np.load(path) as payload:
        calls = {int(value) for value in payload["denoising_call"]}
        layers = {int(value) for value in payload["layer"]}
        heads = {int(value) for value in payload["head"]}
        observed = {
            (int(call), int(layer), int(head))
            for call, layer, head in zip(
                payload["denoising_call"], payload["layer"], payload["head"]
            )
        }
    expected_layers = {
        int(getattr(module, "layer_idx", index)) for index, module in enumerate(modules)
    }
    model_config = getattr(modules[0], "config", None)
    expected_head_count = getattr(model_config, "num_attention_heads", None)
    if expected_head_count is None:
        head_dim = int(getattr(modules[0], "head_dim", 0))
        q_proj = getattr(modules[0], "q_proj", None)
        expected_head_count = (
            int(q_proj.out_features // head_dim)
            if q_proj is not None and head_dim
            else len(heads)
        )
    expected_heads = set(range(int(expected_head_count)))
    expected = {
        (call, layer, head)
        for call in calls
        for layer in expected_layers
        for head in expected_heads
    }
    missing = expected - observed
    return {
        "denoising_calls": len(calls),
        "layers": len(layers),
        "heads": len(heads),
        "expected_layers": len(expected_layers),
        "expected_heads": len(expected_heads),
        "missing_call_layer_head_combinations": len(missing),
        "complete": bool(calls) and layers == expected_layers and heads == expected_heads and not missing,
    }


def load_prompts(args: Any, count: int | None = None) -> list[dict[str, Any]]:
    count = args.num_prompts if count is None else count
    if args.corpus == "ruler8k":
        if args.prompts_jsonl is None:
            raise ValueError("RULER collection requires a tokenizer-specific --prompts-jsonl manifest")
        return load_ruler(args.prompts_jsonl, count, args.split_seed)
    if args.nemo_gym_root is None:
        raise ValueError("Math500 collection requires --nemo-gym-root")
    return load_math500(args.nemo_gym_root, count, args.split_seed)


def _generation_request(args: Any, row: dict[str, Any], index: int) -> GenerationRequest:
    budget = int(row.get("tokens_to_generate", args.max_new_tokens)) if args.corpus.startswith("ruler") else args.max_new_tokens
    extra: dict[str, Any] = {"thinking": False} if args.adapter == "diffusion_gemma" else {}
    if args.temperature > 0 and args.adapter == "diffusion_gemma":
        extra["top_p"] = args.top_p
    return GenerationRequest(
        prompt=row["prompt"],
        max_new_tokens=budget,
        block_size=args.generation_block_size,
        steps=args.steps,
        temperature=args.temperature,
        seed=int(row.get("inference_seed", args.base_seed + index)),
        extra=extra,
    )


def run_collection(args: Any) -> dict[str, Any]:
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    maximum = (
        args.max_distribution_prompts
        if args.auto_extend and args.num_prompts >= 50
        else args.num_prompts
    )
    prompts = load_prompts(args, args.num_prompts)
    if maximum > args.num_prompts:
        full = load_prompts(args, maximum)
        base_ids = {row["request_id"] for row in prompts}
        extension = [row for row in full if row["request_id"] not in base_ids]
        # Keep the canonical first-50 split immutable. Each adaptive 25-prompt
        # tranche receives its own deterministic stratified 80/20 assignment.
        for offset in range(0, len(extension), 25):
            prompts.extend(stratified_split(extension[offset : offset + 25], args.split_seed + 1 + offset // 25))
    model_path = args.model_path or DEFAULT_MODELS[args.adapter]
    revision = args.revision
    if revision is None and args.adapter == "diffusion_gemma":
        revision = DEFAULT_DIFFUSION_GEMMA_REVISION
    adapter = create_adapter(
        args.adapter, model_path, device=args.device, precision=args.precision, revision=revision
    ).load()
    config = CollectorConfig(
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
        reservoir_per_row=args.reservoir_per_row,
        densities=tuple(args.densities),
        reservoir_seed=args.split_seed,
    )
    collector = PromptShardCollector(config)
    shard_root = args.output_dir / "shards" / args.adapter / args.corpus
    shard_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "collection" / args.adapter / args.corpus
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_path = run_dir / "generations.json"
    generation_rows = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.exists()
        else []
    )
    generated_ids = {row["request_id"] for row in generation_rows}
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
        dense_kv_prefix_extractor=_distribution_prefix_extractor(args.adapter, adapter),
        attention_observer=collector,
        integration=adapter.attention_integration,
    )
    prior_parity = (
        json.loads((run_dir / "parity.json").read_text(encoding="utf-8"))
        if (run_dir / "parity.json").exists()
        else {"prompts": []}
    )
    parity_rows = list(prior_parity.get("prompts", []))
    parity_ids = {row["request_id"] for row in parity_rows}
    started = time.time()
    try:
        completed_count = 0
        for index, row in enumerate(prompts):
            safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(row["request_id"]))
            shard = shard_root / f"{index:04d}_{safe_id}.npz"
            if shard.exists() and shard.with_suffix(".json").exists():
                if index < args.parity_prompts and row["request_id"] not in parity_ids:
                    request = _generation_request(args, row, index)
                    binding.runtime.metadata_context = {
                        "request_id": row["request_id"], "corpus": args.corpus,
                        "task": row.get("task", "unclassified"), "split": row["split"],
                        "inference_seed": request.seed,
                    }
                    binding.runtime.force_native_attention = True
                    try:
                        baseline, baseline_values = _run_capture(adapter, request)
                    finally:
                        binding.runtime.force_native_attention = False
                    binding.runtime.attention_observer = lambda *unused_args, **unused_kwargs: None
                    noop_result, noop_values = _run_capture(adapter, request)
                    noop_comparison = compare_captures(baseline_values, noop_values)
                    noop_comparison["completion_tokens_identical"] = (
                        baseline.completion_tokens == noop_result.completion_tokens
                    )
                    binding.runtime.forward_call_index = 0
                    binding.runtime.current_denoising_iteration = -1
                    binding.runtime.attention_output_guard = True
                    binding.runtime.attention_output_guard_checks = 0
                    binding.runtime.attention_output_guard_failures = 0
                    observed, observed_values = _run_capture(adapter, request)
                    comparison = compare_captures(baseline_values, observed_values)
                    comparison.update(
                        request_id=row["request_id"],
                        completion_tokens_identical=baseline.completion_tokens == observed.completion_tokens,
                        text_identical=baseline.text == observed.text,
                        native_noop_parity=noop_comparison,
                        attention_output_guard={
                            "checks": binding.runtime.attention_output_guard_checks,
                            "failures": binding.runtime.attention_output_guard_failures,
                            "passed": binding.runtime.attention_output_guard_checks > 0
                            and binding.runtime.attention_output_guard_failures == 0,
                        },
                        coverage=coverage_check(shard, binding.modules),
                    )
                    parity_rows.append(comparison)
                    parity_ids.add(row["request_id"])
                    _write_json_atomic(
                        run_dir / "parity.json",
                        {"prompts": parity_rows, "passed": all(map(_parity_row_passed, parity_rows))},
                    )
                    if row["request_id"] not in generated_ids:
                        generation_rows.append({
                            "request_id": row["request_id"], "corpus": args.corpus,
                            "task": row.get("task"), "split": row["split"], "seed": request.seed,
                            "completion_tokens": observed.completion_tokens, "text": observed.text,
                        })
                        generated_ids.add(row["request_id"])
                        _write_json_atomic(generation_path, generation_rows)
                    binding.runtime.attention_output_guard = False
                    binding.runtime.attention_observer = collector
                prior = json.loads(shard.with_suffix(".json").read_text(encoding="utf-8"))
                expected_context = {
                    "request_id": row["request_id"],
                    "corpus": args.corpus,
                    "task": row.get("task", "unclassified"),
                    "split": row["split"],
                }
                if any(str(prior.get("context", {}).get(key)) != str(value) for key, value in expected_context.items()):
                    raise RuntimeError(f"existing shard context differs from this run: {shard}")
                if prior.get("config", {}).get("q_block_size") != args.q_block_size or prior.get("config", {}).get("kv_block_size") != args.kv_block_size:
                    raise RuntimeError(f"existing shard block geometry differs from this run: {shard}")
                completed_count = index + 1
                if completed_count >= args.num_prompts and (
                    completed_count == args.num_prompts or completed_count % 25 == 0
                ):
                    convergence = write_convergence(
                        args.output_dir, args.adapter, args.corpus, args.densities, args.split_seed
                    )
                    if convergence["passed"] or completed_count >= args.max_distribution_prompts:
                        prompts = prompts[:completed_count]
                        break
                continue
            request = _generation_request(args, row, index)
            if index < args.parity_prompts:
                binding.runtime.force_native_attention = True
                try:
                    baseline, baseline_values = _run_capture(adapter, request)
                finally:
                    binding.runtime.force_native_attention = False
                binding.runtime.attention_observer = lambda *unused_args, **unused_kwargs: None
                noop_result, noop_values = _run_capture(adapter, request)
                noop_comparison = compare_captures(baseline_values, noop_values)
                noop_comparison["completion_tokens_identical"] = (
                    baseline.completion_tokens == noop_result.completion_tokens
                )
                binding.runtime.attention_observer = collector
            else:
                baseline = None
                baseline_values = []
                noop_comparison = None
            binding.runtime.forward_call_index = 0
            binding.runtime.current_denoising_iteration = -1
            binding.runtime.metadata_context = {
                "request_id": row["request_id"],
                "corpus": args.corpus,
                "task": row.get("task", "unclassified"),
                "split": row["split"],
                "inference_seed": request.seed,
            }
            collector.begin_prompt(**binding.runtime.metadata_context)
            binding.runtime.attention_output_guard = index < args.parity_prompts
            binding.runtime.attention_output_guard_checks = 0
            binding.runtime.attention_output_guard_failures = 0
            observed, observed_values = _run_capture(adapter, request)
            comparison = None
            if index < args.parity_prompts:
                comparison = compare_captures(baseline_values, observed_values)
                comparison.update(
                    request_id=row["request_id"],
                    completion_tokens_identical=baseline.completion_tokens == observed.completion_tokens,
                    text_identical=baseline.text == observed.text,
                    native_noop_parity=noop_comparison,
                    attention_output_guard={
                        "checks": binding.runtime.attention_output_guard_checks,
                        "failures": binding.runtime.attention_output_guard_failures,
                        "passed": (
                            binding.runtime.attention_output_guard_checks > 0
                            and binding.runtime.attention_output_guard_failures == 0
                        ),
                    },
                )
            collector.export_shard(
                shard,
                metadata={
                    "adapter": args.adapter,
                    "model_path": model_path,
                    "revision": revision,
                    "seed": request.seed,
                    "prompt_sha256": __import__("hashlib").sha256(row["prompt"].encode()).hexdigest(),
                },
            )
            if comparison is not None:
                comparison["coverage"] = coverage_check(shard, binding.modules)
                if row["request_id"] not in parity_ids:
                    parity_rows.append(comparison)
                    parity_ids.add(row["request_id"])
                    _write_json_atomic(
                        run_dir / "parity.json",
                        {"prompts": parity_rows, "passed": all(map(_parity_row_passed, parity_rows))},
                    )
            binding.runtime.attention_output_guard = False
            if row["request_id"] not in generated_ids:
                generation_rows.append({
                    "request_id": row["request_id"],
                    "corpus": args.corpus,
                    "task": row.get("task"),
                    "split": row["split"],
                    "seed": request.seed,
                    "completion_tokens": observed.completion_tokens,
                    "text": observed.text,
                })
                generated_ids.add(row["request_id"])
                _write_json_atomic(generation_path, generation_rows)
            completed_count = index + 1
            if completed_count >= args.num_prompts and (
                completed_count == args.num_prompts or completed_count % 25 == 0
            ):
                convergence = write_convergence(
                    args.output_dir, args.adapter, args.corpus, args.densities, args.split_seed
                )
                if convergence["passed"] or completed_count >= args.max_distribution_prompts:
                    prompts = prompts[:completed_count]
                    break
    finally:
        binding.close()
    parity = {
        "prompts": parity_rows,
        "passed": (
            len(parity_rows) == min(args.parity_prompts, len(prompts))
            and all(map(_parity_row_passed, parity_rows))
        ),
    }
    metadata = {
        "schema_version": 1,
        "adapter": args.adapter,
        "model_path": model_path,
        "revision": revision,
        "corpus": args.corpus,
        "num_prompts": len(prompts),
        "split_seed": args.split_seed,
        "prompt_splits": {row["request_id"]: row["split"] for row in prompts},
        "collector": config.__dict__,
        "parity": parity,
        "elapsed_seconds": time.time() - started,
        "runtime": adapter.runtime_metadata(),
        "deterministic_algorithms": bool(args.deterministic),
        "convergence": write_convergence(
            args.output_dir, args.adapter, args.corpus, args.densities, args.split_seed
        ),
    }
    _write_json_atomic(run_dir / "metadata.json", metadata)
    _write_json_atomic(generation_path, generation_rows)
    _write_json_atomic(run_dir / "parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("dense-output parity failed; refusing to profile")
    return metadata
