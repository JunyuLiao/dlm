"""Resumable ten-example prefix-proxy collection."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import GenerationRequest, create_adapter

from experiments.diffusion_attention_threshold_modeling.collect import (
    DEFAULT_DIFFUSION_GEMMA_REVISION,
    DEFAULT_MODELS,
    _distribution_prefix_extractor,
    _generation_request,
    _parity_row_passed,
    _run_capture,
    _write_json_atomic,
    compare_captures,
    coverage_check,
)
from experiments.diffusion_attention_threshold_modeling.datasets import load_math500

from .collector import PrefixProxyConfig, PrefixProxyShardCollector


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def run_collection(args: Any) -> dict[str, Any]:
    if args.num_problems != 10:
        raise ValueError("the redesigned study requires exactly 10 Math500 problems")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if args.corpus == "math500":
        problems = load_math500(args.nemo_gym_root, 10, args.split_seed)
    else:
        from experiments.diffusion_attention_threshold_modeling.datasets import load_ruler
        if args.prompts_jsonl is None:
            raise ValueError("RULER collection requires --prompts-jsonl")
        problems = load_ruler(args.prompts_jsonl, 10, args.split_seed, corpus=args.corpus)
    model_path = args.model_path or DEFAULT_MODELS[args.adapter]
    revision = args.revision
    if revision is None and args.adapter == "diffusion_gemma":
        revision = DEFAULT_DIFFUSION_GEMMA_REVISION
    adapter = create_adapter(
        args.adapter, model_path, device=args.device,
        precision=args.precision, revision=revision,
    ).load()
    config = PrefixProxyConfig(
        q_block_size=64, kv_block_size=64,
        reservoir_per_row=args.reservoir_per_row,
        reservoir_seed=args.split_seed,
    )
    collector = PrefixProxyShardCollector(config)
    root = args.output_dir
    shard_root = root / "shards" / args.adapter
    run_dir = root / "collection" / args.adapter
    shard_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    parity_rows = list(_load_json(run_dir / "parity.json", {"prompts": []}).get("prompts", []))
    parity_ids = {row["request_id"] for row in parity_rows}
    generations = _load_json(run_dir / "generations.json", [])
    generated_ids = {row["request_id"] for row in generations}
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
    started = time.time()
    try:
        for index, problem in enumerate(problems):
            request_id = str(problem["request_id"])
            shard = shard_root / f"{index:02d}_{request_id}.npz"
            request = _generation_request(args, problem, index)
            if shard.exists() and shard.with_suffix(".json").exists():
                continue
            baseline = baseline_values = noop_comparison = None
            if index < args.parity_prompts:
                binding.runtime.force_native_attention = True
                try:
                    baseline, baseline_values = _run_capture(adapter, request)
                finally:
                    binding.runtime.force_native_attention = False
                binding.runtime.attention_observer = lambda *unused, **unused_kwargs: None
                noop_result, noop_values = _run_capture(adapter, request)
                noop_comparison = compare_captures(baseline_values, noop_values)
                noop_comparison["completion_tokens_identical"] = baseline.completion_tokens == noop_result.completion_tokens
                binding.runtime.attention_observer = collector
            binding.runtime.forward_call_index = 0
            binding.runtime.current_denoising_iteration = -1
            binding.runtime.metadata_context = {
                "request_id": request_id, "problem_index": index,
                "corpus": args.corpus, "split": str(problem.get("split", "analysis")),
                "task": str(problem.get("subject", "math")),
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
                    request_id=request_id,
                    completion_tokens_identical=baseline.completion_tokens == observed.completion_tokens,
                    text_identical=baseline.text == observed.text,
                    native_noop_parity=noop_comparison,
                    attention_output_guard={
                        "checks": binding.runtime.attention_output_guard_checks,
                        "failures": binding.runtime.attention_output_guard_failures,
                        "passed": binding.runtime.attention_output_guard_checks > 0 and binding.runtime.attention_output_guard_failures == 0,
                    },
                )
            collector.export_shard(shard, metadata={
                "adapter": args.adapter, "model_path": model_path,
                "revision": revision, "seed": request.seed,
                "prompt_sha256": hashlib.sha256(problem["prompt"].encode()).hexdigest(),
            })
            if comparison is not None:
                comparison["coverage"] = coverage_check(shard, binding.modules)
                parity_rows.append(comparison)
                parity_ids.add(request_id)
                _write_json_atomic(run_dir / "parity.json", {
                    "prompts": parity_rows,
                    "passed": all(_parity_row_passed(row) for row in parity_rows),
                })
            if request_id not in generated_ids:
                generations.append({
                    "request_id": request_id,
                    "problem_index": index,
                    "seed": request.seed,
                    "completion_tokens": observed.completion_tokens,
                    "text": observed.text,
                })
                generated_ids.add(request_id)
                _write_json_atomic(run_dir / "generations.json", generations)
            binding.runtime.attention_output_guard = False
    finally:
        binding.close()
    parity = {
        "prompts": parity_rows,
        "passed": len(parity_rows) == min(args.parity_prompts, len(problems)) and all(_parity_row_passed(row) for row in parity_rows),
    }
    metadata = {
        "schema_version": 1,
        "study": f"{args.corpus}_prefix_proxy_distribution",
        "adapter": args.adapter,
        "model_path": model_path,
        "revision": revision,
        "corpus": args.corpus,
        "num_problems": len(problems),
        "split_seed": args.split_seed,
        "population": "prefix_only",
        "score_transform": "raw pre-softmax block-proxy logits; no row standardization",
        "q_block_size": 64,
        "kv_block_size": 64,
        "reservoir_per_row": args.reservoir_per_row,
        "parity": parity,
        "elapsed_seconds": time.time() - started,
        "runtime": adapter.runtime_metadata(),
        "deterministic_algorithms": bool(args.deterministic),
        "problem_ids": [str(row["request_id"]) for row in problems],
    }
    _write_json_atomic(run_dir / "metadata.json", metadata)
    _write_json_atomic(run_dir / "parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("dense-output parity failed; refusing to finalize prefix distribution")
    return metadata
