"""Single-prompt RULER 16K stepwise prefix-proxy diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from types import MethodType
from pathlib import Path

import torch

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.models import create_adapter
from experiments.diffusion_attention_threshold_modeling.collect import (
    DEFAULT_DIFFUSION_GEMMA_REVISION,
    DEFAULT_MODELS,
    _distribution_prefix_extractor,
    _generation_request,
)
from experiments.diffusion_attention_threshold_modeling.datasets import read_jsonl

from .collector import DiagnosticConfig, StepwisePrefixCollector
from .plots import generate_plots


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--adapter", choices=("diffusion_gemma", "fast_dllm_v2"), required=True)
    collect.add_argument("--model-path")
    collect.add_argument("--revision")
    collect.add_argument("--prompts-jsonl", type=Path, required=True)
    collect.add_argument("--request-id")
    collect.add_argument("--model-index", type=int, default=0)
    collect.add_argument("--reservoir-per-row", type=int, default=256)
    collect.add_argument("--split-seed", type=int, default=20260824)
    collect.add_argument("--base-seed", type=int, default=42)
    collect.add_argument("--max-new-tokens", type=int, default=2048)
    collect.add_argument("--generation-block-size", type=int, default=256)
    collect.add_argument("--steps", type=int)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--top-p", type=float, default=0.95)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--precision", default="bfloat16")
    collect.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--attention-type", default="global")
    plot.add_argument("--layer", type=int, default=0)
    plot.add_argument("--head", type=int, default=0)
    plot.add_argument("--query-block", type=int, default=0)
    return root


def collect(args: argparse.Namespace) -> dict:
    args.corpus = "ruler16k"
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    rows = read_jsonl(args.prompts_jsonl)
    selected = next((row for row in rows if args.request_id and row["request_id"] == args.request_id), None)
    if selected is None:
        selected = rows[args.model_index]
    selected["corpus"] = "ruler16k"
    model_path = args.model_path or DEFAULT_MODELS[args.adapter]
    revision = args.revision
    if revision is None and args.adapter == "diffusion_gemma":
        revision = DEFAULT_DIFFUSION_GEMMA_REVISION
    adapter = create_adapter(args.adapter, model_path, device=args.device, precision=args.precision, revision=revision).load()
    collector = StepwisePrefixCollector(DiagnosticConfig(reservoir_per_row=args.reservoir_per_row, reservoir_seed=args.split_seed))
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
    request = _generation_request(args, selected, 0)
    binding.runtime.forward_call_index = 0
    binding.runtime.current_denoising_iteration = -1
    binding.runtime.metadata_context = {
        "request_id": selected["request_id"], "problem_index": 0,
        "corpus": "ruler16k", "split": str(selected.get("split", "diagnostic")),
        "task": str(selected.get("task", "unknown")), "inference_seed": request.seed,
    }
    collector.begin_prompt(**binding.runtime.metadata_context)
    mask_token_id = adapter.mask_token_id
    if mask_token_id is None and args.adapter == "fast_dllm_v2":
        mask_token_id = 151665
    base_model = getattr(adapter.model, "model", adapter.model)

    def capture_mask_state(_module, hook_args, hook_kwargs):
        input_ids = hook_kwargs.get("input_ids", hook_kwargs.get("decoder_input_ids"))
        if input_ids is None and hook_args and isinstance(hook_args[0], torch.Tensor):
            input_ids = hook_args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or mask_token_id is None:
            return
        pad_token_id = adapter.pad_token_id
        active = torch.ones_like(input_ids, dtype=torch.bool)
        if pad_token_id is not None:
            active &= input_ids != int(pad_token_id)
        masked = int((input_ids == int(mask_token_id)).sum().item())
        active_count = int(active.sum().item())
        collector.set_forward_state(
            masked_tokens=masked,
            active_tokens=active_count,
            mask_ratio=masked / active_count if active_count else 0.0,
        )

    state_hook = base_model.register_forward_pre_hook(capture_mask_state, with_kwargs=True)
    sampler_patch = None
    if args.adapter == "diffusion_gemma":
        # DiffusionGemma has no mask token.  Its discrete denoising state is
        # represented by the entropy sampler: positions not accepted at the
        # current step are re-noised and remain unresolved for the next call.
        # Record that effective unresolved fraction before each decoder call.
        original_prepare_sampler = adapter.model._prepare_sampler

        def prepare_sampler_with_state(self, generation_config):
            sampler = original_prepare_sampler(generation_config)
            canvas_length = int(getattr(self.config, "canvas_length"))
            collector.set_forward_state(
                masked_tokens=canvas_length,
                active_tokens=canvas_length,
                mask_ratio=1.0,
            )
            original_renoise = sampler.renoise_canvas

            def renoise_with_state(accepted_canvas, cur_step):
                output = original_renoise(accepted_canvas, cur_step)
                accepted = getattr(sampler, "accepted_token_mask", None)
                if isinstance(accepted, torch.Tensor):
                    active = int(accepted.numel())
                    unresolved = int((~accepted).sum().item())
                    collector.set_forward_state(
                        masked_tokens=unresolved,
                        active_tokens=active,
                        mask_ratio=unresolved / active if active else 0.0,
                    )
                return output

            sampler.renoise_canvas = renoise_with_state
            return sampler

        adapter.model._prepare_sampler = MethodType(prepare_sampler_with_state, adapter.model)
        sampler_patch = original_prepare_sampler
    try:
        result = adapter.generate(request)
    finally:
        state_hook.remove()
        if sampler_patch is not None:
            adapter.model._prepare_sampler = sampler_patch
        binding.close()
    output_dir = args.output_dir
    shard = output_dir / f"{args.adapter}_{selected['request_id']}.npz"
    collector.export(shard, metadata={
        "adapter": args.adapter, "model_path": model_path, "revision": revision,
        "prompt_sha256": hashlib.sha256(selected["prompt"].encode()).hexdigest(),
        "completion_tokens": result.completion_tokens, "text": result.text,
        "state_definition": (
            "fraction of active input IDs equal to the model mask token"
            if args.adapter == "fast_dllm_v2"
            else "fraction of canvas positions re-noised by DiffusionGemma entropy sampler before the next denoising call"
        ),
        "problem": selected,
    })
    return {"adapter": args.adapter, "request_id": selected["request_id"], "shard": str(shard), "rows": len(collector.rows)}


def main() -> None:
    args = parser().parse_args()
    result = collect(args) if args.command == "collect" else generate_plots(
        args.output_dir, attention_type=args.attention_type, layer=args.layer,
        head=args.head, query_block=args.query_block,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
