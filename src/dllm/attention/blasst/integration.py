"""Binding and direct-SDPA dispatch for the BLASST reference backend."""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch

from ..dense import dense_scaled_dot_product_attention
from .core import (
    Blasst2DConfig,
    Blasst2DRuntime,
    Blasst2DStats,
    _observe_blasst_sweep,
    blasst_2d_attention_forward,
    dense_eager_attention_forward,
)


_REGISTRY_LOCK = threading.RLock()
_REGISTRY_PATCHES: dict[int, dict[str, Any]] = {}


@dataclass
class BlasstBinding:
    runtime: Blasst2DRuntime
    modules: list[torch.nn.Module]
    registry_key: int | None = None
    added_layer_indices: list[torch.nn.Module] | None = None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.runtime.hook_handle is not None:
            self.runtime.hook_handle.remove()
            self.runtime.hook_handle = None
        for module in self.modules:
            for attribute in ("_blasst_2d_runtime", "_dllm_attention_runtime"):
                if getattr(module, attribute, None) is self.runtime:
                    delattr(module, attribute)
            if hasattr(module, "_blasst_kv_already_repeated"):
                delattr(module, "_blasst_kv_already_repeated")
        for module in self.added_layer_indices or ():
            if hasattr(module, "layer_idx"):
                delattr(module, "layer_idx")
        if self.registry_key is not None:
            with _REGISTRY_LOCK:
                state = _REGISTRY_PATCHES[self.registry_key]
                state["references"] -= 1
                if state["references"] == 0:
                    if state["registry"]["sdpa"] is not state["dispatcher"]:
                        raise RuntimeError("Hugging Face SDPA registry changed while BLASST was active")
                    state["registry"]["sdpa"] = state["original"]
                    del _REGISTRY_PATCHES[self.registry_key]

    def __enter__(self) -> "BlasstBinding":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _capture_runtime_state(
    runtime: Blasst2DRuntime,
    mask_token_id: int | None,
    pad_token_id: int | None,
    query_ids_extractor: Callable[[torch.nn.Module, tuple, dict], Any] | None,
    filter_special_query_ids: bool,
    call_selector: Callable[[torch.nn.Module, tuple, dict], bool] | None,
):
    def capture(module, args, kwargs) -> None:
        input_ids = (
            query_ids_extractor(module, args, kwargs)
            if query_ids_extractor is not None
            else kwargs.get("input_ids")
        )
        if (
            query_ids_extractor is None
            and input_ids is None
            and args
            and isinstance(args[0], torch.Tensor)
        ):
            input_ids = args[0]
        runtime.active_call = (
            bool(call_selector(module, args, kwargs))
            if call_selector is not None
            else True
        )
        runtime.metadata = dict(runtime.metadata_context)
        if input_ids is None or input_ids.ndim != 2:
            runtime.active_query_mask = None
            return
        active = torch.ones_like(input_ids, dtype=torch.bool)
        if filter_special_query_ids and pad_token_id is not None:
            active &= input_ids != pad_token_id
        runtime.active_query_mask = active
        if kwargs.get("self_conditioning_logits") is None:
            runtime.current_denoising_iteration = 0
        else:
            runtime.current_denoising_iteration = max(
                1, runtime.current_denoising_iteration + 1
            )
        runtime.metadata["denoising_iteration"] = (
            runtime.current_denoising_iteration
        )
        runtime.metadata["denoising_phase"] = runtime.config.phase_for(
            runtime.current_denoising_iteration
        )
        if filter_special_query_ids and mask_token_id is not None:
            masked = int((input_ids == mask_token_id).sum().item())
            total = int(active.sum().item())
            runtime.metadata.update(
                masked_tokens=masked,
                mask_ratio=masked / total if total else 0.0,
            )
        runtime.metadata["forward_pass_id"] = runtime.forward_call_index
        runtime.metadata["denoising_step"] = runtime.forward_call_index
        runtime.forward_call_index += 1

    return capture


def _attach_direct(
    model: torch.nn.Module,
    modules: list[torch.nn.Module],
    config: Blasst2DConfig,
    stats: Blasst2DStats | None,
    *,
    mask_token_id: int | None,
    pad_token_id: int | None,
    query_ids_extractor: Callable[[torch.nn.Module, tuple, dict], Any] | None,
    filter_special_query_ids: bool,
    call_selector: Callable[[torch.nn.Module, tuple, dict], bool] | None,
    dense_kv_prefix_extractor: Callable[..., int] | None,
    sweep_lambdas: tuple[float, ...],
    attention_observer: Callable[..., None] | None,
) -> BlasstBinding:
    runtime = Blasst2DRuntime(
        config=config,
        stats=stats
        or Blasst2DStats(
            record_layers=config.collect_blasst_layer_stats,
            record_heads=config.collect_blasst_head_stats,
        ),
    )
    runtime.dense_kv_prefix_extractor = dense_kv_prefix_extractor
    runtime.attention_observer = attention_observer
    runtime.sweep_lambdas = sweep_lambdas
    runtime.sweep_stats = {
        value: Blasst2DStats(
            record_layers=config.collect_blasst_layer_stats,
            record_heads=config.collect_blasst_head_stats,
        )
        for value in sweep_lambdas
    }
    added_layer_indices = []
    for index, module in enumerate(modules):
        module._dllm_attention_runtime = runtime
        module._blasst_2d_runtime = runtime
        module._blasst_kv_already_repeated = True
        if not hasattr(module, "layer_idx"):
            module.layer_idx = index
            added_layer_indices.append(module)
    base_model = getattr(model, "model", model)
    runtime.hook_handle = base_model.register_forward_pre_hook(
        _capture_runtime_state(
            runtime,
            mask_token_id,
            pad_token_id,
            query_ids_extractor,
            filter_special_query_ids,
            call_selector,
        ),
        with_kwargs=True,
    )
    return BlasstBinding(
        runtime=runtime,
        modules=modules,
        added_layer_indices=added_layer_indices,
    )


def _attach_registry(
    model: torch.nn.Module,
    modules: list[torch.nn.Module],
    registry: Any,
    config: Blasst2DConfig,
    stats: Blasst2DStats | None,
    *,
    mask_token_id: int | None,
    pad_token_id: int | None,
    query_ids_extractor: Callable[[torch.nn.Module, tuple, dict], Any] | None,
    filter_special_query_ids: bool,
    call_selector: Callable[[torch.nn.Module, tuple, dict], bool] | None,
    dense_kv_prefix_extractor: Callable[..., int] | None,
    sweep_lambdas: tuple[float, ...],
    attention_observer: Callable[..., None] | None,
) -> BlasstBinding:
    runtime = Blasst2DRuntime(
        config=config,
        stats=stats
        or Blasst2DStats(
            record_layers=config.collect_blasst_layer_stats,
            record_heads=config.collect_blasst_head_stats,
        ),
    )
    runtime.dense_kv_prefix_extractor = dense_kv_prefix_extractor
    runtime.attention_observer = attention_observer
    runtime.sweep_lambdas = sweep_lambdas
    runtime.sweep_stats = {
        value: Blasst2DStats(
            record_layers=config.collect_blasst_layer_stats,
            record_heads=config.collect_blasst_head_stats,
        )
        for value in sweep_lambdas
    }
    for module in modules:
        module._blasst_2d_runtime = runtime
    base_model = getattr(model, "model", model)
    runtime.hook_handle = base_model.register_forward_pre_hook(
        _capture_runtime_state(
            runtime,
            mask_token_id,
            pad_token_id,
            query_ids_extractor,
            filter_special_query_ids,
            call_selector,
        ),
        with_kwargs=True,
    )

    registry_key = id(registry)
    with _REGISTRY_LOCK:
        state = _REGISTRY_PATCHES.get(registry_key)
        if state is None:
            original = registry["sdpa"]

            def dispatcher(module, *args, **kwargs):
                tagged = getattr(module, "_blasst_2d_runtime", None)
                if (
                    tagged is None
                    or not tagged.config.enable_blasst_2d
                    or not tagged.active_call
                    or getattr(tagged, "force_native_attention", False)
                ):
                    return original(module, *args, **kwargs)
                if tagged.attention_override is not None:
                    return tagged.attention_override(module, *args, **kwargs)
                if tagged.attention_observer is not None:
                    result = original(module, *args, **kwargs)
                    guarded = getattr(tagged, "attention_output_guard", False)
                    output = result[0] if isinstance(result, tuple) else result
                    snapshot = output.detach().clone() if guarded else None
                    tagged.attention_observer(module, *args, **kwargs)
                    if guarded:
                        tagged.attention_output_guard_checks = getattr(
                            tagged, "attention_output_guard_checks", 0
                        ) + 1
                        # Compare bytes so identical NaN payloads count as
                        # unchanged; torch.equal treats NaN != NaN.
                        if not torch.equal(
                            snapshot.contiguous().view(torch.uint8),
                            output.contiguous().view(torch.uint8),
                        ):
                            tagged.attention_output_guard_failures = getattr(
                                tagged, "attention_output_guard_failures", 0
                            ) + 1
                    return result
                if tagged.sweep_lambdas:
                    _observe_blasst_sweep(module, *args, **kwargs)
                    return dense_eager_attention_forward(module, *args, **kwargs)
                if not tagged.config.apply_blasst_mask:
                    return dense_eager_attention_forward(module, *args, **kwargs)
                return blasst_2d_attention_forward(module, *args, **kwargs)

            state = {
                "registry": registry,
                "original": original,
                "dispatcher": dispatcher,
                "references": 0,
            }
            registry["sdpa"] = dispatcher
            _REGISTRY_PATCHES[registry_key] = state
        elif state["registry"] is not registry:
            raise RuntimeError("attention registry identity collision")
        state["references"] += 1
    return BlasstBinding(
        runtime=runtime,
        modules=modules,
        registry_key=registry_key,
    )


def install_blasst(
    model: torch.nn.Module,
    config: Blasst2DConfig,
    stats: Blasst2DStats | None = None,
    *,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    attention_class_names: tuple[str, ...] | None = None,
    module_selector: Callable[[str, torch.nn.Module], bool] | None = None,
    query_ids_extractor: Callable[[torch.nn.Module, tuple, dict], Any] | None = None,
    filter_special_query_ids: bool = True,
    call_selector: Callable[[torch.nn.Module, tuple, dict], bool] | None = None,
    dense_kv_prefix_extractor: Callable[..., int] | None = None,
    sweep_lambdas: tuple[float, ...] | list[float] = (),
    attention_observer: Callable[..., None] | None = None,
    integration: str = "auto",
) -> BlasstBinding:
    """Install BLASST on any registered current dLLM attention implementation."""
    names = attention_class_names or (
        "Fast_dLLM_QwenAttention",
        "LLaDA2MoeAttention",
        "DreamSdpaAttention",
        "LLaDASequentialBlock",
        "LLaDALlamaBlock",
        "LLaDABlockDiffBlock",
        "LLaDA2Attention",
    )
    modules = [
        module
        for name, module in model.named_modules()
        if (
            module_selector(name, module)
            if module_selector is not None
            else type(module).__name__ in names
        )
    ]
    if not modules:
        raise ValueError(
            "No supported attention modules found; expected one of: "
            + ", ".join(names)
        )
    if integration not in ("auto", "registry", "direct"):
        raise ValueError("integration must be auto, registry, or direct")
    sweep_values = tuple(float(value) for value in sweep_lambdas)
    if any(not 0.0 < value < 1.0 for value in sweep_values):
        raise ValueError("all sweep lambdas must be strictly between 0 and 1")
    modeling = importlib.import_module(type(modules[0]).__module__)
    registry = getattr(modeling, "ALL_ATTENTION_FUNCTIONS", None)
    use_registry = integration == "registry" or (
        integration == "auto" and registry is not None and "sdpa" in registry
    )
    if use_registry:
        if registry is None or "sdpa" not in registry:
            raise ValueError("model adapter requested a missing SDPA registry")
        return _attach_registry(
            model,
            modules,
            registry,
            config,
            stats,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            query_ids_extractor=query_ids_extractor,
            filter_special_query_ids=filter_special_query_ids,
            call_selector=call_selector,
            dense_kv_prefix_extractor=dense_kv_prefix_extractor,
            sweep_lambdas=sweep_values,
            attention_observer=attention_observer,
        )
    return _attach_direct(
        model,
        modules,
        config,
        stats,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        query_ids_extractor=query_ids_extractor,
        filter_special_query_ids=filter_special_query_ids,
        call_selector=call_selector,
        dense_kv_prefix_extractor=dense_kv_prefix_extractor,
        sweep_lambdas=sweep_values,
        attention_observer=attention_observer,
    )


def dispatch_scaled_dot_product_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Use native dense SDPA unless this specific module has BLASST attached."""
    runtime = getattr(module, "_dllm_attention_runtime", None)
    if (
        runtime is None
        or not runtime.config.enable_blasst_2d
        or getattr(runtime, "force_native_attention", False)
    ):
        return dense_scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
    active = runtime.active_query_mask
    if active is not None and active.shape[-1] != query.shape[-2]:
        active = active[..., -query.shape[-2] :]
    kwargs = {
        "dropout": dropout_p,
        "scaling": scale,
        "is_causal": is_causal,
        "blasst_active_query_mask": active,
    }
    if runtime.attention_override is not None:
        output, _ = runtime.attention_override(
            module, query, key, value, attention_mask, **kwargs
        )
    elif runtime.attention_observer is not None:
        output = dense_scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        guarded = getattr(runtime, "attention_output_guard", False)
        snapshot = output.detach().clone() if guarded else None
        runtime.attention_observer(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )
        if guarded:
            runtime.attention_output_guard_checks = getattr(
                runtime, "attention_output_guard_checks", 0
            ) + 1
            if not torch.equal(
                snapshot.contiguous().view(torch.uint8),
                output.contiguous().view(torch.uint8),
            ):
                runtime.attention_output_guard_failures = getattr(
                    runtime, "attention_output_guard_failures", 0
                ) + 1
    elif runtime.sweep_lambdas:
        _observe_blasst_sweep(
            module, query, key, value, attention_mask, **kwargs
        )
        output, _ = dense_eager_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    elif not runtime.config.apply_blasst_mask:
        output, _ = dense_eager_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    else:
        output, _ = blasst_2d_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
        )
    return output.transpose(1, 2).contiguous()
