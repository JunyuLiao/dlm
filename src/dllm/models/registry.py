"""Lazy registry for model adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ModelAdapter


_ADAPTERS: dict[str, str | type["ModelAdapter"]] = {
    "fast_dllm_v1_llada": "dllm.models.adapters.fast_dllm_v1:LLaDAV1Adapter",
    "fast_dllm_v1_dream": "dllm.models.adapters.fast_dllm_v1:DreamV1Adapter",
    "fast_dllm_v2": "dllm.models.adapters.fast_dllm_v2:FastDLLMV2Adapter",
    "llada2_1_mini": "dllm.models.adapters.llada21:LLaDA21Adapter",
    "diffusion_gemma": "dllm.models.adapters.diffusion_gemma:DiffusionGemmaAdapter",
}


def register_adapter(name: str) -> Callable[[type["ModelAdapter"]], type["ModelAdapter"]]:
    def decorator(adapter: type["ModelAdapter"]) -> type["ModelAdapter"]:
        if name in _ADAPTERS:
            raise ValueError(f"model adapter already registered: {name}")
        _ADAPTERS[name] = adapter
        return adapter

    return decorator


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def adapter_class(name: str) -> type["ModelAdapter"]:
    try:
        target = _ADAPTERS[name]
    except KeyError as error:
        choices = ", ".join(adapter_names())
        raise ValueError(f"unknown model adapter {name!r}; choose one of: {choices}") from error
    if isinstance(target, str):
        module_name, symbol = target.split(":", 1)
        target = getattr(importlib.import_module(module_name), symbol)
        _ADAPTERS[name] = target
    return target


def create_adapter(name: str, model_path: str, **kwargs) -> "ModelAdapter":
    return adapter_class(name)(model_path, **kwargs)
