"""Read-only binding to Hugging Face's registered attention interface."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from .proxy import AttentionObserver
from .replay import ReplayAttention


@dataclass
class AttentionObserverBinding:
    model: torch.nn.Module
    modules: list[torch.nn.Module]
    registry: Any
    original: Any
    dispatcher: Any
    hook_handle: Any
    observer: AttentionObserver
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.hook_handle.remove()
        if self.registry["sdpa"] is not self.dispatcher:
            raise RuntimeError("attention registry changed while observation was active")
        self.registry["sdpa"] = self.original
        for module in self.modules:
            if getattr(module, "_diffusion_attention_observer", None) is self.observer:
                delattr(module, "_diffusion_attention_observer")

    def __enter__(self) -> "AttentionObserverBinding":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def install_attention_observer(
    model: torch.nn.Module,
    observer: AttentionObserver,
    *,
    attention_class_name: str = "DiffusionGemmaDecoderTextAttention",
) -> AttentionObserverBinding:
    """Observe selected dense calls and return the native attention result unchanged."""
    modules = [
        module
        for module in model.modules()
        if type(module).__name__ == attention_class_name
    ]
    if not modules:
        raise ValueError(f"no {attention_class_name} modules found")
    modeling = importlib.import_module(type(modules[0]).__module__)
    registry = getattr(modeling, "ALL_ATTENTION_FUNCTIONS", None)
    if registry is None or "sdpa" not in registry:
        raise ValueError("model does not expose a registered sdpa attention interface")
    original = registry["sdpa"]
    for module in modules:
        module._diffusion_attention_observer = observer

    def dispatcher(module, query, key, value, attention_mask, **kwargs):
        tagged = getattr(module, "_diffusion_attention_observer", None)
        if tagged is not None:
            tagged.observe(module, query, key, value, attention_mask, **kwargs)
        return original(module, query, key, value, attention_mask, **kwargs)

    registry["sdpa"] = dispatcher

    def begin_forward(_module, args, kwargs) -> None:
        decoder_ids = kwargs.get("decoder_input_ids")
        if decoder_ids is None and args:
            decoder_ids = args[0] if isinstance(args[0], torch.Tensor) else None
        observer.begin_forward(decoder_ids, kwargs.get("self_conditioning_logits"))

    base_model = getattr(model, "model", model)
    hook_handle = base_model.register_forward_pre_hook(begin_forward, with_kwargs=True)
    return AttentionObserverBinding(
        model=model,
        modules=modules,
        registry=registry,
        original=original,
        dispatcher=dispatcher,
        hook_handle=hook_handle,
        observer=observer,
    )


@dataclass
class AttentionReplayBinding:
    modules: list[torch.nn.Module]
    registry: Any
    original: Any
    dispatcher: Any
    hook_handle: Any
    replay: ReplayAttention
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.hook_handle.remove()
        if self.registry["sdpa"] is not self.dispatcher:
            raise RuntimeError("attention registry changed while replay was active")
        self.registry["sdpa"] = self.original
        for module in self.modules:
            if getattr(module, "_diffusion_attention_replay", None) is self.replay:
                delattr(module, "_diffusion_attention_replay")

    def __enter__(self) -> "AttentionReplayBinding":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def install_attention_replay(
    model: torch.nn.Module,
    replay: ReplayAttention,
    *,
    attention_class_name: str = "DiffusionGemmaDecoderTextAttention",
) -> AttentionReplayBinding:
    modules = [module for module in model.modules() if type(module).__name__ == attention_class_name]
    if not modules:
        raise ValueError(f"no {attention_class_name} modules found")
    modeling = importlib.import_module(type(modules[0]).__module__)
    registry = getattr(modeling, "ALL_ATTENTION_FUNCTIONS", None)
    if registry is None or "sdpa" not in registry:
        raise ValueError("model does not expose a registered sdpa attention interface")
    original = registry["sdpa"]
    for module in modules:
        module._diffusion_attention_replay = replay

    def dispatcher(module, query, key, value, attention_mask, **kwargs):
        tagged = getattr(module, "_diffusion_attention_replay", None)
        layer = int(getattr(module, "layer_idx", -1))
        if tagged is None or not tagged._selected(layer):
            return original(module, query, key, value, attention_mask, **kwargs)
        return tagged.attention_forward(module, query, key, value, attention_mask, **kwargs)

    registry["sdpa"] = dispatcher

    def begin_forward(_module, args, kwargs) -> None:
        replay.begin_forward(
            kwargs.get("decoder_input_ids"),
            kwargs.get("self_conditioning_logits"),
        )

    base_model = getattr(model, "model", model)
    hook_handle = base_model.register_forward_pre_hook(begin_forward, with_kwargs=True)
    return AttentionReplayBinding(
        modules=modules,
        registry=registry,
        original=original,
        dispatcher=dispatcher,
        hook_handle=hook_handle,
        replay=replay,
    )
