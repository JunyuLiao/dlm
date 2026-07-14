"""Minimal LLaDA compatibility helpers used by sparse-attention evaluation."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def _normalize_tied_weight_keys(value: object) -> dict[str, None]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple, set)):
        return {str(item): None for item in value}
    return {str(value): None}


@contextmanager
def patch_llada_transformers_compat() -> Iterator[None]:
    """Shim LLaDA remote-code models while loading with newer Transformers."""

    original_getattr = torch.nn.Module.__getattr__
    original_getattribute = torch.nn.Module.__getattribute__

    def patched_getattr(self: torch.nn.Module, name: str) -> object:
        if name == "all_tied_weights_keys":
            tied = self.__dict__.get("_tied_weights_keys", None)
            if tied is None:
                tied = getattr(type(self), "_tied_weights_keys", None)
            return _normalize_tied_weight_keys(tied)
        return original_getattr(self, name)

    def patched_getattribute(self: torch.nn.Module, name: str) -> object:
        attr = original_getattribute(self, name)
        if name != "tie_weights" or not callable(attr):
            return attr

        def tie_weights_compat(*args: object, **kwargs: object) -> object:
            try:
                return attr(*args, **kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                return attr()

        return tie_weights_compat

    torch.nn.Module.__getattr__ = patched_getattr
    torch.nn.Module.__getattribute__ = patched_getattribute  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.Module.__getattr__ = original_getattr
        torch.nn.Module.__getattribute__ = original_getattribute  # type: ignore[method-assign]


def disable_use_cache(model: torch.nn.Module) -> None:
    """Disable autoregressive KV caching for full bidirectional attention."""

    model.config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False


def ensure_all_tied_weights_keys(model: torch.nn.Module) -> None:
    if not hasattr(model, "all_tied_weights_keys"):
        tied = getattr(model, "_tied_weights_keys", None)
        model.all_tied_weights_keys = _normalize_tied_weight_keys(tied)


def choose_mask_token_id(tokenizer: object, explicit_mask_token_id: int | None) -> int:
    env_mask_token_id = os.environ.get("MASK_TOKEN_ID")
    if env_mask_token_id:
        return int(env_mask_token_id)
    if explicit_mask_token_id is not None:
        return explicit_mask_token_id
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        return int(mask_token_id)
    token_id = tokenizer.convert_tokens_to_ids("[MASK]")  # type: ignore[attr-defined]
    if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
        return int(token_id)
    return 126336
