"""Model integration and metric collection for SPA oracle experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from .reference import (
    SPAConfig,
    SPASupport,
    build_oracle_support,
    build_oracle_support_from_probabilities,
    evaluate_support,
    exact_shared_support_attention,
)


def config_name(config: SPAConfig) -> str:
    selector = config.selector.replace("oracle-", "")
    extras = []
    if config.selector == "oracle-tail":
        extras.append(f"b{config.tail_beta:g}q{config.tail_quantile:g}")
    if config.selector == "oracle-coverage":
        extras.append(config.coverage_objective)
    if config.mandatory_local:
        extras.append("local")
    if config.mandatory_first:
        extras.append("first")
    if config.mandatory_last:
        extras.append("last")
    if config.sink_pages:
        extras.append(f"sink{config.sink_pages}")
    if config.share_heads_experimental:
        extras.append("sharedheads")
    suffix = "-" + "-".join(extras) if extras else ""
    return f"g{config.group_size}-{config.support}-d{config.density:g}-{selector}{suffix}"


def _oracle_sparse_output(
    probabilities: torch.Tensor,
    v: torch.Tensor,
    support: SPASupport,
) -> torch.Tensor:
    """Equivalent conditional output used only to accelerate Phase-A sweeps."""
    query_length = probabilities.shape[2]
    group_ids = torch.arange(query_length, device=probabilities.device) // support.group_size
    row_support = support.key_mask[:, :, group_ids, :]
    selected = probabilities * row_support
    selected = selected / selected.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.einsum("bhqk,bkhd->bqhd", selected, v.float()).to(v.dtype)


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = (left & right).sum(-1).float()
    union = (left | right).sum(-1).float()
    return float(torch.where(union > 0, intersection / union, torch.ones_like(union)).mean())


def _adjacent_group_jaccard(support: SPASupport) -> float:
    if support.key_mask.shape[2] < 2:
        return 1.0
    return _jaccard(support.key_mask[:, :, :-1], support.key_mask[:, :, 1:])


def _row_mass_summary(
    probabilities: torch.Tensor,
    support: SPASupport,
    rows: torch.Tensor,
) -> dict[str, float] | None:
    group_ids = torch.arange(probabilities.shape[2], device=probabilities.device) // support.group_size
    retained = (probabilities * support.key_mask[:, :, group_ids]).sum(-1)
    selected = rows[:, None, :].expand(-1, probabilities.shape[1], -1)
    values = retained[selected]
    if not values.numel():
        return None
    return {
        "mean": float(values.mean()),
        "p1": float(torch.quantile(values, 0.01)),
        "p5": float(torch.quantile(values, 0.05)),
        "minimum": float(values.min()),
    }


def _row_output_error(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    rows: torch.Tensor,
) -> dict[str, float] | None:
    selected = rows[:, :, None, None].expand_as(candidate)
    if not bool(selected.any()):
        return None
    left = candidate.float()[selected]
    right = reference.float()[selected]
    difference = left - right
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(right).clamp_min(1e-30)
        ),
        "cosine_similarity": float(F.cosine_similarity(left, right, dim=0)),
        "maximum_absolute_error": float(difference.abs().max()),
    }


class SPAOracleAnalyzer:
    """Collect per-layer oracle metrics while preserving dense model output."""

    def __init__(self, configs: Sequence[SPAConfig]) -> None:
        if not configs:
            raise ValueError("at least one SPA configuration is required")
        names = [config_name(config) for config in configs]
        if len(names) != len(set(names)):
            raise ValueError("SPA configurations must have unique names")
        self.configs = tuple(configs)
        self.records: list[dict[str, object]] = []
        self.step = 0
        self.remaining_mask_ratio = 1.0
        self.masked_rows: torch.Tensor | None = None
        self.request_ids: tuple[str, ...] = ()
        self._previous_step: dict[tuple[int, str], torch.Tensor] = {}
        self._current_step: dict[tuple[int, str], torch.Tensor] = {}
        self._previous_layer: dict[str, torch.Tensor] = {}

    def set_step(
        self,
        *,
        step: int,
        remaining_mask_ratio: float,
        masked_rows: torch.Tensor,
        request_ids: Sequence[str],
    ) -> None:
        if masked_rows.ndim != 2 or masked_rows.shape[0] != len(request_ids):
            raise ValueError("masked rows must be [request, sequence]")
        self.step = step
        self.remaining_mask_ratio = remaining_mask_ratio
        self.masked_rows = masked_rows
        self.request_ids = tuple(request_ids)
        self._previous_layer = {}
        self._current_step = {}

    def finish_step(self) -> None:
        self._previous_step = self._current_step

    def analyze_layer(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dense_output: torch.Tensor,
        *,
        causal: bool = False,
        softmax_scale: float | None = None,
    ) -> None:
        if self.masked_rows is None:
            raise RuntimeError("set_step must be called before model execution")
        first_support, probabilities = build_oracle_support(
            q, k, self.configs[0], causal=causal, softmax_scale=softmax_scale
        )
        for index, config in enumerate(self.configs):
            name = config_name(config)
            support = (
                first_support
                if index == 0
                else build_oracle_support_from_probabilities(probabilities, config)
            )
            sparse_output = _oracle_sparse_output(probabilities, v, support)
            metrics = evaluate_support(support, probabilities, sparse_output, dense_output)
            mask = support.key_mask.detach().cpu()
            previous_step = self._previous_step.get((layer, name))
            previous_layer = self._previous_layer.get(name)
            row: dict[str, object] = {
                "step": self.step,
                "remaining_mask_ratio": self.remaining_mask_ratio,
                "layer": layer,
                "config": name,
                "parameters": asdict(config),
                **metrics.as_dict(),
                "adjacent_group_support_jaccard": _adjacent_group_jaccard(support),
                "previous_step_support_jaccard": (
                    None if previous_step is None else _jaccard(mask, previous_step)
                ),
                "previous_layer_support_jaccard": (
                    None if previous_layer is None else _jaccard(mask, previous_layer)
                ),
                "masked_retained_mass": _row_mass_summary(
                    probabilities, support, self.masked_rows.to(q.device)
                ),
                "revealed_retained_mass": _row_mass_summary(
                    probabilities, support, ~self.masked_rows.to(q.device)
                ),
                "masked_output_error": _row_output_error(
                    sparse_output, dense_output, self.masked_rows.to(q.device)
                ),
                "revealed_output_error": _row_output_error(
                    sparse_output, dense_output, ~self.masked_rows.to(q.device)
                ),
            }
            self.records.append(row)
            self._current_step[(layer, name)] = mask
            self._previous_layer[name] = mask


@contextmanager
def install_spa_oracle_analyzer(
    model: torch.nn.Module,
    analyzer: SPAOracleAnalyzer,
) -> Iterator[None]:
    """Wrap model FlashAttention calls for read-only Phase-A analysis."""
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    originals = [(module, module.flash_attn_func) for module in modules]
    for layer, (module, original) in enumerate(originals):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer: int = layer,
            _original=original,
            **kwargs: object,
        ) -> torch.Tensor:
            dense = _original(q, k, v, **kwargs)
            analyzer.analyze_layer(
                _layer,
                q,
                k,
                v,
                dense,
                causal=bool(kwargs.get("causal", False)),
                softmax_scale=kwargs.get("softmax_scale"),  # type: ignore[arg-type]
            )
            return dense

        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original


class SPAExactController:
    """Per-layer exact current-step SPA with optional fixed support reuse."""

    def __init__(self, config: SPAConfig, *, refresh_steps: set[int] | None = None) -> None:
        self.config = config
        self.refresh_steps = refresh_steps
        self.step = 0
        self.supports: dict[int, SPASupport] = {}
        self.attention_records: list[dict[str, object]] = []

    def set_step(self, step: int) -> None:
        self.step = step

    def apply(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        refresh = (
            layer not in self.supports
            or self.refresh_steps is None
            or self.step in self.refresh_steps
        )
        if refresh:
            support, probabilities = build_oracle_support(
                q, k, self.config, causal=causal, softmax_scale=softmax_scale
            )
            self.supports[layer] = support
        else:
            support = self.supports[layer]
            _, probabilities = build_oracle_support(
                q, k, self.config, causal=causal, softmax_scale=softmax_scale
            )
        sparse = exact_shared_support_attention(
            q, k, v, support, causal=causal, softmax_scale=softmax_scale
        )
        dense = torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float()).to(q.dtype)
        metrics = evaluate_support(support, probabilities, sparse, dense)
        self.attention_records.append(
            {
                "step": self.step,
                "layer": layer,
                "refreshed": refresh,
                **metrics.as_dict(),
            }
        )
        return sparse


@contextmanager
def install_exact_spa(
    model: torch.nn.Module,
    controller: SPAExactController,
) -> Iterator[None]:
    """Replace attention outputs with the exact selected-key SPA reference."""
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    originals = [(module, module.flash_attn_func) for module in modules]
    for layer, (module, original) in enumerate(originals):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer: int = layer,
            **kwargs: object,
        ) -> torch.Tensor:
            return controller.apply(
                _layer,
                q,
                k,
                v,
                causal=bool(kwargs.get("causal", False)),
                softmax_scale=kwargs.get("softmax_scale"),  # type: ignore[arg-type]
            )

        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = candidate.float() - reference.float()
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
        "cosine_similarity": float(
            F.cosine_similarity(candidate.float().flatten(), reference.float().flatten(), dim=0)
        ),
        "maximum_absolute_error": float(difference.abs().max()),
    }
