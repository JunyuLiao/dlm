"""Reference implementations of the two RULER16K routing rules.

These routines are intentionally framework-light.  They operate on already
projected post-RoPE Q/K tensors and a structural validity mask, which makes
their semantics directly unit-testable without loading DiffusionGemma.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from dllm.attention.blasst.core import (
    _attention_type,
    _prepare_attention_inputs,
    _prepare_attention_scores,
)

from .config import TILE_SIZE, analytic_beta


def _as_valid_mask(mask: torch.Tensor | None, scores: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    value = mask.to(scores.device)
    if value.dtype == torch.bool:
        valid = value
    elif value.is_floating_point():
        valid = torch.isfinite(value) & (value > -1.0e4)
    else:
        valid = value != 0
    return torch.broadcast_to(valid, scores.shape)


def _structural_validity(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    is_causal: bool | None,
    sliding_window: int | None,
) -> torch.Tensor:
    """Build a boolean validity mask using the same conventions as BLASST."""

    scores = torch.empty(
        query.shape[0], query.shape[1], query.shape[-2], key.shape[-2],
        device=query.device, dtype=query.dtype,
    )
    valid = _as_valid_mask(attention_mask, scores)
    causal = bool(is_causal) if is_causal is not None else (
        attention_mask is None and query.shape[-2] > 1
    )
    if causal:
        q_positions = torch.arange(query.shape[-2], device=query.device) + (
            key.shape[-2] - query.shape[-2]
        )
        k_positions = torch.arange(key.shape[-2], device=query.device)
        valid &= k_positions[None, :] <= q_positions[:, None]
    if sliding_window is not None and (attention_mask is None or attention_mask.ndim < 4):
        q_positions = torch.arange(query.shape[-2], device=query.device) + (
            key.shape[-2] - query.shape[-2]
        )
        k_positions = torch.arange(key.shape[-2], device=query.device)
        valid &= k_positions[None, :] >= q_positions[:, None] - int(sliding_window) + 1
    return valid


def mean_pooled_tile_proxies(
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    *,
    q_tile_size: int = TILE_SIZE,
    kv_tile_size: int = TILE_SIZE,
    scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return literal mean-pooled Q/K tile scores and eligibility.

    Means are taken only over positions that participate in at least one valid
    pair in the tile.  This is important for causal/sliding structural masks:
    padding and impossible positions never influence a proxy.  ``query`` and
    ``key`` must already have the same (expanded) head dimension.
    """

    if query.ndim != 4 or key.ndim != 4 or valid.ndim != 4:
        raise ValueError("query, key, and valid must have shape [batch, heads, length, dim]")
    if query.shape[:2] != key.shape[:2] or valid.shape[:2] != query.shape[:2]:
        raise ValueError("query/key/valid batch and head dimensions must agree")
    if q_tile_size <= 0 or kv_tile_size <= 0:
        raise ValueError("tile sizes must be positive")
    batch, heads, q_len, dim = query.shape
    kv_len = key.shape[-2]
    q_tiles = math.ceil(q_len / q_tile_size)
    kv_tiles = math.ceil(kv_len / kv_tile_size)
    scores = torch.zeros((batch, heads, q_tiles, kv_tiles), dtype=torch.float32, device=query.device)
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    scale = float(dim ** -0.5 if scaling is None else scaling)
    q_float, k_float = query.float(), key.float()
    for qi in range(q_tiles):
        qs, qe = qi * q_tile_size, min((qi + 1) * q_tile_size, q_len)
        tile_valid_q = valid[:, :, qs:qe]
        for ki in range(kv_tiles):
            ks, ke = ki * kv_tile_size, min((ki + 1) * kv_tile_size, kv_len)
            pair_valid = tile_valid_q[..., ks:ke]
            tile_eligible = pair_valid.any(dim=(-2, -1))
            q_participates = pair_valid.any(dim=-1)
            k_participates = pair_valid.any(dim=-2)
            q_count = q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
            k_count = k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
            q_mean = (q_float[:, :, qs:qe] * q_participates[..., None]).sum(dim=-2) / q_count
            k_mean = (k_float[:, :, ks:ke] * k_participates[..., None]).sum(dim=-2) / k_count
            scores[:, :, qi, ki] = (q_mean * k_mean).sum(dim=-1) * scale
            eligible[:, :, qi, ki] = tile_eligible
    return scores, eligible


@dataclass
class SolRoutingResult:
    """Sol-Attn tile mask plus auditable counters."""

    allowed: torch.Tensor
    eligible_tiles: torch.Tensor
    skipped_tiles: torch.Tensor
    proxies: torch.Tensor
    standardized: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def retained_tiles(self) -> torch.Tensor:
        return self.eligible_tiles & ~self.skipped_tiles


def sol_gaussian_route(
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    *,
    target_sparsity: float | None = None,
    beta: float | None = None,
    prefix_length: int = 0,
    q_tile_size: int = TILE_SIZE,
    kv_tile_size: int = TILE_SIZE,
    scaling: float | None = None,
) -> SolRoutingResult:
    """Route all eligible prefix and canvas tiles with one universal beta.

    Standardization is performed over the complete eligible KV population for
    each logical query row (prefix and canvas together).  Ties at ``beta`` are
    retained because the skip predicate is strictly ``z < beta``.  Degenerate
    rows retain every eligible tile; if a non-degenerate threshold would remove
    everything, the maximum-proxy tile is retained as an explicit fallback.
    """

    if target_sparsity is None and beta is None:
        raise ValueError("provide target_sparsity or beta")
    if target_sparsity is not None:
        expected = analytic_beta(float(target_sparsity))
        if beta is None:
            beta = expected
        elif abs(float(beta) - expected) > 2.0e-6:
            raise ValueError("beta does not match the analytic target sparsity")
    beta = float(beta)
    if not 0.0 <= int(prefix_length) <= key.shape[-2]:
        raise ValueError("prefix_length must lie inside the KV sequence")
    proxies, eligible = mean_pooled_tile_proxies(
        query, key, valid, q_tile_size=q_tile_size, kv_tile_size=kv_tile_size, scaling=scaling
    )
    batch, heads, q_tiles, kv_tiles = proxies.shape
    standardized = torch.full_like(proxies, float("nan"))
    skipped = torch.zeros_like(eligible)
    allowed = torch.zeros_like(valid, dtype=torch.bool)
    degenerate_rows = fallback_rows = 0
    prefix_candidate = prefix_skipped = canvas_candidate = canvas_skipped = 0

    for b in range(batch):
        for h in range(heads):
            for qi in range(q_tiles):
                mask = eligible[b, h, qi]
                if not bool(mask.any()):
                    continue
                values = proxies[b, h, qi][mask]
                mean = values.mean()
                std = values.std(unbiased=False)
                if not torch.isfinite(std) or float(std) < 1.0e-8:
                    degenerate_rows += 1
                    z = torch.full_like(proxies[b, h, qi], float("nan"))
                    z[mask] = 0.0
                    keep = mask.clone()
                else:
                    z = torch.full_like(proxies[b, h, qi], float("nan"))
                    z[mask] = (proxies[b, h, qi][mask] - mean) / std
                    keep = mask & ~(z < beta)
                    if not bool(keep.any()):
                        fallback_rows += 1
                        best = torch.argmax(torch.where(mask, proxies[b, h, qi], torch.tensor(-torch.inf, device=proxies.device)))
                        keep[best] = True
                standardized[b, h, qi] = z
                skipped[b, h, qi] = mask & ~keep
                qs, qe = qi * q_tile_size, min((qi + 1) * q_tile_size, query.shape[-2])
                for ki in range(kv_tiles):
                    ks, ke = ki * kv_tile_size, min((ki + 1) * kv_tile_size, key.shape[-2])
                    if bool(keep[ki]):
                        allowed[b, h, qs:qe, ks:ke] |= valid[b, h, qs:qe, ks:ke]
                    if bool(mask[ki]):
                        if ks < prefix_length:
                            prefix_candidate += 1; prefix_skipped += int(bool(skipped[b, h, qi, ki]))
                        else:
                            canvas_candidate += 1; canvas_skipped += int(bool(skipped[b, h, qi, ki]))

    stats = {
        "target_sparsity": None if target_sparsity is None else float(target_sparsity),
        "retained_density_target": 1.0 - float(target_sparsity) if target_sparsity is not None else None,
        "beta": beta,
        "eligible_tiles": int(eligible.sum().item()),
        "skipped_tiles": int(skipped.sum().item()),
        "retained_tiles": int((eligible & ~skipped).sum().item()),
        "degenerate_rows": degenerate_rows,
        "fallback_rows": fallback_rows,
        "routing_rows": int(eligible.any(dim=-1).sum().item()),
        "prefix": {"eligible_tiles": prefix_candidate, "skipped_tiles": prefix_skipped},
        "canvas": {"eligible_tiles": canvas_candidate, "skipped_tiles": canvas_skipped},
    }
    return SolRoutingResult(allowed, eligible, skipped, proxies, standardized, stats)


@dataclass
class BlasstRoutingResult:
    """BLASST row mask and physical-tile diagnostics."""

    allowed: torch.Tensor
    row_skip: torch.Tensor
    physical_skip: torch.Tensor
    eligible_tiles: torch.Tensor
    local_max: torch.Tensor
    margins: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)


def blasst_route(
    scores: torch.Tensor,
    valid: torch.Tensor,
    lambda_value: float,
    *,
    active_rows: torch.Tensor | None = None,
    q_tile_size: int = TILE_SIZE,
    kv_tile_size: int = TILE_SIZE,
    prefix_length: int = 0,
) -> BlasstRoutingResult:
    """Apply BLASST's online-softmax block-max rule.

    ``allowed`` is row-granular.  A physical tile is counted as skipped only
    when every valid active query row in that tile votes to skip it.  The
    resulting counters therefore expose both physical tile sparsity and the
    secondary valid-QK-element sparsity.
    """

    value = float(lambda_value)
    if not 0.0 < value < 1.0:
        raise ValueError("lambda must lie strictly between zero and one")
    if scores.ndim != 4:
        raise ValueError("scores must have shape [batch, heads, query, key]")
    valid = _as_valid_mask(valid, scores)
    batch, heads, q_len, kv_len = scores.shape
    if active_rows is None:
        active = torch.ones((batch, heads, q_len), dtype=torch.bool, device=scores.device)
    else:
        active = active_rows.to(scores.device, dtype=torch.bool)
        if active.ndim == 2:
            active = active[:, None, :]
        if active.shape[-1] != q_len:
            if active.shape[-1] < q_len:
                raise ValueError("active_rows is shorter than the query sequence")
            active = active[..., -q_len:]
        active = torch.broadcast_to(active, (batch, heads, q_len))
    q_tiles, kv_tiles = math.ceil(q_len / q_tile_size), math.ceil(kv_len / kv_tile_size)
    local_max = torch.full((batch, heads, q_len, kv_tiles), -torch.inf, dtype=torch.float32, device=scores.device)
    row_skip_tile = torch.zeros_like(local_max, dtype=torch.bool)
    margins = torch.full_like(local_max, float("nan"))
    physical_skip = torch.zeros((batch, heads, q_tiles, kv_tiles), dtype=torch.bool, device=scores.device)
    eligible_tiles = torch.zeros_like(physical_skip)
    log_lambda = math.log(value)
    scores_float = scores.float()
    for qi in range(q_tiles):
        qs, qe = qi * q_tile_size, min((qi + 1) * q_tile_size, q_len)
        running = torch.full((batch, heads, qe - qs), -torch.inf, dtype=torch.float32, device=scores.device)
        for ki in range(kv_tiles):
            ks, ke = ki * kv_tile_size, min((ki + 1) * kv_tile_size, kv_len)
            pair = valid[:, :, qs:qe, ks:ke] & active[:, :, qs:qe, None]
            has = pair.any(dim=-1)
            maxima = scores_float[:, :, qs:qe, ks:ke].masked_fill(~pair, -torch.inf).amax(dim=-1)
            before = running
            margin = maxima - before
            vote = (~has) | (margin < log_lambda)
            # First valid tile in a row cannot be skipped because before=-inf.
            vote &= has
            row_skip_tile[:, :, qs:qe, ki] = vote
            local_max[:, :, qs:qe, ki] = maxima
            margins[:, :, qs:qe, ki] = margin
            running = torch.maximum(running, maxima)
            valid_rows = has & active[:, :, qs:qe]
            eligible = valid_rows.any(dim=-1)
            eligible_tiles[:, :, qi, ki] = eligible
            physical_skip[:, :, qi, ki] = (vote | ~valid_rows).all(dim=-1) & eligible
    row_skip = row_skip_tile.repeat_interleave(kv_tile_size, dim=-1)[..., :kv_len]
    allowed = valid & active[..., None] & ~row_skip
    active_valid = valid & active[..., None]
    skipped_elements = int((active_valid & row_skip).sum().item())
    valid_elements = int(active_valid.sum().item())
    eligible_count = int(eligible_tiles.sum().item())
    physical_count = int(physical_skip.sum().item())
    prefix_candidate = prefix_skipped = canvas_candidate = canvas_skipped = 0
    for qi in range(q_tiles):
        for ki in range(kv_tiles):
            if not bool(eligible_tiles[:, :, qi, ki].any()):
                continue
            count = int(eligible_tiles[:, :, qi, ki].sum().item())
            skipped_count = int(physical_skip[:, :, qi, ki].sum().item())
            if ki * kv_tile_size < int(prefix_length):
                prefix_candidate += count; prefix_skipped += skipped_count
            else:
                canvas_candidate += count; canvas_skipped += skipped_count
    stats = {
        "lambda": value,
        "log_lambda": log_lambda,
        "eligible_tiles": eligible_count,
        "skipped_tiles": physical_count,
        "retained_tiles": eligible_count - physical_count,
        "physical_tile_sparsity": physical_count / eligible_count if eligible_count else 0.0,
        "skipped_valid_qk_elements": skipped_elements,
        "valid_qk_elements": valid_elements,
        "valid_qk_element_sparsity": skipped_elements / valid_elements if valid_elements else 0.0,
        "prefix": {"eligible_tiles": prefix_candidate, "skipped_tiles": prefix_skipped},
        "canvas": {"eligible_tiles": canvas_candidate, "skipped_tiles": canvas_skipped},
    }
    return BlasstRoutingResult(allowed, row_skip_tile, physical_skip, eligible_tiles, local_max, margins, stats)


def route_attention_call(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    method: str,
    target_sparsity: float = 0.0,
    beta: float | None = None,
    lambda_local: float | None = None,
    lambda_global: float | None = None,
    prefix_length: int | None = None,
    scaling: float | None = None,
    is_causal: bool | None = None,
    sliding_window: int | None = None,
    active_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolRoutingResult | BlasstRoutingResult | None]:
    """Prepare native Q/K inputs and route one attention call."""

    if method == "dense":
        from dllm.attention.blasst.core import _finish_eager_attention

        expanded_key, expanded_value, scores, valid = _prepare_attention_scores(
            module, query, key, value, attention_mask,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window,
        )
        return _finish_eager_attention(query, expanded_value, scores, valid, 0.0, module.training)[0], None
    expanded_key, expanded_value, valid, native_scale = _prepare_attention_inputs(
        module, query, key, value, attention_mask,
        scaling=scaling, is_causal=is_causal, sliding_window=sliding_window,
    )
    prefix = max(0, expanded_key.shape[-2] - query.shape[-2]) if prefix_length is None else int(prefix_length)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * native_scale
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        scores = scores + attention_mask[..., : expanded_key.shape[-2]].to(scores.device)
    scores = scores.masked_fill(~valid, -torch.inf)
    if method == "sol_gaussian":
        result = sol_gaussian_route(
            query, expanded_key, valid, target_sparsity=target_sparsity, beta=beta,
            prefix_length=prefix, scaling=native_scale,
        )
        scores = scores.masked_fill(valid & ~result.allowed, -torch.inf)
        from dllm.attention.blasst.core import _finish_eager_attention
        return _finish_eager_attention(query, expanded_value, scores, result.allowed, 0.0, module.training)[0], result
    if method == "blasst_calibrated":
        result = blasst_route(
            scores, valid,
            lambda_local if _attention_type(module, sliding_window) == "local" else lambda_global,
            active_rows=active_rows, prefix_length=prefix,
        )
        scores = scores.masked_fill(valid & ~result.allowed, -torch.inf)
        from dllm.attention.blasst.core import _finish_eager_attention
        return _finish_eager_attention(query, expanded_value, scores, result.allowed, 0.0, module.training)[0], result
    raise ValueError(f"unknown routing method: {method}")


__all__ = [
    "BlasstRoutingResult",
    "SolRoutingResult",
    "blasst_route",
    "mean_pooled_tile_proxies",
    "route_attention_call",
    "sol_gaussian_route",
]
