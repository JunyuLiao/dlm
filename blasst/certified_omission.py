"""Reference primitives for causal, certified pre-QK tile omission.

The certificate propagates an interval for each physical tile's row-wise
log-normalizer.  It uses only Q/K changes and previously certified intervals;
it never reads the current tile's QK scores to make an omission decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogNormalizerInterval:
    """Lower and upper bounds with shape ``[..., kv_tiles, query_rows]``."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError("interval bounds must have identical shapes")
        if torch.any(self.lower > self.upper):
            raise ValueError("interval lower bound exceeds upper bound")

    @classmethod
    def exact(cls, log_normalizer: torch.Tensor) -> "LogNormalizerInterval":
        return cls(log_normalizer, log_normalizer)


def temporal_score_perturbation_bound(
    current_q: torch.Tensor,
    previous_q: torch.Tensor,
    current_k_tiles: torch.Tensor,
    previous_k_tiles: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Bound every row/tile score change without forming a QK matrix.

    Q tensors have shape ``[..., query_rows, dim]`` and tiled K tensors have
    shape ``[..., kv_tiles, kv_rows, dim]``.  The result has shape
    ``[..., kv_tiles, query_rows]``.
    """

    if current_q.shape != previous_q.shape:
        raise ValueError("current and previous Q shapes differ")
    if current_k_tiles.shape != previous_k_tiles.shape:
        raise ValueError("current and previous K shapes differ")
    if current_q.shape[:-2] != current_k_tiles.shape[:-3]:
        raise ValueError("Q and K prefix dimensions differ")
    if current_q.shape[-1] != current_k_tiles.shape[-1]:
        raise ValueError("Q and K head dimensions differ")
    dq = torch.linalg.vector_norm(current_q - previous_q, dim=-1)
    q0 = torch.linalg.vector_norm(previous_q, dim=-1)
    k0 = torch.linalg.vector_norm(previous_k_tiles, dim=-1).amax(-1)
    dk = torch.linalg.vector_norm(current_k_tiles - previous_k_tiles, dim=-1).amax(-1)
    return float(scale) * (
        dq[..., None, :] * k0[..., :, None]
        + q0[..., None, :] * dk[..., :, None]
        + dq[..., None, :] * dk[..., :, None]
    )


def advance_log_normalizer_interval(
    previous: LogNormalizerInterval,
    score_perturbation: torch.Tensor,
) -> LogNormalizerInterval:
    """Propagate a valid log-normalizer interval through one Q/K change."""

    if previous.lower.shape != score_perturbation.shape:
        raise ValueError("interval and perturbation shapes differ")
    if torch.any(score_perturbation < 0):
        raise ValueError("score perturbation bounds must be non-negative")
    return LogNormalizerInterval(
        previous.lower - score_perturbation,
        previous.upper + score_perturbation,
    )


def certified_tile_mass_upper(interval: LogNormalizerInterval) -> torch.Tensor:
    """Upper-bound each tile's current softmax mass for every query row."""

    denominator_lower = torch.logsumexp(interval.lower, dim=-2, keepdim=True)
    # A mass cannot exceed one.  Clamping also prevents exp overflow for very
    # stale intervals without weakening the upper-bound property.
    return torch.exp((interval.upper - denominator_lower).clamp(max=0.0))


def greedy_mass_budget_omission(
    mass_upper: torch.Tensor,
    budget: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select low-bound tiles while respecting every row's aggregate budget.

    Returns a tile mask with shape ``[..., kv_tiles]`` and the accumulated
    row-wise upper bound with shape ``[..., query_rows]``.
    """

    if mass_upper.ndim < 2:
        raise ValueError("mass_upper needs tile and query-row dimensions")
    if budget < 0 or budget >= 1:
        raise ValueError("budget must be in [0, 1)")
    tile_count = mass_upper.shape[-2]
    prefix = mass_upper.shape[:-2]
    flattened = mass_upper.reshape(-1, tile_count, mass_upper.shape[-1])
    order = torch.argsort(flattened.amax(-1), dim=-1, stable=True)
    selected = torch.zeros(
        flattened.shape[:-1], dtype=torch.bool, device=mass_upper.device
    )
    accumulated = torch.zeros(
        (flattened.shape[0], flattened.shape[-1]),
        dtype=mass_upper.dtype,
        device=mass_upper.device,
    )
    batches = torch.arange(flattened.shape[0], device=mass_upper.device)
    for rank in range(tile_count):
        tile = order[:, rank]
        candidate = flattened[batches, tile]
        accept = (accumulated + candidate <= budget).all(-1)
        selected[batches[accept], tile[accept]] = True
        accumulated = torch.where(accept[:, None], accumulated + candidate, accumulated)
    return selected.reshape(*prefix, tile_count), accumulated.reshape(
        *prefix, mass_upper.shape[-1]
    )


def refresh_log_normalizer_interval(
    propagated: LogNormalizerInterval,
    exact_current: torch.Tensor,
    omitted: torch.Tensor,
) -> LogNormalizerInterval:
    """Keep propagated bounds for omitted tiles and reset recomputed tiles."""

    if propagated.lower.shape != exact_current.shape:
        raise ValueError("propagated and exact tensors differ")
    if omitted.shape != exact_current.shape[:-1]:
        raise ValueError("omission mask shape differs")
    keep_interval = omitted[..., None]
    return LogNormalizerInterval(
        torch.where(keep_interval, propagated.lower, exact_current),
        torch.where(keep_interval, propagated.upper, exact_current),
    )


def attention_output_after_omission(
    m: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    omitted: torch.Tensor,
) -> torch.Tensor:
    """Compose current tile statistics after removing certified tiles."""

    if m.shape != l.shape or u.shape[:-1] != m.shape:
        raise ValueError("incompatible (m,l,u) shapes")
    if omitted.shape != m.shape[:-1]:
        raise ValueError("omission mask must omit complete physical tiles")
    if torch.any(omitted.all(-1)):
        raise ValueError("cannot omit every KV tile")
    active = ~omitted[..., None]
    active_m = torch.where(active, m, torch.full_like(m, -torch.inf))
    global_m = active_m.amax(-2)
    weights = torch.where(active, torch.exp(m - global_m[..., None, :]), 0.0)
    total_l = (weights * l).sum(-2)
    total_u = (weights[..., None] * u).sum(-3)
    return total_u / total_l.clamp_min(1e-30)[..., None]
