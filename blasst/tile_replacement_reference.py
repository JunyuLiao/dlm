"""FP32 correctness reference for replacement-aware physical attention tiles.

The reference keeps every exact and approximate contribution in unnormalized
softmax sufficient-statistic form.  It never subtracts an already-normalized
attention output and it composes multiple replacements simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .oracle_mass_pruning import (
    BlasstTileDecisions,
    ExactTileStatistics,
    collect_exact_tile_statistics,
)


ReplacementScope = Literal["veto_only", "all_rows"]


@dataclass(frozen=True)
class ReplacementComponents:
    """Approximate tile contributions in absolute log-mass form.

    Shapes are ``[B,H,T,R]`` for ``log_z`` and ``rows`` and
    ``[B,H,T,R,D]`` for ``conditional_value``.  Entries where ``rows`` is
    false are ignored.
    """

    log_z: torch.Tensor
    conditional_value: torch.Tensor
    rows: torch.Tensor

    def __post_init__(self) -> None:
        if self.log_z.ndim != 4 or self.rows.shape != self.log_z.shape:
            raise ValueError("log_z and rows must have shape [batch, heads, tiles, rows]")
        if (
            self.conditional_value.ndim != 5
            or self.conditional_value.shape[:-1] != self.log_z.shape
        ):
            raise ValueError(
                "conditional_value must have shape [batch, heads, tiles, rows, dim]"
            )


@dataclass(frozen=True)
class ComposedAttention:
    output: torch.Tensor
    log_denominator: torch.Tensor
    scaled_numerator: torch.Tensor
    reference_maximum: torch.Tensor


def collect_tile_replacement_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kv_block_size: int = 64,
    causal_query_offset: int | None = None,
) -> ExactTileStatistics:
    """Collect exact per-tile ``(m,l,u)`` with an FP32 teacher pass.

    ``log Z = m + log(l)``, ``U = u/l``, and the absolute numerator is
    ``N = exp(m) * u``.  The centered representation avoids overflow while
    retaining exactly the requested Z/U/N information.
    """

    return collect_exact_tile_statistics(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        kv_block_size=kv_block_size,
        causal_query_offset=causal_query_offset,
    )


def exact_log_z(statistics: ExactTileStatistics) -> torch.Tensor:
    """Return absolute per-tile ``log Z`` for every query row."""

    return torch.where(
        statistics.l > 0,
        statistics.m + statistics.l.clamp_min(1e-30).log(),
        torch.full_like(statistics.m, -torch.inf),
    )


def exact_conditional_value(statistics: ExactTileStatistics) -> torch.Tensor:
    """Return exact per-tile conditional value ``U``."""

    return statistics.u / statistics.l.clamp_min(1e-30)[..., None]


def replacement_rows(
    decisions: BlasstTileDecisions,
    replaced_tiles: torch.Tensor,
    scope: ReplacementScope,
) -> torch.Tensor:
    """Expand physical replacement decisions to their intended query rows."""

    if replaced_tiles.shape != decisions.keep.shape:
        raise ValueError("replaced_tiles must have shape [batch, heads, tiles]")
    if bool((replaced_tiles & ~decisions.keep).any()):
        raise ValueError("only ordinary-BLASST-kept tiles can be replaced")
    physical = replaced_tiles[..., None]
    if scope == "all_rows":
        return physical.expand_as(decisions.veto_rows)
    if scope == "veto_only":
        return physical & decisions.veto_rows
    raise ValueError(f"unknown replacement scope: {scope}")


def exact_replacement_components(
    statistics: ExactTileStatistics,
    decisions: BlasstTileDecisions,
    replaced_tiles: torch.Tensor,
    *,
    scope: ReplacementScope,
) -> ReplacementComponents:
    """Build an exact replacement, useful as the Oracle-A identity check."""

    rows = replacement_rows(decisions, replaced_tiles, scope)
    return ReplacementComponents(
        log_z=exact_log_z(statistics),
        conditional_value=exact_conditional_value(statistics),
        rows=rows,
    )


def compose_tile_replacements(
    statistics: ExactTileStatistics,
    active_exact_tiles: torch.Tensor,
    replacements: ReplacementComponents | None = None,
) -> torch.Tensor:
    """Compose remaining exact tiles and all approximate tiles at once.

    ``active_exact_tiles`` is physical and has shape ``[B,H,T]``.  Replacement
    rows are row-specific, because a veto-only mechanism replaces the few
    vetoing rows and drops the contribution for the remaining rows.
    """

    return compose_replacement_state(statistics, active_exact_tiles, replacements).output


def compose_replacement_state(
    statistics: ExactTileStatistics,
    active_exact_tiles: torch.Tensor,
    replacements: ReplacementComponents | None = None,
) -> ComposedAttention:
    """Compose replacements and retain denominator/numerator diagnostics."""

    if active_exact_tiles.shape != statistics.m.shape[:-1]:
        raise ValueError("active_exact_tiles must have shape [batch, heads, tiles]")
    exact_active = active_exact_tiles[..., None] & (statistics.l > 0)
    exact_reference = torch.where(exact_active, statistics.m, -torch.inf).amax(dim=2)
    if replacements is None:
        replacement_reference = torch.full_like(exact_reference, -torch.inf)
    else:
        if replacements.log_z.shape != statistics.m.shape:
            raise ValueError("replacement shape does not match exact statistics")
        replacement_reference = torch.where(
            replacements.rows,
            replacements.log_z,
            -torch.inf,
        ).amax(dim=2)
    reference = torch.maximum(exact_reference, replacement_reference)
    if bool((~torch.isfinite(reference)).any()):
        raise ValueError("every row must retain an exact or replacement contribution")

    exact_scale = torch.where(
        exact_active,
        torch.exp(statistics.m - reference[:, :, None]),
        0.0,
    )
    denominator = (exact_scale * statistics.l).sum(dim=2)
    numerator = (exact_scale[..., None] * statistics.u).sum(dim=2)

    if replacements is not None:
        replacement_scale = torch.where(
            replacements.rows,
            torch.exp(replacements.log_z - reference[:, :, None]),
            0.0,
        )
        denominator = denominator + replacement_scale.sum(dim=2)
        numerator = numerator + (
            replacement_scale[..., None] * replacements.conditional_value.float()
        ).sum(dim=2)
    output = numerator / denominator.clamp_min(1e-30)[..., None]
    return ComposedAttention(
        output=output,
        log_denominator=reference + denominator.clamp_min(1e-30).log(),
        scaled_numerator=numerator,
        reference_maximum=reference,
    )


def replace_physical_tiles(
    statistics: ExactTileStatistics,
    decisions: BlasstTileDecisions,
    replaced_tiles: torch.Tensor,
    replacements: ReplacementComponents,
) -> torch.Tensor:
    """Remove selected physical tiles and insert supplied row replacements."""

    if bool((replaced_tiles & ~decisions.keep).any()):
        raise ValueError("replacement is only defined for kept physical tiles")
    return compose_tile_replacements(
        statistics,
        decisions.keep & ~replaced_tiles,
        replacements,
    )
