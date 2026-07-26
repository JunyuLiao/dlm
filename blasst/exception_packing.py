"""Offline feasibility accounting for cross-tile BLASST exception packing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch


GroupingScope = Literal["q-head", "kv-head"]


@dataclass(frozen=True)
class PackabilityMetrics:
    tau: int
    pack_m: int
    grouping: GroupingScope
    candidate_tiles: int
    retained_tiles: int
    all_tiles: int
    exception_rows: int
    grouping_keys: int
    complete_packs: int
    partial_packs: int
    total_packs: int
    full_pack_rows: int
    partial_pack_rows: int
    padded_rows: int
    weighted_utilization: float
    complete_pack_row_fraction: float
    partial_pack_row_fraction: float
    original_v_loads: int
    packed_v_loads: int
    v_load_reduction: float
    v_load_factor: float
    v_bytes_before: int
    v_bytes_after: int
    average_rows_per_original_v_load: float
    average_rows_per_packed_v_load: float
    extra_qk_flops: int
    baseline_qk_flops: int
    candidate_main_pv_flops: int
    packed_pv_flops: int
    pv_flops_avoided: int
    extra_qk_fraction_of_baseline: float
    extra_qk_fraction_of_pv_avoided: float | None
    q_read_bytes: int
    tile_record_bytes: int
    expanded_descriptor_bytes: int
    count_offset_bytes: int
    pack_descriptor_bytes: int
    total_metadata_bytes: int

    @property
    def zero_tile_fraction(self) -> float:
        return 1.0 - self.retained_tiles / self.all_tiles if self.all_tiles else 0.0

    @property
    def candidate_fraction_of_retained(self) -> float:
        return self.candidate_tiles / self.retained_tiles if self.retained_tiles else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "zero_tile_fraction": self.zero_tile_fraction,
            "candidate_fraction_of_retained": self.candidate_fraction_of_retained,
        }


def active_counts(keep_mask: torch.Tensor, query_tile_size: int = 128) -> torch.Tensor:
    """Return active row-vote counts shaped ``[Q heads, Q tiles, KV tiles]``."""
    if keep_mask.ndim != 3:
        raise ValueError("keep_mask must be [query heads, rows, KV tiles]")
    heads, rows, kv_tiles = keep_mask.shape
    q_tiles = math.ceil(rows / query_tile_size)
    padded = q_tiles * query_tile_size
    keep = keep_mask.bool()
    if padded != rows:
        keep = torch.nn.functional.pad(keep, (0, 0, 0, padded - rows))
    return keep.reshape(heads, q_tiles, query_tile_size, kv_tiles).sum(2)


def grouped_exception_totals(
    counts: torch.Tensor,
    tau: int,
    *,
    grouping: GroupingScope,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exception-row totals and source-tile counts for nonempty keys."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if counts.ndim != 3:
        raise ValueError("counts must be [query heads, Q tiles, KV tiles]")
    query_heads, _, _ = counts.shape
    if num_kv_heads <= 0 or query_heads % num_kv_heads:
        raise ValueError("num_kv_heads must evenly divide query heads")
    candidate = (counts > 0) & (counts <= tau)
    candidate_rows = torch.where(candidate, counts, 0)
    if grouping == "q-head":
        totals = candidate_rows.sum(1)
        source_tiles = candidate.sum(1)
    elif grouping == "kv-head":
        q_per_kv = query_heads // num_kv_heads
        totals = candidate_rows.reshape(num_kv_heads, q_per_kv, *counts.shape[1:]).sum((1, 2))
        source_tiles = candidate.reshape(num_kv_heads, q_per_kv, *counts.shape[1:]).sum((1, 2))
    else:
        raise ValueError(f"unknown grouping scope: {grouping}")
    selected = totals > 0
    return totals[selected].to(torch.int64), source_tiles[selected].to(torch.int64)


def exception_tiles_per_row_histogram(
    keep_mask: torch.Tensor,
    counts: torch.Tensor,
    tau: int,
    query_tile_size: int = 128,
) -> torch.Tensor:
    """Histogram of candidate exception KV tiles touching each output row."""
    heads, rows, kv_tiles = keep_mask.shape
    q_tiles = counts.shape[1]
    padded = q_tiles * query_tile_size
    keep = keep_mask.bool()
    if padded != rows:
        keep = torch.nn.functional.pad(keep, (0, 0, 0, padded - rows))
    tiled = keep.reshape(heads, q_tiles, query_tile_size, kv_tiles)
    candidate = ((counts > 0) & (counts <= tau))[:, :, None, :]
    per_row = (tiled & candidate).sum(-1).reshape(heads, padded)[:, :rows]
    return torch.bincount(per_row.flatten(), minlength=kv_tiles + 1)


def summarize_packability(
    counts: torch.Tensor,
    *,
    tau: int,
    pack_m: int,
    grouping: GroupingScope,
    num_kv_heads: int,
    sequence_length: int,
    query_tile_size: int = 128,
    kv_tile_size: int = 64,
    head_dim: int = 128,
    dtype_bytes: int = 2,
    tile_record_size: int = 40,
    expanded_descriptor_size: int = 32,
    pack_descriptor_size: int = 32,
) -> PackabilityMetrics:
    if pack_m not in (64, 128):
        raise ValueError("pack_m must be 64 or 128")
    totals, source_tiles = grouped_exception_totals(
        counts, tau, grouping=grouping, num_kv_heads=num_kv_heads
    )
    candidate = (counts > 0) & (counts <= tau)
    all_tiles = counts.numel()
    retained_tiles = int((counts > 0).sum())
    candidate_tiles = int(candidate.sum())
    exception_rows = int(totals.sum())
    grouping_keys = totals.numel()
    complete = torch.div(totals, pack_m, rounding_mode="floor")
    leftovers = totals.remainder(pack_m)
    partial = leftovers > 0
    complete_packs = int(complete.sum())
    partial_packs = int(partial.sum())
    total_packs = complete_packs + partial_packs
    full_rows = complete_packs * pack_m
    partial_rows = int(leftovers.sum())
    padded_rows = total_packs * pack_m - exception_rows
    original_v_loads = int(source_tiles.sum())
    packed_v_loads = total_packs
    v_tile_bytes = kv_tile_size * head_dim * dtype_bytes
    baseline_qk = 2 * counts.shape[0] * sequence_length * sequence_length * head_dim
    extra_qk = 2 * exception_rows * kv_tile_size * head_dim
    candidate_main_pv = 2 * candidate_tiles * query_tile_size * kv_tile_size * head_dim
    packed_pv = 2 * total_packs * pack_m * kv_tile_size * head_dim
    avoided = candidate_main_pv - packed_pv
    tile_bytes = candidate_tiles * tile_record_size
    row_bytes = exception_rows * expanded_descriptor_size
    count_offset_bytes = grouping_keys * 2 * 4
    pack_bytes = total_packs * pack_descriptor_size
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return PackabilityMetrics(
        tau=tau,
        pack_m=pack_m,
        grouping=grouping,
        candidate_tiles=candidate_tiles,
        retained_tiles=retained_tiles,
        all_tiles=all_tiles,
        exception_rows=exception_rows,
        grouping_keys=grouping_keys,
        complete_packs=complete_packs,
        partial_packs=partial_packs,
        total_packs=total_packs,
        full_pack_rows=full_rows,
        partial_pack_rows=partial_rows,
        padded_rows=padded_rows,
        weighted_utilization=ratio(exception_rows, total_packs * pack_m),
        complete_pack_row_fraction=ratio(full_rows, exception_rows),
        partial_pack_row_fraction=ratio(partial_rows, exception_rows),
        original_v_loads=original_v_loads,
        packed_v_loads=packed_v_loads,
        v_load_reduction=1.0 - ratio(packed_v_loads, original_v_loads),
        v_load_factor=ratio(original_v_loads, packed_v_loads),
        v_bytes_before=original_v_loads * v_tile_bytes,
        v_bytes_after=packed_v_loads * v_tile_bytes,
        average_rows_per_original_v_load=ratio(exception_rows, original_v_loads),
        average_rows_per_packed_v_load=ratio(exception_rows, packed_v_loads),
        extra_qk_flops=extra_qk,
        baseline_qk_flops=baseline_qk,
        candidate_main_pv_flops=candidate_main_pv,
        packed_pv_flops=packed_pv,
        pv_flops_avoided=avoided,
        extra_qk_fraction_of_baseline=ratio(extra_qk, baseline_qk),
        extra_qk_fraction_of_pv_avoided=extra_qk / avoided if avoided > 0 else None,
        q_read_bytes=exception_rows * head_dim * dtype_bytes,
        tile_record_bytes=tile_bytes,
        expanded_descriptor_bytes=row_bytes,
        count_offset_bytes=count_offset_bytes,
        pack_descriptor_bytes=pack_bytes,
        total_metadata_bytes=tile_bytes + row_bytes + count_offset_bytes + pack_bytes,
    )
