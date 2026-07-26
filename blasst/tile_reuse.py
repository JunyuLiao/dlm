"""Reference primitives for composable attention-tile result caching.

This module is intentionally independent from the optimized BLASST kernels. It
provides an exact, easy-to-audit implementation used to validate whether cached
physical-tile sufficient statistics can be mixed with freshly computed tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionTileStatistics:
    """Unnormalized softmax sufficient statistics for one KV tile.

    ``m`` and ``l`` have shape ``[..., query_rows]`` and ``u`` has shape
    ``[..., query_rows, value_dim]``. All tensors are accumulated in float32.
    """

    m: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor

    def normalized_output(self) -> torch.Tensor:
        return self.u / self.l.clamp_min(1e-30)[..., None]


def attention_tile_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
) -> AttentionTileStatistics:
    """Compute exact local ``(m, l, u)`` for one query/KV tile."""

    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have token and feature dimensions")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v prefix dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("incompatible q, k, and v tile shapes")
    qf, kf, vf = q.float(), k.float(), v.float()
    factor = float(scale) if scale is not None else q.shape[-1] ** -0.5
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * factor
    local_max = scores.amax(dim=-1)
    probabilities = torch.exp(scores - local_max[..., None])
    local_sum = probabilities.sum(dim=-1)
    local_value = torch.matmul(probabilities, vf)
    return AttentionTileStatistics(local_max, local_sum, local_value)


def merge_attention_tile_statistics(
    a: AttentionTileStatistics,
    b: AttentionTileStatistics,
) -> AttentionTileStatistics:
    """Compose two independently normalized tile components exactly."""

    if a.m.shape != b.m.shape or a.l.shape != b.l.shape or a.u.shape != b.u.shape:
        raise ValueError("attention tile statistics must have identical shapes")
    merged_max = torch.maximum(a.m, b.m)
    a_scale = torch.exp(a.m - merged_max)
    b_scale = torch.exp(b.m - merged_max)
    merged_sum = a_scale * a.l + b_scale * b.l
    merged_value = a_scale[..., None] * a.u + b_scale[..., None] * b.u
    return AttentionTileStatistics(merged_max, merged_sum, merged_value)


def compose_attention_tile_statistics(
    tiles: Iterable[AttentionTileStatistics],
) -> AttentionTileStatistics:
    """Compose a non-empty iterable of fresh and/or cached tile components."""

    iterator = iter(tiles)
    try:
        result = next(iterator)
    except StopIteration as error:
        raise ValueError("at least one attention tile is required") from error
    for tile in iterator:
        result = merge_attention_tile_statistics(result, tile)
    return result


def tiled_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = 128,
    kv_block_size: int = 64,
    scale: float | None = None,
    return_tiles: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[list[AttentionTileStatistics]]]:
    """Dense attention reconstructed solely from composable tile statistics."""

    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v prefix dimensions must match")
    outputs: list[torch.Tensor] = []
    all_tiles: list[list[AttentionTileStatistics]] = []
    factor = float(scale) if scale is not None else q.shape[-1] ** -0.5
    for q_start in range(0, q.shape[-2], q_block_size):
        q_tile = q[..., q_start : q_start + q_block_size, :]
        tiles = [
            attention_tile_statistics(
                q_tile,
                k[..., k_start : k_start + kv_block_size, :],
                v[..., k_start : k_start + kv_block_size, :],
                scale=factor,
            )
            for k_start in range(0, k.shape[-2], kv_block_size)
        ]
        outputs.append(compose_attention_tile_statistics(tiles).normalized_output())
        if return_tiles:
            all_tiles.append(tiles)
    output = torch.cat(outputs, dim=-2).to(q.dtype)
    return (output, all_tiles) if return_tiles else output


@dataclass(frozen=True)
class TileCacheFootprint:
    batch_size: int
    tiles: int
    bytes_per_tile: int
    total_bytes: int

    @property
    def gib(self) -> float:
        return self.total_bytes / 2**30


def two_dimensional_cache_footprint(
    *,
    batch_size: int,
    sequence_length: int = 4096,
    layers: int = 32,
    heads: int = 32,
    query_tile_size: int = 128,
    kv_tile_size: int = 64,
    head_dim: int = 128,
    ml_bytes: int = 4,
    u_bytes: int = 2,
) -> TileCacheFootprint:
    """Storage for cached per-physical-tile ``m``, ``l``, and ``u``."""

    query_tiles = math.ceil(sequence_length / query_tile_size)
    kv_tiles = math.ceil(sequence_length / kv_tile_size)
    tiles = batch_size * layers * heads * query_tiles * kv_tiles
    bytes_per_tile = query_tile_size * (2 * ml_bytes + head_dim * u_bytes)
    return TileCacheFootprint(batch_size, tiles, bytes_per_tile, tiles * bytes_per_tile)


def query_tile_output_cache_footprint(
    *,
    batch_size: int,
    sequence_length: int = 4096,
    layers: int = 32,
    heads: int = 32,
    query_tile_size: int = 128,
    head_dim: int = 128,
    output_bytes: int = 2,
) -> TileCacheFootprint:
    """Storage for caching only the final output of each complete query tile."""

    query_tiles = math.ceil(sequence_length / query_tile_size)
    tiles = batch_size * layers * heads * query_tiles
    bytes_per_tile = query_tile_size * head_dim * output_bytes
    return TileCacheFootprint(batch_size, tiles, bytes_per_tile, tiles * bytes_per_tile)
