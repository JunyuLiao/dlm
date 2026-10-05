"""Standalone uniform-mask Z center/radius builder; never wired into generation."""
from __future__ import annotations

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _build(Z, OUT, NK: tl.constexpr, TILES: tl.constexpr,
           START: tl.constexpr, COUNT: tl.constexpr):
    tile = tl.program_id(0) + START
    kv = tl.program_id(1)
    key = tile * 64 + tl.arange(0, 64)
    rank = tl.arange(0, 32)
    valid = key < NK
    values = tl.load(Z + (kv * NK + key[:, None]) * 32 + rank[None, :],
                     mask=valid[:, None], other=0.).to(tl.float32)
    count = tl.minimum(64, NK - tile * 64)
    center = tl.sum(values, axis=0) / count
    distance2 = tl.sum((values - center[None, :]) * (values - center[None, :]), axis=1)
    radius = tl.sqrt(tl.max(tl.where(valid, distance2, 0.), axis=0))
    tl.store(OUT + (kv * TILES + tile) * 33 + rank, center)
    tl.store(OUT + (kv * TILES + tile) * 33 + 32, radius)


def build(z: torch.Tensor, *, start_tile: int = 0, end_tile: int | None = None,
          out: torch.Tensor | None = None) -> torch.Tensor:
    if z.ndim != 4 or z.shape[-1] != 32 or z.dtype != torch.float32 or not z.is_cuda or not z.is_contiguous():
        raise ValueError('contiguous CUDA FP32 [batch,kvheads,keys,32] Z required')
    b, hk, nk, _ = z.shape
    tiles = tr.cdiv(nk, 64)
    end_tile = tiles if end_tile is None else end_tile
    if not 0 <= start_tile <= end_tile <= tiles or nk < 1:
        raise ValueError('invalid tile interval')
    if out is None:
        out = torch.empty((b, hk, tiles, 33), device=z.device, dtype=torch.float32)
    if out.shape != (b, hk, tiles, 33) or out.dtype != torch.float32 or out.device != z.device or not out.is_contiguous():
        raise ValueError('invalid output buffer')
    if end_tile > start_tile:
        _build[(end_tile - start_tile, b * hk)](z, out, nk, tiles, start_tile, end_tile - start_tile,
                                                num_warps=4)
    return out


def cpu_reference(z: torch.Tensor) -> torch.Tensor:
    """FP64 arithmetic reference for identity checks, not a safety bound."""
    if z.ndim != 4 or z.shape[-1] != 32:
        raise ValueError('Z shape')
    b, hk, nk, _ = z.shape
    result = torch.empty((b, hk, (nk + 63) // 64, 33), dtype=torch.float64)
    source = z.detach().cpu().double()
    for tile in range(result.shape[2]):
        block = source[..., tile * 64:min(nk, (tile + 1) * 64), :]
        center = block.mean(dim=-2)
        radius = torch.linalg.vector_norm(block - center[..., None, :], dim=-1).amax(dim=-1)
        result[..., tile, :32] = center
        result[..., tile, 32] = radius
    return result
