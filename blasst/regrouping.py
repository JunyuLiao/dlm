"""Training-free query-row grouping and physical-sparsity analysis.

The functions here are deliberately independent of the attention kernel.  A
permutation always maps ``physical_row -> original_row`` and never crosses a
request boundary.  Offline evaluation can therefore establish headroom before
the same permutation is passed to the opt-in Triton prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import torch


GroupingStrategy = Literal["identity", "bitmap", "topk", "simhash", "bucket", "minhash", "oracle_greedy"]


@dataclass(frozen=True)
class RegroupingMetrics:
    row_sparsity: float
    physical_tile_sparsity: float
    average_required_kv_tiles: float
    required_physical_tiles: int
    total_physical_tiles: int
    estimated_v_bytes: int
    estimated_pv_flops: int
    active_rows_mean: float
    active_rows_p50: float
    active_rows_p90: float
    active_rows_p99: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def pack_keep_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pack the final KV-tile dimension, retaining a compact CPU uint8 tensor."""
    values = np.packbits(mask.detach().bool().cpu().numpy(), axis=-1, bitorder="little")
    return torch.from_numpy(values.copy())


def unpack_keep_mask(packed: torch.Tensor, num_kv_tiles: int) -> torch.Tensor:
    values = np.unpackbits(packed.detach().cpu().numpy(), axis=-1, bitorder="little")
    return torch.from_numpy(values[..., :num_kv_tiles].copy()).bool()


def _lexicographic_order(columns: np.ndarray) -> np.ndarray:
    if columns.ndim != 2:
        raise ValueError("signature columns must be two-dimensional")
    rows = columns.shape[0]
    original = np.arange(rows, dtype=np.int64)
    # np.lexsort uses the last key as primary. Original position is the final
    # tie-break, making every strategy deterministic.
    keys = [original]
    keys.extend(columns[:, index] for index in range(columns.shape[1] - 1, -1, -1))
    return np.lexsort(tuple(keys)).astype(np.int64, copy=False)


def aggregate_heads(keep_mask: torch.Tensor, rho: float | None) -> torch.Tensor:
    """Return a row signature shared by a layer from ``[heads, rows, tiles]``."""
    if keep_mask.ndim != 3:
        raise ValueError("keep_mask must have shape [heads, rows, KV tiles]")
    if rho is None:
        return keep_mask.bool().any(0)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return keep_mask.float().mean(0) >= rho


def _bitmap_columns(mask: torch.Tensor) -> np.ndarray:
    return np.packbits(mask.bool().cpu().numpy(), axis=-1, bitorder="big")


def _topk_columns(mask: torch.Tensor, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("top-k must be positive")
    values = mask.bool().cpu().numpy()
    rows, tiles = values.shape
    # Traversal is right-to-left; absent entries sort after real tile IDs.
    result = np.full((rows, k), tiles, dtype=np.int32)
    for row in range(rows):
        kept = np.flatnonzero(values[row])[::-1][:k]
        result[row, : kept.size] = kept
    return result


def _simhash(mask: torch.Tensor, bits: int) -> np.ndarray:
    if bits not in (8, 16, 32):
        raise ValueError("SimHash bits must be 8, 16, or 32")
    values = mask.float().cpu()
    tiles = values.shape[1]
    tile_ids = torch.arange(tiles, dtype=torch.int64)[:, None]
    bit_ids = torch.arange(bits, dtype=torch.int64)[None, :]
    # Stateless integer mixing gives stable balanced hyperplanes without a
    # saved random projection matrix.
    mixed = (tile_ids * 6364136223846793005 + bit_ids * 1442695040888963407) & 0x7FFFFFFFFFFFFFFF
    projection = ((mixed >> 17) & 1).float().mul_(2).sub_(1)
    votes = values @ projection
    result = np.zeros(values.shape[0], dtype=np.uint32)
    positive = votes.numpy() >= 0
    for bit in range(bits):
        result |= positive[:, bit].astype(np.uint32) << np.uint32(bit)
    return result


def _minhash_columns(mask: torch.Tensor, hashes: int) -> np.ndarray:
    if hashes not in (2, 4, 8):
        raise ValueError("MinHash count must be 2, 4, or 8")
    values = mask.bool().cpu().numpy()
    rows, tiles = values.shape
    tile_ids = np.arange(tiles, dtype=np.uint32)[:, None]
    seeds = np.arange(1, hashes + 1, dtype=np.uint32)[None, :]
    mixed = tile_ids * np.uint32(0x9E3779B1) ^ seeds * np.uint32(0x85EBCA77)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    expanded = np.where(values[:, :, None], mixed[None, :, :], np.uint32(0xFFFFFFFF))
    return expanded.min(1)


def oracle_greedy_permutation(mask: torch.Tensor, group_size: int) -> torch.Tensor:
    """Plan-specified greedy union minimizer for one row-signature matrix."""
    if mask.ndim != 2 or group_size <= 0:
        raise ValueError("oracle mask must be [rows, tiles] and group_size positive")
    packed = np.packbits(mask.bool().cpu().numpy(), axis=-1, bitorder="little")
    popcount = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    remaining = np.ones(packed.shape[0], dtype=bool)
    ordered: list[int] = []
    while bool(remaining.any()):
        seed = int(np.flatnonzero(remaining)[0])
        group = [seed]
        remaining[seed] = False
        union = packed[seed].copy()
        while bool(remaining.any()) and len(group) < group_size:
            candidates = np.flatnonzero(remaining)
            new_bits = np.bitwise_and(packed[candidates], np.bitwise_not(union))
            incremental = popcount[new_bits].sum(1)
            hamming = popcount[np.bitwise_xor(packed[candidates], union)].sum(1)
            chosen = int(candidates[np.lexsort((candidates, hamming, incremental))[0]])
            group.append(chosen)
            remaining[chosen] = False
            np.bitwise_or(union, packed[chosen], out=union)
        ordered.extend(group)
    return torch.tensor(ordered, dtype=torch.long)


def build_permutation(
    signal_mask: torch.Tensor,
    strategy: GroupingStrategy,
    *,
    group_size: int = 128,
    topk: int = 4,
    signature_bits: int = 16,
    bucket_bits: int = 8,
    num_hashes: int = 4,
    state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a deterministic request-local permutation from ``[rows, tiles]``."""
    if signal_mask.ndim != 2:
        raise ValueError("signal_mask must have shape [rows, KV tiles]")
    rows = signal_mask.shape[0]
    if strategy == "identity":
        return torch.arange(rows)
    if strategy == "oracle_greedy":
        if state is not None:
            # State-aware oracle is independently greedy inside each class.
            parts = []
            for value in torch.unique(state, sorted=True):
                selected = torch.flatnonzero(state == value)
                local = oracle_greedy_permutation(signal_mask[selected], group_size)
                parts.append(selected[local])
            return torch.cat(parts)
        return oracle_greedy_permutation(signal_mask, group_size)
    if strategy == "bitmap":
        columns = _bitmap_columns(signal_mask)
    elif strategy == "topk":
        columns = _topk_columns(signal_mask, topk)
    elif strategy in ("simhash", "bucket"):
        signature = _simhash(signal_mask, signature_bits)
        if strategy == "bucket":
            if not 1 <= bucket_bits <= signature_bits:
                raise ValueError("bucket bits must be in [1, signature_bits]")
            signature = signature >> np.uint32(signature_bits - bucket_bits)
        columns = signature[:, None]
    elif strategy == "minhash":
        columns = _minhash_columns(signal_mask, num_hashes)
    else:
        raise ValueError(f"unknown grouping strategy: {strategy}")
    if state is not None:
        if state.ndim != 1 or state.numel() != rows:
            raise ValueError("state must contain one class per query row")
        columns = np.concatenate((state.cpu().numpy().astype(np.int32)[:, None], columns), axis=1)
    return torch.from_numpy(_lexicographic_order(columns))


def build_windowed_permutation(
    signal_mask: torch.Tensor,
    strategy: GroupingStrategy,
    *,
    window_size: int,
    group_size: int = 128,
    **kwargs: object,
) -> torch.Tensor:
    """Regroup inside aligned spatial windows larger than one physical tile."""
    if window_size < group_size or window_size % group_size:
        raise ValueError("window_size must be a positive multiple of group_size")
    rows = signal_mask.shape[0]
    parts = []
    state = kwargs.pop("state", None)
    for start in range(0, rows, window_size):
        end = min(rows, start + window_size)
        local_state = None if state is None else state[start:end]  # type: ignore[index]
        local = build_permutation(
            signal_mask[start:end], strategy, group_size=group_size,
            state=local_state, **kwargs,
        )
        parts.append(local + start)
    return torch.cat(parts)


def build_head_group_permutations(
    signal_keep_mask: torch.Tensor,
    strategy: GroupingStrategy,
    *,
    heads_per_group: int,
    rho: float = 0.5,
    group_size: int = 128,
    window_size: int | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Build one permutation per contiguous head cluster and expand to heads."""
    if signal_keep_mask.ndim != 3:
        raise ValueError("signal_keep_mask must be [heads, rows, KV tiles]")
    heads, rows, _ = signal_keep_mask.shape
    if heads_per_group <= 0 or heads % heads_per_group:
        raise ValueError("heads_per_group must evenly divide the head count")
    result = torch.empty((heads, rows), dtype=torch.long)
    for start in range(0, heads, heads_per_group):
        signal = aggregate_heads(signal_keep_mask[start : start + heads_per_group], rho)
        if window_size is None:
            perm = build_permutation(signal, strategy, group_size=group_size, **kwargs)
        else:
            perm = build_windowed_permutation(
                signal, strategy, window_size=window_size, group_size=group_size, **kwargs
            )
        result[start : start + heads_per_group] = perm
    return result


def evaluate_permutation(
    keep_mask: torch.Tensor,
    permutation: torch.Tensor,
    *,
    query_tile_size: int = 128,
    kv_tile_size: int = 64,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> RegroupingMetrics:
    """Evaluate fixed row decisions after packing rows into physical tiles."""
    keep = keep_mask.bool().cpu()
    if keep.ndim == 2:
        keep = keep[None]
    if keep.ndim != 3:
        raise ValueError("keep_mask must have shape [heads, rows, KV tiles]")
    heads, rows, kv_tiles = keep.shape
    perm = permutation.cpu().long()
    if perm.ndim == 1:
        if perm.numel() != rows:
            raise ValueError("shared permutation length does not match rows")
        perm = perm[None].expand(heads, -1)
    if perm.shape != (heads, rows):
        raise ValueError("permutation must be [rows] or [heads, rows]")
    groups = math.ceil(rows / query_tile_size)
    arranged = torch.gather(
        keep, 1, perm[:, :, None].expand(heads, rows, kv_tiles)
    )
    padded_rows = groups * query_tile_size
    if padded_rows != rows:
        arranged = torch.nn.functional.pad(arranged, (0, 0, 0, padded_rows - rows))
    tiled = arranged.reshape(heads, groups, query_tile_size, kv_tiles)
    union = tiled.any(2)
    counts = tiled.sum(2).float().reshape(-1)
    required = int(union.sum())
    group_lengths = torch.full((groups,), query_tile_size, dtype=torch.int64)
    group_lengths[-1] = rows - (groups - 1) * query_tile_size
    pv_flops = int((union.sum((0, 2)) * group_lengths).sum()) * kv_tile_size * head_dim * 2
    total = heads * groups * kv_tiles
    return RegroupingMetrics(
        row_sparsity=1.0 - float(keep.float().mean()),
        physical_tile_sparsity=1.0 - required / total,
        average_required_kv_tiles=required / (heads * groups),
        required_physical_tiles=required,
        total_physical_tiles=total,
        estimated_v_bytes=required * kv_tile_size * head_dim * dtype_bytes,
        estimated_pv_flops=pv_flops,
        active_rows_mean=float(counts.mean()),
        active_rows_p50=float(torch.quantile(counts, 0.50)),
        active_rows_p90=float(torch.quantile(counts, 0.90)),
        active_rows_p99=float(torch.quantile(counts, 0.99)),
    )


def mean_jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("Jaccard masks must have matching shapes")
    left, right = left.bool(), right.bool()
    intersection = (left & right).sum(-1).float()
    union = (left | right).sum(-1).float()
    return float(torch.where(union > 0, intersection / union, torch.ones_like(union)).mean())
