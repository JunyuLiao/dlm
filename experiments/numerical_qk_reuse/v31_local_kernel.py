"""Compact paged-KV consumer for the V31 LOCAL sliding-window map.

The native LOCAL call has a short key window, so FA4's block-list setup can
cost more than the tiles it skips.  This consumer keeps the same Q64/KV64
map, walks only the compact list of retained tiles, and performs the native
bottom-right aligned bidirectional window in the kernel.  It is a consumer
only: selection, budgets, and the held map remain in ``v31_local_sparse``.

The implementation is intentionally separate from the selector so the FA4
consumer remains available as a fallback and the two paths can be qualified
against one another.
"""
from __future__ import annotations

import math

import torch

try:  # Keep CPU-side selector and receipt tests importable without Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in a CPU-only env.
    triton = None
    tl = None


def compact_map(kept: torch.Tensor, n: int):
    """Expand a Q128 held map to Q64 and pack each row's true tile indices.

    ``kept`` has the selector's shape ``[1, H, ceil(n/128), KT]``.  The two
    Q64 halves of a Q128 block intentionally share its selected tiles, which
    preserves the selector's work exactly while matching the short-window
    consumer's launch granularity.
    """
    if kept.ndim != 4 or kept.shape[0] != 1 or n <= 0:
        raise ValueError('LOCAL compact map expects [1, H, QB128, KT] and n > 0')
    qblocks = (int(n) + 63) // 64
    expanded = kept.repeat_interleave(2, dim=2)[..., :qblocks, :]
    counts = expanded.sum(-1).to(torch.int32).contiguous()
    max_keep = int(counts.max().item())
    if max_keep <= 0:
        raise ValueError('LOCAL compact map has no retained tile')
    order = torch.argsort((~expanded).to(torch.int8), dim=-1, stable=True)
    return order[..., :max_keep][0].to(torch.int32).contiguous(), counts[0]


if triton is not None:

    @triton.jit
    def _compact_local(
        Q, K, V, TABLE, IDX, CNT, O,
        SQH, SQQ, SQD,
        SKP, SKT, SKH, SKD,
        SVP, SVT, SVH, SVD,
        NQ, NK,
        GROUP: tl.constexpr, D: tl.constexpr,
        QBLOCKS: tl.constexpr, MAX_KEEP: tl.constexpr,
        SCALE, WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
        BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 64,
    ):
        qb, head = tl.program_id(0), tl.program_id(1)
        qrow = qb * BLOCK_M + tl.arange(0, BLOCK_M)
        dim = tl.arange(0, D)
        key_row = tl.arange(0, BLOCK_N)
        row_ok = qrow < NQ
        kv_head = head // GROUP

        q = tl.load(
            Q + head * SQH + qrow[:, None] * SQQ + dim[None, :] * SQD,
            mask=row_ok[:, None], other=0.0,
        )
        m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        count = tl.load(CNT + head * QBLOCKS + qb)

        for pos in range(MAX_KEEP):
            active = pos < count
            tile = tl.load(IDX + (head * QBLOCKS + qb) * MAX_KEEP + pos,
                           mask=active, other=0)
            logical_key = tile * BLOCK_N + key_row
            key_ok = logical_key < NK
            page = tl.load(TABLE + tile, mask=active, other=0)
            k = tl.load(
                K + page * SKP + key_row[:, None] * SKT + kv_head * SKH + dim[None, :] * SKD,
                mask=(active & key_ok)[:, None], other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * SCALE
            qpos = qrow + (NK - NQ)
            valid = row_ok[:, None] & active & key_ok[None, :] & (
                (qpos[:, None] - logical_key[None, :] <= WINDOW_LEFT)
                & (logical_key[None, :] - qpos[:, None] <= WINDOW_RIGHT)
            )
            scores = tl.where(valid, scores, -float('inf'))
            tile_max = tl.max(scores, 1)
            new_m = tl.maximum(m, tile_max)
            safe_m = tl.where(new_m > -float('inf'), new_m, 0.0)
            alpha = tl.where(m > -float('inf'), tl.exp(m - safe_m), 0.0)
            p = tl.where(valid, tl.exp(scores - safe_m[:, None]), 0.0)
            tile_l = tl.sum(p, 1)
            vv = tl.load(
                V + page * SVP + key_row[:, None] * SVT + kv_head * SVH + dim[None, :] * SVD,
                mask=(active & key_ok)[:, None], other=0.0,
            )
            l = alpha * l + tile_l
            acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), vv)
            m = new_m

        out = acc / tl.maximum(l, 1e-20)[:, None]
        tl.store(
            O + (head * NQ + qrow)[:, None] * D + dim[None, :], out,
            mask=row_ok[:, None],
        )


def forward(query_h: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            table: torch.Tensor, indices: torch.Tensor, counts: torch.Tensor,
            nk: int, scale: float | None = None, window=(1023, 1023)) -> torch.Tensor:
    """Run compact LOCAL attention and return model-major ``[1, Q, H, D]``."""
    if triton is None:
        raise RuntimeError('LOCAL compact consumer requires Triton')
    if query_h.ndim != 4 or query_h.shape[0] != 1:
        raise ValueError('query must be [1, H, Q, D]')
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError('K/V must be [pages64, 64, Hkv, D]')
    if indices.ndim != 3 or counts.ndim != 2 or indices.shape[:2] != counts.shape:
        raise ValueError('compact map shapes must be [H, QB64, K] and [H, QB64]')
    h, nq, d = (int(query_h.shape[1]), int(query_h.shape[2]), int(query_h.shape[3]))
    hk = int(key.shape[2])
    if h % hk or indices.shape[0] != h or indices.shape[1] != (nq + 63) // 64:
        raise ValueError('LOCAL compact map/query GQA geometry mismatch')
    if int(nk) > int(key.shape[0]) * int(key.shape[1]):
        raise ValueError('LOCAL compact key length exceeds paged K/V')
    tab = table[0] if table.ndim == 2 else table
    if tab.ndim != 1:
        raise ValueError('LOCAL page table must be one-dimensional')
    out_h = torch.empty((h, nq, d), device=query_h.device, dtype=query_h.dtype)
    _compact_local[(math.ceil(nq / 64), h)](
        query_h[0], key, value, tab, indices, counts, out_h,
        query_h.stride(1), query_h.stride(2), query_h.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        nq, int(nk), GROUP=h // hk, D=d, QBLOCKS=indices.shape[1],
        MAX_KEEP=indices.shape[2], SCALE=d ** -0.5 if scale is None else float(scale),
        WINDOW_LEFT=int(window[0]), WINDOW_RIGHT=int(window[1]),
        num_warps=4, num_stages=2,
    )
    return out_h.transpose(0, 1).unsqueeze(0)
