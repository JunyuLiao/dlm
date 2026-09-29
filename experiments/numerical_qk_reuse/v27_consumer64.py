"""v27 experimental pre-QK sparse consumer: 64-row query blocks + optional split-KV.

Diagnosis being tested: the production consumer (``_preqk_pv_generic_output``)
runs one program per 16 query rows, so each kept K/V tile is re-read by
16 row-programs x 8 query heads per KV head, and its 16-row dots under-use the
tensor cores. This kernel keeps the same bitmap semantics (support granularity is
Q128 x KV64 per query head; a 64-row block reads the bitmap row of its Q128) but
loads each kept tile once per 64 rows of one head, and can split the KV range
across programs (flash-decoding style) to fill the GPU at a 256-row canvas.
Split-KV changes the reduction order; it is a named numerical variant.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _consume64(Q, K, V, SKIP, ELIG, PO, PM, PL,
               SQH, SQL, SKH, SKL, SVH, SVL,
               NQ, NK, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
               QB: tl.constexpr, KT, SPLITS: tl.constexpr, SCALE,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr = False,
               WINDOW: tl.constexpr = 0):
    mb, h, sp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (H // HK)
    qi = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    di = tl.arange(0, D)
    ki = tl.arange(0, BLOCK_N)
    rows = qi < NQ
    qb = (mb * BLOCK_M) // 128
    q = tl.load(Q + h * SQH + qi[:, None] * SQL + di[None, :], rows[:, None], other=0.)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    per = tl.cdiv(KT, SPLITS)
    lo = sp * per
    hi = tl.minimum(lo + per, KT)
    if CAUSAL:
        # queries sit at the end of the key extent; tiles past the last row's position are empty
        last = (mb * BLOCK_M + BLOCK_M - 1) + (NK - NQ)
        hi = tl.minimum(hi, last // BLOCK_N + 1)
        if WINDOW > 0:
            first = (mb * BLOCK_M) + (NK - NQ) - WINDOW + 1
            lo = tl.maximum(lo, tl.maximum(first, 0) // BLOCK_N)
    for j in range(lo, hi):
        dest = (h * QB + qb) * KT + j
        keep = tl.load(ELIG + dest) & (tl.load(SKIP + dest) == 0)
        if keep:
            kk = j * BLOCK_N + ki
            kv_ok = kk < NK
            qpos = qi + (NK - NQ)
            k = tl.load(K + kh * SKH + kk[:, None] * SKL + di[None, :], kv_ok[:, None], other=0.)
            s = tl.dot(q, tl.trans(k)) * SCALE
            if CAUSAL:
                ok = kv_ok[None, :] & (kk[None, :] <= qpos[:, None])
                if WINDOW > 0:
                    ok = ok & (kk[None, :] > qpos[:, None] - WINDOW)
                s = tl.where(ok, s, -float('inf'))
            else:
                s = tl.where(kv_ok[None, :], s, -float('inf'))
            m_new = tl.maximum(m, tl.max(s, 1))
            safe = tl.where(m_new > -float('inf'), m_new, 0.)
            alpha = tl.where(m > -float('inf'), tl.exp(m - safe), 0.)
            p = tl.exp(s - safe[:, None])
            l = alpha * l + tl.sum(p, 1)
            v = tl.load(V + kh * SVH + kk[:, None] * SVL + di[None, :], kv_ok[:, None], other=0.)
            acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), v)
            m = m_new
    base = (h * SPLITS + sp) * NQ + qi
    tl.store(PM + base, m, rows)
    tl.store(PL + base, l, rows)
    tl.store(PO + base[:, None] * D + di[None, :], acc, rows[:, None])


def dense64(q, k, v, scale, splits=2, causal=False, window=0):
    """Dense attention with the same kernel (every tile kept): [1,Q,H,D] model-major."""
    h, nq, nk = q.shape[1], q.shape[2], k.shape[2]
    shape = (1, h, math.ceil(nq / 128), math.ceil(nk / 64))
    keep = torch.zeros(shape, dtype=torch.bool, device=q.device)
    return consume64(q, k, v, keep, torch.ones_like(keep), scale, splits=splits, causal=causal, window=window)


def consume64(q, k, v, skipped, eligible, scale, block_m=64, splits=1, num_warps=8, num_stages=1, causal=False,
              window=0):
    """q [1,H,Q,D], k/v [1,HK,K,D] (strided views allowed), bitmap [1,H,QB128,KT64]."""
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[2]
    kt = math.ceil(nk / 64)
    qb = math.ceil(nq / 128)
    po = torch.empty((h, splits, nq, d), device=q.device, dtype=torch.float32)
    pm = torch.empty((h, splits, nq), device=q.device, dtype=torch.float32)
    pl = torch.empty((h, splits, nq), device=q.device, dtype=torch.float32)
    grid = (math.ceil(nq / block_m), h, splits)
    _consume64[grid](q, k, v, skipped, eligible, po, pm, pl,
                     q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
                     nq, nk, h, hk, d, qb, kt, splits, scale,
                     BLOCK_M=block_m, BLOCK_N=64, CAUSAL=bool(causal), WINDOW=int(window),
                     num_warps=num_warps, num_stages=num_stages)
    if splits == 1:
        out = po[:, 0] / pl[:, 0, :, None]
    else:
        mx = pm.amax(1, keepdim=True)
        w = torch.exp(pm - mx)
        out = (po * w[..., None]).sum(1) / (pl * w).sum(1)[..., None]
    return out.to(torch.bfloat16).transpose(0, 1).contiguous().unsqueeze(0)  # [1,Q,H,D] model-major
