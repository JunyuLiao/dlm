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


@triton.jit
def _fused_observe(Q, K, V, Z, PO, PM, PL, ZSUM, MUSUM, ACTSUM, BADSUM, TAIL,
                   SQH, SQL, SKH, SKL, SVH, SVL,
                   NQ, NK, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                   QB, PT, KT, KOFF, KSTORE, SPLITS: tl.constexpr, SCALE,
                   MU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MU_PREC: tl.constexpr = 0):
    """v27 fused observation: one dense pass that (a) forms the dense attention output
    (online softmax over every legal tile, FP32 scores, like ``consume64``) and (b)
    writes the M1 prefix summaries of every WHOLLY-prefix KV64 tile -- block log-mass
    ``z``, weighted projected-V ``mu`` (optional), active/bad flags -- from the FP32
    current scores with the route STORE formulas, sharing one exponential per score with
    the output; (c) keeps those FP32 scores of the remaining (canvas/boundary) tiles in
    a compact tail buffer for later decisions. Named variant: FP32 observation scores."""
    mb, h, sp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (H // HK)
    qi = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    di = tl.arange(0, D)
    ki = tl.arange(0, BLOCK_N)
    ri = tl.arange(0, R)
    rows = qi < NQ
    qb = (mb * BLOCK_M) // 128
    r128 = (mb * BLOCK_M) % 128 + tl.arange(0, BLOCK_M)
    q = tl.load(Q + h * SQH + qi[:, None] * SQL + di[None, :], rows[:, None], other=0.)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    per = tl.cdiv(KT, SPLITS)
    lo = sp * per
    hi = tl.minimum(lo + per, KT)
    for j in range(lo, hi):
        kk = j * BLOCK_N + ki
        kv_ok = kk < NK
        k = tl.load(K + kh * SKH + kk[:, None] * SKL + di[None, :], kv_ok[:, None], other=0.)
        valid = rows[:, None] & kv_ok[None, :]
        # One FP32 score tile serves the summary and the output (single exp per element).
        s = tl.where(valid, tl.dot(q, tl.trans(k)) * SCALE, -float('inf'))
        tile_max = tl.max(s, 1)
        active = tile_max > -float('inf')
        safe_tile = tl.where(active, tile_max, 0.)
        p = tl.exp(s - safe_tile[:, None])
        ell = tl.sum(p, 1)
        if j < PT:
            bad_row = tl.sum(((s != s) | (s == float('inf'))).to(tl.int32), 1) > 0
            block_z = tl.where(active, tile_max + tl.log(tl.maximum(ell, 1.e-30)), -float('inf'))
            base = (((h * QB + qb) * PT + j) * 128) + r128
            tl.store(ZSUM + base, block_z, rows)
            tl.store(ACTSUM + base, active.to(tl.int8), rows)
            tl.store(BADSUM + base, bad_row.to(tl.int8), rows)
            if MU:
                sketch = tl.load(Z + (kh * NK + kk[:, None]) * R + ri[None, :], kv_ok[:, None], other=0.)
                w = p / tl.maximum(ell, 1.e-30)[:, None]
                if MU_PREC == 0:
                    mu = tl.dot(w, sketch, input_precision='tf32x3')
                elif MU_PREC == 1:
                    mu = tl.dot(w, sketch, input_precision='tf32')
                else:
                    mu = tl.dot(w.to(tl.bfloat16), sketch.to(tl.bfloat16))
                tl.store(MUSUM + base[:, None] * R + ri[None, :], mu, rows[:, None])
        else:
            tl.store(TAIL + (h * NQ + qi[:, None]) * KSTORE + (kk[None, :] - KOFF), s,
                     rows[:, None] & kv_ok[None, :])
        # online softmax from the same exponentials, rescaled by exp(tile_max - m_new)
        m_new = tl.maximum(m, tile_max)
        safe = tl.where(m_new > -float('inf'), m_new, 0.)
        alpha = tl.where(m > -float('inf'), tl.exp(m - safe), 0.)
        beta = tl.where(active, tl.exp(safe_tile - safe), 0.)
        l = alpha * l + beta * ell
        v = tl.load(V + kh * SVH + kk[:, None] * SVL + di[None, :], kv_ok[:, None], other=0.)
        acc = alpha[:, None] * acc + tl.dot((p * beta[:, None]).to(tl.bfloat16), v)
        m = m_new
    base = (h * SPLITS + sp) * NQ + qi
    tl.store(PM + base, m, rows)
    tl.store(PL + base, l, rows)
    tl.store(PO + base[:, None] * D + di[None, :], acc, rows[:, None])


def fused_observe(q, k, v, sketch, scale, prefix_tiles, summary, splits=2, mu=True, mu_precision='tf32x3'):
    """Dense output [1,Q,H,D] plus summaries written into ``summary`` (PrefixSummary with
    z/mu/active/bad sized for ``prefix_tiles``) and a compact observation-score tail
    [1,H,Q,K-64*prefix_tiles] (FP32) for the remaining tiles."""
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[2]
    if b != 1 or sketch.shape != (1, hk, nk, 32):
        raise ValueError('fused observation expects batch 1 and a [1,KVH,K,32] projected V')
    kt, qb = math.ceil(nk / 64), math.ceil(nq / 128)
    pt = int(prefix_tiles)
    koff = pt * 64
    tail = torch.empty((1, h, nq, nk - koff), device=q.device, dtype=torch.float32)
    po = torch.empty((h, splits, nq, d), device=q.device, dtype=torch.float32)
    pm = torch.empty((h, splits, nq), device=q.device, dtype=torch.float32)
    pl = torch.empty((h, splits, nq), device=q.device, dtype=torch.float32)
    dummy = torch.empty((1,), device=q.device, dtype=torch.float32)
    small = torch.empty((1,), device=q.device, dtype=torch.int8)
    zs = summary.z if pt else dummy
    mus = summary.mu if (pt and mu) else dummy
    acts = summary.active if pt else small
    bads = summary.bad if pt else small
    grid = (math.ceil(nq / 64), h, splits)
    _fused_observe[grid](q, k, v, sketch.contiguous(), po, pm, pl, zs, mus, acts, bads, tail,
                         q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
                         nq, nk, h, hk, d, 32, qb, pt, kt, koff, nk - koff, splits, scale,
                         MU=bool(mu), BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=1,
                         MU_PREC={'tf32x3': 0, 'tf32': 1, 'bf16': 2}[mu_precision])
    if splits == 1:
        out = po[:, 0] / pl[:, 0, :, None]
    else:
        mx = pm.amax(1, keepdim=True)
        w = torch.exp(pm - mx)
        out = (po * w[..., None]).sum(1) / (pl * w).sum(1)[..., None]
    return out.to(torch.bfloat16).transpose(0, 1).contiguous().unsqueeze(0), tail
