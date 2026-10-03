"""Fused control implementation for correctness and scheduling experiments.

No full attention matrix, diagnostic dense PV, or host-dependent skip branch.
This is the explicitly labelled naive fused control, not the production claim.
"""
from dataclasses import dataclass
import math

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _logadd(a, b):
    m = tl.maximum(a, b)
    safe = tl.where(m > -float('inf'), m, 0.)
    return tl.where(m > -float('inf'), safe + lib.log(lib.exp(a-safe)+lib.exp(b-safe)), m)


@tr.jit
def _attention(Q, K, V, Z, REF, MASK, O, SKIP, ELIGIBLE, LSE, STATE, RISK,
               NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
               D: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, QB: tl.constexpr,
               SCALE: tl.constexpr, THRESHOLD: tl.constexpr, MODE: tl.constexpr,
               MASK_KIND: tl.constexpr, MASK_HEADS: tl.constexpr,
               CAUSAL: tl.constexpr, WINDOW: tl.constexpr, PRECISION: tl.constexpr,
               NATIVE_LOGITS: tl.constexpr, TRACE: tl.constexpr):
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (H // HK)
    qi = qb*128+tl.arange(0, 128)
    ki = tl.arange(0, 64)
    di = tl.arange(0, D)
    ri = tl.arange(0, RP)
    q = tl.load(Q+((batch*H+h)*NQ+qi[:, None])*D+di[None, :], qi[:, None] < NQ, other=0.)
    previous = tl.full((128,), -float('inf'), tl.float32)
    projected = tl.full((128, RP), 0., tl.float32)
    out = tl.full((128, D), 0., tl.float32)
    running_max = tl.full((128,), -float('inf'), tl.float32)
    reference = tl.maximum(tl.load(REF+batch*HK+kh), 1e-12)
    for j in range(tr.cdiv(NK, 64)):
        kk = j*64+ki
        k = tl.load(K+((batch*HK+kh)*NK+kk[None, :])*D+di[:, None], kk[None, :] < NK, other=0.)
        logits = tl.dot(q, k)
        if NATIVE_LOGITS:
            logits = (logits.to(tl.bfloat16).to(tl.float32)*SCALE).to(tl.bfloat16).to(tl.float32)
        else:
            logits = logits*SCALE
        valid = (qi[:, None] < NQ) & (kk[None, :] < NK)
        if MASK_KIND:
            mh = h if MASK_HEADS == H else 0
            mask = tl.load(MASK+((batch*MASK_HEADS+mh)*NQ+qi[:, None])*NK+kk[None, :], valid, other=0.)
            if MASK_KIND == 1:
                valid = valid & (mask != 0)
            else:
                valid = valid & (mask > -1.e4) & (mask < float('inf'))
                logits = (logits+mask.to(tl.float32)).to(tl.bfloat16).to(tl.float32) if NATIVE_LOGITS else logits+mask
        if CAUSAL:
            valid = valid & (kk[None, :] <= NK-NQ+qi[:, None])
        if WINDOW:
            valid = valid & (kk[None, :] >= NK-NQ+qi[:, None]-WINDOW+1)
        logits = tl.where(valid, logits, -float('inf'))
        active = tl.sum(valid.to(tl.int32), 1) > 0
        eligible = tl.sum(active.to(tl.int32), 0) > 0
        maximum = tl.max(logits, 1)
        running_max = tl.maximum(running_max, maximum)
        weights = lib.exp(logits-tl.where(active, maximum, 0.)[:, None])
        ell = tl.sum(weights, 1)
        weights = weights/tl.maximum(ell, 1.e-30)[:, None]
        blockz = tl.where(active, maximum+lib.log(tl.maximum(ell, 1.e-30)), -float('inf'))
        combined = _logadd(previous, blockz)
        alpha = tl.where(active, lib.exp(blockz-tl.where(combined > -float('inf'), combined, 0.)), 0.)
        mu = tl.full((128, RP), 0., tl.float32)
        if MODE == 1:
            sketch = tl.load(Z+((batch*HK+kh)*NK+kk[:, None])*R+ri[None, :],
                             (kk[:, None] < NK) & (ri[None, :] < R), other=0.)
            mu = tl.dot(weights, sketch, input_precision=PRECISION)
            delta = alpha[:, None]*(mu-projected)
            risk = lib.log(tl.sqrt(tl.sum(delta*delta, 1))/reference)
        elif MODE == 2:
            risk = maximum-running_max
        else:
            risk = tl.full((128,), float('inf'), tl.float32)
        risk = tl.where(active, tl.where(previous > -float('inf'), risk, float('inf')), -float('inf'))
        worst = tl.max(risk, 0)
        skip = eligible & (worst < THRESHOLD)
        dest = ((batch*H+h)*QB+qb)*tr.cdiv(NK, 64)+j
        tl.store(SKIP+dest, skip)
        tl.store(ELIGIBLE+dest, eligible)
        if TRACE:
            tl.store(RISK+dest, worst)
        if eligible & ~skip:
            safe = tl.where(combined > -float('inf'), combined, 0.)
            a = lib.exp(previous-safe)
            projected = a[:, None]*projected+alpha[:, None]*mu
            v = tl.load(V+((batch*HK+kh)*NK+kk[:, None])*D+di[None, :], kk[:, None] < NK, other=0.)
            # BF16 tensor-core PV; online normalization differs in rounding from
            # historical final-normalized-probability BF16 GEMM, not in operator.
            p = (alpha[:, None]*weights).to(tl.bfloat16)
            out = out*a[:, None]+tl.dot(p, v)
            previous = combined
    tl.store(O+((batch*H+h)*NQ+qi[:, None])*D+di[None, :], out, qi[:, None] < NQ)
    tl.store(LSE+(batch*H+h)*NQ+qi, previous, qi < NQ)
    if TRACE:
        tl.store(STATE+((batch*H+h)*NQ+qi[:, None])*R+ri[None, :], projected,
                 (qi[:, None] < NQ) & (ri[None, :] < R))


@dataclass
class Output:
    output: torch.Tensor
    skipped: torch.Tensor
    eligible: torch.Tensor
    log_normalizer: torch.Tensor
    projected_state: torch.Tensor
    risk: torch.Tensor


def attention(q, k, v, z, reference, *, mask=None, scale=None,
              log_threshold=-math.inf, mode='value', causal=False, window=0,
              precision='ieee', native_logits=True, trace=False, num_warps=8, num_stages=1):
    """Contiguous B,H,Q,D; K/V/Z are native-head tensors, explicit FP32 Z/ref.

    Structural masks are explicit or bottom-right causal/left-window. A supplied
    four-dimensional DiffusionGemma mask already implements its local window;
    callers must then pass window=0, matching the historical reference.
    """
    if not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('CUDA BF16 Q/K/V required')
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('Expected B,H,N,D and matching K/V')
    b, h, nq, d = q.shape
    bk, hk, nk, dk = k.shape
    if b != bk or d != dk or h % hk or d not in (64, 128, 256, 512) or min(nq, nk) <= 0:
        raise ValueError('Unsupported dimensions/GQA')
    if mode not in ('dense', 'value', 'blasst') or precision not in ('ieee', 'tf32x3'):
        raise ValueError('Unsupported mode/precision')
    if math.isnan(log_threshold) or window < 0:
        raise ValueError('Invalid threshold/window')
    if z.shape[:3] != (b, hk, nk) or z.shape[-1] not in (1, 2, 4, 8, 16, 24, 32):
        raise ValueError('Invalid native-head sketches')
    if reference.shape != (b, hk) or z.dtype != torch.float32 or reference.dtype != torch.float32:
        raise ValueError('FP32 sketches and B,HK reference required')
    for x in (q, k, v, z, reference):
        if x.device != q.device or not x.is_contiguous():
            raise ValueError('Tensors must share device and be contiguous')
    kind, mh = 0, 1
    if mask is not None:
        if mask.device != q.device or mask.ndim != 4 or mask.shape[0] != b or mask.shape[1] not in (1, h) or mask.shape[2:] != (nq, nk):
            raise ValueError('Mask must be B,1-or-H,Q,K on the input device')
        if mask.dtype not in (torch.bool, torch.bfloat16):
            raise ValueError('Explicit bool/BF16 masks only; no silent mask cast')
        kind, mh = (1 if mask.dtype == torch.bool else 2), mask.shape[1]
        mask = mask.contiguous()
    r, qb, kt = z.shape[-1], tr.cdiv(nq, 128), tr.cdiv(nk, 64)
    out = torch.empty_like(q)
    skipped = torch.empty((b, h, qb, kt), device=q.device, dtype=torch.bool)
    eligible = torch.empty_like(skipped)
    lse = torch.empty((b, h, nq), device=q.device, dtype=torch.float32)
    state = torch.empty((b, h, nq, r) if trace else (1,), device=q.device, dtype=torch.float32)
    risk = torch.empty(skipped.shape if trace else (1,), device=q.device, dtype=torch.float32)
    _attention[(qb, h, b)](q, k, v, z, reference, mask if mask is not None else q,
        out, skipped, eligible, lse, state, risk, nq, nk, h, hk, d, r, max(16, tr.next_power_of_2(r)), qb,
        d**-.5 if scale is None else scale, log_threshold, {'dense':0, 'value':1, 'blasst':2}[mode],
        kind, mh, causal, window, precision, native_logits, trace, num_warps=num_warps,
        num_stages=num_stages, enable_fp_fusion=False)
    return Output(out, skipped, eligible, lse, state, risk)
