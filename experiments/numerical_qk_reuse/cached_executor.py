"""Bounded Triton consumer of observed transformed scores and current V.

The ordinary path has no current Q or K argument and performs no QK product.
Routing uses Junyu's physical 128x64 decision. A separate 16-row PV program
reads the resulting bitmap and loads V only for retained eligible 64-key tiles.
The producer is responsible for native BF16 QK/scale/bias transformations and
for invalidating the cache when score geometry or legality changes.
"""
from dataclasses import dataclass
import math

import torch

try:
    import triton as tr
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as lib
except ImportError:  # CPU reference/tests do not require Triton.
    tr = None
    tl = None
    lib = None


@dataclass
class Routing:
    """Selector-only result: no attention output, no LSE, no V touched.

    ``invalid_tiles`` flags [B,H,Qtiles,Ktiles] positions whose cached scores
    contained NaN/+inf. It replaces the malformed-score detection that the
    discarded PV pass used to provide; the caller must fail or take a
    declared fallback rather than consuming a routed support built on them.
    """
    skipped: torch.Tensor
    eligible: torch.Tensor
    invalid_tiles: torch.Tensor


@dataclass
class Output:
    output: torch.Tensor
    skipped: torch.Tensor
    eligible: torch.Tensor
    log_normalizer: torch.Tensor
    projected_state: torch.Tensor
    risk: torch.Tensor
    invalid_scores: torch.Tensor
    # [B,H,ceil(Q/16),3]: visited legal tiles, V loads, PV operations.
    counters: torch.Tensor


if tr is not None:
    @tr.jit
    def _logadd(a, b):
        maximum = tl.maximum(a, b)
        safe = tl.where(maximum > -float('inf'), maximum, 0.)
        return tl.where(maximum > -float('inf'),
                        safe + lib.log(lib.exp(a-safe)+lib.exp(b-safe)), maximum)


    @tr.jit
    def _route(S, Z, REF, T, SKIP, ELIGIBLE, BADTILE, LSE, STATE, RISK,
               Q: tl.constexpr, K: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
               R: tl.constexpr, RP: tl.constexpr, QB: tl.constexpr,
               KT: tl.constexpr, THRESHOLD: tl.constexpr, TRACE: tl.constexpr):
        qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        kh = h // (H // HK)
        qi = qb*128 + tl.arange(0, 128)
        ki = tl.arange(0, 64)
        ri = tl.arange(0, RP)
        previous = tl.full((128,), -float('inf'), tl.float32)
        projected = tl.full((128, RP), 0., tl.float32)
        reference = tl.maximum(tl.load(REF+batch*HK+kh), 1e-12)
        sensitivity = tl.load(T+batch*Q+qi, qi<Q, other=1.)
        for j in range(KT):
            kk = j*64+ki
            valid_position = (qi[:, None]<Q) & (kk[None, :]<K)
            score = tl.load(S+((batch*H+h)*Q+qi[:, None])*K+kk[None, :],
                            valid_position, other=-float('inf'))
            finite = valid_position & (score > -float('inf')) & (score < float('inf'))
            invalid = valid_position & ((score != score) | (score == float('inf')))
            clean = tl.where(finite, score, -float('inf'))
            active = tl.sum(finite.to(tl.int32), 1)>0
            bad_row = tl.sum(invalid.to(tl.int32), 1)>0
            eligible = tl.sum((active | bad_row).to(tl.int32), 0)>0
            maximum = tl.max(clean, 1)
            safe_max = tl.where(active, maximum, 0.)
            weights = lib.exp(clean-safe_max[:, None])
            ell = tl.sum(weights, 1)
            weights = weights / tl.maximum(ell, 1.e-30)[:, None]
            block_z = tl.where(active, maximum+lib.log(tl.maximum(ell, 1.e-30)), -float('inf'))
            combined = _logadd(previous, block_z)
            safe = tl.where(combined > -float('inf'), combined, 0.)
            alpha = tl.where(active, lib.exp(block_z-safe), 0.)
            sketch = tl.load(Z+((batch*HK+kh)*K+kk[:, None])*R+ri[None, :],
                             (kk[:, None]<K) & (ri[None, :]<R), other=0.)
            mu = tl.dot(weights, sketch, input_precision='tf32x3')
            delta = alpha[:, None]*(mu-projected)
            norm = tl.sqrt(tl.sum(delta*delta, 1))
            risk = lib.log(norm/reference)+lib.log(sensitivity)
            risk = tl.where(active, tl.where(previous > -float('inf'), risk, float('inf')), -float('inf'))
            risk = tl.where(bad_row, float('inf'), risk)
            worst = tl.max(risk, 0)
            drop = eligible & (worst < THRESHOLD)
            dest = ((batch*H+h)*QB+qb)*KT+j
            tl.store(SKIP+dest, drop)
            tl.store(ELIGIBLE+dest, eligible)
            # Malformed (NaN/+inf) cached scores must stay detectable without
            # relying on a PV pass that route_only never launches. This store
            # does not participate in the skip/eligible arithmetic above.
            tl.store(BADTILE+dest, tl.sum(bad_row.to(tl.int32), 0) > 0)
            if TRACE:
                tl.store(RISK+dest, worst)
            if eligible & ~drop:
                old = lib.exp(previous-safe)
                projected = old[:, None]*projected+alpha[:, None]*mu
                previous = combined
        if TRACE:
            tl.store(LSE+(batch*H+h)*Q+qi, previous, qi<Q)
            tl.store(STATE+((batch*H+h)*Q+qi[:, None])*R+ri[None, :], projected,
                     (qi[:, None]<Q) & (ri[None, :]<R))


    @tr.jit
    def _held_eligible(S, SKIP, ELIGIBLE, Q: tl.constexpr, K: tl.constexpr,
                       H: tl.constexpr, QB: tl.constexpr, KT: tl.constexpr):
        qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        qi = qb*128+tl.arange(0, 128)
        ki = tl.arange(0, 64)
        for j in range(KT):
            kk = j*64+ki
            valid = (qi[:, None]<Q) & (kk[None, :]<K)
            score = tl.load(S+((batch*H+h)*Q+qi[:, None])*K+kk[None, :],
                            valid, other=-float('inf'))
            legal_or_bad = valid & (score != -float('inf'))
            eligible = tl.sum(tl.sum(legal_or_bad.to(tl.int32), 1), 0)>0
            dest = ((batch*H+h)*QB+qb)*KT+j
            tl.store(ELIGIBLE+dest, eligible)


    @tr.jit
    def _pv(S, V, SKIP, ELIGIBLE, O, LSE, INVALID, COUNTERS,
            Q: tl.constexpr, K: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
            D: tl.constexpr, QB: tl.constexpr, KT: tl.constexpr,
            TRACE: tl.constexpr):
        q16, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        kh = h // (H // HK)
        qb = q16 // 8
        qi = q16*16+tl.arange(0, 16)
        ki = tl.arange(0, 64)
        di = tl.arange(0, D)
        maximum = tl.full((16,), -float('inf'), tl.float32)
        denom = tl.full((16,), 0., tl.float32)
        output = tl.full((16, D), 0., tl.float32)
        invalid = tl.full((16,), False, tl.int1)
        visited = 0
        vloads = 0
        pvops = 0
        for j in range(KT):
            dest = ((batch*H+h)*QB+qb)*KT+j
            eligible = tl.load(ELIGIBLE+dest)
            drop = tl.load(SKIP+dest)
            kk = j*64+ki
            valid = (qi[:, None]<Q) & (kk[None, :]<K)
            score = tl.load(S+((batch*H+h)*Q+qi[:, None])*K+kk[None, :],
                            valid, other=-float('inf'))
            bad = valid & ((score != score) | (score == float('inf')))
            bad_row = tl.sum(bad.to(tl.int32), 1)>0
            invalid = invalid | bad_row
            # A stale held decision cannot hide invalid scores; execute the V
            # tile and flag affected rows for fallback instead.
            bad_tile = tl.sum(bad_row.to(tl.int32), 0)>0
            if (eligible | bad_tile) & (~drop | bad_tile):
                visited += 1
                finite = valid & (score > -float('inf')) & (score < float('inf'))
                clean = tl.where(finite, score, -float('inf'))
                tile_max = tl.max(clean, 1)
                next_max = tl.maximum(maximum, tile_max)
                safe = tl.where(next_max > -float('inf'), next_max, 0.)
                old_scale = tl.where(maximum > -float('inf'), lib.exp(maximum-safe), 0.)
                p = lib.exp(clean-safe[:, None])
                denom = old_scale*denom+tl.sum(p, 1)
                output = old_scale[:, None]*output
                # No V read or tensor-core PV is issued for a deleted block.
                v = tl.load(V+((batch*HK+kh)*K+kk[:, None])*D+di[None, :],
                            kk[:, None]<K, other=0.)
                output += tl.dot(p.to(tl.bfloat16), v)
                maximum = next_max
                vloads += 1
                pvops += 1
        normalized = output/tl.maximum(denom[:, None], 1.e-30)
        normalized = tl.where((denom>0)[:, None] & ~invalid[:, None], normalized, 0.)
        logz = tl.where((denom>0) & ~invalid, maximum+lib.log(tl.maximum(denom, 1.e-30)), -float('inf'))
        tl.store(O+((batch*H+h)*Q+qi[:, None])*D+di[None, :], normalized.to(tl.bfloat16), qi[:, None]<Q)
        tl.store(LSE+(batch*H+h)*Q+qi, logz, qi<Q)
        tl.store(INVALID+(batch*H+h)*Q+qi, invalid, qi<Q)
        if TRACE:
            base = ((batch*H+h)*tr.cdiv(Q, 16)+q16)*3
            tl.store(COUNTERS+base+0, visited)
            tl.store(COUNTERS+base+1, vloads)
            tl.store(COUNTERS+base+2, pvops)


def attention(scores, v, z=None, reference=None, *, sensitivity=None,
              log_threshold=-math.inf, skipped=None, eligible=None,
              trace=False, num_warps=8):
    """Run M1, or consume a held M3 bitmap without projected-V routing.

    Output rows flagged in ``invalid_scores`` are zero and require caller
    fallback; they are never silently treated as valid numerical reuse.
    ``trace`` is diagnostic and must be false in timed runs.
    """
    if tr is None:
        raise RuntimeError('Triton is required for the CUDA cached-score executor')
    if scores.ndim != 4 or scores.dtype != torch.float32 or not scores.is_cuda or not scores.is_contiguous():
        raise ValueError('scores must be contiguous CUDA FP32 [B,H,Q,K]')
    if v.ndim != 4 or v.dtype != torch.bfloat16 or not v.is_contiguous() or v.device != scores.device:
        raise ValueError('v must be contiguous CUDA BF16 [B,KVH,K,D]')
    b, h, nq, nk = scores.shape
    hk, d = v.shape[1], v.shape[-1]
    if v.shape[0] != b or v.shape[2] != nk or not nq or not nk or not hk or h % hk or d not in (64, 128, 256, 512):
        raise ValueError('invalid dimensions/GQA')
    if math.isnan(log_threshold):
        raise ValueError('NaN threshold')
    qb, kt = tr.cdiv(nq, 128), tr.cdiv(nk, 64)
    shape = (b, h, qb, kt)
    if skipped is not None:
        if skipped.shape != shape or skipped.dtype != torch.bool or skipped.device != scores.device or not skipped.is_contiguous():
            raise ValueError('held bitmap must be contiguous CUDA bool [B,H,Qtiles,Ktiles]')
        if eligible is not None and (eligible.shape != shape or eligible.dtype != torch.bool or eligible.device != scores.device or not eligible.is_contiguous()):
            raise ValueError('eligible must match held bitmap')
    else:
        if eligible is not None:
            raise ValueError('eligible without skipped bitmap')
        if z is None or reference is None or z.shape[:3] != (b, hk, nk) or z.shape[-1] != 32 or reference.shape != (b, hk):
            raise ValueError('M1 requires rank-32 current projected V and [B,KVH] reference')
        for name, x in [('z', z), ('reference', reference)]:
            if x.dtype != torch.float32 or x.device != scores.device or not x.is_contiguous():
                raise ValueError(f'{name} must be contiguous CUDA FP32')
    if sensitivity is None:
        sensitivity = torch.ones((b, nq), device=scores.device, dtype=torch.float32)
    if sensitivity.shape != (b, nq) or sensitivity.dtype != torch.float32 or sensitivity.device != scores.device or not sensitivity.is_contiguous():
        raise ValueError('sensitivity must be contiguous CUDA FP32 [B,Q]')
    skip = skipped if skipped is not None else torch.empty(shape, device=scores.device, dtype=torch.bool)
    elig = eligible if eligible is not None else torch.empty(shape, device=scores.device, dtype=torch.bool)
    out = torch.empty((b, h, nq, d), device=scores.device, dtype=torch.bfloat16)
    lse = torch.empty((b, h, nq), device=scores.device, dtype=torch.float32)
    invalid = torch.empty((b, h, nq), device=scores.device, dtype=torch.bool)
    state = torch.empty((b, h, nq, 32) if trace and skipped is None else (1,), device=scores.device, dtype=torch.float32)
    risk = torch.empty(shape if trace and skipped is None else (1,), device=scores.device, dtype=torch.float32)
    counters = torch.empty((b, h, tr.cdiv(nq, 16), 3) if trace else (1,), device=scores.device, dtype=torch.int32)
    bad_tiles = torch.empty(shape if skipped is None else (1,), device=scores.device, dtype=torch.bool)
    if skipped is None:
        _route[(qb, h, b)](scores, z, reference, sensitivity, skip, elig, bad_tiles, lse, state, risk,
                           nq, nk, h, hk, 32, 32, qb, kt, log_threshold, trace,
                           num_warps=num_warps, num_stages=1, enable_fp_fusion=False)
    elif eligible is None:
        _held_eligible[(qb, h, b)](scores, skip, elig, nq, nk, h, qb, kt,
                                    num_warps=4, num_stages=1)
    _pv[(tr.cdiv(nq, 16), h, b)](scores, v, skip, elig, out, lse, invalid, counters,
                                 nq, nk, h, hk, d, qb, kt, trace,
                                 num_warps=num_warps, num_stages=1, enable_fp_fusion=False)
    return Output(out, skip, elig, lse, state, risk, invalid, counters)


def route_only(scores, z, reference, *, sensitivity=None, log_threshold=-math.inf,
               num_warps=8):
    """Selector only: the SAME ``_route`` decision, with no PV and no output.

    ``attention(...)`` with ``skipped=None`` also produces a bitmap, but it
    additionally launches ``_pv`` and allocates a full [B,H,Q,D] output plus
    LSE that ``routing_only_current_output`` throws away. This entry point
    exists so that discarded work is never issued. It takes no ``v``: the
    routing decision reads only the transformed scores and the rank-32
    projected V sketch, so V itself is not touched here.

    Arithmetic, dtypes, threshold, sensitivity weighting, scan order, tie /
    first-support handling and the per-query-head Q128 x KV64 grouping are
    the same kernel as before -- this is not a new selection method.
    """
    if tr is None:
        raise RuntimeError('Triton is required for the CUDA cached-score executor')
    if scores.ndim != 4 or scores.dtype != torch.float32 or not scores.is_cuda or not scores.is_contiguous():
        raise ValueError('scores must be contiguous CUDA FP32 [B,H,Q,K]')
    b, h, nq, nk = scores.shape
    if z is None or reference is None or z.ndim != 4 or reference.ndim != 2:
        raise ValueError('route_only requires projected V [B,KVH,K,32] and reference [B,KVH]')
    hk = z.shape[1]
    if z.shape[:3] != (b, hk, nk) or z.shape[-1] != 32 or reference.shape != (b, hk):
        raise ValueError('M1 requires rank-32 current projected V and [B,KVH] reference')
    if not nq or not nk or not hk or h % hk:
        raise ValueError('invalid dimensions/GQA')
    for name, x in [('z', z), ('reference', reference)]:
        if x.dtype != torch.float32 or x.device != scores.device or not x.is_contiguous():
            raise ValueError(f'{name} must be contiguous CUDA FP32')
    if math.isnan(log_threshold):
        raise ValueError('NaN threshold')
    if sensitivity is None:
        sensitivity = torch.ones((b, nq), device=scores.device, dtype=torch.float32)
    if sensitivity.shape != (b, nq) or sensitivity.dtype != torch.float32 or sensitivity.device != scores.device or not sensitivity.is_contiguous():
        raise ValueError('sensitivity must be contiguous CUDA FP32 [B,Q]')
    qb, kt = tr.cdiv(nq, 128), tr.cdiv(nk, 64)
    shape = (b, h, qb, kt)
    skip = torch.empty(shape, device=scores.device, dtype=torch.bool)
    elig = torch.empty(shape, device=scores.device, dtype=torch.bool)
    bad_tiles = torch.empty(shape, device=scores.device, dtype=torch.bool)
    # TRACE=False: LSE/STATE/RISK are never written, so one-element stand-ins
    # keep the launch signature without allocating per-query buffers.
    dummy = torch.empty((1,), device=scores.device, dtype=torch.float32)
    _route[(qb, h, b)](scores, z, reference, sensitivity, skip, elig, bad_tiles,
                       dummy, dummy, dummy,
                       nq, nk, h, hk, 32, 32, qb, kt, log_threshold, False,
                       num_warps=num_warps, num_stages=1, enable_fp_fusion=False)
    return Routing(skip, elig, bad_tiles)
