"""v27 pipelined selector for the summary-LOAD path (M1/M2/M3 decision calls).

The generic ``_route_generic`` scans every KV64 tile in one loop whose body branches on
``summarized = LOAD and j < PREFIX_TILES`` and conditionally updates the retained state.
That control flow keeps Triton from software-pipelining the loop, so each of the ~KT tile
iterations of the only QB x H (= 2 x 16) programs waits for its summary loads: measured
5.1 ms per routed layer at 64K keys (dense 64-row attention of the same layer: 5.4 ms).

This kernel is the same decision, split into two branch-free loops:
  1. prefix tiles [0, PREFIX_TILES): summary loads only (z, mu or the compact pool,
     active, bad) -- independent of the live state, so they can be prefetched;
  2. tail tiles [PREFIX_TILES, KT): the unchanged score path (exact / pooled / compact mu,
     optional TAIL score addressing of the fused observation).
The retained-state update ``if eligible & ~drop`` becomes a select of the identical
expressions. Per-tile arithmetic, operand order, scan order, threshold and stores are the
same as ``_route_generic`` (launched with enable_fp_fusion=False like it), so decisions are
expected to be bit-identical; tests/test_v27_route_pipelined.py checks this against the
generic kernel. TRACE and STORE are not served here (anchor calls keep the generic kernel).
"""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .cached_executor import _logadd


@tr.jit(do_not_specialize=['K', 'KT', 'PREFIX_TILES'])
def _route_load_pipelined(S, Z, REF, T, SKIP, ELIGIBLE, BADTILE,
                          ZSUM, MUSUM, ACTSUM, BADSUM, POOLED, POOLCNT, POOLBAD, KOFF, KSTORE,
                          Q: tl.constexpr, K, H: tl.constexpr, HK: tl.constexpr,
                          R: tl.constexpr, RP: tl.constexpr, QB: tl.constexpr,
                          KT, THRESHOLD: tl.constexpr, PREFIX_TILES,
                          POOL: tl.constexpr, COMPACT: tl.constexpr, KDIV: tl.constexpr,
                          TAIL: tl.constexpr):
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV
    kh = h // (H // HK)
    qi = qb*128 + tl.arange(0, 128)
    ki = tl.arange(0, 64)
    ri = tl.arange(0, RP)
    previous = tl.full((128,), -float('inf'), tl.float32)
    projected = tl.full((128, RP), 0., tl.float32)
    reference = tl.maximum(tl.load(REF+batch*HK+kh), 1e-12)
    sensitivity = tl.load(T+batch*Q+qi, qi<Q, other=1.)
    rows = qi < Q
    # 1. summarized prefix: every load is independent of the retained state.
    for j in range(PREFIX_TILES):
        base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + tl.arange(0, 128)
        block_z = tl.load(ZSUM+base, rows, other=-float('inf'))
        if COMPACT:
            pooled = tl.load(POOLED+((batch*HK+kh)*KT+j)*RP+ri, ri < R, other=0.)
            mu = tl.zeros((128, RP), tl.float32) + pooled[None, :]
        else:
            mu = tl.load(MUSUM+base[:, None]*RP+ri[None, :],
                         rows[:, None] & (ri[None, :] < R), other=0.)
        active = tl.load(ACTSUM+base, rows, other=0) != 0
        bad_row = tl.load(BADSUM+base, rows, other=0) != 0
        eligible = tl.sum((active | bad_row).to(tl.int32), 0)>0
        combined = _logadd(previous, block_z)
        safe = tl.where(combined > -float('inf'), combined, 0.)
        alpha = tl.where(active, lib.exp(block_z-safe), 0.)
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
        tl.store(BADTILE+dest, tl.sum(bad_row.to(tl.int32), 0) > 0)
        keep = eligible & ~drop
        old = lib.exp(previous-safe)
        projected = tl.where(keep, old[:, None]*projected+alpha[:, None]*mu, projected)
        previous = tl.where(keep, combined, previous)
    # 2. tail (canvas and the unaligned boundary): the unchanged score path.
    for j in range(PREFIX_TILES, KT):
        kk = j*64+ki
        valid_position = (qi[:, None]<Q) & (kk[None, :]<K)
        if TAIL:
            score = tl.load(S+((batch*H+h)*Q+qi[:, None])*KSTORE+(kk[None, :]-KOFF),
                            valid_position, other=-float('inf'))
        else:
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
        if COMPACT:
            pooled = tl.load(POOLED+((batch*HK+kh)*KT+j)*RP+ri, ri < R, other=0.)
            mu = tl.zeros((128, RP), tl.float32) + pooled[None, :]
            count = tl.load(POOLCNT+(batch*HK+kh)*KT+j)
            mismatch = rows & (tl.sum(finite.to(tl.int32), 1) != count)
            tl.store(POOLBAD+((batch*H+h)*QB+qb)*KT+j, tl.sum(mismatch.to(tl.int32), 0) > 0)
        elif POOL:
            sketch = tl.load(Z+((batch*HK+kh)*K+kk[:, None])*R+ri[None, :],
                             (kk[:, None]<K) & (ri[None, :]<R), other=0.)
            legal_w = finite.to(tl.float32)
            legal_w = legal_w / tl.maximum(tl.sum(legal_w, 1), 1.)[:, None]
            mu = tl.dot(legal_w, sketch, input_precision='tf32x3')
        else:
            sketch = tl.load(Z+((batch*HK+kh)*K+kk[:, None])*R+ri[None, :],
                             (kk[:, None]<K) & (ri[None, :]<R), other=0.)
            mu = tl.dot(weights, sketch, input_precision='tf32x3')
        combined = _logadd(previous, block_z)
        safe = tl.where(combined > -float('inf'), combined, 0.)
        alpha = tl.where(active, lib.exp(block_z-safe), 0.)
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
        tl.store(BADTILE+dest, tl.sum(bad_row.to(tl.int32), 0) > 0)
        keep = eligible & ~drop
        old = lib.exp(previous-safe)
        projected = tl.where(keep, old[:, None]*projected+alpha[:, None]*mu, projected)
        previous = tl.where(keep, combined, previous)
