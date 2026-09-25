"""Length-generic twins of the cached_executor Triton kernels (v10 CP1).

Bodies are TEXTUALLY IDENTICAL to the static kernels in cached_executor.py
except that the exact key count ``K``, key-tile count ``KT`` and prefix-summary
extent ``PREFIX_TILES`` are runtime scalars instead of ``tl.constexpr``
(``tests/test_v10_generic_kernels.py`` asserts this by normalizing the two
sources). Loops were already runtime ``range`` loops, so the scan order, masks,
arithmetic, dtypes and skip branches are unchanged; only the compile-time value
of the bounds is removed.

Specialization contract (Triton 3.2 ``jit``):
  * K, KT and PREFIX_TILES: ``do_not_specialize`` -- no Triton value classes.
  * KDIV (constexpr) = largest power of two <= 16 dividing K; the launch passes
    K // KDIV and the kernel rebuilds ``K = K * KDIV`` (tl.multiple_of on a
    scalar argument was measured to be ignored). A static constexpr K let the compiler prove
    row alignment (e.g. K % 4 == 0 -> 16-byte fp32 rows), which selects the
    load layout and hence the reduction order; Triton's default integer
    specialization only tracks % 16, and without the hint K=8556 or 8156
    differed from the static kernel in the last ulp (first-mismatch test).
    At most 5 documented alignment classes per mode, never one per length.
  * Remaining constexprs are geometry/mode: Q (canvas 256), H, HK, D, QB, R,
    RP, THRESHOLD (2 frozen values), TRACE, STORE, LOAD, WINDOW.
"""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .cached_executor import _logadd

@tr.jit(do_not_specialize=['K', 'KT', 'PREFIX_TILES'])
def _route_generic(S, Z, REF, T, SKIP, ELIGIBLE, BADTILE, LSE, STATE, RISK,
           ZSUM, MUSUM, ACTSUM, BADSUM,
           Q: tl.constexpr, K, H: tl.constexpr, HK: tl.constexpr,
           R: tl.constexpr, RP: tl.constexpr, QB: tl.constexpr,
           KT, THRESHOLD: tl.constexpr, TRACE: tl.constexpr,
           PREFIX_TILES, STORE: tl.constexpr, LOAD: tl.constexpr, KDIV: tl.constexpr):
    """Selector. Optionally stores or reuses exact per-row prefix summaries.

    For a KV tile lying WHOLLY inside the immutable prefix, ``block_z`` and
    ``mu`` are functions of the frozen cached scores and the frozen prefix
    projected V only. With STORE they are written at the real score anchor;
    with LOAD they are read back instead of recomputed. Everything that
    depends on live state -- alpha, risk, the running retained log mass and
    projected accumulator, the drop decision -- is recomputed every call in
    the same scan order. Reusing the anchor's own FP32 outputs is what makes
    the optimized path bit-identical rather than merely close.
    """
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV  # v10: K arrives as K/KDIV; the product carries K's pow2 alignment
    kh = h // (H // HK)
    qi = qb*128 + tl.arange(0, 128)
    ki = tl.arange(0, 64)
    ri = tl.arange(0, RP)
    previous = tl.full((128,), -float('inf'), tl.float32)
    projected = tl.full((128, RP), 0., tl.float32)
    reference = tl.maximum(tl.load(REF+batch*HK+kh), 1e-12)
    sensitivity = tl.load(T+batch*Q+qi, qi<Q, other=1.)
    rows = qi < Q
    for j in range(KT):
        kk = j*64+ki
        summarized = LOAD and j < PREFIX_TILES
        if summarized:
            base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + tl.arange(0, 128)
            block_z = tl.load(ZSUM+base, rows, other=-float('inf'))
            mu = tl.load(MUSUM+base[:, None]*RP+ri[None, :],
                         rows[:, None] & (ri[None, :] < R), other=0.)
            active = tl.load(ACTSUM+base, rows, other=0) != 0
            bad_row = tl.load(BADSUM+base, rows, other=0) != 0
            eligible = tl.sum((active | bad_row).to(tl.int32), 0)>0
        else:
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
            sketch = tl.load(Z+((batch*HK+kh)*K+kk[:, None])*R+ri[None, :],
                             (kk[:, None]<K) & (ri[None, :]<R), other=0.)
            mu = tl.dot(weights, sketch, input_precision='tf32x3')
            if STORE and j < PREFIX_TILES:
                base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + tl.arange(0, 128)
                tl.store(ZSUM+base, block_z, rows)
                tl.store(MUSUM+base[:, None]*RP+ri[None, :], mu,
                         rows[:, None] & (ri[None, :] < R))
                tl.store(ACTSUM+base, active.to(tl.int8), rows)
                tl.store(BADSUM+base, bad_row.to(tl.int8), rows)
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


@tr.jit(do_not_specialize=['K', 'KT'])
def _held_eligible_generic(S, SKIP, ELIGIBLE, Q: tl.constexpr, K,
                   H: tl.constexpr, QB: tl.constexpr, KT, KDIV: tl.constexpr):
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV  # v10: K arrives as K/KDIV; the product carries K's pow2 alignment
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


@tr.jit(do_not_specialize=['K', 'KT'])
def _pv_generic(S, V, SKIP, ELIGIBLE, O, LSE, INVALID, COUNTERS,
        Q: tl.constexpr, K, H: tl.constexpr, HK: tl.constexpr,
        D: tl.constexpr, QB: tl.constexpr, KT,
        TRACE: tl.constexpr, KDIV: tl.constexpr):
    q16, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV  # v10: K arrives as K/KDIV; the product carries K's pow2 alignment
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


@tr.jit(do_not_specialize=['K', 'KT'])
def _preqk_pv_generic(QQ, KK, V, SKIP, ELIGIBLE, O, LSE, INVALID, COUNTERS,
              SQB, SQH, SQL, SKB, SKH, SKL, SVB, SVH, SVL,
              Q: tl.constexpr, K, H: tl.constexpr, HK: tl.constexpr,
              D: tl.constexpr, QB: tl.constexpr, KT,
              SCALE, WINDOW: tl.constexpr, TRACE: tl.constexpr, KDIV: tl.constexpr):
    """Current-QK/PV consumer that never touches a dropped tile.

    Same 16-row output program and online-softmax accumulation as ``_pv``,
    but the scores are formed here from current Q/K instead of read from a
    materialized [B,H,Q,K] tensor. A dropped tile issues no K load, no V
    load and no dot: the branch is taken before any of them. GQA is handled
    by indexing the KV head, so no repeated K/V is materialized. Q/K/V
    strides are explicit because the model hands us transposed (non
    contiguous) [B,H,L,D] views; forcing a copy would re-introduce the
    layout cost this consumer exists to avoid. The last dimension is
    contiguous in every native layout we accept.
    """
    q16, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV  # v10: K arrives as K/KDIV; the product carries K's pow2 alignment
    kh = h // (H // HK)
    qb = q16 // 8
    qi = q16*16 + tl.arange(0, 16)
    ki = tl.arange(0, 64)
    di = tl.arange(0, D)
    rows = qi < Q
    # The query tile is loaded once per program; dropped tiles cost nothing.
    q_tile = tl.load(QQ+batch*SQB+h*SQH+qi[:, None]*SQL+di[None, :],
                     rows[:, None], other=0.)
    # Absolute query position inside this compact tensor, matching
    # _attention_validity's ``arange(q_len) + (kv_len - q_len)``.
    qpos = qi + (K - Q)
    maximum = tl.full((16,), -float('inf'), tl.float32)
    denom = tl.full((16,), 0., tl.float32)
    output = tl.full((16, D), 0., tl.float32)
    invalid = tl.full((16,), False, tl.int1)
    visited = 0
    kvloads = 0
    qkdots = 0
    for j in range(KT):
        dest = ((batch*H+h)*QB+qb)*KT+j
        eligible = tl.load(ELIGIBLE+dest)
        drop = tl.load(SKIP+dest)
        if eligible & ~drop:
            visited += 1
            kk = j*64+ki
            k_tile = tl.load(KK+batch*SKB+kh*SKH+kk[:, None]*SKL+di[None, :],
                             kk[:, None] < K, other=0.)
            kvloads += 1
            score = tl.dot(q_tile, tl.trans(k_tile))
            qkdots += 1
            # Reference rounding (observe_scores): BF16 matmul output, then
            # BF16 scaling, then FP32 for the softmax. Torch computes the
            # scalar multiply in FP32 opmath and rounds back to BF16.
            score = (score.to(tl.bfloat16).to(tl.float32)*SCALE).to(tl.bfloat16).to(tl.float32)
            valid = rows[:, None] & (kk[None, :] < K)
            if WINDOW > 0:
                valid = valid & (kk[None, :] >= (qpos[:, None] - WINDOW + 1))
            bad = valid & ((score != score) | (score == float('inf')))
            invalid = invalid | (tl.sum(bad.to(tl.int32), 1) > 0)
            finite = valid & (score > -float('inf')) & (score < float('inf'))
            clean = tl.where(finite, score, -float('inf'))
            tile_max = tl.max(clean, 1)
            next_max = tl.maximum(maximum, tile_max)
            safe = tl.where(next_max > -float('inf'), next_max, 0.)
            old_scale = tl.where(maximum > -float('inf'), lib.exp(maximum-safe), 0.)
            p = lib.exp(clean-safe[:, None])
            denom = old_scale*denom+tl.sum(p, 1)
            output = old_scale[:, None]*output
            v = tl.load(V+batch*SVB+kh*SVH+kk[:, None]*SVL+di[None, :],
                        kk[:, None] < K, other=0.)
            output += tl.dot(p.to(tl.bfloat16), v)
            maximum = next_max
    normalized = output/tl.maximum(denom[:, None], 1.e-30)
    normalized = tl.where((denom > 0)[:, None] & ~invalid[:, None], normalized, 0.)
    logz = tl.where((denom > 0) & ~invalid,
                    maximum+lib.log(tl.maximum(denom, 1.e-30)), -float('inf'))
    tl.store(O+((batch*H+h)*Q+qi[:, None])*D+di[None, :], normalized.to(tl.bfloat16), rows[:, None])
    tl.store(LSE+(batch*H+h)*Q+qi, logz, rows)
    tl.store(INVALID+(batch*H+h)*Q+qi, invalid, rows)
    if TRACE:
        base = ((batch*H+h)*tr.cdiv(Q, 16)+q16)*3
        tl.store(COUNTERS+base+0, visited)
        tl.store(COUNTERS+base+1, kvloads)
        tl.store(COUNTERS+base+2, qkdots)
