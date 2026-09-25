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
class PrefixSummary:
    """Exact per-row block summaries for WHOLLY-immutable prefix KV tiles.

    ``z``/``mu`` are the anchor's own FP32 ``block_z`` and ``mu`` values, so a
    reusing call consumes identical bits rather than a recomputation that
    merely ought to agree. Nothing decision-dependent is stored: no alpha, no
    risk, no retained accumulator, no bitmap.

    ``identity`` must carry everything the values depend on -- score-anchor
    generation, prefix owner/version, encoder epoch, projection identity,
    layer/head/layout, mask/scale/dtype, crop and the prefix boundary -- and is
    compared by value. A data pointer alone never authorizes reuse.
    """
    z: torch.Tensor
    mu: torch.Tensor
    active: torch.Tensor
    bad: torch.Tensor
    prefix_tiles: int
    identity: tuple

    def __post_init__(self):
        # Fixed at construction so live accounting is an O(1) host read.
        self.nbytes = self.bytes

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.z, self.mu, self.active, self.bad))

    def matches(self, identity, prefix_tiles):
        return self.identity == identity and self.prefix_tiles == prefix_tiles


def summary_bytes(b, h, qb, prefix_tiles, rank):
    """Bytes allocate_summary would take: FP32 z, FP32 mu[rank], int8 active/bad."""
    if prefix_tiles <= 0:
        return 0
    cells = b * h * qb * int(prefix_tiles) * 128
    return cells * (4 + 4 * rank + 1 + 1)


def allocate_summary(b, h, qb, kt, prefix_tiles, rank, device, identity):
    """Buffers sized for the leading ``prefix_tiles`` tiles only."""
    if prefix_tiles <= 0:
        return None
    shape = (b, h, qb, int(prefix_tiles), 128)
    return PrefixSummary(
        z=torch.empty(shape, device=device, dtype=torch.float32),
        mu=torch.empty(shape + (rank,), device=device, dtype=torch.float32),
        active=torch.empty(shape, device=device, dtype=torch.int8),
        bad=torch.empty(shape, device=device, dtype=torch.int8),
        prefix_tiles=int(prefix_tiles), identity=identity)


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
               ZSUM, MUSUM, ACTSUM, BADSUM,
               Q: tl.constexpr, K: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
               R: tl.constexpr, RP: tl.constexpr, QB: tl.constexpr,
               KT: tl.constexpr, THRESHOLD: tl.constexpr, TRACE: tl.constexpr,
               PREFIX_TILES: tl.constexpr, STORE: tl.constexpr, LOAD: tl.constexpr):
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


if tr is not None:
    @tr.jit
    def _preqk_pv(QQ, KK, V, SKIP, ELIGIBLE, O, LSE, INVALID, COUNTERS,
                  SQB, SQH, SQL, SKB, SKH, SKL, SVB, SVH, SVL,
                  Q: tl.constexpr, K: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
                  D: tl.constexpr, QB: tl.constexpr, KT: tl.constexpr,
                  SCALE, WINDOW: tl.constexpr, TRACE: tl.constexpr):
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


VARIANTS = ('static', 'generic')


_KERNEL_TABLES = {}


def _kernels(variant):
    """static: exact-length constexpr kernels (reference). generic: v10 CP1
    length-generic twins (runtime K/KT/PREFIX_TILES), textually identical bodies."""
    table = _KERNEL_TABLES.get(variant)
    if table is not None:
        return table
    if variant == 'static':
        table = dict(route=_route, held=_held_eligible, pv=_pv, preqk=_preqk_pv)
    elif variant == 'generic':
        from . import generic_kernels as g
        table = dict(route=g._route_generic, held=g._held_eligible_generic,
                     pv=g._pv_generic, preqk=g._preqk_pv_generic)
    else:
        raise ValueError(variant)
    _KERNEL_TABLES[variant] = table
    return table


def _kdiv(nk):
    kdiv = 1
    while kdiv < 16 and nk % (kdiv * 2) == 0:
        kdiv *= 2
    return kdiv


def _karg(variant, nk):
    """The K argument: exact for static; K // KDIV for the generic twins."""
    return nk if variant != 'generic' else nk // _kdiv(nk)


def _extra(variant, nk):
    """Launch-time extras: the generic twins take K's power-of-two alignment."""
    return {} if variant != 'generic' else {'KDIV': _kdiv(nk)}


def attention(scores, v, z=None, reference=None, *, sensitivity=None,
              log_threshold=-math.inf, skipped=None, eligible=None,
              trace=False, num_warps=8, summary=None, store_summary=False,
              variant='static'):
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
        prefix_tiles, zs, mus, acts, bads = _summary_arguments(summary, store_summary, shape, kt,
                                                               scores.device)
        kernels = _kernels(variant)
        kernels['route'][(qb, h, b)](scores, z, reference, sensitivity, skip, elig, bad_tiles, lse, state, risk,
                           zs, mus, acts, bads,
                           nq, _karg(variant, nk), h, hk, 32, 32, qb, kt, log_threshold, trace,
                           prefix_tiles, bool(store_summary),
                           bool(summary is not None and not store_summary),
                           num_warps=num_warps, num_stages=1, enable_fp_fusion=False, **_extra(variant, nk))
    elif summary is not None or store_summary:
        raise ValueError('summaries apply to the selector, not the held-bitmap path')
    elif eligible is None:
        _kernels(variant)['held'][(qb, h, b)](scores, skip, elig, nq, _karg(variant, nk), h, qb, kt,
                                    num_warps=4, num_stages=1, **_extra(variant, nk))
    _kernels(variant)['pv'][(tr.cdiv(nq, 16), h, b)](scores, v, skip, elig, out, lse, invalid, counters,
                                 nq, _karg(variant, nk), h, hk, d, qb, kt, trace,
                                 num_warps=num_warps, num_stages=1, enable_fp_fusion=False, **_extra(variant, nk))
    return Output(out, skip, elig, lse, state, risk, invalid, counters)


def route_only(scores, z, reference, *, sensitivity=None, log_threshold=-math.inf,
               num_warps=8, summary=None, store_summary=False, variant='static'):
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
    prefix_tiles, zs, mus, acts, bads = _summary_arguments(summary, store_summary, shape, kt,
                                                           scores.device)
    _kernels(variant)['route'][(qb, h, b)](scores, z, reference, sensitivity, skip, elig, bad_tiles,
                       dummy, dummy, dummy, zs, mus, acts, bads,
                       nq, _karg(variant, nk), h, hk, 32, 32, qb, kt, log_threshold, False,
                       prefix_tiles, bool(store_summary),
                       bool(summary is not None and not store_summary),
                       num_warps=num_warps, num_stages=1, enable_fp_fusion=False, **_extra(variant, nk))
    return Routing(skip, elig, bad_tiles)


def _summary_arguments(summary, store_summary, shape, kt, device):
    """Validate and unpack summary buffers for a ``_route`` launch."""
    if summary is None:
        dummy = torch.empty((1,), device=device, dtype=torch.float32)
        small = torch.empty((1,), device=device, dtype=torch.int8)
        if store_summary:
            raise ValueError('store_summary requires summary buffers')
        return 0, dummy, dummy, small, small
    if not isinstance(summary.prefix_tiles, int) or not 0 < summary.prefix_tiles <= kt:
        raise ValueError('summary needs a positive prefix tile count within the key tiling')
    expected = shape[:3] + (summary.prefix_tiles, 128)
    for name, tensor, want_shape, want_dtype in (
            ('z', summary.z, expected, torch.float32),
            ('mu', summary.mu, expected + (32,), torch.float32),
            ('active', summary.active, expected, torch.int8),
            ('bad', summary.bad, expected, torch.int8)):
        if tuple(tensor.shape) != want_shape:
            raise ValueError(f'summary {name} shape {tuple(tensor.shape)} != {want_shape}')
        if tensor.dtype != want_dtype:
            raise ValueError(f'summary {name} dtype {tensor.dtype} != {want_dtype}')
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError('summary buffers must be contiguous and co-located')
    return summary.prefix_tiles, summary.z, summary.mu, summary.active, summary.bad


def preqk_attention(q, k, v, skipped, eligible, *, scale, window=None,
                    is_causal=False, trace=False, num_warps=8, variant='static'):
    """Current attention on a preselected support, skipping dropped tiles' QK.

    ``skipped``/``eligible`` are the [B,H,Qtiles,Ktiles] bitmaps produced
    earlier by ``route_only`` from the historical scores. Unlike
    ``attention(scores, ...)``, no [B,H,Q,K] score tensor exists: each output
    program forms current QK for retained tiles only.

    Declared semantic difference from the score-consuming path: a *dropped*
    tile's scores are never computed, so malformed values inside a dropped
    tile cannot be detected here. That is intended -- those scores do not
    reach the output -- and the historical scores that chose the support are
    separately checked via ``route_only``'s ``invalid_tiles``. Retained tiles
    are still checked and still zero + flag their rows.
    """
    if tr is None:
        raise RuntimeError('Triton is required for the CUDA cached-score executor')
    if is_causal:
        raise ValueError('preqk consumer is qualified for the bidirectional decoder only')
    for name, x in (('q', q), ('k', k), ('v', v)):
        if x.ndim != 4 or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError(f'{name} must be a CUDA BF16 [B,heads,len,D] tensor')
        if x.stride(-1) != 1:
            raise ValueError(f'{name} must have a contiguous head dimension')
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[-2]
    if k.shape != v.shape or k.shape[0] != b or q.device != k.device or v.device != k.device:
        raise ValueError('k and v must share [B,KVH,K,D] geometry on q\'s device')
    if not nq or not nk or not hk or h % hk or d != k.shape[-1] or d not in (64, 128, 256, 512):
        raise ValueError('invalid dimensions/GQA')
    if nk < nq:
        raise ValueError('compact key range must contain the current queries')
    qb, kt = tr.cdiv(nq, 128), tr.cdiv(nk, 64)
    shape = (b, h, qb, kt)
    for name, x in (('skipped', skipped), ('eligible', eligible)):
        if x is None or x.shape != shape or x.dtype != torch.bool or x.device != q.device or not x.is_contiguous():
            raise ValueError(f'{name} must be a contiguous CUDA bool {shape} bitmap')
    bound = 0 if not window else int(window)
    if bound < 0:
        raise ValueError('window must be nonnegative')
    out = torch.empty((b, h, nq, d), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, h, nq), device=q.device, dtype=torch.float32)
    invalid = torch.empty((b, h, nq), device=q.device, dtype=torch.bool)
    # v10 glue: the one-element stand-in is never written when trace=False, so
    # it needs no memset; trace=True keeps the zero-initialized counters.
    counters = (torch.zeros if trace else torch.empty)((b, h, tr.cdiv(nq, 16), 3) if trace else (1,),
                           device=q.device, dtype=torch.int32)
    _kernels(variant)['preqk'][(tr.cdiv(nq, 16), h, b)](q, k, v, skipped, eligible, out, lse, invalid, counters,
                                       q.stride(0), q.stride(1), q.stride(2),
                                       k.stride(0), k.stride(1), k.stride(2),
                                       v.stride(0), v.stride(1), v.stride(2),
                                       nq, _karg(variant, nk), h, hk, d, qb, kt, float(scale), bound, trace,
                                       num_warps=num_warps, num_stages=1, enable_fp_fusion=False, **_extra(variant, nk))
    state = torch.empty((1,), device=q.device, dtype=torch.float32)
    return Output(out, skipped, eligible, lse, state, state, invalid, counters)


def warmup_generic(thresholds, device='cuda', *, geometries=None, canvas=256):
    """Deployment warmup for the generic twins, OUTSIDE request latency.

    Compiles the bounded production-mode set on synthetic tensors: for each
    decoder geometry (LOCAL h16/hk8/d256/window1024, GLOBAL h16/hk2/d512) and
    each of the 5 K alignment classes (KDIV 1..16): anchor route+PV with and
    without summary STORE, ordinary route_only with and without summary LOAD,
    and the preqk consumer -- all with trace=False, as in production. Uses the
    frozen policy thresholds because THRESHOLD is a compile-time mode. Never
    touches the model or any answer. Returns (seconds, launches).
    """
    import time
    geometries = geometries or (('local', 16, 8, 256, 1024), ('global', 16, 2, 512, None))
    start, launches = time.perf_counter(), 0
    for kind, h, hk, d, window in geometries:
        threshold = float(thresholds[kind]['log_threshold'])
        for kdiv in (1, 2, 4, 8, 16):
            nk = 1280 + kdiv % 16            # 1281, 1282, 1284, 1288, 1280
            assert _kdiv(nk) == kdiv, (nk, kdiv)
            q = torch.zeros((1, canvas, h, d), device=device, dtype=torch.bfloat16).transpose(1, 2)
            k = torch.zeros((1, hk, nk, d), device=device, dtype=torch.bfloat16)
            v = torch.zeros((1, hk, nk, d), device=device, dtype=torch.bfloat16)
            scores = torch.zeros((1, h, canvas, nk), device=device, dtype=torch.float32)
            z = torch.zeros((1, hk, nk, 32), device=device, dtype=torch.float32)
            ref = torch.ones((1, hk), device=device, dtype=torch.float32)
            sens = torch.ones((1, canvas), device=device, dtype=torch.float32)
            qb, kt = tr.cdiv(canvas, 128), tr.cdiv(nk, 64)
            for store in (False, True):
                summary = allocate_summary(1, h, qb, kt, 16, 32, device, ('warmup',)) if store else None
                attention(scores, v, z, ref, sensitivity=sens, log_threshold=threshold, summary=summary,
                          store_summary=store, variant='generic')
                routed = route_only(scores, z, ref, sensitivity=sens, log_threshold=threshold,
                                    summary=summary, store_summary=False, variant='generic')
                preqk_attention(q, k, v, routed.skipped, routed.eligible, scale=d ** -.5, window=window,
                                variant='generic')
                launches += 5
    torch.cuda.synchronize()
    return time.perf_counter() - start, launches


if tr is not None:
    @tr.jit
    def _guard(OUT, INVALID, TILES, PARTIAL, N_OUT, N_ROWS, N_TILES,
               BLOCK: tl.constexpr, HAS_TILES: tl.constexpr):
        """One pass over the three guarded conditions; each program writes its own
        'ok' slot, so no zero-initialization launch is needed."""
        pid = tl.program_id(0)
        step = tl.num_programs(0) * BLOCK
        bad = tl.zeros((BLOCK,), tl.int32)
        for start in range(pid * BLOCK, N_OUT, step):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(OUT + offs, offs < N_OUT, other=0.).to(tl.float32)
            bad = bad | ((x != x) | (x == float('inf')) | (x == -float('inf'))).to(tl.int32)
        for start in range(pid * BLOCK, N_ROWS, step):
            offs = start + tl.arange(0, BLOCK)
            bad = bad | tl.load(INVALID + offs, offs < N_ROWS, other=0).to(tl.int32)
        if HAS_TILES:
            for start in range(pid * BLOCK, N_TILES, step):
                offs = start + tl.arange(0, BLOCK)
                bad = bad | tl.load(TILES + offs, offs < N_TILES, other=0).to(tl.int32)
        tl.store(PARTIAL + pid, tl.max(bad, 0) == 0)


def guard_flags(output, invalid_rows, invalid_tiles=None, programs=64):
    """Per-program 'ok' flags covering exactly the separate guards' conditions:
    non-finite attention output (torch.isfinite), invalid score rows, and
    malformed cached-score tiles of the routing decision (when given)."""
    for name, x in (('output', output), ('invalid_rows', invalid_rows)) + \
            ((('invalid_tiles', invalid_tiles),) if invalid_tiles is not None else ()):
        if not x.is_cuda or not x.is_contiguous():
            raise ValueError(f'{name} must be a contiguous CUDA tensor')
    partial = torch.empty((programs,), device=output.device, dtype=torch.bool)
    tiles = invalid_tiles if invalid_tiles is not None else invalid_rows
    _guard[(programs,)](output, invalid_rows, tiles, partial, output.numel(), invalid_rows.numel(),
                        tiles.numel(), BLOCK=1024, HAS_TILES=invalid_tiles is not None, num_warps=4)
    return partial


def fused_guard(output, invalid_rows, invalid_tiles=None):
    """Same coverage as the three separate _assert_async guards, one reduction chain."""
    torch._assert_async(guard_flags(output, invalid_rows, invalid_tiles).all(),
                        'Invalid attention: non-finite output, invalid score rows or malformed routed tiles')
