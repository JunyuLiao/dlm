"""v27 M1-DP: the M1 risk against the DENSE prefix state, precomputed at the summary build.

Fan's M1 risk for adding KV64 tile j to the retained state is, per query row,
    log( || alpha_j (mu_j - P) || / ref ) + log T,
where P and the log mass behind alpha are those of the tiles KEPT so far. Because the kept
set depends on every earlier decision, the selector must scan the tiles in order on every
decision call (see v27_route.py for the fastest exact form of that scan).

M1-DP (a named variant; its decisions are NOT those of M1) replaces the kept state by the
dense state of ALL earlier tiles. Every prefix tile's ``log || alpha_j (mu_j - P_dense) ||``
then depends only on the frozen prefix summary, so it is computed once when the summary is
built (``build``). A decision call reads one FP32 per (tile, row) and evaluates
    worst_j = max_rows( lognorm_{j,row} + log T_row ) - log ref,   drop = eligible & worst < THR
for all prefix tiles in parallel (``_dp_decide``), then scans only the few tail tiles
(canvas and boundary) from the stored dense prefix state (``_dp_tail``). The first support,
bad-row and eligibility conventions are the same as the kept-state selector's.

Because the dense mass is at least the kept mass, alpha_dense <= alpha_kept: at an equal
threshold M1-DP keeps no more of a tile's mass share, so its realized sparsity must be
measured and its threshold calibrated separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .cached_executor import _logadd


@dataclass
class DensePrefixState:
    lognorm: torch.Tensor   # [B,H,QB,PT,128] FP32
    eligible: torch.Tensor  # [B,H,QB,PT] int8
    bad: torch.Tensor       # [B,H,QB,PT] int8
    previous: torch.Tensor  # [B,H,QB,128] FP32 dense prefix log mass
    projected: torch.Tensor  # [B,H,QB,128,32] FP32 dense prefix projected mean
    prefix_tiles: int
    identity: tuple

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.lognorm, self.eligible, self.bad,
                                                          self.previous, self.projected))


@tr.jit(do_not_specialize=['KT', 'PREFIX_TILES'])
def _dp_build(ZSUM, MUSUM, ACTSUM, BADSUM, POOLED, LOGN, DPELIG, DPBAD, DPPREV, DPPROJ,
              Q: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
              QB: tl.constexpr, KT, PREFIX_TILES, COMPACT: tl.constexpr):
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (H // HK)
    r128 = tl.arange(0, 128)
    ri = tl.arange(0, RP)
    rows = qb*128 + r128 < Q
    previous = tl.full((128,), -float('inf'), tl.float32)
    projected = tl.full((128, RP), 0., tl.float32)
    for j in range(PREFIX_TILES):
        base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + r128
        block_z = tl.load(ZSUM+base, rows, other=-float('inf'))
        if COMPACT:
            pooled = tl.load(POOLED+((batch*HK+kh)*KT+j)*RP+ri, ri < R, other=0.)
            mu = tl.zeros((128, RP), tl.float32) + pooled[None, :]
        else:
            mu = tl.load(MUSUM+base[:, None]*RP+ri[None, :], rows[:, None] & (ri[None, :] < R), other=0.)
        active = tl.load(ACTSUM+base, rows, other=0) != 0
        bad_row = tl.load(BADSUM+base, rows, other=0) != 0
        eligible = tl.sum((active | bad_row).to(tl.int32), 0) > 0
        combined = _logadd(previous, block_z)
        safe = tl.where(combined > -float('inf'), combined, 0.)
        alpha = tl.where(active, lib.exp(block_z-safe), 0.)
        delta = alpha[:, None]*(mu-projected)
        lognorm = lib.log(tl.sqrt(tl.sum(delta*delta, 1)))
        lognorm = tl.where(active, tl.where(previous > -float('inf'), lognorm, float('inf')), -float('inf'))
        lognorm = tl.where(bad_row, float('inf'), lognorm)
        tl.store(LOGN+base, lognorm, rows)
        tile = ((batch*H+h)*QB+qb)*PREFIX_TILES + j
        tl.store(DPELIG+tile, eligible.to(tl.int8))
        tl.store(DPBAD+tile, (tl.sum(bad_row.to(tl.int32), 0) > 0).to(tl.int8))
        old = lib.exp(previous-safe)
        projected = tl.where(eligible, old[:, None]*projected+alpha[:, None]*mu, projected)
        previous = tl.where(eligible, combined, previous)
    state = ((batch*H+h)*QB+qb)*128 + r128
    tl.store(DPPREV+state, previous, rows)
    tl.store(DPPROJ+state[:, None]*RP+ri[None, :], projected, rows[:, None] & (ri[None, :] < R))


@tr.jit(do_not_specialize=['KT', 'PREFIX_TILES'])
def _dp_decide(LOGN, DPELIG, DPBAD, REF, T, SKIP, ELIGIBLE, BADTILE,
               Q: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, QB: tl.constexpr, KT, PREFIX_TILES,
               THRESHOLD: tl.constexpr, TILES: tl.constexpr):
    qb, bh, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, h = bh // H, bh % H
    kh = h // (H // HK)
    r128 = tl.arange(0, 128)
    qi = qb*128 + r128
    rows = qi < Q
    log_t = lib.log(tl.load(T+batch*Q+qi, rows, other=1.))
    log_ref = lib.log(tl.maximum(tl.load(REF+batch*HK+kh), 1e-12))
    for t in tl.static_range(TILES):
        j = chunk*TILES + t
        if j < PREFIX_TILES:
            base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + r128
            lognorm = tl.load(LOGN+base, rows, other=-float('inf'))
            worst = tl.max(tl.where(rows, lognorm - log_ref + log_t, -float('inf')), 0)
            tile = ((batch*H+h)*QB+qb)*PREFIX_TILES + j
            eligible = tl.load(DPELIG+tile) != 0
            dest = ((batch*H+h)*QB+qb)*KT+j
            tl.store(SKIP+dest, eligible & (worst < THRESHOLD))
            tl.store(ELIGIBLE+dest, eligible)
            tl.store(BADTILE+dest, tl.load(DPBAD+tile) != 0)


@tr.jit(do_not_specialize=['K', 'KT', 'PREFIX_TILES'])
def _dp_tail(S, Z, REF, T, SKIP, ELIGIBLE, BADTILE, DPPREV, DPPROJ, POOLED, POOLCNT, POOLBAD, KOFF, KSTORE,
             Q: tl.constexpr, K, H: tl.constexpr, HK: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
             QB: tl.constexpr, KT, THRESHOLD: tl.constexpr, PREFIX_TILES, POOL: tl.constexpr,
             COMPACT: tl.constexpr, KDIV: tl.constexpr, TAIL: tl.constexpr):
    qb, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    K = K * KDIV
    kh = h // (H // HK)
    qi = qb*128 + tl.arange(0, 128)
    ki = tl.arange(0, 64)
    ri = tl.arange(0, RP)
    rows = qi < Q
    state = ((batch*H+h)*QB+qb)*128 + tl.arange(0, 128)
    if PREFIX_TILES > 0:
        previous = tl.load(DPPREV+state, rows, other=-float('inf'))
        projected = tl.load(DPPROJ+state[:, None]*RP+ri[None, :], rows[:, None] & (ri[None, :] < R), other=0.)
    else:
        previous = tl.full((128,), -float('inf'), tl.float32)
        projected = tl.full((128, RP), 0., tl.float32)
    reference = tl.maximum(tl.load(REF+batch*HK+kh), 1e-12)
    sensitivity = tl.load(T+batch*Q+qi, rows, other=1.)
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
        risk = lib.log(norm)-lib.log(reference)+lib.log(sensitivity)
        risk = tl.where(active, tl.where(previous > -float('inf'), risk, float('inf')), -float('inf'))
        risk = tl.where(bad_row, float('inf'), risk)
        worst = tl.max(risk, 0)
        dest = ((batch*H+h)*QB+qb)*KT+j
        tl.store(SKIP+dest, eligible & (worst < THRESHOLD))
        tl.store(ELIGIBLE+dest, eligible)
        tl.store(BADTILE+dest, tl.sum(bad_row.to(tl.int32), 0) > 0)
        # dense state: every eligible tile joins, kept or not
        old = lib.exp(previous-safe)
        projected = tl.where(eligible, old[:, None]*projected+alpha[:, None]*mu, projected)
        previous = tl.where(eligible, combined, previous)


def budget_skip(lognorm, eligible, sensitivity, reference, nq, log_budget):
    """v27 risk-budget selector (named variant on M1-DP): the prefix tiles a row block may drop are bounded by
    their SUMMED risk instead of each tile's own risk. Per (batch, head, query block) the eligible prefix tiles are
    taken in ascending order of the M1-DP worst-row risk exp(worst_j) and dropped while the running sum stays below
    exp(log_budget). Tiles with infinite risk (first support, bad rows) are never dropped; tiles with no active
    rows (risk 0) always are, as in the per-tile rule. Returns the [B,H,QB,PT] drop map."""
    b, h, qb, pt, width = lognorm.shape
    hk = reference.shape[1]
    dev = lognorm.device
    rows = (torch.arange(qb * width, device=dev) < nq).view(1, 1, qb, 1, width)
    log_t = torch.zeros((b, qb * width), device=dev, dtype=torch.float32)
    log_t[:, :nq] = torch.log(sensitivity.float()[:, :nq])
    value = lognorm.float() + log_t.view(b, 1, qb, 1, width)
    value = torch.where(rows, value, torch.full_like(value, float('-inf')))
    worst = value.amax(-1)
    log_ref = torch.log(reference.float().clamp_min(1e-12))
    worst = worst - log_ref[:, torch.arange(h, device=dev) // (h // hk)].view(b, h, 1, 1)
    candidate = (eligible != 0) & (worst < float('inf')) & ~torch.isnan(worst)
    key = torch.where(candidate, worst, torch.full_like(worst, float('inf')))
    ordered, order = key.sort(-1)
    running = torch.cumsum(torch.exp(ordered.double() - float(log_budget)), -1)
    drop = torch.zeros_like(candidate).scatter(-1, order, running < 1.0)
    return drop & candidate


def _worst_risk(lognorm, sensitivity, reference, nq):
    b, h, qb, pt, width = lognorm.shape
    hk = reference.shape[1]
    dev = lognorm.device
    rows = (torch.arange(qb * width, device=dev) < nq).view(1, 1, qb, 1, width)
    log_t = torch.zeros((b, qb * width), device=dev, dtype=torch.float32)
    log_t[:, :nq] = torch.log(sensitivity.float()[:, :nq])
    value = lognorm.float() + log_t.view(b, 1, qb, 1, width)
    value = torch.where(rows, value, torch.full_like(value, float('-inf')))
    worst = value.amax(-1)
    log_ref = torch.log(reference.float().clamp_min(1e-12))
    return worst - log_ref[:, torch.arange(h, device=dev) // (h // hk)].view(b, h, 1, 1)


def topk_skip(lognorm, eligible, sensitivity, reference, nq, keep):
    """v27 target-sparsity selector (named variant on M1-DP): per (batch, head, query block) keep the ``keep``
    fraction of the eligible prefix tiles with the highest M1-DP worst-row risk and drop the rest, so the prefix
    sparsity is fixed at 1 - keep at every context length (the protocol of target-sparsity comparisons). Tiles with
    infinite risk (first support, bad rows) are never dropped."""
    import math
    worst = _worst_risk(lognorm, sensitivity, reference, nq)
    candidate = (eligible != 0) & (worst < float('inf')) & ~torch.isnan(worst)
    pt = worst.shape[-1]
    n_drop = min(pt, int(math.floor((1.0 - float(keep)) * pt + 1e-9)))   # (1 - 0.6) * 10 is 3.999...
    if n_drop <= 0:
        return torch.zeros_like(candidate)
    key = torch.where(candidate, worst, torch.full_like(worst, float('inf')))
    order = torch.argsort(key, dim=-1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(pt, device=worst.device).expand_as(order))
    return candidate & (rank < n_drop)


def log_mass(summary, lognorm):
    """v27 score-only control (named variant ``risk_value='mass'``): per (tile, row) the log share of the row's
    prefix attention mass that the tile holds, log(exp(z_j) / sum_j' exp(z_j')), from the same observed prefix
    summary the M1 risk uses -- no V term. Tiles whose M1-DP risk is +inf (first support, bad rows) stay +inf, so
    the never-drop conventions are identical; rows without legal keys in a tile are -inf."""
    z = torch.where(summary.active != 0, summary.z.float(), torch.full_like(summary.z, float('-inf')))
    lse = torch.logsumexp(z, dim=3, keepdim=True)
    mass = torch.where(torch.isfinite(lse), z - lse, torch.full_like(z, float('-inf')))
    return torch.where(torch.isposinf(lognorm), lognorm, mass)


def build(summary, nq, kt, hk, pooled=None, identity=None, num_warps=4, num_stages=3):
    """Dense-prefix pass over a filled prefix summary (once per summary)."""
    b, h, qb, pt, _ = summary.z.shape
    compact = pooled is not None
    if not compact and summary.mu.shape[-1] != 32:
        raise ValueError('M1-DP on exact mu needs the rank-32 summary')
    dev = summary.z.device
    state = DensePrefixState(
        lognorm=torch.empty((b, h, qb, pt, 128), device=dev, dtype=torch.float32),
        eligible=torch.empty((b, h, qb, pt), device=dev, dtype=torch.int8),
        bad=torch.empty((b, h, qb, pt), device=dev, dtype=torch.int8),
        previous=torch.empty((b, h, qb, 128), device=dev, dtype=torch.float32),
        projected=torch.zeros((b, h, qb, 128, 32), device=dev, dtype=torch.float32),
        prefix_tiles=pt, identity=identity if identity is not None else summary.identity)
    dummy = torch.empty((1,), device=dev, dtype=torch.float32)
    _dp_build[(qb, h, b)](summary.z, summary.mu if not compact else dummy, summary.active, summary.bad,
                          pooled if compact else dummy, state.lognorm, state.eligible, state.bad,
                          state.previous, state.projected, nq, h, hk, 32, 32,
                          qb, kt, pt, compact, num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return state


def route(scores, z, reference, state, *, sensitivity=None, log_threshold, pool=False, pooled=None,
          pool_count=None, key_offset=0, tiles_per_program=16, num_warps=4, num_stages=3, risk_budget=None,
          risk_topk=None, risk_value=None, summary=None):
    """M1-DP decision call: parallel prefix decisions plus the dense-state tail scan."""
    from .cached_executor import Routing, _extra, _karg, _pool_arguments
    b, h, nq, stored = scores.shape
    nk = stored + key_offset if key_offset else stored
    hk = z.shape[1]
    kt, qb = tr.cdiv(nk, 64), tr.cdiv(nq, 128)
    pt = state.prefix_tiles
    if key_offset and key_offset != pt * 64:
        raise ValueError('tail scores must start exactly after the dense-prefix summary')
    if tuple(state.lognorm.shape) != (b, h, qb, pt, 128):
        raise ValueError('dense-prefix state does not match this call geometry')
    if sensitivity is None:
        sensitivity = torch.ones((b, nq), device=scores.device, dtype=torch.float32)
    shape = (b, h, qb, kt)
    skip = torch.empty(shape, device=scores.device, dtype=torch.bool)
    elig = torch.empty(shape, device=scores.device, dtype=torch.bool)
    bad = torch.empty(shape, device=scores.device, dtype=torch.bool)
    compact = pool == 'compact'
    pool_args = _pool_arguments(pooled, pool_count, shape, scores.device, hk)
    if pt:
        _dp_decide[(qb, b * h, tr.cdiv(pt, tiles_per_program))](
            state.lognorm, state.eligible, state.bad, reference, sensitivity, skip, elig, bad,
            nq, h, hk, qb, kt, pt, log_threshold, tiles_per_program, num_warps=4)
        if risk_budget is not None:
            skip[..., :pt] = budget_skip(state.lognorm, state.eligible, sensitivity, reference, nq, risk_budget)
        elif risk_topk is not None:
            ranked = state.lognorm
            if risk_value == 'mass':
                # score-only control: rank by attention mass share instead of the M1 risk (cached per state)
                ranked = getattr(state, 'logmass', None)
                if ranked is None:
                    if summary is None:
                        raise ValueError('mass ranking needs the prefix summary')
                    ranked = state.logmass = log_mass(summary, state.lognorm)
            elif risk_value is not None:
                raise ValueError(f'unknown risk value {risk_value!r}')
            skip[..., :pt] = topk_skip(ranked, state.eligible, sensitivity, reference, nq, risk_topk)
    _dp_tail[(qb, h, b)](scores, z, reference, sensitivity, skip, elig, bad, state.previous, state.projected,
                         *pool_args, key_offset, nk - key_offset, nq, _karg('generic', nk), h, hk, 32, 32, qb, kt,
                         log_threshold, pt, pool is True, compact, TAIL=bool(key_offset),
                         num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False,
                         **_extra('generic', nk))
    return Routing(skip, elig, bad, pool_args[2] if compact else None)
