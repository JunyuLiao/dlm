"""Hopper selector kernels. Statistics reuse the prior Gaussian32 softmax/PZ design.

No full token attention matrix, full-dimensional PV, or host mask decisions.
Online state is normalized FP32 output plus log mass, including on V2 skips.
Exact greedy scores all candidates in parallel from cached summaries; batch8
is a separately named approximation. Only maps are consumed by V31 FA4.
"""
import math

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .v31_value_selectors import Stats, EPS, SELECTORS, full_support, mandatory_map


@tr.jit(do_not_specialize=['N','NK','JT','KS0'])
def _stats(Q, K, Z, LM, MU, BAD, N, NK,
           H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, JT,
           QB: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr, KS0, KS1: tl.constexpr, SCALE: tl.constexpr):
    chunk, h, j = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qi = chunk*32+tl.arange(0, 32)
    kk = j*64+tl.arange(0, 64)
    dd = tl.arange(0, D)
    rr = tl.arange(0, 32)
    kh = h//(H//HK)
    q = tl.load(Q+h*QS0+qi[:, None]*QS1+dd[None], qi[:, None]<N, 0.)
    k = tl.load(K+kh*KS0+kk[None]*KS1+dd[:, None], kk[None]<NK, 0.)
    scores = tl.dot(q, k)*SCALE
    legal = (qi[:, None]<N)&(kk[None]<NK)
    bad = tl.sum((legal & ((scores != scores)|(scores == float('inf')))).to(tl.int32), 1)>0
    scores = tl.where(legal, scores, -float('inf'))
    mx = tl.max(scores, 1)
    active = mx > -float('inf')
    safe = tl.where(active, mx, 0.)
    weights = lib.exp(scores-safe[:, None])
    den = tl.sum(weights, 1)
    weights = weights/tl.maximum(den, 1.e-30)[:, None]
    z = tl.load(Z+(kh*NK+kk[:, None])*32+rr[None], kk[:, None]<NK, 0.)
    mu = tl.dot(weights, z, input_precision='tf32x3')
    bad |= tl.sum(((mu!=mu) | (tl.abs(mu)==float('inf'))).to(tl.int32),1)>0
    dest = ((h*QB+qi//128)*JT+j)*128+qi%128
    tl.store(LM+dest, tl.where(active & ~bad, safe+lib.log(tl.maximum(den, 1.e-30)), -float('inf')))
    tl.store(MU+dest[:, None]*32+rr[None], tl.where(bad[:, None], 0., mu))
    tl.store(BAD+dest, bad)


def statistics_cuda(q, k, sketch, nu_kv, scale, prefix_tiles):
    _, h, n, d = q.shape
    hk, nk = k.shape[1:3]
    if h % hk or sketch.shape != (1, hk, nk, 32) or q.dtype != torch.bfloat16:
        raise ValueError('native BF16 GQA and FP32 Gaussian32 required')
    qb, kt = tr.cdiv(n, 128), tr.cdiv(nk, 64)
    shape = (h*qb, kt, 128)
    if n % 128:
        lm = torch.full(shape, -math.inf, dtype=torch.float32, device=q.device)
        mu = torch.zeros((*shape, 32), dtype=torch.float32, device=q.device)
        invalid = torch.zeros(shape, dtype=torch.bool, device=q.device)
    else:
        # Every row/tile is overwritten by _stats, so no redundant zeroing pass.
        lm = torch.empty(shape, dtype=torch.float32, device=q.device)
        mu = torch.empty((*shape, 32), dtype=torch.float32, device=q.device)
        invalid = torch.empty(shape, dtype=torch.bool, device=q.device)
    _stats[(tr.cdiv(n, 32), h, kt)](q, k, sketch, lm, mu, invalid, n, nk, h, hk, d, kt, qb,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), float(scale), num_warps=4)
    nu = nu_kv[torch.arange(h, device=q.device)//(h//hk)][:, None, None].expand(h, qb, 128).reshape(h*qb, 128).contiguous()
    return Stats(lm, mu, nu, invalid, prefix_tiles, h, qb)


@tr.jit(do_not_specialize=['JT'])
def _online(LM, MU, NU, BAD, PROT, HELD, KEEP, STATE, FINAL_LM, RISK,
            JT, THRESHOLD, PRESERVE: tl.constexpr,
            STICKY: tl.constexpr):
    unit = tl.program_id(0)
    row, rr = tl.arange(0, 128), tl.arange(0, 32)
    previous = tl.full((128,), -float('inf'), tl.float32)
    output = tl.full((128, 32), 0., tl.float32)
    support = tl.full((128,), False, tl.int1)
    nu = tl.maximum(tl.load(NU+unit*128+row), 1.e-12)
    for j in range(JT):
        dest = (unit*JT+j)*128+row
        lm = tl.load(LM+dest)
        active = lm > -float('inf')
        mx = tl.maximum(previous, lm)
        safe = tl.where(mx > -float('inf'), mx, 0.)
        combined = safe+lib.log(lib.exp(previous-safe)+lib.exp(lm-safe))
        eta = tl.where(active, lib.exp(lm-tl.where(combined > -float('inf'), combined, 0.)), 0.)
        mu = tl.load(MU+dest[:, None]*32+rr[None])
        delta = mu-output
        rho = eta*tl.sqrt(tl.sum(delta*delta, 1))/nu
        risk = tl.max(tl.where(active, lib.log(rho), -float('inf')), 0)
        first = tl.sum((active & ~support).to(tl.int32), 0)>0
        invalid = tl.sum(tl.load(BAD+dest).to(tl.int32), 0)>0
        prot = tl.load(PROT+unit*JT+j)
        held = tl.load(HELD+unit*JT+j)
        take = (tl.sum(active.to(tl.int32), 0)>0) & (first | prot | invalid | (risk+STICKY*held>=THRESHOLD))
        tl.store(KEEP+unit*JT+j, take)
        tl.store(RISK+unit*JT+j, risk)
        output = tl.where(take, (1-eta[:, None])*output+eta[:, None]*mu, output)
        previous = tl.where(take | PRESERVE, combined, previous)
        support = support | (active & take)
    tl.store(STATE+unit*128*32+row[:, None]*32+rr[None], output)
    tl.store(FINAL_LM+unit*128+row, previous)


def online_cuda(stats, threshold, preserve, mandatory, held, sticky):
    u, jt, q = stats.log_mass.shape
    keep = torch.empty((u, jt), dtype=torch.bool, device=stats.log_mass.device)
    state = torch.empty((u, 128, 32), device=keep.device)
    lm = torch.empty((u, 128), device=keep.device)
    risk = torch.empty((u, jt), device=keep.device)
    _online[(u,)](stats.log_mass, stats.mean, stats.nu, stats.invalid, mandatory, held,
        keep, state, lm, risk, jt, math.log(threshold) if threshold else -math.inf,
        preserve, sticky, num_warps=8)
    # _stats validates logits and projected means without a full-sized boolean
    # temporary over the rank dimension. No redundant GiB tensor is allocated.
    bad = stats.invalid.any((1, 2))
    keep |= bad[:, None]
    return keep, state, lm, risk


@tr.jit(do_not_specialize=['JT','PT'])
def _scores(ALPHA, G, A, RESIDUAL, NU, ROWS, KEEP, PROT, HELD, SCORES,
            JT, PT, STICKY: tl.constexpr):
    unit, j = tl.program_id(0), tl.program_id(1)
    removable = tl.load(KEEP+unit*JT+j) & ~tl.load(PROT+unit*JT+j) & (j<PT)
    if not removable:
        tl.store(SCORES+unit*JT+j, float('inf'))
        return
    row, rr = tl.arange(0, 128), tl.arange(0, 32)
    ix = (unit*JT+j)*128+row
    alpha = tl.load(ALPHA+ix)
    remain = tl.load(A+unit*128+row)-alpha
    residual = tl.load(RESIDUAL+unit*128*32+row[:, None]*32+rr[None])
    g = tl.load(G+ix[:, None]*32+rr[None])
    live = tl.load(ROWS+unit*128+row)
    nu = tl.maximum(tl.load(NU+unit*128+row), 1.e-12)
    error = tl.sqrt(tl.sum((residual-g)*(residual-g), 1))/(tl.maximum(remain, 1.e-38)*nu)
    score = tl.max(tl.where(live, error, 0.), 0)
    impossible = tl.sum((live & (remain<=0)).to(tl.int32), 0)>0
    held = tl.load(HELD+unit*JT+j)
    score *= lib.exp(STICKY*held)
    tl.store(SCORES+unit*JT+j, tl.where(impossible | ~removable, float('inf'), score))


def candidate_scores(stats, alpha, g, a, residual, keep, mandatory, held, sticky, live_rows):
    out = torch.empty(keep.shape, dtype=torch.float32, device=keep.device)
    _scores[(keep.shape[0], keep.shape[1])](alpha, g, a, residual, stats.nu, live_rows,
        keep, mandatory, held, out, keep.shape[1], stats.prefix_tiles, sticky, num_warps=4)
    return out


@tr.jit(do_not_specialize=['JT','PT'])
def _exact_update(SCORES, ALPHA, G, A, RESIDUAL, KEEP, TARGET, BAD,
                  JT, PT, SIZE: tl.constexpr):
    unit = tl.program_id(0)
    jj = tl.arange(0, SIZE)
    score = tl.load(SCORES+unit*JT+jj, jj<JT, float('inf'))
    existing = tl.load(KEEP+unit*JT+jj, jj<PT, False)
    remaining = tl.sum(existing.to(tl.int32), 0)
    target = tl.load(TARGET+unit)
    active = remaining>target
    lowest = tl.min(score, 0)
    j = tl.min(tl.where((score==lowest) & (jj<JT), jj, 2147483647), 0)
    valid = (lowest<float('inf')) & active
    row, rr = tl.arange(0, 128), tl.arange(0, 32)
    ix = (unit*JT+j)*128+row
    alpha = tl.load(ALPHA+ix, valid, 0.)
    g = tl.load(G+ix[:, None]*32+rr[None], valid, 0.)
    a = tl.load(A+unit*128+row)
    r = tl.load(RESIDUAL+unit*128*32+row[:, None]*32+rr[None])
    tl.store(A+unit*128+row, a-alpha)
    tl.store(RESIDUAL+unit*128*32+row[:, None]*32+rr[None], r-g)
    tl.store(KEEP+unit*JT+j, False, valid)
    tl.store(BAD+unit, active & ~valid)


@tr.jit(do_not_specialize=['JT','PT'])
def _static_prune(ORDER, VALID, SUPPORT, KEEP, PROT, TARGET, BAD,
                  JT, PT, SIZE: tl.constexpr):
    unit = tl.program_id(0)
    row = tl.arange(0, 128)
    jj = tl.arange(0, SIZE)
    retained = tl.load(KEEP+unit*JT+jj, jj<PT, False)
    remaining = tl.sum(retained.to(tl.int32), 0)
    target = tl.load(TARGET+unit)
    support = tl.load(SUPPORT+unit*128+row)
    for pos in range(PT):
        j = tl.load(ORDER+unit*PT+PT-1-pos)
        candidate = tl.load(KEEP+unit*JT+j) & ~tl.load(PROT+unit*JT+j)
        valid = tl.load(VALID+(unit*JT+j)*128+row)
        safe = tl.sum((valid & (support<=1)).to(tl.int32), 0)==0
        drop = candidate & safe & (remaining>target)
        tl.store(KEEP+unit*JT+j, False, drop)
        support -= tl.where(drop & valid, 1, 0)
        remaining -= drop.to(tl.int32)
    tl.store(BAD+unit, remaining!=target)


def fixed_cuda(stats, budget, mandatory, held, sticky, batch=1, singleton=False):
    from .v31_value_summary import full_support_cuda
    alpha, o, g = full_support_cuda(stats)
    del o
    u, jt, _ = alpha.shape
    pt = stats.prefix_tiles
    eligible = stats.valid.any(-1)
    live_rows = stats.rows.contiguous()
    keep = eligible | mandatory
    # GLOBAL bidirectional units have the same candidate count. This guard is
    # one device assertion per call, never one synchronization per candidate.
    k = min(max(0, int(budget)), pt)
    target = eligible[:, :pt].sum(-1).clamp_max(k)
    torch._assert_async(((mandatory[:, :pt] & eligible[:, :pt]).sum(-1)<=target).all(), 'mandatory support exceeds budget')
    a, residual = alpha.sum(1), g.sum(1)
    evaluations = 0
    if singleton:
        scores = candidate_scores(stats, alpha, g, a, residual, keep, torch.zeros_like(mandatory), held, sticky, live_rows)
        rank = scores[:, :pt].masked_fill(~eligible[:, :pt], -math.inf).masked_fill(mandatory[:, :pt], math.inf)
        # Stable sort defines smallest tile ID on score ties for the new arms.
        order = torch.argsort(rank, descending=True, stable=True)
        valid = stats.valid.contiguous()
        support = (valid & keep[..., None]).sum(1).to(torch.int32)
        failed = torch.empty(u, dtype=torch.bool, device=keep.device)
        _static_prune[(u,)](order, valid, support, keep, mandatory, target, failed,
            jt, pt, tr.next_power_of_2(jt), num_warps=4)
        torch._assert_async((~failed).all(), 'no row-supported singleton set at requested budget')
        evaluations = u*pt
    else:
        failed = torch.zeros(u, dtype=torch.bool, device=keep.device)
        for iteration in range(math.ceil(max(0, pt-k)/batch)):
            count = min(batch, pt-k-iteration*batch)
            scores = candidate_scores(stats, alpha, g, a, residual, keep, mandatory, held, sticky, live_rows)
            if batch == 1:
                _exact_update[(u,)](scores, alpha, g, a, residual, keep, target, failed,
                    jt, pt, tr.next_power_of_2(jt), num_warps=4)
                evaluations += u*(pt-iteration)
                continue
            order = torch.argsort(scores, stable=True)[:, :count]
            selected = scores.gather(1, order)
            active = torch.arange(count, device=keep.device)[None] < (keep[:, :pt].sum(-1)-target)[:, None]
            da = (alpha.gather(1, order[..., None].expand(-1, -1, 128))*active[..., None]).sum(1)
            safe = ((~live_rows | (a-da>0)).all(-1) & (~active | torch.isfinite(selected)).all(-1))
            # On joint support loss, take only the first admissible deletion.
            # Remaining deletions are handled by an exact cleanup phase below.
            active &= safe[:, None] | (torch.arange(count,device=keep.device)[None]==0)
            torch._assert_async((~active | torch.isfinite(selected)).all(), 'no admissible deletion at requested budget')
            drop = torch.zeros_like(keep).scatter_(1, order, active)
            da = (alpha.gather(1, order[..., None].expand(-1, -1, 128))*active[..., None]).sum(1)
            dg = (g.gather(1, order[..., None, None].expand(-1, -1, 128, 32))*active[..., None, None]).sum(1)
            keep &= ~drop
            a -= da
            residual -= dg
            evaluations += u*(pt-iteration*batch)
        if batch > 1:
            # One phase-boundary host read determines cleanup length. There are
            # no host decisions or synchronizations for individual candidates.
            extra = int((keep[:, :pt].sum(-1)-target).max().item())
            for iteration in range(extra):
                scores = candidate_scores(stats,alpha,g,a,residual,keep,mandatory,held,sticky,live_rows)
                _exact_update[(u,)](scores,alpha,g,a,residual,keep,target,failed,
                    jt,pt,tr.next_power_of_2(jt),num_warps=4)
                evaluations += u*(k+extra-iteration)
        torch._assert_async((~failed).all(), 'no admissible exact deletion at requested budget')
    torch._assert_async((~stats.rows | ((alpha*keep[..., None]).sum(1)>0)).all(), 'fixed-k mask removes row support')
    bad = stats.invalid.any((1, 2))
    keep |= bad[:, None]
    return keep, evaluations


def select_cuda(stats, selector, budget, threshold, mandatory, held, sticky):
    mandatory = mandatory_map(stats) if mandatory is None else mandatory.contiguous()
    held = torch.zeros_like(mandatory) if held is None else held.contiguous()
    if selector in SELECTORS[:2]:
        keep, _, _, _ = online_cuda(stats, threshold, selector == SELECTORS[1], mandatory, held, sticky)
        evaluations = stats.log_mass.shape[0]*stats.log_mass.shape[1]
        kernel = 'triton_log_mass_online'
    else:
        keep, evaluations = fixed_cuda(stats, budget, mandatory, held, sticky,
            8 if selector == SELECTORS[4] else 1, selector == SELECTORS[2])
        kernel = 'triton_parallel_cached_deletion'
    return keep.reshape(1, stats.heads, stats.blocks, -1), dict(candidate_evaluations=evaluations,
        kernel=kernel, approximation=selector == SELECTORS[4], summary_bytes=stats.nbytes,
        sticky_rule='multiplicative_exp_bonus' if sticky else 'none')
