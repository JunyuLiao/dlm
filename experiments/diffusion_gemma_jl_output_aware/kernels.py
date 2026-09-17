"""FP32 block softmax/projected-PV and fused retained-state physical routing.

Native QK/logit rounding and final BF16 softmax/PV remain in the parent model
path. These kernels eliminate per-KV-tile Python work and never require full
block PV for a sketch-based decision. No hardware speedup claim is implied.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from experiments.diffusion_gemma_value_aware_gpu.kernels import _logaddexp
from .config import Config

GUARD = 8e-5
METHOD = {'centered': 0, 'contribution': 1, 'mass_exact': 2, 'cancellation_guard': 3}


@tr.jit
def _statistics(S, VALID, SKETCH, NORMS, LOGZ, MU, COUNT, MAXIMUM, WNORM,
                Q: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
                R: tl.constexpr, RP: tl.constexpr, GQA: tl.constexpr):
    h, group, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q = group*16+tl.arange(0, 16)
    kk = tile*64+tl.arange(0, 64)
    rr = tl.arange(0, RP)
    where = (q[:, None] < Q) & (kk[None, :] < K)
    address = (h*Q+q[:, None])*K+kk[None, :]
    valid = tl.load(VALID+address, where, other=0).to(tl.int1)
    logits = tl.load(S+address, where, other=-float('inf')).to(tl.float32)
    logits = tl.where(valid, logits, -float('inf'))
    count = tl.sum(valid.to(tl.int32), 1)
    maximum = tl.max(logits, 1)
    weights = lib.exp(logits-tl.where(count > 0, maximum, 0.)[:, None])
    weights = tl.where(valid, weights, 0.)
    ell = tl.sum(weights, 1)
    weights = weights/tl.maximum(ell, 1e-30)[:, None]
    sk = tl.load(SKETCH+((h//GQA)*K+kk[:, None])*R+rr[None, :],
                 (kk[:, None] < K) & (rr[None, :] < R), other=0.).to(tl.float32)
    # IEEE FP32 products, not BF16/TF32 sketches or full-PV-then-projection.
    mean = tl.dot(weights, sk, input_precision='ieee')
    norm = tl.load(NORMS+(h//GQA)*K+kk, kk < K, other=0.).to(tl.float32)
    wn = tl.sum(weights*norm[None, :], 1)
    logz = tl.where(count > 0, maximum+lib.log(tl.maximum(ell, 1e-30)), -float('inf'))
    off = (h*Q+q)*T+tile
    tl.store(LOGZ+off, logz, q < Q)
    tl.store(COUNT+off, count, q < Q)
    tl.store(MAXIMUM+off, maximum, q < Q)
    tl.store(WNORM+off, wn, q < Q)
    tl.store(MU+off[:, None]*R+rr[None, :], mean, (q[:, None] < Q) & (rr[None, :] < R))


@tr.jit
def _route(Z, MU, N, WN, REF, THRESH, MASK, MARGIN, RISKS,
           Q: tl.constexpr, T: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
           QB: tl.constexpr, GQA: tl.constexpr, METHOD_ID: tl.constexpr,
           PRIMARY_R: tl.constexpr, KAPPA: tl.constexpr, MASS: tl.constexpr,
           STORE_RISKS: tl.constexpr):
    h, qb, policy = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q = qb*128+tl.arange(0, 128)
    rr = tl.arange(0, RP)
    previous = tl.full((128,), -float('inf'), tl.float32)
    output = tl.full((128, RP), 0., tl.float32)
    threshold = tl.load(THRESH+policy)
    ref = tl.maximum(tl.load(REF+h//GQA), 1e-12)
    margin = tl.full((), float('inf'), tl.float32)
    for tile in range(T):
        off = (h*Q+q)*T+tile
        z = tl.load(Z+off, q < Q, other=-float('inf'))
        n = tl.load(N+off, q < Q, other=0)
        mu = tl.load(MU+off[:, None]*R+rr[None, :], (q[:, None] < Q) & (rr[None, :] < R), other=0.)
        active = n > 0
        combined = _logaddexp(previous, z)
        alpha = lib.exp(z-tl.where(combined > -float('inf'), combined, 0.))
        alpha = tl.where(active, alpha, 0.)
        delta = alpha[:, None]*(mu-output)
        if METHOD_ID == 2:
            risk = alpha
        elif METHOD_ID == 1:
            risk = alpha*tl.sqrt(tl.sum(mu*mu, 1))/ref
        elif METHOD_ID == 3:
            first = tl.sqrt(tl.sum(tl.where(rr[None, :] < PRIMARY_R, delta*delta, 0.), 1))/ref
            second = tl.sqrt(tl.sum(tl.where(rr[None, :] >= PRIMARY_R, delta*delta, 0.), 1))/ref
            mean_norm = tl.load(WN+off, q < Q, other=0.)
            kappa = tl.sqrt(tl.sum(tl.where(rr[None, :] < PRIMARY_R, mu*mu, 0.), 1))/(mean_norm+1e-12)
            risk = tl.where((kappa < KAPPA) & (alpha >= MASS), tl.maximum(first, second), first)
        else:
            risk = tl.sqrt(tl.sum(delta*delta, 1))/ref
        log_risk = lib.log(risk)
        log_risk = tl.where(active, tl.where(previous > -float('inf'), log_risk, float('inf')), -float('inf'))
        worst = tl.max(log_risk, 0)
        eligible = tl.sum(active.to(tl.int32), 0) > 0
        skip = eligible & (worst < threshold)
        dest = ((policy*tl.num_programs(0)+h)*QB+qb)*T+tile
        tl.store(MASK+dest, skip)
        if STORE_RISKS:
            tl.store(RISKS+dest, worst)
        margin = tl.minimum(margin, tl.abs(worst-threshold))
        addz = tl.where(active & ~skip, z, -float('inf'))
        newz = _logaddexp(previous, addz)
        safe = tl.where(newz > -float('inf'), newz, 0.)
        a, b = lib.exp(previous-safe), lib.exp(addz-safe)
        output = a[:, None]*output+b[:, None]*mu
        previous = newz
    tl.store(MARGIN+(policy*tl.num_programs(0)+h)*QB+qb, margin)


def block_statistics(scores, valid, sketches, projected_norm=None):
    if scores.ndim != 4 or scores.shape[0] != 1 or not scores.is_cuda:
        raise ValueError('CUDA batch-one B,H,Q,K scores required')
    b, h, q, k = scores.shape
    native, rank = sketches.shape[1], sketches.shape[-1]
    if h % native or sketches.shape[:1]+sketches.shape[2:3] != (b, k):
        raise ValueError('Incompatible native-head sketches/GQA')
    tiles = tr.cdiv(k, 64)
    shape = (b, h, q, tiles)
    z = torch.empty(shape, dtype=torch.float32, device=scores.device)
    maximum, wn = torch.empty_like(z), torch.empty_like(z)
    count = torch.empty(shape, dtype=torch.int32, device=scores.device)
    mu = torch.empty((*shape, rank), dtype=torch.float32, device=scores.device)
    norms = sketches.norm(dim=-1) if projected_norm is None else projected_norm
    _statistics[(h, tr.cdiv(q, 16), tiles)](scores.contiguous(), valid.expand_as(scores).contiguous(),
        sketches.contiguous(), norms.contiguous(), z, mu, count, maximum, wn,
        q, k, tiles, rank, max(16, tr.next_power_of_2(rank)), h//native,
        num_warps=4, enable_fp_fusion=False)
    active = count > 0
    return dict(logz=z, mu=mu, count=count, b=maximum, mean_norm=wn,
                active=active, eligible=active.any(-2))


def route(state, ref, config=None, thresholds=None, store_risks=False):
    c = config or Config()
    _, heads, q, tiles = state['logz'].shape
    rank = state['mu'].shape[-1]
    gqa = heads//ref.shape[-1]
    thresholds = torch.as_tensor([c.log_threshold] if thresholds is None else thresholds,
                                 dtype=torch.float32, device=ref.device).flatten()
    qb = tr.cdiv(q, 128)
    mask = torch.empty((len(thresholds), heads, qb, tiles), dtype=torch.bool, device=ref.device)
    margin = torch.empty(mask.shape[:-1], dtype=torch.float32, device=ref.device)
    risks = torch.empty(mask.shape if store_risks else (1,), dtype=torch.float32, device=ref.device)
    _route[(heads, qb, len(thresholds))](state['logz'], state['mu'], state['count'],
        state['mean_norm'], ref.contiguous(), thresholds, mask, margin, risks,
        q, tiles, rank, tr.next_power_of_2(rank), qb, gqa, METHOD[c.method],
        c.rank, c.cancellation_cutoff, c.substantial_mass, store_risks,
        num_warps=8 if rank >= 128 else 4, enable_fp_fusion=False)
    return mask, margin, risks
