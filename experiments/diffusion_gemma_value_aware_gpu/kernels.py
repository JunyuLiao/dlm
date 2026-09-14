"""Custom CUDA/Triton reductions and physical-tile streaming decisions.

Native QK, BF16-rounded logits, FP32 softmax -> BF16 probabilities and whole-Q
PV stay in PyTorch/cuBLAS. These kernels optimize experiment orchestration,
not the sparse algorithms. No FlashAttention speedup claim is made.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

METHOD = {'blasst': 0, 'value': 1, 'mass': 2, 'risk': 3, 'aligned': 4}
Q_TILE, KV_TILE = 128, 64
GUARD = 2e-5


@tr.jit
def _logaddexp(a, b):
    mx = tl.maximum(a, b)
    return tl.where(mx == -float('inf'), mx,
                    mx + lib.log1p(lib.exp(-tl.abs(a-b))))


@tr.jit
def _stats(S, V, LN, B, Z, N, A,
           Q: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
           QB: tl.constexpr, ALIGNED: tl.constexpr, R: tl.constexpr):
    h = tl.program_id(0)
    qr = tl.program_id(1)*R + tl.arange(0, R)
    tile = tl.program_id(2)
    kk = tile*64 + tl.arange(0, 64)
    inside = (qr[:, None] < Q) & (kk[None, :] < K)
    off = (h*Q + qr[:, None])*K + kk[None, :]
    valid = tl.load(V+off, inside, other=0).to(tl.int1)
    scores = tl.load(S+off, inside, other=-float('inf')).to(tl.float32)
    scores = tl.where(valid, scores, -float('inf'))
    count = tl.sum(valid.to(tl.int32), 1)
    b = tl.max(scores, 1)
    e = lib.exp(scores - tl.where(count > 0, b, 0.)[:, None])
    e = tl.where(valid, e, 0.)
    z = tl.where(count > 0, b+lib.log(tl.maximum(tl.sum(e, 1), 1e-30)), -float('inf'))
    a = b
    if ALIGNED:
        ln = tl.load(LN+h*K+kk, kk < K, other=-float('inf'))
        a = tl.max(scores+ln[None, :], 1)
    # Padding query rows to128 simplifies the physical all-row reduction.
    out = ((h*QB + qr//128)*T+tile)*128 + qr % 128
    tl.store(B+out, b, qr < QB*128)
    tl.store(Z+out, z, qr < QB*128)
    tl.store(N+out, count, qr < QB*128)
    tl.store(A+out, a, qr < QB*128)


@tr.jit
def _route(B, Z, N, A, LR, MASK, MARGIN, THRESH,
           T: tl.constexpr, QB: tl.constexpr, METHOD_ID: tl.constexpr):
    h = tl.program_id(0)
    qb = tl.program_id(1)
    q = tl.arange(0, 128)
    m = tl.full((128,), -float('inf'), tl.float32)
    z = tl.full((128,), -float('inf'), tl.float32)
    seen = tl.full((128,), -float('inf'), tl.float32)
    min_margin = tl.full((), float('inf'), tl.float32)
    threshold = tl.load(THRESH)
    for j in range(T):
        off = ((h*QB+qb)*T+j)*128+q
        b = tl.load(B+off)
        addz = tl.load(Z+off)
        n = tl.load(N+off)
        previous = seen if METHOD_ID == 0 else m
        gap = b-previous
        risk = gap
        if METHOD_ID == 1:
            risk = gap+tl.load(LR+h*T+j)
        elif METHOD_ID == 2 or METHOD_ID == 3:
            boundz = b+lib.log(tl.maximum(n, 1).to(tl.float32))
            risk = boundz-_logaddexp(z, boundz)
            if METHOD_ID == 3:
                ratio = tl.load(LR+h*T+j)
                # PyTorch softplus threshold20 semantics, no overflow rewrite.
                risk += tl.where(ratio > 20., ratio, lib.log1p(lib.exp(ratio)))
        elif METHOD_ID == 4:
            risk = tl.load(A+off)-previous
        risk = tl.where(n > 0, tl.where(z > -float('inf'), risk, float('inf')), -float('inf'))
        worst = tl.max(risk, 0)
        eligible = tl.sum((n > 0).to(tl.int32), 0) > 0
        skip = eligible & (worst < threshold)
        tl.store(MASK+(h*QB+qb)*T+j, skip)
        min_margin = tl.minimum(min_margin, tl.abs(worst-threshold))
        seen = tl.maximum(seen, b)
        include = (n > 0) & ~skip
        z = _logaddexp(z, tl.where(include, addz, -float('inf')))
        m = tl.maximum(m, tl.where(include, b, -float('inf')))
    tl.store(MARGIN+h*QB+qb, min_margin)


def block_statistics(scores, valid, lognorm=None):
    if not scores.is_cuda or scores.ndim != 4:
        raise ValueError('CUDA B,H,Q,K scores required')
    batch, heads, q, k = scores.shape
    qb, tiles = tr.cdiv(q, 128), tr.cdiv(k, 64)
    scores = scores.contiguous()
    valid = torch.broadcast_to(valid, scores.shape).contiguous()
    shape = (batch*heads, qb, tiles, 128)
    b = torch.empty(shape, dtype=torch.float32, device=scores.device)
    z, a = torch.empty_like(b), torch.empty_like(b)
    n = torch.empty(shape, dtype=torch.int32, device=scores.device)
    ln = scores if lognorm is None else lognorm.contiguous()
    _stats[(batch*heads, qb*8, tiles)](scores, valid, ln, b, z, n, a,
        q, k, tiles, qb, lognorm is not None, 16, num_warps=4, enable_fp_fusion=False)
    return b, z, n, a


def route(statistics, config, meta, threshold):
    b, z, n, aligned = statistics
    h, qb, tiles, _ = b.shape
    if config.method not in METHOD or config.aggregation != 'all_rows':
        raise ValueError('Kernel supports only the unchanged requested all-row methods')
    if config.method in ('value', 'risk'):
        ratio = (meta[config.pooling]/meta['ref'][..., None]).clamp_min(1e-30).log().reshape(h, tiles).contiguous()
    else:
        ratio = torch.empty((1,), device=b.device)
    if not torch.is_tensor(threshold):
        threshold = torch.tensor(threshold, dtype=torch.float32, device=b.device)
    mask = torch.empty((h, qb, tiles), dtype=torch.bool, device=b.device)
    margin = torch.empty((h, qb), dtype=torch.float32, device=b.device)
    _route[(h, qb)](b, z, n, aligned, ratio, mask, margin, threshold,
        tiles, qb, METHOD[config.method], num_warps=4, enable_fp_fusion=False)
    return mask, margin


def query_state(statistics, batch, heads, qb, qrows):
    b, z, n, _ = [a[:, qb, :, :qrows].transpose(-1, -2).reshape(batch, heads, qrows, -1) for a in statistics]
    active = n > 0
    total = torch.logsumexp(z, -1, keepdim=True)
    mass = torch.exp(z-torch.where(torch.isfinite(total), total, 0.)).masked_fill(~active, 0.)
    return dict(b=b, logz=z, count=n, active=active, eligible=active.any(-2), mass=mass)
