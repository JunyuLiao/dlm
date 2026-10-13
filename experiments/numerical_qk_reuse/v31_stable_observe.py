"""Bounded duplicate QK observation; no PV and no union across query heads.

Each program computes one Q128 x KV64 rectangle for one query head. GQA
selects that head's native KV head without sharing its routing decision.
"""
import math
import torch


def tile_ids(selected):
    k = int(selected.sum(-1)[0])
    return (~selected).to(torch.int8).argsort(dim=-1, stable=True)[:, :k].contiguous()


def reference_observe(q, k, scale, ids, n):
    h, hk, d = q.shape[1], k.shape[1], q.shape[-1]
    qb = (n+127)//128
    out = torch.full((*ids.shape, 128), -math.inf, device=q.device)
    for u in range(h*qb):
        head, block = divmod(u, qb)
        lo, hi = block*128, min(n, (block+1)*128)
        for t in range(ids.shape[1]):
            j = int(ids[u, t])
            a, b = j*64, min(k.shape[2], (j+1)*64)
            s = q[0, head, lo:hi].float() @ k[0, head//(h//hk), a:b].float().T * scale
            out[u, t, :hi-lo] = s.logsumexp(-1)
    return out


def observe(q, k, scale, ids, n, out=None):
    if not q.is_cuda:
        return reference_observe(q, k, scale, ids, n)
    if out is None:
        out = torch.empty((*ids.shape, 128), dtype=torch.float32, device=q.device)
    if ids.numel():
        _kernel()[(ids.shape[0], ids.shape[1])](q, k, ids, out,
            q.stride(1), q.stride(2), q.stride(3), k.stride(1), k.stride(2), k.stride(3),
            N=n, NK=k.shape[2], QB=(n+127)//128, GROUP=q.shape[1]//k.shape[1],
            D=q.shape[-1], T=ids.shape[1], SCALE=float(scale), num_warps=4)
    return out


_KERNEL = None


def _kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def run(Q, K, IDS, OUT, QH:tl.constexpr, QR:tl.constexpr, QD:tl.constexpr,
            KH:tl.constexpr, KR:tl.constexpr, KD:tl.constexpr, N:tl.constexpr, NK:tl.constexpr,
            QB:tl.constexpr, GROUP:tl.constexpr, D:tl.constexpr, T:tl.constexpr, SCALE:tl.constexpr):
        u, t = tl.program_id(0), tl.program_id(1)
        head, block = u // QB, u % QB
        tile = tl.load(IDS + u*T + t)
        rows = block*128 + tl.arange(0,128)
        keys = tile*64 + tl.arange(0,64)
        dd = tl.arange(0,64)
        acc = tl.zeros((128,64),tl.float32)
        for off in range(tl.cdiv(D,64)):
            di = off*64+dd
            a = tl.load(Q+head*QH+rows[:,None]*QR+di[None,:]*QD,
                        (rows[:,None]<N)&(di[None,:]<D),0)
            b = tl.load(K+(head//GROUP)*KH+keys[None,:]*KR+di[:,None]*KD,
                        (keys[None,:]<NK)&(di[:,None]<D),0)
            acc += tl.dot(a,b, input_precision='ieee')
        s = tl.where(keys[None,:]<NK, acc*SCALE, -float('inf'))
        mx = tl.max(s,1)
        z = mx + tl.log(tl.sum(tl.exp(s-mx[:,None]),1))
        tl.store(OUT+(u*T+t)*128+tl.arange(0,128),tl.where(rows<N,z,-float('inf')))

    _KERNEL = run
    return run
