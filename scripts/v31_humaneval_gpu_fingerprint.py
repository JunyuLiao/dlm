"""Diagnostic 128-bit integer fingerprints; no floating arithmetic or RNG."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mix(x):
    x=(x^(x>>30))*0xbf58476d1ce4e5b9
    x=(x^(x>>27))*0x94d049bb133111eb
    return x^(x>>31)


@triton.jit
def _words(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(0);i=group*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+i,i<N,0).to(tl.uint32).to(tl.uint64)
    position=i.to(tl.uint64)*0x9e3779b97f4a7c15
    a=tl.where(i<N,_mix(value^position^0x243f6a8885a308d3),0)
    b=tl.where(i<N,_mix(value^position^0x13198a2e03707344),0)
    groups=tl.cdiv(N,BLOCK)
    tl.store(Y+group,tl.sum(a,0));tl.store(Y+groups+group,tl.sum(b,0))


@triton.jit
def _fold(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(0);i=group*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(X+i,i<N,0);b=tl.load(X+N+i,i<N,0)
    groups=tl.cdiv(N,BLOCK)
    tl.store(Y+group,tl.sum(a,0));tl.store(Y+groups+group,tl.sum(b,0))


def fingerprint_cuda(value):
    """Fingerprint every raw bit in logical contiguous order on the current stream.

    This is a noncryptographic two-seed SplitMix64 checksum, folded with exact
    integer addition modulo2^64. It is only a first-divergence locator; follow a
    detected change with an actual tensor comparison. It never substitutes for
    output/scoring hashes or establishes mathematical equality.
    """
    if not value.is_cuda:raise ValueError('CUDA tensor required')
    raw=value.detach().contiguous().view(torch.uint8).reshape(-1)
    if not raw.numel() or raw.numel()%4:raise ValueError('nonempty multiple of four bytes required')
    words=raw.view(torch.int32);n=words.numel();block=1024
    groups=triton.cdiv(n,block)
    result=torch.empty((2,groups),device=value.device,dtype=torch.uint64)
    _words[(groups,)](words,result,N=n,BLOCK=block,num_warps=4)
    while groups>1:
        next_groups=triton.cdiv(groups,block)
        folded=torch.empty((2,next_groups),device=value.device,dtype=torch.uint64)
        _fold[(next_groups,)](result,folded,N=groups,BLOCK=block,num_warps=4)
        result=folded;groups=next_groups
    return result.reshape(2)
