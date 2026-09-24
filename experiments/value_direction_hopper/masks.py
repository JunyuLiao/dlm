"""Lossless packing of arbitrary boolean structural masks, not mask inference."""
from dataclasses import dataclass
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _pack(MASK, OUT, NQ:tl.constexpr, NK:tl.constexpr, T:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    bit=tl.arange(0,64)
    valid=tl.load(MASK+row*NK+tile*64+bit,tile*64+bit<NK,other=0).to(tl.uint64)
    word=tl.sum(valid<<bit.to(tl.uint64),0)
    tl.store(OUT+row*T+tile,word)


@dataclass(frozen=True)
class PackedMask:
    tensor: torch.Tensor
    keys: int


def pack(mask):
    if mask.ndim!=4 or mask.dtype!=torch.bool or not mask.is_cuda or not mask.is_contiguous():
        raise ValueError('Contiguous CUDA B,Hmask,Q,K boolean mask required; no additive bias discarded')
    b,h,q,k=mask.shape
    if min(b,h,q,k)<1: raise ValueError('Nonempty mask required')
    tiles=tr.cdiv(k,64)
    result=torch.empty((b,h,q,tiles),device=mask.device,dtype=torch.uint64)
    _pack[(b*h*q,tiles)](mask,result,q,k,tiles,num_warps=4)
    return PackedMask(result,k)


@tr.jit
def _geometry(OUT,Q:tl.constexpr,K,T,CAUSAL:tl.constexpr,WINDOW:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    bit=tl.arange(0,64);key=tile*64+bit
    valid=key<K
    if CAUSAL:valid=valid&(key<=K-Q+row%Q)
    if WINDOW:valid=valid&(key>=K-Q+row%Q-WINDOW+1)
    tl.store(OUT+row*T+tile,tl.sum(valid.to(tl.uint64)<<bit.to(tl.uint64),0))


def geometry(batch,queries,keys,*,device,causal=False,window=0):
    """Exact historical bottom-right causal/window mask without a QK matrix."""
    if min(batch,queries,keys)<1 or window<0:raise ValueError('Invalid geometry')
    tiles=tr.cdiv(keys,64)
    result=torch.empty((batch,1,queries,tiles),device=device,dtype=torch.uint64)
    _geometry[(batch*queries,tiles)](result,queries,keys,tiles,bool(causal),int(window),num_warps=4)
    keyvalid=torch.arange(keys,device=device)>=max(0,keys-queries-int(window)+1 if window else 0)
    return PackedMask(result,keys),keyvalid
