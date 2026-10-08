"""Prior Gaussian32 projection with a bounded current-value RMS reduction.

The inherited projection kernel and norm-then-square convention are reused.
Only the reference-scale reduction is split into 4096-token chunks; its
definition remains sqrt(mean(valid ||v||^2)), with no reference aging.
"""
import torch
import triton as tr
import triton.language as tl

from experiments.value_direction_hopper.projection import _refresh


@tr.jit(do_not_specialize=['N','CHUNKS','VB','VH'])
def _parts(NORM,VALID,PART,N,H:tl.constexpr,CHUNKS,VB,VH,VN:tl.constexpr):
    chunk,head,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    row=chunk*4096+tl.arange(0,4096)
    valid=tl.load(VALID+batch*VB+head*VH+row*VN,row<N,0)
    norm=tl.load(NORM+(batch*H+head)*N+row,row<N,0.)
    base=((batch*H+head)*CHUNKS+chunk)*2
    tl.store(PART+base,tl.sum(tl.where(valid,norm,0.),0))
    tl.store(PART+base+1,tl.sum(valid.to(tl.float32),0))


@tr.jit(do_not_specialize=['CHUNKS'])
def _finish(PART,OUT,CHUNKS):
    unit=tl.program_id(0)
    chunk=tl.arange(0,128)
    base=(unit*CHUNKS+chunk)*2
    numerator=tl.sum(tl.load(PART+base,chunk<CHUNKS,0.),0)
    count=tl.sum(tl.load(PART+base+1,chunk<CHUNKS,0.),0)
    tl.store(OUT+unit,tl.maximum(tl.sqrt(numerator/tl.maximum(count,1.)),1e-12))


def refresh(value,matrix,z,norm,valid,start=0):
    b,h,n,d=value.shape
    if (value.dtype!=torch.bfloat16 or d not in (256,512) or value.stride(-1)!=1
        or matrix.shape!=(h,d,32) or matrix.dtype!=torch.float32 or not matrix.is_contiguous()
        or z.shape!=(b,h,n,32) or norm.shape!=(b,h,n) or valid.shape!=(b,h,n)
        or valid.dtype!=torch.bool or z.dtype!=torch.float32 or norm.dtype!=torch.float32
        or not z.is_contiguous() or not norm.is_contiguous() or not 0<=start<=n
        or any(x.device!=value.device for x in (matrix,z,norm,valid))):
        raise ValueError('invalid native Gaussian32 refresh geometry')
    chunks=tr.cdiv(n,4096)
    if chunks>128:
        raise ValueError('RMS exceeds bounded reduction capacity')
    if start<n:
        _refresh[(tr.cdiv(n-start,32),h,b)](value,matrix,z,norm,n,d,h,start,*value.stride()[:3],
                                         num_warps=4,enable_fp_fusion=False)
    parts=torch.empty((b,h,chunks,2),device=value.device,dtype=torch.float32)
    reference=torch.empty((b,h),device=value.device,dtype=torch.float32)
    _parts[(chunks,h,b)](norm,valid,parts,n,h,chunks,*valid.stride(),num_warps=4,enable_fp_fusion=False)
    _finish[(b*h,)](parts,reference,chunks,num_warps=4,enable_fp_fusion=False)
    return reference
