"""Fused native-head FP32 Gaussian32 projection and value-norm refresh.

Original BF16 V and FP32 R are preserved. TF32x3 implements the FP32 dot to
near-FP32 accuracy, with errors qualified against the original torch projection.
Only changed canvas/boundary tokens are written; no full-dimensional PV occurs.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _refresh(V,R,Z,NORM,N,D:tl.constexpr,H,START,SB,SH,SN):
    block,head,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    row=START+block*32+tl.arange(0,32);col=tl.arange(0,32);kk=tl.arange(0,64)
    acc=tl.full((32,32),0.,tl.float32);norm=tl.full((32,),0.,tl.float32)
    for offset in range(tr.cdiv(D,64)):
        dim=offset*64+kk
        x=tl.load(V+batch*SB+head*SH+row[:,None]*SN+dim[None,:],(row[:,None]<N)&(dim[None,:]<D),other=0).to(tl.float32)
        r=tl.load(R+head*D*32+dim[:,None]*32+col[None,:],dim[:,None]<D,other=0)
        acc+=tl.dot(x,r,input_precision='tf32x3')
        norm+=tl.sum(x*x,1)
    # Match the old norm-then-square convention, including its final rounding.
    norm=tl.sqrt(norm);norm=norm*norm
    tl.store(Z+((batch*H+head)*N+row[:,None])*32+col[None,:],acc,row[:,None]<N)
    tl.store(NORM+(batch*H+head)*N+row,norm,row<N)


@tr.jit
def _reference(NORM,VALID,OUT,N,H,VB,VH,VN,BLOCK:tl.constexpr):
    head,batch=tl.program_id(0),tl.program_id(1);row=tl.arange(0,BLOCK)
    valid=tl.load(VALID+batch*VB+head*VH+row*VN,row<N,other=0)
    norm=tl.load(NORM+(batch*H+head)*N+row,row<N,other=0)
    reference=tl.sqrt(tl.sum(tl.where(valid,norm,0.),0)/tl.maximum(tl.sum(valid.to(tl.int32),0),1))
    tl.store(OUT+batch*H+head,tl.maximum(reference,1.e-12))


def refresh(value,matrix,z,norm,valid,start):
    b,h,n,d=value.shape
    if d not in (256,512) or value.dtype!=torch.bfloat16 or value.stride(-1)!=1:
        raise ValueError('BF16 native-head D256/512 values with contiguous channels required')
    if matrix.shape!=(h,d,32) or matrix.dtype!=torch.float32 or not matrix.is_contiguous():
        raise ValueError('Contiguous native-head FP32 Gaussian32 matrices required')
    if z.shape!=(b,h,n,32) or norm.shape!=(b,h,n) or valid.shape!=(b,h,n) or valid.dtype!=torch.bool:
        raise ValueError('Invalid cache/validity shape')
    if z.dtype!=torch.float32 or norm.dtype!=torch.float32 or not z.is_contiguous() or not norm.is_contiguous():
        raise ValueError('Contiguous FP32 cache required')
    if not 0<=start<=n or any(x.device!=value.device for x in (matrix,z,norm,valid)):raise ValueError('Invalid refresh lease')
    if start<n:
        _refresh[(tr.cdiv(n-start,32),h,b)](value,matrix,z,norm,n,d,h,start,*value.stride()[:3],num_warps=4,enable_fp_fusion=False)
    ref=torch.empty((b,h),device=value.device,dtype=torch.float32)
    _reference[(h,b)](norm,valid,ref,n,h,*valid.stride(),tr.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return ref
