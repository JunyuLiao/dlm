"""Current-value Gaussian32 projection fused with SM90 sketch packing.

Same FP32 Gaussian matrix, TF32x3 dot and norm-then-square convention as the
qualified inherited projection. Stores both row-major audit sketch and packed
BF16 high/residual operands in the projection launch; no separate pack pass.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _project(V,R,Z,NORM,P,N,D:tl.constexpr,H,SB,SH,SN,JT:tl.constexpr,START:tl.constexpr):
    block,head=tl.program_id(0),tl.program_id(1)
    row=START+block*32+tl.arange(0,32);col=tl.arange(0,32);kk=tl.arange(0,64)
    acc=tl.full((32,32),0.,tl.float32);norm=tl.full((32,),0.,tl.float32)
    for offset in range(tr.cdiv(D,64)):
        dim=offset*64+kk
        x=tl.load(V+head*SH+row[:,None]*SN+dim[None,:],(row[:,None]<N)&(dim[None,:]<D),0.).to(tl.float32)
        r=tl.load(R+head*D*32+dim[:,None]*32+col[None,:],dim[:,None]<D,0.)
        acc+=tl.dot(x,r,input_precision='tf32x3');norm+=tl.sum(x*x,1)
    norm=tl.sqrt(norm);norm=norm*norm
    tl.store(Z+(head*N+row[:,None])*32+col[None,:],acc,row[:,None]<N)
    tl.store(NORM+head*N+row,norm,row<N)
    hi=acc.to(tl.bfloat16);lo=(acc-hi.to(tl.float32)).to(tl.bfloat16)
    index=(row[:,None]%64)*32+col[None,:];physical=index^(((index>>6)&3)<<3)
    base=(head*JT+row[:,None]//64)*4096
    tl.store(P+base+physical,hi,row[:,None]<N);tl.store(P+base+2048+physical,lo,row[:,None]<N)
    # Final incomplete tile's invalid rows must be initialized for unconditional
    # packed cp.async loads. These values do not enter valid QK probabilities.
    if block==tl.cdiv(N-START,32)-1:
        extra=N+tl.arange(0,64);tail_index=(extra[:,None]%64)*32+col[None,:];tail_physical=tail_index^(((tail_index>>6)&3)<<3)
        tail_base=(head*JT+extra[:,None]//64)*4096;valid=(extra<N+64)&(extra<JT*64)
        tl.store(P+tail_base+tail_physical,0.,valid[:,None]);tl.store(P+tail_base+2048+tail_physical,0.,valid[:,None])

def project(v,matrix,z,norm,valid,*,start=0,packed=None):
    from .v31_value_projection import _parts,_finish
    b,h,n,d=v.shape
    if b!=1 or d!=512 or not v.is_contiguous():raise ValueError('native B1D512 contiguous values')
    if not 0<=start<n:raise ValueError('valid changed canvas range')
    jt=tr.cdiv(n,64)
    if packed is None:
        if start:raise ValueError('partial update requires existing packed state')
        packed=torch.empty((h,jt,4096),device=v.device,dtype=torch.bfloat16)
    _project[(tr.cdiv(n-start,32),h)](v,matrix,z,norm,packed,n,d,h,*v.stride()[:3],jt,start,num_warps=4,enable_fp_fusion=False)
    chunks=tr.cdiv(n,4096);parts=torch.empty((b,h,chunks,2),device=v.device);nu=torch.empty((b,h),device=v.device)
    _parts[(chunks,h,b)](norm,valid,parts,n,h,chunks,*valid.stride(),num_warps=4,enable_fp_fusion=False)
    _finish[(b*h,)](parts,nu,chunks,num_warps=4,enable_fp_fusion=False)
    return nu,packed
