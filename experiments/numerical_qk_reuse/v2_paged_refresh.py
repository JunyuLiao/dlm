"""Exact selected V2 arithmetic with direct native pages and immutable-prefix reuse."""
import ctypes as ct
import math
import torch
import triton as tr
import triton.language as tl
from .v2_online_sm90 import Kernel, Params
from .v2_online_pipeline import pack_lists

@tr.jit
def _project(V,R,TABLE,Z,NORM,P,N,D:tl.constexpr,H,SH,SN,PAGE:tl.constexpr,JT:tl.constexpr,START:tl.constexpr):
    block,head=tl.program_id(0),tl.program_id(1)
    row=START+block*32+tl.arange(0,32);col=tl.arange(0,32);kk=tl.arange(0,64)
    page=tl.load(TABLE+row//PAGE,row<N,0)
    physical=page*PAGE+row%PAGE
    acc=tl.full((32,32),0.,tl.float32);norm=tl.full((32,),0.,tl.float32)
    for offset in range(tr.cdiv(D,64)):
        dim=offset*64+kk
        x=tl.load(V+head*SH+physical[:,None]*SN+dim[None,:],(row[:,None]<N)&(dim[None,:]<D),0.).to(tl.float32)
        r=tl.load(R+head*D*32+dim[:,None]*32+col[None,:],dim[:,None]<D,0.)
        acc+=tl.dot(x,r,input_precision='tf32x3');norm+=tl.sum(x*x,1)
    norm=tl.sqrt(norm);norm=norm*norm
    tl.store(Z+(head*N+row[:,None])*32+col[None,:],acc,row[:,None]<N)
    tl.store(NORM+head*N+row,norm,row<N)
    hi=acc.to(tl.bfloat16);lo=(acc-hi.to(tl.float32)).to(tl.bfloat16)
    index=(row[:,None]%64)*32+col[None,:];physical_index=index^(((index>>6)&3)<<3)
    base=(head*JT+row[:,None]//64)*4096
    tl.store(P+base+physical_index,hi,row[:,None]<N);tl.store(P+base+2048+physical_index,lo,row[:,None]<N)
    if block==tl.cdiv(N-START,32)-1:
        extra=N+tl.arange(0,64);tail_index=(extra[:,None]%64)*32+col[None,:];tail_physical=tail_index^(((tail_index>>6)&3)<<3)
        tail_base=(head*JT+extra[:,None]//64)*4096;valid=extra<JT*64
        tl.store(P+tail_base+tail_physical,0.,valid[:,None]);tl.store(P+tail_base+2048+tail_physical,0.,valid[:,None])

class PagedLayout(ct.Structure):
    _fields_=[('table',ct.c_void_p),('pages',ct.c_int),('page_size',ct.c_int),('token_stride',ct.c_uint64),('head_stride',ct.c_uint64)]

class PagedRefresh:
    def __init__(self,library,schedule=2):
        self.kernel=Kernel(library);self.schedule=schedule;self.states={}
        self.fn=self.kernel.lib.v2_online_paged_sm90
        self.fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int];self.fn.restype=ct.c_int
        self.reset_counters()
    def reset_counters(self):
        self.full_projection_calls=0;self.canvas_projection_calls=0;self.projected_rows=0;self.reused_rows=0
    def clear(self):
        self.states.clear();self.reset_counters()
    def __call__(self,q,key,value,table,nk,matrix,scale,prefix,threshold,*,held=None,sticky=0.,cache_key=None,canvas=None):
        from .v31_value_projection import _parts,_finish
        b,h,n,d=q.shape;pages,page,hk,kd=key.shape;qb=tr.cdiv(n,128);jt=tr.cdiv(nk,64)
        if b!=1 or d!=512 or kd!=d or h%hk or key.shape!=value.shape or not q.is_contiguous() or q.dtype!=torch.bfloat16:
            raise ValueError('native B1 D512 BF16 Q and paged KV required')
        if page%64 or key.stride(0)!=page*key.stride(1) or key.stride()!=value.stride() or key.stride(-1)!=1 or table.dtype!=torch.int32 or not table.is_contiguous():
            raise ValueError('direct paged stride/table contract')
        if not 0<=prefix<nk or nk-prefix!=n or not math.isfinite(threshold) or threshold<0:
            raise ValueError('current canvas and finite threshold required')
        if held is not None and (held.shape!=(1,h,qb,jt) or held.dtype!=torch.bool or not held.is_contiguous()):
            raise ValueError('current held map required')
        tag=(canvas,nk,prefix,key.data_ptr(),value.data_ptr(),table.data_ptr(),matrix.data_ptr())
        state=self.states.get(cache_key) if cache_key is not None else None
        if state is None or state['tag']!=tag:
            state=dict(tag=tag,z=torch.empty((1,hk,nk,32),device=q.device),norm=torch.empty((1,hk,nk),device=q.device),
                valid=torch.ones((1,hk,nk),device=q.device,dtype=torch.bool),packed=torch.empty((hk,jt,4096),device=q.device,dtype=torch.bfloat16))
            if cache_key is not None:self.states[cache_key]=state
            start=0;self.full_projection_calls+=1
        else:start=prefix;self.canvas_projection_calls+=1
        self.projected_rows+=hk*(nk-start);self.reused_rows+=hk*start
        _project[(tr.cdiv(nk-start,32),hk)](value,matrix,table,state['z'],state['norm'],state['packed'],nk,d,hk,value.stride(2),value.stride(1),page,jt,start,num_warps=4,enable_fp_fusion=False)
        chunks=tr.cdiv(nk,4096);parts=torch.empty((1,hk,chunks,2),device=q.device);nu=torch.empty((1,hk),device=q.device)
        _parts[(chunks,hk,1)](state['norm'],state['valid'],parts,nk,hk,chunks,*state['valid'].stride(),num_warps=4,enable_fp_fusion=False)
        _finish[(hk,)](parts,nu,chunks,num_warps=4,enable_fp_fusion=False)
        out=torch.empty_like(q);skipped=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool);eligible=torch.empty_like(skipped)
        scratch=torch.empty((1,),device=q.device)
        p=Params(*(x.data_ptr() if x is not None else 0 for x in (q,key,value,state['z'],nu,None,out,skipped,eligible,scratch,scratch,scratch)),
            1,h,hk,n,nk,d,0,1,1,0,scale,math.log(threshold) if threshold else -math.inf,0,2,state['packed'].data_ptr(),0,0,0 if held is None else held.data_ptr(),sticky,prefix//64)
        layout=PagedLayout(table.data_ptr(),pages,page,key.stride(1),key.stride(2))
        status=self.fn(ct.byref(p),ct.byref(layout),torch.cuda.current_stream(q.device).cuda_stream,self.schedule)
        if status:raise RuntimeError(f'paged V2 CUDA launch error {status}')
        keep=~skipped
        return out.transpose(1,2),keep,pack_lists(keep)
