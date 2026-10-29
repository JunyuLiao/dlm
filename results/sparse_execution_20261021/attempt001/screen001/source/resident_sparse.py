"""Resident GPU sparse metadata and direct-paged current-KV consumers.

Named execution variant of Yuhan's inherited cross-step reuse. Mask discovery
is separate; held attention performs no selection, list build or KV copy.
SparseD motivates compiled mask reuse; FA4 arithmetic remains the pinned native
implementation. FlexAttention is an independent qualified prototype.
"""
from dataclasses import dataclass
import math
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _pack(KEEP,CNT,IDX,KT:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    unit,split=tl.program_id(0),tl.program_id(1)
    j=tl.arange(0,B);take=tl.load(KEEP+unit*KT+j,j<KT,False)
    pos=tl.cumsum(take.to(tl.int32),0)-1;count=tl.sum(take.to(tl.int32),0)
    lo=count*split//S;hi=count*(split+1)//S
    selected=take&(pos>=lo)&(pos<hi)
    tl.store(IDX+(split*tl.num_programs(0)+unit)*KT+pos-lo,j,selected)
    tl.store(CNT+split*tl.num_programs(0)+unit,hi-lo)

@tr.jit
def _page_table(TABLE,EXPANDED,USED,NK:tl.constexpr,S:tl.constexpr,FACTOR:tl.constexpr,B:tl.constexpr):
    x=tl.arange(0,B);pages=tr.cdiv(NK,64)
    native=tl.load(TABLE+x//FACTOR,x<pages,0)
    tl.store(EXPANDED+x,native*FACTOR+x%FACTOR,x<pages)
    tl.store(USED+tl.arange(0,S),NK)

@tr.jit
def _merge(O,LSE,OUT,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    split=tl.arange(0,S);dim=tl.arange(0,B)
    l=tl.load(LSE+(split*H+head)*N+row);m=tl.max(l,0);safe=tl.where(m>-float('inf'),m,0.)
    w=tl.exp(l-safe);w=w/tl.maximum(tl.sum(w,0),1.e-30)
    val=tl.load(O+((split[:,None]*N+row)*H+head)*D+dim[None,:],dim[None,:]<D,0.).to(tl.float32)
    tl.store(OUT+(row*H+head)*D+dim,tl.sum(w[:,None]*val,0),dim<D)

@dataclass
class ResidentMask:
    keep:torch.Tensor
    lists:object
    splits:int
    kept_prefix:torch.Tensor|None
    eligible_prefix:int

    @classmethod
    def create(cls,keep,*,splits=1,prefix_tiles=0,audit=False):
        from . import v27_fa4
        v27_fa4.load()
        if keep.ndim!=4 or keep.shape[0]!=1 or keep.dtype!=torch.bool or not keep.is_contiguous():raise ValueError('B1 contiguous physical Q128/KV64 map required')
        if splits not in (1,2,4,8):raise ValueError('static split bucket required')
        _,h,qb,kt=keep.shape
        count=torch.empty((splits,h,qb),device=keep.device,dtype=torch.int32)
        idx=torch.zeros((splits,h,qb,kt),device=keep.device,dtype=torch.int32)
        zeros=torch.zeros_like(count);empty=torch.zeros((splits,h,qb,1),device=keep.device,dtype=torch.int32)
        _pack[(h*qb,splits)](keep,count,idx,kt,splits,tr.next_power_of_2(kt),num_warps=4)
        lists=v27_fa4._BST(mask_block_cnt=zeros,mask_block_idx=empty,full_block_cnt=count,full_block_idx=idx,block_size=(128,64))
        return cls(keep,lists,splits,keep[...,:prefix_tiles].sum().double() if audit else None,h*qb*prefix_tiles)

class PagedContext:
    """One canvas's resident descriptors; logical table may update in-place."""
    def __init__(self,key,value,table,nk,*,splits=1):
        if key.shape!=value.shape or key.shape[1]%64 or table.dtype!=torch.int32:raise ValueError('native paged geometry')
        page=key.shape[1];factor=page//64
        if any(x.stride(0)!=page*x.stride(1) for x in (key,value)):raise ValueError('native KV page stride')
        shape=(key.shape[0]*factor,64,key.shape[2],key.shape[3])
        self.k=key.as_strided(shape,(64*key.stride(1),*key.stride()[1:]))
        self.v=value.as_strided(shape,(64*value.stride(1),*value.stride()[1:]))
        self.nk,self.splits=nk,splits;self.source_table=table;self._factor=factor
        self.page_table=torch.empty((splits,tr.cdiv(nk,64)),device=key.device,dtype=torch.int32)
        self.used=torch.full((splits,),nk,device=key.device,dtype=torch.int32)
        self.update_table()
        self.output=None
    def update_table(self):
        # Table expansion is performed once on refresh/canvas setup, never held.
        _page_table[(1,)](self.source_table,self.page_table,self.used,self.nk,self.splits,self._factor,
            tr.next_power_of_2(tr.cdiv(self.nk,64)),num_warps=4)
        if self.splits>1:self.page_table[1:].copy_(self.page_table[:1].expand(self.splits-1,-1))
    def __call__(self,q,mask,scale,*,num_splits=1,overlap=False):
        from . import v27_fa4
        if mask.splits!=self.splits:raise ValueError('resident split mismatch')
        fwd=v27_fa4.load();qs=q.transpose(1,2).expand(self.splits,-1,-1,-1)
        out,lse=fwd(qs,self.k,self.v,softmax_scale=scale,causal=False,page_table=self.page_table,
            seqused_k=self.used,block_sparse_tensors=mask.lists,num_splits=num_splits,pack_gqa=False,
            intra_wg_overlap=overlap,return_lse=self.splits>1)[:2]
        if self.splits==1:return out
        if self.output is None or self.output.shape!=(1,q.shape[2],q.shape[1],q.shape[3]):self.output=torch.empty((1,q.shape[2],q.shape[1],q.shape[3]),device=q.device,dtype=q.dtype)
        _merge[(q.shape[2],q.shape[1])](out,lse,self.output,q.shape[2],q.shape[1],q.shape[3],self.splits,tr.next_power_of_2(q.shape[3]),num_warps=4)
        return self.output

class FlexConsumer:
    """Compiled FlexAttention with resident physical full-block metadata."""
    def __init__(self,keep,n,nk,*,kernel_options=None):
        from torch.nn.attention.flex_attention import BlockMask,flex_attention
        resident=ResidentMask.create(keep)
        count,index=resident.lists.full_block_cnt,resident.lists.full_block_idx
        def mask_mod(b,h,q,k):return (q<n)&(k<nk)
        self.block_mask=BlockMask.from_kv_blocks(torch.zeros_like(count),torch.zeros_like(index),
            full_kv_num_blocks=count,full_kv_indices=index,BLOCK_SIZE=(128,64),mask_mod=mask_mod)
        self.options=kernel_options or dict(BLOCK_M=64,BLOCK_N=64)
        self.fn=torch.compile(flex_attention,fullgraph=True,dynamic=False)
    def __call__(self,q,k,v,scale):
        return self.fn(q,k,v,block_mask=self.block_mask,scale=scale,enable_gqa=True,kernel_options=self.options).transpose(1,2)
