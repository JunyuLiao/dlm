"""Named V31 refresh implementation with fused selection and FA4 observation.

The inherited Yuhan mass qblock-max selector is retained. A private FA4 subclass
observes already exponentiated scores, without recomputing softmax. Installed
FA4 files and published controls are never modified.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from dataclasses import dataclass
from quack import layout_utils
from vllm.vllm_flash_attn.cute.mask import AttentionMask
from vllm.vllm_flash_attn.cute.utils import AuxData
from vllm.vllm_flash_attn.cute import utils


@tr.jit
def _repair(Z,Q,K,N:tl.constexpr,PT,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
            QS0:tl.constexpr,QS1:tl.constexpr,KS0,KS1:tl.constexpr,SCALE:tl.constexpr,J:tl.constexpr):
    unit=tl.program_id(0);qb=tr.cdiv(N,128);h=unit//qb;kh=h//(H//HK)
    row=tl.arange(0,128);jj=tl.arange(0,J)
    qi=(unit%qb)*128+row
    observed=tl.load(Z+unit*PT*128+jj[None,:]*128+row[:,None],jj[None,:]<PT,0.)
    bad=(observed!=observed)|(observed==-float('inf'))
    if tl.sum(bad.to(tl.int32),0).sum(0)>0:
        dd=tl.arange(0,D);kk=tl.arange(0,64)
        q=tl.load(Q+h*QS0+qi[:,None]*QS1+dd[None,:],qi[:,None]<N,0.)
        for j in range(PT):
            z=tl.load(Z+unit*PT*128+j*128+row)
            if tl.sum(((z!=z)|(z==-float('inf'))).to(tl.int32),0)>0:
                keys=j*64+kk
                k=tl.load(K+kh*KS0+keys[None,:]*KS1+dd[:,None])
                score=tl.dot(q,k)*SCALE
                mx=tl.max(score,1)
                den=tl.sum(tl.exp(score-mx[:,None]),1)
                lm=mx+tl.log(den)
                tl.store(Z+unit*PT*128+j*128+row,lm,qi<N)


@tr.jit
def _select(Z,HELD,KEEP,CNT,IDX,N:tl.constexpr,JT,PT,K:tl.constexpr,
            STICKY:tl.constexpr,SINK:tl.constexpr,RECENT:tl.constexpr,J:tl.constexpr):
    unit=tl.program_id(0)
    row,j=tl.arange(0,128),tl.arange(0,J)
    lm=tl.load(Z+unit*PT*128+j[None,:]*128+row[:,None],j[None,:]<PT,-float('inf'))
    valid_row=(unit % tr.cdiv(N,128))*128+row<N
    mx=tl.max(lm,1)
    safe=tl.where(mx > -float('inf'),mx,0.)
    den=tl.sum(tl.exp(lm-safe[:,None]),1)
    lse=safe+tl.log(tl.maximum(den,1.e-30))
    score=tl.max(tl.where(valid_row[:,None],lm-lse[:,None],-float('inf')),0)
    held=tl.load(HELD+unit*JT+j,j<JT,False)
    score+=STICKY*held
    protected=(j<tl.minimum(SINK,PT)) | ((j>=tl.maximum(0,PT-RECENT))&(j<PT))
    score=tl.where(protected,float('inf'),score)
    score=tl.where(j<PT,score,-float('inf'))
    if PT>K:
        sorted_score=tl.sort(score,descending=True)
        cutoff=tl.sum(tl.gather(sorted_score,tl.full((1,),K-1,tl.int32),0),0)
        greater=score>cutoff
        ties=(score==cutoff)&(j<PT)
        tie_rank=tl.cumsum(ties.to(tl.int32),0)
        selected=greater | (ties & (tie_rank<=K-tl.sum(greater.to(tl.int32),0)))
    else:
        selected=j<PT
    selected=(selected & (j<PT)) | ((j>=PT)&(j<JT))
    pos=tl.cumsum(selected.to(tl.int32),0)-1
    tl.store(KEEP+unit*JT+j,selected,j<JT)
    tl.store(IDX+unit*JT+pos,j,selected&(j<JT))
    tl.store(CNT+unit,tl.sum(selected.to(tl.int32),0))


def select(z,n,nk,budget,*,held=None,sticky=0.,sink=0,recent=0):
    from . import v27_fa4
    v27_fa4.load()
    h,qb,pt,_=z.shape
    jt=tr.cdiv(nk,64)
    keep=torch.empty((1,h,qb,jt),device=z.device,dtype=torch.bool)
    held=torch.zeros_like(keep) if held is None else held.contiguous()
    if held.shape!=keep.shape:raise ValueError('held map shape drift')
    count=torch.empty((1,h,qb),device=z.device,dtype=torch.int32)
    indices=torch.zeros((1,h,qb,jt),device=z.device,dtype=torch.int32)
    zeros=torch.zeros_like(count)
    empty=torch.zeros((1,h,qb,1),device=z.device,dtype=torch.int32)
    _select[(h*qb,)](z,held,keep,count,indices,n,jt,pt,max(1,budget//64),float(sticky),
        int(sink),int(recent),tr.next_power_of_2(jt),num_warps=4)
    lists=v27_fa4._BST(mask_block_cnt=zeros,mask_block_idx=empty,full_block_cnt=count,
                     full_block_idx=indices,block_size=(128,64))
    return keep,lists


def build_observer(directory):
    """Bind a private source copy to a unique FA4 subclass and cache key."""
    import vllm.vllm_flash_attn.cute.flash_fwd_sm90 as sm90
    source=Path(sm90.__file__).read_text()
    source=source.replace('class FlashAttentionForwardSm90(', 'class FlashAttentionForwardSm90Refresh(')
    lines=source.splitlines(True)
    patched=[]
    additions=0
    for line in lines:
        patched.append(line)
        if 'row_scale = softmax.online_softmax(acc_S' in line:
            indent=line[:len(line)-len(line.lstrip())]
            patched.extend([indent+'if const_expr(mask_fn is not None):\n',
                indent+'    mask_fn(acc_S=acc_S, n_block=n_block, post_softmax=True, softmax=softmax)\n'])
            additions+=1
    if additions!=5:raise RuntimeError(f'FA4 source hook drift: {additions}')
    source=''.join(patched)
    # Finish outstanding PV before extra observation register work in overlap.
    source=source.replace('''        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block, post_softmax=True, softmax=softmax)
        warpgroup.wait_group(0)
''','''        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block, post_softmax=True, softmax=softmax)
''')
    digest=hashlib.sha256(source.encode()).hexdigest()
    path=Path(directory)/digest[:16]/'flash_fwd_sm90_refresh.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=source:raise RuntimeError('frozen source collision')
    path.write_text(source)
    name='vllm.vllm_flash_attn.cute.refresh_'+digest[:16]
    import sys
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module,digest


def observer_mask():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32
    from dataclasses import dataclass
    from quack import layout_utils
    from vllm.vllm_flash_attn.cute.mask import AttentionMask
    from vllm.vllm_flash_attn.cute.utils import AuxData
    from vllm.vllm_flash_attn.cute import utils
    @dataclass(frozen=True)
    class SoftmaxObservation(AttentionMask):
        @cute.jit
        def apply_mask(self,acc_S,batch_idx,head_idx,m_block,n_block,thr_mma,
                       mask_seqlen=False,mask_causal=False,mask_local=False,mask_mod=None,
                       aux_data:AuxData=AuxData(),fastdiv_mods=(None,None),
                       post_softmax:cutlass.Constexpr[bool]=False,softmax=None):
            if cutlass.const_expr(not post_softmax):
                AttentionMask.apply_mask(self,acc_S,batch_idx,head_idx,m_block,n_block,thr_mma,
                    mask_seqlen,mask_causal,mask_local,mask_mod,aux_data,fastdiv_mods)
            else:
                z=aux_data.tensors[0]
                if n_block<z.shape[2]:
                    p=layout_utils.reshape_acc_to_mn(acc_S)
                    coords=cute.make_identity_tensor((self.tile_m,self.tile_n))
                    coords=layout_utils.reshape_acc_to_mn(thr_mma.partition_C(coords))
                    tid=cute.arch.thread_idx()[0]
                    writer=(tid>=128) and (tid<256) and (tid%4==0)
                    for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
                        total=utils.fadd_reduce(p[r,None].load(),init_val=None)
                        total=cute.arch.warp_reduction_sum(total,threads_in_group=4)
                        grow=m_block*self.tile_m+coords[r,0][0]
                        if writer and grow<self.seqlen_q:
                            mass=-Float32.inf
                            if total>Float32(0.):
                                mass=(softmax.row_max[r]*softmax.scale_log2+cute.math.log2(total,fastmath=True))*0.6931471805599453
                            if total<Float32(1.e-20):
                                mass=-Float32.inf
                            z[head_idx,grow//128,n_block,grow%128]=mass
    return SoftmaxObservation


class Observer:
    def __init__(self,directory):
        self.module,self.source_sha256=build_observer(directory)
        self.module.AttentionMask=observer_mask()
        self.cache={}
        self.dynamic={}
        self.identity={}

    def __call__(self,q,k,v,scale,z,*,overlap=False,num_splits=1,dynamic_causal=False):
        from vllm.vllm_flash_attn.cute import interface
        original=interface.FlashAttentionForwardSm90
        interface.FlashAttentionForwardSm90=self.module.FlashAttentionForwardSm90Refresh
        key=(float(scale),str(q.device))
        if key not in self.cache:self.cache[key]=torch.tensor([scale],device=q.device)
        if str(q.device) not in self.dynamic:self.dynamic[str(q.device)]=torch.tensor([False],device=q.device)
        # Interface compile keys omit the Python subclass; distinguish this
        # private observer from inherited aux2 observation by tensor metadata.
        if str(q.device) not in self.identity:self.identity[str(q.device)]=torch.empty((3,),device=q.device,dtype=torch.int32)
        mask=dict(causal=True,dynamic_causal=self.dynamic[str(q.device)]) if dynamic_causal else dict(causal=False)
        try:
            return interface._flash_attn_fwd(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                **mask,softmax_scale=scale,pack_gqa=False,num_splits=num_splits,
                intra_wg_overlap=overlap,aux_tensors=[z,self.cache[key],self.identity[str(q.device)]])[0]
        finally:interface.FlashAttentionForwardSm90=original

    def refresh(self,q,k,v,scale,prefix,budget,*,overlap=False,held=None,sticky=0.,num_splits=1,dynamic_causal=False):
        z=torch.empty((q.shape[1],tr.cdiv(q.shape[2],128),prefix//64,128),device=q.device)
        out=self(q,k,v,scale,z,overlap=overlap,num_splits=num_splits,dynamic_causal=dynamic_causal)
        # Probabilities scaled by a previous tile's huge max can underflow.
        # Repair only those rare query units, keeping exact prefix statistics.
        if prefix//64:
            _repair[(q.shape[1]*tr.cdiv(q.shape[2],128),)](z,q,k,q.shape[2],prefix//64,
                q.shape[1],k.shape[1],q.shape[-1],q.stride(1),q.stride(2),k.stride(1),k.stride(2),
                float(scale),tr.next_power_of_2(prefix//64),num_warps=4,num_stages=1)
        keep,lists=select(z,q.shape[2],k.shape[2],budget,held=held,sticky=sticky)
        return out,keep,lists
