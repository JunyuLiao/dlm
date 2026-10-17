"""Qualified device-only execution for the two unchanged layer29 value methods."""
import functools
import os
import types
import torch
from .v31_value_decision import Moments
from .v31_value_fast import bounded_proposal_device, dense_output, selected_ids, selected_output
from .v31_value_refinement_runtime import Runtime
from .v31_value_reuse_guard import ResidualDirectionCache


@torch.inference_mode()
def moments_full(q,k,v,prefix,scale,block=64,chunk=128):
    """Reference FP32 moments, without unused tangent or V-RMS calculations."""
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,k,v=q.to(dtype),k.to(dtype),v.to(dtype)
    scores=q@k.T*scale;logz=scores.logsumexp(-1);p=(scores-logz[:,None]).exp()
    pt=prefix//block
    mass=p[:,:pt*block].reshape(len(q),pt,block).sum(-1).T.contiguous()
    num=torch.empty((pt,len(q),v.shape[1]),dtype=dtype,device=q.device)
    for lo in range(0,pt,chunk):
        hi=min(pt,lo+chunk)
        weights=p[:,lo*block:hi*block].reshape(len(q),hi-lo,block).permute(1,0,2)
        num[lo:hi]=weights@v[lo*block:hi*block].reshape(hi-lo,block,-1)
    tail=p[:,pt*block:]
    return Moments(mass,num,tail.sum(-1),tail@v[pt*block:],logz)


class FastRuntime(Runtime):
    backend='device_exact_v1'
    def __init__(self,mode):
        if mode not in ('full_swap8','residual_direction_guard'):raise ValueError(mode)
        super().__init__(mode)
        self.audit=os.environ.get('VALUE_AUDIT')=='1'
        self.clean=os.environ.get('VALUE_CLEAN_TIMING')=='1'
        if self.audit==self.clean:raise ValueError('choose separate clean or audit pass')
        self.events=[]

    def reset(self):
        super().reset();self.events=[]

    def initialize(self,adapter,q,b,scale,prefix,n,out,identity):
        from . import v27_fa4
        start=torch.cuda.Event(enable_timing=True) if self.audit else None
        if start:start.record()
        st=adapter.mage_state[29];kept=st['kept'].clone()
        h,qb,pt=q.shape[1],(n+127)//128,prefix//64
        budget=min(adapter.mage_k//64,pt)
        if self.audit:
            self.add('anchor_budget_units_checked',h*qb)
            self.add('budget_errors',(kept[0,:,:,:pt].sum(-1)!=budget).sum())
        if budget==pt:
            self.state=dict(identity=identity,allkept=True)
            self.add('allkept_shortcuts',1)
            return
        if self.selects and not self.grams:
            w=self.model.model.layers[29].self_attn.o_proj.weight.float()
            for head in range(h):
                part=w[:,head*512:(head+1)*512];self.grams[head]=part.T@part
        if self.corrects:
            dense=torch.empty((n,h,512),device=q.device,dtype=torch.float32)
            sparse=torch.empty_like(dense)
        kh_previous=None;swaps=[]
        for head in range(h):
            kh=head//(h//b['k'].shape[1])
            if kh!=kh_previous:
                # Reused for every query tile and all eight associated query heads.
                k=b['k'][0,kh].float();v=b['v'][0,kh].float();kh_previous=kh
            for qt in range(qb):
                lo,hi=qt*128,min(n,(qt+1)*128)
                m=moments_full(q[0,head,lo:hi],k,v,prefix,scale)
                mask=kept[0,head,qt,:pt].bool()
                if self.selects:
                    new,info=bounded_proposal_device(m,mask,budget,gram=self.grams[head],backend='exact')
                    kept[0,head,qt,:pt]=new
                    if self.audit:swaps.append(info['swaps'])
                else:
                    dense[lo:hi,head]=dense_output(m)
                    num,den=selected_output(m,selected_ids(mask,budget))
                    sparse[lo:hi,head]=num/den.clamp_min(torch.finfo(num.dtype).tiny)[:,None]
                del m
        self.state=dict(identity=identity)
        if self.selects:
            if self.audit:
                self.add('updated_budget_units_checked',h*qb)
                self.add('budget_errors',(kept[0,:,:,:pt].sum(-1)!=budget).sum())
                self.add('swapped_blocks',torch.stack(swaps).sum())
            st['kept']=kept;st['lists']=v27_fa4.block_sparse_tensors(kept)
        else:
            if not hasattr(self,'projection'):
                self.projection=self.model.model.layers[29].self_attn.o_proj.weight.float()
            self.state['direction_cache']=ResidualDirectionCache.build(dense,sparse,self.projection,identity)
            self.add('extra_post_o_projections',2)
        self.add('anchor_events',1)
        self.add('anchor_full_qk_rectangles',h*qb*((prefix+n+63)//64))
        self.add('anchor_full_pv_rectangles',h*qb*((prefix+n+63)//64))
        if start:
            end=torch.cuda.Event(enable_timing=True);end.record();self.events.append((start,end))

    def receipt(self):
        work={k:int(v) if isinstance(v,torch.Tensor) else v for k,v in self.counts.items()}
        result=dict(value_refinement=self.mode,value_refinement_layers=[29],
            value_refinement_output_consumer='native FA4 plus cached correction' if self.corrects else 'unchanged native FA4',
            value_refinement_work=work,value_refinement_clean_timing=self.clean,
            value_refinement_backend=self.backend,
            value_refinement_budget_audited=self.audit)
        if self.audit:
            torch.cuda.synchronize()
            result['value_anchor_module_cuda_elapsed_ms']=sum(a.elapsed_time(b) for a,b in self.events)
        return result


def install(mode):
    from . import vllm_adapter
    from vllm.model_executor.models import diffusion_gemma as dg
    runtime=FastRuntime(mode)
    old_init=dg.DiffusionGemmaForConditionalGeneration.__init__
    @functools.wraps(old_init)
    def init(self,*args,**kwargs):
        old_init(self,*args,**kwargs);runtime.model=self
    dg.DiffusionGemmaForConditionalGeneration.__init__=init
    old_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        if (adapter.arm!='mage' or adapter.mage_granularity!='qblock_max' or adapter.mage_select!='fa4'
            or adapter.stable_refresh_config is not None or adapter.value_selector is not None
            or adapter.cached_refresh_policy is not None):raise ValueError('requires the frozen V31 control')
        old_mage,old_begin,old_receipt=adapter._mage,adapter.begin_request,adapter._value_receipt
        def mage(self,layer,q,b,scale,prefix,n):
            prior=self.mage_state.get(layer);previous=prior.get('lists') if prior is not None else None
            out=old_mage(layer,q,b,scale,prefix,n)
            return runtime.output(self,layer,q,b,scale,prefix,n,out,previous)
        def begin(self,*args,**kwargs):runtime.reset();return old_begin(*args,**kwargs)
        def receipt(self):
            return dict(old_receipt(),value_refinement_prefix_budget=self.mage_k//64,
                value_refinement_sticky_feedback='deployed changed support feeds inherited next-refresh sticky term',**runtime.receipt())
        adapter._mage=types.MethodType(mage,adapter)
        adapter.begin_request=types.MethodType(begin,adapter)
        adapter._value_receipt=types.MethodType(receipt,adapter)
        return old_install(adapter)
    vllm_adapter.install_vllm_patches=install_adapter
    return runtime
