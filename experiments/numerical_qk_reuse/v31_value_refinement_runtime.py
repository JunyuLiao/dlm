"""Opt-in final-GLOBAL-layer native variants; disabled until diagnostic promotion.

Uses the independent selector and an explicit per-request correction cache.
Native bootstrap, progress clock, carry, and sampler are inherited unchanged.
The reconstruction variants change attention output and say so in receipts.
"""
from __future__ import annotations

import functools
import inspect
import types

import torch

from .v31_value_decision import bounded_proposal, c_gate, moments
from .v31_value_reuse_guard import SelectedOnlyCorrection, ResidualDirectionCache, reconstruct_cuda, residual_trust

MODES=('full_swap8','online_swap8','full_cgate_swap8','full_swap8_direction_guard',
       'residual','moments','residual_direction_guard','residual_norm_guard')


class Runtime:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('unknown explicit value refinement')
        self.mode=mode; self.model=None; self.state=None; self.grams={}; self.counts={}
        self.previous_stats=None; self.last_sampler=None; self.stats_canvas=None

    @property
    def selects(self):
        return self.mode.endswith('swap8') or self.mode=='full_swap8_direction_guard'

    @property
    def corrects(self):
        return not self.mode.endswith('swap8')

    @property
    def direction_guard(self):
        return self.mode in ('residual_direction_guard','full_swap8_direction_guard')

    def add(self,name,amount):
        self.counts[name]=self.counts.get(name,0)+amount

    def reset(self):
        self.state=None; self.counts={}
        self.previous_stats=None; self.last_sampler=None; self.stats_canvas=None

    def sample_statistics(self,scaled,canvas,entropy_bound):
        from .v31_same_state_suffix import distribution
        stats=distribution(scaled,entropy_bound)
        old=self.last_sampler if self.stats_canvas==canvas else None
        changed=stats['top1']!=old['top1'] if old is not None else torch.zeros_like(stats['accepted'])
        reversed_=stats['accepted']!=old['accepted'] if old is not None else torch.zeros_like(stats['accepted'])
        self.previous_stats=(stats['accepted'].clone(),changed,reversed_)
        self.last_sampler=dict(top1=stats['top1'].clone(),accepted=stats['accepted'].clone())
        self.stats_canvas=canvas
        self.add('extra_sampler_stat_calls',1)

    @torch.inference_mode()
    def output(self,adapter,layer,q,b,scale,prefix,n,out,previous_list):
        if layer!=29:
            return out
        st=adapter.mage_state.get(layer)
        identity=(adapter.canvas_id,prefix,n)
        if st is None or st['canvas']!=adapter.canvas_id or st['nk']!=prefix+n:
            self.state=None
            return out
        if st['kept'] is None:
            raise RuntimeError('refinement requires an explicit fixed-budget held map')
        anchor=previous_list is not st['lists']
        if anchor:
            self.initialize(adapter,q,b,scale,prefix,n,out,identity)
            return out                         # inherited dense observation output
        if self.state is None or self.state['identity']!=identity:
            return out                         # carry/bootstrap uses native output
        if self.state.get('allkept', False):
            return out
        self.add('held_calls',1)
        if not self.corrects:
            return out                         # FA4 already used the replaced map
        if self.direction_guard:
            cache=self.state['direction_cache']
            value,_=(cache.apply_cuda(out[0],identity) if out.is_cuda else cache.apply(out[0],identity))
            self.add('corrected_calls',1)
            self.add('held_direction_dot_rows',n)
            self.add('direction_guard_fused_calls',int(out.is_cuda))
            return value[None]
        if self.mode=='residual' or self.mode.startswith('residual_'):
            self.add('corrected_calls',1)
            correction=self.state['residual'][None]
            if self.mode.startswith('residual_'):
                current=out.float().reshape(n,-1)@self.projection.T
                kind='direction' if self.mode=='residual_direction_guard' else 'norm'
                alpha=residual_trust(current,self.state['anchor_sparse_post'],self.state['anchor_dense_post'],kind=kind)
                correction=correction*alpha[None,:,None,None]
                self.add('extra_post_o_projections',1)
            return (out.float()+correction).to(out.dtype)
        from .v31_stable_observe import observe, tile_ids
        h,qb=q.shape[1],(n+127)//128
        keep=st['kept'][0].reshape(h*qb,-1)
        ids=tile_ids(keep)
        # Until LSE is exposed from the native FA4 consumer, this is an explicit
        # retained-only QK observation. It does not scan omitted keys, and its
        # rectangles are reported separately from physical consumer work.
        z=observe(q,b['k'],scale,ids,n)
        selected_lse=z.logsumexp(1).reshape(h,qb*128)[:,:n].T.contiguous()
        self.add('held_extra_qk_rectangles',ids.numel())
        self.add('corrected_calls',1)
        return reconstruct_cuda(out[0],selected_lse,self.state['correction'],identity)[None]

    def initialize(self,adapter,q,b,scale,prefix,n,out,identity):
        from . import v27_fa4
        layer=29; st=adapter.mage_state[layer]; kept=st['kept'].clone()
        h,qb,pt=q.shape[1],(n+127)//128,prefix//64
        expected=min(adapter.mage_k//64,pt)
        counts=kept[0,:,:,:pt].sum(-1)
        self.add('anchor_budget_units_checked',counts.numel())
        if not bool((counts==expected).all()):
            raise RuntimeError('fixed per-query-tile budget is not the configured K')
        if expected==pt:
            self.state=dict(identity=identity,allkept=True)
            self.add('allkept_shortcuts',1)
            return
        if not self.grams and self.selects:
            w=self.model.model.layers[29].self_attn.o_proj.weight.float()
            for head in range(h):
                part=w[:,head*512:(head+1)*512]
                self.grams[head]=part.T@part
        dense=torch.empty((n,h,512),device=q.device,dtype=torch.float32)
        sparse=torch.empty_like(dense)
        dl=torch.empty((n,h),device=q.device,dtype=torch.float32)
        sl=torch.empty_like(dl)
        swaps=0
        for head in range(h):
            kh=head//(h//b['k'].shape[1])
            for qt in range(qb):
                lo,hi=qt*128,min(n,(qt+1)*128)
                m=moments(q[0,head,lo:hi],b['k'][0,kh],b['v'][0,kh],prefix,scale)
                mask=kept[0,head,qt,:pt].bool()
                if self.selects:
                    weight=None
                    if self.mode=='full_cgate_swap8' and self.previous_stats is not None and self.stats_canvas==adapter.canvas_id:
                        weight=c_gate(*self.previous_stats)[lo:hi]
                    new,info=bounded_proposal(m,mask,swaps=8,online=self.mode=='online_swap8',gram=self.grams[head],weight=weight)
                    kept[0,head,qt,:pt]=new; swaps+=info['swaps']
                    mask=new
                if self.corrects:
                    dense[lo:hi,head]=m.dense()
                    sparse[lo:hi,head]=m.output(mask)
                    dl[lo:hi,head]=m.log_scale
                    sl[lo:hi,head]=m.log_scale+m.selected(mask)[1].log()
                del m
        self.state=dict(identity=identity)
        if self.selects:
            if not bool((kept[0,:,:,:pt].sum(-1)==expected).all()):
                raise RuntimeError('value refinement changed the per-tile budget')
            self.add('updated_budget_units_checked',h*qb)
            st['kept']=kept; st['lists']=v27_fa4.block_sparse_tensors(kept)
            self.add('swapped_blocks',swaps)
        if self.corrects:
            if self.mode=='moments':
                self.state['correction']=SelectedOnlyCorrection.from_anchor(dense,dl,sparse,sl,identity)
            else:
                self.state['residual']=dense-sparse
            if self.mode.startswith('residual_') or self.direction_guard:
                if not hasattr(self,'projection'):
                    self.projection=self.model.model.layers[29].self_attn.o_proj.weight.float()
                if self.direction_guard:
                    self.state['direction_cache']=ResidualDirectionCache.build(dense,sparse,self.projection,identity)
                    del self.state['residual']
                else:
                    self.state['anchor_dense_post']=dense.reshape(n,-1)@self.projection.T
                    self.state['anchor_sparse_post']=sparse.reshape(n,-1)@self.projection.T
                self.add('extra_post_o_projections',2)
        self.add('anchor_events',1)
        self.add('anchor_full_qk_rectangles',h*qb*((prefix+n+63)//64))
        self.add('anchor_full_pv_rectangles',h*qb*((prefix+n+63)//64))
        self.counts['budget_errors']=0

    def receipt(self):
        return dict(value_refinement=self.mode,value_refinement_layers=[29],
            value_refinement_output_consumer='native FA4 plus cached correction' if self.corrects else 'unchanged native FA4',
            value_refinement_work=dict(self.counts),value_refinement_clean_timing=False)


def install(mode):
    from . import vllm_adapter
    from vllm.model_executor.models import diffusion_gemma as dg
    runtime=Runtime(mode)
    original_init=dg.DiffusionGemmaForConditionalGeneration.__init__
    @functools.wraps(original_init)
    def init(self,*a,**kw):
        original_init(self,*a,**kw);runtime.model=self
    dg.DiffusionGemmaForConditionalGeneration.__init__=init
    if mode=='full_cgate_swap8':
        original_sample=dg._compiled_sample_step;signature=inspect.signature(original_sample)
        @functools.wraps(original_sample)
        def sample(*args,**kwargs):
            bound=signature.bind(*args,**kwargs).arguments
            scaled=original_sample(*args,**kwargs)
            a=vllm_adapter._ACTIVE
            if a is not None and a.bound and a.step_ctx is not None and not a.step_ctx['encoder'] and scaled.shape[0]==1:
                runtime.sample_statistics(scaled[0],a.canvas_id,float(bound['entropy_bound']))
            return scaled
        dg._compiled_sample_step=sample
    original_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        if adapter.arm!='mage' or adapter.mage_granularity!='qblock_max' or adapter.mage_select!='fa4':
            raise ValueError('final-layer refinement requires the frozen V31 mass control')
        if adapter.stable_refresh_config is not None or adapter.value_selector is not None or adapter.cached_refresh_policy is not None:
            raise ValueError('do not silently combine separate selector families')
        old_mage=adapter._mage;old_begin=adapter.begin_request;old_receipt=adapter._value_receipt
        def mage(self,layer,q,b,scale,prefix,n):
            prior=self.mage_state.get(layer); previous_list=prior.get('lists') if prior is not None else None
            out=old_mage(layer,q,b,scale,prefix,n)
            return runtime.output(self,layer,q,b,scale,prefix,n,out,previous_list)
        def begin(self,*a,**kw):
            runtime.reset();return old_begin(*a,**kw)
        def receipt(self):
            return dict(old_receipt(),value_refinement_prefix_budget=self.mage_k//64,
                value_refinement_sticky_feedback='deployed changed support feeds inherited next-refresh sticky term',
                **runtime.receipt())
        adapter._mage=types.MethodType(mage,adapter)
        adapter.begin_request=types.MethodType(begin,adapter)
        adapter._value_receipt=types.MethodType(receipt,adapter)
        return original_install(adapter)
    vllm_adapter.install_vllm_patches=install_adapter
    return runtime
