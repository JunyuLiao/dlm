"""Opt-in final-GLOBAL-layer native variants; disabled until diagnostic promotion.

Uses the independent selector and an explicit per-request correction cache.
Native bootstrap, progress clock, carry, and sampler are inherited unchanged.
The reconstruction variants change attention output and say so in receipts.
"""
from __future__ import annotations

import functools
import types

import torch

from .v31_value_decision import bounded_proposal, moments
from .v31_value_reuse_guard import SelectedOnlyCorrection, reconstruct_cuda, residual_trust

MODES=('full_swap8','online_swap8','residual','moments','residual_direction_guard','residual_norm_guard')


class Runtime:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('unknown explicit value refinement')
        self.mode=mode; self.model=None; self.state=None; self.grams={}; self.counts={}

    def add(self,name,amount):
        self.counts[name]=self.counts.get(name,0)+amount

    def reset(self):
        self.state=None; self.counts={}

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
        if self.mode.endswith('swap8'):
            return out                         # FA4 already used the replaced map
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
        if not bool((counts==expected).all()):
            raise RuntimeError('fixed per-query-tile budget is not the configured K')
        if expected==pt:
            self.state=dict(identity=identity,allkept=True)
            self.add('allkept_shortcuts',1)
            return
        if not self.grams and self.mode.endswith('swap8'):
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
                if self.mode.endswith('swap8'):
                    new,info=bounded_proposal(m,mask,swaps=8,online=self.mode=='online_swap8',gram=self.grams[head])
                    kept[0,head,qt,:pt]=new; swaps+=info['swaps']
                else:
                    dense[lo:hi,head]=m.dense()
                    sparse[lo:hi,head]=m.output(mask)
                    dl[lo:hi,head]=m.log_scale
                    sl[lo:hi,head]=m.log_scale+m.selected(mask)[1].log()
                del m
        self.state=dict(identity=identity)
        if self.mode.endswith('swap8'):
            if not bool((kept[0,:,:,:pt].sum(-1)==expected).all()):
                raise RuntimeError('value refinement changed the per-tile budget')
            st['kept']=kept; st['lists']=v27_fa4.block_sparse_tensors(kept)
            self.add('swapped_blocks',swaps)
        else:
            self.state['residual']=dense-sparse
            self.state['correction']=SelectedOnlyCorrection.from_anchor(dense,dl,sparse,sl,identity)
            if self.mode.startswith('residual_'):
                if not hasattr(self,'projection'):
                    self.projection=self.model.model.layers[29].self_attn.o_proj.weight.float()
                self.state['anchor_dense_post']=dense.reshape(n,-1)@self.projection.T
                self.state['anchor_sparse_post']=sparse.reshape(n,-1)@self.projection.T
                self.add('extra_post_o_projections',2)
        self.add('anchor_events',1)
        self.add('anchor_full_qk_rectangles',h*qb*((prefix+n+63)//64))
        self.add('anchor_full_pv_rectangles',h*qb*((prefix+n+63)//64))
        self.counts['budget_errors']=0

    def receipt(self):
        return dict(value_refinement=self.mode,value_refinement_layers=[29],
            value_refinement_output_consumer='unchanged native FA4' if self.mode.endswith('swap8') else 'native FA4 plus cached correction',
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
            return dict(old_receipt(),**runtime.receipt())
        adapter._mage=types.MethodType(mage,adapter)
        adapter.begin_request=types.MethodType(begin,adapter)
        adapter._value_receipt=types.MethodType(receipt,adapter)
        return original_install(adapter)
    vllm_adapter.install_vllm_patches=install_adapter
    return runtime
