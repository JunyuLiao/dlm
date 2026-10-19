"""Named numerical correction: disable overlap only for mass-observing FA4.

Installation binds after the native mask fix. Plain FA4, routing, budgets,
current K/V, graph dispatch and the native sampler retain their call arguments.
"""
import functools
import hashlib
from pathlib import Path


def wrap_fwd(original,counts):
    @functools.wraps(original)
    def call(q,k,v,*args,**kwargs):
        aux=kwargs.get('aux_tensors')
        if aux:
            import torch
            if (len(aux)!=2 or aux[0].dtype!=torch.float32 or aux[0].ndim!=4 or aux[0].shape[-1]!=128 or
                    aux[1].dtype!=torch.float32 or aux[1].numel()!=1):
                raise RuntimeError('unrecognized auxiliary FA4 path under observation-only numerical policy')
            if kwargs.get('intra_wg_overlap') not in (None,False):
                raise RuntimeError('conflicting explicit FA4 overlap setting')
            kwargs=dict(kwargs,intra_wg_overlap=False)
            counts['observing_calls']+=1
        return original(q,k,v,*args,**kwargs)
    return call


def install():
    from experiments.numerical_qk_reuse import vllm_adapter
    original_install=vllm_adapter.install_vllm_patches
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_receipt=vllm_adapter.VllmMethodAdapter._value_receipt
    @functools.wraps(original_receipt)
    def evidence(adapter,*args,**kwargs):
        counts=getattr(adapter,'_observing_fa4_policy_counts',dict(observing_calls=0))
        return dict(original_receipt(adapter,*args,**kwargs),observing_fa4_numerical_policy='no_intra_wg_overlap',
            observing_fa4_policy_sha256=source_sha,observing_fa4_calls=counts['observing_calls'],
            observing_fa4_intra_wg_overlap=False,observing_fa4_extra_forwards=0,
            observing_fa4_plain_arguments_preserved=True)
    # Shared dispatch captures this class method before adapter installation.
    # Patch that capture point, so control/value/refinement receipts all retain it.
    vllm_adapter.VllmMethodAdapter._value_receipt=evidence
    @functools.wraps(original_install)
    def installed(adapter):
        result=original_install(adapter)
        # Deferred import: the base worker has already installed its native mask fix.
        from experiments.numerical_qk_reuse import v27_fa4,v31_fa4_observe as observe
        from vllm.vllm_flash_attn.cute import interface,flash_fwd_sm90 as sm90
        assert observe._sm90 is sm90 and interface.FlashAttentionForwardSm90 is sm90.FlashAttentionForwardSm90
        v27_fa4.load()
        counts=dict(observing_calls=0)
        adapter._observing_fa4_policy_counts=counts
        observe._flash_attn_fwd=wrap_fwd(observe._flash_attn_fwd,counts)
        v27_fa4._FWD=wrap_fwd(v27_fa4._FWD,counts)
        begin=adapter.begin_request
        @functools.wraps(begin)
        def starting(*args,**kwargs):
            counts['observing_calls']=0
            return begin(*args,**kwargs)
        adapter.begin_request=starting
        return result
    vllm_adapter.install_vllm_patches=installed
