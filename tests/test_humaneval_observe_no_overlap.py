from types import SimpleNamespace
import torch
import pytest
from scripts.v31_humaneval_observe_no_overlap import wrap_fwd,install


def mass_aux():return [torch.empty(16,2,7,128),torch.tensor([1.])]


def test_observer_changes_only_overlap_and_preserves_return_identity():
    calls=[];counts={'observing_calls':0};sentinel=object();q,k,v=object(),object(),object();aux=mass_aux()
    def native(*a,**kw):calls.append((a,kw));return sentinel
    wrapped=wrap_fwd(native,counts);dynamic=torch.tensor([False])
    assert wrapped(q,k,v,softmax_scale=1.,causal=True,dynamic_causal=dynamic,aux_tensors=aux) is sentinel
    args,kwargs=calls[0];assert args==(q,k,v) and len(calls)==1
    assert kwargs==dict(softmax_scale=1.,causal=True,dynamic_causal=dynamic,aux_tensors=aux,intra_wg_overlap=False)
    assert counts['observing_calls']==1 and kwargs['aux_tensors'] is aux


def test_plain_arguments_and_explicit_policy_conflicts():
    calls=[];counts={'observing_calls':0}
    def native(*a,**k):calls.append(k);return a
    wrapped=wrap_fwd(native,counts)
    assert wrapped(1,2,3,causal=False,num_splits=1)==(1,2,3)
    assert calls==[dict(causal=False,num_splits=1)] and counts['observing_calls']==0
    with pytest.raises(RuntimeError,match='conflicting'):wrapped(1,2,3,aux_tensors=mass_aux(),intra_wg_overlap=True)
    assert len(calls)==1


def test_unknown_auxiliary_path_is_rejected_before_native_call():
    counts={'observing_calls':0};calls=[]
    wrapped=wrap_fwd(lambda *a,**k:calls.append(True),counts)
    with pytest.raises(RuntimeError,match='unrecognized'):wrapped(1,2,3,aux_tensors=[torch.ones(1)])
    assert calls==[] and counts['observing_calls']==0


def test_policy_binds_after_adapter_and_resets_receipt_per_request(monkeypatch):
    from experiments.numerical_qk_reuse import vllm_adapter,v27_fa4,v31_fa4_observe as observe
    from vllm.vllm_flash_attn.cute import interface,flash_fwd_sm90 as sm90
    assert observe._sm90 is sm90
    ordered=[];native_calls=[]
    def native(*a,**k):native_calls.append(k);return 'native'
    monkeypatch.setattr(observe,'_flash_attn_fwd',native);monkeypatch.setattr(v27_fa4,'_FWD',native)
    def load():assert ordered;return v27_fa4._FWD
    monkeypatch.setattr(v27_fa4,'load',load)
    def old(adapter):ordered.append('native_install');return adapter
    monkeypatch.setattr(vllm_adapter,'install_vllm_patches',old)
    starts=[]
    class Adapter:
        def begin_request(self,*a,**k):starts.append(a)
        def _value_receipt(self):return dict(prior=1)
    monkeypatch.setattr(vllm_adapter,'VllmMethodAdapter',Adapter)
    install();adapter=Adapter();captured=adapter._value_receipt
    assert vllm_adapter.install_vllm_patches(adapter) is adapter
    # The real shared worker rebinds receipts from this earlier method capture.
    adapter._value_receipt=lambda:dict(captured(),shared_dispatch=1)
    adapter.begin_request('first')
    assert observe._flash_attn_fwd(1,2,3,aux_tensors=mass_aux())=='native'
    assert v27_fa4._FWD(1,2,3,causal=False)=='native'
    receipt=adapter._value_receipt();assert receipt['prior']==1 and receipt['observing_fa4_calls']==1
    assert receipt['observing_fa4_extra_forwards']==0 and receipt['observing_fa4_intra_wg_overlap'] is False
    adapter.begin_request('second');assert adapter._value_receipt()['observing_fa4_calls']==0
    assert starts==[('first',),('second',)] and len(native_calls)==2
