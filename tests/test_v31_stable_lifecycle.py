"""Explicit native-lifecycle sequencing with deterministic fake consumers."""
import torch
import pytest
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
from experiments.numerical_qk_reuse.v31_stable_refresh import StableConfig, StableState


@pytest.mark.parametrize('progress_dense',[True,False])
@pytest.mark.parametrize('period',[0,1,4,8])
def test_dense_reason_clock_noops_carry_and_list_sync(monkeypatch,progress_dense,period):
    from experiments.numerical_qk_reuse import v27_fa4
    import experiments.numerical_qk_reuse.v31_stable_observe as ob
    builds=[]; outputs=[]
    def lists(mask): builds.append(mask.clone()); return mask.clone()
    def sparse(q,k,v,mask,scale): outputs.append(mask.clone()); return torch.tensor(float(mask.sum()))
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',lists)
    monkeypatch.setattr(v27_fa4,'sparse_lists',sparse)
    monkeypatch.setattr(v27_fa4,'dense',lambda *a:torch.tensor(-1.))
    # Favor exactly the held map, so observations are no-ops but the clock advances.
    monkeypatch.setattr(ob,'observe',lambda q,k,scale,ids,n,out=None:torch.zeros((*ids.shape,128)))
    cfg=dict(selector=dict(mode='stable',hysteresis=.1,persistence=2,removal_tolerance=.01),
             period=period,progress_dense=progress_dense)
    a=VllmMethodAdapter(['full_attention'],arm='mage',mage_select='fa4',mage_k=128,
        mage_granularity='qblock_max',mage_select_step=1,mage_carry_first=True,
        mage_reselect_trigger=.15,mage_trigger_signal='settle',mage_sticky=1.386,
        stable_refresh_config=cfg)
    a.begin_request()
    def select(layer,q,b,scale,prefix,n):
        pt=prefix//64; kt=(prefix+n+63)//64
        mask=torch.zeros(1,1,1,kt,dtype=torch.bool); mask[..., :2]=True; mask[...,pt:]=True
        z=torch.full((1,pt,128),-1.); z[:,:2]=0.
        state=StableState(z,mask[0,:,:,:pt].reshape(1,pt),StableConfig(**cfg['selector']))
        a._stable_pending[layer]=state
        return torch.tensor(-2.),mask
    monkeypatch.setattr(a,'_select_initial_refresh',select)
    q=torch.zeros(1,1,128,16); b=dict(k=torch.zeros(1,1,320,16),v=torch.zeros(1,1,320,16))
    a.canvas_id=1
    assert a._mage(0,q,b,.25,192,128)==-1.
    a._mage(0,q,b,.25,192,128)
    first_lists=a.mage_state[0]['lists']
    for step in range(2,11):
        a._trig_at=(1,step) if step==6 else None
        a._mage(0,q,b,.25,192,128)
    reasons=a._stable_reasons
    assert reasons['bootstrap_dense']==1
    assert reasons['canvas_selection_dense']==1
    assert reasons['progress_selection_dense']==int(progress_dense)
    # Nine eligible reuse calls, or split into four + four when progress refresh consumes step 6.
    expected=(8//period if period else 0) if not progress_dense else (2*(4//period) if period else 0)
    if not progress_dense: expected=9//period if period else 0
    assert reasons['cheap_observations']==expected
    assert reasons['actual_updates']==0
    assert len(builds)==1+int(progress_dense)
    assert torch.equal(a.mage_state[0]['kept'],a.mage_state[0]['lists'])
    if not progress_dense: assert a.mage_state[0]['lists'] is first_lists
    a.canvas_id=2
    b=dict(k=torch.zeros(1,1,448,16),v=torch.zeros(1,1,448,16))
    a._mage(0,q,b,.25,320,128)
    assert reasons['carry_calls']==1
    assert reasons['bootstrap_dense']==1
    a._mage(0,q,b,.25,320,128)
    assert reasons['canvas_selection_dense']==2
    assert torch.equal(outputs[0],builds[0])
