"""The updates-disabled study control preserves inherited V31 map sequencing."""
import pytest
import torch
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter


@pytest.mark.parametrize('heads,n,prefix',[(1,128,64),(2,256,512),(2,130,512)])
def test_original_and_study_control_same_dense_carry_and_sparse_sequence(monkeypatch,heads,n,prefix):
    from experiments.numerical_qk_reuse import v27_fa4
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',lambda mask:mask.clone())
    monkeypatch.setattr(v27_fa4,'dense',lambda *args:torch.tensor(-1.))
    weights=lambda mask:(mask.long()*torch.arange(1,mask.shape[-1]+1)).sum()
    monkeypatch.setattr(v27_fa4,'sparse_lists',lambda q,k,v,mask,scale:weights(mask))
    options=dict(arm='mage',mage_select='fa4',mage_k=128,mage_granularity='qblock_max',
                 mage_select_step=1,mage_carry_first=True,mage_reselect_trigger=.15,
                 mage_trigger_signal='settle',mage_sticky=1.386)
    original=VllmMethodAdapter(['full_attention'],**options)
    study=VllmMethodAdapter(['full_attention'],**options,
        stable_refresh_config=dict(period=0,progress_dense=True,selector=dict(mode='raw',updates_enabled=False)))
    selections=[[],[]]
    for idx,adapter in enumerate((original,study)):
        adapter.begin_request()
        def select(layer,q,b,scale,prefix,n,adapter=adapter,idx=idx):
            h,pt,qb=q.shape[1],prefix//64,(n+127)//128;kt=(prefix+n+63)//64
            # Different dense events really re-rank, and use inherited sticky/protection code.
            rows=torch.arange(n)[None,:,None];tiles=torch.arange(pt)[None,None,:]
            z=-(tiles-(len(selections[idx])+rows%3)).abs().float().expand(h,-1,-1).clone()
            mass=torch.softmax(torch.cat((z,torch.zeros(h,n,kt-pt)),dim=-1),dim=-1)
            kept=adapter._mage_units(mass,z,h,n,qb,kt,pt,min(2,pt),1)
            selections[idx].append(kept.clone())
            if adapter.stable_refresh_config is not None:adapter._stable_prepare(layer,z,kept,prefix,n)
            return torch.tensor(-2.),kept
        monkeypatch.setattr(adapter,'_select_initial_refresh',select)
    for canvas in range(1,4):
        current_prefix=prefix+(canvas-1)*n
        q=torch.zeros(1,heads,n,16)
        b=dict(k=torch.zeros(1,1,current_prefix+n,16),v=torch.zeros(1,1,current_prefix+n,16))
        for step in range(12):
            for adapter in (original,study):
                adapter.canvas_id=canvas
                adapter._trig_at=(canvas,step) if step in (4,9) else None
            assert torch.equal(original._mage(0,q,b,.25,current_prefix,n),study._mage(0,q,b,.25,current_prefix,n))
            if original.mage_state.get(0) is not None:
                assert torch.equal(original.mage_state[0]['kept'],study.mage_state[0]['kept'])
    assert len(selections[0])==len(selections[1])
    assert all(torch.equal(a,b) for a,b in zip(*selections))
    for key in ('mage_selections','mage_reselections','mage_reused_calls','mage_carried_calls','mage_warm_dense_calls'):
        assert original.calls[key]==study.calls[key]
