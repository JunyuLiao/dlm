import pytest
import torch
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter


def adapter(**changes):
    options=dict(arm='mage',mage_select='fa4',mage_k=128,mage_granularity='qblock_max',
                 mage_always_refresh=True,value_audit=True)
    options.update(changes)
    a=VllmMethodAdapter(['full_attention'],**options);a.begin_request();return a


def test_each_call_selects_current_map_and_returns_same_call_sparse_output(monkeypatch):
    from experiments.numerical_qk_reuse import v27_fa4
    a=adapter(); seen=[]
    def select(layer,q,b,scale,prefix,n):
        mask=torch.zeros(1,1,1,4,dtype=torch.bool)
        mask[...,int(q[0,0,0,0])%3]=True
        mask[...,(int(q[0,0,0,0])+1)%3]=True
        mask[...,3]=True
        assert a._mage_held is None and a._mage_units_w is None
        seen.append(mask.clone())
        return torch.tensor(999.),mask
    monkeypatch.setattr(a,'_select_initial_refresh',select)
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',lambda x:x)
    monkeypatch.setattr(v27_fa4,'sparse_lists',lambda q,k,v,m,s:m[...,0].float().sum())
    b=dict(k=torch.zeros(1,1,256,16),v=torch.zeros(1,1,256,16))
    a.canvas_id=1
    for number in (0,1,2):
        q=torch.full((1,1,64,16),float(number))
        got=a._mage(0,q,b,.25,192,64)
        assert got==seen[-1][...,0].float().sum() and got!=999
    assert len(seen)==3 and not torch.equal(seen[0],seen[1])
    assert a.calls['every_step_dense_observations']==3
    assert a.calls['mage_reused_calls']==0 and not a.mage_state
    assert a.calls['every_step_observer_qk_rectangles']==12
    assert a.calls['every_step_observer_pv_rectangles']==12
    assert a._cached_receipt()['every_step_output']=='same_call_native_sparse_fa4'


@pytest.mark.parametrize('change',[dict(mage_carry_first=True),dict(mage_select_step=1),
    dict(mage_sticky=1.),dict(stable_refresh_config={}),dict(mage_reselect_trigger=.15)])
def test_stateful_selector_options_are_rejected(change):
    with pytest.raises(ValueError):adapter(**change)


def test_wrong_budget_is_exposed(monkeypatch):
    a=adapter()
    monkeypatch.setattr(a,'_select_initial_refresh',lambda *args:(torch.tensor(0.),torch.ones(1,1,1,4,dtype=torch.bool)))
    with pytest.raises(AssertionError):
        a._mage(0,torch.zeros(1,1,64,16),dict(k=None,v=None),1.,192,64)
