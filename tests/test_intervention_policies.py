import torch
from experiments.value_direction_hopper.intervention_policies import choose_dense,protection,additional


def test_one_based_schedules_and_single_correction():
    assert [choose_dense('first',s) for s in range(1,6)]==[True,False,False,False,False]
    assert [choose_dense('late',s) for s in range(1,6)]==[False,False,False,True,True]
    limits=dict(churn=.2,entropy_floor=.005,progress=0.,acceptance_progress=0.)
    previous=dict(churn=.3,entropy=.1,entropy_progress=.01,accepted_fraction=.8,acceptance_progress=.1)
    assert choose_dense('reactive',3,False,previous,limits)
    assert not choose_dense('reactive',4,True,previous,limits)
    assert not choose_dense('reactive',2,False,dict(previous,churn=None),limits)


def test_protection_opens_on_changed_prediction():
    top=torch.tensor([[1,2,3]]);conf=torch.tensor([[.9999,.9999,.9999]]);margin=conf.clone()
    p,t=protection(None,None,top,conf,margin,None)
    assert not p.any()
    p,t=protection(p,t,top,conf,margin,top)
    assert p.all()
    changed=torch.tensor([[4,2,3]])
    p,t=protection(p,t,changed,torch.tensor([[.9,.99,.97]]),margin,top)
    assert p.tolist()==[[False,True,False]]


def test_random_and_ranked_share_pool_quota_without_changing_global_rng():
    mask=torch.tensor([[True,False,False,False,False]])
    entropy=torch.tensor([[0.,.2,.3,.4,2.]])
    confidence=torch.tensor([[1.,.9,.8,.7,.2]])
    state=torch.random.get_rng_state().clone()
    random,pool=additional(mask,entropy,confidence,2,torch.Generator().manual_seed(42))
    ranked,pool2=additional(mask,entropy,confidence,2)
    assert pool==pool2==[1,2,3]
    assert random.sum()==ranked.sum()==2
    assert not random[0,4] and not ranked[0,4]
    assert torch.equal(state,torch.random.get_rng_state())
