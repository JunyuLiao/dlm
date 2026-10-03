import torch
from experiments.value_direction_hopper.protection_refinement import update_protection,variants


def test_hysteresis_uses_stored_token_probability():
    was=torch.tensor([[True,True,True]])
    stored=torch.tensor([[1,1,1]]);top=torch.tensor([[2,2,1]])
    p=torch.tensor([[.95,.51,.93]]);sp=torch.tensor([[.01,.48,.93]])
    marked,tokens=update_protection(variants()['hysteresis'],was,stored,top,p,torch.zeros_like(p),stored,sp)
    assert marked.tolist()==[[False,True,True]]
    assert tokens.tolist()==stored.tolist()


def test_immediate_vs_one_transition_and_margin():
    top=torch.tensor([[1,2]]);p=torch.tensor([[.9992,.99995]]);margin=torch.tensor([[.9984,.9999]])
    empty=torch.zeros_like(top,dtype=torch.bool)
    args=(None,None,top,p,margin,None,p)
    assert not update_protection(variants()['existing'],*args)[0].any()
    assert update_protection(variants()['immediate'],*args)[0].all()
    assert update_protection(variants()['very_confident'],*args)[0].tolist()==[[False,True]]
    args=(empty,top,top,p,margin,top,p)
    assert update_protection(variants()['confidence_only'],*args)[0].all()
    assert update_protection(variants()['strict_margin'],*args)[0].tolist()==[[False,True]]


def test_old_margin_is_redundant():
    p=torch.linspace(.999,1.,100)
    assert ((2*p-1)>=.998-1e-7).all()
