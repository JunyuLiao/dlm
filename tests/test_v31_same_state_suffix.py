import torch

from experiments.numerical_qk_reuse.v31_same_state_suffix import compare, distribution


def test_acceptance_matches_official_entropy_bound_rule():
    x=torch.tensor([[12.,0.,0.],[1.,2.,3.],[0.,0.,0.],[0.,0.,12.]])
    logp=x.log_softmax(-1); entropy=-(logp.exp()*logp).sum(-1)
    ordered,index=entropy.sort(-1); expected=torch.zeros_like(entropy,dtype=torch.bool)
    expected[index]=(ordered.cumsum(-1)-ordered.cummax(-1).values)<=.1
    assert torch.equal(distribution(x)['accepted'],expected)


def test_identical_logits_have_no_diagnostic_decision_changes():
    x=torch.randn(8,23); noise=torch.randn_like(x)
    out=compare(x,x,noise,x.argmax(-1))
    for key in ('top1_disagreement','acceptance_disagreement','common_noise_disagreement','kl_mean','kl_max','max_logit_error'):
        assert out[key]==0
    assert not out['stop_flip']


def test_stop_uses_both_stability_and_confidence():
    confident=torch.tensor([[30.,0.],[30.,0.]])
    altered=confident.flip(-1)
    out=compare(altered,confident,torch.zeros_like(confident),torch.zeros(2,dtype=torch.long))
    assert out['reference_stop'] and not out['candidate_stop'] and out['stop_flip']
    assert out['top1_disagreement']==1
