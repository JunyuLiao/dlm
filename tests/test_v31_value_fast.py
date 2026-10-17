import pytest
import torch

from experiments.numerical_qk_reuse.v31_value_decision import moments, bounded_proposal, fixed_topk, mass_score
from experiments.numerical_qk_reuse.v31_value_fast import bounded_proposal_device, dense_output


@pytest.mark.parametrize('backend', ['exact', 'factored'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_device_formula_matches_reference_float64(seed, backend):
    g=torch.Generator().manual_seed(seed)
    q=torch.randn(7,16,generator=g,dtype=torch.float64)
    k=torch.randn(64*13+5,16,generator=g,dtype=torch.float64)
    v=torch.randn(k.shape,generator=g,dtype=torch.float64)
    m=moments(q,k,v,64*13+2,.25)
    anchor=fixed_topk(mass_score(m).roll(3),5)
    w=torch.randn(16,16,generator=g,dtype=torch.float64);gram=w.T@w
    expected,info=bounded_proposal(m,anchor,swaps=3,shortlist=4,gram=gram)
    actual,fast=bounded_proposal_device(m,anchor,5,swaps=3,shortlist=4,gram=gram,backend=backend)
    assert torch.equal(expected,actual)
    assert int(actual.sum())==5 and int(fast['swaps'])==info['swaps']
    torch.testing.assert_close(dense_output(m),m.dense(),rtol=0,atol=0)


@pytest.mark.parametrize('budget',[0,3])
def test_allkept_and_empty_keep_budget(budget):
    q=torch.ones(2,4);k=torch.ones(64*3+1,4);v=torch.ones_like(k)
    m=moments(q,k,v,192,.5);anchor=torch.arange(3)<budget
    actual,info=bounded_proposal_device(m,anchor,budget)
    assert torch.equal(actual,anchor) and int(info['swaps'])==0


def test_empty_mass_neighborhood_does_not_add_padded_candidates():
    q=torch.ones(2,4);k=torch.ones(64*8+1,4);v=torch.ones_like(k)
    m=moments(q,k,v,512,.5);anchor=torch.arange(8)<3
    actual,info=bounded_proposal_device(m,anchor,3,extra=0)
    assert torch.equal(actual,anchor) and int(info['swaps'])==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires qualified CUDA')
def test_disjoint_cuda_ties_and_exhaustion():
    from experiments.numerical_qk_reuse.v31_value_fast import rank_disjoint
    p=torch.tensor([[7,1],[7,2],[8,1],[8,2]],device='cuda')
    loss=torch.tensor([1.,1.,1.,1.],device='cuda')
    idx,valid=rank_disjoint(p,loss,torch.tensor(2.,device='cuda'),4)
    assert idx[:2].tolist()==[0,3] and valid.tolist()==[True,True,False,False]
