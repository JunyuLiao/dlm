import importlib.util
from pathlib import Path

import torch

path=Path(__file__).resolve().parents[1]/'code/diagnose.py'
spec=importlib.util.spec_from_file_location('operator_diagnose',path)
diagnose=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnose)


def test_full_qkv_directional_response_matches_central_difference():
    generator=torch.Generator().manual_seed(42)
    q=torch.randn(3,4,generator=generator,dtype=torch.float64)
    k=torch.randn(9,4,generator=generator,dtype=torch.float64)
    v=torch.randn(9,5,generator=generator,dtype=torch.float64)
    dq=torch.randn(q.shape,generator=generator,dtype=torch.float64)
    dk=torch.randn(k.shape,generator=generator,dtype=torch.float64)
    dv=torch.randn(v.shape,generator=generator,dtype=torch.float64)
    scale=.7;eps=1e-6
    p=diagnose.probability(q,k,scale)
    response=diagnose.differential(p,q,k,v,dq,dk,dv,scale)
    finite=(diagnose.probability(q+eps*dq,k+eps*dk,scale)@(v+eps*dv)-
            diagnose.probability(q-eps*dq,k-eps*dk,scale)@(v-eps*dv))/(2*eps)
    torch.testing.assert_close(response['total'],finite,rtol=1e-6,atol=1e-8)


def test_candidates_keep_exact_physical_budget_and_mandatory_tail():
    p=torch.full((4,641),1/641,dtype=torch.float64)
    v=torch.arange(641*3,dtype=torch.float64).reshape(641,3)
    for budget in (1,3,7):
        bank=diagnose.candidate_bank(p,v,449,budget)
        for mask in bank.values():
            support=diagnose.token_support(mask,641)
            assert mask.sum()==budget
            assert support[448:].all()
            assert support[:448].sum()==budget*64
            _,_,mass=diagnose.output(p,v,support)
            assert (mass>0).all()


def test_renormalization_and_prefix_certificate_bound():
    generator=torch.Generator().manual_seed(2608)
    q=torch.randn(4,3,generator=generator,dtype=torch.float64)
    k=torch.randn(200,3,generator=generator,dtype=torch.float64)
    v=torch.randn(200,5,generator=generator,dtype=torch.float64)
    dq=torch.randn(q.shape,generator=generator,dtype=torch.float64)*.01
    mask=torch.tensor([True,False])
    support=diagnose.token_support(mask,200)
    p=diagnose.probability(q,k,.5)
    _,ps,mass=diagnose.output(p,v,support)
    torch.testing.assert_close(ps[:,128:].sum(-1),p[:,128:].sum(-1)/mass)
    u=(p[:,:128]*support[:128]).sum(-1)/p[:,:128].sum(-1)
    bound,_=diagnose.retained_bound(u,dq,k[:128],.5)
    next_p=diagnose.probability(q+dq,k,.5)
    actual=(next_p[:,:128]*support[:128]).sum(-1)/next_p[:,:128].sum(-1)
    assert (bound<=actual+1e-12).all()
