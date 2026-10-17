import pytest
import torch
from experiments.numerical_qk_reuse.v31_value_fast_runtime import moments_full,FastRuntime
from experiments.numerical_qk_reuse.v31_value_decision import moments
from experiments.numerical_qk_reuse.v31_value_fast import dense_output, selected_ids, selected_output
from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache

@pytest.mark.parametrize('rows,prefix,length',[(1,0,7),(31,127,140),(128,256,384),(65,513,578)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_moments_and_correction(rows,prefix,length,dtype):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    g=torch.Generator(device=device).manual_seed(15)
    q=torch.randn(rows,32,device=device,dtype=dtype,generator=g)
    k=torch.randn(length,32,device=device,dtype=dtype,generator=g)
    v=torch.randn(length,32,device=device,dtype=dtype,generator=g)
    ref=moments(q,k,v,prefix,32**-.5);fast=moments_full(q,k,v,prefix,32**-.5)
    for name in ('mass','numerator','tail_mass','tail_numerator','log_scale'):
        assert torch.equal(getattr(ref,name),getattr(fast,name))
    mask=torch.arange(prefix//64,device=device)%2==0;budget=(prefix//64+1)//2
    num,den=selected_output(fast,selected_ids(mask,budget));selected=num/den[:,None]
    assert torch.equal(ref.dense(),dense_output(fast))
    assert torch.equal(ref.output(mask),selected)
    w=torch.randn(16,32,device=device,dtype=dtype,generator=g)
    a=ResidualDirectionCache.build(ref.dense()[:,None],ref.output(mask)[:,None],w,1)
    b=ResidualDirectionCache.build(dense_output(fast)[:,None],selected[:,None],w,1)
    current=selected[:,None]+.01
    assert torch.equal(a.apply(current,1)[0],b.apply(current,1)[0])
    if device=='cuda':assert torch.equal(a.apply_cuda(current,1)[0],b.apply_cuda(current,1)[0])


def test_pass_contract_and_reset(monkeypatch):
    monkeypatch.setenv('VALUE_AUDIT','0');monkeypatch.setenv('VALUE_CLEAN_TIMING','1')
    r=FastRuntime('full_swap8');r.state={'identity':1};r.counts={'x':1};r.events=[1]
    r.reset();assert r.state is None and not r.counts and not r.events
    assert r.receipt()['value_refinement_clean_timing']
    with pytest.raises(ValueError):FastRuntime('residual')
    monkeypatch.setenv('VALUE_AUDIT','1')
    with pytest.raises(ValueError):FastRuntime('full_swap8')
