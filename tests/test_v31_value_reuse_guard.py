import pytest
import torch

from experiments.numerical_qk_reuse.v31_value_reuse_guard import SelectedOnlyCorrection, reconstruct_cuda, residual_trust


def test_direction_guard_is_optimal_for_its_stated_surrogate():
    g=torch.Generator().manual_seed(17)
    old=torch.randn(7,13,generator=g); dense=torch.randn(7,13,generator=g)
    current=torch.randn(7,13,generator=g)
    alpha=residual_trust(current,old,dense)
    chosen=(current+alpha[:,None]*(dense-old)-dense).square().sum(-1)
    for a in torch.linspace(0,1,41):
        assert torch.all(chosen<=(current+a*(dense-old)-dense).square().sum(-1)+1e-5)


def test_guard_handles_zero_residual_and_reaches_old_dense():
    old=torch.zeros(2,3); dense=torch.ones(2,3)
    assert torch.equal(residual_trust(old,old,dense),torch.ones(2))
    assert torch.equal(residual_trust(dense,old,dense),torch.zeros(2))
    assert torch.equal(residual_trust(old,old,old),torch.zeros(2))


def example(device='cpu', dim=8):
    g=torch.Generator().manual_seed(8)
    values=torch.randn((2,5,64,dim),generator=g,device='cpu').to(device)
    scores=torch.randn((2,5,64),generator=g,device='cpu').to(device)
    dense=scores.softmax(-1).unsqueeze(-2)@values
    selected=scores[...,:23].softmax(-1).unsqueeze(-2)@values[...,:23,:]
    dl=scores.logsumexp(-1); sl=scores[...,:23].logsumexp(-1)
    return dense.squeeze(-2),selected.squeeze(-2),dl,sl,scores,values


def test_selected_only_merge_matches_fresh_selected_cached_omitted_oracle():
    dense,selected,dl,sl,scores,v=example()
    cache=SelectedOnlyCorrection.from_anchor(dense,dl,selected,sl,('canvas',1))
    torch.testing.assert_close(cache.reconstruct(selected,sl,('canvas',1)),dense,atol=2e-6,rtol=2e-5)
    changed=scores.clone(); changed[...,:23]+=torch.linspace(-2,2,23)
    new=(changed[...,:23].softmax(-1).unsqueeze(-2)@v[...,:23,:]).squeeze(-2)
    expected=(changed.softmax(-1).unsqueeze(-2)@v).squeeze(-2)
    torch.testing.assert_close(cache.reconstruct(new,changed[...,:23].logsumexp(-1),('canvas',1)),expected,atol=2e-6,rtol=2e-5)
    with pytest.raises(ValueError,match='rebuild'):
        cache.reconstruct(new,sl,('canvas',2))


def test_allkept_is_exact_noop():
    dense,_,dl,_,_,_=example()
    cache=SelectedOnlyCorrection.from_anchor(dense,dl,dense,dl,'all')
    assert torch.equal(cache.reconstruct(dense,dl,'all'),dense)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('dim',[8,512])
def test_fused_merge_matches_reference_without_omitted_inputs(dtype,dim):
    dense,selected,dl,sl,_,_=example('cuda',dim)
    cache=SelectedOnlyCorrection.from_anchor(dense,dl,selected,sl,'fixed')
    actual=reconstruct_cuda(selected.to(dtype),sl,cache,'fixed')
    expected=cache.reconstruct(selected.to(dtype),sl,'fixed').to(dtype)
    torch.testing.assert_close(actual,expected,atol=1e-6 if dtype==torch.float32 else .004,rtol=1e-5 if dtype==torch.float32 else .008)
