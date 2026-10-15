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


def test_direction_guard_regret_bound_including_clipped_optima():
    g=torch.Generator().manual_seed(81)
    old,dense,current,new_dense=[torch.randn(128,17,generator=g,dtype=torch.float64) for _ in range(4)]
    r=dense-old; r2=r.square().sum(-1)
    alpha=residual_trust(current,old,dense)
    oracle=(-(current-new_dense).mul(r).sum(-1)/r2).clamp(0,1)
    regret=(current+alpha[:,None]*r-new_dense).square().sum(-1)-(current+oracle[:,None]*r-new_dense).square().sum(-1)
    bound=((new_dense-dense)*r).sum(-1).square()/r2
    assert bool((regret>=-1e-12).all())
    assert bool((regret<=bound+1e-12).all())


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


def test_adjoint_direction_cache_matches_explicit_post_o_guard():
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
    g=torch.Generator().manual_seed(611)
    selected=torch.randn(23,4,17,dtype=torch.float64,generator=g)
    dense=selected+.3*torch.randn(selected.shape,dtype=torch.float64,generator=g)
    current=selected+.2*torch.randn(selected.shape,dtype=torch.float64,generator=g)
    w=torch.randn(31,68,dtype=torch.float64,generator=g)
    cache=ResidualDirectionCache.build(dense,selected,w,'map1')
    out,a=cache.apply(current,'map1')
    explicit=residual_trust(current.flatten(1)@w.T,selected.flatten(1)@w.T,dense.flatten(1)@w.T)
    torch.testing.assert_close(a,explicit,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(out,current+explicit[:,None,None]*(dense-selected),rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='rebuild'):
        cache.apply(current,'map2')
    zero=ResidualDirectionCache.build(selected,selected,w,'zero')
    assert torch.equal(zero.apply(current,'zero')[0],current)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('width',[68,8192])
def test_fused_adjoint_guard_matches_reference(dtype,width):
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
    g=torch.Generator().manual_seed(612)
    selected=torch.randn(9,1,width,generator=g).cuda()
    dense=selected+.3*torch.randn(selected.shape,generator=g).cuda()
    beta=torch.linspace(-.5,1.5,9,device='cuda')[:,None,None]
    current=(selected+beta*(dense-selected)+.05*torch.randn(selected.shape,generator=g).cuda()).to(dtype)
    w=torch.randn(128,width,generator=g).cuda()/width**.5
    cache=ResidualDirectionCache.build(dense,selected,w,'map')
    expected,a=cache.apply(current,'map')
    actual,b=cache.apply_cuda(current,'map')
    torch.testing.assert_close(b,a,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual,expected,rtol=2e-5 if dtype==torch.float32 else .008,
                               atol=2e-6 if dtype==torch.float32 else .004)
