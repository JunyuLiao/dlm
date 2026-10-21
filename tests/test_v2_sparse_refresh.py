"""H100 operator qualification for the named sparse-refresh variants."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='H100 required')


@pytest.mark.parametrize('n,nk,tau,sticky', [(256, 449, .56, 0.), (137, 641, .3, 1.386), (256, 512, 0., 0.), (256, 1025, 2., 1.386)])
@pytest.mark.parametrize('ratio', [False, True])
def test_fused_scan_matches_independent_ordered_oracle(n, nk, tau, sticky, ratio):
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse.v31_value_selectors import statistics_reference, mandatory_map, online_reference
    g = torch.Generator(device='cuda').manual_seed(nk)
    q = torch.randn((1, 4, n, 512), generator=g, device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, 2, nk, 512), generator=g, device='cuda', dtype=torch.bfloat16)
    z = torch.randn((1, 2, nk, 32), generator=g, device='cuda')
    nu = torch.full((1, 2), 2., device='cuda')
    held = torch.rand((4*((n+127)//128), (nk+63)//64), generator=g, device='cuda') > .5
    pt = max(0, (nk-256)//64)
    keep, state, lm, risk = discover(q, k, z, nu, 512**-.5, pt, tau,
        held=held, sticky=sticky, sink=1, recent=1, trace=True, ratio=ratio)
    # CPU FP32 matmul and independent Python recurrence; no Triton summary oracle.
    scores = q[0].float().cpu() @ k[0].float().cpu().repeat_interleave(2, 0).transpose(-1, -2)
    stats = statistics_reference(scores*512**-.5, z[0].cpu(), nu[0].cpu(), pt)
    reference = online_reference(stats, tau, True, mandatory_map(stats, 1, 1), held.cpu(), sticky)
    assert torch.equal(keep.cpu(), reference[0])
    torch.testing.assert_close(state.cpu(), reference[1], atol=4.e-5, rtol=4.e-5)
    torch.testing.assert_close(lm.cpu(), reference[2], atol=5.e-6, rtol=4.e-6)
    torch.testing.assert_close(risk.cpu(), reference[3], atol=5.e-5, rtol=4.e-5)


def test_sparse_refresh_is_current_masked_full_value_attention():
    from experiments.numerical_qk_reuse.v2_sparse_refresh import refresh
    g = torch.Generator(device='cuda').manual_seed(1729)
    q = torch.randn((1, 4, 256, 512), generator=g, device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, 2, 513, 512), generator=g, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(k.shape, generator=g, device='cuda', dtype=torch.bfloat16)
    matrix = torch.randn((2, 512, 32), generator=g, device='cuda')/32**.5
    out, keep, _ = refresh(q, k, v, matrix, 512**-.5, 4, .56)
    key_keep = keep.repeat_interleave(128, 2)[:, :, :256].repeat_interleave(64, -1)[..., :513]
    scores = q.float() @ k.float().repeat_interleave(2, 1).transpose(-1, -2)*512**-.5
    expected = scores.masked_fill(~key_keep, -torch.inf).softmax(-1) @ v.float().repeat_interleave(2, 1)
    torch.testing.assert_close(out.float(), expected.transpose(1, 2), atol=.004, rtol=.015)
    changed, changed_keep, _ = refresh(q, k, -v, matrix, 512**-.5, 4, .56)
    assert torch.equal(keep, changed_keep)
    assert torch.equal(changed, -out)


@pytest.mark.parametrize('warps,n,nk,tau',[(4,137,641,0.),(8,256,449,.3),(8,256,1025,.08)])
def test_fused_full_pv_operator(warps,n,nk,tau):
    from experiments.numerical_qk_reuse.v2_sparse_refresh import attention,discover
    g=torch.Generator(device='cuda').manual_seed(2718)
    q=torch.randn((1,4,n,512),generator=g,device='cuda',dtype=torch.bfloat16)
    k=torch.randn((1,2,nk,512),generator=g,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(k.shape,generator=g,device='cuda',dtype=torch.bfloat16)
    z=torch.randn((1,2,nk,32),generator=g,device='cuda')
    nu=torch.full((1,2),2.,device='cuda')
    pt=(nk-256)//64
    out,keep=attention(q,k,v,z,nu,512**-.5,pt,tau,warps=warps)
    reference=discover(q,k,z,nu,512**-.5,pt,tau,ratio=True)[0]
    assert torch.equal(keep.reshape_as(reference),reference)
    mask=keep.repeat_interleave(128,2)[:,:,:n].repeat_interleave(64,-1)[...,:nk]
    scores=q.float()@k.float().repeat_interleave(2,1).transpose(-1,-2)*512**-.5
    expected=scores.masked_fill(~mask,-torch.inf).softmax(-1)@v.float().repeat_interleave(2,1)
    torch.testing.assert_close(out.float(),expected.transpose(1,2),atol=.004,rtol=.015)


def test_split_d_preserves_q128_decisions_and_state():
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    g=torch.Generator(device='cuda').manual_seed(42)
    q=torch.randn((1,4,256,512),generator=g,device='cuda',dtype=torch.bfloat16)
    k=torch.randn((1,2,1025,512),generator=g,device='cuda',dtype=torch.bfloat16)
    z=torch.randn((1,2,1025,32),generator=g,device='cuda')
    nu=torch.full((1,2),2.,device='cuda')
    a=discover(q,k,z,nu,512**-.5,12,.2,ratio=True,trace=True)
    b=discover(q,k,z,nu,512**-.5,12,.2,ratio=True,trace=True,split_d=True,warps=4)
    assert torch.equal(a[0],b[0])
    for x,y in zip(a[1:],b[1:]):torch.testing.assert_close(x,y,atol=3.e-5,rtol=3.e-5)


def test_parallel_normalized_scan_matches_sequential_logmass():
    from experiments.numerical_qk_reuse.v2_sparse_refresh import parallel_discover
    g=torch.Generator(device='cuda').manual_seed(71)
    q=torch.randn((1,4,137,512),generator=g,device='cuda',dtype=torch.bfloat16)
    k=torch.randn((1,2,1025,512),generator=g,device='cuda',dtype=torch.bfloat16)
    z=torch.randn((1,2,1025,32),generator=g,device='cuda')
    nu=torch.full((1,2),2.,device='cuda')
    held=torch.rand((8,17),generator=g,device='cuda')>.5
    for tau in [0.,.1,.3,2.]:
        args=dict(held=held,sticky=1.386,sink=1,recent=1)
        a=parallel_discover(q,k,z,nu,512**-.5,12,tau,**args)[0]
        b=parallel_discover(q,k,z,nu,512**-.5,12,tau,fast_scan=True,**args)[0]
        assert torch.equal(a,b)
