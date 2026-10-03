"""H100 kernel acceptance checks; set VD_KERNEL to an explicitly built library."""
import math
import os

import pytest
import torch

from experiments.value_direction_hopper.cuda import Kernel
from experiments.value_direction_hopper.masks import pack
from experiments.value_direction_hopper.validate import inputs,trusted


@pytest.fixture(scope='module')
def kernel():
    library=os.getenv('VD_KERNEL')
    if not library or not torch.cuda.is_available():pytest.skip('VD_KERNEL and H100 required')
    return Kernel(library)


@pytest.mark.parametrize('width',(256,512))
@pytest.mark.parametrize('precision',('ieee','tf32x3','tf32x3_register','tf32x3_shared'))
@pytest.mark.parametrize('threshold',(-math.inf,math.log(.15)))
def test_reference_and_partial_gqa(kernel,width,precision,threshold):
    q,k,v,z,r,m=inputs(129,193,width,4,2,local=True)
    expected=trusted(q,k,v,z,r,m,threshold)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result=kernel(q,k,v,z,r,mask=pack(m),log_threshold=threshold,trace=True,precision=precision)
    torch.cuda.synchronize()
    assert torch.equal(result.skipped,expected['skipped'])
    torch.testing.assert_close(result.log_normalizer,expected['log_normalizer'],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(result.projected_state,expected['projected_state'],atol=2e-5,rtol=2e-5)
    assert (result.output.float()-expected['output'].float()).norm()/expected['output'].float().norm()<.008
    assert torch.isfinite(result.output).all()


@pytest.mark.parametrize('precision',('ieee','tf32x3_register'))
def test_first_support_ties_and_empty_rows(kernel,precision):
    q,k,v,z,r,m=inputs(129,193,256,4,2)
    q.zero_();k.zero_();z.zero_()
    m[...,0,:]=False
    m[...,64:128,:64]=False
    for threshold in (-math.inf,math.log(.1)):
        result=kernel(q,k,v,z,r,mask=pack(m),log_threshold=threshold,trace=True,precision=precision)
        expected=trusted(q,k,v,z,r,m,threshold)
        assert torch.equal(result.skipped,expected['skipped'])
        assert not result.skipped[...,0,0].any()
        assert not result.skipped[...,0,1].any()  # first support for another valid row
        assert (result.output[...,0,:]==0).all()
        assert torch.isneginf(result.log_normalizer[...,0]).all()
        if threshold==-math.inf: assert not result.skipped.any()  # exact zero ties retained
        else: assert result.skipped[...,0,2:].all()


def test_dense_graph_and_invalid_contract(kernel):
    q,k,v,z,r,m=inputs(63,65,256,4,2)
    call=lambda:kernel(q,k,v,z,r,mask=m,mode='dense',trace=True)
    for _ in range(3):call()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result=call()
    graph.replay();torch.cuda.synchronize()
    expected=trusted(q,k,v,z,r,m,-math.inf)
    assert not result.skipped.any()
    torch.testing.assert_close(result.log_normalizer,expected['log_normalizer'],atol=2e-5,rtol=2e-5)
    with pytest.raises(ValueError):kernel(q,k,v,z.half(),r)
    with pytest.raises(ValueError):kernel(q[...,::2],k,v,z,r)


def test_packing_tail_bits(kernel):
    _,_,_,_,_,mask=inputs(63,65,256,4,2)
    mask[...,1,64]=False
    packed=pack(mask).tensor.cpu().numpy()
    assert int(packed[0,0,0,0])==2**64-1
    assert int(packed[0,0,0,1])==1
    assert int(packed[0,0,1,1])==0


@pytest.mark.parametrize('log_lambda',(-2.,0.,1.))
def test_original_and_aggressive_blasst_convention(kernel,log_lambda):
    from experiments.diffusion_gemma_value_aware_gpu.kernels import block_statistics,route
    from experiments.diffusion_gemma_value_aware.operators import Config
    q,k,v,z,r,m=inputs(129,257,256,4,2)
    scores=((q@k.repeat_interleave(2,1).transpose(-2,-1))*256**-.5).masked_fill(~m,-torch.inf)
    st=block_statistics(scores,m)
    mask,_=route(st,Config(method='blasst'),{},log_lambda)
    actual=kernel(q,k,v,z,r,mask=pack(m),mode='blasst',log_threshold=log_lambda)
    assert torch.equal(actual.skipped,mask.unsqueeze(0))


@pytest.mark.parametrize('causal,window',[(False,0),(True,0),(False,128),(True,128)])
def test_compact_structural_geometry(kernel,causal,window):
    from experiments.value_direction_hopper.masks import geometry
    q,k,_,_,_,_=inputs(129,257,256,4,2)
    from dllm.attention.blasst.core import _attention_validity
    expected=_attention_validity(None,q[:,:1],k,is_causal=causal,sliding_window=window or None)
    actual,valid=geometry(1,129,257,device=q.device,causal=causal,window=window)
    assert torch.equal(actual.tensor,pack(expected).tensor)
    assert torch.equal(valid,expected.any((0,1,2)))


def test_producer_cache_invalidation_and_canvas_refresh(kernel):
    from experiments.value_direction_hopper.integration import Sketches
    from types import SimpleNamespace
    encoder=type('DiffusionGemmaEncoderModel',(torch.nn.Module,),{})()
    model=torch.nn.Module();model.encoder=encoder
    adapter=SimpleNamespace(model=model,is_blasst_attention_module=lambda name,module:False)
    cache=Sketches(adapter)
    try:
        v=torch.randn((1,2,193,256),device='cuda',dtype=torch.bfloat16)
        source=v[...,:129,:].clone();cache.sources[0]=source
        valid=torch.ones((1,2,193),device='cuda',dtype=torch.bool)
        z,_=cache.get(0,v,valid,129);first=z.clone()
        v[...,129:,:].add_(.1)
        changed,_=cache.get(0,v,valid,129)
        assert cache.reused_tokens==1*2*128
        torch.testing.assert_close(changed[...,:128,:],first[...,:128,:],rtol=0,atol=0)
        assert not torch.equal(changed[...,129:,:],first[...,129:,:])
        source.add_(.25);v[...,:129,:]=source
        z,_=cache.get(0,v,valid,129)
        matrix=cache.projections.get(0,2,256,'gaussian',32,1729,v.device)
        torch.testing.assert_close(z,v.float()@matrix,rtol=0,atol=0)
        before=cache.projected_tokens;cache.invalidate();cache.sources[0]=source
        cache.get(0,v,valid,129)
        assert cache.projected_tokens-before==1*2*193
    finally:cache.close()


@pytest.mark.parametrize('schedule',('tma','split'))
def test_async_schedules(kernel,schedule):
    q,k,v,z,r,m=inputs(129,257,512,4,2,local=True)
    expected=trusted(q,k,v,z,r,m,math.log(.15))
    a=kernel(q,k,v,z,r,mask=pack(m),log_threshold=math.log(.15),precision='tf32x3_register',trace=True,
             tma=schedule=='tma',split_pv=schedule=='split')
    assert torch.equal(a.skipped,expected['skipped'])
    torch.testing.assert_close(a.projected_state,expected['projected_state'],rtol=2e-5,atol=2e-5)
    assert (a.output.float()-expected['output'].float()).norm()/expected['output'].float().norm()<.008


@pytest.mark.parametrize('width',(256,512))
def test_tma_batch_head_masks_and_cuda_packing(kernel,width):
    q,k,v,z,r,m=inputs(131,259,width,4,2,local=True)
    q,k,v,z,r=(torch.cat((x,x*.7),0).contiguous() for x in (q,k,v,z,r))
    m=m.expand(2,4,131,259).clone();m[1,1,65:,:64]=False;m[0,2,0,:]=False
    packed=kernel.pack_mask(m)
    assert torch.equal(packed.tensor,pack(m).tensor)
    expected=trusted(q,k,v,z,r,m,math.log(.15))
    scores=torch.empty_like(m,dtype=torch.float32)
    actual=kernel(q,k,v,z,r,mask=packed,log_threshold=math.log(.15),trace=True,precision='tf32x3_register',tma=True,debug_scores=scores)
    assert torch.equal(actual.skipped,expected['skipped'])
    # TMA indexing/state arithmetic must match on identical logits. Independent
    # BF16 dot-product reductions need not round identically (native QK audit).
    replay=trusted(q,k,v,z,r,m,math.log(.15),scores=scores)
    torch.testing.assert_close(actual.projected_state,replay['projected_state'],rtol=2e-5,atol=2e-5)
    assert torch.isfinite(actual.output).all()


def test_additive_bias_is_not_discarded(kernel):
    q,k,v,z,r,m=inputs(65,131,256,4,2)
    bias=torch.linspace(-2,2,131,device='cuda',dtype=torch.bfloat16).expand(1,4,65,131).clone()
    bias[...,0,:]=-torch.inf;mask=torch.isfinite(bias)
    expected=trusted(q,k,v,z,r,mask,math.log(.15),bias=bias)
    actual=kernel(q,k,v,z,r,mask=bias,log_threshold=math.log(.15),trace=True,precision='tf32x3_register')
    assert torch.equal(actual.skipped,expected['skipped'])
    torch.testing.assert_close(actual.projected_state,expected['projected_state'],rtol=2e-5,atol=2e-5)
    assert (actual.output[...,0,:]==0).all()


def test_identity_subspace_redundancy_and_cancellation(kernel):
    q,k,v,z,r,m=inputs(128,256,256,4,2)
    q.zero_();k.zero_();v.zero_()
    v[...,:128,0]=1.  # First block and redundant second block.
    v[...,128:192:2,0]=1.;v[...,129:192:2,0]=-1.  # Internally cancelling.
    v[...,192:,0]=-1.  # Distinctive direction.
    z=v[...,:32].float().contiguous();r=v.float().square().sum(-1).mean(-1).sqrt()
    expected=trusted(q,k,v,z,r,m,math.log(.1))
    actual=kernel(q,k,v,z,r,mask=pack(m),log_threshold=math.log(.1),trace=True,precision='tf32x3_register',tma=True)
    assert torch.equal(actual.skipped,expected['skipped'])
    assert actual.skipped[...,1].all()
    assert not actual.skipped[...,2].any()  # Cancellation changes normalization.
    assert not actual.skipped[...,3].any()
    torch.testing.assert_close(actual.projected_state,expected['projected_state'],rtol=2e-5,atol=2e-5)


def test_c_abi_rejects_missing_tma_pointers(kernel):
    import ctypes as ct
    from experiments.value_direction_hopper.cuda import Params
    p=Params();p.batch=1;p.heads=4;p.kv_heads=2;p.queries=128;p.keys=128;p.width=256
    p.mode=1;p.mask_kind=3;p.mask_heads=1;p.projected_precision=2;p.scale=1.
    assert kernel.fn(ct.byref(p),0,1)!=0
    assert kernel.tma_fn(ct.byref(p),0,1)!=0


@pytest.mark.parametrize('width',(256,512))
def test_fused_projection_refresh_and_prefix_preservation(kernel,width):
    from experiments.value_direction_hopper.projection import refresh
    q,k,v,_,_,m=inputs(129,259,width,4,2)
    matrix=torch.randn((2,width,32),device='cuda',dtype=torch.float32)/math.sqrt(32)
    valid=m.any(-2).expand(1,2,259);z=torch.empty((1,2,259,32),device='cuda');norm=torch.empty((1,2,259),device='cuda')
    ref=refresh(v,matrix,z,norm,valid,0);expected=v.float()@matrix
    torch.testing.assert_close(z,expected,rtol=2e-5,atol=2e-5)
    original=z[...,:128,:].clone();v[...,128:,:].add_(.5)
    ref=refresh(v,matrix,z,norm,valid,128)
    assert torch.equal(z[...,:128,:],original)
    torch.testing.assert_close(z,v.float()@matrix,rtol=2e-5,atol=2e-5)
    expected_ref=v.float().norm(dim=-1).square().mean(-1).sqrt()
    torch.testing.assert_close(ref,expected_ref,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('width',(256,512))
def test_aten_bridge_exact_stream_and_graph_parity(kernel,width):
    bridge=os.getenv('VD_TORCH_KERNEL')
    if not bridge:pytest.skip('Explicit ATen bridge required')
    fast=Kernel(kernel.path,torch_library=bridge)
    q,k,v,z,r,m=inputs(129,257,width,4,2,local=True);mask=pack(m)
    for mode in ('value','blasst','dense'):
        options=dict(mask=mask,mode=mode,trace=True,log_threshold=math.log(.15),precision='tf32x3_register',tma=mode=='value')
        expected=kernel(q,k,v,z,r,**options)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):actual=fast(q,k,v,z,r,**options)
        torch.cuda.synchronize()
        for field in ('output','skipped','eligible','log_normalizer','projected_state','risk'):
            torch.testing.assert_close(getattr(actual,field),getattr(expected,field),rtol=0,atol=0,equal_nan=True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=fast(q,k,v,z,r,**options)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(actual.output,expected.output,rtol=0,atol=0)


@pytest.mark.parametrize('width',(256,512))
def test_tma_blasst_load_schedule_matches_cp(kernel,width):
    if not os.getenv('VD_TMA_CONTROLS'):pytest.skip('Optional new TMA controls must be explicitly enabled')
    q,k,v,z,r,m=inputs(129,259,width,4,2,local=True);mask=pack(m)
    for mode in ('blasst','dense'):
        for threshold in (-2.,0.,1.):
            options=dict(mask=mask,mode=mode,log_threshold=threshold,precision='tf32x3_register',trace=True)
            a=kernel(q,k,v,z,r,**options);b=kernel(q,k,v,z,r,tma=True,**options)
            for field in ('output','skipped','eligible','log_normalizer','projected_state','risk'):
                torch.testing.assert_close(getattr(a,field),getattr(b,field),rtol=0,atol=0,equal_nan=True)
