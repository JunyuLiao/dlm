"""H100 qualification of packed-sketch Q128 scheduling and changing masks."""
import os
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')


@pytest.mark.parametrize('nk,n',[(449,137),(1025,256),(4096,256)])
def test_compact_lists_preserve_fa4_output(nk,n):
    from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
    from experiments.numerical_qk_reuse import v27_fa4
    g=torch.Generator(device='cuda').manual_seed(nk)
    q=torch.randn((1,4,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g)
    v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
    keep=torch.rand((1,4,(n+127)//128,(nk+63)//64),device='cuda',generator=g)>.5
    keep[...,0]=True
    fast=pack_lists(keep);old=v27_fa4.block_sparse_tensors(keep)
    assert torch.equal(fast.full_block_cnt,old.full_block_cnt)
    for u in range(keep.numel()//keep.shape[-1]):
        cnt=int(fast.full_block_cnt.flatten()[u]);got=fast.full_block_idx.reshape(-1,keep.shape[-1])[u]
        assert torch.equal(got[:cnt],torch.where(keep.reshape(-1,keep.shape[-1])[u])[0].int())
        assert bool((got[cnt:]==0).all())
    assert torch.equal(v27_fa4.sparse_lists(q,k,v,fast,512**-.5),v27_fa4.sparse_lists(q,k,v,old,512**-.5))


def _kernel():
    path=os.environ.get('V2_WARPSPEC_LIBRARY')
    if not path:pytest.skip('explicit qualified binary required')
    from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel
    from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
    import json
    packed=json.loads((Path(path).parent/'build_receipt.json').read_text())['options'].get('packed')
    return PackedKernel(Path(path)) if packed else Kernel(Path(path))


@pytest.mark.parametrize('nk,n',[(641,137),(1025,256),(4096,256)])
@pytest.mark.parametrize('schedule',[0,2,3])
def test_online_graph_replay_with_changing_support(nk,n,schedule):
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
    kernel=_kernel()
    g=torch.Generator(device='cuda').manual_seed(43)
    q=torch.randn((1,4,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g)
    v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
    z=torch.randn((1,2,nk,32),device='cuda',generator=g);nu=torch.full((1,2),2.,device='cuda')
    held=torch.zeros((1,4,(n+127)//128,(nk+63)//64),device='cuda',dtype=torch.bool)
    pt=(nk-n)//64;scale=512**-.5;tau=.08
    def run():return kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,held=held,sticky=1.386,schedule=schedule)
    for _ in range(3):run()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out,keep,*_=run()
    previous=None;transitions=0
    for iteration in range(20):
        q.copy_(torch.randn(q.shape,device='cuda',dtype=q.dtype,generator=g))
        z.copy_(torch.randn(z.shape,device='cuda',generator=g))
        # Alternate a near-zero sketch with ordinary scale to force decisions
        # to change even at very short prefixes with mandatory support.
        if iteration%2==0:z.mul_(.001)
        held.copy_(torch.rand(held.shape,device='cuda',generator=g)>.5)
        expected=discover(q,k,z,nu,scale,pt,tau,held=held,sticky=1.386,ratio=True)[0].reshape_as(keep)
        graph.replay()
        assert torch.equal(keep,expected)
        if previous is not None:transitions+=int((previous!=keep).sum())
        previous=keep.clone()
        reference=v27_fa4.sparse_lists(q,k,v,pack_lists(keep),scale)
        assert float((out.float()-reference.float()).norm()/reference.float().norm())<.01
    assert transitions>0


def test_online_extreme_logits_preserve_ordered_selector():
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
    kernel=_kernel();g=torch.Generator(device='cuda').manual_seed(44)
    q=torch.zeros((1,4,137,512),device='cuda',dtype=torch.bfloat16);q[...,0]=1
    k=torch.zeros((1,2,641,512),device='cuda',dtype=torch.bfloat16);k[...,0]=-32768;k[:,:,192:256,0]=32768
    v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
    z=torch.randn((1,2,641,32),device='cuda',generator=g);nu=torch.full((1,2),2.,device='cuda')
    held=torch.rand((1,4,2,11),device='cuda',generator=g)>.5;pt=(641-137)//64
    expected=discover(q,k,z,nu,512**-.5,pt,.08,held=held,sticky=1.386,ratio=True)[0]
    for schedule in [0,2,3]:
        out,keep,*_=kernel(q,k,v,z,nu,512**-.5,.08,prefix_tiles=pt,held=held,sticky=1.386,schedule=schedule)
        assert torch.equal(keep.reshape_as(expected),expected)
        reference=v27_fa4.sparse_lists(q,k,v,pack_lists(keep),512**-.5)
        assert bool(torch.isfinite(out).all())
        assert float((out.float()-reference.float()).norm()/reference.float().norm())<.01


@pytest.mark.parametrize('schedule',[2,3])
def test_complete_refresh_entry_point_uses_current_values(schedule):
    from experiments.numerical_qk_reuse.v2_online_pipeline import OnlineRefresh
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from experiments.numerical_qk_reuse import v27_fa4
    kernel=_kernel()
    if not kernel.options.get('tma_all'):pytest.skip('complete entry point requires all-TMA binary')
    from experiments.numerical_qk_reuse.v2_online_warpspec import PackedRefresh
    op=PackedRefresh(kernel.path,schedule=schedule)
    g=torch.Generator(device='cuda').manual_seed(45)
    q=torch.randn((1,4,256,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,2,1024,512),device='cuda',dtype=torch.bfloat16,generator=g)
    v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
    matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device)
    for _ in range(2):
        v.copy_(torch.randn(v.shape,device='cuda',dtype=v.dtype,generator=g))
        out,keep,lists=op(q,k,v,matrix,512**-.5,12,.114013671875)
        z=torch.empty((1,2,1024,32),device='cuda');norm=torch.empty((1,2,1024),device='cuda')
        nu=project(v,matrix,z,norm,torch.ones_like(norm,dtype=torch.bool),0)
        expected=discover(q,k,z,nu,512**-.5,12,.114013671875,ratio=True)[0]
        assert torch.equal(keep.reshape_as(expected),expected)
        reference=v27_fa4.sparse_lists(q,k,v,lists,512**-.5)
        assert float((out.float()-reference.float()).norm()/reference.float().norm())<.01


def test_packed_kernel_rejects_malformed_support():
    kernel=_kernel();q=torch.zeros((1,4,128,512),device='cuda',dtype=torch.bfloat16)
    k=torch.zeros((1,2,256,512),device='cuda',dtype=torch.bfloat16);v=torch.zeros_like(k)
    z=torch.zeros((1,2,256,32),device='cuda');nu=torch.ones((1,2),device='cuda')
    held=torch.zeros((1,4,1,3),device='cuda',dtype=torch.bool)
    with pytest.raises(ValueError,match='held map'):
        kernel(q,k,v,z,nu,512**-.5,.1,prefix_tiles=0,held=held)
