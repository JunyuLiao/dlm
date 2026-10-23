import os
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('nk,n,page',[(641,137,64),(1025,256,128),(8192,256,128)])
@pytest.mark.parametrize('schedule',[2,3])
def test_paged_refresh_exact_current_values_and_graph(nk,n,page,schedule):
    from experiments.numerical_qk_reuse.v2_paged_refresh import PagedRefresh
    from experiments.numerical_qk_reuse.v2_online_warpspec import FusedPackedRefresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    library=os.environ.get('V2_PAGED_LIBRARY')
    if not library:pytest.skip('explicit paged candidate required')
    previous=Path('results/sparse_execution_20261021/attempt002/build/200354cb919fb6b8/v2_online_sm90.so')
    op=PagedRefresh(Path(library),schedule);reference=FusedPackedRefresh(previous,schedule)
    g=torch.Generator(device='cuda').manual_seed(nk);pages=(nk+page-1)//page
    q=torch.randn((1,4,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,2,nk,512),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
    cache=torch.empty((pages,page,2,1024),device='cuda',dtype=q.dtype);key,value=cache[...,:512],cache[...,512:]
    table=torch.randperm(pages,device='cuda',generator=g).int()
    def scatter():
        pk=torch.nn.functional.pad(k[0].transpose(0,1),(0,0,0,0,0,pages*page-nk))
        pv=torch.nn.functional.pad(v[0].transpose(0,1),(0,0,0,0,0,pages*page-nk))
        key[table.long()]=pk.reshape_as(key);value[table.long()]=pv.reshape_as(value)
    scatter();matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device);prefix=nk-n
    held=torch.rand((1,4,(n+127)//128,(nk+63)//64),device=q.device,generator=g)>.5
    def run():return op(q,key,value,table,nk,matrix,512**-.5,prefix,.08,held=held,sticky=1.386,cache_key=5,canvas=0)
    for _ in range(3):run()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out,keep,_=run()
    for step in range(5):
        q.copy_(torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=g))
        k.copy_(torch.randn(k.shape,device=k.device,dtype=k.dtype,generator=g))
        v[:,:,prefix:].copy_(torch.randn(v[:,:,prefix:].shape,device=v.device,dtype=v.dtype,generator=g))
        held.logical_not_();scatter();graph.replay()
        ref,mask,_=reference(q,k,v,matrix,512**-.5,prefix//64,.08,held=held,sticky=1.386)
        assert torch.equal(keep,mask)
        assert torch.equal(out,ref), float((out.float()-ref.float()).abs().max())
    op.clear();v.neg_();scatter();actual,mask,_=run()
    ref,expected,_=reference(q,k,v,matrix,512**-.5,prefix//64,.08,held=held,sticky=1.386)
    assert torch.equal(actual,ref) and torch.equal(mask,expected)
