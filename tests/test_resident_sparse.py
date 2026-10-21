import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('splits',[1,2,4])
@pytest.mark.parametrize('nk,n',[(641,137),(4096,256)])
def test_resident_current_paged_kv_graph_and_mask_changes(splits,nk,n):
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.resident_sparse import ResidentMask,PagedContext,_pack
    from experiments.numerical_qk_reuse.v31_local_sparse import alias64
    import triton as tr
    g=torch.Generator(device='cuda').manual_seed(nk);h=4;hk=2;d=512;pages=(nk+127)//128;kt=(nk+63)//64
    cache=torch.randn((pages,128,hk,2*d),device='cuda',dtype=torch.bfloat16,generator=g);key,value=cache[...,:d],cache[...,d:]
    table=torch.randperm(pages,device='cuda',generator=g).int();q=torch.randn((1,h,n,d),device='cuda',dtype=cache.dtype,generator=g)
    keep=torch.rand((1,h,(n+127)//128,kt),device='cuda',generator=g)>.5;keep[...,0]=True;mask=ResidentMask.create(keep,splits=splits)
    ctx=PagedContext(key,value,table,nk,splits=splits)
    native_out=torch.empty((1,n,h,d),device=q.device,dtype=q.dtype)
    for _ in range(2):ctx(q,mask,d**-.5,out=native_out)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=ctx(q,mask,d**-.5,out=native_out)
    assert out.data_ptr()==native_out.data_ptr()
    ptrs=(ctx.page_table.data_ptr(),ctx.used.data_ptr(),mask.lists.full_block_cnt.data_ptr(),mask.lists.full_block_idx.data_ptr(),ctx.partial.data_ptr())
    for i in range(10):
        cache.copy_(torch.randn(cache.shape,device='cuda',dtype=cache.dtype,generator=g));q.copy_(torch.randn(q.shape,device='cuda',dtype=q.dtype,generator=g))
        table.copy_(torch.randperm(pages,device='cuda',generator=g).int());ctx.update_table()
        keep.copy_(torch.rand(keep.shape,device='cuda',generator=g)>.5);keep[...,0]=True
        _pack[(h*((n+127)//128),splits)](keep,mask.lists.full_block_cnt,mask.lists.full_block_idx,kt,splits,tr.next_power_of_2(kt),num_warps=4)
        graph.replay()
        pk=key[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous();pv=value[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous()
        ref=v27_fa4.sparse_lists(q,pk,pv,v27_fa4.block_sparse_tensors(keep),d**-.5)
        assert bool(torch.isfinite(out).all())
        assert float((out.float()-ref.float()).norm()/ref.float().norm())<.004
        assert ptrs==(ctx.page_table.data_ptr(),ctx.used.data_ptr(),mask.lists.full_block_cnt.data_ptr(),mask.lists.full_block_idx.data_ptr(),ctx.partial.data_ptr())
