"""Numerical/current-input qualification for the matched all-kept control."""
from types import SimpleNamespace
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('nk,n',[(641,137),(8192,256)])
def test_resident_allkept_current_kv_and_canvas_metadata(nk,n):
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.resident_execution_runtime import install_allkept
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    g=torch.Generator(device='cuda').manual_seed(nk)
    h,hk,d=4,2,512;pages=(nk+127)//128
    cache=torch.randn((pages,128,hk,2*d),device='cuda',dtype=torch.bfloat16,generator=g)
    table=torch.randperm(pages,device='cuda',generator=g).int()[None]
    adapter=VllmMethodAdapter(['full_attention']*30,arm='allkept',value_audit=True)
    state=install_allkept(adapter);adapter.begin_request()
    impl=SimpleNamespace(head_size=d,scale=d**-.5)
    metadata=SimpleNamespace(num_actual_tokens=n,block_table=table)
    output=torch.empty((n,h*d),device='cuda',dtype=torch.bfloat16)
    for iteration in range(3):
        cache.copy_(torch.randn(cache.shape,device='cuda',dtype=cache.dtype,generator=g))
        if iteration==2:
            adapter.canvas_id+=1
            table.copy_(torch.randperm(pages,device='cuda',generator=g).int()[None])
        query=torch.randn((n,h,d),device='cuda',dtype=cache.dtype,generator=g)
        adapter.step_ctx=dict(seq_len=nk,n=n);adapter._canvas_step=iteration
        native_cache=cache.transpose(1,2)
        adapter.forward(impl,5,query,native_cache,metadata,output)
        key,value=cache[...,:d],cache[...,d:]
        k=key[table[0].long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous()
        v=value[table[0].long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous()
        q=query.transpose(0,1)[None]
        ref=v27_fa4.dense_plain(q,k,v,d**-.5)
        assert bool(torch.isfinite(output).all())
        assert float((output.reshape_as(ref).float()-ref.float()).norm()/ref.float().norm())<.004
    assert state['metadata_builds']==2
    assert state['calls']==3
    receipt=adapter._value_receipt()
    assert receipt['resident_allkept_kv_copy_calls']==0
    assert receipt['resident_allkept_extra_forwards']==0
    assert receipt['global_kept_tiles']==receipt['global_eligible_tiles']
    assert receipt['global_eligible_tiles']==3*h*((n+127)//128)*((nk+63)//64)
    phases=adapter._value_phase_receipt()['value_phase_tiles']
    assert phases['dense']['kept_tiles']==phases['dense']['eligible_tiles']
    assert phases['dense']['calls']==3
