import os
from pathlib import Path
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('method',['v2','v31'])
def test_native_refresh_current_qkv_lists_and_accounting(method,tmp_path,monkeypatch):
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    from experiments.numerical_qk_reuse.optimized_refresh_runtime import install
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse.v31_value_projection import refresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',v27_fa4.block_sparse_tensors)
    lib=Path(os.environ['V2_WARPSPEC_LIBRARY'])
    adapter=VllmMethodAdapter(['full_attention']*30,arm='mage',mage_select='fa4',mage_granularity='qblock_max',mage_k=128,
        value_selector='value_v2_online_preserve_mass' if method=='v2' else None,value_threshold=.08 if method=='v2' else None,value_audit=True)
    install(adapter,method=method,library=lib,observer_directory=tmp_path)
    g=torch.Generator(device='cuda').manual_seed(44)
    qnative=torch.randn((137,4,512),device='cuda',dtype=torch.bfloat16,generator=g);q=qnative.transpose(0,1).unsqueeze(0)
    k=torch.randn((1,2,641,512),device='cuda',dtype=torch.bfloat16,generator=g);v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=g)
    adapter.begin_request();adapter._mage_held=torch.rand((1,4,2,11),device='cuda',generator=g)>.5;adapter.mage_sticky=1.386
    outputs=[]
    for _ in range(2):
        out,keep=adapter._select_initial_refresh(5,q,dict(k=k,v=v),512**-.5,504,137)
        lists=v27_fa4.block_sparse_tensors(keep)
        ref=v27_fa4.sparse_lists(q,k,v,lists,512**-.5) if method=='v2' else v27_fa4.dense_plain(q,k,v,512**-.5)
        assert float((out.float()-ref.float()).norm()/ref.float().norm())<.01
        if method=='v2':
            z=torch.empty((1,2,641,32),device='cuda');norm=torch.empty((1,2,641),device='cuda');valid=torch.ones_like(norm,dtype=torch.bool)
            matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device);nu=refresh(v,matrix,z,norm,valid,0)
            mask=discover(q.contiguous(),k,z,nu,512**-.5,7,.08,held=adapter._mage_held,sticky=1.386,ratio=True)[0]
            assert torch.equal(keep.reshape_as(mask),mask)
        else:assert bool((keep[...,:7].sum(-1)==2).all())
        outputs.append(out.clone());v.mul_(-1)
    assert not torch.equal(*outputs)
    receipt=adapter._value_receipt()
    assert receipt['optimized_refresh_calls']==2 and receipt['optimized_refresh_extra_model_forwards']==0
    assert receipt['optimized_refresh_gpu_ms']>0
    assert receipt['optimized_refresh_budget_errors']==0
    if method=='v2':assert adapter._sparse_total==2*4*2*7
