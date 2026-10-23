import os
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

def test_cached_native_refresh_resets_and_uses_current_canvas(monkeypatch,tmp_path):
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    from experiments.numerical_qk_reuse.optimized_refresh_runtime import install
    from experiments.numerical_qk_reuse.v2_online_warpspec import FusedPackedRefresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',v27_fa4.block_sparse_tensors)
    monkeypatch.setenv('V2_CANVAS_PROJECTION_REUSE','1')
    monkeypatch.delenv('V2_PAGED_REFRESH_LIBRARY',raising=False)
    library=Path(os.environ['V2_WARPSPEC_LIBRARY'])
    adapter=VllmMethodAdapter(['full_attention']*30,arm='mage',mage_select='fa4',mage_granularity='qblock_max',mage_k=128,
        value_selector='value_v2_online_preserve_mass',value_threshold=.08,value_audit=True)
    install(adapter,method='v2',library=library,observer_directory=tmp_path)
    old=FusedPackedRefresh(Path('results/sparse_execution_20261021/attempt002/build/200354cb919fb6b8/v2_online_sm90.so'),2)
    g=torch.Generator(device='cuda').manual_seed(45)
    q=torch.randn((1,4,256,512),device='cuda',dtype=torch.bfloat16,generator=g)
    k=torch.randn((1,2,8192,512),device=q.device,dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
    matrix=Projections().get(5,2,512,'gaussian',32,1729,q.device)
    for request in range(2):
        adapter.begin_request();adapter._mage_held=None;v.neg_()
        for step in range(3):
            v[:,:,7936:].copy_(torch.randn(v[:,:,7936:].shape,device=v.device,dtype=v.dtype,generator=g))
            actual,keep=adapter._select_initial_refresh(5,q,dict(k=k,v=v),512**-.5,7936,256)
            expected,mask,_=old(q,k,v,matrix,512**-.5,124,.08)
            assert torch.equal(actual,expected) and torch.equal(keep,mask)
        receipt=adapter._value_receipt()
        assert receipt['optimized_refresh_projection_full_calls']==1
        assert receipt['optimized_refresh_projection_canvas_calls']==2
        assert receipt['optimized_refresh_reused_projection_rows']==2*2*7936
