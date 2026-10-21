import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('nk',[449,1024,4096,8192])
def test_fused_projection_and_pack_are_original_operands(nk):
    from experiments.numerical_qk_reuse.fused_value_pack import project
    from experiments.numerical_qk_reuse.v31_value_projection import refresh
    from experiments.numerical_qk_reuse.v2_online_warpspec import pack_z
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    g=torch.Generator(device='cuda').manual_seed(nk);v=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g)
    matrix=Projections().get(5,2,512,'gaussian',32,1729,v.device);z=torch.empty((1,2,nk,32),device=v.device);norm=torch.empty((1,2,nk),device=v.device);valid=torch.ones_like(norm,dtype=torch.bool)
    oldz=torch.empty_like(z);oldnorm=torch.empty_like(norm)
    ref=refresh(v,matrix,oldz,oldnorm,valid,0);nu,packed=project(v,matrix,z,norm,valid)
    assert torch.equal(z,oldz) and torch.equal(norm,oldnorm) and torch.equal(nu,ref)
    assert torch.equal(packed,pack_z(oldz))

@pytest.mark.parametrize('nk,prefix',[(641,504),(4096,3840)])
def test_partial_current_canvas_pack_preserves_unchanged_prefix(nk,prefix):
    from experiments.numerical_qk_reuse.fused_value_pack import project
    from experiments.numerical_qk_reuse.v31_value_projection import refresh
    from experiments.numerical_qk_reuse.v2_online_warpspec import pack_z
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    g=torch.Generator(device='cuda').manual_seed(nk);v=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g);matrix=Projections().get(5,2,512,'gaussian',32,1729,v.device)
    z=torch.empty((1,2,nk,32),device=v.device);norm=torch.empty((1,2,nk),device=v.device);valid=torch.ones_like(norm,dtype=torch.bool)
    _,packed=project(v,matrix,z,norm,valid);oldz=torch.empty_like(z);oldnorm=torch.empty_like(norm)
    for _ in range(4):
        v[:,:,prefix:].copy_(torch.randn(v[:,:,prefix:].shape,device=v.device,dtype=v.dtype,generator=g))
        nu,updated=project(v,matrix,z,norm,valid,start=prefix,packed=packed);ref=refresh(v,matrix,oldz,oldnorm,valid,0)
        assert updated.data_ptr()==packed.data_ptr() and torch.equal(z,oldz) and torch.equal(norm,oldnorm) and torch.equal(nu,ref)
        assert torch.equal(packed,pack_z(oldz))

def test_cached_canvas_refresh_matches_full_current_values():
    import os
    from pathlib import Path
    from experiments.numerical_qk_reuse.v2_online_warpspec import CanvasPackedRefresh,PackedRefresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    library=os.environ.get('V2_WARPSPEC_LIBRARY')
    if not library:pytest.skip('explicit selected library required')
    g=torch.Generator(device='cuda').manual_seed(1729);q=torch.randn((1,4,256,512),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,2,1024,512),device=q.device,dtype=q.dtype,generator=g);v=torch.randn(k.shape,device=q.device,dtype=q.dtype,generator=g)
    matrix=Projections().get(5,2,512,'gaussian',32,1729,v.device);op=CanvasPackedRefresh(Path(library),2);full=PackedRefresh(Path(library),2)
    for canvas in [0,1]:
        v.copy_(torch.randn(v.shape,device=v.device,dtype=v.dtype,generator=g))
        for step in range(3):
            v[:,:,768:].copy_(torch.randn(v[:,:,768:].shape,device=v.device,dtype=v.dtype,generator=g))
            out,keep,_=op(q,k,v,matrix,512**-.5,12,.12,cache_key=5,canvas=canvas,prefix=768)
            ref,chosen,_=full(q,k,v,matrix,512**-.5,12,.12)
            assert torch.equal(out,ref) and torch.equal(keep,chosen)
