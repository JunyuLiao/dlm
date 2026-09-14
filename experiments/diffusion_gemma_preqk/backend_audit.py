"""Check the existing physical reference before treating it as a drop-in backend."""
import inspect
from pathlib import Path
import torch
from dllm.attention.blasst.core import _finish_eager_attention
from experiments.diffusion_attention_threshold_modeling.routing import _physical_sparse_attention
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .run import digest


def audit(root):
    generator=torch.Generator().manual_seed(42)
    q=torch.randn(1,2,64,8,generator=generator,dtype=torch.bfloat16)
    k=torch.randn(1,2,128,8,generator=generator,dtype=torch.bfloat16);v=torch.randn(k.shape,generator=generator,dtype=torch.bfloat16)
    valid=torch.ones(1,2,64,128,dtype=torch.bool);scale=8**-.5
    scores=(q @ k.transpose(-2,-1))*scale
    dense=_finish_eager_attention(q,v,scores,valid,0.,False)[0]
    physical=_physical_sparse_attention(q,k,v,valid,None,scaling=scale,dropout=0.,training=False,block_size=(64,64))
    assert dense.shape==physical.shape and physical.isfinite().all()
    path=Path(inspect.getsourcefile(_physical_sparse_attention))
    result=dict(source=str(path),source_sha256=digest(path),function='_physical_sparse_attention',
        exists=True,kind='Python two-pass physical reference, not an optimized GPU-tile kernel',
        precision='FP32 QK/PV and accumulation versus frozen native BF16 QK/PV contract',
        exposes_block_observations=False,
        no_skip_cpu_fixture=dict(shape=list(q.shape),bit_exact=bool(torch.equal(dense,physical)),
            different_elements=int((dense!=physical).sum()),elements=dense.numel(),max_abs_difference=float((dense-physical).abs().max())),
        compatible_drop_in=False,
        decision='Not a validated replacement for the frozen online-history backend: different numerical contract and output-only interface (no required observed block mass/peak). Integrating/revalidating it is new backend work. No claim that no physical reference exists.',
        gpu_latency_measured=False,scope='Source/interface and small CPU BF16 numerical check only; no sparse-kernel performance claim')
    assert not result['no_skip_cpu_fixture']['bit_exact']
    _write(root/'backend_compatibility.json',result);return result


if __name__=='__main__':
    from .config import ROOT
    audit(ROOT)
