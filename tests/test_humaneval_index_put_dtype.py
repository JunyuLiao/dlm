"""Reproduce the native guarded assignment defect and validate its narrow fix."""
import pytest
import torch
from torch._inductor import lowering
from scripts.v31_humaneval_index_put_dtype import install

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compiler diagnostic')


def test_guarded_bf16_value_fp32_cache_assignment_reproduces_and_is_repaired():
    old_guard=torch.are_deterministic_algorithms_enabled()
    old_fill=torch.utils.deterministic.fill_uninitialized_memory
    original=lowering.index_put_fallback
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory=False
    def update(cache,slots,embeds,normalizer,keep):
        cache[slots]=(embeds*normalizer)*keep
        return cache
    options=dict(deterministic=True,benchmark_combo_kernel=False,fx_graph_cache=False)
    target=torch.zeros((2,2,4),device='cuda',dtype=torch.float32)
    slots=torch.tensor([0],device='cuda',dtype=torch.int64)
    values=torch.tensor([[[1.125,-2.25,3.5,4.],[.25,.5,-.75,1.]]],device='cuda',dtype=torch.bfloat16)
    normalizer=torch.tensor(2.,device='cuda',dtype=torch.float32)
    keep=torch.ones((1,1,1),device='cuda',dtype=torch.bool)
    try:
        unpatched=torch.compile(update,dynamic=True,options=options)
        with pytest.raises(RuntimeError,match='scalar type|same dtype|dtypes match'):
            unpatched(target.clone(),slots,values,normalizer,keep)
            torch.cuda.synchronize()
        metadata=install()
        assert metadata['native_sampler_python_changed'] is False
        torch._dynamo.reset()
        patched=torch.compile(update,dynamic=True,options=options)
        result=patched(target.clone(),slots,values,normalizer,keep)
        expected=target.cpu()
        expected[0]=(values[0].cpu().float()*2)
        assert torch.equal(result.cpu(),expected)
        assert patched._torchdynamo_orig_callable is update
        assert torch.equal(values.cpu(),torch.tensor([[[1.125,-2.25,3.5,4.],[.25,.5,-.75,1.]]],dtype=torch.bfloat16))
        assert not bool(target.any())
    finally:
        lowering.index_put_fallback=original
        torch._dynamo.reset()
        torch.use_deterministic_algorithms(old_guard)
        torch.utils.deterministic.fill_uninitialized_memory=old_fill
