"""Pinned compiler adapter for guarded CUDA FP32 <- BF16 indexed assignment."""
import functools
import hashlib
import inspect


def install():
    import torch
    from torch._inductor import lowering
    original=lowering.index_put_fallback
    implementation=inspect.getsource(lowering.index_put_impl_)
    if not (implementation.index('if torch.are_deterministic_algorithms_enabled():') <
            implementation.index('values = to_dtype(values, self.get_dtype())')):
        raise RuntimeError('pinned fallback-before-cast compiler structure changed')
    if getattr(original,'_humaneval_fp32_bf16_cast',False):
        raise RuntimeError('compiler adapter already installed')

    @functools.wraps(original)
    def fallback(target,indices,values,accumulate):
        device=target.get_device()
        integer_indices=all(index is None or index.get_dtype() in (torch.int32,torch.int64) for index in indices)
        if (torch.are_deterministic_algorithms_enabled() and not accumulate and
            device is not None and device.type=='cuda' and integer_indices and
            target.get_dtype()==torch.float32 and values.get_dtype()==torch.bfloat16):
            # The ordinary index_put_impl_ path performs this exact conversion
            # before its scatter. The guarded fallback returned before it.
            values=lowering.to_dtype(values,target.get_dtype())
        return original(target,indices,values,accumulate)

    fallback._humaneval_fp32_bf16_cast=True
    lowering.index_put_fallback=fallback
    return dict(scope='guarded CUDA integer-index nonaccumulating FP32 target, BF16 values',
        original_fallback_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_implementation_sha256=hashlib.sha256(implementation.encode()).hexdigest(),
        # inspect.getsource(function) unwraps functools.wraps and would fingerprint
        # the original fallback. The code object identifies the installed body.
        adapter_sha256=hashlib.sha256(inspect.getsource(fallback.__code__).encode()).hexdigest(),
        native_sampler_python_changed=False)
