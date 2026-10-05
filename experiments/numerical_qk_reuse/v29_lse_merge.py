"""Standalone alias2 LSE merge + original-row writeback experiment.

No adapter integration or FA4 loader change. Lazy torch/Triton imports permit
CPU contract tests. Both natural and grouped arms must use this same merge.
"""
import numpy as np
from functools import lru_cache


def validate_order(order, heads, rows):
    order = np.asarray(order)
    if (order.shape != (heads, rows) or order.dtype.kind not in 'iu'
            or heads < 1 or rows < 1
            or not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(rows), order.shape))):
        raise ValueError('per-head grouped-slot to original-row bijection required')
    return order


def merge_reference(partials, lse, order):
    """FP64 CPU oracle; shapes [2,Q,H,D], [2,H,Q], [H,Q]."""
    partials, lse = np.asarray(partials), np.asarray(lse)
    if partials.ndim != 4 or partials.shape[0] != 2 or min(partials.shape[1:]) < 1:
        raise ValueError('exactly two nonempty alias partials required')
    _, q, h, _ = partials.shape
    order = validate_order(order, h, q)
    if lse.shape != (2, h, q) or np.isnan(lse).any() or np.isposinf(lse).any():
        raise ValueError('finite or negative-infinite LSE shape required')
    if (~np.isfinite(lse)).all(axis=0).any():
        raise ValueError('at least one nonempty split per output row required')
    if not np.isfinite(partials).all():
        raise ValueError('finite partial outputs required')
    lse = lse.astype(np.float64)
    weights = np.exp(lse-lse.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    grouped = (partials.astype(np.float64)*weights.transpose(0, 2, 1)[..., None]).sum(axis=0)
    result = np.empty_like(grouped)
    result[order.T, np.arange(h)[None, :]] = grouped
    return result[None]


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def merge(O, L, Map, Y, Q: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              OS: tl.constexpr, OQ: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
              LS: tl.constexpr, LH: tl.constexpr, LQ: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        valid = i < Q*H*D
        d = i % D
        h = (i//D) % H
        q = i//(D*H)
        orig = tl.load(Map+h*Q+q, mask=valid, other=0)
        l0 = tl.load(L+h*LH+q*LQ, mask=valid, other=-float('inf'))
        l1 = tl.load(L+LS+h*LH+q*LQ, mask=valid, other=-float('inf'))
        m = tl.maximum(l0, l1)
        e0, e1 = tl.exp(l0-m), tl.exp(l1-m)
        a = tl.load(O+q*OQ+h*OH+d*OD, mask=valid, other=0).to(tl.float32)
        b = tl.load(O+OS+q*OQ+h*OH+d*OD, mask=valid, other=0).to(tl.float32)
        denom = e0+e1
        w0, w1 = e0/denom, e1/denom
        value = a*w0+b*w1
        tl.store(Y+(orig*H+h)*D+d, value, mask=valid)

    return merge


class Alias2MappedMerge:
    """Frozen map validated once; no oracle/D2H in warmed calls."""

    def __init__(self, order):
        import torch
        if (order.ndim != 2 or order.dtype != torch.long or not order.is_cuda
                or not order.is_contiguous()):
            raise ValueError('contiguous CUDA int64 per-head map required')
        self.heads, self.rows = order.shape
        validate_order(order.cpu().numpy(), self.heads, self.rows)
        self.order = order
        self.kernel = _kernel()

    @classmethod
    def identity(cls, heads, rows, device):
        """Construct known immutable identity on CPU; no device-to-host read."""
        import torch
        if any(type(x) is not int or x <= 0 for x in (heads, rows)):
            raise ValueError('positive integer identity dimensions required')
        order = np.broadcast_to(np.arange(rows, dtype=np.int64), (heads, rows)).copy()
        validate_order(order, heads, rows)
        obj = cls.__new__(cls)
        obj.heads, obj.rows = heads, rows
        obj.order = torch.from_numpy(order).to(device=device, dtype=torch.long)
        obj.kernel = _kernel()
        return obj

    def __call__(self, partials, lse):
        import torch
        import triton
        if (partials.ndim != 4 or partials.shape[:3] != (2, self.rows, self.heads)
                or partials.shape[-1] < 1 or partials.dtype != torch.bfloat16
                or lse.shape != (2, self.heads, self.rows) or lse.dtype != torch.float32
                or not partials.is_cuda or not lse.is_cuda
                or partials.device != self.order.device or lse.device != self.order.device
                or any(s <= 0 for s in (*partials.stride(), *lse.stride()))):
            raise ValueError('alias2 CUDA BF16 partials and FP32 LSE geometry required')
        d = partials.shape[-1]
        result = torch.empty((1, self.rows, self.heads, d), device=partials.device, dtype=partials.dtype)
        self.kernel[(triton.cdiv(self.rows*self.heads*d, 1024),)](
            partials, lse, self.order, result, self.rows, self.heads, d,
            *partials.stride(), *lse.stride(), 1024, num_warps=4, enable_fp_fusion=False)
        return result
