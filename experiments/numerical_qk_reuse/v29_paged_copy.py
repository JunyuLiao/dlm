"""Opt-in exact paged KV copy; no routing, arithmetic, stream or sampler change.

Fuse the adapter's advanced-index gather, transpose and copy into preallocated
head-major buffers. Supports actual interleaved K/V cache strides. No CPU read
of device contents, allocation, synchronization or first-call oracle here;
the caller must warm and qualify before timed use.
"""
from functools import lru_cache


def validate_geometry(shape, kstrides, vstrides, target_shape, table_size, start, count):
    if (len(shape) != 4 or len(kstrides) != 4 or len(vstrides) != 4
            or any(type(x) is not int or x <= 0 for x in (*shape, *kstrides, *vstrides))):
        raise ValueError('positive four-dimensional cache shape/strides required')
    pages, page, heads, dim = shape
    if (len(target_shape) != 4 or target_shape[0] != 1
            or target_shape[1] != heads or target_shape[3] != dim):
        raise ValueError('destination must be [1, KV heads, sequence, dim]')
    nk = target_shape[2]
    if any(type(x) is not int for x in (nk, table_size, start, count)):
        raise ValueError('integer copy geometry required')
    if nk <= 0 or start < 0 or count <= 0 or start + count > nk:
        raise ValueError('copy range outside destination')
    if table_size < (start + count + page - 1) // page:
        raise ValueError('page table does not cover requested range')
    return page, heads, dim, nk


def cpu_address(logical_token, head, dim_index, table, page_size, source_strides, nk, heads, dim):
    """Independent scalar address specification for unit tests, never timed."""
    if not (0 <= logical_token < nk and 0 <= head < heads and 0 <= dim_index < dim):
        raise ValueError('coordinate outside destination')
    physical_page = table[logical_token // page_size]
    sp, st, sh, sd = source_strides
    return (physical_page * sp + (logical_token % page_size) * st + head * sh + dim_index * sd,
            (head * nk + logical_token) * dim + dim_index)


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['START', 'COUNT', 'NK'])
    def copy(K, V, Table, KO, VO,
             START, COUNT, NK,
             PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
             VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
             TS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < COUNT * D
        d = i % D
        token = START + i // D
        head = tl.program_id(1)
        physical = tl.load(Table + (token // PAGE) * TS, mask=valid, other=0)
        slot = token % PAGE
        k = tl.load(K + physical * KP + slot * KT + head * KH + d * KD, mask=valid, other=0)
        v = tl.load(V + physical * VP + slot * VT + head * VH + d * VD, mask=valid, other=0)
        out = (head * NK + token) * D + d
        tl.store(KO + out, k, mask=valid)
        tl.store(VO + out, v, mask=valid)
    return copy


def copy_paged_kv(key, value, table, out_key, out_value, start, count):
    import torch
    import triton
    if key.shape != value.shape or out_key.shape != out_value.shape:
        raise ValueError('K/V shape mismatch')
    page, heads, dim, nk = validate_geometry(tuple(key.shape), tuple(key.stride()), tuple(value.stride()),
                                             tuple(out_key.shape), table.numel(), start, count)
    tensors = (key, value, table, out_key, out_value)
    if any(not x.is_cuda or x.device != key.device for x in tensors):
        raise ValueError('all buffers must be on the same CUDA device')
    if (key.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or any(x.dtype != key.dtype for x in (value, out_key, out_value))
            or table.ndim != 1 or table.dtype not in (torch.int32, torch.int64)
            or table.stride(0) <= 0 or not out_key.is_contiguous() or not out_value.is_contiguous()):
        raise ValueError('unsupported copy dtype or layout')
    # Page IDs are trusted native metadata, just as in the original adapter.
    # Full value/range checks belong to untimed GPU qualification, not this path.
    _kernel()[(triton.cdiv(count * dim, 1024), heads)](
        key, value, table, out_key, out_value, start, count, nk, page, heads, dim,
        *key.stride(), *value.stride(), table.stride(0), 1024, num_warps=4)
