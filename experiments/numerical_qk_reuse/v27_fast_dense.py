"""v27 strongest-known dense baseline: native attention with a faster GLOBAL SDPA call.

The installed native path calls torch SDPA with ``enable_gqa=True`` for the GLOBAL
layers (16 query heads, 2 KV heads, head_dim 512). On H100 that dispatch measured
5.24 ms at 17.5K keys, while repeating K/V to 16 heads and calling SDPA without GQA
measured 2.33 ms. This control keeps the model and every LOCAL layer untouched and
only issues the GLOBAL calls through the repeated-KV SDPA. The mathematics is dense
attention; the kernel and therefore the rounding may differ from native.
"""
from __future__ import annotations

from contextlib import contextmanager

PLUGIN = 'experiments.numerical_qk_reuse.v27_fast_dense:install'
CONDITION = 'v27_fast_dense_global_repeat_kv'
# D_fa4: FlashAttention-4's plain dense path. D_fa4_allkept: the same kernel through its block-sparse interface with
# every tile kept -- bitwise identical output, 4-6% faster at this shape, and the exact code path the M1/M2/M3
# FA4 consumer uses, so dense vs sparse differ only in skipped tiles. Both are kept (user request 2026-09-30).
CONTROLS = ('D_fast', 'D_c64', 'D_fa4', 'D_fa4_allkept')


@contextmanager
def install(adapter, config: dict, condition: str):
    import importlib

    import torch
    if condition != CONDITION or config.get('control') not in CONTROLS:
        raise ValueError('v27 fast dense control identity drift')
    model = adapter.model
    modeling = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_'))
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    native = registry['sdpa']
    counts = dict(fast_dense_global_calls=0, native_calls=0)

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                  is_causal=None, **kwargs):
        heads, kv_heads = query.shape[1], key.shape[1]
        is_global = query.shape[-1] == 512 and not getattr(module, 'is_sliding', False)
        if (not is_global or attention_mask is not None or is_causal is not False or dropout
                or heads % kv_heads):
            counts['native_calls'] += 1
            return native(module, query, key, value, attention_mask, dropout=dropout,
                          scaling=scaling, is_causal=is_causal, **kwargs)
        if config['control'] in ('D_fa4', 'D_fa4_allkept'):
            # the official SOTA dense kernel for these layers: FlashAttention-4 (vLLM fork, SM90 hd512)
            from . import v27_fa4
            counts['fast_dense_global_calls'] += 1
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            if config['control'] == 'D_fa4':
                return v27_fa4.dense_plain(query, key, value, scale), None
            return v27_fa4.dense(query, key, value, scale), None
        if config['control'] == 'D_c64':
            # Same 64-row kernel as the sparse consumer, every tile kept.
            from .v27_consumer64 import dense64
            counts['fast_dense_global_calls'] += 1
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            return dense64(query, key, value, scale, splits=config.get('c64_splits', 2)), None
        groups = heads // kv_heads
        k = key.repeat_interleave(groups, dim=1)
        v = value.repeat_interleave(groups, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=None, dropout_p=0.0,
                                                               scale=scaling, is_causal=False)
        counts['fast_dense_global_calls'] += 1
        return out.transpose(1, 2).contiguous(), None

    registry['sdpa'] = attention
    try:
        yield dict(counters=lambda: dict(counts))
    finally:
        registry['sdpa'] = native
