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


@contextmanager
def install(adapter, config: dict, condition: str):
    import importlib

    import torch
    if condition != CONDITION or config.get('control') != 'D_fast':
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
