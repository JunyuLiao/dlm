"""piecewise_v4: the decoder's static LOCAL shape (left-padded sliding cache + persistent key mask) is value-identical."""
import types

import pytest
import torch

from experiments.numerical_qk_reuse.v27_substrate import (SUBSTRATES, local_attention_v4, pad_sliding_cache,
                                                         prefill_kernel, unpad_sliding_cache)


class _Layer:
    is_sliding = True

    def __init__(self, n, full=7):
        self.keys = torch.randn(1, 2, n, 4)
        self.values = torch.randn(1, 2, n, 4)
        self.full = full

    def update(self, k, v):
        # HF DynamicSlidingWindowLayer: keep the last sliding_window - 1 tokens
        self.keys = torch.cat([self.keys, k], dim=-2)[:, :, -self.full:]
        self.values = torch.cat([self.values, v], dim=-2)[:, :, -self.full:]


class _Global:
    is_sliding = False

    def __init__(self, n):
        self.keys = torch.randn(1, 2, n, 4)
        self.values = torch.randn(1, 2, n, 4)


def test_v4_is_registered_with_fa4_prefill():
    assert 'piecewise_v4' in SUBSTRATES and prefill_kernel('piecewise_v4') == 'fa4'


def test_pad_then_unpad_restores_the_real_cache_and_updates_match_unpadded():
    torch.manual_seed(0)
    cache = types.SimpleNamespace(layers=[_Layer(3), _Global(9), _Layer(3)])
    real = [(l.keys.clone(), l.values.clone()) for l in cache.layers]
    assert pad_sliding_cache(cache, 7) == 4
    for layer, (k, v) in zip(cache.layers, real):
        if layer.is_sliding:
            assert layer.keys.shape[-2] == 7 and torch.equal(layer.keys[:, :, 4:], k)
            assert torch.count_nonzero(layer.keys[:, :, :4]) == 0
        else:
            assert torch.equal(layer.keys, k)                      # GLOBAL layers are never touched
    unpad_sliding_cache(cache)
    for layer, (k, v) in zip(cache.layers, real):
        assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
    # an encoder append after unpadding gives exactly the unpadded result
    reference = types.SimpleNamespace(layers=[_Layer(3)])
    reference.layers[0].keys, reference.layers[0].values = real[0]
    new_k, new_v = torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)
    cache.layers[0].update(new_k, new_v)
    reference.layers[0].update(new_k, new_v)
    assert torch.equal(cache.layers[0].keys, reference.layers[0].keys)


def test_full_window_is_left_alone_and_double_padding_is_refused():
    cache = types.SimpleNamespace(layers=[_Layer(7), _Layer(7)])
    before = cache.layers[0].keys
    assert pad_sliding_cache(cache, 7) == 0 and cache.layers[0].keys is before
    short = types.SimpleNamespace(layers=[_Layer(2)])
    pad_sliding_cache(short, 7)
    with pytest.raises(RuntimeError):
        pad_sliding_cache(short, 7)
    with pytest.raises(ValueError):
        pad_sliding_cache(types.SimpleNamespace(layers=[_Layer(2), _Layer(3)]), 7)


def test_padded_masked_local_attention_equals_unpadded_attention():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    torch.manual_seed(1)
    heads, kv_heads, d, canvas, real, full = 4, 2, 8, 5, 3, 7
    module = types.SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=False, training=False)
    q = torch.randn(1, heads, canvas, d)
    k_real, v_real = torch.randn(1, kv_heads, real, d), torch.randn(1, kv_heads, real, d)
    k_canvas, v_canvas = torch.randn(1, kv_heads, canvas, d), torch.randn(1, kv_heads, canvas, d)
    expected, _ = sdpa_attention_forward(module, q, torch.cat([k_real, k_canvas], -2),
                                         torch.cat([v_real, v_canvas], -2), None, is_causal=False)
    pad = full - real
    mask = torch.ones((1, 1, 1, full), dtype=torch.bool)
    mask[..., :pad] = False
    module._v27_pad = dict(active=True, mask=mask, full=full)
    zeros = torch.zeros(1, kv_heads, pad, d)
    got, _ = local_attention_v4(sdpa_attention_forward)(
        module, q, torch.cat([zeros, k_real, k_canvas], -2), torch.cat([zeros, v_real, v_canvas], -2), None,
        is_causal=False)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    module._v27_pad['active'] = False      # full window: plain native call, no mask
    plain, _ = local_attention_v4(sdpa_attention_forward)(
        module, q, torch.cat([k_real, k_canvas], -2), torch.cat([v_real, v_canvas], -2), None, is_causal=False)
    torch.testing.assert_close(plain, expected, rtol=0, atol=0)
