"""v27 D_fast: GLOBAL calls use repeated-KV SDPA; everything else stays native."""
import sys
import types

import pytest
import torch

from experiments.numerical_qk_reuse import v27_fast_dense as fast


def _fake_adapter(native):
    mod = types.ModuleType('fake_modeling_x')
    mod.ALL_ATTENTION_FUNCTIONS = {'sdpa': native}
    sys.modules['fake_modeling_x'] = mod
    model_cls = type('M', (), {'__module__': 'fake_generation_x'})
    return types.SimpleNamespace(model=model_cls()), mod


def _native(module, q, k, v, mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    g = q.shape[1] // k.shape[1]
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), scale=scaling, is_causal=bool(is_causal))
    return out.transpose(1, 2).contiguous(), 'native'


def test_global_calls_match_dense_math_and_others_delegate():
    adapter, mod = _fake_adapter(_native)
    q = torch.randn(1, 16, 8, 512)
    k, v = torch.randn(1, 2, 40, 512), torch.randn(1, 2, 40, 512)
    with fast.install(adapter, dict(control='D_fast'), fast.CONDITION) as runtime:
        attn = mod.ALL_ATTENTION_FUNCTIONS['sdpa']
        out, tag = attn(types.SimpleNamespace(is_sliding=False), q, k, v, None, scaling=.1, is_causal=False)
        assert tag is None
        torch.testing.assert_close(out, _native(None, q, k, v, None, scaling=.1, is_causal=False)[0])
        local_q = torch.randn(1, 16, 8, 256)
        local = attn(types.SimpleNamespace(is_sliding=True), local_q, torch.randn(1, 8, 40, 256),
                     torch.randn(1, 8, 40, 256), None, scaling=.1, is_causal=False)
        assert local[1] == 'native'
        masked = attn(types.SimpleNamespace(is_sliding=False), q, k, v, torch.zeros(1, 1, 8, 40),
                      scaling=.1, is_causal=False)
        assert masked[1] == 'native'
        assert runtime['counters']() == dict(fast_dense_global_calls=1, native_calls=2)
    assert mod.ALL_ATTENTION_FUNCTIONS['sdpa'] is _native


def test_identity_guard():
    adapter, _ = _fake_adapter(_native)
    with pytest.raises(ValueError):
        with fast.install(adapter, dict(control='other'), fast.CONDITION):
            pass


def test_runner_accepts_dense_control_without_router():
    from experiments.numerical_qk_reuse.runner import _runtime
    adapter, mod = _fake_adapter(_native)
    config = dict(control='D_c64', plugin=fast.PLUGIN, diagnostic=False)
    with _runtime(adapter, fast.CONDITION, config) as runtime:
        assert runtime['router'] is None and runtime['state'] is None
        assert mod.ALL_ATTENTION_FUNCTIONS['sdpa'] is not _native
        assert runtime['counters']() == dict(fast_dense_global_calls=0, native_calls=0)
    assert mod.ALL_ATTENTION_FUNCTIONS['sdpa'] is _native
    with pytest.raises(ValueError):
        with _runtime(adapter, fast.CONDITION, dict(config, plugin='x:y')):
            pass
