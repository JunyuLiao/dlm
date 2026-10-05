"""CPU geometry qualification for the opt-in native-legal attention domain."""
import hashlib
import inspect

import pytest
import torch


NATIVE_SDPA_SHA256 = '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564'


def test_native_sdpa_source_identity():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    source = inspect.getfile(sdpa_attention_forward)
    assert hashlib.sha256(open(source, 'rb').read()).hexdigest() == NATIVE_SDPA_SHA256


def make_router(monkeypatch, support_geometry):
    from experiments.value_direction_hopper import integration as mod

    captured = {}

    class FakeKernel:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, q, k, v, z, ref, **kwargs):
            captured['kernel'] = kwargs
            return type('Result', (), dict(output=q, skipped=torch.zeros(1, 1, 1, 2, dtype=torch.bool),
                                           eligible=torch.ones(1, 1, 1, 2, dtype=torch.bool)))()

    class FakeSketches:
        def __init__(self, *_args, **_kwargs):
            pass

        def get(self, layer, v, validkv, prefix):
            captured['validkv'] = validkv.clone()
            return torch.zeros(*v.shape[:-1], 32), torch.ones(v.shape[:2])

    def fake_geometry(batch, nq, nk, *, device, causal, window):
        captured['geometry'] = dict(causal=causal, window=window)
        key = torch.arange(nk)
        rows = torch.arange(nq) + nk - nq
        valid = (key[None, :] < nk).expand(nq, nk)
        if causal:
            valid = valid & (key[None, :] <= rows[:, None])
        if window:
            valid = valid & (key[None, :] >= rows[:, None] - window + 1)
        captured['dense_geometry'] = valid
        return valid, valid.any(0)

    monkeypatch.setattr(mod, 'Kernel', FakeKernel)
    monkeypatch.setattr(mod, 'Sketches', FakeSketches)
    monkeypatch.setattr(mod, 'geometry', fake_geometry)
    monkeypatch.setattr(mod, 'pack', lambda valid: valid)
    monkeypatch.setattr(mod, '_attention_type', lambda module, window: 'local' if window else 'global')
    router = mod.Attention(None, 'unused', {'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}},
                           collect=False, projection='torch', support_geometry=support_geometry)
    return router, captured


@pytest.mark.parametrize('window', [None, 4])
def test_native_geometry_matches_unmasked_sdpa_and_keeps_policy_kind(monkeypatch, window):
    router, captured = make_router(monkeypatch, 'native_legal')
    q = torch.zeros(1, 1, 3, 2)
    kv = torch.zeros(1, 1, 70, 2)  # partial 64-key tile and a short current canvas
    module = type('Module', (), dict(layer_idx=0, training=False))()
    router(module, q, kv, kv, None, is_causal=False, sliding_window=window)
    assert captured['geometry'] == dict(causal=False, window=0)
    assert captured['dense_geometry'].shape == (3, 70)
    assert captured['dense_geometry'].all() and captured['validkv'].all()
    assert captured['kernel']['log_threshold'] == (-1. if window else -2.)


def test_native_geometry_preserves_explicit_padding_mask(monkeypatch):
    router, captured = make_router(monkeypatch, 'native_legal')
    q = torch.zeros(1, 1, 3, 2)
    kv = torch.zeros(1, 1, 70, 2)
    mask = torch.ones(1, 1, 3, 70, dtype=torch.bool)
    mask[..., -5:] = False
    module = type('Module', (), dict(layer_idx=0, training=False))()
    router(module, q, kv, kv, mask, is_causal=False, sliding_window=4)
    assert torch.equal(captured['kernel']['mask'], mask)
    assert not captured['validkv'][..., -5:].any()
    assert captured['kernel']['log_threshold'] == -1.


def test_native_geometry_rejects_implicit_or_causal_call(monkeypatch):
    router, _ = make_router(monkeypatch, 'native_legal')
    q = torch.zeros(1, 1, 3, 2)
    kv = torch.zeros(1, 1, 70, 2)
    module = type('Module', (), dict(layer_idx=0, training=False))()
    for causal in (None, True):
        with pytest.raises(ValueError, match='explicit is_causal=False'):
            router(module, q, kv, kv, None, is_causal=causal, sliding_window=4)


def test_legacy_geometry_still_uses_window(monkeypatch):
    router, captured = make_router(monkeypatch, 'legacy_junyu')
    q = torch.zeros(1, 1, 3, 2)
    kv = torch.zeros(1, 1, 70, 2)
    module = type('Module', (), dict(layer_idx=0, training=False))()
    router(module, q, kv, kv, None, is_causal=False, sliding_window=4)
    assert captured['geometry'] == dict(causal=False, window=4)
    assert not captured['dense_geometry'][:, :64].any()


def test_frontier_scope_rejects_wrong_arm_and_dense_threshold():
    from experiments.value_direction_hopper.frontier_scope import install

    adapter = type('Adapter', (), {'model': object()})()
    with pytest.raises(ValueError, match='named frontier arm'):
        with install(adapter, {'frontier_arm': 'invalid'}, 'native_legal_all_layers'):
            pass
    config = dict(frontier_arm='D_matched', method='kernel_dense', diagnostic=False,
                  policy={'local': {'log_threshold': -1.}, 'global': {'log_threshold': -float('inf')}})
    with pytest.raises(ValueError, match='retain every legal tile'):
        with install(adapter, config, 'native_legal_all_layers'):
            pass
