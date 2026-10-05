"""v27b long prefill: the O(n^2) prefill mask mapping is elided only for a batch-1, unpadded,
empty-cache prefill longer than 256, after the model's builder is checked once (CPU)."""
import sys
import types

import pytest
import torch

from experiments.numerical_qk_reuse import v27_long


class Cache:
    def __init__(self, n=0):
        self.n = n

    def get_seq_length(self):
        return self.n


class Config:
    def get_text_config(self):
        return types.SimpleNamespace(sliding_window=16)


def _builder(window=16, broken=False):
    calls = []

    def build(config, inputs_embeds, attention_mask, past_key_values, position_ids, mm_token_type_ids=None):
        n = inputs_embeds.shape[1]
        calls.append(n)
        i = torch.arange(n)
        full = i[None, :] <= i[:, None]
        if broken:
            full = torch.ones(n, n, dtype=torch.bool)
        sliding = full & (i[None, :] > i[:, None] - window)
        return {'full_attention': full[None, None], 'sliding_attention': sliding[None, None]}
    return build, calls


def _model(builder):
    mod = types.ModuleType('fake_modeling_long')
    mod.ALL_ATTENTION_FUNCTIONS = {'sdpa': lambda *a, **k: ('native', None)}
    sys.modules['fake_modeling_long'] = mod
    # like the real model: a class-level staticmethod, shadowed per instance by the wrapper
    encoder_cls = type('DiffusionGemmaEncoderModel', (torch.nn.Module,),
                       {'create_masks_for_generate': staticmethod(builder)})
    encoder = encoder_cls()
    model_cls = type('M', (torch.nn.Module,), {'__module__': 'fake_generation_long'})
    model = model_cls()
    model.encoder = encoder
    return model, encoder


def _kw(n, cache=0, padded=False, batch=1):
    mask = torch.ones(batch, n, dtype=torch.bool)
    if padded:
        mask[0, 0] = False
    return dict(config=Config(), inputs_embeds=torch.empty(batch, n, 0), attention_mask=mask,
                past_key_values=Cache(cache), position_ids=torch.arange(n)[None])


def test_long_unpadded_prefill_is_elided_after_one_semantic_check(monkeypatch):
    monkeypatch.setattr(v27_long, '_SEMANTICS_CHECKED', [])
    build, calls = _builder()
    model, encoder = _model(build)
    with v27_long.prefill_dense64(model, True):
        out = encoder.create_masks_for_generate(**_kw(3000))
        assert out == {'full_attention': None, 'sliding_attention': None}
        assert calls == [1500]                     # the small semantic check only
        encoder.create_masks_for_generate(**_kw(4000))
        assert calls == [1500]                     # checked once per process
        for kw in (_kw(256), _kw(3000, cache=5), _kw(3000, padded=True), _kw(3000, batch=2)):
            assert encoder.create_masks_for_generate(**kw)['full_attention'] is not None
    assert encoder.create_masks_for_generate is build   # restored


def test_prompt_shorter_than_the_probe_is_its_own_semantic_check(monkeypatch):
    monkeypatch.setattr(v27_long, '_SEMANTICS_CHECKED', [])
    build, calls = _builder()
    model, encoder = _model(build)
    with v27_long.prefill_dense64(model, True, kernel='fa4'):
        out = encoder.create_masks_for_generate(**_kw(700))
        assert out == {'full_attention': None, 'sliding_attention': None}
        assert calls == [700]


def test_non_causal_builder_refuses_elision(monkeypatch):
    monkeypatch.setattr(v27_long, '_SEMANTICS_CHECKED', [])
    build, _ = _builder(broken=True)
    model, encoder = _model(build)
    with v27_long.prefill_dense64(model, True):
        with pytest.raises(RuntimeError):
            encoder.create_masks_for_generate(**_kw(3000))


def test_disabled_wrapper_touches_nothing():
    build, calls = _builder()
    model, encoder = _model(build)
    with v27_long.prefill_dense64(model, False):
        assert encoder.create_masks_for_generate(**_kw(3000))['full_attention'] is not None
