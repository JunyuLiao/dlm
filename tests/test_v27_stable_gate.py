"""v27 dense-confirmation gate: the rest of a canvas runs dense once the argmax canvas stops changing."""
import types

import torch

from experiments.numerical_qk_reuse.integration import DENSITY_GATES, Attention, density_gate_fires


def test_stable_preset_fires_only_on_an_unchanged_argmax():
    preset = DENSITY_GATES['stable1']
    assert density_gate_fires(preset, 3, None, 10.0, 9.0, 0, changed=0)[0]
    assert not density_gate_fires(preset, 3, None, 10.0, 9.0, 0, changed=2)[0]
    assert not density_gate_fires(preset, 3, None, 10.0, 9.0, 0, changed=None)[0]
    # the old presets ignore the argmax signal entirely
    assert not density_gate_fires(DENSITY_GATES['ent0.05'], 3, 0.2, 10.0, 9.0, 0, changed=0)[0]


def _router(preset):
    r = Attention.__new__(Attention)
    r.cache = types.SimpleNamespace(clear=lambda: None)
    r.valid_keys, r.summaries, r._fa4_lists = {}, {}, []
    r.density_gate = dict(DENSITY_GATES[preset])
    r.canvas, r.step = -1, -1
    r.gate_dense, r.gate_stalled, r.gate_previous_accepted = False, 0, None
    r.gate_entries, r.gate_dense_calls = [], 0
    r._sampler_entropy = r._sampler_accepted = None
    r._sampler_changed = r._previous_top = None
    return r


def _logits(tokens, vocab=7):
    x = torch.full((1, len(tokens), vocab), -5.0)
    x[0, torch.arange(len(tokens)), torch.tensor(tokens)] = 5.0
    return x


def test_gate_sequence_over_a_canvas_and_reset_at_the_next_canvas():
    r = _router('stable1')
    accepted = torch.ones(1, 4, dtype=torch.bool)
    for step, tokens in enumerate([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5], [1, 2, 3, 5]]):
        r.begin_step(0, step)
        if step <= 2:
            assert not r.gate_dense            # steps 0-2: no unchanged pair has been observed yet
        r.observe_sampler(_logits(tokens), accepted)
        assert r._sampler_entropy is None           # the stable-only preset never computes the entropy
    r.begin_step(0, 4)
    assert r.gate_dense and r.gate_entries == [3]   # step 2 repeated step 1's argmax -> dense from step 3
    r.begin_step(1, 0)                               # new canvas: the gate and the argmax history reset
    assert not r.gate_dense and r._previous_top is None


def test_entropy_presets_unchanged():
    r = _router('ent0.05')
    accepted = torch.ones(1, 4, dtype=torch.bool)
    r.begin_step(0, 0)
    r.observe_sampler(_logits([1, 2, 3, 4]), accepted)    # near-zero entropy -> fires at step 1
    assert r._sampler_entropy is not None and r._sampler_changed is None
    r.begin_step(0, 1)
    assert r.gate_dense and r.gate_entries == [1]
