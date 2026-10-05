"""GPU qualification only; never run as part of CPU unit tests.

Reuses the predeclared v11 Hopper-vs-FP32/Triton envelope and source tensors.
Set V11_SUPPORT_BUILD to the verified build_identity.json on the target host.
No generation, gold, scorer or remote dispatch occurs in this module.
"""
import math
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

_v11_spec = importlib.util.spec_from_file_location(
    '_v20_pinned_v11_support_reference', Path(__file__).with_name('test_v11_support_consumer.py'))
v11 = importlib.util.module_from_spec(_v11_spec)
_v11_spec.loader.exec_module(v11)


@pytest.fixture(scope='module')
def support():
    identity = os.environ.get('V11_SUPPORT_BUILD')
    if not identity:
        pytest.skip('set V11_SUPPORT_BUILD to verified support build_identity.json')
    from experiments.value_direction_hopper import support as consumer
    consumer.load(identity)
    return consumer


def _source(prefix=129, nq=128, seed=17):
    q, k, v, native_window, scale = v11.make(seed, v11.LOCAL, prefix, nq=nq, crop=False)
    cache = SimpleNamespace(
        layers={2: SimpleNamespace(keys=k[..., :prefix, :].clone(),
                                   values=v[..., :prefix, :].clone())},
        is_compileable=False, get_seq_length=lambda: prefix)
    module = SimpleNamespace(layer_idx=2, training=False, is_sliding=True,
                             layer_type='sliding_attention')
    return q, k, v, native_window, scale, cache, module


def _router(interval, support_build, *, consumer='hopper', prefix=129):
    from experiments.numerical_qk_reuse.integration import Attention
    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x
    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda *_: False)
    return Attention(adapter, {k: {'log_threshold': -2.0} for k in ('local', 'global')},
                     score_period=8, decision_interval=interval, support='native_mask',
                     output_mode='historical_route_preqk_current_output',
                     selector='legacy_recompute', selector_layers='all',
                     kernel_variant='generic', telemetry='minimal', guard_mode='fused',
                     consumer=consumer, support_build=support_build if consumer == 'hopper' else None,
                     max_cache_bytes=4 * 1024**3, max_summary_bytes=1024**3)


def _call(router, source, step):
    q, k, v, window, scale, cache, module = source
    router.begin_step(0, step)
    router.identify(module, (), {'past_key_values': cache})
    router.sketches.identify(module, (), {'past_key_values': cache})
    output, _ = router(module, q, k, v, None, scaling=scale, is_causal=False,
                       sliding_window=window)
    entry = router.cache.entries[2]
    return output, entry.decision.skipped.clone(), entry.decision.eligible.clone()


def test_native_mask_local_does_not_reintroduce_window(support):
    source = _source(prefix=1645, nq=128)
    router = _router(1, os.environ['V11_SUPPORT_BUILD'])
    try:
        output, skipped, eligible = _call(router, source, 0)
        q, k, v, window, scale, _, _ = source
        scores = router.cache.entries[2].scores
        assert scores.shape[-1] == k.shape[-2]  # no prefix crop
        assert torch.isfinite(scores[..., :64]).any()  # earliest prefix remains legal
        assert torch.isfinite(output.float()).all()
        assert router.counters()['unsupported_mask_refreshes'] == 0
    finally:
        router.close()


@pytest.mark.parametrize('geometry,prefix', [(v11.LOCAL, 1645), (v11.GLOBAL, 129)])
def test_same_support_hopper_triton_predeclared_envelope(support, geometry, prefix):
    # LOCAL uses window=0 on the native-legal domain; no historical local crop.
    q, k, v, _window, scale = v11.make(prefix, geometry, prefix, crop=False)
    skipped, eligible = v11.routed(q, k, v, 0, scale, -1.0, prefix)
    v11.check(support, q, k, v, skipped, eligible, 0, scale)


def test_dropped_tile_nan_poison_never_reaches_current_output(support):
    q, k, v, _window, scale = v11.make(31, v11.LOCAL, 129, crop=False)
    kt = (k.shape[2] + 63) // 64
    eligible = torch.ones(1, q.shape[1], (q.shape[2] + 127) // 128, kt,
                          device='cuda', dtype=torch.bool)
    skipped = torch.zeros_like(eligible)
    skipped[..., 1] = True
    clean, _, invalid_clean, counters = support.attention(q, k, v, skipped, eligible,
                                                           scale=scale, window=0, counters=True)
    kp, vp = k.clone(), v.clone()
    kp[..., 64:128, :] = math.nan
    vp[..., 64:128, :] = math.nan
    poisoned, _, invalid_poisoned, poisoned_counters = support.attention(
        q, kp, vp, skipped, eligible, scale=scale, window=0, counters=True)
    torch.cuda.synchronize()
    assert not invalid_clean.any() and not invalid_poisoned.any()
    assert torch.equal(clean.view(torch.int16), poisoned.view(torch.int16))
    assert torch.equal(counters, poisoned_counters)


def test_r1_matches_m1_and_r_at_least_a_matches_held_on_actual_calls(support):
    source = _source(prefix=129, nq=128)
    build = os.environ['V11_SUPPORT_BUILD']
    baseline_m1, v20_r1, held_a8, r9 = (_router(i, build) for i in (1, 1, 8, 9))
    try:
        for step in range(9):
            m1 = _call(baseline_m1, source, step)
            r1 = _call(v20_r1, source, step)
            b = _call(held_a8, source, step)
            larger = _call(r9, source, step)
            for left, right in ((m1, r1), (b, larger)):
                assert torch.equal(left[0].view(torch.int16), right[0].view(torch.int16))
                assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
        assert held_a8.score_calls == r9.score_calls == 2
        assert held_a8.decision_calls == r9.decision_calls == 2
        assert v20_r1.decision_calls == baseline_m1.decision_calls == 9
    finally:
        for router in (baseline_m1, v20_r1, held_a8, r9):
            router.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('fraction,target', [(0.75, 3), (0.30, 1)])
def test_ported_mass_bitmap_native_q128_invariants(device, fraction, target):
    from experiments.numerical_qk_reuse.v20_controls import mass_bitmap
    # Five wholly immutable prefix tiles, then current canvas. All scores tie;
    # stable ranking must choose low physical tile indices after mandatory tile0.
    prefix, nq = 320, 128
    scores = torch.zeros(1, 2, nq, prefix + nq, device=device)
    skipped, eligible = mass_bitmap(scores, prefix, fraction)
    assert skipped.shape == (1, 2, 1, 7)
    assert eligible.all()
    assert not skipped[..., 0].any()  # first argmax tile mandatory in every row
    assert not skipped[..., prefix // 64:].any()  # whole current canvas mandatory
    assert int(skipped.sum(-1).max()) <= target
    assert torch.equal(skipped[0, 0, 0, 1:1 + target],
                       torch.ones(target, device=device, dtype=torch.bool))
    # An unaligned prefix boundary is mandatory, even though it intersects a
    # nominal prefix; only floor(prefix/64) full tiles can enter the ranking.
    boundary = 350
    scores_boundary = torch.zeros(1, 1, nq, boundary + nq, device=device)
    skipped_boundary, _ = mass_bitmap(scores_boundary, boundary, fraction)
    assert not skipped_boundary[..., boundary // 64:].any()
    assert int(skipped_boundary.sum(-1).max()) <= int((boundary // 64) * fraction)
    # Two distinct row winners within the same physical Q128 block cannot be
    # deleted, even if the target exceeds the remaining eligible candidates.
    winners = scores.clone()
    winners[0, :, 0, 3 * 64] = 20
    winners[0, :, 1, 4 * 64] = 20
    skipped_winners, _ = mass_bitmap(winners, prefix, fraction)
    assert not skipped_winners[..., [0, 3, 4]].any()
    assert int(skipped_winners.sum(-1).max()) <= target


def test_ported_held_control_observes_step1_then_holds(support):
    from experiments.numerical_qk_reuse.v20_controls import Consumer
    source = _source(prefix=129, nq=128)
    model = torch.nn.Module()
    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda *_: False)
    config = dict(policy={k: {'log_threshold': -2.0} for k in ('local', 'global')},
                  consumer='hopper', support_build=os.environ['V11_SUPPORT_BUILD'])
    control = Consumer(adapter, config, held=True)
    try:
        q, k, v, window, scale, cache, module = source
        for step in range(3):
            control.begin_step(0, step)
            control.owner.identify(module, (), {'past_key_values': cache})
            output, _ = control(module, q, k, v, None, scaling=scale, is_causal=False,
                                sliding_window=window)
            assert torch.isfinite(output.float()).all()
        counts = control.counters()
        assert (counts['bootstrap_calls'], counts['bitmap_observation_calls'],
                counts['held_decision_calls']) == (1, 1, 1)
        assert counts['phase_counts'] == {'A': 2, 'D': 0, 'H': 1}
    finally:
        control.close()


def test_fast_t_exact_completed_logit_history_gpu():
    """Same fixed completed logits yield identical chosen T weights at every step."""
    from experiments.value_direction_hopper.query_adaptive import State
    reference_router, fast_router = SimpleNamespace(query_sensitivity=None), SimpleNamespace(query_sensitivity=None)
    reference = State('T', reference_router, m_ref=14.258454322814941,
                      beta=3., gamma=.5, diagnostics=False, fast_t=False)
    fast = State('T', fast_router, m_ref=14.258454322814941,
                 beta=3., gamma=.5, diagnostics=False, fast_t=True)
    canvas = torch.zeros((1, 4), device='cuda', dtype=torch.long)
    generator = torch.Generator(device='cuda').manual_seed(20260927)
    completed = [torch.randn((1, 4, 97), device='cuda', generator=generator)
                 for _ in range(5)]
    # Force both no-flip and flip transitions without relying on random ties.
    completed[1] = completed[0].clone()
    completed[2][..., 0] = 50.
    accepted = torch.ones_like(canvas, dtype=torch.bool)
    for step, logits in enumerate(completed):
        for state in (reference, fast):
            state.begin(48-step, canvas)
        if step == 0:
            assert reference.used_weights is None and fast.used_weights is None
        else:
            assert torch.equal(reference.used_weights.view(torch.int32),
                               fast.used_weights.view(torch.int32))
            assert torch.equal(reference_router.query_sensitivity.view(torch.int32),
                               fast_router.query_sensitivity.view(torch.int32))
            if step == 1:
                assert torch.equal(fast.used_weights, torch.ones_like(fast.used_weights))
        for state in (reference, fast):
            state.observe_logits(logits, accepted, 48-step)
        assert torch.equal(reference.temporal.view(torch.int32), fast.temporal.view(torch.int32))
        assert torch.equal(reference.previous_top, fast.previous_top)
