"""v9 section 4: live/peak memory accounting and guards for prefix summaries.

``summary_bytes`` used to be a tally updated only on build, so after a commit,
a new canvas or close it still reported buffers that no longer existed, and
``max_cache_bytes`` bounded the score cache only. These tests drive the real
router (real ScoreCache, real Triton selector) through build -> hit ->
replacement -> invalidation and check the live number each time.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_router(**kwargs):
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda name, module: False)
    router = Attention(adapter, {k: {'log_threshold': -1.0} for k in ('local', 'global')},
                       decision_interval=1, score_period=8,
                       output_mode='historical_route_preqk_current_output',
                       selector='prefix_block_summary', **kwargs)
    modules = {layer: SimpleNamespace(layer_idx=layer, training=False, is_sliding=True,
                                      layer_type='sliding_attention') for layer in (0, 1)}
    return router, modules, model


def inputs(generator, layer, prefix=192, canvas=64, hk=2, h=4, d=128):
    """Persistent per-layer tensors: identity carries the prefix OBJECT, so a
    fresh tensor per call would (correctly) force a refresh every time."""
    total = prefix + canvas
    k = torch.randn(1, hk, total, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    v = torch.randn(1, hk, total, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(1, h, canvas, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    cache = SimpleNamespace(layers={layer: SimpleNamespace(keys=k[..., :prefix, :], values=v[..., :prefix, :])},
                            is_compileable=False, get_seq_length=lambda: prefix)
    return dict(q=q, k=k, v=v, cache=cache, d=d)


def call(router, module, x):
    router.identify(module, (), {'past_key_values': x['cache']})
    router.sketches.identify(module, (), {'past_key_values': x['cache']})
    return router(module, x['q'], x['k'], x['v'], None, scaling=x['d'] ** -.5,
                  is_causal=False, sliding_window=4096)[0]


def expected_bytes(prefix, canvas=64, h=4):
    from experiments.numerical_qk_reuse.cached_executor import summary_bytes
    return summary_bytes(1, h, (canvas + 127) // 128, prefix // 64, 32)


def test_live_bytes_track_build_hit_replacement_and_every_invalidation():
    router, modules, model = make_router()
    generator = torch.Generator(device='cuda').manual_seed(9)
    x = {layer: inputs(generator, layer) for layer in (0, 1)}
    one = expected_bytes(192)
    try:
        router.begin_step(0, 0)
        call(router, modules[0], x[0])                  # anchor: build layer 0
        assert router.summary_bytes == one
        call(router, modules[1], x[1])                  # anchor: build layer 1
        assert router.summary_bytes == 2 * one
        assert router.counters()['summary_live_bytes'] == 2 * one
        router.begin_step(0, 1)
        call(router, modules[0], x[0])                  # ordinary: hit, no growth
        assert router.summary_hits >= 1 and router.summary_bytes == 2 * one
        router.begin_step(0, 8)                              # next anchor: replacement
        call(router, modules[0], x[0])
        assert router.summary_builds == 3 and router.summary_bytes == 2 * one
        assert router.counters()['summary_peak_bytes'] == 2 * one   # never double-held
        model.encoder(torch.zeros(1, device='cuda'))        # encoder commit
        assert router.summary_bytes == 0 and router.counters()['summary_live_bytes'] == 0
        router.begin_step(0, 9)
        call(router, modules[0], x[0])                  # identity changed -> rebuild
        assert router.summary_bytes == one
        router.begin_step(1, 0)                              # new canvas
        assert router.summary_bytes == 0
        router.begin_step(1, 0)
        call(router, modules[0], x[0])
        assert router.summary_bytes == one
        counters = router.counters()
        assert counters['score_live_bytes'] == router.cache.storage_bytes > 0
        assert counters['history_summary_peak_bytes'] >= counters['score_peak_bytes']
        assert counters['score_peak_transient_bytes'] >= counters['score_peak_bytes']
    finally:
        router.close()
    assert router.summary_bytes == 0


def test_summary_budget_is_enforced_before_allocation_with_exact_fallback():
    one = expected_bytes(192)
    outputs = {}
    for budget in (10 * one, one):          # second layer does not fit in the tight budget
        router, modules, _ = make_router(max_summary_bytes=budget)
        generator = torch.Generator(device='cuda').manual_seed(11)
        x = {layer: inputs(generator, layer) for layer in (0, 1)}
        try:
            router.begin_step(0, 0)
            first = [call(router, modules[layer], x[layer]) for layer in (0, 1)]
            router.begin_step(0, 1)
            second = [call(router, modules[layer], x[layer]) for layer in (0, 1)]
            torch.cuda.synchronize()
            outputs[budget] = first + second
            assert router.summary_bytes <= budget
            if budget == one:
                assert router.summary_budget_declines == 1 and router.summary_misses == 1
        finally:
            router.close()
    for loose, tight in zip(outputs[10 * one], outputs[one]):
        assert torch.equal(loose, tight)     # declining a summary never changes the output


def test_summary_arguments_reject_wrong_dtype_shape_and_prefix_range():
    from experiments.numerical_qk_reuse.cached_executor import _summary_arguments, allocate_summary
    shape, kt = (1, 4, 1, 4), 4
    good = allocate_summary(1, 4, 1, kt, 3, 32, torch.device('cuda', 0), ('id',))
    assert _summary_arguments(good, True, shape, kt, good.z.device)[0] == 3
    for field, bad in (('active', good.active.to(torch.bool)), ('bad', good.bad.to(torch.int32)),
                       ('z', good.z.double()), ('mu', good.mu.half()),
                       ('active', good.active[..., :64].contiguous()),
                       ('bad', good.bad.transpose(-1, -2).contiguous().transpose(-1, -2))):
        broken = allocate_summary(1, 4, 1, kt, 3, 32, torch.device('cuda', 0), ('id',))
        setattr(broken, field, bad)
        with pytest.raises(ValueError):
            _summary_arguments(broken, True, shape, kt, good.z.device)
    over = allocate_summary(1, 4, 1, 8, 5, 32, torch.device('cuda', 0), ('id',))
    with pytest.raises(ValueError):
        _summary_arguments(over, True, shape, kt, good.z.device)
    zero = allocate_summary(1, 4, 1, kt, 3, 32, torch.device('cuda', 0), ('id',))
    zero.prefix_tiles = 0
    with pytest.raises(ValueError):
        _summary_arguments(zero, True, shape, kt, good.z.device)
