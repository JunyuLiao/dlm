"""v10 CP2 repair: minimal production telemetry must not change anything the
model consumes, must keep every correctness guard, and must report per-tile
statistics as N/A rather than zeros."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def run(telemetry, variant, monkeypatch=None):
    from tests.test_numerical_reuse_summary_memory import call, inputs, make_router
    router, modules, _ = make_router(telemetry=telemetry, kernel_variant=variant)
    generator = torch.Generator(device='cuda').manual_seed(21)
    x = {layer: inputs(generator, layer, prefix=320) for layer in (0, 1)}
    outputs = []
    try:
        for step in range(10):                      # anchor at 0 and 8, ordinary otherwise
            router.begin_step(0, step)
            for layer in (0, 1):
                outputs.append(call(router, modules[layer], x[layer]))
        torch.cuda.synchronize()
        return outputs, router.counters(), router.records(), len(router.pending), len(router.call_metadata)
    finally:
        router.close()


@pytest.mark.parametrize('variant', ['static', 'generic'])
def test_minimal_telemetry_outputs_bit_identical_and_statistics_na(variant):
    full, full_counters, full_records, full_pending, full_meta = run('full', variant)
    minimal, minimal_counters, minimal_records, pending, meta = run('minimal', variant)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(full, minimal))
    assert full_records and len(full_records) == 20 and full_pending == full_meta == 20
    assert minimal_records is None and pending == 0 and meta == 0
    assert 'N/A' in minimal_counters['per_tile_statistics']
    same = ('attention_calls', 'score_refresh_calls', 'decision_refresh_calls', 'preqk_consumer_calls',
            'summary_builds', 'summary_hits', 'summary_misses', 'score_peak_bytes', 'summary_peak_bytes',
            'current_qk_elements', 'reused_qk_elements')
    assert {k: full_counters[k] for k in same} == {k: minimal_counters[k] for k in same}
    assert minimal_counters['summary_hits'] > 0 and minimal_counters['score_refresh_calls'] == 4


def test_minimal_telemetry_keeps_every_guard(monkeypatch):
    calls = {'full': 0, 'minimal': 0}
    original = torch._assert_async
    for mode in ('full', 'minimal'):
        def counting(*args, mode=mode, **kwargs):
            calls[mode] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(torch, '_assert_async', counting)
        run(mode, 'static')
    monkeypatch.setattr(torch, '_assert_async', original)
    assert calls['full'] == calls['minimal'] > 0
