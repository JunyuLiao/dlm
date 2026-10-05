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


@pytest.mark.parametrize('case', ['clean', 'nan_first', 'posinf_last', 'neginf_mid', 'invalid_row_last',
                                  'invalid_tile_last', 'tile_flag_absent_clean'])
def test_fused_guard_flags_match_the_separate_guard_conditions(case):
    from experiments.numerical_qk_reuse.cached_executor import guard_flags
    g = torch.Generator(device='cuda').manual_seed(3)
    out = torch.randn(1, 16, 256, 512, generator=g, device='cuda').to(torch.bfloat16)
    rows = torch.zeros(1, 16, 256, dtype=torch.bool, device='cuda')
    tiles = torch.zeros(1, 16, 2, 37, dtype=torch.bool, device='cuda')
    flat = out.view(-1)
    if case == 'nan_first':
        flat[0] = float('nan')
    elif case == 'posinf_last':
        flat[-1] = float('inf')
    elif case == 'neginf_mid':
        flat[flat.numel() // 2 + 7] = -float('inf')
    elif case == 'invalid_row_last':
        rows.view(-1)[-1] = True
    elif case == 'invalid_tile_last':
        tiles.view(-1)[-1] = True
    reference = bool(torch.isfinite(out).all() & ~rows.any() & ~tiles.any())
    ok = bool(guard_flags(out, rows, tiles if case != 'tile_flag_absent_clean' else None).all())
    assert ok == reference
    assert ok == (case in ('clean', 'tile_flag_absent_clean'))


@pytest.mark.parametrize('variant', ['static', 'generic'])
def test_lean_router_outputs_bit_identical_with_fewer_guard_chains(monkeypatch, variant):
    from tests.test_numerical_reuse_summary_memory import call, inputs, make_router
    original = torch._assert_async
    results = {}
    for mode in ('reference', 'lean'):
        count = {'n': 0}

        def counting(*args, **kwargs):
            count['n'] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(torch, '_assert_async', counting)
        kwargs = dict(telemetry='minimal', guard_mode='fused') if mode == 'lean' else {}
        router, modules, _ = make_router(kernel_variant=variant, **kwargs)
        generator = torch.Generator(device='cuda').manual_seed(33)
        x = {layer: inputs(generator, layer, prefix=320) for layer in (0, 1)}
        outs = []
        try:
            for step in range(10):
                router.begin_step(0, step)
                outs.extend(call(router, modules[layer], x[layer]) for layer in (0, 1))
            torch.cuda.synchronize()
        finally:
            router.close()
        results[mode] = (outs, count['n'])
    monkeypatch.setattr(torch, '_assert_async', original)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
               for a, b in zip(results['reference'][0], results['lean'][0]))
    assert results['lean'][1] == 20 and results['reference'][1] > results['lean'][1]
