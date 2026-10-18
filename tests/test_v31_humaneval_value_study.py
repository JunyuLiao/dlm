from types import SimpleNamespace
import pytest
import torch
from scripts import v31_humaneval_value_study as study
from scripts.v31_humaneval_value_worker import bind, CONTROL, V2, V3A


def row(index, eligible, kept):
    return dict(index=index, receipts=dict(adapter=dict(value_canvas_phase_tiles={
        '1': {'held': dict(eligible_tiles=eligible, kept_tiles=kept)},
        '2': {'held': dict(eligible_tiles=10000, kept_tiles=0)}})))


def test_first_canvas_matching_ignores_later_work_and_answer_quality():
    reference = study.work([row(0, 100, 90), row(1, 200, 170)])
    assert reference['kept_fraction'] == pytest.approx(260/300)
    candidates = {.01: study.work([row(0, 200, 180)]),
                  .03: study.work([row(0, 200, 173)]),
                  .1: study.work([row(0, 200, 176)])}
    assert study.choose_threshold(reference, candidates) == .03
    assert study.match(reference, candidates[.03])['matched']
    assert not study.match(reference, candidates[.01])['matched']


def test_schedule_covers_requested_methods_and_keeps_control_bookends():
    cells = [dict(index=i, seed=42) for i in range(3)]
    a, arms, groups, warm = study.schedule(cells, cells[:2], 'shared', 'clean', .03)
    _, audit, _, _ = study.schedule(cells, cells[:2], 'shared', 'audit', .03)
    assert warm == 8 and len(a) == 26 and len(arms) == 27
    for g in groups:
        first, last = g['start']+1, g['end']
        assert arms[first]['label'] == 'control_before'
        assert arms[last]['label'] == 'control_after'
        assert [x['arm'] for x in arms[first+1:last]] == [x['arm'] for x in audit[first+1:last]][::-1]
        assert {x['arm'] for x in arms[first+1:last]} == set(study.CANDIDATES)


def test_request_switching_resets_value_mode_and_keeps_native_control():
    runtime = SimpleNamespace(mode=None)
    calls = []
    adapter = SimpleNamespace(mage_k=256, value_audit=False,
        begin_request=lambda: calls.append('native reset'),
        _mage='refined', _value_receipt=lambda: {'refined': True},
        _select_initial_refresh=lambda *a: None)
    original = lambda: {'original': True}
    schedule = [study.entry(CONTROL), study.entry(V2, .03), study.entry('full_swap8'), study.entry(V3A), study.entry(CONTROL)]
    bind(adapter, runtime, 'original', original, schedule)
    for i, e in enumerate(schedule):
        adapter.begin_request()
        assert adapter.value_selector == (e['arm'] if e['arm'] in (V2, V3A) else None)
        assert adapter._mage == ('refined' if e['arm'] in study.REFINEMENTS else 'original')
        assert adapter._value_receipt()['humaneval_live_arm'] == e['arm']
    assert len(calls) == 5 and adapter.value_threshold is None
    with pytest.raises(RuntimeError, match='frozen schedule'):
        adapter.begin_request()


def test_non_thinking_without_channel_is_scored_and_thinking_requires_final():
    text = '```python\ndef twice(x):\n    return 2*x\n```<turn|>'
    assert study.extract(text, 'twice', False)[1] == 'parsed'
    assert study.extract(text, 'twice', True)[0] is None
    assert study.extract('<channel|>'+text, 'twice', True)[1] == 'parsed'


def test_length_strata_weight_full_population_without_duplicate_items():
    manifest = [dict(prompt_tokens=[0]*(80+i)) for i in range(164)]
    indices, weights = study.calibration_indices(manifest)
    assert len(set(indices)) == 24 and sum(weights.values()) == 164


def test_calibration_refines_work_transition_instead_of_dense_plateau():
    ref = study.work([row(0, 100, 99)])
    points = {t: study.work([row(0, 100, k)]) for t, k in [(0., 100), (.01, 100), (.1, 100), (.3, 80)]}
    assert study.choose_threshold(ref, points) == 0.
    assert all(.1 < t < .3 for t in study.refinement_points(ref, points))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='qualified H100 required')
@pytest.mark.parametrize('prefix_tiles,budget', [(3, 3), (5, 4), (8, 4), (13, 4)])
def test_short_context_k4_swaps_match_independent_reference(prefix_tiles, budget):
    from experiments.numerical_qk_reuse.v31_value_decision import moments, bounded_proposal, fixed_topk, mass_score
    from experiments.numerical_qk_reuse.v31_value_fast import bounded_proposal_device
    g = torch.Generator(device='cuda').manual_seed(prefix_tiles)
    q = torch.randn((17, 32), generator=g, device='cuda')
    k = torch.randn((64*prefix_tiles+256, 32), generator=g, device='cuda')
    v = torch.randn(k.shape, generator=g, device='cuda')
    m = moments(q, k, v, prefix_tiles*64, 32**-.5)
    anchor = fixed_topk(mass_score(m).roll(2), budget)
    expected, info = bounded_proposal(m, anchor, swaps=8)
    actual, fast = bounded_proposal_device(m, anchor, budget)
    assert torch.equal(expected, actual)
    assert int(actual.sum()) == budget and int(fast['swaps']) == info['swaps']
