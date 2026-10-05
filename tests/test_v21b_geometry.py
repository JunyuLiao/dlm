"""CPU checks for the untimed v21b geometry reference, without Triton/GPU."""

import pytest
import torch

from experiments.numerical_qk_reuse.geometry_diagnostic import (
    _group_any, _historical_mass_budget, _select, compare_geometries,
    matched_q16_screen,
)


def _inputs(*, heads=8, queries=5, keys=96, kv_heads=1):
    gen = torch.Generator().manual_seed(101)
    scores = torch.randn(1, heads, queries, keys, generator=gen)
    scores[..., 0] = float('-inf')
    projected = torch.randn(1, kv_heads, keys, 32, generator=gen)
    reference = torch.ones(1, kv_heads)
    sensitivity = torch.ones(1, queries)
    q = torch.randn(1, heads, queries, 16, generator=gen)
    k = torch.randn(1, kv_heads, keys, 16, generator=gen)
    v = torch.randn(1, kv_heads, keys, 16, generator=gen)
    legal = torch.ones(1, 1, queries, keys, dtype=torch.bool)
    legal[..., -3:] = False
    return scores, projected, reference, sensitivity, q, k, v, legal


def test_group_closure_counts_legal_extra_pairs_and_partial_keys():
    logical = torch.zeros(1, 8, 5, 3, dtype=torch.bool)
    logical[0, 0, 0, 0] = True
    logical[0, 2, 4, 2] = True
    coarse, expanded = _group_any(logical, 1, 16, 64)
    assert coarse.shape == (1, 8, 1, 2)
    assert expanded.shape == logical.shape
    assert expanded[0, 0, 4, 1]
    assert expanded[0, 2, 0, 2]
    assert not expanded[0, 1].any()


def test_mass_reference_retains_each_live_row_argmax_even_over_budget():
    scores = torch.zeros(1, 1, 2, 64)
    scores[0, 0, 0, 0] = 5
    scores[0, 0, 1, 40] = 5
    actual = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    bits, meta = _historical_mass_budget(scores, actual, 1, 2, 32)
    assert meta['groups_mandatory_exceeded_budget'] == 1
    assert bits[0, 0, 0].tolist() == [True, True]
    assert bits[0, 0, 1].tolist() == [True, True]


def test_mass_mandatory_uses_tile_mass_not_top_token():
    # Tile 0 has the largest single token, but tile 1 has more total mass.
    scores = torch.tensor([[[[3., float('-inf'), 2.5, 2.5]]]])
    actual = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
    bits, meta = _historical_mass_budget(scores, actual, 1, 1, 2)
    assert bits[0, 0, 0].tolist() == [False, True]
    assert meta['mandatory_tiles'] == 1


def test_each_geometry_reruns_retained_state_not_split_bitmap():
    scores, projected, reference, sensitivity, *_ = _inputs()
    loose = _select(scores, projected, reference, sensitivity,
                    heads=1, queries=1, keys=32, threshold=-1.)
    grouped = _select(scores, projected, reference, sensitivity,
                      heads=1, queries=16, keys=64, threshold=-1.)
    assert loose['bits'].shape == (1, 8, 5, 3)
    assert grouped['bits'].shape == (1, 8, 5, 2)
    assert torch.isfinite(loose['risk'][:, :, :, 1:]).any()
    assert grouped['torch_reference_selection_seconds'] >= 0


def test_compare_geometries_json_scalars_and_same_logical_g0():
    args = _inputs()
    out = compare_geometries(*args[:-1], scale=.25, threshold=-1.,
                             kind='GLOBAL', legal=args[-1])
    assert out['schema'] == 'v21b_geometry_diagnostic_001'
    assert out['coarse_route_only_parity']['status'] == 'not_run'
    assert set(out['independent_sequential']) == {
        'one_head_Q128_K64', 'one_head_Q16_K64', 'GLOBAL_8head_Q2_K32'}
    assert 'one_head_Q16_K32' in out['g0_common_logical']['closures']
    assert out['denominator_legal_pairs'] == 8*5*93
    assert out['masked_illegal_score_positions'] == 8*5*3
    assert out['projected_v_setup_seconds'] is None
    for case in out['independent_sequential'].values():
        assert 0 <= case['removed_mass_per_row']['mean'] <= 1
        assert case['output_vs_full_fp32']['finite']
        assert len(case['support_sha256']) == 64
    logical = out['g0_common_logical']['rowwise_kept_legal_pairs']
    for case in out['g0_common_logical']['closures'].values():
        assert case['legal_extra_pairs_over_logical'] == case['kept_legal_pairs']-logical
    for name, case in out['historical_mass_argmax_matched_tile_budget'].items():
        assert case['status'] == 'compared', name
        assert case['legal_pair_symmetric_difference'] == (
            case['legal_pairs_added_vs_value_aware'] + case['legal_pairs_removed_vs_value_aware'])


def test_illegal_scores_cannot_change_selection_or_mass_reference():
    args = list(_inputs())
    first = compare_geometries(*args[:-1], scale=.25, threshold=-1.,
                               kind='GLOBAL', legal=args[-1])
    args[0][..., -3:] = float('nan')
    second = compare_geometries(*args[:-1], scale=.25, threshold=-1.,
                                kind='GLOBAL', legal=args[-1])
    for name in first['independent_sequential']:
        assert first['independent_sequential'][name]['support_sha256'] == (
            second['independent_sequential'][name]['support_sha256'])
        assert first['historical_mass_argmax_matched_tile_budget'][name]['support_sha256'] == (
            second['historical_mass_argmax_matched_tile_budget'][name]['support_sha256'])


def test_matched_screen_uses_only_five_frozen_offsets_and_reports_work_error():
    args = _inputs(queries=33)
    screen = matched_q16_screen(*args[:-1], scale=.25, threshold=-3.18,
                                kind='GLOBAL', legal=args[-1])
    assert screen['schema'] == 'v21b_matched_q16_screen_001'
    assert screen['offsets'] == [0., -.25, -.5, -1., -2.]
    assert [row['offset'] for row in screen['q16_candidates']] == screen['offsets']
    assert screen['coarse_baseline']['bitmap_bytes'] == 8*1*2
    assert all(row['bitmap_bytes'] == 8*3*2 for row in screen['q16_candidates'])
    assert all(row['output_vs_full_fp32']['finite'] for row in screen['q16_candidates'])
    assert screen['denominator_legal_pairs'] == 8*33*93
    assert all(set(x) == {'previous_offset', 'current_offset', 'previous_pairs', 'current_pairs'}
               for x in screen['nonmonotonic_retained_work_samples'])


def test_matched_screen_rejects_local_and_illegal_score_perturbation():
    args = list(_inputs())
    with pytest.raises(ValueError, match='GLOBAL'):
        matched_q16_screen(*args[:-1], scale=.25, threshold=-3.18,
                           kind='LOCAL', legal=args[-1])
    first = matched_q16_screen(*args[:-1], scale=.25, threshold=-3.18,
                               kind='GLOBAL', legal=args[-1])
    args[0][..., -3:] = float('nan')
    second = matched_q16_screen(*args[:-1], scale=.25, threshold=-3.18,
                                kind='GLOBAL', legal=args[-1])
    assert [x['support_sha256'] for x in first['q16_candidates']] == [
        x['support_sha256'] for x in second['q16_candidates']]


def test_fail_closed_gqa_and_explicit_legality():
    args = list(_inputs())
    # Eight heads over one KV head is valid GQA.
    compare_geometries(*args[:-1], scale=.25, threshold=-1., kind='GLOBAL', legal=args[-1])
    split = list(_inputs(kv_heads=2))
    with pytest.raises(ValueError, match='within each KV head group'):
        compare_geometries(*split[:-1], scale=.25, threshold=-1.,
                           kind='GLOBAL', legal=split[-1])
    args[0] = args[0][:, :7]
    args[4] = args[4][:, :7]
    with pytest.raises(ValueError, match='GQA|8-head'):
        compare_geometries(*args[:-1], scale=.25, threshold=-1., kind='GLOBAL', legal=args[-1])
    args = list(_inputs())
    args[-1] = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match='broadcast'):
        compare_geometries(*args[:-1], scale=.25, threshold=-1., kind='GLOBAL', legal=args[-1])


def test_bootstrap_none_sensitivity_equals_explicit_ones_and_live_t_is_used():
    # At threshold .5 neutral T drops tiles while this live T keeps them all.
    args = list(_inputs(queries=33))
    ones = compare_geometries(*args[:-1], scale=.25, threshold=.5,
                              kind='GLOBAL', legal=args[-1])
    args[3] = None
    none = compare_geometries(*args[:-1], scale=.25, threshold=.5,
                              kind='GLOBAL', legal=args[-1])
    live = list(_inputs(queries=33))
    live[3] = torch.linspace(.05, 20., 33).reshape(1, 33)
    varied = compare_geometries(*live[:-1], scale=.25, threshold=.5,
                                kind='GLOBAL', legal=live[-1])
    hashes = lambda out: {k: v['support_sha256'] for k, v in out['independent_sequential'].items()}
    assert hashes(ones) == hashes(none)
    assert hashes(varied) != hashes(ones)
    screen_ones = matched_q16_screen(*_inputs(queries=33)[:-1], scale=.25, threshold=.5,
                                     kind='GLOBAL', legal=args[-1])
    screen_none = matched_q16_screen(*args[:-1], scale=.25, threshold=.5,
                                     kind='GLOBAL', legal=args[-1])
    assert [x['support_sha256'] for x in screen_ones['q16_candidates']] == [
        x['support_sha256'] for x in screen_none['q16_candidates']]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='capture timing helper synchronizes CUDA')
def test_capture_maps_only_absent_t_to_ones_and_passes_live_t_through():
    pytest.importorskip('dllm.attention.blasst.core')
    from types import SimpleNamespace
    from scripts import v21b_geometry_capture as capture
    scores, _, _, _, q, k, v, _ = _inputs(heads=8, queries=33, keys=96)
    matrix = torch.randn(16, 32, generator=torch.Generator().manual_seed(7))
    owner = SimpleNamespace(
        thresholds={'global': {'log_threshold': .5}},
        projections=SimpleNamespace(get=lambda *a, **kw: matrix))
    def record(sensitivity):
        return dict(module=SimpleNamespace(layer_idx=5), q=q, k=k, v=v, args=(),
                    kwargs=dict(attention_mask=None, is_causal=False, scaling=.25),
                    scores=scores.clone(), sensitivity=sensitivity, production_norm=None)
    absent = capture._geometry(owner, record(None), 5)
    explicit = capture._geometry(owner, record(torch.ones(1, 33)), 5)
    live_t = torch.linspace(.05, 20., 33).reshape(1, 33)
    live = capture._geometry(owner, record(live_t), 5)
    assert absent['query_sensitivity'] == 'production_none_means_neutral_ones'
    assert explicit['query_sensitivity'] == live['query_sensitivity'] == 'live_T'
    hashes = lambda out: {k: v['support_sha256']
                          for k, v in out['comparison']['independent_sequential'].items()}
    assert hashes(absent) == hashes(explicit)
    assert hashes(live) != hashes(absent)


def test_completeness_rejects_zero_exit_with_failed_layers():
    from scripts.v21b_geometry_capture import completeness
    ok_call = {'layers': {'0': {'status': 'qualified'}, '5': {'status': 'qualified',
               'geometry': {'comparison': {'matched_q16_calibration': {
                   'q16_candidates': [{}]*5}}}}}}
    report = dict(errors=[], groups={
        'longbench_v2|x|canvas0': dict(dataset='longbench_v2', status='complete',
                                       selected_calls=[0, 3], missing_requested_calls=[],
                                       calls={'0': ok_call, '3': ok_call}),
        'ruler4k|y|canvas0': dict(dataset='ruler4k', status='missing_canvas',
                                  selected_calls=[], missing_requested_calls=[0], calls={})})
    verdict = completeness(report, bootstrap_calibration=True)
    assert verdict['status'] == 'complete'
    assert verdict['native_unreachable'] == ['ruler4k|y|canvas0:0']
    report['groups']['longbench_v2|x|canvas0']['calls']['0'] = {
        'layers': {'0': {'status': 'failed'}, '5': {'status': 'qualified'}}}
    report['errors'].append({'group': 'longbench_v2|x|canvas0'})
    verdict = completeness(report, bootstrap_calibration=True)
    assert verdict['status'] == 'incomplete'
    assert verdict['failed_layers'] == ['longbench_v2|x|canvas0:0:0']


def test_validation_completeness_requires_every_screen_and_share():
    from scripts.v21b_geometry_capture import completeness
    screen = {'status': 'qualified', 'geometry': {'comparison': {'matched_q16_calibration': {
        'q16_candidates': [{}]*5}}}}
    group = dict(dataset='aime26', status='complete', selected_calls=[0, 4],
                 missing_requested_calls=[], calls={'0': {'layers': {'5': screen}},
                                                    '4': {'layers': {'5': screen}}},
                 attention_share={'0': {'status': 'qualified'}, '4': {'status': 'qualified'}})
    report = dict(errors=[], groups={'aime26|a|canvas0': group})
    assert completeness(report, bootstrap_calibration=False, q16_validation=True)['status'] == 'complete'
    group['attention_share']['4'] = {'status': 'failed'}
    verdict = completeness(report, bootstrap_calibration=False, q16_validation=True)
    assert verdict['status'] == 'incomplete'
    assert verdict['failed_layers'] == ['aime26|a|canvas0:4:attention_share']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA event timing')
def test_attention_share_oracles_visit_every_layer_and_zero_only_dropped_kind():
    pytest.importorskip('triton')
    from types import SimpleNamespace
    from scripts import v21b_geometry_capture as capture
    modules = [SimpleNamespace(layer_idx=i, is_sliding=(i % 6 != 5)) for i in range(30)]
    registry = {}
    def native(module, q, k, v, *args, **kwargs):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return (out.transpose(1, 2).contiguous(), None)
    registry['sdpa'] = native
    seen = []
    class Model:
        def modules(self):
            return modules + modules  # encoder and decoder share indices
        def forward(self, decoder_input_ids, **kwargs):
            total = torch.zeros((), device='cuda')
            for module in modules:
                out = registry['sdpa'](module, decoder_input_ids, decoder_input_ids, decoder_input_ids)[0]
                seen.append((module.layer_idx, bool(out.abs().sum() > 0)))
                total = total + out.float().sum()
            return SimpleNamespace(logits=total.reshape(1))
    x = torch.randn(1, 2, 16, 8, device='cuda', generator=torch.Generator('cuda').manual_seed(3))
    snapshot = SimpleNamespace(prepare=lambda controller=None: dict(
        current_canvas=x, self_conditioning_logits=None, mask_mapping=None,
        past_key_values=None, decoder_position_ids=None))
    model = Model()
    kinds = capture._layer_kinds(model)
    result = capture._attention_share(model, snapshot, registry, native, kinds, reps=2)
    assert registry['sdpa'] is native
    assert result['status'] == 'qualified'
    assert {r['attention_calls'] for r in result['samples']['native']} == {30}
    digests = {name: {r['output_digest'] for r in rows} for name, rows in result['samples'].items()}
    assert all(len(d) == 1 for d in digests.values())
    assert len({next(iter(d)) for d in digests.values()}) == 4
    seen.clear()
    registry['sdpa'] = native
    result = capture._attention_share(model, snapshot, registry, native, kinds, reps=1)
    assert result['summary']['native']['GLOBAL_attention_in_forward_median_ms'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton qualification host')
def test_actual_coarse_route_only_bitmap_parity_cuda():
    pytest.importorskip('triton')
    args = _inputs(heads=8, queries=256, keys=128)
    args = tuple(x.cuda() for x in args)
    out = compare_geometries(*args[:-1], scale=.25, threshold=3.,
                             kind='GLOBAL', legal=args[-1])
    parity = out['coarse_route_only_parity']
    assert parity['status'] == 'match', parity
    assert parity['kept_mismatch_tiles'] == 0
    assert parity['eligible_mismatch_tiles'] == 0


def test_grouped_q_producer_preserves_gqa_head_order_and_mask():
    pytest.importorskip('dllm.attention.blasst.core')
    from experiments.numerical_qk_reuse.integration import Attention
    gen = torch.Generator().manual_seed(23)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Small integers make every product/sum exact, so any mismatch is a
    # head/row mapping error rather than GEMM rounding.
    q = torch.randint(-3, 4, (1, 8, 5, 16), generator=gen).to(device, torch.bfloat16)
    q = q.transpose(1, 2).contiguous().transpose(1, 2)        # native non-contiguous view
    k = torch.randint(-3, 4, (1, 2, 70, 16), generator=gen).to(device, torch.bfloat16)
    legacy = Attention.observe_scores(q, k, None, .25, False, None, 0)
    grouped = Attention.observe_scores_grouped(q, k, None, .25, False, None, 0)
    assert grouped.shape == legacy.shape == (1, 8, 5, 70)
    assert grouped.dtype == legacy.dtype == torch.float32 and grouped.is_contiguous()
    assert torch.equal(legacy, grouped)
