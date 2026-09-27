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
