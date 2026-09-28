"""CPU tests for v25b guards: physical score-cache accounting and strict qualification gates."""
import math

import pytest
import torch

from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
from experiments.numerical_qk_reuse.integration import Attention, PREQK_MODE


def _ident(layer, keys, queries=8, heads=4):
    return Identity(0, 0, 0, layer, 1, heads, 2, queries, keys, 16, 100, 0, keys, .1,
                    'torch.float32', 'cpu', (False, None, None), 1)


def _owner(storage='aligned16', max_bytes=10**9):
    owner = Attention.__new__(Attention)
    owner.route_storage, owner.output_score_precision, owner.output_mode = (
        storage, 'fp32_scores_bf16_pv', PREQK_MODE)
    owner.aligned_score_copies = owner.aligned_pad_bytes = owner.aligned_sketch_pads = 0
    owner.aligned_extra_pad_bytes = owner.aligned_copy_bytes = 0
    owner.mu_mode = 'exact'
    owner.peak_score_transient_bytes = owner.peak_score_physical_bytes = 0
    owner.cache = ScoreCache(8, 3, max_bytes)
    return owner


def test_physical_counts_padded_pitch_logical_does_not():
    owner = _owner()
    ident = _ident(5, 37)
    stored = owner._store_scores(torch.randn(1, 4, 8, 37))
    owner.cache.publish_scores(ident, 0, stored)
    assert owner.cache.storage_bytes == 1 * 4 * 8 * 37 * 4
    assert owner.cache.physical_bytes == 1 * 4 * 8 * 48 * 4
    assert owner.aligned_extra_pad_bytes == 1 * 4 * 8 * 11 * 4
    assert owner.aligned_copy_bytes == 1 * 4 * 8 * 37 * 4
    assert owner.aligned_pad_bytes == owner.cache.physical_bytes


def test_aligned_k_multilayer_replacement_shared_and_release():
    owner = _owner()
    a = owner._store_scores(torch.randn(1, 4, 8, 64))   # already aligned: no copy
    owner.cache.publish_scores(_ident(5, 64), 0, a)
    owner.cache.publish_scores(_ident(11, 64), 0, a)     # same storage under two layers counts once
    assert owner.cache.physical_bytes == a.numel() * 4 and owner.aligned_score_copies == 0
    b = owner._store_scores(torch.randn(1, 4, 8, 50))
    owner.cache.publish_scores(_ident(5, 50), 8, b)      # replacement of layer 5
    assert owner.cache.physical_bytes == a.numel() * 4 + b.numel() * 4
    owner.cache.clear()
    assert owner.cache.physical_bytes == 0


def test_reserve_physical_budget_boundary_before_allocation():
    need = 1 * 4 * 8 * 48 * 4
    owner = _owner(max_bytes=need)
    ident = _ident(5, 37)
    owner._reserve_observation(ident, 1, 4, 8, 37)       # exactly at the bound: allowed
    owner.cache.publish_scores(ident, 0, owner._store_scores(torch.randn(1, 4, 8, 37)))
    with pytest.raises(MemoryError):
        owner._reserve_observation(_ident(11, 37), 1, 4, 8, 37)   # a second layer exceeds it
    owner._reserve_observation(_ident(5, 37), 1, 4, 8, 37)        # replacing layer 5 is fine


def test_transient_peak_includes_old_logical_and_padded_new():
    owner = _owner()
    ident = _ident(5, 37)
    owner.cache.publish_scores(ident, 0, owner._store_scores(torch.randn(1, 4, 8, 37)))
    owner._reserve_observation(ident, 1, 4, 8, 37)
    old, logical, padded = 4 * 8 * 48 * 4, 4 * 8 * 37 * 4, 4 * 8 * 48 * 4
    assert owner.peak_score_transient_bytes == old + logical + padded
    logical_owner = _owner('logical')
    logical_owner._reserve_observation(ident, 1, 4, 8, 37)
    assert logical_owner.peak_score_transient_bytes == logical


capture = pytest.importorskip('scripts.v21b_geometry_capture')
G = capture.GLOBAL_LAYERS


def _record(idx, phase, *, layers=G, summaries=(), flip=None, logit_shift=0.):
    logits = torch.zeros(1, 4, 6)
    logits[..., 0] += logit_shift
    decisions = {l: (torch.zeros(1, 2, 2, 3, dtype=torch.bool), torch.ones(1, 2, 2, 3, dtype=torch.bool))
                 for l in layers}
    if flip is not None:
        decisions[flip][0][0, 0, 0, 0] = True
    return dict(call_index=idx, phase=phase, logits=logits, argmax=logits.argmax(-1),
                decisions=decisions, clocks={l: (1, 1) for l in layers},
                summaries={l: (torch.zeros(3), torch.zeros(3, 2)) for l in summaries})


def test_b0_empty_sets_are_the_only_legal_empty_case():
    ok = capture._compare_call(_record(0, 'B0', layers=()), _record(0, 'B0', layers=()), expect_summaries=True)
    assert ok['level2_failures'] == []
    bad = capture._compare_call(_record(2, 'H', layers=()), _record(2, 'H', layers=()), expect_summaries=True)
    assert 'decisions_layer_set' in bad['level2_failures']


@pytest.mark.parametrize('cand,failure', [
    (dict(layers=G[:-1]), 'decisions_layer_set'),
    (dict(flip=5), 'decisions'),
    (dict(logit_shift=1e-3), 'logits'),
])
def test_missing_layer_flipped_bit_and_logit_change_fail(cand, failure):
    result = capture._compare_call(_record(3, 'H'), _record(3, 'H', **cand), expect_summaries=False)
    assert failure in result['level2_failures']


def test_phase_mismatch_and_missing_summary_layer_fail():
    phase = capture._compare_call(_record(4, 'D'), _record(4, 'H'), expect_summaries=False)
    assert 'call_index_or_phase' in phase['level2_failures']
    summ = capture._compare_call(_record(1, 'BO', summaries=G), _record(1, 'BO', summaries=G[:-1]),
                                 expect_summaries=True)
    assert 'summary_layer_set' in summ['level2_failures']


def test_sequence_verdict_rejects_missing_call_and_passes_clean_run():
    rows = [capture._compare_call(_record(i, 'H'), _record(i, 'H'), expect_summaries=False) for i in range(2, 5)]
    parent = [dict(call_index=i, phase='H') for i in range(2, 5)]
    good = capture._sequence_verdict(parent, rows, expected_calls=3)
    assert good['level2_functional_identity'] and good['pilot_eligible'] and good['executed']
    short = capture._sequence_verdict(parent, rows[:2], expected_calls=3)
    assert not short['level2_functional_identity'] and 'call_count' in short['level2_failures']
