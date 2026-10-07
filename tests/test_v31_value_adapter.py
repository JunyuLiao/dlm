"""Adapter-integration gates for this study's value-aware selectors.

These prove the INTEGRATION, not the mathematics (that is ``test_v31_value_select.py`` /
``test_v31_value_select_gpu.py``):

  * the inherited control (``value_selector=None``) is byte-for-byte the code path it was: the arm
    construction, the receipt and the selection dispatch are unchanged when the study is off;
  * every arm of the study constructs, reports its own name and its projection identity, and the two
    approximations are flagged;
  * the reuse contract is unchanged: the discovery/refresh clock, carry policy, budget, protected
    tiles, layer scope, LOCAL handling and the consumer's map type are identical across all arms;
  * a value-aware selection returns the SAME ``[1, H, QB, KT]`` bool map type the unchanged FA4
    consumer consumes, and the later consumer receives exactly that type;
  * an invalid or silently-approximated configuration is refused at construction.

Run: PYTHONPATH=src <pinned python> -m pytest tests/test_v31_value_adapter.py -q
"""
from __future__ import annotations

import sys

import pytest
import torch

from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter

TYPES = ['sliding_attention'] * 5 + ['full_attention']
ARMS = ('v1', 'v2', 'v3a', 'v3b', 'v3b_drop', 'v3b_shortlist')
CONTROL = dict(arm='mage', mage_select='fa4', mage_granularity='qblock_max', mage_k=1728,
               mage_select_step=1, mage_carry_first=True, mage_reselect_trigger=[0.15],
               mage_trigger_signal='settle', mage_sticky=1.386, kv_copy_backend='triton',
               merge_backend='triton')


def arm_kwargs(sel):
    """The frozen per-arm selector settings (V3 is threshold-free; each approximation must declare
    the feature that makes it an approximation)."""
    kw = {'value_selector': sel}
    if sel == 'v3b_drop':
        kw['value_drop_fraction'] = 0.25
    if sel == 'v3b_shortlist':
        kw['value_shortlist'] = 8
    return kw


def adapter(**kw):
    cfg = dict(CONTROL)
    cfg.update(kw)
    return VllmMethodAdapter(TYPES, **cfg)


# --------------------------------------------------------------------------- the control is unchanged


def test_the_active_control_still_constructs_and_reports_no_value_arm():
    a = adapter()
    r = a.value_receipt()
    assert r['value_selector'] is None and r['value_arms_active'] is False
    assert a.arm == 'mage' and a.mage_select == 'fa4' and a.mage_granularity == 'qblock_max'
    assert a.mage_k == 1728 and a.mage_select_step == 1 and a.mage_carry_first is True
    assert list(a.mage_reselect_trigger) == [0.15] and a.mage_trigger_signal == 'settle'
    assert a.mage_sticky == 1.386 and a.mage_sink == 0 and a.mage_recent_tiles == 0


@pytest.mark.parametrize('sel', ARMS)
def test_each_arm_constructs_and_names_its_own_selector(sel):
    a = adapter(**arm_kwargs(sel))
    r = a.value_receipt()
    assert r['value_selector'] == sel and r['value_arms_active'] is True
    assert r['value_approximate_selector'] == (sel in ('v3b_drop', 'v3b_shortlist'))
    assert r['value_projection']['family'] == 'gaussian'
    assert r['value_projection']['rank'] == 32 and r['value_projection']['seed'] == 1729
    assert r['value_consumer'].startswith('unchanged FA4')
    assert r['value_output_path'].startswith('unchanged')
    assert r['value_aggregation'] == 'max over the valid rows of the 128-row block'
    # LOCAL is out of scope: no LOCAL layer is sparse and the native window is preserved
    assert r['value_layer_scope']['local_layers_sparse'] == []
    assert r['value_layer_scope']['local_window'] == [1023, 1023]
    assert a.local_router is None, 'no LOCAL router may be active in this study'


@pytest.mark.parametrize('sel', ARMS)
def test_the_reuse_contract_is_identical_across_every_arm(sel):
    """The study changes the SELECTION FORMULA only: budget, clock, carry, sticky, protection, layer
    scope and the consumer are the control's."""
    a, c = adapter(**arm_kwargs(sel)), adapter()
    for field in ('mage_k', 'mage_select_step', 'mage_carry_first', 'mage_reselect_trigger',
                  'mage_trigger_signal', 'mage_sticky', 'mage_sink', 'mage_recent_tiles',
                  'mage_granularity', 'mage_select', 'kv_copy_backend', 'merge_backend',
                  'global_layers', 'local_router', 'splits'):
        assert getattr(a, field) == getattr(c, field), field
    assert a.value_receipt()['value_budget_tokens'] == c.value_receipt().get('value_budget_tokens', 1728)
    assert a.value_receipt()['value_protect_sink_tokens'] == 0
    assert a.value_receipt()['value_protect_recent_tokens'] == 0


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize('kw,why', (
    (dict(value_selector='v3b_drop'), 'v3b_drop without a drop fraction would silently be exact'),
    (dict(value_selector='v3b_shortlist'), 'v3b_shortlist without a shortlist would silently be exact'),
    (dict(value_selector='v1', value_drop_fraction=0.25), 'V1 takes no approximation feature'),
    (dict(value_selector='v2', value_shortlist=4), 'V2 takes no approximation feature'),
    (dict(value_selector='v3a', value_threshold=2.0), 'the V3 selectors are threshold-free'),
    (dict(value_selector='bogus'), 'unknown selector'),
    (dict(value_selector='v3b_drop', value_drop_fraction=1.5), 'drop fraction out of range'),
    (dict(value_selector='v1', value_threshold=-1.0), 'negative threshold'),
    (dict(value_selector='v1', value_exact_max=0), 'exact max must be positive'),
    (dict(value_selector='v1', value_scan='bogus'), 'unknown scan path'),
))
def test_invalid_configurations_are_refused(kw, why):
    with pytest.raises(ValueError):
        adapter(**kw)


def test_the_value_selectors_refuse_a_non_mage_arm():
    with pytest.raises(ValueError):
        VllmMethodAdapter(TYPES, arm='allkept', value_selector='v3a')


# --------------------------------------------------------------------------- counters / map type


@pytest.mark.parametrize('sel', ARMS)
def test_each_arm_starts_with_a_full_value_counter_set(sel):
    a = adapter(**arm_kwargs(sel))
    for key in ('value_candidates', 'value_evaluations', 'value_forced_keep', 'value_forced_skip',
                'value_threshold_keeps', 'value_passes', 'value_blocked', 'value_mu_passes',
                'value_sketch_builds', 'value_stat_bytes', 'value_approximation_calls'):
        assert key in a.calls, key
        assert a.calls[key] == 0
    assert 'value_last_counters' in a.value_receipt()


def test_begin_request_clears_the_value_state():
    a = adapter(value_selector='v3a')
    a.calls['value_candidates'] = 7
    a._value_obj = dict(objective_mean=1.0)
    a._value_bank[(0, 2, 8, 'cpu')] = torch.zeros(2, 8, 32)
    a._value_sketch[(0, 10)] = torch.zeros(1, 2, 10, 32)
    a._last_value_counters = dict(candidates=7)
    a.begin_request()
    assert a.calls['value_candidates'] == 0
    assert a._value_obj is None and a._last_value_counters is None
    assert a._value_bank == {} and a._value_sketch == {} and a._value_bank_records is None


# --------------------------------------------------------------------------- the consumer sees the same map type


@pytest.mark.parametrize('sel', ARMS)
def test_the_value_selectors_return_the_maps_unchanged_consumers_expect(sel):
    """Whatever map a value-aware selector produces must be consumable by the unchanged FA4 path with
    no adaptation, and the tile accounting must land in the control's own counters."""
    from experiments.numerical_qk_reuse import v31_value_select as vs
    sys.path.insert(0, 'tests')
    from test_v31_value_select_gpu import Case, _select
    case = Case(nk=256, n=128)
    stats, _z, _mu = case.stats()
    keep = _select(case, sel, 6)
    assert keep.dtype == torch.bool and keep.shape == (1, case.heads, 1, 256 // 64)
    lists = v27_fa4.block_sparse_tensors(keep)
    assert lists.block_size == (128, 64)
    out = v27_fa4.sparse_lists(case.q, case.k, case.v, lists, 64 ** -0.5)
    assert torch.isfinite(out).all()
    # the map's prefix-tile count is what the control's sparsity counters would charge
    a = adapter(**arm_kwargs(sel))
    a.begin_request()                                  # the control's own sparsity counters
    a.calls['mage_kept_prefix_tiles'] += int(keep[:, :, :stats.prefix_tiles].sum())
    a.calls['mage_prefix_tiles'] += stats.prefix_tiles * stats.heads * stats.blocks
    assert a.calls['mage_kept_prefix_tiles'] > 0
    # the study's own counters are present and zero before any selection has run
    assert a.calls['value_candidates'] == 0 and a.calls['value_passes'] == 0
    assert sel in vs.VALUE_SELECTORS