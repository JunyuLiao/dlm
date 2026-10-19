from types import SimpleNamespace

import pytest
import torch

from scripts.v31_humaneval_cfgdet_worker import bind,CONTROL
from scripts.v31_humaneval_cfgdet_study import entry,schedule


def test_cpu_emission_intervals_preserve_uncommitted_tail_without_gpu_work():
    from scripts.v31_humaneval_cfgdet_base_worker import NativeEmissionIntervals
    trace=NativeEmissionIntervals()
    trace.observe(0,0)
    trace.observe(256,4)
    trace.observe(256,5)
    trace.observe(500,7)
    receipt=trace.receipt(9,500,2)
    assert receipt['native_interval_trace_available']
    assert receipt['native_interval_steps']==[3,2,2]
    assert sum(receipt['native_interval_steps'])==9-2


def test_coalesced_or_non_cumulative_emissions_are_not_fabricated_as_canvases():
    from scripts.v31_humaneval_cfgdet_base_worker import NativeEmissionIntervals
    trace=NativeEmissionIntervals();trace.observe(512,7)
    assert trace.receipt(7,512,2)['native_interval_trace_available'] is False


@pytest.mark.parametrize('tokens',[128,256])
@pytest.mark.parametrize('prefix_tiles',[0,1,2,3,7])
def test_fixed_budget_is_per_query_head_tile_and_excludes_dense_tail(tokens,prefix_tiles):
    selected=min(tokens//64,prefix_tiles)
    kept=torch.ones((1,4,2,prefix_tiles+5),dtype=torch.bool)
    kept[...,:prefix_tiles]=False
    kept[...,:selected]=True
    adapter=SimpleNamespace(mage_k=tokens,value_audit=True,begin_request=lambda:None,
        _mage=None,_value_receipt=lambda:{},_select_initial_refresh=lambda *args:(None,kept))
    bind(adapter,SimpleNamespace(mode=None),None,lambda:{},[entry(CONTROL)])
    adapter.begin_request()
    adapter._select_initial_refresh(29,None,None,None,64*prefix_tiles+31,256)
    receipt=adapter._value_receipt()
    assert receipt['humaneval_prefix_budget_tokens']==tokens
    assert receipt['humaneval_budget_policy']==f'fixed_prefix_k{tokens//64}'
    assert receipt['humaneval_fixed_budget_units']==8
    assert receipt['humaneval_fixed_budget_errors']==0
    if selected:
        # One broken query tile must be detected even if aggregate work looks
        # almost unchanged and the mandatory dense tail remains fully kept.
        kept[0,0,0,selected-1]=False
        adapter._select_initial_refresh(29,None,None,None,64*prefix_tiles+31,256)
        assert adapter._value_receipt()['humaneval_fixed_budget_errors']==1


def test_qualification_prefix_matches_primary_order_before_extra_seed():
    cohort=[dict(index=i,seed=42) for i in [0,23,129,106,145,116]]
    warm=cohort[2:4]
    extra=[dict(c,seed=43) for c in cohort]
    remaining=[dict(index=i,seed=42) for i in range(164) if i not in {c['index'] for c in cohort}]
    for pass_name in ('clean','audit'):
        for family in ('shared','dense_full_fix51994','allkept_fa4'):
            q,qa,_,qw=schedule(cohort+extra,warm,family,pass_name,.2)
            primary,pa,_,pw=schedule(cohort+remaining[:76],warm,family,pass_name,.2)
            # Compare the complete common request prefix including implicit and
            # explicit warmups and per-method dispatch, before seed43 begins.
            count=qw+len(cohort)*(6 if family=='shared' else 1)
            assert qw==pw and q[:count]==primary[:count]
            if family=='shared':assert qa[:count+1]==pa[:count+1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='qualified H100 required')
@pytest.mark.parametrize('prefix_tiles,budget,dim',[(1,1,512),(2,2,512),(3,2,512),
                                                  (5,2,512),(8,2,512),(13,2,512),
                                                  (129,2,512),(8,4,512)])
def test_k2_full_value_swaps_match_independent_same_state_reference(prefix_tiles,budget,dim):
    from experiments.numerical_qk_reuse.v31_value_decision import moments,bounded_proposal,fixed_topk,mass_score
    from experiments.numerical_qk_reuse.v31_value_fast import bounded_proposal_device
    generator=torch.Generator(device='cuda').manual_seed(41+prefix_tiles)
    q=torch.randn((17,dim),generator=generator,device='cuda')
    k=torch.randn((64*prefix_tiles+256,dim),generator=generator,device='cuda')
    v=torch.randn(k.shape,generator=generator,device='cuda')
    stats=moments(q,k,v,prefix_tiles*64,dim**-.5)
    anchor=fixed_topk(mass_score(stats).roll(2),budget)
    expected,info=bounded_proposal(stats,anchor,swaps=8)
    actual,fast=bounded_proposal_device(stats,anchor,budget)
    assert torch.equal(expected,actual)
    assert int(actual.sum())==budget and int(fast['swaps'])==info['swaps']
