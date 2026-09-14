import pytest
import torch

from dllm.attention.blasst.core import Blasst2DConfig, apply_blasst_2d, slow_blasst_2d


def test_endpoint_retains_records_and_ties_and_matches_literal_loop():
    scores = torch.tensor([[[[1., 1., 2., 2., 2., 2., 0., 0.],
                              [2., 2., 1., 1., 2., 2., 0., 0.]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    config = Blasst2DConfig(blasst_lambda=1., local_blasst_lambda=1., global_blasst_lambda=1., q_tile_size=2, kv_tile_size=2)
    fast, decisions = apply_blasst_2d(scores, valid, None, config)
    slow, reference = slow_blasst_2d(scores, valid, None, config)
    assert torch.equal(fast, slow)
    assert torch.equal(decisions.skip_mask, reference.skip_mask)
    assert decisions.skip_mask.tolist() == [[[[False, False, False, True]]]]
    assert decisions.row_skip_mask.tolist() == [[[[False, False, False, True], [False, True, False, True]]]]
    assert config.lambda_for("local", 0) == config.lambda_for("global", 0) == 1.


@pytest.mark.parametrize("value", [0., -1., 1.000001, float("nan")])
def test_endpoint_rejects_out_of_domain(value):
    with pytest.raises(ValueError):
        Blasst2DConfig(blasst_lambda=value)


from experiments.diffusion_gemma_blasst_diagnosis.diagnostics import (
    online_mask, gaussian_mask, group_counts, mask_metrics, select_rows,
)


def test_online_order_ties_and_updated_max_equivalence():
    maxima = torch.tensor([[1., 3., 2., 3., -torch.inf], [2., 1., 2., 0., 4.]])
    valid = torch.isfinite(maxima)
    for lam in (.001, .5, .9999, 1.):
        keep, _ = online_mask(maxima, valid, lam)
        updated = maxima - maxima.cummax(-1).values
        assert torch.equal(keep, valid & (updated >= __import__('math').log(lam)))
    forward, _ = online_mask(maxima, valid)
    reverse, _ = online_mask(maxima, valid, order='reverse')
    assert not torch.equal(forward, reverse)
    assert forward[0].tolist() == [True, True, False, True, False]


def test_physical_closure_recovers_mass_without_changing_tiles():
    counts = torch.ones(2, 3, dtype=torch.long)
    mass = torch.tensor([[.1,.6,.3],[.2,.1,.7]])
    contributions = mass[...,None] * torch.tensor([[[1.],[2.],[3.]]])
    keep = torch.tensor([[True,False,True],[False,True,False]])
    raw = mask_metrics(keep,counts,mass,contributions)
    closed = mask_metrics(keep.any(0)[None,:].expand_as(keep),counts,mass,contributions)
    assert raw['skipped_tiles'] == closed['skipped_tiles'] == 0
    assert closed['retained_mass_sum'] > raw['retained_mass_sum']
    assert closed['output_squared_error'] == 0


def test_regrouping_fixed_row_mask_changes_physical_union():
    keep = torch.tensor([[True,False],[False,True],[True,False],[False,True]])
    valid = torch.ones_like(keep)
    original = group_counts(keep,valid,size=2)
    regrouped = group_counts(keep,valid,size=2,order=torch.tensor([0,2,1,3]))
    assert original['eligible_tiles'] == regrouped['eligible_tiles'] == 4
    assert original['skipped_tiles'] == 0 and regrouped['skipped_tiles'] == 2


def test_gaussian_and_frozen_disjoint_diagnostic_selection():
    eligible=torch.ones(3,dtype=torch.bool)
    keep,degenerate,fallback=gaussian_mask(torch.tensor([-1.,0.,1.]),eligible,0.)
    assert keep.tolist()==[False,True,True]
    assert not degenerate and not fallback
    assert gaussian_mask(torch.ones(3),eligible,1.)[1]
    assert gaussian_mask(torch.arange(3.).float(),eligible,10.)[2]
    _,a=select_rows(); _,b=select_rows()
    assert a==b and len(a)==16
    assert sum(r['diagnostic_split']=='final' for r in a)==8


@pytest.mark.parametrize('order', ['forward', 'reverse'])
def test_reference_physical_mask_partial_tile_gqa_structural_mask(order):
    from types import SimpleNamespace
    from dllm.attention.blasst.core import (
        blasst_2d_attention_forward, _prepare_attention_scores, Blasst2DStats,
    )
    torch.manual_seed(3)
    module=torch.nn.Module()
    module.num_key_value_groups=2
    module.layer_idx=0
    module._blasst_2d_runtime=SimpleNamespace(
        config=Blasst2DConfig(blasst_lambda=1.,q_tile_size=2,kv_tile_size=2,collect_blasst_stats=True),
        active_query_mask=None,dense_kv_prefix_extractor=None,metadata={},stats=Blasst2DStats())
    q=torch.randn(1,2,3,4); k=torch.randn(1,1,5,4); v=torch.randn(1,1,5,4)
    mask=torch.ones(1,1,3,5,dtype=torch.bool); mask[...,0,3:]=False
    output,_=blasst_2d_attention_forward(module,q,k,v,mask,blasst_tile_order=order,is_causal=False)
    _,expanded,scores,valid=_prepare_attention_scores(module,q,k,v,mask,scaling=None,is_causal=False,sliding_window=None)
    blocks=torch.nn.functional.pad(scores,(0,1),value=-torch.inf).reshape(1,2,3,3,2).max(-1).values
    allowed=torch.isfinite(blocks)
    keep,_=online_mask(blocks,allowed,order=order)
    # The execution mask is the union of retained row votes in each query
    # tile, including the partial final query tile.
    keep=torch.nn.functional.pad(keep,(0,0,0,1),value=False).reshape(1,2,2,2,3).any(-2).repeat_interleave(2,-2)[...,:3,:]
    expected=(scores.masked_fill(~keep.repeat_interleave(2,-1)[...,:5],-torch.inf).softmax(-1) @ expanded).transpose(1,2)
    assert torch.equal(output,expected)
    assert module._blasst_2d_runtime.stats.summary()['eligible_tiles']>0
    probabilities=scores.softmax(-1)
    expected_mass=(probabilities*keep.repeat_interleave(2,-1)[...,:5]).sum(-1).mean()
    stats=module._blasst_2d_runtime.stats.summary()
    assert stats['retained_dense_attention_mass']==pytest.approx(float(expected_mass))
    element_skip=~keep.repeat_interleave(2,-1)[...,:5] & valid
    assert stats['skipped_valid_elements']==int(element_skip.sum())


def test_observer_end_to_end_compact_shard(tmp_path):
    from types import SimpleNamespace
    from experiments.diffusion_gemma_blasst_diagnosis.diagnostics import Observer
    module=torch.nn.Module(); module.layer_idx=0; module.num_key_value_groups=2
    module._blasst_2d_runtime=SimpleNamespace(metadata={'example_id':'fake'})
    observer=Observer(tmp_path,[{'sample_id':'fake','diagnostic_split':'calibration'}])
    q=torch.randn(1,2,64,8); k=torch.randn(1,1,129,8); v=torch.randn_like(k)
    original=q.clone()
    for _ in range(2):
        observer(module,q,k,v,is_causal=False)
    observer.flush()
    assert torch.equal(q,original)
    assert len(observer.records)==2
    assert observer.records[1]['reuse']['step_sol_50']['union']>0
    assert (tmp_path/'shards/fake/snapshots.npz').exists()
    assert all(p['empty_rows']==0 for p in observer.records[0]['policies'].values())
