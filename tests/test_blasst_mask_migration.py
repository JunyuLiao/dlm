"""Whole-tile execution, offline metrics, and legacy-shard safety."""
import json
from types import SimpleNamespace

import pytest
import torch

from dllm.attention.blasst import (
    BLASST_MASK_SEMANTICS, Blasst2DConfig, Blasst2DRuntime, Blasst2DStats,
    validate_blasst_output_directory,
)
from dllm.attention.blasst.core import blasst_2d_attention_forward, apply_blasst_2d
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.routing import blasst_route
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.calibration import (
    evaluate_margin_trace, evaluate_lambda_grid, evaluate_lambda_grid_jsonl,
    evaluate_lambda_grid_jsonl_by_type,
)


@pytest.mark.parametrize('order', ['forward', 'reverse'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_inactive_row_losing_only_valid_key_returns_zero(order, dtype):
    config=Blasst2DConfig(blasst_lambda=.5,q_tile_size=2,kv_tile_size=1,collect_blasst_stats=True)
    runtime=Blasst2DRuntime(config)
    runtime.active_query_mask=torch.tensor([[True,False]])
    module=SimpleNamespace(_blasst_2d_runtime=runtime,num_key_value_groups=1,layer_idx=0,training=False)
    q=torch.eye(2,dtype=dtype).reshape(1,1,2,2)
    k=torch.tensor([[[[5.,0.],[0.,4.]]]],dtype=dtype)
    v=torch.tensor([[[[1.],[2.]]]],dtype=dtype)
    mask=torch.tensor([[[[True,True],[False,True]]]])
    if order=='reverse':
        k,v,mask=k.flip(-2),v.flip(-2),mask.flip(-1)
    result,_=blasst_2d_attention_forward(module,q,k,v,mask,scaling=1.,is_causal=False,blasst_tile_order=order)
    assert torch.isfinite(result).all()
    assert result[0,0,0,0]==1.
    assert result[0,1,0,0]==0.
    assert runtime.stats.summary()['skipped_valid_elements']==1


@pytest.mark.parametrize('active', [None,torch.tensor([[True,False]])])
def test_experiment_router_matches_shared_physical_execution(active):
    scores=torch.tensor([[[[5.,4.,0.,0.,-1.],[1.,0.,4.,3.,-2.]]]])
    valid=torch.ones_like(scores,dtype=torch.bool)
    routed=blasst_route(scores,valid,1.,active_rows=active,q_tile_size=2,kv_tile_size=2)
    masked,decisions=apply_blasst_2d(scores,valid,active,Blasst2DConfig(blasst_lambda=1.,q_tile_size=2,kv_tile_size=2))
    assert torch.equal(routed.allowed,torch.isfinite(masked))
    assert torch.equal(routed.physical_skip,decisions.skip_mask)
    assert routed.stats['skipped_valid_qk_elements']==int(decisions.skipped_valid_elements.sum())


def test_offline_calibration_elements_follow_physical_not_row_votes(tmp_path):
    trace=dict(attention_type='global',valid_kv_length=5,
        margins=[[float('inf'),-5.,-6.],[float('inf'),3.,-7.]],
        valid_rows=[[True,True,True],[True,True,False]],
        eligible_tiles=[True,True,True],valid_elements=[[2,2,1],[2,2,0]])
    path=tmp_path/'traces.jsonl'; path.write_text(json.dumps(trace)+'\n')
    direct=evaluate_margin_trace(trace,1.,metric='valid_qk')
    variants=[direct,evaluate_lambda_grid([trace],[1.],metric='valid_qk')[0],
              evaluate_lambda_grid_jsonl(path,[1.],metric='valid_qk')[0],
              evaluate_lambda_grid_jsonl_by_type(path,[1.],metric='valid_qk')['global'][0]]
    for row in variants:
        assert row['blasst_mask_semantics']==BLASST_MASK_SEMANTICS
        assert row['skipped_valid_qk_elements']==1  # not three from row votes
        assert row['valid_qk_element_sparsity']==pytest.approx(1/9)
        assert row['physical_sparsity']==pytest.approx(1/3)


@pytest.mark.parametrize('artifact', ['run_config.json','task_run_config.json','ruler_run_config.json',
                                      'dualcache_run_config.json','summary.json','attention_stats/summary.json'])
def test_legacy_provenance_is_rejected_without_writes(tmp_path,artifact):
    path=tmp_path/artifact; path.parent.mkdir(parents=True,exist_ok=True)
    original='{"eligible_tiles": 42}\n'; path.write_text(original)
    with pytest.raises(ValueError,match='fresh output'):
        validate_blasst_output_directory(tmp_path)
    assert path.read_text()==original


def test_current_stats_resume_but_legacy_stats_cannot_load_or_overwrite(tmp_path):
    stats=Blasst2DStats(); config=Blasst2DConfig()
    stats.export(tmp_path,config)
    assert Blasst2DStats.load_export(tmp_path).summary()==stats.summary()
    path=tmp_path/'summary.json'
    legacy=json.loads(path.read_text()); legacy.pop('blasst_mask_semantics')
    original=json.dumps(legacy); path.write_text(original)
    with pytest.raises(ValueError,match='mask semantics'):
        Blasst2DStats.load_export(tmp_path)
    with pytest.raises(ValueError,match='mask semantics'):
        stats.export(tmp_path,config)
    assert path.read_text()==original


def test_run_rejects_old_directory_before_log_or_model_load(tmp_path,monkeypatch):
    from dllm.evaluation.ruler import runner
    out=tmp_path/'old'; out.mkdir()
    config_path=out/'run_config.json'; config_path.write_text('{"fingerprint":"legacy"}')
    monkeypatch.setattr(runner,'_load_manifest',lambda config:({},[]))
    monkeypatch.setattr(runner,'create_adapter',lambda *a,**k:pytest.fail('model must not load'))
    config=runner.RulerRunConfig(model_adapter='diffusion_gemma',model_path='fake',manifest_path='unused',
        ruler_root='unused',output_dir=str(out),attention_backend='blasst-reference',
        num_samples=1,context_length=8)
    with pytest.raises(ValueError,match='mask semantics'):
        runner.run_evaluation(config)
    assert not (out/'progress.log').exists()
    assert config_path.read_text()=='{"fingerprint":"legacy"}'


def test_report_rejects_mixed_semantics_before_refreshing_shards(tmp_path, monkeypatch):
    from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k import report
    dense = tmp_path / 'dense'
    dense.mkdir()
    (dense / 'predictions.jsonl').write_text('')
    for name, config in [('blasst_calibrated_s25', {}),
                         ('blasst_calibrated_s50', {'blasst_mask_semantics': BLASST_MASK_SEMANTICS})]:
        path = tmp_path / name
        path.mkdir()
        (path / 'summary.json').write_text('{}')
        (path / 'run_config.json').write_text(json.dumps(config))
        (path / 'predictions.jsonl').write_text('{"sample_id":"example"}\n')
    monkeypatch.setattr(report, 'write_json', lambda *a, **k: pytest.fail('no derived writes before preflight'))
    with pytest.raises(ValueError, match='Mixed legacy'):
        report.build_report(tmp_path)
