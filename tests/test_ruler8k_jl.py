from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments import diffusion_gemma_ruler8k_jl as study
from experiments import diffusion_gemma_ruler8k_jl_report as reporting
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections, SketchCache
from experiments.diffusion_gemma_jl_output_aware import reference
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts


def test_cached_pool_selection_is_deterministic_and_disjoint():
    pool = [json.loads(x) for x in (study.POOL/'samples.jsonl').read_text().splitlines()]
    a = study.select(pool); b = study.select(list(reversed(pool)))
    assert a == b and a != study.select(pool, 43)
    seen = set()
    for split, n in (('calibration',2), ('final',10), ('development',1)):
        assert Counter(r['task'] for r in a[split]) == dict.fromkeys(study.official.PAPER_TASKS,n)
        ids = {r['sample_id'] for r in a[split]}; assert not ids & seen; seen |= ids


def test_prepared_manifest_exact_budgets_prompts_and_nonoverlap():
    setup = study.prepare(); study.audit_setup(setup)
    assert len(setup['final']) == 130 and len(setup['calibration']) == 26
    assert len(study.CONDITIONS) == 9 and study.EXPECTED == 1170
    assert setup['tile_size'] == [128,64] and setup['regions'] == 'prefix_plus_canvas'
    for row in setup['final']:
        assert row['generation_budget'] == (30 if row['task']=='vt' else 120 if row['task']=='cwe' else 50 if row['task']=='fwe' else 32 if row['task'].startswith('qa_') else 128)
    bad = deepcopy(setup); bad['final'][0] = dict(bad['calibration'][0], split='final',calibration=False)
    with pytest.raises(ValueError): study.audit_setup(bad)


@pytest.mark.parametrize('target,local,global_,counts,aggressive', [
    (.5,.6,.9,{'local':1,'global':1},False),
    (.75,.4,.9,{'local':1,'global':5},False),
    (.75,.4,.7,{'local':1,'global':5},True),
    (.5,.5,.5,{'local':1,'global':1},False),
])
def test_above_one_requires_joint_overall_ceiling_failure(target,local,global_,counts,aggressive):
    overall=(counts['local']*local+counts['global']*global_)/sum(counts.values())
    ceiling=dict(achieved={'local':local,'global':global_,'overall':overall},eligible=counts)
    allowed, goals=study.goals_for(target,ceiling)
    assert allowed == aggressive
    if not allowed:
        assert goals['local']<=local and goals['global']<=global_
        assert sum(goals[k]*counts[k] for k in counts)/sum(counts.values()) == pytest.approx(target)
    else: assert goals=={'local':target,'global':target}


def test_local_shortfall_does_not_unlock_aggressive():
    ceiling=dict(achieved={'local':.4,'global':.9,'overall':.8},eligible={'local':1,'global':4})
    allowed,goals=study.goals_for(.75,ceiling)
    assert not allowed and goals==pytest.approx({'local':.4,'global':.8375})
    assert study.within({'local':.41,'global':.84,'overall':.75},.75,goals)
    assert not study.within({'local':.41,'global':.84,'overall':.72},.75,goals)


def test_length_dependent_lambda_and_cap_audit():
    policy={k:dict(log_scale=8.,cap_one=True) for k in study.KINDS}
    calls=[]
    for layer in range(30):
        kind='global' if layer%6==5 else 'local';length=8192 if kind=='global' else 1024
        logt=min(0.,8.-math.log(length))
        calls.append(dict(layer=layer,step=0,attention_type=kind,valid_kv_length=length,log_lambda=logt,lambda_value=math.exp(logt)))
    out=dict(effective_lambda=calls,finite_calls=30)
    assert len(reporting.audit_lambda(out,policy))==30
    bad=deepcopy(out);bad['effective_lambda'][0]['lambda_value']=1.1
    with pytest.raises(ValueError):reporting.audit_lambda(bad,policy)
    bad=deepcopy(out);bad['effective_lambda'].pop()
    with pytest.raises(ValueError):reporting.audit_lambda(bad,policy)


def test_physical_weighting_mass_and_variable_length_agreement():
    a=dict(eligible=10,skipped=9,mass_sum=1.5,rows=2,error_sq=1.,dense_sq=4.)
    b=dict(eligible=90,skipped=9,mass_sum=2.5,rows=5,error_sq=3.,dense_sq=12.)
    c=aggregate([a,b]);assert c['physical_sparsity']==.18 and c['mass']==4/7 and c['relative_error']==.5
    assert token_counts([1,2,3],[1,4,3,9])==(2,4)
    assert token_counts([1,2,3],[1])==(1,3)


def test_official_ruler_scores_and_scoped_dispatch():
    row=dict(task_base='niah',outputs=['apple','pear'])
    assert study.score(row,'APPLE')==.5
    assert study.score(dict(row,task_base='qa'),'APPLE')==1.
    old=study.runner.score
    with study.scope():
        assert study.runner.score is study.score and Config(rank=2).rank==2
    assert study.runner.score is old


def test_projection_linearity_and_cache_refresh_rank2():
    torch.manual_seed(31);v=torch.randn(1,2,137,64);valid=torch.ones(v.shape[:-1],dtype=torch.bool)
    bank=Projections();matrix=bank.get(5,2,64,'gaussian',2,1729,'cpu')
    assert torch.equal(matrix,Projections().get(5,2,64,'gaussian',2,1729,'cpu'))
    scores=torch.randn(1,4,3,137);mask=torch.rand_like(scores)>.2
    full=reference.block_statistics(scores,mask,v.repeat_interleave(2,1))
    small=reference.block_statistics(scores,mask,(v@matrix).repeat_interleave(2,1))
    torch.testing.assert_close(small['mu'],torch.einsum('bhqtd,hdr->bhqtr',full['mu'],matrix.repeat_interleave(2,0)),atol=3e-6,rtol=3e-5)
    with study.scope():
        cache=SketchCache(Config(rank=2));old=cache.get(5,v,valid,90)['z'].clone()
        assert torch.equal(cache.get(5,v,valid,90)['z'],old)
        v[...,70,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,70,:],old[...,70,:])
        v[...,5,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,5,:],old[...,5,:])


def test_stratified_bootstrap_is_paired_reproducible():
    group=[dict(id=str(i),task=str(i//10),accuracy=float(i%2),dense_accuracy=float(i%2)) for i in range(130)]
    assert reporting.bootstrap(group)==[0.,0.]
    for r in group:r['accuracy']=1.
    assert reporting.bootstrap(group)==reporting.bootstrap(list(reversed(group)))


def test_complete_report_regenerates_only_from1170_final_shards(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    setup=study.prepare();contract=dict(fingerprint='synthetic',sources={})
    monkeypatch.setattr(study,'prepare',lambda root:setup)
    monkeypatch.setattr(study,'execution',lambda root:contract)
    monkeypatch.setattr(study,'smoke',lambda *args:dict(passed=True,sources={}))
    monkeypatch.setattr(study,'audit_policy',lambda *args:None)
    monkeypatch.setattr(reporting.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(reporting,'diagnostic_summary',lambda *args:([],{}))
    monkeypatch.setattr(reporting,'plots',lambda *args:[])
    monkeypatch.setattr(reporting,'audit_lambda',lambda *args:[])
    source=tmp_path/'records.json';source.write_text('{}')
    states=[dict(path='synthetic.pt',id=setup['calibration'][0]['id'],split='calibration',layer=0,step=0)]
    study._write(tmp_path/'shared_state_index.json',states);index=[]
    for label in study.CONDITIONS:
        name='dense' if label=='dense' else label.rsplit('_s',1)[0]
        target=0. if name=='dense' else int(label.rsplit('_s',1)[1])/100
        policy=None if name=='dense' else {k:dict(log_threshold=-1.) for k in study.KINDS}
        cfg=dict(name=name,target=target,config={} if name=='dense' else study.CONFIGS[name],thresholds={study.BENCHMARK:policy},fingerprint='synthetic',sources={})
        study._write(tmp_path/'final_configs'/f'{label}.json',cfg)
        if name!='dense':
            study._write(tmp_path/'policies'/study.BENCHMARK/f'{label}.json',dict(name=name,target=target,policy=policy,goals=dict.fromkeys(study.KINDS,target),
                aggressive_allowed=False,measured=dict.fromkeys(('overall',*study.KINDS),target)))
            path=tmp_path/'diagnostics'/f'{label}.json';study._write(path,dict(identity=dict(source=states[0],name=name,target=target,fingerprint='synthetic')))
            index.append(dict(path=str(path),sha256=study.sha(path.read_bytes())))
        for row in setup['final']:
            study._write(study.shard_path(tmp_path,'dense' if name=='dense' else 'final',label,row['id']),{})
    study._write(tmp_path/'shared_diagnostics_index.json',index)
    def cached(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None and stage in ('dense','final') and row['split']=='final'
        records=[];target=0 if name=='dense' else int(label.rsplit('_s',1)[1])/100
        for layer in (0,5):
            count=100 if layer==0 else 900
            r=dict.fromkeys(FIELDS,0);r.update(layer=layer,head=0,step=0,attention_type='local' if layer==0 else 'global',probe='execution',
                eligible=count,skipped=count*target,rows=4,mass_sum=4,dense_sq=1,prefix_eligible=count)
            records.append(r)
        return dict(id=row['id'],prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
            config=cfg,thresholds=thresholds,prediction=' '.join(row['outputs']),completion_tokens=[1,2],generation_metadata={},termination_reason='eos',
            records=records,records_source=dict(path=str(source),sha256=study.sha(source.read_bytes())))
    monkeypatch.setattr(study.runner,'cached',cached)
    with study.scope():
        audit=reporting.regenerate(tmp_path)
        assert audit['complete'] and audit['completed']==1170 and not audit['violations']
        assert reporting.regenerate(tmp_path)==audit
    rows=study.read(tmp_path/'summary.json');assert len(rows)==9 and {r['count'] for r in rows}=={130}
    assert all(r['accuracy']==1. for r in rows)
    assert all(r['overall_physical_sparsity']==r['target'] for r in rows)
    assert len(study.read(tmp_path/'per_task.json'))==117


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA validation')
@pytest.mark.parametrize('name',list(study.CONFIGS))
def test_gpu_routers_and_lambda_observer_match_trusted_gqa_masks(name):
    from experiments.diffusion_gemma_value_aware.routing import Attention as Legacy
    from experiments.diffusion_gemma_jl_output_aware.routing import Attention as Projected
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(71)
    q=torch.randn(1,4,129,32,device='cuda',dtype=torch.bfloat16)
    k,v=[torch.randn(1,2,270,32,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    valid=torch.ones(1,1,129,270,device='cuda',dtype=torch.bool);valid[...,0,:]=False;valid[...,-1]=False
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_sliding=True,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    policy={x:dict(log_threshold=-1.) for x in study.KINDS};cfg=study.CONFIGS[name]
    with study.scope():
        if name in study.PROJECTED:
            fast=Projected(cfg,policy,validate=True);ref=Projected(cfg,policy,trusted=True)
        else: fast=study.LambdaAttention(cfg,policy,validate=True);ref=Legacy(cfg,policy)
        got=fast(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=256)[0]
        expected=ref(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=256)[0]
    assert torch.equal(got,expected) and torch.isfinite(got).all()
    for kind in ('eligible','skipped','prefix_eligible','canvas_eligible','boundary_eligible'):
        assert sum(r[kind] for r in fast.records)==sum(r[kind] for r in ref.records)
    if name=='blasst':assert len(fast.lambda_records)==1 and fast.lambda_records[0]['lambda_value']==math.exp(-1.)
