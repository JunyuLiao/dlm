from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments import diffusion_gemma_jl_lowrank_multibench_v2 as study
from experiments import diffusion_gemma_jl_lowrank_multibench_v2_report as reporting
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections,SketchCache
from experiments.diffusion_gemma_jl_output_aware import reference


def test_scoped_rank1_restores_frozen_configuration():
    before=study.tiny.previous.configuration.DIMENSIONS
    with pytest.raises(ValueError):Config(rank=1)
    with study.dimension_scope():
        for r in (1,2,4,8,16,24,32):assert Config(rank=r).rank==r
    assert study.tiny.previous.configuration.DIMENSIONS==before


def test_rank1_projection_linearity_gqa_and_cache_invalidation():
    torch.manual_seed(31);v=torch.randn(1,2,137,64);valid=torch.ones(v.shape[:-1],dtype=torch.bool)
    bank=Projections();matrix=bank.get(5,2,64,'gaussian',1,1729,'cpu');other=Projections()
    assert torch.equal(matrix,other.get(5,2,64,'gaussian',1,1729,'cpu')) and bank.manifest==other.manifest
    scores=torch.randn(1,4,3,137);mask=torch.rand_like(scores)>.2
    full=reference.block_statistics(scores,mask,v.repeat_interleave(2,1))
    small=reference.block_statistics(scores,mask,(v@matrix).repeat_interleave(2,1))
    expected=torch.einsum('bhqtd,hdr->bhqtr',full['mu'],matrix.repeat_interleave(2,0))
    torch.testing.assert_close(small['mu'],expected,atol=3e-6,rtol=3e-5)
    with study.dimension_scope():
        cache=SketchCache(Config(rank=1));old=cache.get(5,v,valid,90)['z'].clone()
        torch.testing.assert_close(cache.get(5,v,valid,90)['z'],old);assert cache.work['reused_tokens']==128
        v[...,70,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,70,:],old[...,70,:])
        v[...,5,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,5,:],old[...,5,:])
        valid[...,5]=False
        expected=(v.norm(dim=-1).square().masked_fill(~valid,0.).sum(-1)/valid.sum(-1)).sqrt()
        torch.testing.assert_close(cache.get(5,v,valid,90)['ref'],expected)


def test_exact_30_and100_cached_manifests_and_asymmetric_condition_scope(tmp_path):
    setup=study.prepare(tmp_path)
    assert [r for r in setup['final'] if r['benchmark']=='aime26']==study.read(study.AIME/'setup.json')['final']
    def infer(rows):return [{k:v for k,v in r.items() if k!='calibration'} for r in rows]
    assert infer([r for r in setup['final'] if r['benchmark']=='longbench_v2'])==infer(study.read(study.LB/'setup.json')['final'])
    assert len(setup['conditions_by_benchmark']['aime26'])==12 and len(setup['conditions_by_benchmark']['longbench_v2'])==9
    assert sum(study.COUNTS[b]*len(v) for b,v in setup['conditions_by_benchmark'].items())==1260
    assert setup['new_by_benchmark']=={'aime26':['jl_gaussian_r1'],'longbench_v2':['jl_gaussian_r1','jl_gaussian_r2','jl_gaussian_r8']}
    assert sum(r['calibration'] for r in setup['final'])==12 and setup==study.prepare(tmp_path)


def test_preserved_calibration_membership_full_budgets_and_tolerance(tmp_path):
    setup=study.prepare(tmp_path)
    for b in study.COUNTS:assert len(study.calibration_rows(setup,b))==6
    bad=deepcopy(setup);bad['calibration'][-1]['generation_budget']=512
    with pytest.raises(ValueError):study.calibration_rows(bad,'longbench_v2')
    bad=deepcopy(setup);bad['calibration'][0]['id']='aime26/wrong'
    with pytest.raises(ValueError):study.calibration_rows(bad,'aime26')
    assert study.fb.within({'overall':.5,'global':.48,'local':.52},.5)
    assert not study.fb.within({'overall':.5,'global':.4799,'local':.52},.5)


def test_two_fifty_cohorts_reuse_exact_full100_sources():
    a=next(r for r in study.read(study.V6/'setup.json')['final'] if r['benchmark']=='longbench_v2')
    b=study.read(study.V7/'setup.json')['final'][0]
    for row,source in ((a,study.V6),(b,study.V7)):
        for label in ('full_centered_s50','jl_gaussian_r32_s50'):assert study.source_for('longbench_v2',label,row['id'])==source
        for label in study.remaining.BASELINE_LABELS:assert study.source_for('longbench_v2',label,row['id'])==study.LB
    assert study.source_for('aime26','jl_gaussian_r2_s50','aime26/2')==study.AIME
    dense=study.read(study.LB/'final_configs/dense.json')
    assert dense['thresholds']=={} and study.validate_cached_condition(dense,'dense','longbench_v2',None)=={}
    sparse=study.read(study.LB/'final_configs/mass_s50.json')
    threshold=study.read(study.imported_policy_path('longbench_v2','mass'))['policy']
    assert study.validate_cached_condition(sparse,'mass','longbench_v2',threshold)==study.CONFIGS['mass']
    sparse['thresholds']={}
    with pytest.raises(ValueError,match='mismatch'):study.validate_cached_condition(sparse,'mass','longbench_v2',threshold)


def test_benchmark_specific_unverified_warm_trial_and_cdfs(tmp_path,monkeypatch):
    calls=[]
    def distribution(root,name,b):
        calls.append((root,name,b));return {k:np.array([-np.inf,-2.,0.]) for k in ('local','global')},{}
    monkeypatch.setattr(study.screen,'distributions',distribution)
    for b,names in study.NEW_BY_BENCHMARK.items():
        for name in names:
            old,values,sources=study.warm_start(tmp_path,name,b,.5)
            assert old['policy']==study.read(study.imported_policy_path(b,'jl_gaussian_r32'))['policy']
            assert old['config']==dict(family='gaussian',rank=32) and all(np.isfinite(v).all() for v in values.values())
            assert sources
    assert len(calls)==4 and {x[2] for x in calls}==set(study.COUNTS)
    with pytest.raises(ValueError):study.warm_start(tmp_path,'jl_gaussian_r2','aime26',.5)


def imported_policy(b,label):
    name=label.rsplit('_s',1)[0];src=study.imported_policy_path(b,name);old=study.read(src)
    p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
    p.update(fingerprint='test',heldout_used=False,imported_policy=str(src),sources={});return p


def test_cached_policy_fields_and_benchmark_identity_are_checked(tmp_path):
    setup=study.prepare(tmp_path);contract=dict(fingerprint='test')
    for b in study.COUNTS:
        p=imported_policy(b,'jl_gaussian_r32_s50');study.audit_policy(tmp_path,p,setup,contract)
        p['policy']['local']['log_threshold']=100.
        with pytest.raises(ValueError,match='changed'):study.audit_policy(tmp_path,p,setup,contract)
    p=imported_policy('aime26','jl_gaussian_r4_s50');p['benchmark']='longbench_v2'
    with pytest.raises(ValueError):study.audit_policy(tmp_path,p,setup,contract)


def test_freeze_only_available_benchmark_policies(tmp_path,monkeypatch):
    setup=study.prepare(tmp_path);contract=dict(fingerprint='test',sources={})
    monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    (tmp_path/'predecessor_audit.json').write_text('{"sources":{}}')
    for b,labels in study.REUSED_BY_BENCHMARK.items():
        folder=tmp_path/'policies'/b;folder.mkdir(parents=True)
        for label in labels:
            if label!='dense':(folder/f'{label}.json').write_text(json.dumps(imported_policy(b,label)))
    result=study.freeze_conditions(tmp_path,setup,contract)
    assert 'jl_gaussian_r1_s50' not in result
    assert set(result['jl_gaussian_r2_s50']['thresholds'])=={'aime26'}
    assert result['jl_gaussian_r32_s50']['expected_per_benchmark']=={'aime26':30,'longbench_v2':100}
    policies=[study.read(p) for p in (tmp_path/'policies').glob('*/*.json')]
    reporting.validate_conditions(result,policies)
    result['jl_gaussian_r32_s50']['thresholds']['longbench_v2']['local']['log_threshold']=100.
    with pytest.raises(ValueError,match='threshold differs'):reporting.validate_conditions(result,policies)


def test_accounting_counts_only_new_validation_and_generation(tmp_path):
    for stage,outputs in {'calibration':[{}],'final':[{},dict(imported_source={'path':'old'})],
                          'dense':[dict(imported_source={'path':'dense'})]}.items():
        folder=tmp_path/stage/'label/shards';folder.mkdir(parents=True)
        for i,out in enumerate(outputs):(folder/f'{i}.json').write_text(json.dumps(out))
    folder=tmp_path/'validation/fp/smoke/label/shards';folder.mkdir(parents=True)
    (folder/'fresh.json').write_text('{}');(folder/'reuse.json').write_text('{"imported_source":{"path":"old"}}')
    (tmp_path/'native_smoke_imports.json').write_text('{"a":{},"b":{}}')
    a=study.accounting(tmp_path)
    assert a['completed_new_inference']==3 and a['reused_sparse_smoke_calls']==1 and a['new_short_validation_calls']==1
    assert a['final_generation_slots']==330 and a['reused_final_baselines_and_references']==930


def synthetic_setup_and_outputs(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=study.prepare(tmp_path);conditions={}
    for label in study.CONDITIONS:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        conditions[label]=dict(fingerprint='test',name=name,target=0. if name=='dense' else .5,config={} if name=='dense' else study.CONFIGS[name],sources={},
            thresholds={b:None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')}
                        for b in study.COUNTS if label in study.CONDITIONS_BY_BENCHMARK[b]},
            expected_per_benchmark={b:study.COUNTS[b] for b in study.COUNTS if label in study.CONDITIONS_BY_BENCHMARK[b]})
    raw=tmp_path/'raw.json';raw.write_text('{}')
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        records=[]
        for i in range(2):
            eligible=(10 if row['id']==setup['final'][0]['id'] else 30)*(i+1)
            r=dict.fromkeys(FIELDS,0);r.update(layer=i*5,step=0,head=0,attention_type=('local','global')[i],probe='execution',
                eligible=eligible,skipped=0 if label=='dense' else eligible//2,rows=2,mass_sum=2 if label=='dense' else 1.5,
                dense_sq=2,error_sq=0 if label=='dense' else .1);records.append(r)
        return dict(prediction='42',completion_tokens=[1,2],records=records,backend='synthetic',records_source=dict(path=str(raw),sha256=study.sha(raw.read_bytes())))
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],accuracy=1.,dense_accuracy=1.,
            matching=2,compared=3,exact_match=False,termination_reason='length',unparsed_answer=False,output_length=2,
            aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k]) for k in ('overall','local','global')})
    return setup,conditions,output,pair,raw


def test_hundred_question_and_noncalibration94_summaries(tmp_path,monkeypatch):
    setup,conditions,output,pair,_=synthetic_setup_and_outputs(tmp_path,monkeypatch);raw=[]
    for row in setup['final']:
        for label in setup['conditions_by_benchmark'][row['benchmark']]:
            c=conditions[label];out=output(None,tmp_path,row,'final',label,c['name'],c['config'],c['thresholds'][row['benchmark']],{})
            raw.append(dict(condition=label,**pair(row,out,out)))
    rows,tasks,subtasks,groups=reporting.summarize_raw(raw,conditions)
    assert len(rows)==63 and len(raw)==1260
    assert {r['count'] for r in rows if r['benchmark']=='longbench_v2' and r['split']=='full'}=={100}
    assert {r['count'] for r in rows if r['benchmark']=='longbench_v2' and r['split']=='noncalibration94'}=={94}
    assert all(r['overall_physical_sparsity']==.5 for r in rows if r['name']!='dense')
    direct=reporting.direct_comparisons(rows,groups);assert direct and all(r['delta_pp']==0 and r['paired_ci95_pp']==[0.,0.] for r in direct)


def test_complete1260_raw_report_roundtrip_missing_and_policy_scope(tmp_path,monkeypatch):
    setup,conditions,output,pair,raw=synthetic_setup_and_outputs(tmp_path,monkeypatch)
    monkeypatch.setattr(study,'prepare',lambda root:setup);monkeypatch.setattr(study,'execution',lambda root:dict(fingerprint='test',sources={}))
    monkeypatch.setattr(study,'audit_policy',lambda *args:None);monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    monkeypatch.setattr(study,'shard_path',lambda *args:raw);monkeypatch.setattr(study.runner,'cached',output)
    monkeypatch.setattr(reporting,'audit_diagnostics',lambda *args:([],{}))
    monkeypatch.setattr(reporting.core.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(reporting.core.evidence,'pair',pair)
    monkeypatch.setattr(reporting.core,'nemo_pairs',lambda row,outputs:([dict(condition=n,**pair(row,o,outputs['dense'])) for n,o in outputs.items()],[]))
    monkeypatch.setattr(reporting.core,'plots',lambda *args:[])
    (tmp_path/'final_configs').mkdir()
    for label,c in conditions.items():(tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    for label,c in conditions.items():
        if label=='dense':continue
        for b,threshold in c['thresholds'].items():
            folder=tmp_path/'policies'/b;folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{label}.json').write_text(json.dumps(dict(benchmark=b,name=c['name'],config=c['config'],policy=threshold)))
    audit=study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==audit['expected']==1260 and len(study.read(tmp_path/'summary.json'))==63
    assert audit==study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    def missing(*a,**kw):
        if a[2]['id']==setup['final'][-1]['id'] and a[4]=='jl_gaussian_r1_s50':raise FileNotFoundError('missing')
        return output(*a,**kw)
    monkeypatch.setattr(study.runner,'cached',missing)
    incomplete=study.regenerate(tmp_path);assert not incomplete['complete'] and incomplete['completed']==1259
    assert not any(r['benchmark']=='longbench_v2' and r['condition']=='jl_gaussian_r1_s50' and r['split']=='full' for r in study.read(tmp_path/'summary.json'))


@pytest.mark.parametrize('rank',(1,2))
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA rank1 validation')
def test_cuda_tiny_rank_statistics_masks_gqa_trace_and_first_support(rank):
    from experiments.diffusion_gemma_jl_output_aware import kernels
    from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
    torch.manual_seed(918);torch.backends.cuda.matmul.allow_tf32=False
    scores=torch.randn(1,4,131,193,device='cuda').bfloat16();valid=torch.rand(scores.shape,device='cuda')>.2
    valid[...,0,:]=False;valid[...,1,:64]=False;valid[...,2,128:]=False
    v=torch.randn(1,2,193,64,device='cuda');matrix=Projections().get(5,2,64,'gaussian',rank,1729,'cuda');z=v@matrix
    actual=kernels.block_statistics(scores,valid,z);expected=reference.block_statistics(scores,valid,z.repeat_interleave(2,1))
    for k in ('b','logz','mu','mean_norm'):torch.testing.assert_close(actual[k],expected[k],atol=4e-6,rtol=3e-5)
    assert torch.equal(actual['count'],expected['count']) and actual['mu'].shape[-1]==rank
    ref=v.norm(dim=-1).square().mean(-1).sqrt()
    with study.dimension_scope():
        c=Config(rank=rank);thresholds=[-math.inf,-3.,-1.,0.];masks,margin,risks=kernels.route(actual,ref,c,thresholds,True)
        for pi,t in enumerate(thresholds):
            for qb,start in enumerate(range(0,131,128)):
                st={k:v[...,start:start+128,:] for k,v in expected.items() if k not in ('mu','eligible')}
                st['mu']=expected['mu'][...,start:start+128,:,:];st['eligible']=st['active'].any(-2)
                cc=replace(c,log_threshold=t);selected,rows=reference.route(st,ref.repeat_interleave(2,1),cc,True)
                assert torch.equal(selected[0],masks[pi,:,qb]) and not selected[...,0].any()
                if pi==1:
                    traced=trace(st,ref.repeat_interleave(2,1),selected,cc)
                    for j,row in enumerate(rows):
                        torch.testing.assert_close(traced['centered'][...,j],row['delta'].norm(dim=-1)/ref.repeat_interleave(2,1)[...,None],atol=3e-6,rtol=3e-5)
