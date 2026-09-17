from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch
from experiments import diffusion_gemma_jl_aime_tiny_dimensions as study
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections,SketchCache
from experiments.diffusion_gemma_jl_output_aware import reference


def test_scoped_tiny_rank_support_restores_legacy():
    before=study.previous.configuration.DIMENSIONS
    for rank in (2,4):
        with pytest.raises(ValueError):Config(rank=rank)
    with study.dimension_scope():
        for rank in (2,4,8,16,24,32):assert Config(rank=rank).rank==rank
        assert study.runner.PROJECTED==study.ALL
    assert study.previous.configuration.DIMENSIONS==before


@pytest.mark.parametrize('rank',(2,4))
def test_deterministic_linearity_gqa_and_sketch_refresh(rank):
    torch.manual_seed(31)
    v=torch.randn(1,2,137,64);valid=torch.ones(v.shape[:-1],dtype=torch.bool)
    bank=Projections();matrix=bank.get(5,2,64,'gaussian',rank,1729,'cpu')
    other=Projections();assert torch.equal(matrix,other.get(5,2,64,'gaussian',rank,1729,'cpu'))
    assert bank.manifest==other.manifest and not torch.equal(matrix[0],matrix[1])
    scores=torch.randn(1,4,3,137);mask=torch.rand_like(scores)>.2
    small=reference.block_statistics(scores,mask,(v@matrix).repeat_interleave(2,1))
    full=reference.block_statistics(scores,mask,v.repeat_interleave(2,1))
    expected=torch.einsum('bhqtd,hdr->bhqtr',full['mu'],matrix.repeat_interleave(2,0))
    torch.testing.assert_close(small['mu'],expected,atol=2e-6,rtol=3e-5)
    with study.dimension_scope():
        cache=SketchCache(Config(rank=rank));old=cache.get(5,v,valid,90)['z'].clone()
        torch.testing.assert_close(cache.get(5,v,valid,90)['z'],old)
        assert cache.work['reused_tokens']==128
        v[...,70,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,70,:],old[...,70,:])
        v[...,5,:]+=1;assert not torch.equal(cache.get(5,v,valid,90)['z'][...,5,:],old[...,5,:])
        valid[...,5]=False;result=cache.get(5,v,valid,90)
        expected=(v.norm(dim=-1).square().masked_fill(~valid,0.).sum(-1)/valid.sum(-1)).sqrt()
        torch.testing.assert_close(result['ref'],expected)


def test_exact_existing_manifests_and_complete_comparison_set(tmp_path):
    setup=study.prepare(tmp_path);old=study.read(study.PREVIOUS/'setup.json')
    for split in ('final','calibration','development'):assert setup[split]==old[split]
    assert len(setup['final'])==30 and len(setup['calibration'])==6
    assert len(setup['conditions'])==11 and setup['targets']==[.5]
    assert set(study.REUSED)==set(old['conditions']) and set(study.NEW)=={'jl_gaussian_r2','jl_gaussian_r4'}
    assert setup['final_generation_slots']==60 and setup['reused_reference_and_baseline_slots']==270
    assert setup==study.prepare(tmp_path)


def test_full_budget_calibration_and_strict_acceptance(tmp_path):
    setup=study.prepare(tmp_path)
    assert len(study.previous.prior.calibration_rows(setup,'aime26'))==6
    bad=deepcopy(setup);bad['calibration'][0]['generation_budget']=512
    with pytest.raises(ValueError):study.previous.prior.calibration_rows(bad,'aime26')
    bad=deepcopy(setup);bad['calibration'][0]['id']='aime26/new'
    with pytest.raises(ValueError):study.previous.prior.calibration_rows(bad,'aime26')
    assert study.fb.within({'overall':.5,'local':.48,'global':.52},.5)
    assert not study.fb.within({'overall':.5,'local':.48,'global':.52001},.5)


def test_rank_specific_proposals_and_unverified_common_initial_trial(tmp_path,monkeypatch):
    calls=[]
    def distributions(root,name,benchmark):
        calls.append((root,name,benchmark));return {k:np.array([-np.inf,-2.,0.]) for k in ('local','global')},{}
    monkeypatch.setattr(study.screen,'distributions',distributions)
    expected=study.read(study.PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json')['policy']
    for name in study.NEW:
        old,values,sources=study.warm_start(tmp_path,name,'aime26',.5)
        assert old['policy']==expected and old['config']==dict(family='gaussian',rank=32)
        assert all(np.isfinite(v).all() for v in values.values()) and sources
    assert [c[1] for c in calls]==list(study.NEW) and all(c[0]==tmp_path/'rank_proposals' for c in calls)
    with pytest.raises(ValueError):study.warm_start(tmp_path,'jl_gaussian_r2','longbench_v2',.5)


def test_accounting_does_not_count_aliases_as_inference(tmp_path):
    for stage,outputs in {'calibration':[{},{}],'final':[{},dict(imported_source={'path':'cal'})],
                          'dense':[dict(imported_source={'path':'old'})]}.items():
        folder=tmp_path/stage/'label/shards';folder.mkdir(parents=True)
        for i,out in enumerate(outputs):(folder/f'{i}.json').write_text(json.dumps(out))
    folder=tmp_path/'validation/fp/smoke/label/shards';folder.mkdir(parents=True);(folder/'one.json').write_text('{}')
    (tmp_path/'native_smoke_imports.json').write_text('{"a":{},"b":{}}')
    result=study.accounting(tmp_path)
    assert result['completed_new_inference']==4 and result['new_short_validation_calls']==1
    assert result['final_generation_slots']==60 and result['reused_final_baselines_and_references']==270


def test_imported_policy_audit_and_condition_freeze_are_nonrecursive(tmp_path,monkeypatch):
    setup=study.prepare(tmp_path);contract=dict(fingerprint='test',sources={})
    monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    (tmp_path/'predecessor_audit.json').write_text('{"sources":{}}')
    folder=tmp_path/'policies/aime26';folder.mkdir(parents=True)
    for label in study.REUSED:
        if label=='dense':continue
        src=study.PREVIOUS/'policies/aime26'/f'{label}.json';old=study.read(src)
        p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
        p.update(fingerprint='test',heldout_used=False,imported_policy=str(src),sources={})
        (folder/f'{label}.json').write_text(json.dumps(p))
    before=study.previous.NEW
    conditions=study.freeze_conditions(tmp_path,setup,contract)
    assert set(conditions)==set(study.REUSED) and study.previous.NEW==before
    assert study.previous.audit_policy is study.PREVIOUS_AUDIT
    p['policy']['local']['log_threshold']=100.
    with pytest.raises(ValueError,match='changed'):study.audit_policy(tmp_path,p,setup,contract)


def test_complete330_report_pairwise_regeneration_and_missing_detection(tmp_path,monkeypatch):
    report=study.backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=study.prepare(tmp_path)
    monkeypatch.setattr(study,'prepare',lambda root:setup)
    monkeypatch.setattr(study,'execution',lambda root:dict(fingerprint='test',sources={}))
    monkeypatch.setattr(study,'audit_policy',lambda *a:None);monkeypatch.setattr(study,'smoke_audit',lambda *a:{})
    monkeypatch.setattr(report.common,'diagnostic_summary',lambda *a:([],{}))
    monkeypatch.setattr(report.common,'audit_matrices',lambda *a:None)
    monkeypatch.setattr(report,'selected_sources',lambda *a:{})
    monkeypatch.setattr(study.previous,'BASE_PLOTS',lambda *a:[])
    for n in ('shared_state_index','shared_diagnostics_index'):(tmp_path/f'{n}.json').write_text('[]')
    raw=tmp_path/'raw.json';raw.write_text('{}');monkeypatch.setattr(report,'shard_path',lambda *a:raw)
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        records=[]
        for i in range(2):
            r=dict.fromkeys(FIELDS,0);r.update(layer=i*5,step=0,head=0,attention_type=('local','global')[i],probe='execution',
                eligible=10*(i+1),skipped=0 if label=='dense' else 5*(i+1),rows=2,
                mass_sum=2 if label=='dense' else 1.5,dense_sq=2,error_sq=0 if label=='dense' else .1);records.append(r)
        return dict(prediction='42',completion_tokens=[1,2],records=records,backend='synthetic',records_source=dict(path=str(raw),sha256=study.sha(raw.read_bytes())))
    monkeypatch.setattr(report,'cached',output)
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],accuracy=1.,dense_accuracy=1.,
            matching=2,compared=3,exact_match=False,termination_reason='length',unparsed_answer=False,output_length=2,
            aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k]) for k in ('overall','local','global')})
    monkeypatch.setattr(report.evidence,'pair',pair);(tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        c=dict(fingerprint='test',name=name,target=0. if name=='dense' else .5,config={} if name=='dense' else study.CONFIGS[name],sources={},
            thresholds={'aime26':None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')}})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    audit=study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==audit['expected']==330
    assert len(study.read(tmp_path/'summary.json'))==33 and len(study.read(tmp_path/'dimension_pairwise.json'))==30
    assert audit==study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    assert '330 audited' in (tmp_path/'report.md').read_text()
    def missing(*a,**kw):
        if a[2]['id']==setup['final'][0]['id'] and a[4]=='jl_gaussian_r2_s50':raise FileNotFoundError('missing')
        return output(*a,**kw)
    monkeypatch.setattr(report,'cached',missing);assert not study.regenerate(tmp_path)['complete']


@pytest.mark.parametrize('rank',(2,4))
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Tiny-rank CUDA kernel validation')
def test_cuda_tiny_ranks_statistics_masks_gqa_and_trace(rank):
    from experiments.diffusion_gemma_jl_output_aware import kernels
    from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
    torch.manual_seed(918);torch.backends.cuda.matmul.allow_tf32=False
    scores=torch.randn(1,4,131,193,device='cuda').bfloat16();valid=torch.rand(scores.shape,device='cuda')>.2
    valid[...,0,:]=False;valid[...,1,:64]=False;valid[...,2,128:]=False
    value=torch.randn(1,2,193,64,device='cuda');matrix=Projections().get(5,2,64,'gaussian',rank,1729,'cuda');z=value@matrix
    actual=kernels.block_statistics(scores,valid,z);expected=reference.block_statistics(scores,valid,z.repeat_interleave(2,1))
    for key in ('b','logz','mu','mean_norm'):torch.testing.assert_close(actual[key],expected[key],atol=4e-6,rtol=3e-5)
    assert torch.equal(actual['count'],expected['count']) and actual['mu'].shape[-1]==rank
    ref=value.norm(dim=-1).square().mean(-1).sqrt()
    with study.dimension_scope():
        c=Config(rank=rank);thresholds=[-math.inf,-3.,-1.,0.]
        masks,margin,risks=kernels.route(actual,ref,c,thresholds,True)
        for pi,t in enumerate(thresholds):
            for qb,start in enumerate(range(0,131,128)):
                st={k:v[...,start:start+128,:] for k,v in expected.items() if k not in ('mu','eligible')}
                st['mu']=expected['mu'][...,start:start+128,:,:];st['eligible']=st['active'].any(-2)
                cc=replace(c,log_threshold=t);selected,rows=reference.route(st,ref.repeat_interleave(2,1),cc,True)
                assert torch.equal(selected[0],masks[pi,:,qb]) and not selected[...,0].any()
                if pi==1:
                    traced=trace(st,ref.repeat_interleave(2,1),selected,cc)
                    for j,row in enumerate(rows):
                        centered=row['delta'].norm(dim=-1)/ref.repeat_interleave(2,1)[...,None]
                        torch.testing.assert_close(traced['centered'][...,j],centered,atol=3e-6,rtol=3e-5)
