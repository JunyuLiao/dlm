from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch

from experiments import diffusion_gemma_jl_aime_dimensions as study
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections, SketchCache
from experiments.diffusion_gemma_jl_output_aware import reference


def test_rank24_allow_list_is_scoped_and_restores_legacy():
    before=study.configuration.DIMENSIONS
    with pytest.raises(ValueError): Config(rank=24)
    with study.dimension_scope():
        assert Config(rank=24).rank==24
        assert study.runner.PROJECTED==study.ALL
        for r in (8,16,32): assert Config(rank=r).rank==r
    assert study.configuration.DIMENSIONS==before
    with pytest.raises(ValueError): Config(rank=24)


def test_rank24_matrix_linearity_gqa_and_cache_invalidation():
    torch.manual_seed(31)
    v=torch.randn(1,2,137,64); valid=torch.ones(v.shape[:-1],dtype=torch.bool)
    a=Projections(); matrix=a.get(5,2,64,'gaussian',24,1729,'cpu')
    b=Projections(); assert torch.equal(matrix,b.get(5,2,64,'gaussian',24,1729,'cpu'))
    assert a.manifest==b.manifest and not torch.equal(matrix[0],matrix[1])
    scores=torch.randn(1,4,3,137); mask=torch.rand_like(scores)>.2
    small=reference.block_statistics(scores,mask,(v@matrix).repeat_interleave(2,1))
    full=reference.block_statistics(scores,mask,v.repeat_interleave(2,1))
    expected=torch.einsum('bhqtd,hdr->bhqtr',full['mu'],matrix.repeat_interleave(2,0))
    torch.testing.assert_close(small['mu'],expected,atol=1e-6,rtol=2e-5)
    with study.dimension_scope():
        cache=SketchCache(Config(rank=24)); old=cache.get(5,v,valid,90)['z'].clone()
        torch.testing.assert_close(cache.get(5,v,valid,90)['z'],old)
        assert cache.work['reused_tokens']==128
        v[...,70,:]+=1; boundary=cache.get(5,v,valid,90)['z']
        assert not torch.equal(boundary[...,70,:],old[...,70,:])
        v[...,5,:]+=1; changed=cache.get(5,v,valid,90)['z']
        assert not torch.equal(changed[...,5,:],old[...,5,:])
        valid[...,5]=False
        result=cache.get(5,v,valid,90)
        expected=(v.norm(dim=-1).square().masked_fill(~valid,0.).sum(-1)/valid.sum(-1)).sqrt()
        torch.testing.assert_close(result['ref'],expected)


def test_existing30_and_six_calibration_rows_are_verbatim(tmp_path):
    setup=study.prepare(tmp_path); old=study.read(study.PREVIOUS/'setup.json')
    assert setup['final']==[r for r in old['final'] if r['benchmark']=='aime26']
    assert setup['calibration']==[r for r in old['calibration'] if r['benchmark']=='aime26']
    assert len(setup['final'])==30 and len(setup['calibration'])==6
    assert len(setup['conditions'])==9 and setup['targets']==[.5]
    assert setup['final_generation_slots']==90 and setup['reused_reference_and_baseline_slots']==180
    assert setup==study.prepare(tmp_path)


def test_calibration_scope_full_budget_and_strict_acceptance(tmp_path):
    setup=study.prepare(tmp_path)
    assert len(study.prior.calibration_rows(setup,'aime26'))==6
    bad=deepcopy(setup);bad['calibration'][0]['generation_budget']=512
    with pytest.raises(ValueError):study.prior.calibration_rows(bad,'aime26')
    bad=deepcopy(setup);bad['calibration'][0]['id']='aime26/not-calibration'
    with pytest.raises(ValueError):study.prior.calibration_rows(bad,'aime26')
    assert study.fb.within({'overall':.50,'local':.48,'global':.52},.5)
    assert not study.fb.within({'overall':.50,'local':.48,'global':.52001},.5)


def test_common_warm_start_never_counts_as_new_rank_calibration(tmp_path,monkeypatch):
    called=[]
    def distributions(root,name,benchmark):
        called.append((root,name,benchmark))
        return {k:np.array([-np.inf,-2.,0.]) for k in ('local','global')},{}
    monkeypatch.setattr(study.screen,'distributions',distributions)
    expected=study.read(study.PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json')['policy']
    for name in study.NEW:
        old,values,sources=study.warm_start(tmp_path,name,'aime26',.5)
        assert old['policy']==expected and old['config']=={'family':'gaussian','rank':32}
        assert all(np.isfinite(x).all() for x in values.values()) and sources
    assert called[-1][0]==tmp_path/'rank24_proposals'
    assert called[0][0]==called[1][0]==study.prior.parent.OLD
    with pytest.raises(ValueError):study.warm_start(tmp_path,'jl_gaussian_r24','longbench_v2',.5)


def test_call_accounting_includes_calibration_smoke_and_exact_final_aliases(tmp_path):
    for stage,outputs in {'calibration':[{},{}],'final':[{},dict(imported_source={"path":"cal"})],
                          'dense':[dict(imported_source={"path":"cached"})]}.items():
        folder=tmp_path/stage/'label/shards';folder.mkdir(parents=True)
        for i,o in enumerate(outputs):(folder/f'{i}.json').write_text(json.dumps(o))
    folder=tmp_path/'validation/fp/smoke/label/shards';folder.mkdir(parents=True)
    (folder/'one.json').write_text('{}')
    (tmp_path/'native_smoke_imports.json').write_text('{"one":{},"two":{}}')
    a=study.accounting(tmp_path)
    assert a['completed_new_inference']==4 and a['new_short_validation_calls']==1
    assert a['reused_native_smoke_calls']==2 and a['final_generation_slots']==90


def test_complete270_report_raw_roundtrip_and_missing_detection(tmp_path,monkeypatch):
    report=study.backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=study.prepare(tmp_path)
    monkeypatch.setattr(study,'prepare',lambda root:setup)
    monkeypatch.setattr(study,'execution',lambda root:dict(fingerprint='test',sources={}))
    monkeypatch.setattr(study,'audit_policy',lambda *args:None)
    monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    monkeypatch.setattr(report.common,'diagnostic_summary',lambda *args:([],{}))
    monkeypatch.setattr(report.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(report,'selected_sources',lambda *args:{})
    monkeypatch.setattr(study,'BASE_PLOTS',lambda *args:[])
    for n in ('shared_state_index','shared_diagnostics_index'):(tmp_path/f'{n}.json').write_text('[]')
    raw=tmp_path/'raw.json';raw.write_text('{}');monkeypatch.setattr(report,'shard_path',lambda *args:raw)
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        rs=[]
        for i in range(2):
            r=dict.fromkeys(FIELDS,0);r.update(layer=i*5,step=0,head=0,attention_type=('local','global')[i],probe='execution',
                eligible=10*(i+1),skipped=0 if label=='dense' else 5*(i+1),rows=2,
                mass_sum=2 if label=='dense' else 1.5,dense_sq=2,error_sq=0 if label=='dense' else .1)
            rs.append(r)
        return dict(prediction='42',completion_tokens=[1,2],records=rs,backend='synthetic',records_source=dict(path=str(raw),sha256=study.sha(raw.read_bytes())))
    monkeypatch.setattr(report,'cached',output)
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
            accuracy=1.,dense_accuracy=1.,matching=2,compared=3,exact_match=False,termination_reason='length',
            unparsed_answer=False,output_length=2,aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k]) for k in ('overall','local','global')})
    monkeypatch.setattr(report.evidence,'pair',pair)
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        c=dict(fingerprint='test',name=name,target=0. if name=='dense' else .5,config={} if name=='dense' else study.CONFIGS[name],sources={},
            thresholds={'aime26':None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')}})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    a=study.regenerate(tmp_path)
    assert a['complete'] and a['completed']==a['expected']==270
    assert len(study.read(tmp_path/'summary.json'))==27
    assert a==study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    assert '270 audited' in (tmp_path/'report.md').read_text()
    def missing(*args,**kwargs):
        if args[2]['id']==setup['final'][0]['id'] and args[4]=='jl_gaussian_r24_s50':raise FileNotFoundError('missing')
        return output(*args,**kwargs)
    monkeypatch.setattr(report,'cached',missing)
    assert not study.regenerate(tmp_path)['complete']


@pytest.mark.parametrize('rank',(8,16,24,32))
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA padded-rank validation')
def test_cuda_rank_statistics_masks_gqa_padding_first_support(rank):
    from experiments.diffusion_gemma_jl_output_aware import kernels
    from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
    torch.manual_seed(918);torch.backends.cuda.matmul.allow_tf32=False
    scores=torch.randn(1,4,131,193,device='cuda').bfloat16()
    valid=torch.rand(scores.shape,device='cuda')>.2
    valid[...,0,:]=False;valid[...,1,:64]=False;valid[...,2,128:]=False
    value=torch.randn(1,2,193,64,device='cuda')
    matrix=Projections().get(5,2,64,'gaussian',rank,1729,'cuda');z=value@matrix
    actual=kernels.block_statistics(scores,valid,z)
    expected=reference.block_statistics(scores,valid,z.repeat_interleave(2,1))
    for k in ('b','logz','mu','mean_norm'):
        torch.testing.assert_close(actual[k],expected[k],atol=4e-6,rtol=3e-5)
    assert torch.equal(actual['count'],expected['count']) and actual['mu'].shape[-1]==rank
    ref=value.norm(dim=-1).square().mean(-1).sqrt()
    with study.dimension_scope():
        c=Config(rank=rank);thresholds=[-math.inf,-3.,-1.,0.]
        masks,margin,risks=kernels.route(actual,ref,c,thresholds,True)
        for pi,t in enumerate(thresholds):
            for qb,start in enumerate(range(0,131,128)):
                st={k:v[...,start:start+128,:] for k,v in expected.items() if k not in ('mu','eligible')}
                st['mu']=expected['mu'][...,start:start+128,:,:];st['eligible']=st['active'].any(-2)
                cc=replace(c,log_threshold=t);selected,trs=reference.route(st,ref.repeat_interleave(2,1),cc,True)
                assert torch.equal(selected[0],masks[pi,:,qb])
                assert not selected[...,0].any()
                if rank==24 and pi==1:
                    traced=trace(st,ref.repeat_interleave(2,1),selected,cc)
                    for j,row in enumerate(trs):
                        centered=row['delta'].norm(dim=-1)/ref.repeat_interleave(2,1)[...,None]
                        torch.testing.assert_close(traced['centered'][...,j],centered,atol=3e-6,rtol=3e-5)
