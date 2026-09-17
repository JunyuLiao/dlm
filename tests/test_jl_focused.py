"""Focused orchestration/provenance tests; synthetic scores are not evidence."""
from copy import deepcopy
import json

import pytest

from experiments.diffusion_gemma_jl_focused import protocol as p


def test_subset_is_deterministic_score_blind_and_source_exact():
    rows=p.read(p.LB/'setup.json')['final']
    chosen,quotas=p.select(rows)
    other=deepcopy(list(reversed(rows)))
    for r in other:
        r['score']=-123;r['prediction']='adversarial';r['expected']='X'
    assert [r['id'] for r in p.select(other)[0]]==[r['id'] for r in chosen]
    assert len(chosen)==50 and all(r in rows for r in chosen)
    assert sorted(quotas['domains'].values())==[3,4,5,8,13,17]
    assert [r['id'] for r in p.select(rows,seed=1)[0]]!=[r['id'] for r in chosen]
    assert sum(sum(v.values()) for v in quotas['subtasks'].values())==50


def test_focused_manifest_and_reduced_scope():
    setup=p.prepare()
    audit=p.audit(setup)
    assert audit['cached_baselines']==560 and audit['new_generations']==480
    assert len(p.CONDITIONS)==13 and len(p.PROJECTED)==3
    assert p.PROJECTED['contribution_gaussian_r32']['rank']==32
    assert p.PROJECTED['jl_gaussian_r32']['rank']==32
    assert set(p.PROJECTED)=={'jl_gaussian_r32','contribution_gaussian_r32','full_centered'}
    assert len(audit['aime_calibration_overlap'])==6 and not audit['longbench_calibration_overlap']
    bad=deepcopy(setup);bad['final'][-1]['generation_budget']=128
    with pytest.raises(ValueError):p.audit(bad)
    bad=deepcopy(setup);bad['calibration'][0]=bad['final'][-1]
    with pytest.raises(ValueError):p.audit(bad)


def test_new_contribution_dispatch_uses_projected_router(monkeypatch):
    from experiments.diffusion_gemma_jl_focused.reuse import dispatch
    from experiments.diffusion_gemma_jl_output_aware import runner
    old=runner.PROJECTED
    called=[]
    class FakeAttention:
        def __init__(self,config,*args):
            called.append(config)
            self.cache=type('Cache',(),{'projections':type('Bank',(),{'manifest':{}})()})()
        def work_accounting(self):return {}
    def backend(adapter,row,config,thresholds,validate=False):
        runner.backend.Attention(config,thresholds,validate=validate)
        return {}
    monkeypatch.setattr(runner,'Attention',FakeAttention)
    monkeypatch.setattr(runner.backend,'generate',backend)
    with dispatch():
        c=p.PROJECTED['contribution_gaussian_r32']
        out=runner.generate(None,{},'contribution_gaussian_r32',c,{})
        assert out['candidate']=='contribution_gaussian_r32'
        assert out['backend']=='jl_fp32_custom_gpu_native_output'
    assert called==[c] and runner.PROJECTED is old


def test_frozen_sources_unchanged():
    from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
    check_sources(p.read(p.OLD/'execution_contract.json')['sources'])


def test_full_report_roundtrip_and_missing_shard(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_jl_focused import report
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=p.prepare()
    monkeypatch.setattr(report,'prepare',lambda root:setup)
    monkeypatch.setattr(report,'execution',lambda root:dict(fingerprint='test',sources={}))
    monkeypatch.setattr(report,'audit_policy',lambda *args:None)
    monkeypatch.setattr(report,'smoke_audit',lambda *args:{})
    monkeypatch.setattr(report.common,'diagnostic_summary',lambda *args:([],{}))
    monkeypatch.setattr(report.common,'shared_seed_summary',lambda *args:([],{}))
    monkeypatch.setattr(report.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(report,'selected_sources',lambda *args:{})
    monkeypatch.setattr(report,'plots',lambda *args:[])
    for name in ('shared_state_index','shared_diagnostics_index'):
        (tmp_path/f'{name}.json').write_text('[]')
    raw=tmp_path/'raw.json';raw.write_text('{}')
    monkeypatch.setattr(report,'shard_path',lambda *args:raw)
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        recs=[]
        for index in range(2):
            rec=dict.fromkeys(FIELDS,0)
            rec.update(layer=index*5,step=0,head=0,attention_type=('local','global')[index],probe='execution',
                calls=1,eligible=10*(index+1),skipped=0 if label=='dense' else 5*(index+1),rows=2,
                mass_sum=2 if label=='dense' else 1.5,dense_sq=2,error_sq=0 if label=='dense' else .1)
            recs.append(rec)
        return dict(prediction='A',completion_tokens=[1,2],records=recs,backend='synthetic',
            records_source=dict(path=str(raw),sha256=p.sha(raw.read_bytes())))
    monkeypatch.setattr(report,'cached',output)
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
            accuracy=1.,dense_accuracy=1.,matching=2,compared=3,exact_match=False,termination_reason='length',
            unparsed_answer=False,output_length=2,aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k])
                for k in ('overall','global','local')})
    monkeypatch.setattr(report.evidence,'pair',pair)
    monkeypatch.setattr(report,'nemo_pairs',lambda row,outs:([dict(condition=n,**pair(row,o,outs['dense'])) for n,o in outs.items()],[]))
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        target=int(label.rsplit('_s',1)[1])/100 if label!='dense' else 0.
        cfg={} if name=='dense' else p.CONFIGS[name]
        condition=dict(fingerprint='test',name=name,target=target,config=cfg,sources={},
            thresholds={b:None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')} for b in ('aime26','longbench_v2')})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(condition))
    audit=report.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==1040
    rows=p.read(tmp_path/'summary.json')
    assert len(rows)==13*4
    assert all(r['overall_physical_sparsity']==(0. if r['name']=='dense' else .5) for r in rows)
    assert all(r['token_agreement']==2/3 for r in rows)
    assert audit==report.regenerate(tmp_path)
    assert report.verify(tmp_path)['passed']
    def absent(*args,**kwargs):
        if args[2]['id']==setup['final'][0]['id'] and args[4]=='jl_gaussian_r32_s50':
            raise FileNotFoundError('synthetic missing shard')
        return output(*args,**kwargs)
    monkeypatch.setattr(report,'cached',absent)
    incomplete=report.regenerate(tmp_path)
    assert not incomplete['complete'] and incomplete['completed']==1039
    assert not any(r['condition']=='jl_gaussian_r32_s50' and r['benchmark']=='aime26' and r['split']=='full'
        for r in p.read(tmp_path/'summary.json'))
