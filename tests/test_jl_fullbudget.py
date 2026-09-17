"""Full-budget isolation, exact aliases, count weighting and raw-only reporting."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import diffusion_gemma_jl_fullbudget as fb


def test_full_budget_calibration_ids_and_scope(tmp_path):
    s=fb.prepare(tmp_path)
    assert len(s['conditions'])==9 and s['projected_methods_paused']
    assert [r['id'] for r in fb.calibration_rows(s,'aime26')]==['aime26/2','aime26/8','aime26/14','aime26/20','aime26/23','aime26/30']
    assert all(r['generation_budget']==4096 for r in fb.calibration_rows(s,'longbench_v2'))
    bad=deepcopy(s);bad['calibration'][0]['generation_budget']=512
    with pytest.raises(ValueError):fb.calibration_rows(bad,'aime26')
    bad=deepcopy(s);bad['final'].append(fb.calibration_rows(s,'longbench_v2')[0])
    with pytest.raises(ValueError):fb.calibration_rows(bad,'longbench_v2')


def test_weighted_counts_and_strict_deployment_gate():
    outputs=[{'records':[dict(probe='execution',attention_type='local',eligible=10,skipped=5)]},
             {'records':[dict(probe='execution',attention_type='global',eligible=90,skipped=75)]}]
    _,s=fb.measured(outputs)
    assert s=={'overall':.8,'local':.5,'global':75/90}
    assert not fb.within(s,.5)
    assert fb.within({'overall':.5,'local':.48,'global':.52},.5)
    assert not fb.within({'overall':.5,'local':.479,'global':.501},.5)


def test_short_budget_alias_rejected_and_repeated_proposal_fallback(tmp_path,monkeypatch):
    s=fb.prepare(tmp_path);row=fb.calibration_rows(s,'aime26')[0]
    old=fb.read(fb.parent.OLD/'policies/aime26/full_centered_s50.json')
    point=next(x for x in old['trace'] if x['iteration']==old['selected_round'])
    src=Path(point['sources'][row['id']]['path'])
    fp=fb.read(fb.parent.OLD/'execution_contract.json')['fingerprint']
    with pytest.raises(ValueError,match='generation_budget'):
        fb.alias(tmp_path,row,'calibration','short','full_centered',fb.CONFIG,old['policy'],{'fingerprint':'test'},src,fp)
    assert not (tmp_path/'calibration').exists()
    old_policy={k:dict(log_threshold=-2.) for k in ('local','global')}
    new_policy={k:dict(log_threshold=-3.) for k in ('local','global')}
    trace=[dict(policy=old_policy,achieved={'local':.6,'global':.7})]
    monkeypatch.setattr(fb,'rank_next',lambda *args:deepcopy(old_policy))
    monkeypatch.setattr(fb,'scalar_next',lambda *args:deepcopy(new_policy))
    assert fb.proposal(trace,.5,{})==new_policy


def test_fullbudget_report720_raw_roundtrip(tmp_path,monkeypatch):
    report=fb.report_backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=fb.prepare(tmp_path)
    monkeypatch.setattr(fb,'prepare',lambda root:setup)
    monkeypatch.setattr(fb,'execution',lambda root:dict(fingerprint='test',sources={}))
    monkeypatch.setattr(fb,'audit_policy',lambda *args:None)
    monkeypatch.setattr(fb,'inherited_smoke',lambda *args:{})
    monkeypatch.setattr(report.common,'diagnostic_summary',lambda *args:([],{}))
    monkeypatch.setattr(report.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(report,'selected_sources',lambda *args:{})
    monkeypatch.setattr(report,'plots',lambda *args:[])
    for name in ('shared_state_index','shared_diagnostics_index'):(tmp_path/f'{name}.json').write_text('[]')
    raw=tmp_path/'raw.json';raw.write_text('{}');monkeypatch.setattr(report,'shard_path',lambda *args:raw)
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        records=[]
        for index in range(2):
            r=dict.fromkeys(FIELDS,0);r.update(layer=index*5,step=0,head=0,attention_type=('local','global')[index],probe='execution',
                eligible=10*(index+1),skipped=0 if label=='dense' else 5*(index+1),rows=2,mass_sum=2 if label=='dense' else 1.5,
                dense_sq=2,error_sq=0 if label=='dense' else .1)
            records.append(r)
        return dict(prediction='A',completion_tokens=[1,2],records=records,backend='synthetic',
            records_source=dict(path=str(raw),sha256=fb.sha(raw.read_bytes())))
    monkeypatch.setattr(report,'cached',output)
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
            accuracy=1.,dense_accuracy=1.,matching=2,compared=3,exact_match=False,termination_reason='length',unparsed_answer=False,
            output_length=2,aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k]) for k in ('overall','local','global')})
    monkeypatch.setattr(report.evidence,'pair',pair)
    monkeypatch.setattr(report,'nemo_pairs',lambda row,outs:([dict(condition=n,**pair(row,o,outs['dense'])) for n,o in outs.items()],[]))
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        cfg={} if name=='dense' else fb.CONFIG if name=='full_centered' else fb.parent.BASELINES[name]
        c=dict(fingerprint='test',name=name,target=0. if name=='dense' else int(label.rsplit('_s',1)[1])/100,config=cfg,sources={},
            thresholds={b:None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')} for b in ('aime26','longbench_v2')})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    audit=fb.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==audit['expected']==720
    assert len(fb.read(tmp_path/'summary.json'))==36
    assert 'full-budget calibration v3' in (tmp_path/'report.md').read_text()
    assert '512-token verification;' not in (tmp_path/'report.md').read_text()
    assert audit==fb.regenerate(tmp_path) and fb.verify(tmp_path)['passed']
    def missing(*args,**kwargs):
        if args[2]['id']==setup['final'][0]['id'] and args[4]=='full_centered_s50':raise FileNotFoundError('synthetic missing')
        return output(*args,**kwargs)
    monkeypatch.setattr(report,'cached',missing)
    assert not fb.regenerate(tmp_path)['complete']
