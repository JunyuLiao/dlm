"""Scheduling changes do not alter science;720-output scoped report roundtrip."""
from copy import deepcopy
import json
import pytest
from experiments import diffusion_gemma_jl_reference_first as ref


def test_scope_and_order():
    assert ref.LABELS==('full_centered_s50','full_centered_s75')
    assert len(ref.REPORT_LABELS)==9 and len(set(ref.REPORT_LABELS))==9
    assert not any(x.startswith(('jl_','contribution_')) for x in ref.REPORT_LABELS)
    assert ref.MODULE=='experiments.diffusion_gemma_jl_reference_first'
    old=ref.read(ref.ROOT/'execution_contract.json')
    ref.check_sources(old['sources'])


def test_report_alias_preserves_inputs_and_rejects_retarget(tmp_path):
    original=tmp_path/'source.json';original.write_text('{"immutable":true}')
    alias=tmp_path/'view/source.json'
    ref.link(alias,original);ref.link(alias,original)
    assert alias.is_symlink() and alias.read_bytes()==original.read_bytes()
    changed=tmp_path/'other.json';changed.write_text('{}')
    with pytest.raises(ValueError):ref.link(alias,changed)
    assert original.read_text()=='{"immutable":true}'


def test_reference_report720_roundtrip_and_missing_shard(tmp_path,monkeypatch):
    report=ref.report_backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=deepcopy(ref.protocol.prepare());setup['conditions']=list(ref.REPORT_LABELS)
    monkeypatch.setattr(ref,'scheduling_contract',lambda *args,**kwargs:(setup,dict(fingerprint='test',sources={}),dict(sources={})))
    view=tmp_path/'reference_first/report';view.mkdir(parents=True)
    (tmp_path/'reference_first/scheduling_contract.json').write_text('{}')
    (view/'scope.json').write_text('{}')
    monkeypatch.setattr(ref,'report_view',lambda *args:(view,setup))
    monkeypatch.setattr(report,'audit_policy',lambda *args:None)
    monkeypatch.setattr(report,'smoke_audit',lambda *args:{})
    monkeypatch.setattr(report.common,'diagnostic_summary',lambda *args:([],{}))
    monkeypatch.setattr(report.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(report,'selected_sources',lambda *args:{})
    monkeypatch.setattr(report,'plots',lambda *args:[])
    for name in ('shared_state_index','shared_diagnostics_index'):(view/f'{name}.json').write_text('[]')
    raw=view/'raw.json';raw.write_text('{}')
    monkeypatch.setattr(report,'shard_path',lambda *args:raw)
    def output(adapter,root,row,stage,label,name,cfg,thresholds,contract):
        assert adapter is None
        recs=[]
        for index in range(2):
            rec=dict.fromkeys(FIELDS,0)
            rec.update(layer=index*5,step=0,head=0,attention_type=('local','global')[index],probe='execution',
                eligible=10*(index+1),skipped=0 if label=='dense' else 5*(index+1),rows=2,
                mass_sum=2 if label=='dense' else 1.5,dense_sq=2,error_sq=0 if label=='dense' else .1)
            recs.append(rec)
        return dict(prediction='A',completion_tokens=[1,2],records=recs,backend='synthetic',
            records_source=dict(path=str(raw),sha256=ref.sha(raw.read_bytes())))
    monkeypatch.setattr(report,'cached',output)
    def pair(row,out,dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
            accuracy=1.,dense_accuracy=1.,matching=2,compared=3,exact_match=False,termination_reason='length',
            unparsed_answer=False,output_length=2,aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k])
                for k in ('overall','global','local')})
    monkeypatch.setattr(report.evidence,'pair',pair)
    monkeypatch.setattr(report,'nemo_pairs',lambda row,outs:([dict(condition=n,**pair(row,o,outs['dense'])) for n,o in outs.items()],[]))
    (view/'final_configs').mkdir()
    for label in setup['conditions']:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        target=int(label.rsplit('_s',1)[1])/100 if label!='dense' else 0.
        cfg={} if name=='dense' else ref.protocol.CONFIGS[name]
        condition=dict(fingerprint='test',name=name,target=target,config=cfg,sources={},
            thresholds={b:None if name=='dense' else {k:dict(log_threshold=-2.) for k in ('local','global')} for b in ('aime26','longbench_v2')})
        (view/'final_configs'/f'{label}.json').write_text(json.dumps(condition))
    audit=ref.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==audit['expected']==720
    rows=ref.read(view/'summary.json')
    assert len(rows)==9*4 and all(r['token_agreement']==2/3 for r in rows)
    assert all(r['overall_physical_sparsity']==(0. if r['name']=='dense' else .5) for r in rows)
    assert audit==ref.regenerate(tmp_path)
    assert ref.verify(tmp_path)['passed']
    def absent(*args,**kwargs):
        if args[2]['id']==setup['final'][0]['id'] and args[4]=='full_centered_s50':raise FileNotFoundError('synthetic missing shard')
        return output(*args,**kwargs)
    monkeypatch.setattr(report,'cached',absent)
    incomplete=ref.regenerate(tmp_path)
    assert not incomplete['complete'] and incomplete['completed']==719
