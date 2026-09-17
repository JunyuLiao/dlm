from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from experiments import diffusion_gemma_jl_aime_seed43 as study
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_jl_output_aware.config import Config


def test_only_generation_seed_changes_all30_rows(tmp_path):
    setup=study.prepare(tmp_path);old=study.read(study.PREVIOUS/'setup.json')
    assert setup['final']==[dict(r,seed=43) for r in old['final']]
    assert setup['decoding']==old['decoding'] and setup['projection_seed']==1729
    assert setup['calibration_ids']==old['calibration_ids'] and len(setup['calibration'])==6
    assert all(r['seed']==43 and r['generation_budget']==2048 for r in setup['final'])
    assert set(study.CONDITIONS)=={'dense','blasst_aggressive_s50','mass_s50','full_centered_s50','jl_gaussian_r8_s50'}
    assert study.EXPECTED==30*len(study.CONDITIONS)==150 and setup==study.prepare(tmp_path)


def test_projection_seed_and_hashes_unchanged_by_generation_seed():
    import torch
    c=Config(**study.ALL['jl_gaussian_r8']);assert c.projection_seed==1729
    a,b=Projections(),Projections()
    torch.manual_seed(42);ma=a.get(5,2,512,'gaussian',8,c.projection_seed,'cpu')
    torch.manual_seed(43);mb=b.get(5,2,512,'gaussian',8,c.projection_seed,'cpu')
    assert torch.equal(ma,mb) and a.manifest==b.manifest


def policy(name):
    path=study.source_policy(name);old=study.read(path)
    p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
    p.update(fingerprint='test',heldout_used=False,imported_policy=str(path),sources={})
    return p


def test_exact_previous_policies_no_refitting_or_benchmark_substitution(tmp_path):
    setup=study.prepare(tmp_path)
    for name in study.CONFIGS:
        p=policy(name);study.audit_policy(tmp_path,p,setup,{'fingerprint':'test'})
        p['policy']['local']['log_threshold']=123.
        with pytest.raises(ValueError,match='threshold changed'):study.audit_policy(tmp_path,p,setup,{'fingerprint':'test'})
    p=policy('mass');p['benchmark']='longbench_v2'
    with pytest.raises(ValueError,match='identity'):study.audit_policy(tmp_path,p,setup,{'fingerprint':'test'})


def test_freeze_conditions_ties_every_threshold_to_policy(tmp_path,monkeypatch):
    setup=study.prepare(tmp_path);contract=dict(fingerprint='test',sources={})
    monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    study.frozen_write(tmp_path/'predecessor_audit.json',dict(sources={}))
    for name in study.CONFIGS:study.frozen_write(tmp_path/'policies/aime26'/f'{name}_s50.json',policy(name))
    conditions=study.freeze_conditions(tmp_path,setup,contract)
    policies=[study.read(p) for p in (tmp_path/'policies').glob('*/*.json')]
    assert list(conditions)==study.CONDITIONS
    with patch.object(study.reporting,'study',study):
        study.reporting.validate_conditions(conditions,policies)
        conditions['mass_s50']['thresholds']['aime26']['local']['log_threshold']=123.
        with pytest.raises(ValueError,match='threshold differs'):study.reporting.validate_conditions(conditions,policies)


def test_seed_comparison_pairs_questions_not_dense_seed42(tmp_path):
    setup=study.prepare(tmp_path);old=[];new=[]
    for label in study.CONDITIONS:
        for i,r in enumerate(setup['final']):
            base={k:r[k] for k in ('id','prompt_hash','generation_budget')}
            old.append(dict(base,condition=label,seed=42,accuracy=float(i<15)))
            new.append(dict(base,condition=label,seed=43,accuracy=float(i<16)))
    result=study.seed_comparisons(new,old)
    assert all(r['seed42_correct']==15 and r['seed43_correct']==16 and r['delta_pp']==pytest.approx(100/30) for r in result)
    new[0]['seed']=42
    with pytest.raises(ValueError,match='identity'):study.seed_comparisons(new,old)
    new[0]['seed']=43
    with pytest.raises(ValueError,match='paired30'):study.seed_comparisons(new[:-1],old)


def test_queue_cannot_start_before_successful_final_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(study.parent,'ROOT',tmp_path)
    assert study.parent_ready() is None
    study.frozen_write(tmp_path/'supervisor_terminal.json',dict(exit_code=1))
    with pytest.raises(ValueError,match='failed'):study.parent_ready()
    (tmp_path/'supervisor_terminal.json').write_text(json.dumps(dict(exit_code=0)))
    study.frozen_write(tmp_path/'terminal.json',dict(complete=True,completed=1260))
    calls=[]
    monkeypatch.setattr(study.parent.remaining,'require_complete',lambda root,count:calls.append((root,count)) or {'audit':'hash'})
    assert study.parent_ready()=={'audit':'hash'} and calls==[(tmp_path,1260)]


def test_accounting_fresh150_and_validation_only(tmp_path):
    for stage,n in (('dense',30),('final',120)):
        folder=tmp_path/stage/'label/shards';folder.mkdir(parents=True)
        for i in range(n):(folder/f'{i}.json').write_text('{}')
    folder=tmp_path/'validation/fp/smoke/label/shards';folder.mkdir(parents=True)
    for i in range(6):(folder/f'{i}.json').write_text('{}')
    for name in ('aa','bb'):(tmp_path/'validation/fp'/f'{name}.native.json').write_text('{}')
    data=study.accounting(tmp_path)
    assert data['completed_new_inference']==158 and data['new_native_smoke_calls']==2
    assert data['final_generation_slots']==150 and data['reused_final_baselines_and_references']==0
    assert data['completed_by_stage']['calibration']['stored']==0


def test_historical_diagnostics_cover_all_four_methods_without_gpu(tmp_path):
    setup=study.prepare(tmp_path);contract={'fingerprint':'test'}
    # Exercise real inherited source proofs, threshold import, diagnostic mapping
    # and exact method/state coverage. No inference and no CUDA required.
    study.reuse(tmp_path,setup,contract)
    index=study.read(tmp_path/'shared_diagnostics_index.json')
    assert len(index)==4*32
    assert {study.read(x['path'])['identity']['name'] for x in index}==set(study.CONFIGS)
    assert not (tmp_path/'dense').exists() and not (tmp_path/'final').exists()


def test_raw_report_complete150_repeatable_and_missing_detected(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup=study.prepare(tmp_path);contract=dict(fingerprint='test',sources={});report=study.reporting
    monkeypatch.setattr(study,'execution',lambda root:contract)
    monkeypatch.setattr(study,'smoke_audit',lambda *args:{})
    study.frozen_write(tmp_path/'predecessor_audit.json',dict(sources={}))
    for name in study.CONFIGS:study.frozen_write(tmp_path/'policies/aime26'/f'{name}_s50.json',policy(name))
    study.freeze_conditions(tmp_path,setup,contract)
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
        return dict(**{k:row[k] for k in ('id','benchmark','task','calibration','seed','prompt_hash','generation_budget')},
            accuracy=1.,dense_accuracy=1.,matching=2,compared=3,exact_match=False,termination_reason='length',unparsed_answer=False,output_length=2,
            aggregates={k:aggregate([r for r in out['records'] if k=='overall' or r['attention_type']==k]) for k in ('overall','local','global')})
    monkeypatch.setattr(study,'shard_path',lambda *args:raw);monkeypatch.setattr(study.runner,'cached',output)
    monkeypatch.setattr(report,'audit_diagnostics',lambda *args:([],{}))
    monkeypatch.setattr(report.core.common,'audit_matrices',lambda *args:None)
    monkeypatch.setattr(report.core.evidence,'pair',pair);monkeypatch.setattr(report.core,'plots',lambda *args:[])
    audit=study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed']==150 and len(study.read(tmp_path/'summary.json'))==15
    assert len(study.read(tmp_path/'seed_comparison.json'))==5
    assert all(r['overall_physical_sparsity']==.5 for r in study.read(tmp_path/'summary.json') if r['name']!='dense')
    assert audit==study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    def missing(*a,**kw):
        if a[2]['id']==setup['final'][-1]['id'] and a[4]=='mass_s50':raise FileNotFoundError('missing')
        return output(*a,**kw)
    monkeypatch.setattr(study.runner,'cached',missing)
    audit=study.regenerate(tmp_path);assert not audit['complete'] and audit['completed']==149
