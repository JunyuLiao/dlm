"""Candidate selection rejects corrupt sparse statistics or unmatched dense data."""
import json
import pytest
from experiments.diffusion_gemma_value_aware import development_report as report
from experiments.diffusion_gemma_value_aware.run import shard_path


def artifacts(root):
    row=dict(id='fixture/1',benchmark='longbench',task='fixture',prompt_hash='prompt',seed=42,generation_budget=8)
    config=dict(method='mass');policy={k:dict(log_threshold=-1.) for k in ('local','global')}
    records=[dict(probe='execution',layer=l,head=h,step=0,
        attention_type='global' if l%6==5 else 'local',eligible=1,skipped=0,pv_omitted=0,
        softmax_skipped=0,compensated=0,rows=1,mass_sum=1,denominator_mass_sum=1,
        error_sq=0,dense_sq=1,prefix_eligible=1,canvas_eligible=0,boundary_eligible=0,
        prefix_skipped=0,canvas_skipped=0,boundary_skipped=0) for l in range(30) for h in range(16)]
    sparse=dict(row,fingerprint='fp',config=config,thresholds=policy,records=records,
        finite_calls=30,termination_reason='eos',completion_tokens=[1],prediction='answer')
    dense=dict(sparse,config={},thresholds=None)
    paths=[root/'sparse.json',shard_path(root,'screen','dense',row['id'])]
    for path,value in zip(paths,[sparse,dense]):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return row,config,policy,paths


def test_valid_sparse_and_dense_evidence(tmp_path,monkeypatch):
    row,config,policy,paths=artifacts(tmp_path)
    monkeypatch.setattr(report,'score',lambda row,prediction:1.)
    value,sources=report.example(tmp_path,row,paths[0],'fp',config,policy)
    assert value['accuracy']==1 and value['matching']==1 and len(sources)==2


@pytest.mark.parametrize('which,field,value',[(0,'skipped',2),(1,'skipped',2),(1,'prompt_hash','wrong'),(1,'seed',99)])
def test_rejects_corrupted_selection_evidence(tmp_path,which,field,value):
    row,config,policy,paths=artifacts(tmp_path)
    data=json.loads(paths[which].read_text())
    if field=='skipped':data['records'][0][field]=value
    else:data[field]=value
    paths[which].write_text(json.dumps(data))
    with pytest.raises(ValueError,match='development evidence audit'):
        report.example(tmp_path,row,paths[0],'fp',config,policy)


def test_freeze_rejects_legacy_identity_only_audit(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware import evaluate
    monkeypatch.setattr(evaluate,'prepare',lambda root:{})
    monkeypatch.setattr(evaluate,'fingerprint',lambda root:'fp')
    monkeypatch.setattr(evaluate,'candidate_configs',lambda root:{'mass':dict(method='mass')})
    (tmp_path/'candidate_decision.json').write_text(json.dumps(dict(fingerprint='fp',heldout_used=False,
        selected_methods=['mass'],rationale='fixture')))
    (tmp_path/'development_audit.json').write_text(json.dumps(dict(complete=True,heldout_used=False,fingerprint='fp')))
    with pytest.raises(AssertionError,match='shared raw sparse/dense audit'):
        evaluate.freeze(tmp_path,['mass'])


def test_freeze_preserves_raw_evidence_and_rejects_hash_conflicts():
    from experiments.diffusion_gemma_value_aware.evaluate import merge_frozen_sources
    assert merge_frozen_sources({'audit':'a'},{'raw_dense':'d'},{'raw_cal':'c','raw_dense':'d'})=={
        'audit':'a','raw_dense':'d','raw_cal':'c'}
    with pytest.raises(ValueError,match='inconsistent frozen evidence'):
        merge_frozen_sources({'raw_cal':'old'},{'raw_cal':'new'})


def test_calibration_table_retains_repaired_cap_labels_and_source_role():
    from experiments.diffusion_gemma_value_aware.scientific_report import calibration_policy_row
    policy=dict(benchmark='longbench',name='blasst_original',target=.5,config=dict(method='blasst'),
        measured={'local':.25,'global':.51},policy={},cap_one_unattainable=None,
        independent_cap1_repair={'fixed_at_one':{'local':True,'global':False}})
    row=calibration_policy_row(policy,'verified_policies/longbench/blasst_original_s50.json',{'blasst_original_s50':{}})
    assert row['cap1_unattainable']=={'local':True,'global':False} and row['used_by_final_condition']
    assert not calibration_policy_row(policy,'verified_policies/longbench/blasst_original_s50.json',{})['used_by_final_condition']
