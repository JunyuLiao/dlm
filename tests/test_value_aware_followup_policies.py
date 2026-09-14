import copy
import json
from pathlib import Path
import pytest

from experiments.diffusion_gemma_value_aware_followup import policies
from experiments.diffusion_gemma_value_aware_followup.calibrate import fitting_done,smoke_verified
from experiments.diffusion_gemma_value_aware_followup.protocol import sha


def test_main_candidates_include_six_families_and_matched_controls():
    selection=dict(selected={n:dict(pooling='vector_mean') for n in ('value','mass_value','risk')})
    configs=policies.candidate_configs(selection)
    assert set(configs)==set(policies.METHODS)
    assert all(n in configs for n in ('value','mass_value','risk','aligned','centered','compensate'))
    assert configs['no_value_control']['mode']=='no_value_control'
    assert configs['mass_exact']['mode']=='exact_mass'


def test_calibration_rejects_v1_and_non_calibration_examples():
    setup=dict(calibration=[dict(id=str(i),benchmark='longbench_v2',split='calibration') for i in range(6)])
    assert len(policies.calibration_rows(setup,'longbench_v2'))==6
    with pytest.raises(ValueError,match='v1'):policies.calibration_rows(setup,'longbench')
    setup['calibration'][0]['split']='final'
    with pytest.raises(ValueError,match='final/development'):policies.calibration_rows(setup,'longbench_v2')


def test_starting_policies_keep_blasst_inverse_length_and_new_risks_scalar():
    p=dict(policies={'longbench_v2':dict(
        blasst={k:dict(targets={'0.5':dict(log_scale=3.)}) for k in policies.KINDS},
        value_max={k:{'0.5':dict(log_threshold=-.1,unattainable=False)} for k in policies.KINDS})})
    original=policies.starting_policy(p,'longbench_v2','blasst_original',dict(method='blasst'),.5)
    aggressive=policies.starting_policy(p,'longbench_v2','blasst_aggressive',dict(method='blasst'),.5)
    value=policies.starting_policy(p,'longbench_v2','value',dict(method='value',pooling='max'),.5)
    assert all(e['cap_one'] and e['log_scale']==3. for e in original.values())
    assert all(not e['cap_one'] for e in aggressive.values())
    assert all('log_scale' not in e and e['log_threshold']==-.1 for e in value.values())


def test_physical_aggregation_uses_total_counts_and_keeps_pv_omission_separate():
    records=[dict(probe='execution',attention_type='global',eligible=100,skipped=90,pv_omitted=90,rows=2,mass_sum=1),
        dict(probe='execution',attention_type='local',eligible=10,skipped=0,pv_omitted=0,rows=1,mass_sum=1)]
    metrics,achieved=policies.measurements([dict(records=records)],dict(method='value'))
    assert metrics['overall']['physical_sparsity']==pytest.approx(90/110)
    assert metrics['overall']['mass']==pytest.approx(2/3)
    assert achieved=={'local':0.,'global':.9}
    records[0]['skipped']=0
    metrics,achieved=policies.measurements([dict(records=records)],dict(method='compensate'))
    assert metrics['overall']['physical_sparsity']==0
    assert achieved=={'local':0.,'global':.9}


def test_original_boundary_is_independent_and_requires_actual_verification():
    point=dict(policy={'local':dict(cap_one=True,log_threshold=0.),'global':dict(cap_one=True,log_scale=2.)},
        achieved={'local':.2,'global':.5})
    assert fitting_done([point],.5,'blasst_original',10.)
    assert not fitting_done([point],.5,'blasst_aggressive',10.)
    point['achieved']['global']=.3
    assert not fitting_done([point],.5,'blasst_original',10.)
    point['policy']['local']=dict(cap_one=True,log_scale=1.)
    assert not fitting_done([point],.5,'blasst_original',10.)


def test_boundary_reexpression_preserves_other_type_and_all_final_lengths():
    point=dict(policy={'local':dict(log_scale=10.,cap_one=True),
        'global':dict(log_scale=2.,cap_one=True)},achieved={'local':.2,'global':.5})
    result=policies.explicit_boundaries(point,{'local':True,'global':False},10.)
    assert result['source_policy']==point['policy']
    assert result['policy']['global']==point['policy']['global']
    assert policies.constant_one(result['policy']['local'])
    assert not policies.constant_one(point['policy']['local'])
    assert point['policy']['local']['log_scale']==10.  # source remains immutable
    with pytest.raises(ValueError,match='unmeasured'):
        policies.explicit_boundaries(point,{'global':True},10.)


def test_old_aime_policies_never_substitute_for_v2(tmp_path):
    assert policies.import_aime_policy(tmp_path,[dict(benchmark='longbench_v2')],
        'mass',dict(method='mass'),.5,{}) is None


def test_smoke_failure_does_not_disable_independent_methods():
    tests=[dict(name=n,id=i,unpruned=u,passed=True) for n in ('mass','value')
        for i in ('aime','v2') for u in (True,False)]
    tests[0]['passed']=False
    assert smoke_verified(tests,['mass','value'],['aime','v2'])==['value']
    assert smoke_verified(tests[:-1],['mass','value'],['aime','v2'])==[]


def test_control_cdf_refuses_final_data_before_io(tmp_path):
    with pytest.raises(ValueError,match='only calibration'):
        policies.distributions(tmp_path,[dict(split='final')],'no_value_control',{}, {})


def test_audit_point_checks_split_fingerprint_hash_and_raw_counts(tmp_path,monkeypatch):
    row=dict(id='x',split='calibration',benchmark='longbench_v2')
    raw=dict(records=[dict(probe='execution',attention_type=k,eligible=10,skipped=5) for k in policies.KINDS])
    path=tmp_path/'raw.json';path.write_text(json.dumps(raw))
    source=dict(path=str(path),sha256=sha(path.read_bytes()),fingerprint='new',stage='calibration')
    point=dict(source_ids=['x'],sources={'x':source},policy={},achieved={'local':.5,'global':.5})
    execution=dict(fingerprint='new',previous_fingerprint='old')
    monkeypatch.setattr(policies,'check_result',lambda *a,**kw:None)
    assert policies.audit_point(point,[row],dict(method='mass'),execution)['global']['physical_sparsity']==.5
    bad=copy.deepcopy(point);bad['sources']['x']['stage']='final'
    with pytest.raises(ValueError,match='development/final'):policies.audit_point(bad,[row],dict(method='mass'),execution)
    bad=copy.deepcopy(point);bad['sources']['x']['fingerprint']='old';bad['imported_policy']={}
    with pytest.raises(ValueError,match='incompatible imported'):policies.audit_point(bad,[row],dict(method='mass'),execution)
    bad=copy.deepcopy(point);bad['sources']['x']['sha256']='bad'
    with pytest.raises(ValueError,match='source changed'):policies.audit_point(bad,[row],dict(method='mass'),execution)
    bad=copy.deepcopy(point);bad['achieved']['local']=.9
    with pytest.raises(ValueError,match='raw physical'):policies.audit_point(bad,[row],dict(method='mass'),execution)
