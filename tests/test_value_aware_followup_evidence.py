import copy
import json
import pytest

from experiments.diffusion_gemma_value_aware_followup import evidence
from experiments.diffusion_gemma_value_aware_followup.final import validate_scope,secondary_conditions,phase_targets
from experiments.diffusion_gemma_value_aware_followup.policies import METHODS,candidate_configs
from experiments.diffusion_gemma_value_aware.ranking_guards import GUARD


def sample():
    row=dict(id='a',benchmark='longbench_v2',task='Single-Document QA',seed=42,prompt_hash='p',generation_budget=128,expected='B')
    data=dict(id='a',seed=42,prompt_hash='p',generation_budget=128,config={},thresholds=None,
        generation_metadata=dict(thinking=False),prediction='The correct answer is (B)',score=1.,
        completion_tokens=[1,2,3],termination_reason='eos',records=[dict(probe='execution',
            attention_type=k,eligible=10,skipped=0,rows=1,mass_sum=1) for k in ('local','global')])
    return row,data


def test_pair_reuses_variable_length_positional_agreement_after_divergence():
    row,dense=sample();sparse=copy.deepcopy(dense);sparse['completion_tokens']=[1,9,3,4]
    sparse['config']=dict(method='value');result=evidence.pair(row,sparse,dense)
    assert result['matching']==2 and result['compared']==4
    assert not result['exact_match'] and result['parsed_answer']=='B'
    assert not result['unparsed_answer']


def test_pair_flags_unparsed_official_v2_answer_and_refuses_unrelated_changes():
    row,dense=sample();sparse=copy.deepcopy(dense)
    sparse.update(prediction='B',score=0.)
    result=evidence.pair(row,sparse,dense)
    assert result['unparsed_answer'] and result['accuracy']==0
    sparse['generation_metadata']['thinking']=True
    with pytest.raises(ValueError,match='unrelated decoding'):evidence.pair(row,sparse,dense)


def test_pair_rejects_changed_manifest_and_stale_score():
    row,dense=sample();sparse=copy.deepcopy(dense);sparse['seed']=0
    with pytest.raises(ValueError,match='prompt/seed'):evidence.pair(row,sparse,dense)
    sparse=copy.deepcopy(dense);sparse['score']=0
    with pytest.raises(ValueError,match='official rescoring'):evidence.pair(row,sparse,dense)


def test_report_reader_never_imports_v1_for_v2(tmp_path):
    row=dict(id='v2',benchmark='longbench_v2')
    (tmp_path/'imported_sources.json').write_text(json.dumps({'final/x/v2':dict(path='unread',fingerprint='old')}))
    with pytest.raises(ValueError,match='matching AIME'):
        evidence.read_result(tmp_path,row,'final','x',{},None,dict(fingerprint='new',previous_fingerprint='old'))


def test_source_snapshots_cannot_be_silently_overwritten():
    assert evidence.merge_sources({'p':'1'},{'q':'2'})=={'p':'1','q':'2'}
    with pytest.raises(ValueError,match='inconsistent'):evidence.merge_sources({'p':'1'},{'p':'2'})


def frozen(phase):
    conditions={'dense':dict(config={})}
    configs=candidate_configs(dict(selected={n:dict(pooling='rms') for n in ('value','mass_value','risk')}))
    for name in METHODS:
        for target in phase_targets(phase):
            method=configs[name]['method']
            conditions[f'{name}_s{round(target*100)}']=dict(target=target,config=configs[name],
                thresholds={'aime26':{},'longbench_v2':{}},
                target_metric='pv_omission' if method in ('compensate','zero_pv') else 'physical_sparsity')
    if phase=='full':conditions.update(secondary_conditions())
    return dict(phase=phase,conditions=conditions,expected_per_condition={'aime26':30,'longbench_v2':30},
        heldout_used_for_selection=False,aime24_previously_exposed=True)


def test_broad_and_full_contracts_cannot_silently_drop_families_or_aime_questions():
    broad=frozen('broad50');validate_scope(broad);assert len(broad['conditions'])==13
    full=frozen('full');validate_scope(full);assert len(full['conditions'])==75
    del full['conditions']['risk_s90']
    with pytest.raises(ValueError,match='no silent scope reduction'):validate_scope(full)
    broad['expected_per_condition']['aime26']=24
    with pytest.raises(ValueError,match='all30 AIME'):validate_scope(broad)


def test_pv_compensation_is_not_a_physical_sparsity_target():
    data=frozen('broad50');data['conditions']['compensate_s50']['target_metric']='physical_sparsity'
    with pytest.raises(ValueError,match='PV replacement'):validate_scope(data)


def test_secondary_scope_uses_guarded_declared_signals_and_nonnegative_top_p():
    conditions=secondary_conditions();assert len(conditions)==26
    assert all(c['config']['aggregation']==GUARD for c in conditions.values())
    topp=[c for c in conditions.values() if c['config']['mode']=='topp']
    assert len(topp)==4 and all(c['config']['pooling'] in ('mass','contribution') for c in topp)
    assert all(c['retained_nonnegative_signal'] in (.95,.99) and c['target'] is None for c in topp)


def test_scope_checks_family_operators_not_just_condition_names():
    data=frozen('broad50');data['conditions']['blasst_original_s50']['config']=dict(method='value')
    with pytest.raises(ValueError,match='family label'):validate_scope(data)


def test_queue_dependency_uses_actual_process_identity_not_pid_alone():
    from experiments.diffusion_gemma_value_aware_followup.workflow import same_live_process,stage_command
    old=dict(pid=12,start_ticks='100',state='S')
    assert same_live_process(old,dict(old))
    assert not same_live_process(old,None)
    assert not same_live_process(old,dict(old,start_ticks='200'))
    assert not same_live_process(old,dict(old,state='Z'))
    assert stage_command('development','root',[])[-3:]==['run','--output','root']
