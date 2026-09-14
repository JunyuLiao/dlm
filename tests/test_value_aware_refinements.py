import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dllm.attention.blasst.core import Blasst2DConfig,Blasst2DRuntime,dense_eager_attention_forward
from experiments.diffusion_gemma_value_aware.operators import (Config,block_state,value_summaries,streaming_mask,reference_output,screen_risks)
from experiments.diffusion_gemma_value_aware.refinements import refined_mask,refined_screen_risks,RefinedAttention
from experiments.diffusion_gemma_value_aware.analysis import ratio_from_risks


def test_exact_mass_corrects_peaky_block_bound_and_accounts_softmax():
    s=torch.zeros(1,1,2,128);s[...,65:]=-20
    valid=torch.ones_like(s,dtype=torch.bool);v=torch.randn(1,1,128,4)
    meta=value_summaries(v,valid.any(-2));state=block_state(s,valid,v)
    plain=Config(method='mass',log_threshold=math.log(.1))
    exact=Config(method='mass',mode='exact_mass',log_threshold=math.log(.1))
    assert streaming_mask(state,meta,plain).tolist()==[[[False,False]]]
    assert refined_mask(state,meta,exact).tolist()==[[[False,True]]]
    risk,_=refined_screen_risks(state,meta,exact)
    assert risk[0,0,1].exp()==pytest.approx(1/65,rel=1e-5)


def test_exact_mass_log_probability_keeps_near_one_resolution():
    s=torch.zeros(1,1,1,128);s[...,64:]=20.
    valid=torch.ones_like(s,dtype=torch.bool);v=torch.ones(1,1,128,2)
    state=block_state(s,valid,v);meta=value_summaries(v,valid.any(-2))
    r,_=refined_screen_risks(state,meta,Config(method='mass',mode='exact_mass'))
    assert r[0,0,1]<0
    assert r[0,0,1].item()==pytest.approx(-math.log1p(math.exp(-20)),rel=2e-6)


@pytest.mark.parametrize('mode',['exact_mass','previous_output','previous_output_exact_mass'])
def test_refined_operator_unpruned_nonempty_and_stable(mode):
    torch.manual_seed(5)
    s=torch.randn(1,2,8,193)*20;v=torch.randn(1,2,193,4)
    valid=torch.ones_like(s,dtype=torch.bool);valid[...,0,:]=False;valid[...,1,:64]=False
    state=block_state(s.masked_fill(~valid,-torch.inf),valid,v);meta=value_summaries(v,valid.any(-2))
    method='mass' if mode=='exact_mass' else 'centered'
    prev=torch.randn(1,2,8,4)
    assert not refined_mask(state,meta,Config(method=method,mode=mode,log_threshold=-math.inf),prev).any()
    c=Config(method=method,mode=mode,log_threshold=100.)
    mask=refined_mask(state,meta,c,prev)
    assert (state['active']&~mask[...,None,:]).any(-1).equal(valid.any(-1))
    assert torch.isfinite(reference_output(s,valid,v,mask,c,meta['vectors'])).all()


def test_previous_output_changes_only_the_available_output_estimate():
    s=torch.zeros(1,1,1,128);valid=torch.ones_like(s,dtype=torch.bool)
    v=torch.ones(1,1,128,1);v[...,:64,:]=-1
    state=block_state(s,valid,v);meta=value_summaries(v,valid.any(-2))
    c=Config(method='centered',mode='previous_output',log_threshold=math.log(.1))
    assert refined_mask(state,meta,c,None).tolist()==[[[False,False]]]
    assert refined_mask(state,meta,c,torch.ones(1,1,1,1)).tolist()==[[[False,True]]]
    # An arbitrary current dense output stored in diagnostics is not consulted.
    state['current_dense_output']=torch.full((1,1,1,1),1e6)
    assert refined_mask(state,meta,c,None).tolist()==[[[False,False]]]


def test_previous_cache_invalidates_canvas_and_step_gaps():
    torch.manual_seed(1)
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.randn(1,1,3,4);k=torch.randn(1,1,128,4);v=torch.randn_like(k)
    valid=torch.ones(1,1,3,128,dtype=torch.bool)
    r=RefinedAttention(dict(method='centered',mode='previous_output',log_threshold=-100.))
    reference=dense_eager_attention_forward(module,q,k,v,valid,is_causal=False)[0]
    for step in (0,1,3):
        runtime.current_denoising_iteration=step
        assert torch.equal(r(module,q,k,v,valid,is_causal=False)[0],reference)
    assert r.previous_calls==1 and r.warmup_calls==2
    runtime.current_denoising_iteration=4
    r(module,q,k[...,:127,:],v[...,:127,:],valid[...,:127],is_causal=False)
    assert r.previous_calls==1 and r.warmup_calls==3


def test_scoped_injection_does_not_modify_base_router():
    from experiments.diffusion_gemma_value_aware import routing
    original=routing.streaming_mask
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.ones(1,1,2,2);k=torch.ones(1,1,128,2);v=torch.ones_like(k)
    mask=torch.ones(1,1,2,128,dtype=torch.bool)
    r=RefinedAttention(dict(method='mass',mode='exact_mass',log_threshold=10.))
    r(module,q,k,v,mask,is_causal=False)
    assert routing.streaming_mask is original
    assert sum(x['skipped'] for x in r.records)>0
    assert sum(x['softmax_skipped'] for x in r.records)==0


def test_block_metadata_reconstruction_from_aligned_risks():
    torch.manual_seed(9)
    s=torch.randn(1,2,3,128);valid=torch.ones_like(s,dtype=torch.bool);v=torch.randn(1,2,128,5)
    state=block_state(s,valid,v);meta=value_summaries(v,valid.any(-2))
    mass,_=screen_risks(state,meta,Config(method='mass'))
    value,_=screen_risks(state,meta,Config(method='mass_value',pooling='vector_mean'))
    ratios=ratio_from_risks(value.numpy(),mass.numpy())
    expected=(meta['vector_mean']/meta['ref'][...,None])[...,1:].flatten().numpy()
    assert np.allclose(ratios,expected,atol=1e-6)


def test_targeted_observer_preserves_dense_and_serializes_diagnostics():
    import json
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    torch.manual_seed(17)
    q=torch.randn(1,1,3,4);k=torch.randn(1,1,128,4);v=torch.randn_like(k)
    mask=torch.ones(1,1,3,128,dtype=torch.bool)
    r=RefinedAttention(screen=True)
    dense=dense_eager_attention_forward(module,q,k,v,mask,is_causal=False)[0]
    out=r(module,q,k,v,mask,is_causal=False)[0]
    assert torch.equal(out,dense)
    assert len(r.risk_arrays)==3 and r.refinement_diagnostics
    json.dumps(r.refinement_diagnostics,allow_nan=False)


def test_process_wait_fallback_checks_real_identity():
    import os
    from experiments.diffusion_gemma_value_aware.queue import process_state,same_live_process
    pid=os.getpid();state=process_state(pid)
    assert state is not None and same_live_process(pid,state['start_ticks'])
    assert not same_live_process(pid,'incorrect_start_time')
    assert process_state(2**30) is None


def test_unattainable_local_target_does_not_force_global_to_one():
    from experiments.diffusion_gemma_value_aware.repair_original import boundary_plan
    trace=[dict(policy={'local':dict(log_scale=8.,cap_one=True),'global':dict(log_scale=7.9,cap_one=True)},
                achieved={'local':.245,'global':.475}),
           dict(policy={'local':dict(log_scale=80.,cap_one=True),'global':dict(log_scale=80.,cap_one=True)},
                achieved={'local':.246,'global':.766})]
    policy,fixed,boundary=boundary_plan(trace,.5)
    assert fixed=={'local':True,'global':False}
    assert policy['local']['log_threshold']==0 and policy['local']['unattainable']
    assert policy['global']['log_scale']==7.9
    assert boundary['global']==.766


def test_both_unattainable_types_are_independently_capped():
    from experiments.diffusion_gemma_value_aware.repair_original import boundary_plan
    trace=[dict(policy={k:dict(log_scale=80.,cap_one=True) for k in ('local','global')},
                achieved={'local':.26,'global':.36})]
    policy,fixed,_=boundary_plan(trace,.75)
    assert all(fixed.values()) and all(e['log_threshold']==0 for e in policy.values())


def test_cap1_repair_reuses_joint_boundary_when_only_local_is_unattainable(tmp_path,monkeypatch):
    import json
    from experiments.diffusion_gemma_value_aware import repair_original as repair
    boundary={k:dict(log_scale=80.,cap_one=True) for k in ('local','global')}
    old=dict(name='blasst_original',benchmark='longbench',target=.75,fingerprint='fp',
        config=dict(method='blasst'),policy=boundary,measured={'global':.766,'local':.246},
        trace=[dict(iteration=0,policy=boundary,achieved={'global':.766,'local':.246},
            source_condition='historical_joint_one')])
    path=tmp_path/'verified_policies'/'longbench'/'blasst_original_s75.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(old))
    (tmp_path/'job.json').write_text(json.dumps(dict(pid=123)))
    monkeypatch.setattr(repair,'prepare',lambda root:dict(calibration=[
        dict(benchmark='longbench',prompt_tokens=100,generation_budget=32)]))
    monkeypatch.setattr(repair,'fingerprint',lambda root:'fp')
    monkeypatch.setattr(repair,'process_state',lambda pid:None)
    def no_inference(*args,**kwargs):raise AssertionError('identical boundary must reuse cached evidence')
    monkeypatch.setattr(repair,'create_adapter',no_inference)
    repair.repair(tmp_path)
    new=json.loads(path.read_text())
    assert new['trace'][0]['reused_exact_lambda1_source']=='historical_joint_one'
    assert new['independent_cap1_repair']['fixed_at_one']=={'local':True,'global':False}
    assert new['within_two_points_on_attainable_types']
    assert new['policy']['local']['unattainable'] and 'unattainable' not in new['policy']['global']


def test_new_calibration_search_keeps_feasible_type_while_probing_boundary():
    from experiments.diffusion_gemma_value_aware.policy_search import next_policy,select_point
    p={k:dict(log_scale=7.,cap_one=True) for k in ('local','global')}
    trace=[dict(policy=p,achieved={'local':.245,'global':.495})]
    new=next_policy(trace,.5,original=True,upper=10.)
    assert new['local']['log_scale']==10. and new['local']['lambda_at_one']
    assert new['global']==p['global']
    trace.append(dict(policy=new,achieved={'local':.246,'global':.505}))
    best,fixed=select_point(trace,.5,original=True,upper=10.)
    assert best is trace[1] and fixed=={'local':True,'global':False}
    updated=next_policy(trace,.5,original=True,upper=10.)
    assert updated['local']['unattainable'] and updated['global']==p['global']


def test_cap_boundary_does_not_dominate_attainable_type_selection():
    from experiments.diffusion_gemma_value_aware.policy_search import select_point
    trace=[dict(policy={'local':dict(log_scale=10.,cap_one=True),
                        'global':dict(log_scale=g,cap_one=True)},
                achieved={'local':.2,'global':s}) for g,s in [(10.,.77),(7.,.51),(6.,.49)]]
    best,fixed=select_point(trace,.5,original=True,upper=10.)
    assert best['policy']['global']['log_scale']!=10.
    assert fixed=={'local':True,'global':False}


def test_refinement_dispatch_and_policy_source_are_explicit():
    from experiments.diffusion_gemma_value_aware.execution import execution_cache,provenance,assert_policy_provenance
    from experiments.diffusion_gemma_value_aware.run import cache
    from experiments.diffusion_gemma_value_aware.refinements import cache_refined
    c=dict(method='mass',mode='exact_mass')
    assert execution_cache(c) is cache_refined and execution_cache(dict(method='mass')) is cache
    p=dict(fingerprint='fp',config=c,**provenance(c))
    assert_policy_provenance(p,c,'fp')
    with pytest.raises(RuntimeError,match='source mismatch'):
        assert_policy_provenance(dict(p,refinement_sha256='old'),c,'fp')


def test_refined_policy_requires_matching_cuda_smoke(tmp_path):
    import json
    from experiments.diffusion_gemma_value_aware.execution import require_refinement_smoke,provenance
    c=dict(method='mass',mode='exact_mass')
    require_refinement_smoke(tmp_path,'fp',[{}])
    (tmp_path/'refinement_smoke.json').write_text(json.dumps(dict(passed=True,fingerprint='fp',**provenance(c))))
    require_refinement_smoke(tmp_path,'fp',[c])
    with pytest.raises(RuntimeError):require_refinement_smoke(tmp_path,'wrong',[c])


def test_report_region_work_sums_and_marginals_are_count_weighted():
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate,accumulate_marginals,SUM_FIELDS
    records=[dict(layer=0,head=0,step=0,attention_type='local',eligible=2,skipped=1,
                prefix_eligible=1,prefix_skipped=1,canvas_eligible=1,canvas_skipped=0,
                pv_omitted=1,softmax_skipped=1,rows=1,mass_sum=.9,denominator_mass_sum=.9),
             dict(layer=1,head=0,step=0,attention_type='local',eligible=8,skipped=0,
                prefix_eligible=7,prefix_skipped=0,canvas_eligible=1,canvas_skipped=0,
                pv_omitted=0,softmax_skipped=0,rows=3,mass_sum=3,denominator_mass_sum=3)]
    direct=aggregate(records);merged=aggregate([aggregate([r]) for r in records])
    assert merged==direct and direct['physical_sparsity']==.1 and direct['prefix_sparsity']==.125
    assert direct['mass']==pytest.approx(.975)
    dimensions={};accumulate_marginals(dimensions,records,'bench','method')
    sums=dimensions['bench','method','head',0,'local']
    assert aggregate([dict(zip(SUM_FIELDS,sums))])==direct


def test_matched_comparison_requires_actual_sparsity_and_heldout_scope():
    from experiments.diffusion_gemma_value_aware.report_metrics import matched_blasst
    def row(name,method,s,accuracy,split='heldout24'):
        return dict(benchmark='aime26',split=split,condition=name,method=method,
            overall_physical_sparsity=s,accuracy=accuracy,target=.5)
    rows=[row('blasst_s50','blasst',.25,.5),row('value_s50','value',.5,.7),
        row('value_s25','value',.27,.6),row('value_full','value',.25,1.,split='full')]
    result=matched_blasst(rows)
    assert len(result)==2
    assert result[0]['score_delta'] is None and not result[0]['comparable_within_three_points']
    assert result[1]['score_delta']==pytest.approx(.1) and result[1]['comparable_within_three_points']


def test_descriptive_correlation_handles_constant_or_too_few_points():
    from experiments.diffusion_gemma_value_aware.report_metrics import correlation
    assert correlation([1,1,1],[1,2,3]) is None
    assert correlation([1,2],[2,4]) is None
    assert correlation([1,2,3],[2,4,6])==pytest.approx(1.)


def test_confirmation_reuses_frozen_thresholds_but_has_a_distinct_seed():
    from experiments.diffusion_gemma_value_aware.confirmation import build_contract
    final=dict(conditions={'dense':dict(config={},target=0.),
        'value_s50':dict(config=dict(method='value'),thresholds={'aime26':{'local':1,'global':2}})})
    result=build_contract(final,'fp',['value_s50'],10000)
    assert list(result['conditions'])==['dense','value_s50'] and not result['retuned']
    assert result['conditions']['value_s50']==final['conditions']['value_s50']
    assert result['conditions']['dense']['config']==dict(method='dense')
    assert final['conditions']['dense']['config']=={}
    assert result['expected_per_condition']=={'aime26':30,'longbench':50}
    with pytest.raises(ValueError):build_contract(final,'fp',['value_s50'],0)
    with pytest.raises(ValueError):build_contract(final,'fp',['new_unfrozen_method'],10000)


def test_development_evidence_must_identify_exact_selected_policy():
    from experiments.diffusion_gemma_value_aware.development_report import selected_point
    p=dict(policy={'local':1,'global':2},selected_round=2,
        trace=[dict(iteration=1,policy={}),dict(iteration=2,policy={'local':1,'global':2})])
    assert selected_point(p) is p['trace'][1]
    with pytest.raises(ValueError):selected_point(dict(p,policy={}))
    with pytest.raises(ValueError):selected_point(dict(p,selected_round=3))


def test_scientific_report_renders_from_audited_synthetic_sums(tmp_path):
    import csv,json
    from experiments.diffusion_gemma_value_aware.report import summary,plot
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    from experiments.diffusion_gemma_value_aware.scientific_report import write_report
    stats=aggregate([dict(eligible=10,skipped=0,pv_omitted=0,rows=10,mass_sum=10,
        denominator_mass_sum=10,dense_sq=10,error_sq=0,prefix_eligible=8,canvas_eligible=2)])
    sample=dict(task='fixture',accuracy=1.,dense_accuracy=1.,matching=1,compared=1,exact_match=True,
        aggregates={k:stats for k in ('overall','global','local')},output_length=1,termination_reason='eos')
    rows=[dict(benchmark=b,split=s,condition='dense',method='dense',pooling=None,target=0.,beta=None,mode=None,
        thresholds=None,**summary([sample])) for b,s in [('aime26','full'),('aime26','heldout24'),('longbench','full')]]
    comp_records=[dict(eligible=10,skipped=0,pv_omitted=n,compensated=n,rows=10,mass_sum=10-n,
        denominator_mass_sum=10,dense_sq=10,error_sq=1,prefix_eligible=8,canvas_eligible=2) for n in (8,6)]
    comp_sample=dict(sample,aggregates=dict(overall=aggregate(comp_records),
        global_=aggregate(comp_records[:1]),local=aggregate(comp_records[1:])))
    comp_sample['aggregates']['global']=comp_sample['aggregates'].pop('global_')
    for b,s in [('aime26','full'),('aime26','heldout24'),('longbench','full')]:
        rows.append(dict(benchmark=b,split=s,condition='compensate_s75',method='compensate',pooling=None,
            target=.75,beta=None,mode=None,thresholds=None,**summary([comp_sample])))
    setup=dict(revision='synthetic-test-only',final=[dict(benchmark=b,task='fixture',prompt_tokens=10,generation_budget=32)
        for b in ('aime26','longbench')],calibration=[],development=[])
    contract=dict(conditions={'dense':dict(config={}), 'compensate_s75':dict(config=dict(method='compensate'))},sources={})
    (tmp_path/'audit.json').write_text(json.dumps(dict(complete=True,completed=4)))
    write_report(tmp_path,setup,contract,rows,[],[],[])
    text=(tmp_path/'report.md').read_text()
    assert 'decoder denoising-attention' in text and 'not a kernel benchmark' in text
    assert 'Mechanism-attribution limitation' in text and 'not a value-only intervention' in text
    assert 'same-threshold-family no-value control' in text
    assert 'PV-replacement budgets — separate from physical sparsity' in text
    with (tmp_path/'main_ablation.csv').open() as f:
        comp=next(r for r in csv.DictReader(f) if r['method']=='compensate_s75')
    assert comp['target_metric']=='pv_omission' and float(comp['physical'])==0.
    assert [float(comp[k]) for k in ('pv_omission','global_pv','local_pv')]==[70.,80.,60.]
    assert float(comp['retained_mass'])==100. and float(comp['exact_PV_mass'])==30.
    assert (tmp_path/'execution_work.csv').exists() and (tmp_path/'flashattention_compatibility.csv').exists()
    from experiments.diffusion_gemma_value_aware.pruning import evidence
    (tmp_path/'screening_pruning.json').write_text(json.dumps(dict(
        pruned_candidate_name='centered',scope='remaining75% calibration only',
        heldout_used=False,decision_time_utc='synthetic',aime26_s50=dict(score=0.))))
    with pytest.raises(ValueError,match='frozen provenance'):
        write_report(tmp_path,setup,contract,rows,[],[],[])
    _,contract['sources']=evidence(tmp_path)
    write_report(tmp_path,setup,contract,rows,[],[],[])
    text=(tmp_path/'report.md').read_text()
    assert 'Pruned development configurations' in text
    assert 'not completed final conditions' in text and 'remaining75% calibration only' in text
    plot(tmp_path,rows)
    assert (tmp_path/'figures'/'aime26_target_vs_actual.png').exists()


def test_pruning_evidence_includes_linked_sources_and_rejects_changed_policy(tmp_path):
    import hashlib,json
    from experiments.diffusion_gemma_value_aware.pruning import evidence
    (tmp_path/'policy.json').write_text('{}')
    (tmp_path/'findings.md').write_text('Calibration-only failure, not final evidence.')
    risk=dict(pruned_candidate_name='risk',scope='remaining75%',heldout_used=False,
        reason_source='findings.md',aime26_s50=dict(policy_source='policy.json',
            policy_sha256=hashlib.sha256(b'{}').hexdigest()))
    (tmp_path/'risk.json').write_text(json.dumps(risk))
    initial=dict(pruned_candidate_name='centered',scope='remaining75%',heldout_used=False,
        subsequent_decisions=[dict(source='risk.json',findings='findings.md',heldout_used=False)])
    (tmp_path/'screening_pruning.json').write_text(json.dumps(initial))
    decisions,sources=evidence(tmp_path)
    assert [d['pruned_candidate_name'] for d in decisions]==['centered','risk']
    assert {p.rsplit('/',1)[-1] for p in sources}=={'screening_pruning.json','risk.json','findings.md','policy.json'}
    (tmp_path/'policy.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='policy changed'):evidence(tmp_path)
    initial['heldout_used']=True
    (tmp_path/'screening_pruning.json').write_text(json.dumps(initial))
    with pytest.raises(ValueError,match='calibration-only'):evidence(tmp_path)


def test_policy_audit_recomputes_physical_counts_and_rejects_false_claims():
    from copy import deepcopy
    from experiments.diffusion_gemma_value_aware.policy_audit import verify_measurement
    row=dict(id='cal1',prompt_hash='hash',seed=42,prompt_tokens=100,generation_budget=32)
    thresholds={k:dict(log_threshold=-1.) for k in ('local','global')}
    p=dict(name='mass',fingerprint='fp',config=dict(method='mass'),policy=thresholds,
        calibration_ids=['cal1'],heldout_used=False,measured={'local':.25,'global':.5},selected_round=0,
        trace=[dict(iteration=0,policy=thresholds,achieved={'local':.25,'global':.5})])
    out=dict(row,fingerprint='fp',config=p['config'],thresholds=thresholds,
        records=[dict(probe='execution',attention_type=k,eligible=n,skipped=s) for k,n,s in [('local',8,2),('global',2,1)]])
    assert verify_measurement(p,[out],[row])['local']['physical_sparsity']==.25
    bad=deepcopy(p);bad['measured']['local']=.5
    with pytest.raises(ValueError,match='raw eligible/skipped counts'):verify_measurement(bad,[out],[row])
    with pytest.raises(ValueError,match='incomplete or duplicated'):verify_measurement(p,[out,out],[row])
    with pytest.raises(ValueError,match='seed mismatch'):verify_measurement(p,[dict(out,seed=99)],[row])
    with pytest.raises(ValueError,match='threshold mismatch'):verify_measurement(p,[dict(out,thresholds={})],[row])


def test_policy_audit_accepts_only_exact_boundary_metadata_reuse():
    from copy import deepcopy
    from experiments.diffusion_gemma_value_aware.policy_audit import verify_measurement
    row=dict(id='cal1',prompt_hash='hash',seed=42,prompt_tokens=100,generation_budget=32)
    old={k:dict(log_scale=80.,cap_one=True) for k in ('local','global')}
    new={k:dict(log_threshold=0.,cap_one=True,unattainable=True) for k in old}
    p=dict(name='blasst_original',fingerprint='fp',config=dict(method='blasst'),policy=new,target=.9,
        calibration_ids=['cal1'],heldout_used=False,measured={'local':.25,'global':.5},selected_round=0,
        trace=[dict(iteration=0,policy=new,achieved={'local':.25,'global':.5},reused_exact_lambda1_source='old')])
    out=dict(row,fingerprint='fp',config=p['config'],thresholds=old,
        records=[dict(probe='execution',attention_type=k,eligible=n,skipped=s) for k,n,s in [('local',8,2),('global',2,1)]])
    verify_measurement(p,[out],[row])
    bad=deepcopy(out);bad['thresholds']['local']['log_scale']=-1.
    with pytest.raises(ValueError,match='threshold mismatch'):verify_measurement(p,[bad],[row])


def test_rank_search_scales_updates_to_narrow_risk_distribution():
    from experiments.diffusion_gemma_value_aware.rank_calibration import next_policy
    values={k:np.linspace(-.01,-.001,1000) for k in ('local','global')}
    policy={k:dict(log_threshold=-.0055) for k in values}
    observations=[dict(policy=policy,achieved={'global':.62,'local':.498})]
    updated=next_policy(observations,.5,values)
    assert updated['local']==policy['local']
    assert -.01<updated['global']['log_threshold']<-.0055
    # The old generic update moved nearly one entire log unit for this error;
    # a CDF-rank update stays on the observed risk's sub-.01 scale.
    assert abs(updated['global']['log_threshold']-policy['global']['log_threshold'])<.002


def test_rank_search_brackets_in_cdf_coordinates_and_rejects_heldout(tmp_path):
    from experiments.diffusion_gemma_value_aware.rank_calibration import next_policy,distributions
    values={k:np.linspace(-.01,-.001,1000) for k in ('local','global')}
    observations=[dict(policy={k:dict(log_threshold=x) for k in values},achieved={k:s for k in values})
        for x,s in [(-.008,.1),(-.003,.8)]]
    result=next_policy(observations,.5,values)
    assert all(-.008<v['log_threshold']<-.003 for v in result.values())
    with pytest.raises(ValueError,match='only calibration'):
        distributions(tmp_path,[dict(split='final')],'mass',dict(method='mass'))


def test_resume_keeps_prior_immutable_and_recovers_unpublished_verified_points(tmp_path):
    import json
    from copy import deepcopy
    from experiments.diffusion_gemma_value_aware.rank_calibration import resume_observations
    prior=dict(trace=[dict(iteration=0,policy={'local':dict(log_threshold=-1.)})])
    original=deepcopy(prior);checkpoint=tmp_path/'trace.json'
    checkpoint.write_text(json.dumps(prior['trace']+[dict(iteration=1,policy={})]))
    observations=resume_observations(prior,checkpoint)
    assert len(observations)==2
    observations[0]['policy']['local']['log_threshold']=99
    observations.append(dict(iteration=2))
    assert prior==original
    checkpoint.write_text(json.dumps([dict(iteration=9)]))
    with pytest.raises(ValueError,match='not an extension'):resume_observations(prior,checkpoint)


def test_calibrator_publishes_completed_checkpoint_without_rerunning_gpu(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from experiments.diffusion_gemma_value_aware import evaluate as e
    from experiments.diffusion_gemma_value_aware.run import shard_path
    from experiments.diffusion_gemma_value_aware.protocol import sha
    rows=[dict(id=b+'/cal',benchmark=b,split='calibration',prompt_hash='h',seed=42,
        generation_budget=32,prompt_tokens=100) for b in ('aime26','longbench')]
    config=dict(method='mass');oldfiles={}
    for row in rows:
        b=row['benchmark'];points=[]
        for iteration,achieved in enumerate((.4,.51)):
            policy={k:dict(log_threshold=-1.+iteration*.1) for k in ('local','global')}
            condition=f'mass_s50/{b}/point{iteration}'
            point=dict(iteration=iteration,policy=policy,achieved={k:achieved for k in policy},
                error=abs(achieved-.5),source_ids=[row['id']],source_condition=condition)
            points.append(point)
            path=shard_path(tmp_path,'calibration',condition,row['id']);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(row,config=config,fingerprint='fp',thresholds=policy)))
        prior=dict(fingerprint='fp',name='mass',benchmark=b,target=.5,error=.1,config=config,
            trace=points[:1],policy=points[0]['policy'])
        path=tmp_path/'verified_policies'/b/'mass_s50.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(prior));oldfiles[b]=prior
        trace=tmp_path/'calibration_traces'/b/'mass_s50.json';trace.parent.mkdir(parents=True,exist_ok=True)
        trace.write_text(json.dumps(points))
    monkeypatch.setattr(e,'prepare',lambda root:dict(calibration=rows))
    monkeypatch.setattr(e,'fingerprint',lambda root:'fp')
    monkeypatch.setattr(e,'screen_summary',lambda root:None)
    monkeypatch.setattr(e,'candidate_configs',lambda root:dict(mass=config))
    monkeypatch.setattr(e,'create_adapter',lambda *a,**k:SimpleNamespace(load=lambda:object()))
    monkeypatch.setattr(e,'smoke',lambda *a:None)
    monkeypatch.setattr(e,'rank_distributions',lambda *a:({k:np.array([-1.,0.]) for k in ('local','global')},{'test':True}))
    def unexpected_gpu(*a,**k):raise AssertionError('completed verification was unnecessarily rerun')
    monkeypatch.setattr(e,'run_group',unexpected_gpu)
    e.calibrate(tmp_path,['mass'],[.5],3)
    for b in oldfiles:
        final=json.loads((tmp_path/'verified_policies'/b/'mass_s50.json').read_text())
        assert final['selected_round']==1 and final['error']==pytest.approx(.01)
        assert json.loads(__import__('pathlib').Path(final['previous_policy_archive']).read_text())==oldfiles[b]
