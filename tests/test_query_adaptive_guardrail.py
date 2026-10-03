import torch

from experiments.value_direction_hopper.query_adaptive_guardrail import (
    PhaseState, initial_policies, phase_policy, pair, profile, violations,
)


class Router:
    query_sensitivity = None
    policy_selector = None


def test_phase_thresholds_and_early_unit_weights():
    policy=phase_policy(pair(-1.,-2.),pair(.1,-.2))
    canvas=torch.zeros((1,256),dtype=torch.int64)
    router=Router()
    state=PhaseState('C',router,m_ref=1.,beta=3.,gamma=.5,
        allocation='normal',bootstrap=False,diagnostics=False,seed=42,
        phase_thresholds=policy)
    state.begin(48,canvas)
    assert router.policy_selector(48,'local')==policy['early']['local']
    assert router.query_sensitivity is None
    logits=torch.zeros((1,256,3));logits[...,0]=2
    state.observe_logits(logits,torch.ones_like(canvas,dtype=torch.bool),48)
    state.begin(47,canvas)
    assert router.policy_selector(47,'global')==policy['early']['global']
    assert router.query_sensitivity is None
    state.observe_logits(logits,torch.ones_like(canvas,dtype=torch.bool),47)
    state.begin(46,canvas)
    assert router.policy_selector(46,'local')==policy['late']['local']
    assert router.query_sensitivity is not None
    assert torch.all(router.query_sensitivity>=1)
    state.begin(48,canvas)
    assert router.query_sensitivity is None
    assert state.iteration==1


def test_tile_mean_broadcast_only_in_late_phase():
    policy=phase_policy(pair(-1.,-2.),pair(.1,-.2))
    canvas=torch.zeros((1,256),dtype=torch.int64)
    router=Router()
    state=PhaseState('T',router,m_ref=1.,beta=3.,gamma=.5,
        allocation='normal',bootstrap=False,diagnostics=False,seed=42,
        phase_thresholds=policy,tile_mode='mean')
    a=torch.zeros((1,256,3));a[...,0]=2
    state.begin(48,canvas);state.observe_logits(a,torch.ones_like(canvas,dtype=torch.bool),48)
    state.begin(47,canvas)
    b=a.clone();b[:,:64,0]=0;b[:,:64,1]=3
    state.observe_logits(b,torch.ones_like(canvas,dtype=torch.bool),47)
    state.begin(46,canvas)
    assert torch.allclose(router.query_sensitivity[:,:128],torch.full((1,128),1.75))
    assert torch.all(router.query_sensitivity[:,128:]==1)


def test_profile_pools_physical_counts_and_rejects_early_overpruning():
    def counts(eligible,skipped):
        return {k:dict(eligible=eligible,skipped=skipped)
                for k in ('whole','local','global')}
    row=dict(steps=3,score=1.,counts=counts(210,149),step_records=[
        dict(iteration=1,counts=counts(10,9),accepted=1,processed_entropy_mean=3.),
        dict(iteration=2,counts=counts(100,80),accepted=1,processed_entropy_mean=2.),
        dict(iteration=3,counts=counts(100,60),accepted=200,processed_entropy_mean=.1)])
    x=profile([row])
    assert x['overall']['whole']==149/210
    assert x['phase_sparsity']['step1']['whole']==.9
    assert 'step1_whole' in violations(x,70)
    assert x['accepted_mean']['step1']==1


def test_initial_grid_contains_archived_and_fresh_policies():
    policies=initial_policies(70)
    assert len(policies)>=5
    assert all(p['early']==p['late'] for p in policies)
    assert any(abs(p['early']['local']['log_threshold']+.08442977964878079)<1e-8
               for p in policies)
