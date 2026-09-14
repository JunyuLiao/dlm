from copy import deepcopy
import pytest
from experiments.diffusion_gemma_value_aware.boundary_reuse import proves_exact_one,retarget


@pytest.mark.parametrize('entry,expected',[
    (dict(cap_one=True,log_threshold=0.),True),
    (dict(cap_one=True,log_scale=80.),True),
    (dict(cap_one=True,log_scale=4.,unattainable=True),True),
    (dict(cap_one=True,log_scale=4.,lambda_at_one=True),False),
    (dict(cap_one=False,log_scale=80.),False),
    (dict(cap_one=True,log_threshold=-.1),False),
    (dict(cap_one=True,log_scale=float('nan'),unattainable=True),False),
])
def test_boundary_proof_uses_executed_threshold_not_annotation(entry,expected):
    assert proves_exact_one(entry,10.)==expected


def source_fixture():
    policy={k:dict(log_scale=80.,cap_one=True) for k in ('local','global')}
    return dict(name='blasst_original',config=dict(method='blasst'),heldout_used=False,
        fingerprint='fp',benchmark='longbench',target=.75,policy=policy,selected_round=2,
        trace=[dict(iteration=2,policy=policy,source_condition='old/source')],calibration_ids=['cal1'])


def test_retarget_preserves_source_and_exposes_unattainable_reuse():
    source=source_fixture();before=deepcopy(source)
    metrics={k:dict(physical_sparsity=v) for k,v in [('global',.76),('local',.25)]}
    result=retarget(source,.9,metrics,10.,'source.json','digest')
    assert source==before and result['target']==.9 and result['config']==source['config']
    assert result['trace'][0]['reused_exact_lambda1_source']=='old/source'
    assert all(e['log_threshold']==0 and e['unattainable'] for e in result['policy'].values())
    assert result['cap_one_unattainable']=={'local':True,'global':True} and not result['within_two_points']
    assert result['boundary_reuse']['source_target']==.75


def test_retarget_refuses_feasible_type_wrong_method_and_heldout():
    source=source_fixture();metrics={k:dict(physical_sparsity=v) for k,v in [('global',.89),('local',.25)]}
    with pytest.raises(ValueError,match='both types'):retarget(source,.9,metrics,10.,'source','hash')
    metrics['global']['physical_sparsity']=.76
    with pytest.raises(ValueError):retarget(dict(source,heldout_used=True),.9,metrics,10.,'source','hash')
    with pytest.raises(ValueError):retarget(dict(source,name='blasst_aggressive'),.9,metrics,10.,'source','hash')
    source['policy']['global']['log_scale']=4.
    with pytest.raises(ValueError,match='proven joint'):retarget(source,.9,metrics,10.,'source','hash')
