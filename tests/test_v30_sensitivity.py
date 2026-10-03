import math
from types import SimpleNamespace
import pytest
import torch

from experiments.value_direction_hopper.query_adaptive import State
from experiments.numerical_qk_reuse.v30_sensitivity import SensitivityOverride


def state():
    router=SimpleNamespace(query_sensitivity=None,carry_first=True,cache=object())
    return State('T',router,m_ref=1.,beta=3.,diagnostics=False,fast_t=True),router


def logits(probabilities):
    return torch.tensor([[[math.log(p),math.log((1-p)/2),math.log((1-p)/2)] for p in probabilities]])


def test_unit_uses_no_logits_or_acceptance_and_preserves_router_identity():
    s,r=state();cache=r.cache;override=SensitivityOverride(s,'unit_v30');canvas=torch.zeros((1,2),dtype=torch.long)
    for cur in (48,47,46):
        s.begin(cur,canvas);assert r.query_sensitivity is None
        s.observe_logits(object(),None,cur)  # Would fail if logits were read.
    assert r.cache is cache and r.carry_first is True and s.previous_top is None
    assert override.receipt()['unit_steps']==3 and override.receipt()['observe']==3
    override.close();assert not hasattr(s,'_v30_sensitivity_override')


def test_confidence_prior_uses_only_previous_call_and_resets_at_canvas():
    s,r=state();override=SensitivityOverride(s,'confidence_v30');canvas=torch.zeros((1,2),dtype=torch.long)
    s.begin(48,canvas);torch.testing.assert_close(r.query_sensitivity,torch.full((1,2),4.))
    x=logits([.5,.9]);before=x.clone();s.observe_logits(x,None,48)
    torch.testing.assert_close(x,before,rtol=0,atol=0)
    torch.testing.assert_close(r.query_sensitivity,torch.full((1,2),4.))
    s.begin(47,canvas)
    torch.testing.assert_close(r.query_sensitivity,1+3*torch.sqrt(torch.tensor([[.5,.1]])))
    s.observe_logits(logits([.8,.6]),None,47)
    s.begin(48,canvas);torch.testing.assert_close(r.query_sensitivity,torch.full((1,2),4.))
    assert override.receipt()['canvas_resets']==2 and override.receipt()['confidence_steps']==1


def test_observation_clock_duplicates_and_shape_fail_closed():
    s,r=state();SensitivityOverride(s,'confidence_v30');canvas=torch.zeros((1,2),dtype=torch.long)
    with pytest.raises(RuntimeError):s.observe_logits(logits([.5,.9]),None,48)
    s.begin(48,canvas)
    with pytest.raises(ValueError):s.observe_logits(logits([.5,.9]),None,47)
    with pytest.raises(ValueError):s.observe_logits(logits([.5]),None,48)
    s.observe_logits(logits([.5,.9]),None,48)
    with pytest.raises(RuntimeError):s.observe_logits(logits([.5,.9]),None,48)


def test_modes_and_unsupported_state_combinations_fail():
    s,r=state()
    with pytest.raises(ValueError):SensitivityOverride(s,'cgate')
    r.density_gate='stable1'
    with pytest.raises(ValueError):SensitivityOverride(s,'unit_v30')
    r.density_gate=None;SensitivityOverride(s,'unit_v30')
    with pytest.raises(RuntimeError):SensitivityOverride(s,'confidence_v30')


def test_confidence_rejects_mid_canvas_geometry_change_before_routing():
    s,r=state();SensitivityOverride(s,'confidence_v30')
    s.begin(48,torch.zeros((1,2),dtype=torch.long));s.observe_logits(logits([.5,.9]),None,48)
    with pytest.raises(ValueError,match='geometry changed'):
        s.begin(47,torch.zeros((1,3),dtype=torch.long))
    s.begin(48,torch.zeros((1,3),dtype=torch.long))
    torch.testing.assert_close(r.query_sensitivity,torch.full((1,3),4.))


@pytest.mark.parametrize('mode',(None,'unit_v30','confidence_v30'))
def test_vllm_dispatch_preserves_main_and_trims_only_new_variant_padding(mode):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    adapter=VllmMethodAdapter.__new__(VllmMethodAdapter)
    calls=[]
    adapter.calls=dict(observes=0,order_errors=0)
    adapter.pending_sample=True;adapter.step_ctx=dict(step=7,n=2)
    adapter.config={} if mode is None else dict(sensitivity=mode)
    adapter.runtime={'state':SimpleNamespace(observe_logits=lambda *args:calls.append(args))}
    x=torch.randn(1,4,3);before=x.clone()
    adapter.on_sample(x)
    assert len(calls)==1 and calls[0][1] is None
    assert calls[0][2]==(None if mode is None else 41)
    if mode is None:assert calls[0][0] is x
    else:torch.testing.assert_close(calls[0][0],x[:,:2,:])
    torch.testing.assert_close(x,before,rtol=0,atol=0)
    assert adapter.calls==dict(observes=1,order_errors=0)


@pytest.mark.parametrize('mode',('unit_v30','confidence_v30'))
def test_effective_configuration_keeps_m3_clocks_and_carry(mode):
    from experiments.numerical_qk_reuse import v21
    base=dict(diagnostic=False,policy={'local':{'log_threshold':-3.},'global':{'log_threshold':-2.}})
    line=dict(output_score_precision='fp32_scores_bf16_pv',bootstrap_policy='native_bootstrap2_observe1',
              output_layout='model_major',consumer64=2,memory_caps='long',fa4_consumer=True,score_period=64,
              fused_observe=True,async_route=True,risk_state='dense_prefix',decision_interval=6,
              threshold_shift='minus_ln2',carry_first=True)
    cfg=v21.effective_config(base,'M3_R3_A8_current_output','GLOBAL_ONLY_NATIVE_LOCAL',sensitivity=mode,**line)
    v21.validate_effective(cfg,cfg['condition'])
    assert cfg['sensitivity']==mode and cfg['score_period']==64 and cfg['decision_interval']==6 and cfg['carry_first']
    with pytest.raises(ValueError):
        v21.effective_config(base,'M3_R3_A8_current_output','GLOBAL_ONLY_NATIVE_LOCAL',sensitivity=mode,**dict(line,carry_first=False))
