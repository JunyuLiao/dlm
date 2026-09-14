from types import SimpleNamespace
import pytest
import torch
from dllm.attention.blasst.core import Blasst2DConfig,Blasst2DRuntime,dense_eager_attention_forward
from experiments.diffusion_gemma_value_aware.operators import Config,proxy_keep
from experiments.diffusion_gemma_value_aware.ranking_guards import GUARD,signal_rescue,RankingAttention,JointScreen,diagnostic_signal


def test_signal_rescue_never_needs_dense_mass_and_respects_row_support():
    keep=torch.zeros(1,1,3,dtype=torch.bool)
    active=torch.tensor([[[[True,True,False],[True,False,True],[False,False,False]]]])
    signal=torch.tensor([[[1.,3.,2.]]])
    repaired,empty=signal_rescue(keep,active,signal)
    assert repaired.tolist()==[[[False,True,True]]]
    assert empty.tolist()==[[[True,True,False]]]
    # A stable tie picks the lowest supporting KV index.
    repaired,_=signal_rescue(keep,active,torch.zeros_like(signal))
    assert repaired.tolist()==[[[True,False,False]]]


def test_sol_gaussian_ninety_fallback_uses_proxy_not_attention_probability():
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.tensor([[[[1.],[-.9]]]])
    k=torch.cat([-torch.ones(1,1,64,1),torch.ones(1,1,64,1)],-2)
    v=torch.cat([torch.zeros(1,1,64,1),torch.ones(1,1,64,1)*2],-2)
    valid=torch.ones(1,1,2,128,dtype=torch.bool)
    c=dict(method='sol',mode='gaussian',amount=.9,aggregation=GUARD)
    router=RankingAttention(c)
    out=router(module,q,k,v,valid,is_causal=False)[0]
    # z={-1,+1}, beta1.2815 drops both. Mean-Q proxy prefers block1 for both
    # rows, even though row1's actual dense attention prefers block0.
    assert torch.allclose(out,torch.full_like(out,2.))
    assert sum(r['skipped'] for r in router.records)==1
    assert sum(r['rescued_rows'] for r in router.records)==2
    # Changing V cannot alter plain Sol's proxy mask/nonempty decision.
    altered=RankingAttention(c);altered(module,q,k,-v*100,valid,is_causal=False)
    assert sum(r['skipped'] for r in altered.records)==1


@pytest.mark.parametrize('method,pool',[('sol','rms'),('diagnostic','qk'),('diagnostic','mass'),('diagnostic','contribution')])
def test_guarded_unpruned_dense_parity_and_scoped_restore(method,pool):
    from experiments.diffusion_gemma_value_aware import routing
    original_rescue=routing.rescue;original_proxy=routing.proxy_keep
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    torch.manual_seed(9);q=torch.randn(1,2,129,4);k=torch.randn(1,1,193,4);v=torch.randn_like(k)
    mask=torch.ones(1,1,129,193,dtype=torch.bool)
    dense=dense_eager_attention_forward(module,q,k,v,mask,is_causal=False)[0]
    r=RankingAttention(dict(method=method,pooling=pool,mode='topk',amount=0.,aggregation=GUARD))
    assert torch.equal(r(module,q,k,v,mask,is_causal=False)[0],dense)
    assert routing.rescue is original_rescue and routing.proxy_keep is original_proxy


def test_qk_signal_does_not_read_mass_or_values():
    state={'b':torch.tensor([[[[1.,3.],[2.,1.]]]])}
    assert diagnostic_signal(state,'qk').tolist()==[[[2.,3.]]]


def test_mass_ranking_work_accounting_does_not_claim_softmax_skipping():
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.ones(1,1,2,2);k=torch.ones(1,1,128,2);v=torch.ones_like(k);valid=torch.ones(1,1,2,128,dtype=torch.bool)
    r=RankingAttention(dict(method='diagnostic',pooling='mass',mode='topk',amount=.5,aggregation=GUARD))
    r(module,q,k,v,valid,is_causal=False)
    assert sum(x['skipped'] for x in r.records)==1
    assert sum(x['softmax_skipped'] for x in r.records)==0


def test_joint_refinement_and_guarded_screen_preserves_dense():
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    torch.manual_seed(4);q=torch.randn(1,1,3,4);k=torch.randn(1,1,128,4);v=torch.randn_like(k)
    valid=torch.ones(1,1,3,128,dtype=torch.bool)
    r=JointScreen(screen=True)
    dense=dense_eager_attention_forward(module,q,k,v,valid,is_causal=False)[0]
    assert torch.equal(r(module,q,k,v,valid,is_causal=False)[0],dense)
    assert len(r.risk_arrays)==3
    probes={x['probe'] for x in r.records}
    assert sum(p.startswith('guarded_sol/') for p in probes)==16
    assert sum(p.startswith('guarded_diagnostic/') for p in probes)==12
    assert sum(p.startswith('refine/') for p in probes)==12


def test_selection_cost_prevents_contribution_ranking_from_claiming_pv_savings():
    from experiments.diffusion_gemma_value_aware.report_metrics import work_opportunities
    row=dict(method='diagnostic',pooling='contribution',mode='topk',overall_skipped=9,
        overall_pv_omitted=9,overall_softmax_skipped=0)
    result=work_opportunities(row)
    assert result['operator_full_PV_replaced_tiles']==9
    assert result['selection_requires_all_block_PV'] and result['selection_aware_full_PV_omission_tiles']==0
    result=work_opportunities(dict(row,pooling='mass'))
    assert not result['selection_requires_all_block_PV'] and result['selection_aware_full_PV_omission_tiles']==9
    assert result['selection_aware_softmax_omission_tiles']==0
