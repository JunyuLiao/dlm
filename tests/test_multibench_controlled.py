from types import SimpleNamespace
import pytest
import torch
from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DStats, Blasst2DRuntime, dense_eager_attention_forward, blasst_2d_attention_forward
from experiments.diffusion_attention_threshold_modeling.routing import FreshRoutingAttention
from experiments.diffusion_gemma_solattn_blasst_multibench.diagnostics import DenseDiagnostics, sol_config
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_dataset import take, score
from experiments.diffusion_gemma_solattn_blasst_multibench.config import condition_map, BETAS


@pytest.mark.parametrize('local',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_dense_counterfactual_matches_existing_routers(local,dtype):
    torch.manual_seed(42)
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.randn(1,2,65,4,dtype=dtype);k=torch.randn(1,1,257,4,dtype=dtype);v=torch.randn_like(k)
    valid=torch.ones(1,1,65,257,dtype=torch.bool)
    valid[...,0,:]=False
    if local: valid[...,:128]=False
    kw=dict(scaling=.5,is_causal=False,sliding_window=128 if local else None)
    diag=DenseDiagnostics();out,_=diag(module,q,k,v,valid,**kw)
    dense,_=dense_eager_attention_forward(module,q,k,v,valid,**kw)
    assert torch.equal(out,dense)
    for name,condition in condition_map().items():
        expected=diag.calls[name][0]
        if condition.method=='sol':
            router=FreshRoutingAttention(sol_config(condition.target_sparsity));router(module,q,k,v,valid,**kw)
            actual=router.stats.per_call[0]
            assert expected['eligible_tiles']==actual['physical_total_tiles']
            assert expected['skipped_tiles']==actual['physical_skipped_tiles']
            assert expected['retained_dense_attention_mass']==pytest.approx(actual['retained_dense_attention_mass'],abs=2e-6)
        elif condition.method=='blasst':
            runtime.config=diag.bconfigs[name]
            from dataclasses import replace
            runtime.config=replace(runtime.config,collect_blasst_stats=True)
            runtime.stats=Blasst2DStats()
            blasst_2d_attention_forward(module,q,k,v,valid,**kw)
            actual=runtime.stats.summary()
            assert expected['eligible_tiles']==actual['eligible_tiles']
            assert expected['skipped_tiles']==actual['skipped_tiles']
            assert expected['retained_dense_attention_mass']==pytest.approx(actual['retained_dense_attention_mass'],abs=2e-6)
    assert all(c[0]['eligible_tiles']==diag.calls['dense'][0]['eligible_tiles'] for c in diag.calls.values())


def test_fixed_gaussian_and_existing_length_rule():
    from scipy.stats import norm
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _policy_for_target
    for s,beta in BETAS.items(): assert beta==pytest.approx(norm.ppf(s),abs=1e-6)
    c=Blasst2DConfig(length_aware_policy=_policy_for_target(.25))
    assert c.lambda_for('global',0,20000)==pytest.approx(c.lambda_for('global',0,10000)/2)
    c=Blasst2DConfig(length_aware_policy=_policy_for_target(.90))
    assert c.lambda_for('local',0,10000)==1.
    assert c.lambda_for('global',0,1)==1.


def test_deterministic_sample_selection_and_fractional_ruler():
    rows=[{'_id':str(i)} for i in range(20)]
    assert take(rows,5,'task')==take(rows[::-1],5,'task')
    value=score(dict(benchmark='ruler16k',task_base='niah',expected=['alpha','beta']), 'alpha')
    assert value==.5


def test_report_uses_summed_tiles_rows_and_union_token_positions():
    from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate, token_counts
    rows=[dict(attention_type='local',eligible_tiles=2,skipped_tiles=1,valid_rows=1,retained_dense_attention_mass=.9),
          dict(attention_type='global',eligible_tiles=10,skipped_tiles=9,valid_rows=9,retained_dense_attention_mass=.5)]
    r=aggregate(rows)
    assert r['overall']['full_tile_sparsity']==pytest.approx(10/12)
    assert r['overall']['retained_dense_attention_mass']==pytest.approx(.54)
    assert r['local']['full_tile_sparsity']==.5
    assert r['global']['full_tile_sparsity']==.9
    assert token_counts([1,2,3,4],[1,9,3])==(2,4)
    assert token_counts([],[])==(0,0)
