from types import SimpleNamespace
import pytest
import torch
from experiments.diffusion_gemma_oracle.routing import select, signals, masked_diagnostics, OracleAttention
from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DRuntime, dense_eager_attention_forward, apply_blasst_2d


def test_budgets_ties_and_structural_exclusion():
    x=torch.tensor([[4.,4.,2.,1.,999.],[1.,2.,3.,4.,5.]])
    valid=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool)
    assert select(x,valid,'topk',.5).tolist()==[[True,True,False,False,False],[False,False,True,True,True]]
    assert select(x,valid,'topp',.7)[0].tolist()==[True,True,False,False,False]
    assert torch.equal(select(x,valid,'topk',0),valid)
    assert torch.equal(select(x*0,valid,'topp',.95),valid)


def test_exact_mass_contribution_and_renormalized_error():
    scores=torch.tensor([[[[0.,0.,0.,0.],[0.,0.,0.,0.]]]])
    valid=torch.ones_like(scores,dtype=torch.bool)
    values=torch.tensor([[[[2.,0.],[-2.,0.],[0.,1.],[0.,1.]]]])
    imp,eligible,mass,contrib,has,_=signals(scores,valid,values,width=2)
    assert imp['mass'].tolist()==[[[1.,1.]]]
    assert imp['contribution'][0,0,0]==0 # V cancellation matters
    keep=torch.tensor([[[False,True]]])
    d=masked_diagnostics(keep,mass,contrib,has)
    assert d['mass_sum'].item()==1
    assert d['error_sq'].item()==pytest.approx(.5)
    assert d['dense_sq'].item()==pytest.approx(.5)


def test_output_error_matches_explicit_multitile_mask():
    torch.manual_seed(123)
    scores=torch.randn(1,2,5,137)
    valid=torch.ones_like(scores,dtype=torch.bool)
    valid[...,0,:]=False;valid[...,1,:64]=False
    scores=scores.masked_fill(~valid,-torch.inf)
    values=torch.randn(1,2,137,7)
    imp,eligible,mass,contrib,has,_=signals(scores,valid,values)
    keep=torch.tensor([[[False,True,True],[False,True,True]]])
    diag=masked_diagnostics(keep,mass,contrib,has)
    token_keep=keep.repeat_interleave(64,-1)[...,:137]
    dense_probs=torch.softmax(torch.where(has[...,None],scores,0),-1)*valid
    sparse_probs=torch.softmax(torch.where(has[...,None],scores.masked_fill(~token_keep[...,None,:],-torch.inf),0),-1)*valid
    dense=dense_probs @ values;sparse=sparse_probs @ values
    assert torch.allclose(diag['error_sq'],(dense-sparse).square().sum((-1,-2)),atol=1e-6)
    assert torch.allclose(diag['mass_sum'],(dense_probs*token_keep[...,None,:]).sum((-1,-2)),atol=1e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_gqa_dense_parity_and_physical_mask(dtype):
    torch.manual_seed(7)
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,
        _blasst_2d_runtime=Blasst2DRuntime(Blasst2DConfig()))
    module._blasst_2d_runtime.current_denoising_iteration=0
    q=torch.randn(1,2,65,4,dtype=dtype);k=torch.randn(1,1,193,4,dtype=dtype);v=torch.randn_like(k)
    valid=torch.ones(1,1,65,193,dtype=torch.bool);valid[...,0,:]=False
    kw=dict(scaling=.5,is_causal=False)
    dense=dense_eager_attention_forward(module,q,k,v,valid,**kw)[0]
    for config in (dict(amount=0.),dict(observer=True)):
        router=OracleAttention(**config);output=router(module,q,k,v,valid,**kw)[0]
        assert torch.equal(dense,output)
        assert all(r['skipped']==0 for r in router.records if not config.get('observer') or r['probe']=='dense')
    router=OracleAttention(amount=.5);out=router(module,q,k,v,valid,**kw)[0]
    assert torch.isfinite(out).all()
    assert sum(r['skipped'] for r in router.records)==8
    assert sum(r['eligible'] for r in router.records)==16


def test_blasst_margin_reconstructs_physical_vote():
    torch.manual_seed(9)
    scores=torch.randn(1,2,64,256);valid=torch.ones_like(scores,dtype=torch.bool)
    valid[..., :64]=False
    values=torch.randn(1,2,256,4)
    imp,eligible,*_=signals(scores,valid,values)
    for lam in (.1,.5,1.):
        _,d=apply_blasst_2d(scores,valid,None,Blasst2DConfig(q_tile_size=64,kv_tile_size=64,blasst_lambda=lam))
        expected=eligible & (imp['blasst'] < __import__('math').log(lam))
        assert torch.equal(d.skip_mask[...,0,:],expected)


def test_disjoint_structural_support_rescued():
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,
        _blasst_2d_runtime=Blasst2DRuntime(Blasst2DConfig()))
    q=torch.ones(1,1,2,2);k=torch.ones(1,1,128,2);v=torch.ones_like(k)
    valid=torch.zeros(1,1,2,128,dtype=torch.bool)
    valid[...,0,:64]=True;valid[...,1,64:]=True
    router=OracleAttention(amount=.9);out=router(module,q,k,v,valid,is_causal=False)[0]
    assert torch.isfinite(out).all()
    assert router.records[0]['skipped']==0
    assert router.records[0]['rescued_rows']==1


def test_weighted_counts_and_ranking_overlap():
    from experiments.diffusion_gemma_oracle.report import summed,ranking_metrics
    records=[dict(eligible=4,skipped=2,mass_sum=1,rows=2,error_sq=1,dense_sq=4),
             dict(eligible=12,skipped=3,mass_sum=8,rows=8,error_sq=0,dense_sq=5)]
    m=summed(records);assert m['sparsity']==5/16;assert m['mass']==.9
    r=dict(layer=0,head=0,step=0,attention_type='global',scores=dict(blasst=[3,2,1],mass=[1,2,3]),retained={'0.5':[True,False,False]})
    metrics=ranking_metrics(r);mass=next(m for m in metrics if m['signal']=='mass')
    assert mass['rank_correlation']==-1;assert mass['budget_overlap']==0


def test_rank_correlation_matches_scipy_with_ties_and_infinity():
    from scipy.stats import spearmanr
    from experiments.diffusion_gemma_oracle.report import ranking_metrics
    x=[float('inf'),3,3,-2,1];y=[0,2,1,1,4]
    r=dict(layer=0,head=0,step=0,attention_type='local',scores=dict(blasst=x,mass=y),retained={'0.5':[True,True,False,False,False]})
    value=next(m for m in ranking_metrics(r) if m['signal']=='mass')['rank_correlation']
    assert value==pytest.approx(spearmanr(x,y).statistic,abs=1e-14)


def test_error_tolerance_frontier_does_not_assume_monotonicity():
    from experiments.diffusion_gemma_oracle.report import attention_tolerance_frontiers
    records=[]
    for s,error in ((.25,.1),(.5,.04),(.75,.2),(.9,.3)):
        records.append(dict(benchmark='aime24',id='a',attention_type='local',layer=0,head=0,step=0,
            condition='dense_observer',probe=f'mass_topk_{s}',eligible=100,skipped=100*s,
            mass_sum=1,rows=1,error_sq=error**2,dense_sq=1))
    rows=attention_tolerance_frontiers(records)
    result=next(r for r in rows if r['relative_error_tolerance']==.05)
    assert result['largest_tested_sparsity']==.5
    assert result['qualifying_sparse_points']==1
    assert next(r for r in rows if r['relative_error_tolerance']==.01)['largest_tested_sparsity']==0
