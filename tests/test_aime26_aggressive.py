import math
import pytest
import torch
import numpy as np
from dllm.attention.blasst.core import Blasst2DConfig,evaluate_blasst_thresholds,apply_blasst_2d,slow_blasst_2d
from experiments.diffusion_gemma_aime26.calibration import margins,fit

def test_opt_in_finite_threshold_validation():
    with pytest.raises(ValueError): Blasst2DConfig(blasst_lambda=2.)
    assert Blasst2DConfig(blasst_lambda=2.,allow_lambda_above_one=True).log_lambda==math.log(2.)
    for x in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError): Blasst2DConfig(blasst_lambda=x,allow_lambda_above_one=True)

def test_128_query_votes_margins_and_nonempty_rows():
    torch.manual_seed(42)
    scores=torch.randn(1,2,129,259);valid=torch.ones_like(scores,dtype=torch.bool)
    valid[...,0,:64]=False;valid[...,1,:128]=False;valid[...,2,:]=False
    scores=scores.masked_fill(~valid,-torch.inf)
    m,e=margins(scores,valid)
    previous=0
    for lam in (.1,1.,2.,100.,1e20):
        c=Blasst2DConfig(blasst_lambda=lam,q_tile_size=128,kv_tile_size=64,allow_lambda_above_one=True)
        masked,d=apply_blasst_2d(scores,valid,None,c)
        slow,sd=slow_blasst_2d(scores,valid,None,c)
        assert torch.equal(masked,slow)
        assert torch.equal(d.skip_mask,sd.skip_mask)
        assert torch.equal(d.skip_mask,(m<math.log(lam))&e)
        assert torch.equal(torch.isfinite(masked).any(-1),valid.any(-1))
        assert int(d.skip_mask.sum())>=previous;previous=int(d.skip_mask.sum())

def test_calibration_reaches_targets_and_preserves_ceiling():
    a=np.r_[np.linspace(-5,20,9900),np.full(100,np.inf)]
    result=fit([a])
    for target,p in result['targets'].items():
        assert abs(p['achieved_dense_sparsity']-float(target))<.02
    limited=fit([np.r_[np.linspace(-5,20,6000),np.full(4000,np.inf)]])
    assert limited['targets']['0.9']['unattainable']

def test_new_maxima_skip_above_one_but_one_dissent_retains_tile():
    scores=torch.zeros(1,1,128,128)
    scores[...,:64]=1.
    scores[...,64:]=1.25  # New maxima, still below previous+log(2).
    config=Blasst2DConfig(blasst_lambda=2.,q_tile_size=128,kv_tile_size=64,allow_lambda_above_one=True)
    _,d=apply_blasst_2d(scores,None,None,config)
    assert d.skip_mask.tolist()==[[[[False,True]]]]
    scores[...,127,64:]=2.
    _,d=apply_blasst_2d(scores,None,None,config)
    assert not d.skip_mask.any()
    assert int(d.row_skippable[...,1].sum())==127
    assert int(d.row_total[...,1].sum())==128
