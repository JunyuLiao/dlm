import math
import torch
from experiments.numerical_qk_reuse.multi_model_blasst import select_maxima,reference


def test_lambda_above_one_skips_new_max_but_keeps_first_and_current():
    maximum=torch.tensor([[[1.,1.],[1.25,1.25],[2.,2.]]])
    mandatory=torch.tensor([[False,False,True]])
    assert select_maxima(maximum,math.log(2.),mandatory).tolist()==[[True,False,True]]


def test_skipped_tile_updates_seen_maximum():
    maximum=torch.tensor([[[0.],[.5],[1.]]]);mandatory=torch.zeros((1,3),dtype=torch.bool)
    assert select_maxima(maximum,.6,mandatory).tolist()==[[True,False,False]]


def test_tie_and_one_query_dissent_retain():
    maximum=torch.tensor([[[0.,0.],[.5,.6],[.5,.6]]]);mandatory=torch.zeros((1,3),dtype=torch.bool)
    assert select_maxima(maximum,.6,mandatory).tolist()==[[True,True,False]]


def test_padding_first_support_dense_and_invalid_fail_closed():
    maximum=torch.tensor([[[-math.inf,0.],[-math.inf,-1.],[0.,-2.]]]);mandatory=torch.zeros((1,3),dtype=torch.bool)
    assert select_maxima(maximum,10.,mandatory).tolist()==[[True,False,True]]
    assert select_maxima(maximum,-math.inf,mandatory).all()
    maximum[0,1,0]=math.nan
    assert select_maxima(maximum,100.,mandatory).all()


def test_vectorized_matches_sequential_random_rules():
    torch.manual_seed(42);maximum=torch.randn((4,19,128));maximum[:,:,7:]=-math.inf
    mandatory=torch.zeros((4,19),dtype=torch.bool);mandatory[:,-1]=True
    for log_lambda in (-math.inf,-2.,0.,1.,4.):
        assert torch.equal(select_maxima(maximum,log_lambda,mandatory),reference(maximum,log_lambda,mandatory))
