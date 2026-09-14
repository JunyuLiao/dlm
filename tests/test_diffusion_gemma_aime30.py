import math
import numpy as np
import pytest
import torch
from dllm.attention.blasst.core import evaluate_blasst_thresholds
from experiments.diffusion_gemma_aime30.calibration import physical_margins, calibrate
from experiments.diffusion_gemma_aime30.protocol import numeric_score, prompt


@pytest.mark.parametrize('text', [r'\boxed{073}', 'Final answer: 73.0', 'The answer is 7.3e1', r'\boxed{\frac{146}{2}}', 'Thus 73'])
def test_flexible_numeric(text): assert numeric_score(text,'73')['correct']


@pytest.mark.parametrize('text', [r'73 was an intermediate result. \boxed{74}', 'Final answer: -73', r'\boxed{73+1}', 'no answer'])
def test_no_gold_search_or_ambiguous_formula(text): assert not numeric_score(text,'73')['correct']


@pytest.mark.parametrize('q,k',[(65,193),(64,256),(3,7)])
def test_compressed_margins_match_physical_blasst(q,k):
    torch.manual_seed(42)
    scores=torch.randn(1,2,q,k)
    valid=torch.rand_like(scores)>.3;valid[...,0,:]=False;valid[...,:2]=False
    scores=scores.masked_fill(~valid,-torch.inf)
    margin,eligible=physical_margins(scores,valid)
    for lam,decision in evaluate_blasst_thresholds(scores,valid,None,[1e-8,.01,.4,1.],q_tile_size=64,kv_tile_size=64).items():
        assert torch.equal(eligible,decision.eligible_mask)
        assert torch.equal(eligible & (margin < math.log(lam)),decision.skip_mask)


def test_calibration_ceiling_and_target():
    values=np.asarray([-10.,-8.,-6.,-4.,-2.,-1.,0.,float('inf')],dtype=np.float32)
    p=calibrate(values)
    assert p['lambda1_ceiling']==.75
    assert p['targets']['0.9']['unattainable']
    assert p['targets']['0.9']['lambda_value']==1.
    for s in (.25,.5,.75):
        assert p['targets'][str(s)]['sparsity']==s
        assert not p['targets'][str(s)]['unattainable']


def test_prompt_demonstrations_identical_and_no_query_answer():
    demos=[dict(problem=f'demo {i}',reference_solution=f'Work {i}',expected_answer=str(i)) for i in range(5)]
    first=prompt('query one',demos);second=prompt('query two',demos)
    assert first.split('### Problem to solve')[0]==second.split('### Problem to solve')[0]
    assert first.count('### Worked example')==5
    assert prompt('query one',[]) .count('### Worked example')==0
