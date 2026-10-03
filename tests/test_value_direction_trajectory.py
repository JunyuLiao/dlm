import torch

from experiments.value_direction_hopper.trajectory import distribution, SamplerProbe, StopProbe
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler, EntropyBoundSamplerConfig, StableAndConfidentStoppingCriteria,
)


def test_confidence_and_margin():
    x=torch.tensor([[[0.,0.],[0.,2.]]])
    result=distribution(x)
    assert abs(result['confidence'][0]-.5)<1e-6
    assert result['margin'][0]==0
    assert result['top1'][0]==0
    assert result['top1'][1]==1
    assert abs(result['margin'][1]-.761594)<1e-5


def test_acceptance_observation_preserves_rng_and_reversible_selection():
    sampler=EntropyBoundSampler(EntropyBoundSamplerConfig(entropy_bound=.1),3,3,48)
    record={}
    probe=SamplerProbe(sampler,record)
    x=torch.tensor([[[15.,0.,0.],[15.,0.,0.],[0.,0.,0.]]])
    state=torch.random.get_rng_state().clone()
    probe.accept_canvas(torch.zeros((1,3),dtype=torch.long),torch.ones((1,3),dtype=torch.long),x,48)
    assert torch.equal(state,torch.random.get_rng_state())
    assert record['accepted']==[True,True,True]  # Largest entropy is excluded from bound.
    x=torch.zeros_like(x)
    probe.accept_canvas(torch.zeros((1,3),dtype=torch.long),torch.ones((1,3),dtype=torch.long),x,47)
    assert sum(record['accepted'])==1


def test_fixed_step_suppresses_return_but_preserves_stopper_history():
    stopper=StableAndConfidentStoppingCriteria(1,.005)
    record={}
    probe=StopProbe(stopper,record,True)
    x=torch.tensor([[[20.,0.],[0.,20.]]])
    canvas=x.argmax(-1)
    assert not probe(canvas,x).item()
    assert not record['would_stop']
    assert not probe(canvas,x).item()
    assert record['would_stop'] and record['stable']
    assert record['unstable_positions']==[False,False]


def test_stopping_boundary_uses_native_entropy_dtype():
    from experiments.value_direction_hopper.trajectory_report import native_confidence
    values=torch.tensor([.005,.005,.005,.00497],dtype=torch.bfloat16)
    assert float(values.float().mean())<.005
    _,confident=native_confidence(values.tolist(),'torch.bfloat16',.005)
    assert not confident
