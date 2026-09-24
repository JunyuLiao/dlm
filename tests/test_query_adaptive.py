import torch

from experiments.value_direction_hopper.query_adaptive import State,shuffle_within_tiles,weight
from experiments.value_direction_hopper.query_adaptive_replay import accepted_from_logits
from experiments.value_direction_hopper.query_adaptive_granularity import route_rows
from experiments.value_direction_hopper.query_adaptive_report import paired_interval


def test_weight_bounds_and_first_iteration():
    margin=torch.tensor([[0.,1.,10.]]);confidence=torch.tensor([[0.,.75,1.]])
    temporal=torch.tensor([[0.,.5,1.]])
    assert weight('M',None,None,None) is None
    for method in ('M','C','T','MT','CT'):
        values=weight(method,margin,confidence,temporal)
        assert values.shape==(1,3)
        assert bool(((values>=1)&(values<=4)).all())
    assert weight('M',margin,confidence,temporal)[0,0]==4
    assert weight('T',margin,confidence,temporal)[0,0]==1
    assert weight('T',margin,confidence,temporal)[0,2]==4


def test_tile_reduction_uses_same_row_product():
    score=torch.tensor([.1,.9])
    sensitivity=torch.tensor([4.,1.])
    actual=(score*sensitivity).max()
    assert actual==.9
    assert actual!=score.max()*sensitivity.max()
    assert bool(((score*2)>=score).all())


def test_shuffle_preserves_each_physical_tile_distribution():
    values=torch.arange(256,dtype=torch.float32)[None]
    generator=torch.Generator().manual_seed(41)
    result=shuffle_within_tiles(values,generator)
    assert torch.equal(result[:,:128].sort().values,values[:,:128])
    assert torch.equal(result[:,128:].sort().values,values[:,128:])
    assert not torch.equal(result,values)


def test_causal_history_and_reset():
    class Router:pass
    router=Router();router.query_sensitivity=None;router.thresholds={'local':{},'global':{}};router.policy_selector=None
    state=State('CT',router,m_ref=1.,diagnostics=False)
    canvas=torch.zeros((1,256),dtype=torch.long)
    state.begin(48,canvas)
    assert router.query_sensitivity is None
    logits=torch.zeros((1,256,3));logits[...,0]=1
    state.observe_logits(logits,torch.ones((1,256),dtype=torch.bool),48)
    state.begin(47,canvas)
    assert router.query_sensitivity is not None
    assert bool((router.query_sensitivity>=1).all())
    state.begin(48,canvas)
    assert router.query_sensitivity is None
    assert state.previous_top is None


def test_replay_acceptance_rule_matches_native_sampler():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSampler,EntropyBoundSamplerConfig
    sampler=EntropyBoundSampler(config=EntropyBoundSamplerConfig(entropy_bound=.1),vocab_size=3,canvas_length=4,max_denoising_steps=48)
    logits=torch.tensor([[[9.,0,0],[2.,1,0],[0.,0,0],[8.,0,0]]])
    current=torch.zeros((1,4),dtype=torch.long);proposal=torch.ones_like(current)
    sampler.accept_canvas(current,proposal,logits,torch.tensor(48))
    expected=sampler.accepted_token_mask
    observed,_=accepted_from_logits(logits,sampler.entropy_bound)
    assert torch.equal(expected,observed)


def test_task_stratified_prompt_bootstrap_uses_pairs():
    a=[dict(id='a',task='t1',score=1,steps=2),dict(id='b',task='t1',score=0,steps=4),
       dict(id='c',task='t2',score=1,steps=5),dict(id='d',task='t2',score=1,steps=6)]
    b=[dict(x,score=0,steps=x['steps']+1) for x in a]
    assert paired_interval(a,b,'steps')==[-1.,-1.]
    interval=paired_interval(a,b,'score')
    assert interval[0]>=0 and interval[1]<=1


def test_reference_row_votes_and_first_support():
    state=dict(logz=torch.zeros((1,1,2,2)),
        mu=torch.tensor([[[[[0.],[.2]],[[0.],[1.8]]]]]),
        active=torch.ones((1,1,2,2),dtype=torch.bool),eligible=torch.ones((1,1,2),dtype=torch.bool))
    reference=torch.ones((1,1))
    base=route_rows(state,reference,torch.ones((1,2)),0.)
    weighted=route_rows(state,reference,torch.tensor([[1.,4.]]),0.)
    assert not bool(base['physical'][0,0,0])  # no previously retained support
    assert bool(base['physical'][0,0,1])
    assert not bool(weighted['physical'][0,0,1])
    assert bool(base['row_skip'][0,0,0,1])
    assert not bool(weighted['row_skip'][0,0,1,1])
