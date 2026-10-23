import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from production_observer import ProductionObserver, distribution_stats


def native_sampler():
    source = Path('/home/exouser/dyh/v31_env_20261005/lib/python3.12/site-packages/vllm/model_executor/models/diffusion_gemma.py')
    module = ast.parse(source.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_compiled_sample_step')
    function.decorator_list = []
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_compiled_sample_step']


def state():
    cl, vocabulary, hidden = 4, 8, 3
    args = [torch.tensor([[15., 0., -1., -2., -3., -4., -5., -6.]]).repeat(cl,1),
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([cl]),
        torch.zeros((1,cl),dtype=torch.long), torch.zeros((1,cl),dtype=torch.long),
        torch.zeros(1,dtype=torch.int32), torch.zeros(1,dtype=torch.bool),
        torch.zeros(1,dtype=torch.bool), torch.zeros((1,cl,hidden)),
        torch.arange(vocabulary*hidden,dtype=torch.float32).reshape(vocabulary,hidden),
        torch.tensor(1.), torch.zeros((1,2,cl),dtype=torch.long),
        torch.zeros(1,dtype=torch.long), torch.zeros((1,cl),dtype=torch.long),
        torch.zeros(1,dtype=torch.long), torch.zeros((1,cl),dtype=torch.long)]
    kwargs = dict(max_denoising_steps=48., t_min=.4,t_max=.8,confidence_threshold=.005,
        vocab_size=vocabulary,CL=cl,ST=2,entropy_bound=.1,sc_vocab_start=0,
        sc_vocab_end=vocabulary,tp_size=1,tp_group_name='unused')
    return args,kwargs


def test_rich_observer_preserves_actual_native_sampler_states_and_rng():
    native=native_sampler()
    traces=[]
    for mode in ('minimal','rich','minimal'):
        args,kwargs=state(); observer=ProductionObserver();observer.begin(mode)
        torch.manual_seed(51994)
        for _ in range(4): observer.sample(native,*args,**kwargs)
        traces.append(observer.fingerprints)
        if mode=='rich':
            summary=observer.summary()
            assert summary['native_predicate_mismatches']==0
            assert summary['denoising_calls']==3
            assert summary['commit_calls']==1
            assert summary['observer_rng_checks']==8
            assert observer.records[0]['argmax_flip_count'] is None
            assert observer.records[-1]['canvas_id']==1
            assert observer.records[-1]['stable_dwell_p50']==0
    assert traces[0]==traces[1]==traces[2]


def test_statistics_reconstruct_native_entropy_and_acceptance():
    logits=torch.tensor([[10.,0.,-1.],[0.,0.,0.],[5.,2.,-1.]])
    before=logits.clone();rng=torch.get_rng_state()
    entropy,margin,probability,accepted=distribution_stats(logits,.1)
    logp=logits.log_softmax(-1)
    torch.testing.assert_close(entropy,-(logp.exp()*logp).sum(-1))
    assert accepted.sum()==2
    assert torch.equal(before,logits) and torch.equal(rng,torch.get_rng_state())


def test_census_has_a_required_smoke_gate_and_no_sampler_override():
    source=(CODE/'worker.py').read_text()
    assert 'smoke/final_complete.json' in source
    assert "full_cache_fork_restore_validated=False" in source
    assert 'max_denoising_steps=' not in source
    assert "full30_complete=False" in source
