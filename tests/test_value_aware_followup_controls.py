from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
import torch

from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DRuntime, dense_eager_attention_forward
from experiments.diffusion_gemma_value_aware.operators import (
    Config, value_summaries, block_state, streaming_mask, screen_risks)
from experiments.diffusion_gemma_value_aware_followup.controls import (
    CONFIG, unity_metadata, UnityCache, ControlAttention)
from experiments.diffusion_gemma_value_aware_followup.supplement import SupplementalScreen


def test_unity_metadata_is_exact_and_does_not_mutate_real_value_pools():
    values = torch.randn(1, 2, 150, 4)
    valid = torch.ones(1, 2, 150, dtype=torch.bool)
    meta = value_summaries(values, valid); original = meta['rms'].clone()
    unit = unity_metadata(meta)
    assert torch.equal(unit['rms']/unit['ref'][..., None], torch.ones_like(unit['rms']))
    assert torch.equal(meta['rms'], original)
    assert unit['vectors'] is meta['vectors']


@pytest.mark.parametrize('threshold', [-2., 0., 1., 5.])
def test_no_value_masks_ignore_v_but_preserve_retained_state(threshold):
    scores = torch.randn(1, 2, 7, 192)
    valid = torch.ones_like(scores, dtype=torch.bool); valid[..., 0, :64] = False
    masks = []; risks = []
    for values in (torch.zeros(1,2,192,4), torch.randn(1,2,192,4)*100):
        meta = unity_metadata(value_summaries(values, valid.any(-2)))
        state = block_state(scores, valid, values)
        config = Config(**CONFIG, log_threshold=threshold)
        masks.append(streaming_mask(state, meta, config)); risks.append(screen_risks(state,meta,config)[0])
        if threshold <= 0:
            assert torch.equal(masks[-1], streaming_mask(state, meta,
                Config(method='blasst', log_threshold=threshold)))
        assert (state['active'] & ~masks[-1][...,None,:]).any(-1).equal(valid.any(-1))
    assert torch.equal(masks[0], masks[1]) and torch.equal(risks[0], risks[1])


def test_control_retained_max_is_distinct_from_aggressive_blasst_seen_max():
    scores = torch.zeros(1,1,2,192); scores[...,64:128]=.5; scores[...,128:]=1.2
    valid = torch.ones_like(scores, dtype=torch.bool); values=torch.randn(1,1,192,4)
    meta=unity_metadata(value_summaries(values,valid.any(-2))); state=block_state(scores,valid,values)
    assert streaming_mask(state,meta,Config(**CONFIG,log_threshold=1.)).tolist()==[[[False,True,False]]]
    assert streaming_mask(state,meta,Config(method='blasst',log_threshold=1.)).tolist()==[[[False,True,True]]]


def test_value_rule_can_change_masks_while_matched_control_retains_threshold_ties():
    scores=torch.zeros(1,1,2,128);valid=torch.ones_like(scores,dtype=torch.bool)
    choices=[]
    for left,right in ((100.,.01),(.01,100.)):
        values=torch.ones(1,1,128,1);values[...,:64,:]*=left;values[...,64:,:]*=right
        meta=value_summaries(values,valid.any(-2));state=block_state(scores,valid,values)
        choices.append(streaming_mask(state,meta,Config(method='value',log_threshold=0.)).tolist())
        control=streaming_mask(state,unity_metadata(meta),Config(**CONFIG,log_threshold=0.))
        assert control.tolist()==[[[False,False]]]
    assert choices==[[[[False,True]]],[[[False,False]]]]


def test_unity_cache_keeps_real_vectors_and_refreshes_changed_canvas():
    cache=UnityCache(); values=torch.randn(1,1,150,4); valid=torch.ones(1,1,150,dtype=torch.bool)
    a=cache.get(0,values,valid,128); values[...,140,:]*=5
    b=cache.get(0,values,valid,128)
    assert cache.reused_blocks==2
    assert not torch.equal(a['vectors'],b['vectors'])
    assert torch.equal(b['rms']/b['ref'][...,None],torch.ones_like(b['rms']))


def test_supplement_covers_base_pools_and_control_without_changing_dense():
    torch.manual_seed(0)
    runtime=Blasst2DRuntime(Blasst2DConfig()); runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.randn(1,2,5,8); k=torch.randn(1,1,150,8); v=torch.randn_like(k)
    valid=torch.ones(1,1,5,150,dtype=torch.bool)
    dense=dense_eager_attention_forward(module,q,k,v,valid,is_causal=False)[0]
    router=SupplementalScreen(screen=True)
    assert torch.equal(router(module,q,k,v,valid,is_causal=False)[0],dense)
    arrays=router.arrays()
    for name in ('blasst','mass','aligned','centered','compensate','zero_pv','no_value_control'):
        assert f'{name}__global' in arrays
    for name in ('value','mass_value','risk'):
        for pool in ('max','mean','rms','p95','vector_mean'):
            assert f'{name}_{pool}__global' in arrays
    control=ControlAttention(dict(CONFIG,log_threshold=-math.inf))
    assert torch.equal(control(module,q,k,v,valid,is_causal=False)[0],dense)


def test_control_rejects_implicit_or_wrong_pool_configuration():
    with pytest.raises(ValueError): ControlAttention(dict(method='value'))
    with pytest.raises(ValueError): ControlAttention(dict(CONFIG,pooling='max'))
