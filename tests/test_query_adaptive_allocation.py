"""CPU checks for the frozen fresh-prompt temporal-allocation comparison."""
from collections import Counter
import json
from pathlib import Path

import torch

from experiments.value_direction_hopper.query_adaptive import State, shuffle_within_tiles
from experiments.value_direction_hopper.query_adaptive_allocation import (
    TASKS, _selected_pool, conditions, specs,
)


def test_new_task_balanced_splits_do_not_reuse_previous_pool_ids_or_prompts():
    selected=_selected_pool()
    assert {key:len(rows) for key,rows in selected.items()}=={
        'calibration':26,'final':130,'development':13}
    old=json.loads(Path('results/query_adaptive_v3/configs/final_manifest.json').read_text())
    old+=json.loads(Path('results/query_adaptive_v3/configs/calibration_manifest.json').read_text())
    old+=json.loads(Path('results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16/development_manifest.json').read_text())
    old_ids={r['source_id'] for r in old};old_prompts={r['prompt'] for r in old}
    all_ids=set();all_prompts=set()
    for split,per_task in (('calibration',2),('final',10),('development',1)):
        rows=selected[split]
        assert Counter(r['task'] for r in rows)==dict.fromkeys(TASKS,per_task)
        for r in rows:
            identifier=f'ruler_4096_{r["task"]}_p{int(r["_pool_index"]):04d}'
            assert identifier not in old_ids|all_ids
            assert r['input'] not in old_prompts|all_prompts
            all_ids.add(identifier);all_prompts.add(r['input'])


def test_only_allocation_changes_between_temporal_arms():
    mapping=specs()
    assert [x for x in conditions() if x[1] is not None]==[
        (name,target) for target in (50,70)
        for name in ('unweighted','T','T_shuffle','T_uniform')]
    assert {mapping[name]['method'] for name in ('T','T_shuffle','T_uniform')}=={'T'}
    assert [mapping[name]['allocation'] for name in ('T','T_shuffle','T_uniform')]==[
        'normal','shuffle','uniform']
    assert {mapping[name]['bootstrap'] for name in ('T','T_shuffle','T_uniform')}=={False}


def test_shuffled_weights_preserve_each_physical_query_tile_multiset():
    values=torch.arange(1,258,dtype=torch.float32).reshape(1,257)
    generator=torch.Generator().manual_seed(42)
    shuffled=shuffle_within_tiles(values,generator)
    for start in (0,128,256):
        assert torch.equal(values[:,start:start+128].sort(-1).values,
            shuffled[:,start:start+128].sort(-1).values)
    assert not torch.equal(values[:,:128],shuffled[:,:128])


def test_temporal_history_cannot_affect_first_two_iterations():
    state=State('T',None,m_ref=1.,seed=42,diagnostics=False)
    canvas=torch.zeros((1,256),dtype=torch.int64)
    state.begin(48,canvas)
    assert state.used_weights is None
    logits=torch.zeros((1,256,3));logits[...,0]=2
    state.observe_logits(logits,torch.ones_like(canvas,dtype=torch.bool),48)
    state.begin(47,canvas)
    assert torch.equal(state.used_weights,torch.ones_like(state.used_weights))
    logits[...,0]=0;logits[...,1]=3
    state.observe_logits(logits,torch.ones_like(canvas,dtype=torch.bool),47)
    state.begin(46,canvas)
    assert torch.all(state.used_weights>1)
    state.begin(48,canvas)
    assert state.used_weights is None
