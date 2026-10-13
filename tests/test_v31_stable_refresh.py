import math
import pytest
import torch
from experiments.numerical_qk_reuse.v31_stable_refresh import (
    StableConfig, StableState, rows_to_tiles, mass_scores, stable_topk,
    corrected_bounded, assert_budget, support_drift)
from experiments.numerical_qk_reuse.v31_stable_observe import reference_observe, observe


DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def test_distinguishable_dense_layout_and_padding():
    x = torch.arange(2*259*7).reshape(2,259,7).float()
    z = rows_to_tiles(x)
    for h,r,j in [(0,0,0),(0,127,3),(0,128,4),(1,258,6)]:
        assert z[h*3+r//128,j,r%128]==x[h,r,j]
    assert torch.isneginf(z[2,:,3:]).all()
    assert torch.isneginf(z[5,:,3:]).all()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('scores,current,k,m,cap',[
    ([5.,4.,3.],[1,1,0],2,2,8), # empty outside
    ([5.,4.,3.],[1,1,1],3,3,8), # all-kept
    ([1.,1.,1.],[1,0,1],2,3,8), # no beneficial ties
    ([2.,1.,3.],[1,1,0],2,3,8), # one candidate, then exhaustion
])
def test_corrected_bounded_noop_exhaustion(device,scores,current,k,m,cap):
    score=torch.tensor([scores],device=device)
    cur=torch.tensor([current],dtype=torch.bool,device=device)
    got=corrected_bounded(score,cur,k,m,cap)
    assert got.sum()==k
    ref=corrected_bounded(score.cpu(),cur.cpu(),k,m,cap)
    assert torch.equal(ref,got.cpu())
    if scores[0]>=scores[-1]: assert torch.equal(got,cur)


@pytest.mark.parametrize('device',DEVICES)
def test_budget_failure_is_exposed_and_never_repaired(device):
    # GPU invalid-map async assertions are tested separately in an isolated process.
    if device=='cuda': return
    with pytest.raises(AssertionError):
        corrected_bounded(torch.zeros(1,4),torch.tensor([[1,0,0,0]],dtype=torch.bool),2,3)


def test_complete_support_snapshot_ranking_and_immutable_cache():
    gen=torch.Generator().manual_seed(4)
    raw=torch.randn(3,9,128,generator=gen)
    chosen=stable_topk(mass_scores(raw),4)
    state=StableState(raw,chosen,StableConfig(mode='raw',hysteresis=0))
    anchor=state.anchor.clone()
    got,_=state.observe(raw.clone(),torch.zeros(3,128))
    assert torch.equal(got,chosen)
    assert torch.equal(state.anchor,anchor)
    fresh=raw.clone(); fresh[~chosen]=10000.
    state.observe(fresh,torch.zeros(3,128))
    assert torch.equal(state.selected,chosen)
    assert torch.equal(state.anchor,anchor)


@pytest.mark.parametrize('device',DEVICES)
def test_hysteresis_persistence_budget_guard_and_protection(device):
    raw=torch.tensor([[[0.],[0.],[-0.1],[-0.2],[-0.3]]],device=device).expand(1,5,128).clone()
    cur=torch.tensor([[1,1,0,0,0]],dtype=torch.bool,device=device)
    mandatory=torch.tensor([[1,0,0,0,0]],dtype=torch.bool,device=device)
    cfg=StableConfig(hysteresis=0.1,persistence=2,event_cap=2,cumulative_cap=1,removal_tolerance=1.)
    state=StableState(raw,cur,cfg,mandatory)
    fresh=raw.clone(); fresh[:,1]-=1.
    first,_=state.observe(fresh,torch.zeros(1,128,device=device))
    assert torch.equal(first,cur)
    second,_=state.observe(fresh,torch.zeros(1,128,device=device))
    assert second[0,0] and second[0,2] and not second[0,1]
    assert state.cumulative.item()==1
    for _ in range(8): state.observe(fresh,torch.zeros(1,128,device=device))
    assert state.cumulative.item()==1
    assert_budget(state.selected,2,mandatory)


@pytest.mark.parametrize('device',DEVICES)
def test_row_removal_guard_includes_mandatory_tail(device):
    raw=torch.tensor([[[0.],[-0.1],[0.2]]],device=device).expand(1,3,128).clone()
    cur=torch.tensor([[1,1,0]],dtype=torch.bool,device=device)
    cfg=StableConfig(hysteresis=0,persistence=1,removal_tolerance=.01)
    blocked=StableState(raw,cur,cfg)
    out,info=blocked.observe(raw,torch.full((1,128),-math.inf,device=device))
    assert torch.equal(out,cur) and info['rejected_guard'].any()
    tail=StableState(raw,cur,cfg)
    out,_=tail.observe(raw,torch.full((1,128),10.,device=device))
    assert (out!=cur).any() # large mandatory tail makes removal share small


@pytest.mark.parametrize('device',DEVICES)
def test_allkept_and_disabled_updates_preserve_map_and_rng(device):
    raw=torch.arange(5.,device=device)[None,:,None].expand(1,5,128)
    cur=torch.ones(1,5,dtype=torch.bool,device=device)
    before=torch.get_rng_state().clone()
    state=StableState(raw,cur)
    state.observe(raw,torch.zeros(1,128,device=device))
    assert torch.equal(state.selected,cur)
    assert torch.equal(torch.get_rng_state(),before)
    hold=StableState(raw,torch.tensor([[1,1,0,0,0]],dtype=torch.bool,device=device),StableConfig(updates_enabled=False))
    out,_=hold.observe(raw,torch.zeros(1,128,device=device))
    assert torch.equal(out,hold.initial)


def test_nonfinite_fails_without_implicit_dense_rescue():
    raw=torch.zeros(1,3,128); cur=torch.tensor([[1,1,0]],dtype=torch.bool)
    state=StableState(raw,cur)
    bad=raw.clone(); bad[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError): state.observe(bad,torch.zeros(1,128))
    assert torch.equal(state.selected,cur)


@pytest.mark.parametrize('device',DEVICES)
def test_bounded_outside_probe_semantics_and_counter_budget(device):
    raw=torch.tensor([[[1.],[0.],[2.],[1.5]]],device=device).expand(1,4,128).clone()
    selected=torch.tensor([[1,1,0,0]],dtype=torch.bool,device=device)
    cfg=StableConfig(hysteresis=0,persistence=1,removal_tolerance=1.,probes=1)
    state=StableState(raw,selected,cfg)
    fresh=raw.clone();fresh[:,2]=-10.
    probe=torch.tensor([[0,0,1,0]],dtype=torch.bool,device=device)
    actual,_=state.observe(fresh,torch.zeros(1,128,device=device),probe)
    assert actual[0,3] and not actual[0,2]
    assert torch.equal(state.anchor,raw)
    assert state.cumulative.item()==1


def test_drift_common_shift_separated_from_normalized_change():
    raw=torch.arange(5.)[None,:,None].expand(1,5,128); mask=torch.ones(1,5,dtype=torch.bool)
    drift=support_drift(raw+7,raw,mask)
    torch.testing.assert_close(drift,torch.tensor([7.,7.,0.,0.]),atol=1e-6,rtol=1e-5)


def test_cpu_cuda_policy_against_independent_python_oracle():
    import numpy as np
    gen=torch.Generator().manual_seed(97)
    raw=torch.randn(3,13,128,generator=gen)
    current=stable_topk(mass_scores(raw),5)
    for device in DEVICES:
        state=StableState(raw.to(device),current.to(device),StableConfig(hysteresis=.1,persistence=2,removal_tolerance=.05))
        held=current.numpy().copy(); streak=np.zeros(held.shape,dtype=int); cumulative=np.zeros(3,dtype=int)
        for step in range(6):
            fresh=raw+torch.randn(raw.shape,generator=gen)*.4
            # Independent scalar policy with explicit per-unit and row sums.
            hybrid=np.where(held[:,:,None],fresh.numpy(),raw.numpy())
            mx=hybrid.max(axis=1,keepdims=True)
            logden=mx+np.log(np.exp(hybrid-mx).sum(axis=1,keepdims=True))
            score=(hybrid-logden).max(axis=2)
            nxt=held.copy()
            for u in range(3):
                order=sorted(range(13),key=lambda j:(-(float(score[u,j])+.1*held[u,j]),j))
                proposed=set(order[:5]); favorable=np.array([j in proposed and not held[u,j] for j in range(13)])
                streak[u]=np.where(favorable,streak[u]+1,0)
                candidates=sorted((j for j in proposed if favorable[j] and streak[u,j]>=2),key=lambda j:(-score[u,j],j))
                removed=sorted((j for j in range(13) if held[u,j] and j not in proposed),key=lambda j:(score[u,j],j))
                all_log=np.concatenate([fresh[u,held[u]].numpy(),np.zeros((1,128))],axis=0)
                rowmax=all_log.max(0); den=np.exp(all_log-rowmax).sum(0)*np.exp(rowmax)
                mass=np.zeros(128); used=0
                for ci,wi in zip(candidates,removed):
                    r=np.exp(fresh[u,wi].numpy())/den
                    if used<2 and cumulative[u]<8 and score[u,ci]-score[u,wi]>.1 and (mass+r).max()<=.05:
                        nxt[u,wi]=False; nxt[u,ci]=True; mass+=r; used+=1; cumulative[u]+=1
            actual,_=state.observe(fresh.to(device),torch.zeros(3,128,device=device))
            np.testing.assert_array_equal(actual.cpu().numpy(),nxt)
            held=nxt


@pytest.mark.parametrize('device',DEVICES)
def test_observe_gqa_per_head_padded_rows_native_statistics(device):
    gen=torch.Generator().manual_seed(3)
    q=torch.randn(1,4,129,64,generator=gen).to(device,dtype=torch.bfloat16)
    k=torch.randn(1,2,209,64,generator=gen).to(device,dtype=torch.bfloat16)
    ids=torch.tensor([[0,1],[1,2],[0,2],[2,3],[1,3],[0,1],[0,3],[1,2]],device=device)
    expected=reference_observe(q,k,.125,ids,129)
    rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
    actual=observe(q,k,.125,ids,129)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=2e-5)
    after=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
    assert torch.equal(rng,after)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 needed')
def test_dense_fa4_retained_observation_same_snapshot_and_output_bitwise():
    from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    gen=torch.Generator(device='cuda').manual_seed(9)
    q=torch.randn(1,16,256,512,device='cuda',generator=gen,dtype=torch.bfloat16)
    k=torch.randn(1,2,6464,512,device='cuda',generator=gen,dtype=torch.bfloat16)
    v=torch.randn(k.shape,device='cuda',generator=gen,dtype=torch.bfloat16)
    z=torch.full((16,2,97,128),-math.inf,device='cuda')
    scale=1/math.sqrt(512)
    out=observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale,z)
    plain=_flash_attn_fwd(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),softmax_scale=scale,
        causal=True,dynamic_causal=torch.tensor([False],device='cuda'),num_splits=0,pack_gqa=False)[0]
    assert torch.equal(out,plain)
    ids=torch.arange(10,device='cuda')[None].expand(32,-1).contiguous()
    fresh=observe(q,k,scale,ids,256)
    expected=z.reshape(32,97,128).gather(1,ids[:,:,None].expand(-1,-1,128))
    torch.testing.assert_close(fresh,expected,atol=3e-4,rtol=2e-5)
    from experiments.numerical_qk_reuse import v27_fa4
    selected=stable_topk(mass_scores(z.reshape(32,97,128)),16)
    full=torch.ones(1,16,2,101,dtype=torch.bool,device='cuda')
    full[0,:,:,:97]=selected.reshape(16,2,97)
    lists=v27_fa4.block_sparse_tensors(full)
    held=v27_fa4.sparse_lists(q,k,v,lists,scale)
    rng=torch.cuda.get_rng_state().clone()
    state=StableState(z.reshape(32,97,128),selected,StableConfig(updates_enabled=False))
    all_ids=selected.to(torch.int8).argsort(dim=-1,descending=True,stable=True)[:,:16].contiguous()
    retained=observe(q,k,scale,all_ids,256)
    fresh_full=torch.full_like(state.anchor,-math.inf)
    fresh_full.scatter_(1,all_ids[:,:,None].expand(-1,-1,128),retained)
    state.observe(fresh_full,torch.zeros(32,128,device='cuda'))
    again=v27_fa4.sparse_lists(q,k,v,lists,scale)
    assert torch.equal(held,again)
    assert torch.equal(torch.cuda.get_rng_state(),rng)
