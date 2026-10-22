import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('nk,n,budget,sticky',[(1024,256,128,0.),(449,137,128,1.386),(4096,256,1728,1.386)])
def test_fused_selection_matches_inherited(nk,n,budget,sticky):
    from experiments.numerical_qk_reuse.refresh_v31_fast import select
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    g=torch.Generator(device='cuda').manual_seed(nk)
    pt=(nk-n)//64;qb=(n+127)//128;h=4
    z=torch.randn((h,qb,pt,128),device='cuda',generator=g)
    held=torch.rand((1,h,qb,(nk+63)//64),device='cuda',generator=g)>.5
    adapter=VllmMethodAdapter(['full_attention']*30,arm='mage',mage_select='fa4',mage_granularity='qblock_max',mage_k=budget)
    adapter.calls.update(mage_kept_prefix_tiles=0,mage_prefix_tiles=0)
    adapter._mage_held=held;adapter.mage_sticky=sticky
    head=z.permute(0,1,3,2).reshape(h,qb*128,pt)[:,:n]
    reference=adapter._mage_units(None,head,h,n,qb,(nk+63)//64,pt,min(budget//64,pt))
    keep,lists=select(z,n,nk,budget,held=held,sticky=sticky)
    assert torch.equal(keep,reference)
    for u in range(h*qb):
        ids=lists.full_block_idx.reshape(h*qb,-1)[u,:lists.full_block_cnt.flatten()[u]]
        assert torch.equal(ids,torch.where(keep.reshape(h*qb,-1)[u])[0].int())


@pytest.mark.parametrize('overlap',[False])
def test_softmax_observer_preserves_output_and_mass(tmp_path,overlap):
    from experiments.numerical_qk_reuse.refresh_v31_fast import Observer
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    g=torch.Generator(device='cuda').manual_seed(31)
    q=torch.randn((1,16,256,512),device='cuda',generator=g,dtype=torch.bfloat16)
    k=torch.randn((1,2,1025,512),device='cuda',generator=g,dtype=torch.bfloat16)
    v=torch.randn(k.shape,device='cuda',generator=g,dtype=torch.bfloat16)
    z=torch.empty((16,2,12,128),device='cuda')
    observer=Observer(tmp_path)
    logits=q[0].float()@k[0,:,:768].float().repeat_interleave(8,0).transpose(-1,-2)*512**-.5
    expected=logits.reshape(16,2,128,12,64).logsumexp(-1).permute(0,1,3,2)
    for overlap in (overlap,):
        reference=_flash_attn_fwd(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=False,
            softmax_scale=512**-.5,pack_gqa=False,num_splits=1,intra_wg_overlap=overlap)[0]
        for _ in range(5):
            out=observer(q,k,v,512**-.5,z,overlap=overlap)
            assert torch.equal(out,reference)
            torch.testing.assert_close(z,expected,atol=2.e-5,rtol=2.e-5)


def test_observer_cache_is_distinct_from_inherited_and_shape_changes(tmp_path):
    from experiments.numerical_qk_reuse.refresh_v31_fast import Observer
    from experiments.numerical_qk_reuse import v31_fa4_observe
    from scripts.v31_humaneval_observe_no_overlap import wrap_fwd
    v31_fa4_observe._flash_attn_fwd=wrap_fwd(v31_fa4_observe._flash_attn_fwd,dict(observing_calls=0))
    observer=Observer(tmp_path)
    for nk,n in [(449,137),(4096,256),(8192,256)]:
        g=torch.Generator(device='cuda').manual_seed(nk)
        q=torch.randn((1,16,n,512),device='cuda',generator=g,dtype=torch.bfloat16)
        k=torch.randn((1,2,nk,512),device='cuda',generator=g,dtype=torch.bfloat16)
        v=torch.randn(k.shape,device='cuda',generator=g,dtype=torch.bfloat16)
        pt=(nk-n)//64;qb=(n+127)//128
        old=torch.empty((16,qb,pt,128),device='cuda');new=torch.empty_like(old)
        old_output=v31_fa4_observe.observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),512**-.5,old)
        result=observer(q,k,v,512**-.5,new,overlap=False,num_splits=0,dynamic_causal=True)
        assert torch.equal(result,old_output)
        rows=(torch.arange(qb*128,device='cuda').reshape(qb,128)<n)[None,:,None,:].expand_as(old)
        torch.testing.assert_close(new[rows],old[rows],atol=2.e-5,rtol=2.e-5)


def test_prefix_observation_recovers_from_dominant_canvas_underflow(tmp_path):
    from experiments.numerical_qk_reuse.refresh_v31_fast import Observer
    from experiments.numerical_qk_reuse import v31_fa4_observe
    from scripts.v31_humaneval_observe_no_overlap import wrap_fwd
    v31_fa4_observe._flash_attn_fwd=wrap_fwd(v31_fa4_observe._flash_attn_fwd,dict(observing_calls=0))
    q=torch.zeros((1,16,256,512),device='cuda',dtype=torch.bfloat16);q[...,0]=1
    k=torch.zeros((1,2,512,512),device='cuda',dtype=torch.bfloat16);k[:,:,256:,0]=4096
    k[:,:,:64,0]=5
    v=torch.randn_like(k)
    z=torch.empty((16,2,4,128),device='cuda')
    v31_fa4_observe.observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),512**-.5,z)
    out,keep,lists=Observer(tmp_path).refresh(q,k,v,512**-.5,256,128)
    assert bool(keep[...,0].all())
    assert bool(torch.isfinite(out).all())
    assert bool((keep[...,:4].sum(-1)==2).all())


@pytest.mark.parametrize('n,pt,budget,sticky',[(256,640,128,0.),(137,2204,8192,1.386)])
def test_long_selector_matches_bounded_reference(n,pt,budget,sticky):
    from experiments.numerical_qk_reuse.refresh_v31_fast import select
    h=2;qb=(n+127)//128;nk=pt*64+n;jt=(nk+63)//64
    g=torch.Generator(device='cuda').manual_seed(pt)
    z=torch.randn((h,qb,pt,128),device='cuda',generator=g)
    # Invalid query rows must not dominate qblock-max selection.
    if n%128:z[:,-1,:,n%128:]=100.
    held=torch.rand((1,h,qb,jt),device='cuda',generator=g)>.5
    lm=z.permute(0,1,3,2)
    scores=lm-lm.logsumexp(-1,keepdim=True)
    valid=torch.arange(qb*128,device='cuda').reshape(qb,128)<n
    scores=scores.masked_fill(~valid[None,:,:,None],-float('inf')).amax(-2)
    scores+=sticky*held[0,...,:pt]
    expected=torch.zeros_like(held)
    expected[0,...,:pt].scatter_(-1,scores.topk(min(pt,budget//64),-1).indices,True)
    expected[...,pt:]=True
    keep,lists=select(z,n,nk,budget,held=held,sticky=sticky)
    assert torch.equal(keep,expected)
    assert torch.equal(lists.full_block_cnt,keep.sum(-1).int())


def test_long_selector_ties_are_stable():
    from experiments.numerical_qk_reuse.refresh_v31_fast import select
    z=torch.zeros((1,2,2204,128),device='cuda')
    keep,lists=select(z,256,2208*64,8192)
    expected=torch.zeros_like(keep);expected[...,:128]=True;expected[...,2204:]=True
    assert torch.equal(keep,expected)
    assert bool((lists.full_block_cnt==132).all())
