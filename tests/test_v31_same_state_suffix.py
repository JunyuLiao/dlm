import torch
import pytest

from experiments.numerical_qk_reuse.v31_same_state_suffix import compare, distribution


def test_acceptance_matches_official_entropy_bound_rule():
    x=torch.tensor([[12.,0.,0.],[1.,2.,3.],[0.,0.,0.],[0.,0.,12.]])
    logp=x.log_softmax(-1); entropy=-(logp.exp()*logp).sum(-1)
    ordered,index=entropy.sort(-1); expected=torch.zeros_like(entropy,dtype=torch.bool)
    expected[index]=(ordered.cumsum(-1)-ordered.cummax(-1).values)<=.1
    assert torch.equal(distribution(x)['accepted'],expected)


def test_identical_logits_have_no_diagnostic_decision_changes():
    x=torch.randn(8,23); noise=torch.randn_like(x)
    out=compare(x,x,noise,x.argmax(-1))
    for key in ('top1_disagreement','acceptance_disagreement','common_noise_disagreement','kl_mean','kl_max','max_logit_error'):
        assert out[key]==0
    assert not out['stop_flip']


def test_stop_uses_both_stability_and_confidence():
    confident=torch.tensor([[30.,0.],[30.,0.]])
    altered=confident.flip(-1)
    out=compare(altered,confident,torch.zeros_like(confident),torch.zeros(2,dtype=torch.long))
    assert out['reference_stop'] and not out['candidate_stop'] and out['stop_flip']
    assert out['top1_disagreement']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 FA4 required')
def test_selected_fa4_lse_and_cached_merge_match_full_dimension_oracle():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v31_value_decision import moments, OmittedCache
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import SelectedOnlyCorrection, reconstruct_cuda
    g=torch.Generator().manual_seed(381)
    q=torch.randn(1,2,128,512,generator=g).to(device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,1,384,512,generator=g).to(device='cuda',dtype=torch.bfloat16)
    v=torch.randn(1,1,384,512,generator=g).to(device='cuda',dtype=torch.bfloat16)
    scale=512**-.5
    kept=torch.tensor([[[[1,0,1,0,1,1]],[[0,1,0,1,1,1]]]],device='cuda',dtype=torch.bool)
    fwd=v27_fa4.load();lists=v27_fa4.block_sparse_tensors(kept)
    selected,lse=fwd(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),softmax_scale=scale,
        causal=False,block_sparse_tensors=lists,return_lse=True)[:2]
    dl=torch.empty(128,2,device='cuda');sl=torch.empty_like(dl)
    dense=torch.empty(128,2,512,device='cuda');old=torch.empty_like(dense);expected=[]
    changed=q*.9
    for head in range(2):
        m=moments(q[0,head],k[0,0],v[0,0],256,scale);mask=kept[0,head,0,:4]
        dense[:,head]=m.dense();old[:,head]=m.output(mask)
        dl[:,head]=m.log_scale;sl[:,head]=m.log_scale+m.selected(mask)[1].log()
        now=moments(changed[0,head],k[0,0],v[0,0],256,scale)
        expected.append(OmittedCache.build(m,mask).reconstruct(now,mask))
    torch.testing.assert_close(lse[0].T,sl,rtol=1e-5,atol=2e-5)
    from experiments.numerical_qk_reuse.v31_stable_observe import observe, tile_ids
    observed=observe(q,k,scale,tile_ids(kept[0].reshape(2,-1)),128)
    torch.testing.assert_close(observed.logsumexp(1).reshape(2,128).T,sl,rtol=1e-5,atol=2e-5)
    torch.testing.assert_close(selected[0].float(),old,rtol=.02,atol=.002)
    cache=SelectedOnlyCorrection.from_anchor(dense,dl,old,sl,'fixed')
    current,new_lse=fwd(changed.transpose(1,2),k.transpose(1,2),v.transpose(1,2),softmax_scale=scale,
        causal=False,block_sparse_tensors=lists,return_lse=True)[:2]
    merged=reconstruct_cuda(current[0],new_lse[0].T.contiguous(),cache,'fixed')
    torch.testing.assert_close(merged.float(),torch.stack(expected,1),rtol=.02,atol=.002)
