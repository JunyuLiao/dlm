from types import SimpleNamespace

import torch

from experiments.numerical_qk_reuse.v31_value_refinement_runtime import Runtime


def state():
    lists=object()
    st=dict(canvas=4,nk=12,kept=torch.ones(1,1,1,1,dtype=torch.bool),lists=lists)
    adapter=SimpleNamespace(canvas_id=4,mage_state={29:st},mage_k=8192)
    return adapter,lists,torch.randn(1,4,1,8)


def test_correction_never_crosses_canvas_or_request():
    a,lists,out=state();r=Runtime('residual')
    r.state=dict(identity=(3,8,4),residual=torch.ones(4,1,8))
    result=r.output(a,29,None,None,1.,8,4,out,lists)
    assert result is out
    r.state=dict(identity=(4,8,4),residual=torch.ones(4,1,8))
    r.reset()
    assert r.state is None and r.output(a,29,None,None,1.,8,4,out,lists) is out


def test_earlier_layers_are_untouched_and_residual_counts_are_explicit():
    a,lists,out=state();r=Runtime('residual')
    r.state=dict(identity=(4,8,4),residual=torch.ones(4,1,8))
    assert r.output(a,23,None,None,1.,8,4,out,lists) is out
    torch.testing.assert_close(r.output(a,29,None,None,1.,8,4,out,lists),out+1)
    assert r.receipt()['value_refinement_work']['corrected_calls']==1
    assert 'plus cached correction' in r.receipt()['value_refinement_output_consumer']


def test_allkept_shortcut_does_not_run_correction_or_observation():
    a,lists,out=state();r=Runtime('moments')
    r.state=dict(identity=(4,8,4),allkept=True)
    assert r.output(a,29,None,None,1.,8,4,out,lists) is out
    assert not r.counts


def test_direction_guard_does_not_correct_a_row_already_at_dense_anchor():
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
    a,lists,out=state();r=Runtime('residual_direction_guard');out=torch.ones_like(out)
    identity=(4,8,4)
    cache=ResidualDirectionCache.build(torch.ones(4,1,8),torch.zeros(4,1,8),torch.eye(8),identity)
    r.state=dict(identity=identity,direction_cache=cache)
    assert torch.equal(r.output(a,29,None,None,1.,8,4,out,lists),out)
    assert not r.counts.get('extra_post_o_projections',0)
    assert r.counts['held_direction_dot_rows']==4


def test_native_anchor_cache_rebuild_and_carry_lifecycle(monkeypatch):
    from experiments.numerical_qk_reuse.v31_value_decision import moments
    g=torch.Generator().manual_seed(45)
    q=torch.randn(1,1,4,512,generator=g)
    b=dict(k=torch.randn(1,1,132,512,generator=g),v=torch.randn(1,1,132,512,generator=g))
    scale=512**-.5
    st=dict(canvas=4,nk=132,kept=torch.tensor([[[[True,False,True]]]]),lists=object())
    a=SimpleNamespace(canvas_id=4,mage_state={29:st},mage_k=64)
    layer=SimpleNamespace(self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=torch.randn(7,512,generator=g))))
    r=Runtime('residual_direction_guard');r.model=SimpleNamespace(model=SimpleNamespace(layers=[None]*29+[layer]))
    m=moments(q[0,0],b['k'][0,0],b['v'][0,0],128,scale)
    dense=m.dense()[None,:,None,:]
    assert r.output(a,29,q,b,scale,128,4,dense,None) is dense
    first=r.state['direction_cache'];old_list=st['lists']
    selected=m.output(st['kept'][0,0,0,:2])[None,:,None,:]
    torch.testing.assert_close(r.output(a,29,q,b,scale,128,4,selected,old_list),dense,rtol=1e-5,atol=1e-6)
    st['lists']=object();st['kept']=torch.tensor([[[[False,True,True]]]])
    r.output(a,29,q,b,scale,128,4,dense,old_list)
    assert r.state['direction_cache'] is not first
    assert r.counts['anchor_events']==2
    a.canvas_id=5
    assert r.output(a,29,q,b,scale,128,4,selected,st['lists']) is selected
    assert r.state is None


def test_cgate_statistics_are_previous_call_only_and_reset_at_canvas():
    r=Runtime('full_cgate_swap8')
    first=torch.tensor([[30.,0.],[0.,30.],[1.,1.]])
    r.sample_statistics(first,4,.1)
    assert not r.previous_stats[1].any() and not r.previous_stats[2].any()
    changed=first.clone();changed[0]=changed[0].flip(0)
    r.sample_statistics(changed,4,.1)
    assert torch.equal(r.previous_stats[1],torch.tensor([True,False,False]))
    r.sample_statistics(first,5,.1)
    assert not r.previous_stats[1].any() and r.stats_canvas==5
    r.reset();assert r.previous_stats is None and r.stats_canvas is None


def test_combined_variant_both_selects_and_corrects():
    r=Runtime('full_swap8_direction_guard')
    assert r.selects and r.corrects and r.direction_guard
    assert 'plus cached correction' in r.receipt()['value_refinement_output_consumer']
    assert not Runtime('full_cgate_swap8').corrects


def test_combined_cache_is_built_for_changed_support(monkeypatch):
    from experiments.numerical_qk_reuse import v27_fa4, v31_value_refinement_runtime as module
    from experiments.numerical_qk_reuse.v31_value_decision import moments
    monkeypatch.setattr(v27_fa4,'block_sparse_tensors',lambda mask:object())
    monkeypatch.setattr(module,'bounded_proposal',lambda m,mask,**kw:(~mask,{'swaps':1}))
    g=torch.Generator().manual_seed(103)
    q=torch.randn(1,1,4,512,generator=g)
    b=dict(k=torch.randn(1,1,132,512,generator=g),v=torch.randn(1,1,132,512,generator=g))
    st=dict(canvas=4,nk=132,kept=torch.tensor([[[[True,False,True]]]]),lists=object())
    a=SimpleNamespace(canvas_id=4,mage_state={29:st},mage_k=64)
    layer=SimpleNamespace(self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=torch.randn(7,512,generator=g))))
    r=Runtime('full_swap8_direction_guard');r.model=SimpleNamespace(model=SimpleNamespace(layers=[None]*29+[layer]))
    m=moments(q[0,0],b['k'][0,0],b['v'][0,0],128,512**-.5)
    dense=m.dense()[None,:,None,:]
    r.output(a,29,q,b,512**-.5,128,4,dense,None)
    assert torch.equal(st['kept'],torch.tensor([[[[False,True,True]]]]))
    selected=m.output(st['kept'][0,0,0,:2])[None,:,None,:]
    torch.testing.assert_close(r.output(a,29,q,b,512**-.5,128,4,selected,st['lists']),dense,atol=1e-6,rtol=1e-5)
    assert r.counts['swapped_blocks']==1 and r.counts['budget_errors']==0
