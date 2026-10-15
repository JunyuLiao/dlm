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
