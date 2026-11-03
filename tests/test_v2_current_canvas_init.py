import torch
from experiments.numerical_qk_reuse.v31_value_selectors import Stats
from experiments.numerical_qk_reuse.v2_current_canvas_init import current_canvas_reference


def test_mandatory_suffix_is_normalized_before_optional_scan():
    z=torch.zeros((1,4,1));mu=torch.tensor([4.,0.,0.,2.]).reshape(1,4,1,1)
    s=Stats(z,mu,torch.ones(1,1),torch.zeros_like(z,dtype=torch.bool),2,1,1)
    mandatory=torch.tensor([[False,False,True,True]])
    keep,state,mass,_=current_canvas_reference(s,.9,mandatory,2)
    assert keep.tolist()==[[True,False,True,True]]
    assert torch.allclose(state,torch.tensor([[[2.]]]))
    assert torch.allclose(mass,torch.tensor([[4.]]).log())


def test_same_values_do_not_force_chronological_first():
    z=torch.zeros((1,6,2));mu=torch.ones((1,6,2,3))
    s=Stats(z,mu,torch.ones(1,2),torch.zeros_like(z,dtype=torch.bool),2,1,1)
    mandatory=torch.tensor([[False,False,True,True,True,True]])
    keep,state,mass,_=current_canvas_reference(s,.1,mandatory,2)
    assert torch.equal(keep,mandatory)
    assert torch.allclose(state,torch.ones_like(state))
    assert torch.allclose(mass,torch.full_like(mass,6.).log())
