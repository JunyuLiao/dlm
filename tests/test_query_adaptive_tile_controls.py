"""CPU checks for the added 128-row tile sensitivity broadcasts."""
import torch

from experiments.value_direction_hopper.query_adaptive_tile_controls import TileState,tile_broadcast


def test_tile_mean_and_max_broadcast_each_128_row_group():
    values=torch.cat((torch.arange(128,dtype=torch.float32),
        200+torch.arange(128,dtype=torch.float32),torch.tensor([400.]))).reshape(1,-1)
    mean=tile_broadcast(values,'mean');maximum=tile_broadcast(values,'max')
    assert mean.shape==maximum.shape==values.shape
    assert torch.all(mean[:,:128]==63.5)
    assert torch.all(maximum[:,:128]==127.)
    assert torch.all(mean[:,128:256]==263.5)
    assert torch.all(maximum[:,128:256]==327.)
    assert mean[0,256]==maximum[0,256]==400.


def test_tile_control_uses_only_previous_deterministic_argmax_history():
    class Router:
        query_sensitivity=None
    canvas=torch.zeros((1,256),dtype=torch.int64)
    for mode in ('mean','max'):
        router=Router();state=TileState('T',router,m_ref=1.,beta=3.,gamma=.5,
            seed=42,diagnostics=False,tile_mode=mode)
        state.begin(48,canvas)
        assert state.used_weights is None
        a=torch.zeros((1,256,3));a[...,0]=2
        state.observe_logits(a,torch.ones_like(canvas,dtype=torch.bool),48)
        state.begin(47,canvas)
        assert torch.all(state.used_weights==1)
        b=a.clone();b[:,:64,0]=0;b[:,:64,1]=3
        state.observe_logits(b,torch.ones_like(canvas,dtype=torch.bool),47)
        state.begin(46,canvas)
        expected=1+3*.5*(.5 if mode=='mean' else 1.)
        assert torch.allclose(state.used_weights[:,:128],torch.full((1,128),expected))
        assert torch.all(state.used_weights[:,128:]==1)
        assert torch.equal(router.query_sensitivity,state.used_weights)
        state.begin(48,canvas)
        assert state.used_weights is None and router.query_sensitivity is None
