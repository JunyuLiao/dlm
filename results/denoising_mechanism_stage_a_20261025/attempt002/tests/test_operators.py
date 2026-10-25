import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from operators import anchors,operator,differential,support_bank,score_bank,evaluate_future,directions,pair_match


def test_anchors_do_not_force_continuation_or_substitute_canvas():
    assert anchors(2)==[]
    assert anchors(3)==[2]
    assert anchors(11)==[5,9]
    assert anchors(48)==[24,46]


def test_full_qkv_differential_matches_central_difference_with_prefix_fixed():
    gen=torch.Generator().manual_seed(181)
    q=torch.randn((4,7),generator=gen,dtype=torch.float64)
    k=torch.randn((192,7),generator=gen,dtype=torch.float64)
    v=torch.randn((192,5),generator=gen,dtype=torch.float64)
    d=[torch.randn(x.shape,generator=gen,dtype=x.dtype)*.01 for x in [q,k[128:],v[128:]]]
    p,o=operator(q,k,v,.3,[0]);actual=differential(p,q,k,v,d,128,.3)
    def f(sign):
        kk=k.clone();vv=v.clone();kk[128:]+=sign*1e-5*d[1];vv[128:]+=sign*1e-5*d[2]
        return operator(q+sign*1e-5*d[0],kk,vv,.3,[0])[1]
    assert torch.allclose(actual,(f(1)-f(-1))/2e-5,atol=1e-9)


def test_bank_has_whole_set_diversity_and_no_native_rng_draw():
    gen=torch.Generator().manual_seed(182)
    p=torch.rand((4,4096),generator=gen);p/=p.sum(-1,keepdim=True)
    v=torch.randn((4096,8),generator=gen)
    before=torch.get_rng_state().clone()
    a=support_bank(p,v,60,6,0)
    assert len(a)==48 and len(set(a))==48 and all(len(x)==6 for x in a)
    assert any(len(set(x)^set(a[0]))>=6 for x in a)
    assert torch.equal(before,torch.get_rng_state())


def test_direction_sham_preserves_component_motion_scale():
    q=torch.arange(24.).reshape(4,6);k=torch.ones(192,6);v=k*3
    prev=dict(q=q*.8,k_tail=k[128:]*.7,v_tail=v[128:]*.9)
    d=directions(q,k,v,prev,128,9)
    assert all(torch.allclose(a.norm(),b.norm()) for a,b in zip(d['qkv_past'],d['direction_sham']))


def test_future_results_cannot_change_frozen_bank_or_choices():
    import copy
    g=torch.Generator().manual_seed(183)
    q=torch.randn(4,6,generator=g)*.1;k=torch.randn(384,6,generator=g)*.1
    v=torch.ones(384,8)+torch.randn(384,8,generator=g)*.01
    prev=dict(q=q*.9,k_tail=k[256:]*.8,v_tail=v[256:]*.95)
    b,s=score_bank(q,k,v,prev,256,.1,0,1);original=copy.deepcopy(b)
    out=evaluate_future(b,s,q+.01,k,v,256,.1)
    assert b==original and len(out)==len(b['candidates'])
    for x in s['candidates']:
        assert torch.allclose(sum(x['qkv_components']),x['responses']['qkv_past'],atol=1e-6)


def test_error_mass_matching_does_not_confuse_residual_and_full():
    a=dict(current_error=.01,deleted_mass=.1,scores={'qkv_past':1.})
    b=dict(current_error=.0103,deleted_mass=.105,scores={'qkv_past':1.1})
    assert pair_match(a,b)==(True,False)
    b['scores']['qkv_past']=2
    assert pair_match(a,b)==(True,True)


def test_diagnostic_work_counts_every_actual_gemm_and_tail():
    from operators import WORK
    q=torch.ones(4,6);k=torch.ones(192,6);v=torch.ones(192,8)
    WORK.clear();p,_=operator(q,k,v,.1)
    differential(p,q,k,v,(q,k[128:],v[128:]),128,.1)
    assert WORK['QK_rectangles']==3+3+1
    assert WORK['PV_rectangles']==3+3+1
