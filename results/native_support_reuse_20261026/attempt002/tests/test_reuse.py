from pathlib import Path
import sys,copy
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from reuse_selectors import active_schedule,union_window,canonical_numeric,full_policy_regressions
from qualification import core_trace_equal,exact_request
from observations import observe
from op_bank import bias_metrics,matched,budget

@pytest.mark.parametrize('life,expected',[('SA',[0]),('AS',[1]),('held2',[0,1]),('refresh2',[0,1]),('matched_change2',[0,1]),('held4',[0,1,2,3]),('refresh4',[0,1,2,3])])
def test_schedules_and_natural_boundary(life,expected):
    assert [o for o in range(-1,6) if active_schedule(life,o)]==expected
    # The iterator ends at the real stop; it cannot fabricate a second call.
    observed_native_calls=[0]
    assert [o for o in observed_native_calls if active_schedule(life,o)]==[o for o in expected if o==0]

def test_overlapping_windows_are_unions_not_first_anchor():
    a={'late':dict(canvas=0,after_completed_calls=15),'early':dict(canvas=0,after_completed_calls=3)}
    assert union_window(a,0,4) and union_window(a,0,18)
    assert not union_window(a,1,4)

def test_exact_previous_pair_includes_unaligned_native_winners_and_ties():
    z=torch.tensor([[[2.,2.,0.],[0.,3.,1.],[2.,0.,3.],[3.,2.,1.]]]);winner=z.argmax(-1)
    args=[None]*15;args[1]=torch.tensor([0]);args[6]=winner;args[7]=torch.tensor([3]);args[8]=torch.tensor([False]);args[9]=torch.tensor([False]);args[13]=torch.stack([winner[0],winner[0]])[None];args[14]=torch.tensor([2])
    prev=dict(winner=torch.tensor([0,0,0,2]),competitor=torch.tensor([1,2,2,1]))
    row,p=observe(z,args,dict(ST=2),previous=prev,positions=True)
    assert p['winner'].tolist()==[0,1,2,0]
    assert p['previous_pair_current_margin'].tolist()==[0.,-1.,-1.,-1.]
    assert p['previous_pair_aligned'].tolist()==[True,False,False,False]
    assert p['margin'][0]==0

def test_signed_zero_is_not_numerical_SC_change():
    a=torch.tensor([0.,-0.,1.]);b=torch.tensor([-0.,0.,1.])
    assert a.view(torch.uint8).tolist()!=b.view(torch.uint8).tolist()
    assert canonical_numeric(a).view(torch.uint8).tolist()==canonical_numeric(b).view(torch.uint8).tolist()
    assert torch.signbit(a)[1] # copies only

def test_quality_gate_includes_unexposed_fallback():
    c={('q',42):dict(correct=True)};p=[dict(id='q',seed=42,correct=False,exposed=False)]
    assert full_policy_regressions(p,c)==[dict(id='q',seed=42)]

def test_tiny_actual_bias_cannot_certify_math_direction():
    a=torch.ones(4,128,3,dtype=torch.bfloat16);s=a.clone();m,b=bias_metrics(s,a,torch.full_like(a.float(),1e-5))
    assert m['actual_vs_FP32_cosine'] is None and m['roundoff_or_tiny_bias']
    s[0,0,0]+=.125;m,b=bias_metrics(s,a,torch.full_like(a.float(),1e-5))
    assert m['actual_vs_FP32_relative_residual']>1

def test_active_native_history_comparison_excludes_only_unready_fraction():
    a=dict(history_ready=False,stable_position_fraction=.1,converged=False);b=dict(a,stable_position_fraction=.7)
    assert core_trace_equal([a],[b])
    assert not core_trace_equal([dict(a,history_ready=True)],[dict(b,history_ready=True)])
    assert not core_trace_equal([a],[dict(b,converged=True)])

def test_integer_budget_alias_and_matching():
    assert budget(4,.05)==budget(4,.1)==1
    assert matched(dict(current_error=.1,deleted_mass=.1),dict(current_error=.099,deleted_mass=.101))
    assert not matched(dict(current_error=.1,deleted_mass=.1),dict(current_error=.08,deleted_mass=.101))

def test_shadow_bound_at_same_and_changed_queries():
    gen=torch.Generator().manual_seed(91);k=torch.randn(16,5,generator=gen);q0=torch.randn(3,5,generator=gen);q=q0+.1*torch.randn(3,5,generator=gen);keep=torch.arange(16)%3!=0;tail=torch.randn(4,5,generator=gen)
    for query in [q0,q]:
        z0=q0@k.T*.3;z=query@tail.T*.3;d=(query-q0).norm(dim=-1)*k.norm(dim=-1).max()*.3
        bound=(torch.logaddexp(z0[:,keep].logsumexp(-1)-d,z.logsumexp(-1))-torch.logaddexp(z0.logsumexp(-1)+d,z.logsumexp(-1))).exp()
        current=query@k.T*.3;actual=(torch.logaddexp(current[:,keep].logsumexp(-1),z.logsumexp(-1))-torch.logaddexp(current.logsumexp(-1),z.logsumexp(-1))).exp()
        assert bool((bound<=actual+1e-6).all())

def test_public_worker_imports_preserve_stdlib_selectors():
    import subprocess,selectors
    assert hasattr(selectors,'DefaultSelector')
    import worker,plan_branches,reuse_runtime
    assert worker.ROOT==plan_branches.ROOT==reuse_runtime.ROOT

def test_common_bank_adaptations_share_feasibility_and_do_not_draw_rng():
    from op_bank import build
    from reuse_selectors import augment,SELECTORS
    torch.set_num_threads(1);gen=torch.Generator().manual_seed(19)
    q=torch.randn(256,16,4,generator=gen)*.05;k=torch.randn(516,2,4,generator=gen)*.1;v=torch.randn(516,2,4,generator=gen)*.1+2
    probe=torch.randn(4,128,4,generator=gen);e=torch.ones(4,128)/256;stable=torch.zeros(4,128,dtype=torch.bool);margin=torch.ones(4,128)
    def operator(units):
        out=torch.empty_like(q);masks={(u['head'],u['qb']):u['tiles'] for u in units or []}
        for h in range(16):
            for qb in range(2):
                z=q[qb*128:(qb+1)*128,h]@k[:,h//8].T*.5
                for tile in masks.get((h,qb),[]):z[:,tile*64:(tile+1)*64]=-torch.inf
                out[qb*128:(qb+1)*128,h]=z.softmax(-1)@v[:,h//8]
        return out
    before=torch.get_rng_state().clone();bank,p,w=build(q,k,v,260,.5,.1,probe,e,stable,margin,operator,'frozen-toy')
    choices,work=augment(bank,p,q,k,v,.5,q*.99,dict(stable_positions=torch.zeros(256,dtype=torch.bool),margin=torch.ones(256)))
    assert set(choices)==set(SELECTORS) and all(bank['candidates'][i]['feasible'] for i in choices.values())
    assert torch.equal(before,torch.get_rng_state()) and work['response_QK_rectangles']>0
    assert all(r['response_J_S']>=0 and r['response_residual']>=0 and r['previous_protected_empty'] for r in bank['candidates'])
    assert bank['candidates'][choices['signed_high']]['signed_output_alignment']>=bank['candidates'][choices['signed_low']]['signed_output_alignment']
