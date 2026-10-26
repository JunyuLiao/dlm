from pathlib import Path
import os,sys,subprocess
import torch
CODE=Path(__file__).resolve().parents[1]/'code';sys.path.insert(0,str(CODE))
from op_bank import budget,matched,bias_metrics,build,UNITS,gather_units
from pulse_runtime import core_trace_equal,active_site

def test_budget_rounding_and_matching_floors():
    assert budget(4,.05)==budget(4,.1)==1
    assert budget(100,.05)==5 and budget(100,.1)==10
    assert matched(dict(current_error=0.,deleted_mass=0.),dict(current_error=4e-8,deleted_mass=.01))
    assert not matched(dict(current_error=.1,deleted_mass=0.),dict(current_error=.09,deleted_mass=0.))

def test_actual_difference_near_roundoff_is_not_FP32_direction():
    a=torch.ones(4,128,3,dtype=torch.bfloat16);s=a.clone();m,b=bias_metrics(s,a,torch.full_like(a.float(),1e-5))
    assert b.abs().sum()==0 and m['actual_vs_FP32_cosine'] is None and m['roundoff_or_tiny_bias']
    s[0,0,0]+=0.125;m,b=bias_metrics(s,a)
    assert m['bias_direction_available'] and m['current_error']>0

def test_scalar_average_does_not_hide_position_changes():
    a=torch.ones(4,128,2);s=a.clone();s[0,0]+=.1;s[0,1]-=.1
    m,b=bias_metrics(s,a)
    assert b.mean().abs()<1e-8 and m['bias_Frobenius']>0 and m['current_error']>.09

def test_early_stop_is_not_next_canvas_and_trace_optional_metrics():
    anchor=dict(canvas=1,after_completed_calls=4)
    assert active_site(dict(canvas=1,step=5),anchor)
    assert not active_site(dict(canvas=2,step=1),anchor)
    assert core_trace_equal([dict(commit=False,entropy_by_position=[1],native_step=5)],[dict(commit=False,native_step=5)])
    assert not core_trace_equal([], [dict(commit=True)])

def test_diverse_bank_uses_actual_S_minus_A_and_diagnostic_rng_only():
    torch.set_num_threads(1);gen=torch.Generator().manual_seed(762)
    q=torch.randn(256,16,4,generator=gen)*.05;k=torch.randn(512,2,4,generator=gen)*.1;v=torch.randn(512,2,4,generator=gen)*.1+2
    probe=torch.randn(4,128,4,generator=gen);ew=torch.ones(4,128)/256;protected=torch.zeros(4,128,dtype=torch.bool);margin=torch.ones(4,128)
    def operator(units):
        masks={(u['head'],u['qb']):u['tiles'] for u in units or []};out=torch.empty_like(q)
        for h in range(16):
            for qb in range(2):
                scores=q[qb*128:(qb+1)*128,h]@k[:,h//8].T*.5
                for tile in masks.get((h,qb),[]):scores[:,tile*64:(tile+1)*64]=-torch.inf
                out[qb*128:(qb+1)*128,h]=scores.softmax(-1)@v[:,h//8]
        return out
    before=torch.get_rng_state().clone();bank,priv,work=build(q,k,v,256,.5,.1,probe,ew,protected,margin,operator,'toy-frozen')
    assert torch.equal(before,torch.get_rng_state())
    assert bank['allkept_repeat_exact'] and len(bank['screen'])>=4 and len(bank['screen'])<=8
    ce=bank['candidates'][bank['selections']['current_error']]
    for f in ['support_sham','readout_margin']:assert matched(ce,bank['candidates'][bank['selections'][f]])
    for i,c in enumerate(bank['candidates']):
        assert all(len(u['tiles'])==1 and max(u['tiles'])<4 for u in c['units'])
        assert torch.equal(priv['candidates'][i]['actual_bias'],priv['candidates'][i]['sparse'].float()-priv['dense'].float())
    assert work['native_S_QK_rectangles']>0

def test_worker_imports_do_not_shadow_stdlib():
    p=subprocess.run([sys.executable,'-c','import torch,select,op_bank,pulse_runtime,bank_plan,study_worker;assert hasattr(select,"select")'],cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
    assert p.returncode==0,p.stderr.decode()[-1200:]

def test_native_history_is_compared_across_history_slots_per_position():
    from observations import observe
    # Real indexing retains a leading request dimension: [1,ST,CL].
    z=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]]);winner=z.argmax(-1)
    args=[None]*15;args[1]=torch.tensor([0]);args[6]=winner;args[7]=torch.tensor([2]);args[8]=torch.tensor([False]);args[9]=torch.tensor([False]);args[13]=torch.stack([winner[0],winner[0].clone()])[None];args[14]=torch.tensor([2])
    args[13][0,1,2]=1
    row,private=observe(z,args,dict(ST=2))
    assert private['stable_positions'].tolist()==[True,True,False,True] and not row['stable']
