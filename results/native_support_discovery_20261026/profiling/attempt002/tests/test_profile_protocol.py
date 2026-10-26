from pathlib import Path
import importlib.util,os,subprocess,sys
import torch

CODE=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(CODE))
from native_harness import anchors

def row(canvas,t,stable,confidence=False,converged=False):
    return dict(commit=False,canvas=canvas,native_step=t,stable=stable,history_ready=t>=2,confidence=confidence,converged=converged,cap_reached=t==48)

def test_first_observable_strata_and_native_next_call_room():
    trace=[row(0,1,False),row(0,2,False),row(0,3,False),row(0,4,True),row(0,5,True,True,True),dict(commit=True,canvas=0,native_step=0),row(1,1,False)]
    a=anchors(trace,300)
    assert a['instability']['after_completed_calls']==3 and a['instability']['optimistic_U']==1
    assert a['confidence']['after_completed_calls']==4 and a['confidence']['optimistic_U']==0

def test_tiny_prefix_missing_and_no_later_search_within_stratum():
    trace=[row(0,3,False),row(0,4,True),row(1,2,True),row(1,3,True),row(1,4,True,True,True)]
    a=anchors(trace,100)
    assert 'instability' not in a and a['confidence']['canvas']==1 and a['confidence']['after_completed_calls']==2

def test_native_cap_does_not_force_next_call():
    a=anchors([row(0,46,True),row(0,47,True),row(0,48,True,False,True)],300)
    assert a['confidence']['after_completed_calls']==46 and a['confidence']['reference_cap'] and a['confidence']['optimistic_U']==1

def test_worker_directory_has_no_stdlib_module_collision():
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-c','import torch,select,json; import shared_consumer,observations,native_harness; assert hasattr(select,"select")'],cwd=CODE,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert p.returncode==0,p.stderr.decode()[-1200:]

def test_actual_native_winner_not_topk_tie_substitute():
    from observations import observe
    # A tied vocabulary's topk index may differ from argmax. Native argmax wins.
    z=torch.tensor([[[3.,3.,3.,0.],[0.,2.,1.,0.]]]);native=z.argmax(-1)
    args=[None]*15;args[1]=torch.tensor([0]);args[6]=native.clone();args[7]=torch.tensor([2]);args[8]=torch.tensor([False]);args[9]=torch.tensor([False]);args[13]=torch.stack([native[0],native[0]])[None];args[14]=torch.tensor([2])
    measured,private=observe(z,args,dict(ST=2))
    assert torch.equal(private['winner'],native.reshape(-1)) and measured['history_ready']
    assert private['margin'][0]==0 and private['competitor'][0]!=private['winner'][0]
