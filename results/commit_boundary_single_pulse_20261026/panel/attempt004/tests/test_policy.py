from pathlib import Path
import importlib.util,sys
import torch
C=Path(__file__).resolve().parents[1]/'code';sys.path.insert(0,str(C));from bank import build,matching

def operands():return torch.zeros(256,16,8),torch.zeros(512,2,8),torch.ones(512,2,8)
def test_independent_ce_without_direction_pair():
 q,k,v=operands();b,_,_=build(q,k,v,256,.1)
 assert b['feasible_common'] and not b['qualified'];assert set(b['selections'])=={'current_error','support_sham'}
 assert all(len(u['tiles'])==1 for u in b['selections']['current_error']['units'])
def test_measurement_only_clean_preserves_choices_and_rng():
 q,k,v=operands();rng=torch.get_rng_state().clone();a,_,_=build(q,k,v,256,.1,True);b,p,_=build(q,k,v,256,.1,False)
 assert a['selections']==b['selections'] and p==[] and torch.equal(rng,torch.get_rng_state())
def test_legacy_qualified_choices_unchanged():
 spec=importlib.util.spec_from_file_location('legacy_bank',C.parents[3]/'response_behavior_and_stopping_headroom_20261025/track_h/attempt001/code/bank.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 g=torch.Generator().manual_seed(3844);q=torch.randn(256,16,8,generator=g);k=torch.randn(1024,2,8,generator=g);v=torch.randn(1024,2,8,generator=g)
 a,_,_=old.build(q,k,v,768,.1);b,_,_=build(q,k,v,768,.1)
 assert a['qualified']==b['qualified']
 for name,chosen in a['selections'].items():assert chosen==b['selections'][name]
def test_matching_boundary_and_floor():
 assert matching(dict(current_error=0,deleted_mass=0),dict(current_error=0,deleted_mass=.01))
 assert not matching(dict(current_error=.1,deleted_mass=.1),dict(current_error=.09,deleted_mass=.1))
def test_anchor_first_only():
 def trigger(C,ready,stable,conf,conv,step,count,anchor):
  count=count+1 if C>=1 and ready and stable and not conf and not conv else 0
  return count, anchor or (dict(canvas=C,after_completed_calls=step) if count>=2 and step<=46 else None)
 c,a=trigger(0,True,True,False,False,4,1,None);assert a is None
 c,a=trigger(1,True,True,False,False,46,1,None);assert a['after_completed_calls']==46
 c,b=trigger(2,True,True,False,False,20,1,a);assert b==a
 c,a=trigger(1,True,True,False,False,47,1,None);assert a is None
