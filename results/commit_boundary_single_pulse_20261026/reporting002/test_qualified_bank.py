"""Non-vacuous regression: a feasible opposite-bias directional bank."""
from pathlib import Path
import importlib.util
import torch
REPO=Path(__file__).resolve().parents[3]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=load('qualified_original',REPO/'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/code/bank.py')
NEW=load('qualified_online',REPO/'results/commit_boundary_single_pulse_20261026/panel/attempt004/code/bank.py')
def test_qualified_old_new_clean_supports_equal():
 q=torch.zeros(256,16,8);k=torch.zeros(512,2,8);v=torch.ones(512,2,8)
 for block,(dim,sign) in enumerate([(0,1),(0,-1),(1,1),(1,-1)]):v[block*64:(block+1)*64,:,dim]+=sign*.125
 before=torch.get_rng_state().clone();a,_,_=OLD.build(q,k,v,256,.1);b,_,_=NEW.build(q,k,v,256,.1,True);c,_,_=NEW.build(q,k,v,256,.1,False)
 assert a['qualified'] and b['qualified'] and c['qualified']
 assert set(a['selections'])=={'current_error','support_sham','direction_high','direction_low'}
 assert a['selections']==b['selections']==c['selections']
 assert b['directional_contrast']['cosine']<=-.99
 assert torch.equal(before,torch.get_rng_state())
