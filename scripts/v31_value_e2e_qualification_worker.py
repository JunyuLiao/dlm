"""Native same-state oracle at every actual value observation; qualification only."""
import os
import runpy
from types import SimpleNamespace
import torch
from experiments.numerical_qk_reuse.v31_value_fast_runtime import FastRuntime
from experiments.numerical_qk_reuse.v31_value_refinement_runtime import Runtime

original=FastRuntime.initialize

def initialize(self,adapter,q,b,scale,prefix,n,out,identity):
    st=dict(adapter.mage_state[29]);st['kept']=st['kept'].clone()
    reference_adapter=SimpleNamespace(mage_state={29:st},mage_k=adapter.mage_k)
    oracle=Runtime(self.mode);oracle.model=self.model;oracle.grams=self.grams
    oracle.initialize(reference_adapter,q,b,scale,prefix,n,out,identity)
    original(self,adapter,q,b,scale,prefix,n,out,identity)
    if not torch.equal(st['kept'],adapter.mage_state[29]['kept']):
        raise RuntimeError('native same-state support mismatch')
    if not self.state.get('allkept') and self.corrects:
        a,b=oracle.state['direction_cache'],self.state['direction_cache']
        for name in ('anchor_selected','residual','dual','residual_squared'):
            if not torch.equal(getattr(a,name),getattr(b,name)):
                raise RuntimeError('native same-state correction cache mismatch: '+name)
        probe=a.anchor_selected.to(out.dtype)
        if not torch.equal(a.apply_cuda(probe,identity)[0],b.apply_cuda(probe,identity)[0]):
            raise RuntimeError('native same-state held correction mismatch')
    self.add('native_same_state_oracle_checks',1)

if __name__=='__main__':
    if os.environ.get('VALUE_AUDIT')!='1':raise RuntimeError('oracle is qualification-only')
    FastRuntime.initialize=initialize
    runpy.run_module('scripts.v31_value_e2e_worker',run_name='__main__')
