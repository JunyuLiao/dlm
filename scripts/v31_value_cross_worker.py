"""Predeclared reference/device implementation cross in one native engine."""
import json
import os
from pathlib import Path
import runpy
import types


def bind(adapter, fast, reference, original_mage, original_receipt, schedule):
    old_begin=adapter.begin_request
    state=dict(index=-1,entry=None)
    def begin(self,*args,**kwargs):
        state['index']+=1
        if state['index']>=len(schedule):raise RuntimeError('schedule overflow')
        state['entry']=schedule[state['index']]
        entry=state['entry'];result=old_begin(*args,**kwargs)
        reference.reset();reference.model=fast.model
        if entry['backend']!='control':
            runtime=fast if entry['backend']=='device_exact' else reference
            runtime.mode=entry['mode']
        return result
    def mage(self,layer,q,b,scale,prefix,n):
        prior=self.mage_state.get(layer);previous=prior.get('lists') if prior is not None else None
        out=original_mage(layer,q,b,scale,prefix,n)
        entry=state['entry']
        if entry['backend']=='control':return out
        runtime=fast if entry['backend']=='device_exact' else reference
        return runtime.output(self,layer,q,b,scale,prefix,n,out,previous)
    def receipt(self):
        result=original_receipt();entry=state['entry']
        if entry['backend']!='control':
            runtime=fast if entry['backend']=='device_exact' else reference
            result=dict(result,**runtime.receipt(),value_refinement_prefix_budget=self.mage_k//64)
        return dict(result,cross_backend=entry['backend'],cross_mode=entry['mode'],cross_label=entry['label'],cross_request_index=state['index'])
    adapter.begin_request=types.MethodType(begin,adapter)
    adapter._mage=types.MethodType(mage,adapter)
    adapter._value_receipt=types.MethodType(receipt,adapter)


if __name__=='__main__':
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import install
    from experiments.numerical_qk_reuse.v31_value_refinement_runtime import Runtime
    schedule=json.loads(Path(os.environ['VALUE_CROSS_SCHEDULE']).read_text())
    if any(e['backend'] not in ('control','reference','device_exact') or
           e['mode'] not in ('current_v31_control','full_swap8','residual_direction_guard') for e in schedule):raise RuntimeError('unknown arm')
    fast=install('full_swap8');reference=Runtime('full_swap8')
    old_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        original_mage,original_receipt=adapter._mage,adapter._value_receipt
        result=old_install(adapter)
        bind(adapter,fast,reference,original_mage,original_receipt,schedule)
        return result
    vllm_adapter.install_vllm_patches=install_adapter
    runpy.run_path(os.environ['VALUE_CROSS_BASE_WORKER'],run_name='__main__')
