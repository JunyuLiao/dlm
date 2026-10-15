"""Process-matched diagnostic worker; one native engine, explicit request arms.

All request state is reset by the unchanged adapter lifecycle. Control requests
call the original bound V31 method directly. No sampler or forward is added.
This worker requires a frozen schedule, whose first entry is the warmup arm.
"""
import functools
import json
import os
from pathlib import Path
import runpy
import types

CONTROL='current_v31_control'
ALLOWED=(CONTROL,'full_swap8','residual','residual_direction_guard','full_swap8_direction_guard')


def bind_requests(adapter,runtime,original_mage,original_receipt,schedule):
    if not schedule or schedule[0]!=CONTROL or any(x not in ALLOWED for x in schedule):
        raise ValueError('explicit control warmup and supported request arms required')
    refined_mage=adapter._mage;refined_receipt=adapter._value_receipt
    old_begin=adapter.begin_request
    state=dict(index=-1,arm=None)
    @functools.wraps(old_begin)
    def begin(self,*args,**kwargs):
        index=state['index']+1
        if index>=len(schedule):raise RuntimeError('request exceeded frozen arm schedule')
        arm=schedule[index]
        if arm!=CONTROL:runtime.mode=arm
        self._mage=original_mage if arm==CONTROL else refined_mage
        result=old_begin(*args,**kwargs)  # Includes refinement reset and native reset.
        state.update(index=index,arm=arm)
        return result
    def receipt(self):
        source=original_receipt if state['arm']==CONTROL else refined_receipt
        return dict(source(),comparison_request_index=state['index'],comparison_arm=state['arm'],
                    comparison_design='same warmed engine; original native request reset')
    adapter.begin_request=types.MethodType(begin,adapter)
    adapter._value_receipt=types.MethodType(receipt,adapter)
    return state


if __name__=='__main__':
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_value_refinement_runtime import install
    gate=json.loads(Path(os.environ['VALUE_REFINEMENT_GATE']).read_text())
    schedule=json.loads(Path(os.environ['NATIVE_INTERLEAVED_ARMS']).read_text())
    if gate.get('status')!='passed' or not set(schedule).difference({CONTROL})<=set(gate.get('promoted_modes',[])):
        raise RuntimeError('request arms require the original same-state promotion gate')
    if os.environ.get('REPEATS','1')!='1':raise RuntimeError('frozen schedule requires REPEATS=1')
    runtime=install('full_swap8')
    original_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        original_mage=adapter._mage;original_receipt=adapter._value_receipt
        result=original_install(adapter)
        bind_requests(adapter,runtime,original_mage,original_receipt,schedule)
        return result
    vllm_adapter.install_vllm_patches=install_adapter
    runpy.run_path(os.environ['NATIVE_INTERLEAVED_BASE_WORKER'],run_name='__main__')
