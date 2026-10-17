"""Explicit common-engine schedule for the qualified optimized value backend."""
import json
import os
from pathlib import Path
import runpy
from scripts.v31_value_interleaved_worker import bind_requests

if __name__=='__main__':
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import install
    schedule=json.loads(Path(os.environ['NATIVE_INTERLEAVED_ARMS']).read_text())
    if set(schedule)-{'current_v31_control','full_swap8','residual_direction_guard'}:
        raise RuntimeError('unplanned method')
    if os.environ.get('REPEATS','1')!='1':raise RuntimeError('single seed42 trial required')
    runtime=install('full_swap8')
    old_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        original_mage,original_receipt=adapter._mage,adapter._value_receipt
        result=old_install(adapter)
        bind_requests(adapter,runtime,original_mage,original_receipt,schedule)
        return result
    vllm_adapter.install_vllm_patches=install_adapter
    runpy.run_path(os.environ['NATIVE_INTERLEAVED_BASE_WORKER'],run_name='__main__')
