"""One native benchmark worker with explicit frozen optimized refresh dispatch."""
import os
import runpy
from pathlib import Path
import torch
import torch._inductor.config as cfg
cfg.deterministic=True
cfg.benchmark_combo_kernel=False
cfg.combo_kernels=False
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy,install_kernel_inventory
policy=install_compilation_policy()
install_kernel_inventory(Path(os.environ["OPTIMIZED_KERNEL_INVENTORY"]),policy)
from experiments.numerical_qk_reuse import v31_value_native_audit as native_audit_module
OriginalAudit=native_audit_module.NativeGeometryAudit
class CanvasAudit(OriginalAudit):
    def begin_request(self):
        super().begin_request();self._canvas_index=-1;self._previous_nk=None;self._canvas_tiles={}
    def record(self,encoder,draft,n,nk,mode=None):
        if self.bound and not encoder and draft:
            if nk!=self._previous_nk:self._canvas_index+=1;self._previous_nk=nk
            tiles=self.global_layers*native_audit_module.rectangles(self.heads,n,nk)
            self._canvas_tiles[str(self._canvas_index)]=self._canvas_tiles.get(str(self._canvas_index),0)+tiles
        return super().record(encoder,draft,n,nk,mode)
    def receipt(self):
        result=super().receipt()
        result['value_canvas_phase_tiles']={k:{'dense':dict(eligible_tiles=v,kept_tiles=v,skipped_tiles=0)} for k,v in self._canvas_tiles.items()}
        return result
native_audit_module.NativeGeometryAudit=CanvasAudit
from experiments.numerical_qk_reuse import vllm_adapter
from experiments.numerical_qk_reuse.optimized_refresh_runtime import install
original=vllm_adapter.install_vllm_patches

def bind(adapter):
    result=original(adapter)
    method=os.environ['OPTIMIZED_REFRESH_METHOD']
    if method in ('v2','v31'):
        install(adapter,method=method,library=os.environ['OPTIMIZED_REFRESH_LIBRARY'],
            observer_directory=os.environ['OPTIMIZED_REFRESH_OBSERVER_DIR'],schedule=2)
        if os.environ.get('RESIDENT_SPARSE_EXECUTION')=='1':
            from experiments.numerical_qk_reuse.resident_execution_runtime import install as resident_install
            resident_install(adapter,graph_buckets=os.environ.get('RESIDENT_SPARSE_GRAPH_BUCKETS')=='1',
                v31_paged_refresh=method=='v31' and os.environ.get('RESIDENT_V31_PAGED_REFRESH')=='1')
    return result
vllm_adapter.install_vllm_patches=bind
runpy.run_path(os.environ['OPTIMIZED_REFRESH_BASE_WORKER'],run_name='__main__')
