"""One-factor dense native diagnostics; no model or sampler formulas change."""
import os
import runpy
import atexit
import json
from pathlib import Path


if __name__=='__main__':
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    factor=os.environ['HUMANEVAL_REPRO_FACTOR']
    if factor=='without_geometry':
        NativeGeometryAudit.install=lambda self:None
    elif factor=='without_logit_check':
        NativeGeometryAudit.logits=lambda self,scaled:None
    elif factor in ('deterministic_fill','deterministic_no_fill'):
        import torch
        import torch.utils.deterministic
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=(factor=='deterministic_fill')
    elif factor=='uva_depth64':
        from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
        original=UvaBufferPool.__init__
        def initialize(self,size,dtype,max_concurrency=None):
            return original(self,size,dtype,max(64,max_concurrency or 0))
        UvaBufferPool.__init__=initialize
    elif factor=='uva_device_copy':
        from vllm.v1.worker.gpu import buffer_utils
        buffer_utils.is_uva_available=lambda:False
    elif factor=='zero_padded_inputs':
        import vllm.model_executor.models.diffusion_gemma as dg
        original=dg.DiffusionGemmaModelState.prepare_inputs
        def prepare(self,input_batch,req_states):
            result=original(self,input_batch,req_states)
            if input_batch.num_tokens_after_padding>input_batch.num_tokens:
                result['inputs_embeds'][input_batch.num_tokens:].zero_()
            return result
        dg.DiffusionGemmaModelState.prepare_inputs=prepare
    elif factor=='full_precision_bf16_reduction':
        import torch
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
    elif factor=='prefer_cublaslt':
        import torch
        torch.backends.cuda.preferred_blas_library(backend='cublaslt')
    else:
        raise ValueError(factor)
    if os.environ.get('HUMANEVAL_REPRO_FACTOR_RECEIPT'):
        def receipt():
            import torch
            path=Path(os.environ['HUMANEVAL_REPRO_FACTOR_RECEIPT'])
            if path.exists():raise RuntimeError('immutable factor receipt required')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(factor=factor,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                blas_library=str(torch.backends.cuda.preferred_blas_library()),
                float32_matmul_precision=torch.get_float32_matmul_precision()),indent=2)+'\n')
        atexit.register(receipt)
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
