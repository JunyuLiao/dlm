"""Private startup identity probe: hashes actual loaded parameters and buffers."""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy


def install(path):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=dg.DiffusionGemmaForConditionalGeneration.load_weights
    original_runner=GPUModelRunner.load_model
    state={}

    def tensor_hash(t):
        h=hashlib.sha256()
        if not t.is_contiguous():
            h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            flat=t.detach().reshape(-1).view(torch.uint8)
            for start in range(0,flat.numel(),128*1024*1024):
                h.update(flat[start:start+128*1024*1024].cpu().numpy().tobytes())
        return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=h.hexdigest())

    @functools.wraps(original)
    def load(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        state['loaded_names']=sorted(result) if result is not None else None
        return result

    @functools.wraps(original_runner)
    def runner_load(self,*args,**kwargs):
        result=original_runner(self,*args,**kwargs)
        params={n:tensor_hash(p) for n,p in self.model.named_parameters()}
        buffers={n:tensor_hash(p) for n,p in self.model.named_buffers()}
        if path.exists():raise RuntimeError('immutable startup identity required')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(parameters=params,buffers=buffers,
            loaded_names=state.get('loaded_names'),
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,default_dtype=str(torch.get_default_dtype()),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            tensor_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())),indent=2)+'\n')
        return result
    dg.DiffusionGemmaForConditionalGeneration.load_weights=load
    GPUModelRunner.load_model=runner_load


if __name__=='__main__':
    install(Path(os.environ['HUMANEVAL_REPRO_MODEL_IDENTITY']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
