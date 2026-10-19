"""Private startup identity probe: hashes actual loaded parameters and buffers."""
import functools
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def install_cache_probe(path):
    from vllm.vllm_flash_attn.cute import interface
    original=interface._flash_attn_fwd.compile_cache
    observed={}
    class Probe:
        def __contains__(self,key):
            frame=sys._getframe(1)
            values=frame.f_locals
            if frame.f_code.co_name=='_flash_attn_fwd':
                dyn=values.get('dynamic_causal')
                tag=(dyn is not None,str(dyn.dtype) if dyn is not None else None)
                h=hashlib.sha256(repr(key).encode()).hexdigest()
                entry=observed.setdefault(h,dict(calls=0,contexts=[]))
                entry['calls']+=1
                if list(tag) not in entry['contexts']:entry['contexts'].append(list(tag))
            return key in original
        def __getitem__(self,key):return original[key]
        def __setitem__(self,key,value):original[key]=value
        def __getattr__(self,key):return getattr(original,key)
    interface._flash_attn_fwd.compile_cache=Probe()
    def flush():
        if path.exists():raise RuntimeError('immutable cache probe required')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(cache_type=type(original).__name__,keys=observed,
            dynamic_context_collisions=sum(len(e['contexts'])>1 for e in observed.values())),indent=2)+'\n')
    atexit.register(flush)


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
    path=Path(os.environ['HUMANEVAL_REPRO_MODEL_IDENTITY'])
    install_cache_probe(path.with_name(path.stem+'_fa4_cache.json'))
    install(path)
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
