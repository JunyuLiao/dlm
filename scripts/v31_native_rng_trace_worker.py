"""Private first-step trace without changing native sampling or tensor returns.

Capture the first two denoising calls of the first measured request. Tensors
and token IDs stay private. No random draw, extra model call or seed reset is
performed by the observer. The original worker still owns all generation.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch
from experiments.numerical_qk_reuse import vllm_adapter
from vllm.model_executor.models import diffusion_gemma as dg


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__=='__main__':
    root=Path(os.environ['NATIVE_RNG_TRACE_DIR']);root.mkdir(parents=True)
    state=dict(request=0,calls=0)
    original_begin=vllm_adapter.VllmMethodAdapter.begin_request
    @functools.wraps(original_begin)
    def begin(self,*args,**kwargs):
        state['request']+=1;state['calls']=0
        return original_begin(self,*args,**kwargs)
    vllm_adapter.VllmMethodAdapter.begin_request=begin
    original=dg._compiled_sample_step
    @functools.wraps(original)
    def sample(*args,**kwargs):
        adapter=vllm_adapter._ACTIVE
        selected=(state['request']==2 and state['calls']<2 and adapter is not None and
                  adapter.bound and adapter.step_ctx is not None and not adapter.step_ctx['encoder'])
        if not selected:return original(*args,**kwargs)
        index=state['calls'];state['calls']+=1
        before=dict(logits=args[0].detach().cpu(),canvas=args[5].detach().cpu(),
                    cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state())
        result=original(*args,**kwargs)
        after=dict(canvas=args[5].detach().cpu(),argmax=args[6].detach().cpu(),
                   cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state(),
                   scaled=result.detach().cpu())
        path=root/f'call{index:03d}.pt'
        torch.save(dict(before=before,after=after),path)
        summary=dict(request=state['request'],call=index,logits_shape=list(before['logits'].shape),
            before={name:digest(value) for name,value in before.items()},
            after={name:digest(value) for name,value in after.items()},
            scope='read-only observation; all tensor payloads remain private')
        (root/f'call{index:03d}.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        return result
    dg._compiled_sample_step=sample
    runpy.run_path(os.environ['NATIVE_RNG_BASE_WORKER'],run_name='__main__')
