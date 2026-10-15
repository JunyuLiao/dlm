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
import re

import torch
from experiments.numerical_qk_reuse import vllm_adapter
from vllm.model_executor.models import diffusion_gemma as dg


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__=='__main__':
    root=Path(os.environ['NATIVE_RNG_TRACE_DIR']);root.mkdir(parents=True)
    state=dict(request=0,calls=0)
    if os.environ.get('NATIVE_LAYER_TRACE')=='1':
        original_init=dg.DiffusionGemmaForConditionalGeneration.__init__
        @functools.wraps(original_init)
        def initialize(self,*args,**kwargs):
            original_init(self,*args,**kwargs);state['model']=self
        dg.DiffusionGemmaForConditionalGeneration.__init__=initialize
        original_install=vllm_adapter.install_vllm_patches
        def install(adapter):
            result=original_install(adapter)
            from vllm.v1.attention.backends import flash_attn as fa
            original_attention=fa.FlashAttentionImpl.forward
            @functools.wraps(original_attention)
            def attention(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs):
                active=vllm_adapter._ACTIVE;context=active.step_ctx if active is not None else None
                match=re.search(r'layers\.(\d+)\.',getattr(layer,'layer_name',''))
                label=None
                if state['request']==2 and state['calls']==0 and context is not None and active.bound and match:
                    phase='prefill' if context['encoder'] else 'decode'
                    label=f'{phase}_layer{int(match[1]):02d}'
                    if label in state['seen']:label=None
                if label is not None:
                    state['seen'].add(label)
                    actual_tokens=int(attn_metadata.num_actual_tokens)
                    snapshot={name:t[:actual_tokens].detach().cpu() if t is not None else None for name,t in
                              (('query',query),('key',key),('value',value))}
                    snapshot['actual_tokens']=actual_tokens
                    snapshot['metadata']={}
                    for name in ('seq_lens','query_start_loc','block_table','causal','max_query_len','max_seq_len'):
                        value_=getattr(attn_metadata,name,None)
                        if isinstance(value_,torch.Tensor):value_=value_.detach().cpu()
                        if isinstance(value_,(torch.Tensor,int,float,bool,tuple,list)) or value_ is None:
                            snapshot['metadata'][name]=value_
                result=original_attention(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs)
                if label is not None:
                    actual=result if isinstance(result,torch.Tensor) else output
                    snapshot['output']=actual[:actual_tokens].detach().cpu()
                    torch.save(snapshot,root/f'{label}.pt')
                return result
            fa.FlashAttentionImpl.forward=attention
            return result
        vllm_adapter.install_vllm_patches=install
    original_begin=vllm_adapter.VllmMethodAdapter.begin_request
    @functools.wraps(original_begin)
    def begin(self,*args,**kwargs):
        state['request']+=1;state['calls']=0;state['seen']=set()
        result=original_begin(self,*args,**kwargs)
        if state['request']==2 and os.environ.get('NATIVE_LAYER_TRACE')=='1':
            parameters={}
            for name,p in state['model'].named_parameters():
                if name.startswith(('model.layers.0.','model.embed_tokens.','self_conditioning.')):
                    parameters[name]=dict(shape=list(p.shape),dtype=str(p.dtype),sha256=digest(p))
            for name,p in state['model'].named_buffers():
                if name=='normalizer' or name.startswith(('model.layers.0.','model.embed_tokens.','self_conditioning.')):
                    parameters['buffer:'+name]=dict(shape=list(p.shape),dtype=str(p.dtype),sha256=digest(p))
            (root/'parameter_fingerprints.json').write_text(json.dumps(parameters,indent=2,sort_keys=True)+'\n')
        return result
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
