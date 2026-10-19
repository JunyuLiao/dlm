"""Localize first native forward divergence using same-engine clean bookends.

Prefill attention executes outside FULL decode graphs. Capture its actual Q/K/V
and output, then logical cached K/V after prefill and before first sampling.
No attention consumer, forward count, or native sampler formula is changed.
"""
import functools
import json
import os
from pathlib import Path
import runpy


def install(schedule,root):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    state=dict(index=-1,active=False,prefilled=False,call=0,layers={},cache={},data={})
    add,step=LLMEngine.add_request,LLMEngine.step
    forward=FlashAttentionImpl.forward
    prefill=dg.DiffusionGemmaSampler._finish_prefills
    sample=dg._compiled_sample_step
    def clone(x):return x.detach().clone() if isinstance(x,torch.Tensor) else None
    @functools.wraps(add)
    def begin(self,request_id,*args,**kwargs):
        index=state['index']+1
        state.update(index=index,active=schedule[index]=='observe',prefilled=False,call=0,
                     layers={},cache={},data={},focus='-humaneval-0-' in request_id)
        return add(self,request_id,*args,**kwargs)
    @functools.wraps(forward)
    def attention(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs):
        capture=(state['active'] and not state['prefilled'] and attn_metadata is not None
                 and not torch.cuda.is_current_stream_capturing())
        name=getattr(layer,'layer_name',None)
        if capture and name is None:raise RuntimeError('missing native layer identity')
        result=forward(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs)
        if capture:
            n=int(attn_metadata.num_actual_tokens)
            state['layers'][name]=dict(q=clone(query[:n]),key_input=clone(key[:n]) if key is not None else None,
                value_input=clone(value[:n]) if value is not None else None,output=clone(output[:n]))
            state['cache'][name]=(kv_cache,attn_metadata.block_table,self.head_size)
        return result
    def logical(n):
        result={}
        for name,(cache,table,d) in state['cache'].items():
            k,v=cache.transpose(1,2).split(d,dim=-1)
            page=k.shape[1];pages=table[0,:-(-n//page)].long()
            result[name]=dict(k=k[pages].reshape(-1,k.shape[2],d)[:n].clone(),
                v=v[pages].reshape(-1,v.shape[2],d)[:n].clone(),page=page)
        return result
    @functools.wraps(prefill)
    def finish_prefill(self,input_batch,prefill_indices_np):
        result=prefill(self,input_batch,prefill_indices_np)
        if state['active'] and not state['prefilled']:
            if input_batch.num_reqs!=1 or input_batch.num_tokens>460:
                raise RuntimeError('this observer requires frozen one-chunk diagnostic prefills')
            state['prefix']=int(input_batch.num_tokens)
            state['data']['prefill_cache']=logical(state['prefix'])
            state['prefilled']=True
        return result
    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        if state['active'] and state['call']==0:
            state['data']['first_denoise_cache']=logical(state['prefix']+kwargs['CL'])
            state['data']['first_logits_sample']=clone(args[0][:,::512])
        state['call']+=1
        return sample(*args,**kwargs)
    def cpu(x):
        if isinstance(x,torch.Tensor):return x.cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        return x
    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        out=step(self,*args,**kwargs)
        if state['active'] and any(o.finished for o in out):
            state['active']=False
            if len(state['layers'])!=30:raise RuntimeError('incomplete native prefill attention coverage')
            p=root/f"request{state['index']:03d}.pt"
            if p.exists():raise RuntimeError('immutable attention observation required')
            root.mkdir(parents=True,exist_ok=True)
            torch.save(cpu(dict(prefill_attention=state['layers'],**state['data'])),p)
            state['layers']={};state['cache']={};state['data']={}
        return out
    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    FlashAttentionImpl.forward=attention
    dg.DiffusionGemmaSampler._finish_prefills=finish_prefill
    dg._compiled_sample_step=sampler


if __name__=='__main__':
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_REPRO_TRACE']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
