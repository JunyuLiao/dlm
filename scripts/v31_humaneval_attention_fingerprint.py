"""Locate attention/routing differences; diagnostic integer hashes only."""
import functools
import json
import os
from pathlib import Path
import re


def install(state,destination):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from experiments.numerical_qk_reuse import vllm_adapter
    from humaneval_gpu_fingerprint import fingerprint_cuda
    add,step,sample=LLMEngine.add_request,LLMEngine.step,dg._compiled_sample_step
    install_adapter=vllm_adapter.install_vllm_patches
    sparse=observe=None
    raw_targets=json.loads(os.environ.get('HUMANEVAL_ATTENTION_CAPTURE','[]'))

    def fingerprint(value):
        if value is None:return None
        if not value.numel():return dict(shape=list(value.shape),dtype=str(value.dtype),empty=True)
        return dict(shape=list(value.shape),dtype=str(value.dtype),gpu128=fingerprint_cuda(value))

    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
        return value

    @functools.wraps(add)
    def begin(self,*args,**kwargs):
        result=add(self,*args,**kwargs)
        state.update(attention_events=[],attention_samples=0,attention_raw=[])
        return result

    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        result=sample(*args,**kwargs)
        if state['active']:state['attention_samples']+=1
        return result

    def installed(adapter):
        nonlocal sparse,observe
        result=install_adapter(adapter)
        # The base worker installs the native attention mask fixes before
        # adapter construction. Do not import/bind observer classes earlier.
        from experiments.numerical_qk_reuse import v27_fa4,v31_fa4_observe
        sparse,observe=v27_fa4.sparse_lists,v31_fa4_observe.observe_dense
        v27_fa4.sparse_lists=functools.wraps(sparse)(sparse_consumer)
        v31_fa4_observe.observe_dense=functools.wraps(observe)(observing_consumer)
        forward=FlashAttentionImpl.forward

        @functools.wraps(forward)
        def attention(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs):
            active=state['active'] and attn_metadata is not None
            if not active:return forward(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs)
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('attention diagnostic captured')
            name=getattr(layer,'layer_name','');match=re.search(r'\.layers\.(\d+)\.',name)
            if not match:raise RuntimeError('missing attention layer identity')
            index=int(match.group(1));n=int(attn_metadata.num_actual_tokens)
            context=dict(adapter.step_ctx or {})
            old=adapter.mage_state.get(index);old_lists=old.get('lists') if old else None
            event=dict(layer=index,sampler_index=state['attention_samples'],
                context=context,canvas=adapter.canvas_id,
                current_inputs={k:fingerprint(v[:n]) if v is not None else None
                    for k,v in [('q',query),('k',key),('v',value)]},
                trigger_at=adapter._trig_at,trigger_count=adapter._trig_count)
            if index==0:
                clock=getattr(adapter,'_jc',None)
                event['progress_state']=None if clock is None else {k:fingerprint(v) for k,v in clock.items()}
            capture=any(t['request']==state['index'] and t['layer']==index and
                t['sampler_index']==state['attention_samples'] for t in raw_targets)
            raw={k:v[:n].detach().clone() if v is not None else None
                 for k,v in [('q',query),('k',key),('v',value)]} if capture else None
            state['attention_current_event']=event
            try:out=forward(self,layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs)
            finally:state['attention_current_event']=None
            event['output']=fingerprint(output[:n])
            current=adapter.mage_state.get(index);buffer=adapter.buffers.get(index)
            prefix=context.get('seq_len',n)-n
            valid_buffer=(not context.get('encoder',True) and buffer is not None and
                          buffer['nk']==prefix+n and buffer['prefix']==prefix)
            if valid_buffer:
                event['logical_global_cache']={k:fingerprint(buffer[k]) for k in ('k','v')}
            if current is not None and current['canvas']==adapter.canvas_id and current['nk']==prefix+n:
                event['kept']=fingerprint(current['kept']) if current.get('kept') is not None else None
                event['new_lists']=old_lists is not current.get('lists')
                lists=current.get('lists')
                event['lists']={k:fingerprint(getattr(lists,k)) for k in (
                    'full_block_cnt','full_block_idx','mask_block_cnt','mask_block_idx')}
            if capture:
                raw.update(output=output[:n].detach().clone(),event=event,scale=float(self.scale))
                if valid_buffer:raw.update(logical_k=buffer['k'].detach().clone(),logical_v=buffer['v'].detach().clone())
                if current is not None and current.get('lists') is not None:
                    raw['lists']={k:getattr(current['lists'],k).detach().clone() for k in (
                        'full_block_cnt','full_block_idx','mask_block_cnt','mask_block_idx')}
                state['attention_raw'].append(raw)
            state['attention_events'].append(event)
            return out
        FlashAttentionImpl.forward=attention
        return result

    def sparse_consumer(q,k,v,lists,scale,*args,**kwargs):
        result=sparse(q,k,v,lists,scale,*args,**kwargs)
        event=state.get('attention_current_event')
        if event is not None:
            event['consumer']='full_block_lists'
            event['consumer_lists']={name:fingerprint(getattr(lists,name)) for name in (
                'full_block_cnt','full_block_idx','mask_block_cnt','mask_block_idx')}
            event['consumer_output']=fingerprint(result)
        return result

    def observing_consumer(q,k,v,scale,z,*args,**kwargs):
        result=observe(q,k,v,scale,z,*args,**kwargs)
        event=state.get('attention_current_event')
        if event is not None:
            event['consumer']='observing_dense'
            valid=z.permute(0,1,3,2).reshape(z.shape[0],-1,z.shape[2])[:,:q.shape[1],:] if z.numel() else z
            event['observed_log_mass']=fingerprint(valid)
            event['consumer_output']=fingerprint(result)
        return result

    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        active,index=state['active'],state['index']
        result=step(self,*args,**kwargs)
        if active and any(o.finished for o in result):
            destination.mkdir(parents=True,exist_ok=True)
            with (destination/f'request{index:03d}.json').open('x') as f:
                json.dump(cpu(dict(events=state['attention_events'],extra_forwards=0,
                    integer_fingerprint_locator=True,clean_timing=False)),f,sort_keys=True)
            if state['attention_raw']:
                def tensor_cpu(x):
                    if isinstance(x,torch.Tensor):return x.detach().cpu()
                    if isinstance(x,dict):return {k:tensor_cpu(v) for k,v in x.items()}
                    if isinstance(x,list):return [tensor_cpu(v) for v in x]
                    return x
                path=destination/f'request{index:03d}.private.pt'
                if path.exists():raise RuntimeError('immutable raw capture required')
                torch.save(tensor_cpu(state['attention_raw']),path)
            state.update(attention_events=[],attention_raw=[])
        return result

    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    dg._compiled_sample_step=sampler
    vllm_adapter.install_vllm_patches=installed
