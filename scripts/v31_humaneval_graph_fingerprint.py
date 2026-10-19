"""Observe native graph boundaries without bypassing or replaying a graph."""
import functools
import json
import os
from pathlib import Path


def install(state,destination):
    import torch
    from torch.utils._pytree import tree_flatten
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import is_forward_context_available,get_forward_context
    import vllm.model_executor.models.diffusion_gemma as dg
    from humaneval_gpu_fingerprint import fingerprint_cuda
    add,step,sample=LLMEngine.add_request,LLMEngine.step,dg._compiled_sample_step
    graph=CUDAGraphWrapper.__call__
    identities={}
    state.update(graph_mode='off',graph_events=[],graph_logits=[],graph_samples=0,graph_calls=0)
    modes=json.loads(Path(os.environ['HUMANEVAL_GRAPH_MODES']).read_text()) if os.environ.get('HUMANEVAL_GRAPH_MODES') else None

    def values(value):
        leaves,_=tree_flatten(value)
        result=[]
        for i,x in enumerate(leaves):
            if not isinstance(x,torch.Tensor) or not x.is_cuda or not x.numel():continue
            # Exclude model parameters and large persistent storage. This is an
            # activation-boundary locator, not a claim that every argument is read.
            if x.numel()>4*256*2816 or (x.ndim>=2 and x.shape[0]>512) or (x.ndim==1 and x.numel()>512):continue
            result.append(dict(leaf=i,shape=list(x.shape),stride=list(x.stride()),
                dtype=str(x.dtype),gpu128=fingerprint_cuda(x)))
        return result

    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        return value

    @functools.wraps(add)
    def begin(self,*args,**kwargs):
        result=add(self,*args,**kwargs)
        mode=modes[state['index']] if modes is not None else 'full' if state['active'] else 'off'
        if mode not in ('off','full','graph_only'):raise RuntimeError('unknown graph observation mode')
        state.update(graph_events=[],graph_logits=[],graph_samples=0,graph_calls=0,graph_mode=mode)
        return result

    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        if state['graph_mode']=='graph_only' and args:
            state['graph_logits'].append(dict(sampler_index=state['graph_samples'],gpu128=fingerprint_cuda(args[0])))
        result=sample(*args,**kwargs)
        if state['graph_mode']!='off':state['graph_samples']+=1
        return result

    @functools.wraps(graph)
    def boundary(self,*args,**kwargs):
        if state['graph_mode']=='off':return graph(self,*args,**kwargs)
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('boundary observer entered capture')
        identity=identities.setdefault(id(self),len(identities))
        call=state['graph_calls'];state['graph_calls']+=1
        context=get_forward_context() if is_forward_context_available() else None
        descriptor=getattr(context,'batch_descriptor',None)
        entries=getattr(self,'concrete_cudagraph_entries',{})
        entry=entries.get(descriptor) if descriptor is not None else None
        metadata=dict(call=call,wrapper=identity,wrapper_mode=self.runtime_mode.name,
            native_runtime_mode=context.cudagraph_runtime_mode.name if context is not None else 'NO_CONTEXT',
            captured_entry_ready=entry is not None and entry.cudagraph is not None,
            sampler_index=state['graph_samples'])
        state['graph_events'].append(dict(point='before',**metadata,values=values((args,kwargs))))
        result=graph(self,*args,**kwargs)
        state['graph_events'].append(dict(point='after',**metadata,values=values(result)))
        return result

    @functools.wraps(step)
    def finish(self,*args,**kwargs):
        active,index=state['graph_mode']!='off',state['index']
        result=step(self,*args,**kwargs)
        if active and any(o.finished for o in result):
            destination.mkdir(parents=True,exist_ok=True)
            with (destination/f'request{index:03d}.json').open('x') as f:
                json.dump(cpu(dict(events=state['graph_events'],logits_before=state['graph_logits'],mode=state['graph_mode'],extra_forwards=0,
                    native_graph_dispatch_preserved=True,graph_bypasses=0,
                    partial_activation_argument_scope=True,clean_timing=False)),f,sort_keys=True)
            state.update(graph_events=[],graph_logits=[])
        return result

    LLMEngine.add_request,LLMEngine.step=begin,finish
    dg._compiled_sample_step=sampler
    CUDAGraphWrapper.__call__=boundary
