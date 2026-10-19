"""Observe native graph boundaries without bypassing or replaying a graph."""
import functools
import json


def install(state,destination):
    import torch
    from torch.utils._pytree import tree_flatten
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    import vllm.model_executor.models.diffusion_gemma as dg
    from humaneval_gpu_fingerprint import fingerprint_cuda
    add,step,sample=LLMEngine.add_request,LLMEngine.step,dg._compiled_sample_step
    graph=CUDAGraphWrapper.__call__
    identities={}

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
        state.update(graph_events=[],graph_samples=0,graph_calls=0)
        return result

    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        result=sample(*args,**kwargs)
        if state['active']:state['graph_samples']+=1
        return result

    @functools.wraps(graph)
    def boundary(self,*args,**kwargs):
        if not state['active']:return graph(self,*args,**kwargs)
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('boundary observer entered capture')
        identity=identities.setdefault(id(self),len(identities))
        call=state['graph_calls'];state['graph_calls']+=1
        metadata=dict(call=call,wrapper=identity,mode=self.runtime_mode.name,sampler_index=state['graph_samples'])
        state['graph_events'].append(dict(point='before',**metadata,values=values((args,kwargs))))
        result=graph(self,*args,**kwargs)
        state['graph_events'].append(dict(point='after',**metadata,values=values(result)))
        return result

    @functools.wraps(step)
    def finish(self,*args,**kwargs):
        active,index=state['active'],state['index']
        result=step(self,*args,**kwargs)
        if active and any(o.finished for o in result):
            destination.mkdir(parents=True,exist_ok=True)
            with (destination/f'request{index:03d}.json').open('x') as f:
                json.dump(cpu(dict(events=state['graph_events'],extra_forwards=0,
                    native_graph_dispatch_preserved=True,graph_bypasses=0,
                    partial_activation_argument_scope=True,clean_timing=False)),f,sort_keys=True)
            state.update(graph_events=[])
        return result

    LLMEngine.add_request,LLMEngine.step=begin,finish
    dg._compiled_sample_step=sampler
    CUDAGraphWrapper.__call__=boundary
