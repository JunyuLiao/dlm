"""Qualified-policy native diagnostic with residual fingerprints and bookends."""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy


def install(schedule, destination):
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import FastRuntime
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
    state=dict(index=-1,active=False,production_anchor=False,events=[])
    add,step=LLMEngine.add_request,LLMEngine.step
    initialize=FastRuntime.initialize
    build=ResidualDirectionCache.build.__func__
    apply=ResidualDirectionCache.apply_cuda

    def digest(value):
        host=value.detach().cpu().contiguous()
        return dict(shape=list(host.shape),dtype=str(host.dtype),
            sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest())

    @functools.wraps(add)
    def begin(self,*args,**kwargs):
        index=state['index']+1
        state.update(index=index,active=schedule[index]=='observe',events=[])
        return add(self,*args,**kwargs)

    @functools.wraps(initialize)
    def anchor(self,adapter,q,b,scale,prefix,n,out,identity):
        active=state['active'] and self.corrects
        if active:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('diagnostic captured')
            state['events'].append(dict(kind='anchor_inputs',identity=identity,q=digest(q),
                support=digest(adapter.mage_state[29]['kept']),native_output=digest(out)))
        state['production_anchor']=active
        try:return initialize(self,adapter,q,b,scale,prefix,n,out,identity)
        finally:state['production_anchor']=False

    @classmethod
    @functools.wraps(build)
    def cache(cls,dense,selected,projection,identity):
        active=state['production_anchor']
        before=dict(dense=digest(dense),selected=digest(selected)) if active else None
        if active and not any(x['kind']=='projection' for x in state['events']):
            state['events'].append(dict(kind='projection',weight=digest(projection)))
        result=build(cls,dense,selected,projection,identity)
        if active:
            state['events'].append(dict(kind='anchor_cache',identity=identity,**before,
                cache={name:digest(getattr(result,name)) for name in (
                    'anchor_selected','residual','dual','residual_squared')}))
        return result

    @functools.wraps(apply)
    def held(self,current,identity):
        active=state['active']
        before=digest(current) if active else None
        result,alpha=apply(self,current,identity)
        if active:state['events'].append(dict(kind='held_correction',identity=identity,
            current=before,output=digest(result),alpha=digest(alpha)))
        return result,alpha

    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        result=step(self,*args,**kwargs)
        if state['active'] and any(o.finished for o in result):
            destination.mkdir(parents=True,exist_ok=True)
            path=destination/f"request{state['index']:03d}.json"
            with path.open('x') as f:json.dump(dict(events=state['events'],extra_forwards=0,
                cpu_fingerprints=True,clean_timing=False),f,sort_keys=True)
            state.update(active=False,events=[])
        return result

    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    FastRuntime.initialize=anchor
    ResidualDirectionCache.build=cache
    ResidualDirectionCache.apply_cuda=held
    return state


if __name__=='__main__':
    policy=Path(os.environ['HUMANEVAL_POLICY_SOURCE'])
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=os.environ['HUMANEVAL_POLICY_SOURCE_SHA256']:
        raise RuntimeError('qualified numerical policy drift')
    functions=runpy.run_path(str(policy));receipt=functions['install_compilation_policy']()
    receipt.update(intermediate_observer=True,residual_fingerprints=True)
    functions['install_kernel_inventory'](Path(os.environ['HUMANEVAL_KERNEL_INVENTORY']),receipt)
    schedule=json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text())
    state=install(schedule,Path(os.environ['HUMANEVAL_RESIDUAL_TRACE_DIR']))
    if os.environ.get('HUMANEVAL_ATTENTION_TRACE_DIR'):
        attention=runpy.run_path(os.environ['HUMANEVAL_ATTENTION_TRACE_SOURCE'])
        attention['install'](state,Path(os.environ['HUMANEVAL_ATTENTION_TRACE_DIR']))
    if os.environ.get('HUMANEVAL_GRAPH_TRACE_DIR'):
        graphs=runpy.run_path(os.environ['HUMANEVAL_GRAPH_TRACE_SOURCE'])
        graphs['install'](state,Path(os.environ['HUMANEVAL_GRAPH_TRACE_DIR']))
    trajectory=runpy.run_path(os.environ['HUMANEVAL_TRAJECTORY_SOURCE'])
    trajectory['install'](schedule,Path(os.environ['HUMANEVAL_TRAJECTORY_DIR']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
