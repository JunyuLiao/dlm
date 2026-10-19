"""Diagnostic noninterference of unused graph-padding embedding rows."""
import functools
import json
import os
from pathlib import Path
import runpy


def install(schedule,root):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    state=dict(index=-1,mode=None,logits=None,call=0,focus=False)
    add,step=LLMEngine.add_request,LLMEngine.step
    prepare=dg.DiffusionGemmaModelState.prepare_inputs
    sample=dg._compiled_sample_step
    @functools.wraps(add)
    def begin(self,request_id,*args,**kwargs):
        index=state['index']+1
        state.update(index=index,mode=schedule[index],logits=None,call=0,
                     focus='-humaneval-0-' in request_id)
        return add(self,request_id,*args,**kwargs)
    @functools.wraps(prepare)
    def inputs(self,input_batch,req_states):
        out=prepare(self,input_batch,req_states)
        if input_batch.num_tokens_after_padding>input_batch.num_tokens:
            if state['mode']=='zero_pad':out['inputs_embeds'][input_batch.num_tokens:].zero_()
            elif state['mode']=='finite_pad':out['inputs_embeds'][input_batch.num_tokens:].fill_(100.)
        return out
    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        if state['focus'] and state['call']==0:
            state['logits']=args[0].detach().clone()
        state['call']+=1
        return sample(*args,**kwargs)
    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        out=step(self,*args,**kwargs)
        if state['logits'] is not None and any(o.finished for o in out):
            p=root/f"request{state['index']:03d}.pt"
            if p.exists():raise RuntimeError('immutable padding probe required')
            root.mkdir(parents=True,exist_ok=True)
            torch.save(dict(mode=state['mode'],logits=state['logits'].cpu()),p)
            state['logits']=None
        return out
    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    dg.DiffusionGemmaModelState.prepare_inputs=inputs
    dg._compiled_sample_step=sampler


if __name__=='__main__':
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_REPRO_TRACE']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
