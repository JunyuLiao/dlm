"""Qualify deferred observation with same-engine native clean bookends."""
import functools
import json
import os
from pathlib import Path
import runpy


def install(schedule,root):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    state=dict(index=-1,active=False,call=0,entries=[])
    add,step=LLMEngine.add_request,LLMEngine.step
    initial=dg.DiffusionGemmaRequestStates.init_canvas
    prepare=dg.DiffusionGemmaModelState.prepare_inputs
    attn=dg.DiffusionGemmaModelState.prepare_attn
    sample=dg._compiled_sample_step
    def rng():return dict(cpu_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state().clone())
    def clone(t):return t.detach().clone()
    def enabled():return state['active'] and state['call']<8
    def record(kind,**data):
        state['entries'].append(dict(kind=kind,call=state['call'],stream=int(torch.cuda.current_stream().cuda_stream),**data))

    @functools.wraps(add)
    def begin(self,request_id,*args,**kwargs):
        i=state['index']+1
        state.update(index=i,active=schedule[i]=='observe',call=0,entries=[])
        state['focus']='-humaneval-0-' in request_id
        if state['active']:record('request',**rng())
        return add(self,request_id,*args,**kwargs)

    @functools.wraps(initial)
    def init_canvas(self,slots):
        before=rng() if enabled() else None
        result=initial(self,slots)
        if enabled():record('initial',canvas=clone(self.canvas[slots]),before=before,after=rng())
        return result

    @functools.wraps(prepare)
    def inputs(self,batch,req_states):
        out=prepare(self,batch,req_states)
        if enabled():
            record('model_inputs',n=batch.num_tokens,padded=batch.num_tokens_after_padding,
                draft=batch.num_draft_tokens,input_ids=clone(batch.input_ids),positions=clone(batch.positions),
                seq_lens=clone(batch.seq_lens),query_start_loc=clone(batch.query_start_loc),
                embeddings=clone(out['inputs_embeds']))
        return out

    @functools.wraps(attn)
    def metadata(self,*args,**kwargs):
        out=attn(self,*args,**kwargs)
        if enabled():
            values={}
            for name,meta in out.items():
                fields={k:clone(v) for k,v in vars(meta).items() if isinstance(v,torch.Tensor)} if hasattr(meta,'__dict__') else {}
                values[str(name)]=fields
            record('attention_metadata',causal=clone(self._causal_buf),metadata=values)
        return out

    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        if not enabled():return sample(*args,**kwargs)
        slots=args[1];logits=args[0]
        before=dict(canvas=clone(args[5][slots]),sc=clone(args[10][slots]),step=clone(args[7][slots]),
            encoder=clone(args[8][slots]),logits_slice=clone(logits[:,::512]),logits_argmax=logits.argmax(-1),**rng())
        if state['focus'] and state['call']==0:before['full_logits']=clone(logits)
        out=sample(*args,**kwargs)
        after=dict(canvas=clone(args[5][slots]),argmax=clone(args[6][slots]),step=clone(args[7][slots]),
            encoder=clone(args[8][slots]),sc=clone(args[10][slots]),**rng())
        record('sample',before=before,after=after)
        state['call']+=1
        return out

    def cpu(x):
        if isinstance(x,torch.Tensor):return x.cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        return x

    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        out=step(self,*args,**kwargs)
        if state['active'] and any(o.finished for o in out):
            state['active']=False
            p=root/f"request{state['index']:03d}.pt"
            if p.exists():raise RuntimeError('immutable observation attempt required')
            root.mkdir(parents=True,exist_ok=True);torch.save(cpu(state['entries']),p)
            state['entries']=[]
        return out

    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    dg.DiffusionGemmaRequestStates.init_canvas=init_canvas
    dg.DiffusionGemmaModelState.prepare_inputs=inputs
    dg.DiffusionGemmaModelState.prepare_attn=metadata
    dg._compiled_sample_step=sampler


if __name__=='__main__':
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_REPRO_TRACE']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
