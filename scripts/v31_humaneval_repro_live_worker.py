"""Same-engine dense factor interventions with native RNG and sampling intact."""
import functools
import json
import os
from pathlib import Path
import runpy


def install(schedule):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    state=dict(index=-1,mode=None,finite=[])
    add,step=LLMEngine.add_request,LLMEngine.step
    prepare=dg.DiffusionGemmaModelState.prepare_attn
    sample=dg._compiled_sample_step

    @functools.wraps(add)
    def begin(self,*args,**kwargs):
        state['index']+=1
        state.update(mode=schedule[state['index']],finite=[])
        return add(self,*args,**kwargs)

    @functools.wraps(step)
    def engine_step(self,*args,**kwargs):
        out=step(self,*args,**kwargs)
        if state['mode'] in ('sync','audit'):
            torch.cuda.synchronize()
        return out

    @functools.wraps(prepare)
    def attn(self,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,
             kv_cache_config,for_capture=False,ubatch_idx=0):
        if state['mode'] in ('geometry','audit') and not for_capture and input_batch.num_reqs==1:
            slot=int(input_batch.idx_mapping_np[0])
            torch.stack((self.diffusion_states.is_encoder_phase[slot].long(),
                         input_batch.seq_lens[0].long())).tolist()
        return prepare(self,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,
                       kv_cache_config,for_capture=for_capture,ubatch_idx=ubatch_idx)

    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        out=sample(*args,**kwargs)
        if state['mode'] in ('finite','audit'):
            state['finite'].append(~torch.isfinite(out).all())
        return out

    LLMEngine.add_request,LLMEngine.step=begin,engine_step
    dg.DiffusionGemmaModelState.prepare_attn,dg._compiled_sample_step=attn,sampler


if __name__=='__main__':
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
