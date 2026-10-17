"""Read-only early native divergence trace; no added forwards or RNG draws.

Full logits are fingerprinted, not written. Native canvas/state tensors and
reconstructed entropy acceptance masks remain private. This is instrumented
mechanism diagnosis, never a source of clean latency.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import types
import torch


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def trace(adapter,root,runtimes=None,schedule=None):
    from vllm.model_executor.models import diffusion_gemma as dg
    from experiments.numerical_qk_reuse.v31_same_state_suffix import distribution
    state=dict(index=-1,call=0,attention=None)
    first=int(os.environ['VALUE_TRACE_FIRST_REQUEST']);last=first+4
    limit=int(os.environ.get('VALUE_TRACE_CALL_LIMIT','6'))
    begin=adapter.begin_request;mage=adapter._mage;sample=dg._compiled_sample_step
    def selected():return first<=state['index']<last and state['call']<limit
    def start(self,*args,**kwargs):
        result=begin(*args,**kwargs);state['index']+=1;state['call']=0;state['attention']=None
        return result
    def attention(self,layer,q,b,scale,prefix,n):
        capture=selected() and layer==29
        old=self.mage_state.get(layer)
        if capture:
            rec=dict(q_hash=digest(q),canvas=self.canvas_id,prefix=prefix,
                     prior_support=old['kept'].detach().cpu() if old and old.get('kept') is not None else None,
                     prior_list_exists=old is not None and old.get('lists') is not None)
            old_list=old.get('lists') if old else None
        out=mage(layer,q,b,scale,prefix,n)
        if capture:
            new=self.mage_state.get(layer)
            rec.update(output_hash=digest(out),output=out.detach().cpu() if state['call']<6 else None,
                       selected_support=new['kept'].detach().cpu() if new and new.get('kept') is not None else None,
                       observed=new is not None and new.get('lists') is not old_list)
            if rec['observed'] or state['call']==0:
                rec.update(k_hash=digest(b['k']),v_hash=digest(b['v']))
            if runtimes is not None:
                runtime=runtimes.get(schedule[state['index']]['backend'])
                cache=runtime.state.get('direction_cache') if runtime is not None and runtime.state is not None else None
                rec['cache']={name:getattr(cache,name).detach().cpu() for name in
                              ('anchor_selected','residual','dual','residual_squared')} if cache is not None and rec['observed'] else None
            state['attention']=rec
        return out
    @functools.wraps(sample)
    def sampler(*args,**kwargs):
        capture=selected() and adapter.bound and adapter.step_ctx is not None and not adapter.step_ctx['encoder']
        if not capture:return sample(*args,**kwargs)
        slots=args[1]
        before=dict(logits_hash=digest(args[0]) if state['call']<6 else None,canvas=args[5][slots].detach().cpu(),
                    step=args[7][slots].detach().cpu(),encoder=args[8][slots].detach().cpu(),
                    cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state())
        result=sample(*args,**kwargs)
        after=dict(canvas=args[5][slots].detach().cpu(),argmax=args[6][slots].detach().cpu(),
                   step=args[7][slots].detach().cpu(),encoder=args[8][slots].detach().cpu(),
                   confident=args[9][slots].detach().cpu(),self_conditioning_hash=digest(args[10][slots]),
                   cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state())
        stats=distribution(result[0],kwargs.get('entropy_bound',.1))
        after.update(entropy=stats['entropy'].detach().cpu(),reconstructed_acceptance=stats['accepted'].detach().cpu())
        folder=root/f"request{state['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"call{state['call']:03d}.pt"
        if path.exists():raise RuntimeError('trace overwrite')
        torch.save(dict(before=before,after=after,attention=state['attention']),path)
        state['call']+=1;state['attention']=None
        return result
    adapter.begin_request=types.MethodType(start,adapter);adapter._mage=types.MethodType(attention,adapter)
    dg._compiled_sample_step=sampler


if __name__=='__main__':
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import install
    from scripts.v31_value_interleaved_worker import bind_requests
    from scripts.v31_value_cross_worker import bind
    from experiments.numerical_qk_reuse.v31_value_refinement_runtime import Runtime
    root=Path(os.environ['VALUE_TRACE_ROOT']);root.mkdir(parents=True)
    schedule=json.loads(Path(os.environ['NATIVE_INTERLEAVED_ARMS']).read_text())
    runtime=install('full_swap8');reference=Runtime('residual_direction_guard');old_install=vllm_adapter.install_vllm_patches
    def install_adapter(adapter):
        original_mage,original_receipt=adapter._mage,adapter._value_receipt
        result=old_install(adapter)
        if isinstance(schedule[0],dict):
            bind(adapter,runtime,reference,original_mage,original_receipt,schedule)
            trace(adapter,root,{'reference':reference,'device_exact':runtime},schedule)
        else:
            bind_requests(adapter,runtime,original_mage,original_receipt,schedule)
            trace(adapter,root)
        return result
    vllm_adapter.install_vllm_patches=install_adapter
    runpy.run_path(os.environ['NATIVE_INTERLEAVED_BASE_WORKER'],run_name='__main__')
