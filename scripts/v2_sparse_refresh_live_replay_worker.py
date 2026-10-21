"""Native HumanEval refresh replay: GPU-only operands, numeric results only.

This diagnostic repeats attention operations on current operands after the
unchanged V31 output has been computed. No extra model forward or persisted
activation, prompt, token, answer or generation is introduced.
"""
import atexit
import json
import os
from pathlib import Path
import runpy
import time


def main():
    import torch
    from experiments.numerical_qk_reuse import vllm_adapter, v27_fa4
    from experiments.numerical_qk_reuse.v2_sparse_refresh import refresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from scripts.v2_sparse_refresh_bench import timed, graph_functions
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_observe_no_overlap import install as corrected_observation
    install_compilation_policy()
    corrected_observation()
    destination = Path(os.environ['V2_REFRESH_REPLAY_RECEIPT'])
    if destination.exists(): raise RuntimeError('new diagnostic attempt required')
    original_select = vllm_adapter.VllmMethodAdapter._select_initial_refresh
    begin = vllm_adapter.VllmMethodAdapter.begin_request
    state = dict(request=-1, rows=[], phases={}, started=time.monotonic())
    projections = Projections()

    def starting(self, *args, **kwargs):
        state['request'] += 1
        state['phases'] = {}
        return begin(self, *args, **kwargs)

    def selecting(self, layer, q, b, scale, prefix, n):
        out, kept = original_select(self, layer, q, b, scale, prefix, n)
        if (state['request'] >= 2 and layer in (5, 29) and self.canvas_id in (0, 1, 3, 7, 15, 31)):
            identity = (layer, self.canvas_id)
            phase = state['phases'].get(identity, 0)
            state['phases'][identity] = phase+1
            k, v = b['k'], b['v']
            matrix = projections.get(layer,k.shape[1],v.shape[-1],'gaussian',32,1729,q.device)
            extra = dict(held=self._mage_held,sticky=0. if self._mage_held is None else self.mage_sticky)
            def reference():
                output, chosen = original_select(self,layer,q,b,scale,prefix,n)
                return output, v27_fa4.block_sparse_tensors(chosen)
            functions = dict(v31_complete_refresh=reference,
                dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),
                v2_sparse_parallel=lambda:refresh(q,k,v,matrix,scale,prefix//64,.560071620607,backend='parallel',**extra),
                v2_sparse_parallel_fast_scan=lambda:refresh(q,k,v,matrix,scale,prefix//64,.560071620607,backend='parallel',fast_scan=True,**extra),
                v2_sparse_fused=lambda:refresh(q,k,v,matrix,scale,prefix//64,.560071620607,ratio=True,**extra),
                v2_sparse_conditional_pv=lambda:refresh(q,k,v,matrix,scale,prefix//64,.560071620607,backend='attention',**extra))
            # V31 replay increments Python counters. Restore them and do not
            # retain any repeat output as a model result or reusable map.
            counters = dict(self.calls)
            host = timed(functions, rounds=5, repeats=4)
            graph_calls, owners = graph_functions({name:fn for name,fn in functions.items() if name != 'v31_complete_refresh'})
            device = timed(graph_calls, rounds=5, repeats=20)
            candidate = functions['v2_sparse_fused']()
            difference = (candidate[0].float()-out.float())
            row = dict(request=state['request'],canvas=int(self.canvas_id),layer=layer,phase=phase,
                queries=n,keys=k.shape[2],threshold=.560071620607,
                global_eligible_tiles=kept.numel(),v31_kept_tiles=int(kept.sum()),v2_kept_tiles=int(candidate[1].sum()),
                host_dispatch_timing=host,graph_replay_timing=device,
                v2_dense_output_relative_l2=float(difference.norm()/out.float().norm().clamp_min(1.e-12)),
                comparison='same current QKV; unchanged V31 trajectory; attention replay only; no accuracy or clean request claim')
            self.calls.clear();self.calls.update(counters)
            state['rows'].append(row)
            destination.write_text(json.dumps(dict(status='running',rows=state['rows']),indent=2)+'\n')
            print(json.dumps(dict(event='refresh_replay',request=row['request'],canvas=row['canvas'],layer=layer,
                keys=row['keys'],sparsity=1-row['v2_kept_tiles']/row['global_eligible_tiles'],
                ms={name:value['median_ms'] for name,value in host.items()})),flush=True)
        return out, kept
    vllm_adapter.VllmMethodAdapter.begin_request=starting
    vllm_adapter.VllmMethodAdapter._select_initial_refresh=selecting
    def save():
        destination.write_text(json.dumps(dict(status='complete',rows=state['rows'],
            gpu_reservation_seconds=time.monotonic()-state['started'],
            operands_persisted=False,raw_generations_persisted=False,extra_model_forwards=0,
            native_output_path='unchanged corrected V31; diagnostic attention replay'),indent=2)+'\n')
    atexit.register(save)
    runpy.run_path(os.environ['V2_REFRESH_BASE_WORKER'],run_name='__main__')


if __name__=='__main__':main()
