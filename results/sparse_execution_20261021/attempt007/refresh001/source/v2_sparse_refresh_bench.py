"""Controlled H100 refresh benchmark. Synthetic results are diagnostics only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch


def timed(functions, rounds=9, repeats=12):
    for f in functions.values():
        for _ in range(3): f()
    torch.cuda.synchronize()
    results = {name: [] for name in functions}
    names = list(functions)
    for r in range(rounds):
        for name in names[r % len(names):]+names[:r % len(names)]:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            begin.record()
            for _ in range(repeats): functions[name]()
            end.record(); end.synchronize()
            results[name].append(begin.elapsed_time(end)/repeats)
    return {name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in results.items()}


def graph_functions(functions):
    """Construct graphs outside timing; retain their output allocations."""
    graphs, outputs = {}, {}
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): outputs[name] = fn()
        graphs[name] = graph
    return {name: graph.replay for name, graph in graphs.items()}, (graphs, outputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--keys', type=int, nargs='+', default=[449, 1024, 4096, 8192])
    p.add_argument('--synthetic-sparsity', type=float)
    a = p.parse_args()
    if a.out.exists(): raise RuntimeError('new attempt required')
    a.out.mkdir(parents=True)
    started = time.monotonic()
    from experiments.numerical_qk_reuse import v27_fa4, v31_fa4_observe as observation
    from experiments.numerical_qk_reuse.v2_sparse_refresh import refresh, discover, parallel_discover
    from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
    from scripts.v31_humaneval_observe_no_overlap import wrap_fwd
    observation._flash_attn_fwd = wrap_fwd(observation._flash_attn_fwd, dict(observing_calls=0))
    capture_count = [0]
    original_capture = torch.cuda.CUDAGraph.capture_begin
    def counted_capture(graph, *args, **kwargs):
        capture_count[0] += 1
        return original_capture(graph, *args, **kwargs)
    torch.cuda.CUDAGraph.capture_begin = counted_capture
    proof = dict(status='running', synthetic=True,
        device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in
            [Path(__file__), Path('experiments/numerical_qk_reuse/v2_sparse_refresh.py')]}, rows=[])
    g = torch.Generator(device='cuda').manual_seed(1729)
    inputs = []
    for nk in a.keys:
        q = torch.randn((1, 16, 256, 512), device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn((1, 2, nk, 512), device='cuda', dtype=torch.bfloat16, generator=g)
        inputs.append((str(nk), dict(q=q, k=k, v=torch.randn(k.shape, device='cuda', dtype=torch.bfloat16, generator=g),
            prefix=nk-256, scale=512**-.5, threshold=.56, layer=5)))
    for label, x in inputs:
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        q, k, v = x['q'], x['k'], x['v']
        nk, n, h, hk, d = k.shape[2], q.shape[2], q.shape[1], k.shape[1], q.shape[-1]
        pt, scale = x['prefix']//64, x['scale']
        matrix = Projections().get(x.get('layer', 5), hk, d, 'gaussian', 32, 1729, q.device)
        z = torch.empty((1, hk, nk, 32), device='cuda')
        norm = torch.empty((1, hk, nk), device='cuda')
        nu = project(v, matrix, z, norm, torch.ones_like(norm, dtype=torch.bool), 0)
        held = x.get('held')
        extra = dict(held=held, sticky=1.386 if held is not None else 0.)
        tau = x.get('threshold', .560071620607)
        calibration = None
        if a.synthetic_sparsity is not None:
            low, high = 0., 2.
            target = h*((n+127)//128)*((nk+63)//64)*(1-a.synthetic_sparsity)
            trials = []
            for _ in range(12):
                candidate = (low+high)/2
                chosen = discover(q,k,z,nu,scale,pt,candidate,ratio=True,**extra)[0]
                count = int(chosen.sum())
                trials.append(dict(threshold=candidate,kept=count))
                if count > target: low=candidate
                else: high=candidate
            winner = min(trials,key=lambda row:abs(row['kept']-target))
            tau = winner['threshold']
            calibration = dict(target_sparsity=a.synthetic_sparsity,trials=trials,
                achieved_sparsity=1-winner['kept']/(h*((n+127)//128)*((nk+63)//64)),
                scope='synthetic kernel diagnostic only; no benchmark calibration or accuracy use')
        observed = torch.empty((h, (n+127)//128, pt, 128), device='cuda')
        from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
        adapter = VllmMethodAdapter(['full_attention']*30, arm='mage', mage_select='fa4',
            mage_granularity='qblock_max', mage_k=128)
        adapter.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
        def v31_refresh():
            output, chosen = adapter._mage_select_fa4(q,k,v,scale,x['prefix'],n)
            return output, v27_fa4.block_sparse_tensors(chosen)
        functions = dict(
            v31_complete_refresh=v31_refresh,
            dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),
            dense_observing=lambda:observation.observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale,observed),
            parallel_discovery=lambda:parallel_discover(q,k,z,nu,scale,pt,tau,**extra),
            parallel_discovery_fast_scan=lambda:parallel_discover(q,k,z,nu,scale,pt,tau,fast_scan=True,**extra),
            fused_discovery_w8=lambda:discover(q,k,z,nu,scale,pt,tau,**extra),
            fused_discovery_w16=lambda:discover(q,k,z,nu,scale,pt,tau,warps=16,**extra),
            fused_discovery_ratio=lambda:discover(q,k,z,nu,scale,pt,tau,ratio=True,**extra),
            fused_discovery_ratio_w4=lambda:discover(q,k,z,nu,scale,pt,tau,ratio=True,warps=4,**extra),
            fused_discovery_split_d=lambda:discover(q,k,z,nu,scale,pt,tau,ratio=True,split_d=True,warps=4,**extra),
            sparse_refresh_parallel=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='parallel',**extra),
            sparse_refresh_parallel_fast_scan=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='parallel',fast_scan=True,**extra),
            sparse_refresh_fused_w8=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='fused',**extra),
            sparse_refresh_ratio=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='fused',ratio=True,**extra),
            sparse_refresh_ratio_w4=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='fused',ratio=True,warps=4,**extra),
            sparse_refresh_split_d=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='fused',ratio=True,split_d=True,warps=4,**extra),
            sparse_refresh_attention=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='attention',**extra),
            sparse_refresh_attention_w4=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='attention',warps=4,**extra),
            sparse_refresh_fused_w16=lambda:refresh(q,k,v,matrix,scale,pt,tau,backend='fused',warps=16,**extra))
        before_captures = capture_count[0]
        timing = timed(functions)
        if capture_count[0] != before_captures: raise RuntimeError('capture in kernel timing')
        graph_calls, graph_owners = graph_functions(functions)
        after_graph_build = capture_count[0]
        device_timing = timed(graph_calls, repeats=30)
        if capture_count[0] != after_graph_build: raise RuntimeError('capture during graph replay timing')
        keep = discover(q,k,z,nu,scale,pt,tau,**extra)[0]
        ref = parallel_discover(q,k,z,nu,scale,pt,tau,**extra)[0]
        ratio = discover(q,k,z,nu,scale,pt,tau,ratio=True,**extra)[0]
        attention_out,attention_keep,attention_lists=functions['sparse_refresh_attention']()
        masked_reference=v27_fa4.sparse_lists(q,k,v,attention_lists,scale)
        error=attention_out.float()-masked_reference.float()
        relative_error=float(error.norm()/masked_reference.float().norm().clamp_min(1.e-12))
        if relative_error>.01:raise RuntimeError('fused output operator tolerance exceeded')
        row = dict(label=label, queries=n, keys=nk, prefix_tiles=pt, threshold=tau,
            global_eligible_tiles=keep.numel(), global_kept_tiles=int(keep.sum()),
            prefix_eligible_tiles=keep.shape[0]*pt, prefix_kept_tiles=int(keep[:,:pt].sum()),
            fused_parallel_mask_differences=int((keep != ref).sum()), timing=timing,
            fused_ratio_mask_differences=int((keep!=ratio).sum()),
            attention_ratio_mask_differences=int((attention_keep.reshape_as(ratio)!=ratio).sum()),
            attention_fa4_relative_l2=relative_error,attention_fa4_max_abs=float(error.abs().max()),
            graph_replay_timing=device_timing, graph_captures_outside_timing=after_graph_build-before_captures,
            synthetic_work_calibration=calibration)
        proof['rows'].append(row)
        (a.out/'receipt.json').write_text(json.dumps(proof, indent=2)+'\n')
        print(json.dumps(dict(label=label,kept=int(keep.sum()),eligible=keep.numel(),
            host_inclusive_ms={k:round(v['median_ms'],5) for k,v in timing.items()},
            graph_replay_ms={k:round(v['median_ms'],5) for k,v in device_timing.items()})), flush=True)
    proof.update(status='complete',gpu_reservation_seconds=time.monotonic()-started,
                 total_graph_captures=capture_count[0],captures_during_timing=0,
                 timing='CUDA events; warmed interleaved; actual CUDAGraph.capture_begin counter; graph build separate')
    (a.out/'receipt.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__': main()
