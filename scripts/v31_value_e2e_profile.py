"""Isolated overhead profile; never produces accuracy or end-to-end claims."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from scripts import v31_value_decision_replay as io
from experiments.numerical_qk_reuse.v31_value_decision import moments, bounded_proposal
from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache


def measure(fn, repeats=5):
    for _ in range(2):
        value = fn()
    torch.cuda.synchronize()
    times, walls = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize()
        host = time.perf_counter()
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        walls.append(1000 * (time.perf_counter() - host))
        times.append(start.elapsed_time(end))
    return dict(cuda_elapsed_ms=statistics.median(times), host_ms=statistics.median(walls),
                cuda_samples_ms=times, host_samples_ms=walls,
                peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-before)


def run(output):
    if output.exists():
        raise RuntimeError('Preserve the prior profile; use a new attempt')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('H100 must be idle')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    from safetensors import safe_open
    index = io.read(io.MODEL / 'model.safetensors.index.json')['weight_map']
    weight_name = 'model.decoder.layers.29.self_attn.o_proj.weight'
    with safe_open(str(io.MODEL/index[weight_name]), framework='pt', device='cpu') as f:
        w = f.get_tensor(weight_name).float().cuda()
    gram = w[:, :512].T @ w[:, :512]
    groups = io.group_paths()
    selected = []
    for request in ('request001', 'request002'):
        for group, paths in sorted(groups.items()):
            if group[0] == request and group[2] == 'layer29':
                for path in paths:
                    state = torch.load(path, weights_only=True, map_location='cpu')
                    if state['selection_event'] and state['next'] is not None:
                        selected.append(path)
                        break
                if len(selected) and selected[-1].parent.name == request:
                    break
    if len(selected) != 2:
        raise RuntimeError('Missing predeclared LongBench layer29 states')
    config = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_sha256=io.sha(__file__), inputs={str(p.relative_to(io.SOURCE)):io.sha(p) for p in selected},
                  input_policy='First actual layer29 selection event per request001/002, independent of correctness',
                  tf32=False, device=torch.cuda.get_device_name(), timing='isolated kernel/module diagnostics only')
    io.write(output/'config.json', config, True)
    result=[]; start_time=time.monotonic()
    for path in selected:
        data=torch.load(path,weights_only=True,map_location='cpu')
        prefix=torch.load(path.parent/data['prefix_file'],weights_only=True,map_location='cuda')
        k=torch.cat((prefix['k'],data['tail_k'].cuda()),2)[0,0]
        v=torch.cat((prefix['v'],data['tail_v'].cuda()),2)[0,0]
        q=data['q'][0,0,:128].cuda()
        anchor=data['next'][0,0,0,:data['prefix']//64].cuda().bool()
        m=moments(q,k,v,data['prefix'],data['scale'])
        record=dict(request=path.parent.name,prefix=data['prefix'],rows=len(q),head=0,query_tile=0,
                    k=int(anchor.sum()),selection_event=data['selection_event'])
        record['moments']=measure(lambda:moments(q,k,v,data['prefix'],data['scale']))
        record['bounded_selection']=measure(lambda:bounded_proposal(m,anchor,swaps=8,gram=gram))
        k32,v32=k.float(),v.float()
        record['moments_preconverted_kv']=measure(lambda:moments(q,k32,v32,data['prefix'],data['scale']))
        record['dense_selected_reductions']=measure(lambda:(m.dense(),m.output(anchor)))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            bounded_proposal(m,anchor,swaps=8,gram=gram)
            torch.cuda.synchronize()
        names=('aten::_local_scalar_dense','aten::nonzero','aten::mm','aten::bmm','aten::index','aten::sort')
        record['selection_profile']={x.key:dict(count=x.count,self_cpu_us=x.self_cpu_time_total,
            self_device_us=x.self_device_time_total) for x in prof.key_averages() if x.key in names}
        result.append(record)
        io.write(output/f'{path.parent.name}.json',record,True)
        print(json.dumps(record),flush=True)
        del m,k32,v32,k,v,prefix
    generator=torch.Generator(device='cuda').manual_seed(1729)
    dense=torch.randn((256,16,512),device='cuda',generator=generator)*.1
    sparse=dense+torch.randn(dense.shape,device='cuda',generator=generator)*.01
    current=(sparse+.005).bfloat16()
    cache=ResidualDirectionCache.build(dense,sparse,w,'profile')
    correction=dict(shape=list(current.shape),source='synthetic fixed-seed inputs; real O projection',
                    build=measure(lambda:ResidualDirectionCache.build(dense,sparse,w,'profile')),
                    held=measure(lambda:cache.apply_cuda(current,'profile'),repeats=20))
    io.write(output/'correction.json',correction,True)
    io.write(output/'final_complete.json',dict(status='complete',config_sha256=io.sha(output/'config.json'),
        results_sha256={p.name:io.sha(p) for p in sorted(output.glob('*.json'))},
        gpu_reservation_s=time.monotonic()-start_time,timing='diagnostic only'),True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
