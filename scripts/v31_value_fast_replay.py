"""Source-bound support-equivalence and speed screen on saved LongBench states."""
import argparse
from pathlib import Path
import subprocess
import time

import torch

from scripts import v31_value_decision_replay as io
from scripts.v31_value_e2e_profile import measure
from experiments.numerical_qk_reuse.v31_value_decision import moments, bounded_proposal
from experiments.numerical_qk_reuse.v31_value_fast import bounded_proposal_device, dense_output


CODE = ['scripts/v31_value_fast_replay.py', 'scripts/v31_value_e2e_profile.py',
        'experiments/numerical_qk_reuse/v31_value_fast.py',
        'experiments/numerical_qk_reuse/v31_value_fast_kernel.py',
        'experiments/numerical_qk_reuse/v31_value_decision.py']


def run(output, limit):
    if output.exists():raise RuntimeError('Preserve immutable attempt')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('H100 must be idle')
    output.mkdir(parents=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    paths=[]
    for group, files in sorted(io.group_paths().items()):
        if group[0] not in ('request001','request002') or group[2]!='layer29':continue
        for path in files:
            state=torch.load(path,weights_only=True,map_location='cpu')
            if state['selection_event'] and state['next'] is not None:paths.append(path)
    if limit:paths=paths[:limit]
    if not paths:raise RuntimeError('No actual selection states')
    from safetensors import safe_open
    name='model.decoder.layers.29.self_attn.o_proj.weight'
    shard=io.read(io.MODEL/'model.safetensors.index.json')['weight_map'][name]
    with safe_open(str(io.MODEL/shard),framework='pt',device='cpu') as f:w=f.get_tensor(name).float().cuda()
    grams=[w[:,h*512:(h+1)*512].T@w[:,h*512:(h+1)*512] for h in range(16)]
    del w
    config=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:io.sha(io.ROOT/p) for p in CODE},
        input_sha256={str(p.relative_to(io.SOURCE)):io.sha(p) for p in paths},
        model_projection_shard_sha256=io.sha(io.MODEL/shard),frames=len(paths),limit_frames=limit,
        scope='All heads and Q128 tiles of saved LongBench layer29 selection states',
        fixed_prefix_k=128,tf32=False,backends=['exact','factored'])
    io.write(output/'config.json',config,True)
    records=[];totals={b:dict(units=0,mask_mismatches=0,swap_count_mismatches=0,budget_errors=0) for b in config['backends']}
    reservation=time.monotonic();profiled=set()
    for frame,path in enumerate(paths):
        d=torch.load(path,weights_only=True,map_location='cpu')
        prefix=torch.load(path.parent/d['prefix_file'],weights_only=True,map_location='cuda')
        k=torch.cat((prefix['k'],d['tail_k'].cuda()),2)[0].float()
        v=torch.cat((prefix['v'],d['tail_v'].cuda()),2)[0].float()
        record=dict(frame=frame,request=path.parent.name,prefix=d['prefix'],units=0,
                    mismatches=[],budget_errors=[],dense_reduction_mismatches=0)
        for h in range(16):
            for qt in range((d['n']+127)//128):
                q=d['q'][0,h,qt*128:min((qt+1)*128,d['n'])].cuda()
                anchor=d['next'][0,h,qt,:d['prefix']//64].cuda().bool();budget=int(anchor.sum())
                assert budget==min(128,len(anchor))
                m=moments(q,k[h//8],v[h//8],d['prefix'],d['scale'])
                ref,info=bounded_proposal(m,anchor,swaps=8,gram=grams[h])
                record['dense_reduction_mismatches']+=not torch.equal(dense_output(m),m.dense())
                for backend in config['backends']:
                    fast,fi=bounded_proposal_device(m,anchor,budget,gram=grams[h],backend=backend)
                    bad=not torch.equal(fast,ref);bad_count=int(fi['swaps'])!=info['swaps'];bad_budget=int(fast.sum())!=budget
                    totals[backend]['units']+=1;totals[backend]['mask_mismatches']+=bad
                    totals[backend]['swap_count_mismatches']+=bad_count;totals[backend]['budget_errors']+=bad_budget
                    if bad:record['mismatches'].append(dict(head=h,query_tile=qt,backend=backend,
                        different_blocks=int((fast!=ref).sum()),reference_swaps=info['swaps'],fast_swaps=int(fi['swaps'])))
                    if bad_budget:record['budget_errors'].append(dict(head=h,query_tile=qt,backend=backend))
                record['units']+=1
                if path.parent.name not in profiled:
                    timing={'reference':measure(lambda:bounded_proposal(m,anchor,swaps=8,gram=grams[h]))}
                    for backend in config['backends']:
                        timing[backend]=measure(lambda:bounded_proposal_device(m,anchor,budget,gram=grams[h],backend=backend))
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            bounded_proposal_device(m,anchor,budget,gram=grams[h],backend=backend)
                            torch.cuda.synchronize()
                        timing[backend]['host_scalar_reads']=sum(x.count for x in prof.key_averages() if x.key=='aten::_local_scalar_dense')
                        timing[backend]['dynamic_nonzero_calls']=sum(x.count for x in prof.key_averages() if x.key=='aten::nonzero')
                    record['timing']=timing;profiled.add(path.parent.name)
                del m
        records.append(record)
        io.write(output/f'frame{frame:03d}.json',record,True)
        print(dict(frame=frame,frames=len(paths),units=record['units'],mismatches=len(record['mismatches'])),flush=True)
        del prefix,k,v
    result=dict(status='complete_diagnostic',config_sha256=io.sha(output/'config.json'),totals=totals,
                frames=len(records),dense_reduction_mismatches=sum(r['dense_reduction_mismatches'] for r in records),
                timing={r['request']:r['timing'] for r in records if 'timing' in r},
                interpretation='Support equivalence on recorded states only; native qualification still required',
                gpu_reservation_s=time.monotonic()-reservation,
                records_sha256={f'frame{i:03d}.json':io.sha(output/f'frame{i:03d}.json') for i in range(len(records))})
    io.write(output/'summary.json',result,True)
    io.write(output/'final_complete.json',dict(status='complete',summary_sha256=io.sha(output/'summary.json')),True)
    print(totals,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit-frames',type=int,default=0)
    a=parser.parse_args();run(a.output.resolve(),a.limit_frames)
