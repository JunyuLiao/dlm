"""Real identical-Q/K/V qualification for the no-reuse same-call baseline."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
from scripts.v31_refresh_factorial_replay import cpu_scores,cpu_topk
from scripts.v31_refresh_factor_analysis import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('preserve prior qualification attempt')
    source=args.source_study.resolve();inv=json.loads((source/'specs/real_snapshot_inventory.json').read_text())
    for rel,h in inv['sha256'].items():
        if sha(source/rel)!=h:raise RuntimeError('frozen QKV source changed')
    binding=json.loads((source/'operator/trajectory_binding.json').read_text())
    leaf=source/binding['completed_leaf']/'private/trajectories'
    groups={}
    for f in sorted(leaf.glob('request*/canvas*_call*_layer*.pt')):
        ds='longbench_v2_0shot_think' if f.parent.name in ('request001','request002') else 'aime26'
        groups.setdefault((ds,f.name.rsplit('_',1)[1]),[]).append(f)
    files=[f for _,paths in sorted(groups.items()) for f in (paths[0],paths[-1])]
    rows=[];capture={'count':0};original=torch.cuda.CUDAGraph.capture_begin
    def track(graph,*a,**kw):capture['count']+=1;return original(graph,*a,**kw)
    torch.cuda.CUDAGraph.capture_begin=track
    for f in files:
        d=torch.load(f,weights_only=True);prefix=torch.load(f.parent/d['prefix_file'],map_location='cuda',weights_only=True)
        q=d['q'].cuda();k=torch.cat((prefix['k'],d['tail_k'].cuda()),2);v=torch.cat((prefix['v'],d['tail_v'].cuda()),2)
        n,pt,h=d['n'],d['prefix']//64,q.shape[1];qb=(n+127)//128
        a=VllmMethodAdapter(['full_attention'],arm='mage',mage_select='fa4',mage_k=8192,
            mage_granularity='qblock_max',mage_always_refresh=True,value_audit=True)
        a.begin_request();a._mage_units_w=a._mage_held=a._k_override=None
        captured={};select=a._select_initial_refresh
        def remember(*args):
            out,mask=select(*args);captured['mask']=mask;return out,mask
        a._select_initial_refresh=remember
        out=a._mage(d['layer'],q,dict(k=k,v=v),d['scale'],d['prefix'],n)
        z=torch.empty(h,qb,pt,128,device='cuda')
        dense=observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),d['scale'],z)
        expected=cpu_topk(cpu_scores(z.reshape(h*qb,pt,128).cpu().numpy().astype(np.float64)),min(128,pt))
        mask=captured['mask'];matched=np.array_equal(mask[0,:,:,:pt].reshape(h*qb,pt).cpu().numpy(),expected)
        reference=v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(mask),d['scale'])
        error=(out.float()-dense.float()).norm(dim=-1)
        if not matched or not torch.equal(out,reference) or not torch.isfinite(out).all():
            raise RuntimeError('fresh map or same-call native sparse output parity failure')
        rows.append(dict(dataset=d['dataset'],layer=d['layer'],fixed_K=min(128,pt),CPU_map_match=True,
            native_sparse_output_bitwise_match=True,dense_output_error_mean=float(error.mean()),
            dense_output_error_max=float(error.max()),receipt=a._cached_receipt()))
        del q,k,v,z,dense,prefix,a,out,reference
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if capture['count']:raise RuntimeError('operator CUDA graph captures')
    args.output.write_text(json.dumps(dict(status='complete',snapshots=len(rows),validated_input_files=inv['files'],
        capture_count=capture['count'],rows=rows),indent=2)+'\n')
    print(json.dumps(dict(status='complete',snapshots=len(rows),capture_count=0)))


if __name__=='__main__':main()
