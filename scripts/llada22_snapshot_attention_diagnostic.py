"""Same-snapshot native mass/output and independent selector-oracle audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from scripts.llada22_eos_trajectory_diagnostic import (
    V2, V31, V3A, TAU, immutable, pinned_modules, output_metrics, distribution, sha,
)


def mass_count_matched(scores, mandatory, counts):
    ranking = scores.masked_fill(mandatory, torch.inf).argsort(-1, descending=True, stable=True)
    out = torch.zeros_like(mandatory)
    out.scatter_(1, ranking, torch.arange(scores.shape[1],device=scores.device)[None] < counts[:,None])
    return out


@torch.inference_mode()
def probe(base, sources, out):
    torch.set_num_threads(4)
    attention, runtime = pinned_modules(base)
    kernels = __import__('experiments.numerical_qk_reuse.v31_value_kernels',fromlist=['statistics_cuda'])
    reference = __import__('experiments.numerical_qk_reuse.v31_value_selectors',fromlist=['statistics_reference'])
    rows, oracle = [], []
    for path in sources:
        snapshots = torch.load(path, map_location='cpu', weights_only=False)
        item = path.stem.replace('_snapshots','').replace('humaneval_','humaneval/')
        for s in snapshots:
            for layer, saved in s['layers'].items():
                q,k,v = (saved[n].cuda() for n in ['q','k','v'])
                keep = saved['keep'].cuda().reshape(q.shape[1],-1)
                rt = attention.AttentionRuntime(V2,TAU)
                z,nu = rt.project(v,layer)
                kt = keep.shape[-1]
                actual_stats = kernels.statistics_cuda(q,k,z,nu,saved['scale'],kt)
                h,nq = q.shape[1:3]
                kh = torch.arange(h) // (h//k.shape[1])
                score_cpu = torch.matmul(q[0].float().cpu(),k[0,kh.cuda()].float().cpu().transpose(-1,-2))*saved['scale']
                cpu_stats = reference.statistics_reference(score_cpu,z[0].cpu(),nu.cpu(),kt)
                valid = cpu_stats.valid
                lm_delta = actual_stats.log_mass.cpu()[valid]-cpu_stats.log_mass[valid]
                mu_delta = actual_stats.mean.cpu()[valid]-cpu_stats.mean[valid]
                mandatory = torch.zeros_like(keep)
                mandatory[:,s['block_start']//64:math_ceil(s['block_end'],64)] = True
                recompute,_,_,_ = kernels.online_cuda(actual_stats,TAU,True,mandatory,torch.zeros_like(mandatory),0.)
                cpu_keep,_,_,_ = reference.online_reference(cpu_stats,TAU,preserve_mass=True,
                                        mandatory=mandatory.cpu(),held=torch.zeros_like(mandatory.cpu()),sticky=0.)
                full_stats = reference.statistics_reference(score_cpu,v[0].float().cpu(),nu.cpu(),kt)
                full_keep,_,_,_ = reference.online_reference(full_stats,TAU,preserve_mass=True,
                                        mandatory=mandatory.cpu(),held=torch.zeros_like(mandatory.cpu()),sticky=0.)
                lm=actual_stats.log_mass
                normalizer=torch.logsumexp(lm,1,keepdim=True)
                score=(lm-torch.where(torch.isfinite(normalizer),normalizer,0.)).amax(-1)
                count_map=mass_count_matched(score,mandatory,keep.sum(-1))
                half_map=attention.budget_map(score,mandatory,actual_stats.valid.any(-1))
                oracle.append(dict(id=item,label=s['label'],layer=layer,
                                    logmass_max_abs=float(lm_delta.abs().max()),
                                    projected_mean_max_abs=float(mu_delta.abs().max()),
                                    independent_fp32_cpu_vs_cuda_v2_map_mismatches=int((cpu_keep.cuda()!=recompute).sum()),
                                    fresh_v2_vs_actual_map_mismatches=int((keep!=recompute).sum()),
                                    projected_vs_full_v_iou=float((recompute&full_keep.cuda()).sum()/(recompute|full_keep.cuda()).sum())))
                for name,map_ in [('actual_v2',keep),('fresh_v2',recompute),('mass_same_count_each_head',count_map),
                                   ('v31_half',half_map),('full_v_online_same_tau',full_keep.cuda())]:
                    met,full,sparse=output_metrics(q,k,v,map_,saved['scale'])
                    all_probs=score_cpu.softmax(-1)
                    tile_mass=torch.stack([all_probs[...,j*64:min((j+1)*64,k.shape[2])].sum(-1).mean(-1) for j in range(kt)],-1)
                    met.update(id=item,label=s['label'],layer=layer,method=name,call=s['call_in_block'],
                               eligible=map_.numel(),kept=int(map_.sum()),
                               refresh=saved['refresh'],tile_keep_fraction=map_.float().mean(0).tolist(),
                               tile_mass_mean=tile_mass.mean(0).tolist(),
                               tile_mass_max=tile_mass.max(0).values.tolist(),
                               first_and_mandatory_only_heads=int((map_==(mandatory | (torch.arange(kt,device=map_.device)[None]==0))).all(-1).sum()),
                               iou_with_actual_v2=float((keep&map_).sum()/(keep|map_).sum()))
                    if name=='actual_v2':
                        cons=saved['output'].cuda()[0].transpose(0,1).float()
                        met['consumer_vs_fp32_relative_l2']=float((cons-sparse).norm()/sparse.norm())
                    rows.append(met)
                del full_stats, cpu_stats, actual_stats
        print(json.dumps(dict(event='snapshot_audit_complete',item=item)),flush=True)
    immutable(out/'metrics.json',rows)
    immutable(out/'oracle.json',oracle)
    immutable(out/'final_complete.json',dict(status='complete',snapshot_layer_cases=len(oracle),
                                             metrics_sha256=sha(out/'metrics.json'),oracle_sha256=sha(out/'oracle.json')))


def math_ceil(x,b):
    return (x+b-1)//b


def main():
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args()
    protocol=json.loads((args.out/'protocol.json').read_text())
    if sha(__file__)!=protocol['runner_sha256'] or (args.out/'final_complete.json').exists():
        raise RuntimeError('fresh frozen snapshot diagnostic required')
    if not (args.source/'final_complete.json').exists():
        raise RuntimeError('source trajectory diagnostic is incomplete')
    paths=sorted((args.source/'private').glob('*_snapshots.pt'))
    if len(paths)!=protocol['source_item_count']:
        raise RuntimeError('unexpected snapshot cohort')
    probe(args.base.resolve(),paths,args.out.resolve())


if __name__=='__main__':
    main()
