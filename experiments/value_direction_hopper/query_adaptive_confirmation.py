"""Predeclared two-seed confirmation, conditional on a matched-budget signal.

Seeds and selection gate were written before inspection of the final 130-run
matrix. This is still confirmation on reused prompts, not new held-out data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from .experiment import atomic
from .query_adaptive_study import cached,aggregate

SEEDS=(43,44)


def select(summary):
    index={r['condition']:r for r in summary}
    chosen=[]
    for target in (50,70):
        base=index[f'unweighted_s{target}']
        candidates=[]
        for method in ('M','C','T','MT','CT'):
            candidate=index[f'{method}_s{target}']
            matched=all(abs(candidate[k]-base[k])<=.02 for k in ('sparsity','global_sparsity','local_sparsity'))
            improvement=(candidate['accuracy']>=base['accuracy']+.02 or
                candidate['accuracy']>=base['accuracy']-.01 and
                candidate['mean_iterations']<=base['mean_iterations']-1.)
            if matched and improvement:candidates.append(candidate)
        if candidates:
            best=max(candidates,key=lambda r:(r['accuracy']-base['accuracy'])+
                .01*(base['mean_iterations']-r['mean_iterations']))
            chosen.append((best['method'],target))
    return chosen


def run(root):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    frozen=json.loads((root/'configs/frozen_policies.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if summary['missing']:raise ValueError('Primary matrix incomplete')
    selected=select(summary['summary'])
    atomic(root/'confirmation_plan.json',dict(seeds=SEEDS,selected=selected,
        rule='At each target, require overall/global/local within2pp and either >=2pp accuracy gain or >=1 fewer mean calls with accuracy no worse than1pp; choose highest predeclared score.'))
    if not selected:return
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
    projections=Projections();rows=json.loads((root/'configs/final_manifest.json').read_text())
    records=[]
    for seed in SEEDS:
        for method,target in selected:
            for name in ('unweighted',method):
                condition=f'{name}_s{target}';spec=cfg['methods'][name];policy=frozen['policies'][condition]
                for row in rows:
                    copy=dict(row,seed=seed)
                    result=cached(adapter,root,f'confirmation/seed{seed}',name,target,copy,spec,policy,cfg,projections,frozen['m_ref'])
                    records.append(dict(result,confirmation_seed=seed))
    table=[]
    for seed in SEEDS:
        for method,target in selected:
            for name in ('unweighted',method):
                group=[r for r in records if r['confirmation_seed']==seed and r['condition']==f'{name}_s{target}']
                sparsity,_=aggregate(group)
                table.append(dict(seed=seed,condition=f'{name}_s{target}',n=len(group),
                    accuracy=float(np.mean([r['score'] for r in group])),
                    mean_iterations=float(np.mean([r['steps'] for r in group])),
                    sparsity=sparsity['whole'],global_sparsity=sparsity['global'],local_sparsity=sparsity['local']))
    atomic(root/'confirmation_summary.json',dict(table=table,seeds=SEEDS,
        caveat='Same repeatedly examined prompts; different generation seeds are not fresh held-out samples'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    args=parser.parse_args();run(args.root)
