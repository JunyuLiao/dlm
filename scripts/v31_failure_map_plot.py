"""Standalone scientific failure-map figures from completed sanitized artifacts."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/v31_failure_map_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import v31_failure_map_study as s


def plot(study):
    analysis=s.base.read(study/'analysis/attempt001/analysis.json')
    verification=s.base.read(study/'verification/attempt001/verified_complete.json')
    dest=study/'figures/attempt001'
    if dest.exists():raise RuntimeError('new figure attempt required')
    dest.mkdir(parents=True)
    data=sorted(analysis['paired_question_cluster_intervals']['screen'],key=lambda x:x['k'])
    metrics=[('accuracy_delta_pp','Accuracy difference (percentage points)',0),('NC_ratio','Calls per canvas / dense FULL',1)]
    if verification['clean_timing_jit_qualified']:metrics.append(('W_ratio','Request time / dense FULL',1))
    fig,axes=plt.subplots(1,len(metrics),figsize=(4.1*len(metrics),3.5),layout='constrained')
    for ax,(metric,label,reference) in zip(axes,metrics):
        y=np.array([x['estimates'][metric]['estimate'] for x in data])
        ci=np.array([x['estimates'][metric]['ci95'] for x in data])
        ax.errorbar(range(len(data)),y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o-',color='#225ea8',capsize=4)
        ax.axhline(reference,color='#666666',ls='--',lw=1)
        ax.set_xticks(range(len(data)),[str(x['k']) for x in data]);ax.set_xlabel('V31 prefix budget (tokens)')
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('V31 budget screen: 16 development questions, two seeds\n95% question-cluster intervals; native sampler unchanged',fontsize=11)
    fig.savefig(dest/'budget_frontier.png',dpi=180);fig.savefig(dest/'budget_frontier.pdf');plt.close(fig)
    rows=s.load_phase(study,'screen');p=s.base.read(study/'protocol.json')
    cohort=sorted(p['cohort'],key=lambda x:(x['prompt_tokens'],x['index']))
    grid=np.zeros((len(cohort),len(s.KS)),dtype=int)
    all_cases={}
    for j,k in enumerate(s.KS):
        found=s.cases(rows,k,[1,42],[0],('dense_full_fix51994','allkept_fa4'))
        all_cases[str(k)]=found;by={x['index']:x for x in found}
        for i,c in enumerate(cohort):
            if c['index'] in by:grid[i,j]=1 if by[c['index']]['failure_type']=='accuracy' else 2
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(7.1,6.5),layout='constrained')
    ax.imshow(grid,cmap=ListedColormap(['#eeeeee','#e69f00','#56b4e9']),vmin=0,vmax=2,aspect='auto')
    ax.set_xticks(range(len(s.KS)),[str(x) for x in s.KS]);ax.set_xlabel('V31 prefix budget (tokens)')
    ax.set_yticks(range(len(cohort)),[f"Index {x['index']} | {x['prompt_tokens']/1024:.1f}K" for x in cohort],fontsize=8)
    for (i,j),v in np.ndenumerate(grid):ax.text(j,i,('—','A','C')[v],ha='center',va='center')
    ax.set_title('Repeated candidate failures on both screening seeds\nA: accuracy; C: convergence; —: gate not met (not proof of safety)',fontsize=10)
    fig.savefig(dest/'candidate_failure_map.png',dpi=180);fig.savefig(dest/'candidate_failure_map.pdf');plt.close(fig)
    s.base.write(dest/'figure_data.json',dict(cases=all_cases,cohort=cohort,comparisons=data,script_sha256=s.base.sha(Path(__file__))),True)
    print(dest)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    plot(ap.parse_args().study.resolve())
