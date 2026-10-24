"""Standalone scientific native-trajectory plots from sanitized audit records."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=root/'trajectory_plots001';out.mkdir(exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR',str(root/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=defaultdict(lambda:defaultdict(list))
    for p in (root/'runs/panel').glob('*/*/audit[0-9][0-9][0-9].json'):
        r=json.loads(p.read_text());data[r['benchmark']][r['arm']].append(r)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for axis,task in zip(axes,['aime26','humaneval','longbench_v2']):
        for arm,records in data[task].items():
            values=[]
            for r in records:
                pending=0
                for event in r['native_trace']:
                    if event['commit']: values.append(pending);pending=0
                    else: pending+=1
            # Uncommitted native tail is not a completed canvas histogram bin.
            counts=np.bincount(values,minlength=49).astype(float)
            axis.plot(np.arange(1,49),counts[1:49]/counts.sum(),label=arm,linewidth=1)
        axis.set_title(task);axis.set_xlabel('Native calls per canvas');axis.set_ylabel('Fraction of observed canvases')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'calls_per_canvas.png',dpi=180);fig.savefig(out/'calls_per_canvas.pdf');plt.close(fig)
    # Average within each question first, then across questions, keeping seed
    # exposure separate from independent item count. Missing late calls are not
    # imputed: only surviving first canvases contribute at those native steps.
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for axis,task in zip(axes,['aime26','humaneval','longbench_v2']):
        for arm,records in data[task].items():
            curves=defaultdict(list)
            for r in records:
                first=[]
                for call in r['native_trace']:
                    if call['commit']:break
                    first.append(call['entropy_mean'])
                curves[r['id']].append(first)
            y=[]
            for index in range(48):
                items=[]
                for curves_per_seed in curves.values():
                    available=[c[index] for c in curves_per_seed if len(c)>index]
                    if available:items.append(np.mean(available))
                y.append(np.mean(items) if items else np.nan)
            axis.plot(np.arange(1,49),y,label=arm,linewidth=1)
        axis.axhline(.005,color='black',linestyle='--',linewidth=.8)
        axis.set_yscale('log');axis.set_title(task);axis.set_xlabel('Native first-canvas call');axis.set_ylabel('Survivor mean entropy (item weighted)')
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'first_canvas_entropy.png',dpi=180);fig.savefig(out/'first_canvas_entropy.pdf');plt.close(fig)
    (out/'plot_receipt.json').write_text(json.dumps(dict(passed=True,
        caveats=['Canvas histogram is descriptive, not independent-task inference.',
                 'Late entropy points condition on canvases that have not stopped; missing calls are not zero or forced continuation.',
                 'Entropy recomputed from native returned scaled logits; actual native confidence/stopping flags are separate.']),indent=2)+'\n')


if __name__=='__main__':main()
