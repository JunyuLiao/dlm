"""Standalone scientific plots; no performance inference from tile budgets."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=R/'boundary/attempt001/analysis001/summary.json';c=R/'panel/attempt003/analysis001/summary.json';B=json.loads(b.read_text());C=json.loads(c.read_text());out=R/'figures001';out.mkdir(exist_ok=False)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for i,x in enumerate(B['cases']):
  gain=x['A_remaining']-x['remaining'];ratio=x['N']/x['A_N'];axes[0].scatter(gain,ratio,color='tab:red' if not x['correct'] else 'tab:green');axes[0].annotate(x['label'].removeprefix('cell'),(gain,ratio),fontsize=7,xytext=(3,3+i%2*7),textcoords='offset points')
 axes[0].axhline(1,color='gray',linestyle='--');axes[0].set(xlabel='Remaining-call gain in target canvas (A minus S)',ylabel='Complete-request N ratio (S/A)',title='Historical states replayed; all ten treatments wrong')
 methods=['current_error_single_pulse','support_sham_single_pulse','direction_high_single_pulse','direction_low_single_pulse'];colors=['tab:blue','tab:orange','tab:green','tab:purple']
 for j,task in enumerate(['aime26','humaneval','longbench_v2']):
  for i,(arm,color) in enumerate(zip(methods,colors)):
   x=next(v for v in C['comparisons'] if v['task']==task and v['stratum']=='all' and v['cohort']=='policy_all' and v['method']==arm and v['reference']=='matched_allkept_layer29');e=x['estimates']['N'];point=e['point'];lo,hi=e['CI95'];pos=j+(i-1.5)*.14
   axes[1].errorbar(pos,point,yerr=[[max(0,point-lo)],[max(0,hi-point)]],color=color,marker='o',capsize=2,label=arm.removesuffix('_single_pulse') if j==0 else None)
 axes[1].axhline(1,color='gray',linestyle='--');axes[1].set_xticks(range(3),['AIME26','HumanEval','LongBench-v2']);axes[1].set(ylabel='Paired complete-request N ratio versus common A',title='Online policies;4question clusters/task, both seeds');axes[1].legend(fontsize=8)
 fig.savefig(out/'local_and_complete_work.png',dpi=180);fig.savefig(out/'local_and_complete_work.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained');arms=['corrected_dense_full','dense_piecewise','matched_allkept_layer29',*methods]
 for ax,task in zip(axes,['aime26','humaneval','longbench_v2']):
  rows=[next(x for x in C['aggregates'] if x['task']==task and x['stratum']=='all' and x['arm']==arm) for arm in arms]
  ax.bar(range(7),[x['correct'] for x in rows],color=['gray','silver','black',*colors]);ax.set_xticks(range(7),['FULL','PW','A','CE','sham','high','low'],rotation=35);ax.set_ylim(0,8.8);ax.set(title=task,ylabel='Official correct /8seed-cells (4questions)')
  for i,x in enumerate(rows):ax.text(i,x['correct']+.12,f"{x['exposed']} pulses",ha='center',fontsize=7,rotation=45)
 fig.savefig(out/'quality_and_exposure.png',dpi=180);fig.savefig(out/'quality_and_exposure.pdf');plt.close(fig)
 with (out/'receipt.json').open('x') as f:json.dump(dict(inputs={str(b.relative_to(R)):sha(b),str(c.relative_to(R)):sha(c)},scope='Descriptive finite study; quality/work and exposure jointly reported; no noninferiority or production promotion'),f,indent=2)
if __name__=='__main__':main()
