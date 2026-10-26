"""Publication figures from complete finite diagnostics; no generation."""
from pathlib import Path
from collections import Counter
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    bp=R/'boundary/attempt001/analysis001/summary.json';sp=R/'panel/attempt004/analysis001/summary.json';pp=R/'panel/attempt004/paired_readout001/summary.json'
    b=json.loads(bp.read_text());s=json.loads(sp.read_text());p=json.loads(pp.read_text());out=R/'figures002';out.mkdir(exist_ok=False)
    colors=['#555555','#111111','#2878b5','#ed8b23','#45985a','#9363ad'];arms=['corrected_dense_full','matched_allkept_layer29','current_error_single_pulse','support_sham_single_pulse','direction_high_single_pulse','direction_low_single_pulse'];names=['FULL','A','CE','sham','high','low']
    fig,axes=plt.subplots(5,3,figsize=(12,13),layout='constrained')
    for j,task in enumerate(['aime26','humaneval','longbench_v2']):
        rows=[next(x for x in s['aggregates'] if x['task']==task and x['stratum']=='all' and x['arm']==arm) for arm in arms]
        for i,(key,label) in enumerate([('N','Total calls / A'),('C','Committed canvases / A'),('T','Output tokens / A')]):
            ax=axes[i,j];ax.bar(range(6),[x[key]/rows[1][key] for x in rows],color=colors);ax.axhline(1,color='gray',linestyle='--',linewidth=1);ax.set_ylabel(label);ax.set_xticks(range(6),names,rotation=25);ax.set_ylim(0,1.6)
            if i==0:ax.set_title(task)
        ax=axes[3,j];ax.bar(range(6),[x['correct'] for x in rows],color=colors);ax.set_xticks(range(6),names,rotation=25);ax.set_ylabel('Official correct / 8 seed cells');ax.set_ylim(0,8.4)
        ax=axes[4,j]
        for i,arm in enumerate(arms[2:]):
            v=next(x for x in p['primary_timing'] if x['task']==task and x['stratum']=='all' and x['method']==arm and x['reference']==arms[1])['estimates']['W'];lo,hi=v['CI95'];point=v['point']
            ax.errorbar(i,point,yerr=[[point-lo],[hi-point]],fmt='o',color=colors[i+2],capsize=3)
        ax.axhline(1,color='gray',linestyle='--');ax.set_ylim(.99,1.12);ax.set_xticks(range(4),names[2:]);ax.set_ylabel('Paired clean W / A, 95% CI')
    fig.suptitle('Single-pulse panel: 4 question clusters/task, paired seeds42/43\nTargeted finite cohort; all-kept consumer effects remain separate from deletion',fontsize=12)
    fig.savefig(out/'work_quality_latency.png',dpi=180);fig.savefig(out/'work_quality_latency.pdf');plt.close(fig)
    points=Counter((x['A_remaining']-x['remaining'],x['N']/x['A_N']) for x in b['cases'])
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    for (gain,ratio),count in points.items():
        ax.scatter(gain,ratio,s=45+count*25,color='#bd3535');ax.annotate(f'{count} support cases',(gain,ratio),xytext=(-5 if gain==3 else 8,12),textcoords='offset points',ha='right' if gain==3 else 'left',fontsize=9)
    ax.axhline(1,color='gray',linestyle='--');ax.set(xlim=(-.3,3.4),ylim=(.8,4.2),xlabel='Target-canvas remaining-call gain (A minus S)',ylabel='Complete-request N ratio (S / A)',title='Boundary replay: ten deleting cases, all wrong\nCoincident cases grouped; no independent-mask inference')
    fig.savefig(out/'boundary_local_vs_complete.png',dpi=180);fig.savefig(out/'boundary_local_vs_complete.pdf');plt.close(fig)
    with (out/'receipt.json').open('x') as f:json.dump(dict(inputs={str(x.relative_to(R)):sha(x) for x in [bp,sp,pp]},reader_sha256=sha(Path(__file__)),scope='Descriptive finite cohort; arithmetic work totals, official quality and paired clean timing remain distinct. PW equals FULL output/counters and is omitted only in this display.'),f,indent=2)
    print('Publication figures002 created.')
if __name__=='__main__':main()
