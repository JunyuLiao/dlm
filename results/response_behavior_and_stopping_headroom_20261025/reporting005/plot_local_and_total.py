"""Show all controls and both local/global endpoints; descriptive development data."""
import hashlib
import json
import os
from pathlib import Path

S=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(S/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=S/'track_h/attempt001/closeout001/native_pulse_effects.json';data=json.loads(p.read_text())
out=S/'figures001';out.mkdir(exist_ok=False)
states=[x for x in data['coverage'] if x['bank_qualified']]
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
styles=[('current_error','tab:blue','D',-.24),('support_sham','tab:green','s',-.08),('direction_high','tab:orange','^',.08),('direction_low','tab:red','v',.24)]
for ax,metric in zip(axes,['remaining_calls','N']):
    refs={}
    for x in states:
        key=(x['id'],x['seed']);v=next(y['A'] for y in data['consumer_controls'] if (y['id'],y['seed'])==key)
        refs[key]=v[metric]
    ax.bar(range(len(states)),[refs[(x['id'],x['seed'])] for x in states],width=.7,color='.8',label='Common all-kept A')
    for arm,color,marker,offset in styles:
        vals={(x['id'],x['seed']):x[metric] for x in data['effects'] if x['arm']==arm}
        ax.scatter([i+offset for i in range(len(states))],[vals[(x['id'],x['seed'])] for x in states],label=arm,color=color,marker=marker,zorder=3,s=50)
    ax.set_xticks(range(len(states)),[x['id'].split('/')[-1][:8]+' / seed'+str(x['seed']) for x in states],rotation=15,ha='right')
    ax.set_ylabel('Remaining calls to the same canvas commit' if metric=='remaining_calls' else 'Total complete-request denoising calls N')
axes[0].set_title('Local headroom; no directional lead over current-error')
axes[1].set_title('Earlier canvas stopping can increase total work')
handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
fig.suptitle('Development only: 3/8 identifying cells; every deleting request is wrong and output-capped')
fig.savefig(out/'local_vs_total_calls.png',dpi=170);fig.savefig(out/'local_vs_total_calls.pdf');plt.close(fig)
with (out/'receipt.json').open('x') as f:json.dump(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    model_forwards=0,identified_cells=3,independent_development_questions=2,confirmation_banks=0,
    sham_alias_note='Sham and current-error are the same support at states1 and3, not independent controls.'),f,indent=2)
print('Exported local-versus-total calls figure with all matched controls')
