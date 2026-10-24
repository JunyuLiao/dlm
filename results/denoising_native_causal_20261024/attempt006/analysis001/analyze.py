"""Sanitized descriptive readout; never consumes prompts or generations."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def corr(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2:return None
    r=float(spearmanr(x,y).statistic)
    return r if math.isfinite(r) else None

def main():
    protocol=read(ROOT/'protocol.json');records=[];items=[];inputs={}
    for group in ['smoke','short','long']:
        marker=ROOT/group/'final_complete.json'
        if not marker.exists():raise RuntimeError('missing complete '+group)
        complete=read(marker)
        if not complete['passed'] or complete['protocol_sha256']!=sha(ROOT/'protocol.json'):raise RuntimeError('bad identity')
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        for folder in sorted((ROOT/group).glob('item*')):
            item=read(folder/'receipt.json');controls=read(folder/'controls.json')
            if not controls['restore_native_equal']:raise RuntimeError('native restore failed')
            rows=[]
            for f in sorted(folder.glob('candidate*.json')):
                x=read(f);x.update(id=item['id'],group=group,heldout=item['id'] in protocol['heldout_ids'])
                x['deleted_blocks']=x['prefix_blocks']-x['kept_prefix_blocks']
                x['first_logit_motion']=x['native_vs_allkept'][0]['logit_l2']
                x['next_logit_motion']=x['native_vs_allkept'][1]['logit_l2']
                for call in x['execution']['physical']:
                    if call['eligible_tiles']!=call['kept_tiles']+call['skipped_tiles']:raise RuntimeError('physical sum')
                    if call['skipped_tiles']!=x['deleted_blocks']:raise RuntimeError('support did not execute')
                    if call['current_canvas_deleted_tiles'] or call['local_skipped_tiles']:raise RuntimeError('scope violation')
                for key in ['rng_cpu','rng_cuda']:
                    if x['execution'][key]!=controls['dense'][key]:raise RuntimeError('RNG order changed')
                if x['execution']['forwards']!=x['execution']['steps']:raise RuntimeError('hidden forward')
                if not x['deleted_blocks'] and any(v['logit_l2'] for v in x['native_vs_allkept']):raise RuntimeError('allkept duplicate mismatch')
                rows.append(x);records.append(x)
            selections=[]
            for k in sorted(set(x['kept_prefix_blocks'] for x in rows)):
                subset=[x for x in rows if x['kept_prefix_blocks']==k and x['deleted_blocks']>0]
                if not subset:continue
                feasible=[x for x in subset if x['feasible']]
                mass=next((x for x in subset if x['candidate']=='mass'),subset[0])
                pick=lambda key:min(feasible,key=lambda x:(key(x),x['support_sha256'])) if feasible else mass
                choices=dict(response=pick(lambda x:x['response']),current_error=pick(lambda x:x['current_output_error']),
                    margin_influence=pick(lambda x:abs(x['native_vs_allkept'][0]['margin_delta'])),
                    decision_influence=pick(lambda x:(x['native_vs_allkept'][0]['accepted_changed'],x['native_vs_allkept'][0]['top1_changed'],abs(x['native_vs_allkept'][0]['margin_delta']))),
                    sham=min(subset,key=lambda x:x['support_sha256']))
                selections.append(dict(kept=k,candidates=len(subset),feasible=len(feasible),fallback=not feasible,
                    selected={name:dict(candidate=x['candidate'],support_sha256=x['support_sha256'],next_logit_motion=x['next_logit_motion'],next_top1_changed=x['native_vs_allkept'][1]['top1_changed'],next_accepted_changed=x['native_vs_allkept'][1]['accepted_changed']) for name,x in choices.items()}))
            sparse=[x for x in rows if x['deleted_blocks']]
            item.update(group=group,heldout=item['id'] in protocol['heldout_ids'],sparse_candidates=len(sparse),
                correlations={key:corr([x[key] for x in sparse],[x['next_logit_motion'] for x in sparse]) for key in ['response','current_output_error','first_logit_motion']},
                allkept_vs_native=controls['allkept_vs_native'],selections=selections)
            items.append(item)
            for f in folder.glob('*.json'):inputs[str(f.relative_to(ROOT))]=sha(f)
    held=[x for x in records if x['heldout'] and x['deleted_blocks']]
    selections=[(i,s) for i in items if i['heldout'] for s in i['selections']]
    differentiated=[(i,s) for i,s in selections if s['selected']['response']['support_sha256']!=s['selected']['current_error']['support_sha256']]
    summary=dict(protocol_sha256=sha(ROOT/'protocol.json'),items=items,candidates=len(records),sparse_candidates=sum(bool(x['deleted_blocks']) for x in records),
        heldout_sparse_candidates=len(held),heldout_items_with_sparse_candidates=len(set(x['id'] for x in held)),
        heldout_response_error_distinct_selections=len(differentiated),
        zero_stop_changes=all(not v['native_stop_changed'] for x in records for v in x['native_vs_allkept']),
        guard_current_agreement_future_changes=[dict(id=x['id'],candidate=x['candidate'],current_output_error=x['current_output_error'],deleted_blocks=x['deleted_blocks'],next_top1=x['native_vs_allkept'][1]['top1_changed'],next_accepted=x['native_vs_allkept'][1]['accepted_changed']) for x in records if x['feasible'] and x['deleted_blocks'] and x['native_vs_allkept'][0]['top1_changed']==0 and x['native_vs_allkept'][0]['accepted_changed']==0 and x['native_vs_allkept'][1]['top1_changed']>0],
        scope='first-canvas layer29 head0 queryblock0, two calls, seed42; correlated descriptive states; no task-accuracy or trajectory claim')
    write(OUT/'summary.json',summary);write(OUT/'input_hashes.json',inputs)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    colors=dict(aime26='tab:orange',humaneval='tab:blue',longbench_v2='tab:green')
    for x in records:
        if not x['deleted_blocks']:continue
        color=colors[x['id'].split('/')[0]]
        for ax,key in zip(axs,['current_output_error','response']):
            ax.scatter(x[key],x['next_logit_motion'],c=color,marker='o' if x['feasible'] else 'x',alpha=.7)
    for ax,key in zip(axs,['current output error','Q/K/V response']):
        ax.set_xscale('symlog',linthresh=1e-5);ax.set_xlabel(key);ax.set_ylabel('Next-call logit L2 vs all-kept');ax.grid(alpha=.2)
    fig.suptitle('Native two-call diagnostics: orange AIME, blue HumanEval, green LongBench')
    fig.tight_layout();fig.savefig(OUT/'native_prediction.png',dpi=180);fig.savefig(OUT/'native_prediction.pdf');plt.close(fig)
    write(OUT/'analysis_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),summary_sha256=sha(OUT/'summary.json'),script_sha256=sha(Path(__file__)),scope='descriptive analysis only'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['items','guard_current_agreement_future_changes']},indent=2))
    for i in items:print(i['id'],i['correlations'],json.dumps(i['selections']))

if __name__=='__main__':main()
