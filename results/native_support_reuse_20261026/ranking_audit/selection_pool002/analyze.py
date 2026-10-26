from pathlib import Path
from collections import defaultdict
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;REPO=R.parents[3];E=REPO/'results/native_support_discovery_20261026/experiment/attempt006';S=R.parent.parent;inputs={}
def read(p):inputs[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def write(n,v):
    with (R/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def sign(x):return int(x>0)-int(x<0)
plan=read(E/'branch_manifest.json');lookup={b['identity']:b for c in plan['cells'] for b in c['branches']};reported={b['identity']:b for b in read(E/'analysis001/branches.json')};pairs=read(S/'ranking_audit/attempt001/matched_pairs.json');corrected=[]
for p in pairs:
    a,b=[lookup[i] for i in p['identities']];site=read(E/a['bank_relative']);bank=next(z for z in site['banks'] if z['drop_count']==a['drop_count'] and 'alias_of' not in z);ce=bank['selections']['current_error'];ref=bank['candidates'][ce]
    def eligible(i):
        r=bank['candidates'][i];return abs(r['current_error']-ref['current_error'])/max(r['current_error'],ref['current_error'],1e-6)<=.05 and abs(r['deleted_mass']-ref['deleted_mass'])<=.01
    ia,ib=a['candidate_index'],b['candidate_index'];inside=eligible(ia) and eligible(ib);utility=dict(p['utility'])
    if inside and (ia==ce or ib==ce):
        outcome=sign(reported[a['identity']]['local_gain']-reported[b['identity']]['local_gain']);utility['support_sham']=(-1 if ia==ce else 1)*outcome
    corrected.append(dict(p,both_original_pool=inside,faithful_original_utility=utility if inside else None))
summaries=[]
for view in ['original_pool','original_pool_meaningful','original_pool_correct_reference']:
    ps=[p for p in corrected if p['both_original_pool'] and ('meaningful' not in view or p['meaningful']) and ('correct_reference' not in view or p['reference_correct'])];state=defaultdict(list)
    for p in ps:state[p['id'],p['seed'],p['anchor'],p['drop']].append(p)
    byseed=defaultdict(list)
    for (q,s,a,d),vs in state.items():byseed[q,s].append({k:float(np.mean([p['faithful_original_utility']['readout']-p['faithful_original_utility'][k] for p in vs])) for k in ['CE','support_sham','direction_sham_mean']})
    seedrows=[dict(id=q,seed=s,**{k:float(np.mean([v[k] for v in vs])) for k in vs[0]}) for (q,s),vs in byseed.items()];byq=defaultdict(list)
    for x in seedrows:byq[x['id']].append(x)
    qrows={q:{k:float(np.mean([v[k] for v in vs])) for k in ['CE','support_sham','direction_sham_mean']} for q,vs in byq.items()};both=[q for q,vs in byq.items() if len(vs)==2 and all(v['CE']>0 and v['direction_sham_mean']>0 for v in vs)]
    means={k:float(np.mean([v[k] for v in qrows.values()])) for k in ['CE','support_sham','direction_sham_mean']};loo={k:{q:float(np.mean([v[k] for other,v in qrows.items() if other!=q])) for q in qrows if len(qrows)>1} for k in means};intervals={}
    for k in means:
        a=np.array([v[k] for v in qrows.values()]);rng=np.random.default_rng(2601026);boot=a[rng.integers(0,len(a),(20000,len(a)))].mean(1);intervals[k]=np.quantile(boot,[.025,.975]).tolist()
    summaries.append(dict(view=view,pairs=len(ps),per_seed=seedrows,per_question=qrows,mean_advantage=means,question_cluster_CI95=intervals,both_seed_positive_vs_CE_direction=both,LOO=loo,screen=len(both)>=3 and len({q.split('/')[0] for q in both})>=2 and all(v>0 for k in ['CE','direction_sham_mean'] for v in loo[k].values())))
write('pairs.json',corrected);write('summary.json',summaries);write('input_hashes.json',inputs);write('final_complete.json',dict(passed=True,all_pairs=len(pairs),primary_original_pool_pairs=sum(x['both_original_pool'] for x in corrected),old_score_validated=all(x['screen'] for x in summaries),native_forwards=0));print(json.dumps([dict(view=x['view'],pairs=x['pairs'],mean_advantage=x['mean_advantage'],screen=x['screen']) for x in summaries]))
