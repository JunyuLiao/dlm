"""Frozen observed-pool ranking analysis; CUDA disabled and no model calls."""
from pathlib import Path
from collections import defaultdict
import hashlib, json, itertools, math
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
E=REPO/'results/native_support_discovery_20261026/experiment/attempt006'
V=REPO/'results/native_support_discovery_review_20261026/review001'
INPUTS={}
def read(p):
    INPUTS[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def write(n,x):
    with (OUT/n).open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def sign(x):return int(x>0)-int(x<0)
def cluster(values):
    if not values:return None
    a=np.array(list(values.values()));rng=np.random.default_rng(2601026)
    boot=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)
    return dict(question_values=values,mean=float(a.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),question_clusters=len(a))
def matched(a,b):return abs(a['current_error']-b['current_error'])/max(a['current_error'],b['current_error'],1e-6)<=.05 and abs(a['deleted_mass']-b['deleted_mass'])<=.01
def main():
    torch.set_num_threads(1)
    perms=read(OUT/'direction_sham_permutations.json')
    plan=read(E/'branch_manifest.json');reported={r['identity']:r for r in read(E/'analysis001/branches.json')}
    pairs=[];bankrows=[];states=[]
    for cell in plan['cells']:
        cid=hashlib.sha256(cell['id'].encode()).hexdigest()[:12]+f"_seed{cell['seed']}"
        ref=read(E/f'runs/references/{cid}.json');groups=defaultdict(list)
        for j,b in enumerate(cell['branches']):
            if b['layer']!=29:continue
            raw=read(E/f'runs/treatments/{cid}/s{j:03d}.json')
            assert raw['branch_identity']==b['identity']
            groups[(b['bank_relative'],b['drop_count'])].append((b,raw))
        for (name,drop),branches in groups.items():
            site=read(E/name);bank=next(b for b in site['banks'] if b['drop_count']==drop and 'alias_of' not in b)
            p=E/bank['private_relative'];digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==bank['private_sha256'];INPUTS[str(p.relative_to(REPO))]=digest
            priv=torch.load(p,map_location='cpu',weights_only=False);rs=bank['candidates'];g=site['geometry']
            identity=f"{cell['id']}|{cell['seed']}|{g['canvas']}|{g['step']}|29"
            keys={};shams={}
            base=torch.cat([priv['base_margin'][0],priv['base_margin'][1]])
            stable=torch.cat([priv['protected'][0],priv['protected'][1]])
            ew=torch.cat([priv['entropy_weight'][0],priv['entropy_weight'][1]])
            for b,raw in branches:
                i=b['candidate_index'];r=rs[i];bias=priv['candidates'][i]['actual_bias'].float()
                keys[i]=dict(readout=(r['projected_stable_crossings'],-r['projected_entropy_weighted_gain'],r['current_error'],r['support_sha256']),CE=(r['current_error'],r['support_sha256']),support_sham=(hashlib.sha256(('support-sham-v1|'+identity+'|'+r['support_sha256']).encode()).hexdigest(),))
                for n,perm in enumerate(perms):
                    probe=torch.stack([priv['readout_probe'][u,perm[u]] for u in range(4)])
                    projected=(bias*probe).sum(-1);pos=torch.cat([projected[0]+projected[2],projected[1]+projected[3]])
                    keys[i][f'direction_sham{n:02d}']=(int(((base+pos<0)&stable).sum()),-float((pos*ew).sum()),r['current_error'],r['support_sha256'])
                def at(v,t):return next((x for x in v['native_trace'] if not x['commit'] and x['canvas']==g['canvas'] and x['native_step']==t),None)
                a=at(ref,g['step']);s=at(raw,g['step']);an=at(ref,g['step']+1);sn=at(raw,g['step']+1)
                states.append(dict(id=cell['id'],seed=cell['seed'],identity=b['identity'],score_gain=r['projected_entropy_weighted_gain'],score_crossings=r['projected_stable_crossings'],immediate_entropy_reduction=a['entropy_mean']-s['entropy_mean'],next_entropy_reduction=an['entropy_mean']-sn['entropy_mean'] if an and sn else None,gain=reported[b['identity']]['local_gain']))
            for (ba,ra),(bb,rb) in itertools.combinations(branches,2):
                i,j=ba['candidate_index'],bb['candidate_index'];a,b=rs[i],rs[j]
                if not matched(a,b):continue
                x,y=(priv['candidates'][t]['actual_bias'].double() for t in [i,j]);sep=float((x-y).norm())/max(float(x.norm()),float(y.norm()),1e-12)
                outcome=sign(reported[ba['identity']]['local_gain']-reported[bb['identity']]['local_gain'])
                utility={k:sign(int(keys[j][k]>keys[i][k])-int(keys[j][k]<keys[i][k]))*outcome for k in keys[i]}
                utility['direction_sham_mean']=float(np.mean([utility[f'direction_sham{n:02d}'] for n in range(16)]))
                pairs.append(dict(id=cell['id'],seed=cell['seed'],benchmark=cell['benchmark'],anchor=ba['stratum'],drop=drop,identities=[ba['identity'],bb['identity']],meaningful=a['bias_direction_available'] and b['bias_direction_available'] and sep>=.25,bias_separation=sep,reference_correct=ref['correct'],outcome_tie=outcome==0,utility=utility))
            bankrows.append(dict(id=cell['id'],seed=cell['seed'],site=name,drop=drop,observed_candidates=len(branches),operator_candidates=len(rs)))
    assert len(pairs)==131 and sum(p['meaningful'] for p in pairs)==93
    summaries=[]
    for view in ['all_matched','meaningful','correct_reference']:
        pp=[p for p in pairs if (view!='meaningful' or p['meaningful']) and (view!='correct_reference' or p['reference_correct'])]
        bycell=defaultdict(list)
        for p in pp:bycell[(p['id'],p['seed'],p['anchor'],p['drop'])].append(p)
        byseed=defaultdict(list)
        for (qid,seed,anchor,drop),ps in bycell.items():byseed[(qid,seed)].append({k:float(np.mean([p['utility'][k] for p in ps])) for k in ['readout','CE','support_sham','direction_sham_mean']})
        seeds=[dict(id=q,seed=s,**{k:float(np.mean([z[k] for z in zs])) for k in zs[0]}) for (q,s),zs in byseed.items()]
        qs=defaultdict(list)
        for s in seeds:qs[s['id']].append(s)
        advantages={control:{q:float(np.mean([z['readout']-z[control] for z in zs])) for q,zs in qs.items()} for control in ['CE','support_sham','direction_sham_mean']}
        both=[q for q,zs in qs.items() if len(zs)==2 and all(z['readout']>z['CE'] and z['readout']>z['direction_sham_mean'] for z in zs)]
        loo={k:{q:float(np.mean([x for other,x in v.items() if other!=q])) for q in v if len(v)>1} for k,v in advantages.items()}
        summaries.append(dict(view=view,pairs=len(pp),outcome_ties=sum(p['outcome_tie'] for p in pp),per_seed=seeds,advantages={k:cluster(v) for k,v in advantages.items()},both_seed_questions=both,leave_one_question_out=loo,screen=len(both)>=3 and len({q.split('/')[0] for q in both})>=2 and all(x>0 for k in ['CE','direction_sham_mean'] for x in loo[k].values())))
    from scipy.stats import spearmanr
    links=[]
    for q in sorted({s['id'] for s in states}):
        ss=[s for s in states if s['id']==q]
        for x,y in [('score_gain','immediate_entropy_reduction'),('score_gain','next_entropy_reduction'),('immediate_entropy_reduction','gain'),('next_entropy_reduction','gain')]:
            valid=[z for z in ss if z[x] is not None and z[y] is not None];rho=None
            if len({z[x] for z in valid})>1 and len({z[y] for z in valid})>1:rho=float(spearmanr([z[x] for z in valid],[z[y] for z in valid]).statistic)
            links.append(dict(id=q,predictor=x,outcome=y,n=len(valid),rho=rho,descriptive_selected_pool=True))
    write('matched_pairs.json',pairs);write('ranking_summary.json',summaries);write('response_links.json',links);write('banks.json',bankrows)
    write('input_hashes.json',INPUTS);write('final_complete.json',dict(passed=True,matched=131,meaningful=93,old_score_revalidated=all(x['screen'] for x in summaries),scope='conditional exposed observed-pool CPU analysis, not confirmation',native_forwards=0))
    print(json.dumps([dict(view=x['view'],pairs=x['pairs'],both_seed_questions=x['both_seed_questions'],screen=x['screen']) for x in summaries]))
if __name__=='__main__':main()
