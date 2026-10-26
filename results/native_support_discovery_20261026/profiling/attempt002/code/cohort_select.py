"""Baseline-only targeted/coverage selection, partitions fixed beforehand."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    roster=json.loads((R/'private/roster.json').read_text());public={x['id']:x for x in json.loads((R/'roster_identity.json').read_text())['identities']};profiles={};inputs={};arms=['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29']
    for arm in arms:
        leaf=R/'runs/profile'/arm;assert json.loads((leaf/'final_complete.json').read_text())['passed']
        for p in sorted(leaf.glob('audit*.json')):
            v=json.loads(p.read_text());key=(v['id'],v['seed'],arm);assert key not in profiles;profiles[key]=v;inputs[str(p.relative_to(R))]=sha(p)
    assert len(profiles)==216
    selected=[];identities=[];rankings=[]
    for task in ['aime26','humaneval','longbench_v2']:
        for part in ['development','confirmation']:
            pool=[r for r in roster if r['benchmark']==task and r['partition']==part]
            def rank(r):
                both=0;correct_A=0;reach=0;anyanchor=False;rooms=[]
                for seed in [42,43]:
                    a=profiles[(r['id'],seed,arms[2])];f=profiles[(r['id'],seed,arms[0])];room=max([z['optimistic_U'] for z in a['anchors'].values()] or [0]);available=bool(a['anchors']);anyanchor|=available
                    if available:rooms.append(room)
                    if room>=1:
                        reach+=1;correct_A+=int(a['correct']);both+=int(a['correct'] and f['correct'])
                tier=0 if both==2 else 1 if both else 2 if correct_A else 3 if reach else 4 if anyanchor else 5
                return tier,-both,-correct_A,-reach,-(min(rooms) if rooms else 0),r['roster_key']
            target=min(pool,key=rank);remaining=sorted([r for r in pool if r['id']!=target['id']],key=lambda r:r['roster_key'])
            if task=='longbench_v2':different=[r for r in remaining if r['domain']!=target['domain']];coverage=(different or remaining)[0]
            else:coverage=remaining[0]
            for row,stratum in [(target,'targeted'),(coverage,'coverage')]:
                rr=dict(row,stratum=stratum);selected.append(rr)
                identities.append(dict(public[row['id']],stratum=stratum,target_tier=rank(row)[0] if stratum=='targeted' else None,
                    anchor_records={str(seed):profiles[(row['id'],seed,arms[2])]['anchors'] for seed in [42,43]},
                    A_correct={str(seed):profiles[(row['id'],seed,arms[2])]['correct'] for seed in [42,43]},FULL_correct={str(seed):profiles[(row['id'],seed,arms[0])]['correct'] for seed in [42,43]}))
            rankings.extend(dict(id=row['id'],benchmark=task,partition=part,rank=list(rank(row)),selected=row['id'] in [target['id'],coverage['id']]) for row in sorted(pool,key=rank))
    assert len(selected)==12
    save(R/'private/manifest.json',selected);save(R/'manifest_identity.json',dict(identities=identities,rankings=rankings,private_manifest_sha256=sha(R/'private/manifest.json'),roster_sha256=sha(R/'roster_identity.json'),baseline_input_hashes=inputs,no_sparse_outcomes_used=True))
    print(json.dumps(dict(selected=[dict(id=x['id'],partition=x['partition'],stratum=x['stratum'],tier=x['target_tier']) for x in identities])))
if __name__=='__main__':main()
