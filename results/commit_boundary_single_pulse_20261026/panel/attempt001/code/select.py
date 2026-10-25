"""Baseline-only selection, exactly one finite roster; freeze before treatments."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)
def main():
 roster=json.loads((ROOT/'private/roster.json').read_text());public={r['id']:r for r in json.loads((ROOT/'roster_identity.json').read_text())['identities']};profiles={};inputs={}
 for arm in ['corrected_dense_full','matched_allkept_layer29']:
  leaf=ROOT/'runs/profile'/arm;assert json.loads((leaf/'final_complete.json').read_text())['passed']
  for p in sorted(leaf.glob('audit*.json')):
   x=json.loads(p.read_text());assert x['seed']==42;profiles[(x['id'],arm)]=x;inputs[str(p.relative_to(ROOT))]=sha(p)
 selected=[];identities=[];rankings=[]
 for task in ['aime26','humaneval','longbench_v2']:
  pool=[x for x in roster if x['benchmark']==task]
  def rank(r):
   a=profiles[(r['id'],'matched_allkept_layer29')];f=profiles[(r['id'],'corrected_dense_full')];headroom=a['remaining_calls'] or 0
   tier=0 if a['anchor'] and headroom>=3 and a['correct'] and f['correct'] else 1 if a['anchor'] and headroom>=3 and a['correct'] else 2 if a['anchor'] and headroom>=3 else 3 if a['anchor'] else 4
   return tier,-headroom,r['roster_key']
  targets=sorted(pool,key=rank)[:2];rest=sorted([x for x in pool if x not in targets],key=lambda x:x['roster_key'])[:2]
  for r in targets+rest:
   stratum='targeted' if r in targets else 'coverage';rr=dict(r,stratum=stratum,selection_tier=rank(r)[0] if stratum=='targeted' else None);selected.append(rr)
   identities.append(dict(public[r['id']],stratum=stratum,selection_tier=rr['selection_tier']))
  rankings.extend(dict(id=r['id'],task=task,tier=rank(r)[0],headroom=-rank(r)[1],A_correct=profiles[(r['id'],'matched_allkept_layer29')]['correct'],FULL_correct=profiles[(r['id'],'corrected_dense_full')]['correct'],selected=any(x['id']==r['id'] for x in targets+rest)) for r in sorted(pool,key=rank))
 assert len(selected)==12 and len({r['id'] for r in selected})==12
 save(ROOT/'private/manifest.json',selected);save(ROOT/'manifest_identity.json',dict(identities=identities,private_manifest_sha256=sha(ROOT/'private/manifest.json'),roster_sha256=sha(ROOT/'roster_identity.json'),baseline_input_hashes=inputs,rankings=rankings,no_sparse_outcomes_used=True))
 print(json.dumps(dict(selected=[x['id'] for x in identities],targeted_tiers=[(x['id'],x['selection_tier']) for x in identities if x['stratum']=='targeted'])))
if __name__=='__main__':main()
