"""Freeze initial support/schedule aliases without reading sparse outcomes."""
from pathlib import Path
import json,hashlib
from reuse_runtime import ROOT,save,sha
from reuse_selectors import SELECTORS
from worker import cid

def load(p):return json.loads(p.read_text())
def main():
    assert load(ROOT/'runs/references/final_complete.json')['passed']
    rows=[r for r in load(ROOT.parent/'private/manifest.json') if r['partition']=='development'];cells=[];inputs={}
    for row in rows:
        for seed in [42,43]:
            rp=ROOT/f'runs/references/{cid(row,seed)}.json';r=load(rp);inputs[str(rp.relative_to(ROOT))]=sha(rp);a=r['anchors'].get('instability');branches=[];seen={};labels=[]
            if r['initial_site']:
                site=load(ROOT/r['initial_site']);inputs[r['initial_site']]=sha(ROOT/r['initial_site']);banks={}
                for b in site['banks']:
                    if 'relative' in b:banks[b['drop_count']]=load(ROOT/b['relative']);inputs[b['relative']]=sha(ROOT/b['relative'])
                for fraction in [.05,.1]:
                    drop=max(1,int(fraction*(a['prefix']//64)));bank=banks[drop]
                    for selector in SELECTORS:
                        index=bank['all_selections'].get(selector)
                        for lifecycle in ['SA','AS','held2','refresh2','matched_change2','held4','refresh4']:
                            if index is None:labels.append(dict(fraction=fraction,selector=selector,lifecycle=lifecycle,branch=None,reason='no_initial_feasible_support'));continue
                            c=bank['candidates'][index];future=selector if lifecycle.startswith('refresh') else None
                            key=(drop,c['support_sha256'],lifecycle,future)
                            if key not in seen:
                                identity=hashlib.sha256(json.dumps([row['id'],seed,*key]).encode()).hexdigest()
                                b=dict(identity=identity,drop_count=drop,fraction=fraction,selector=selector,lifecycle=lifecycle,units=c['units'],support_sha256=c['support_sha256'],sparse_sha256=c['sparse_sha256'],geometry=site['geometry'],labels=[])
                                branches.append(b);seen[key]=b
                            b=seen[key];label=dict(fraction=fraction,selector=selector,lifecycle=lifecycle,branch=b['identity']);b['labels'].append(label);labels.append(label)
            else:
                labels=[dict(fraction=f,selector=s,lifecycle=l,branch=None,reason='no_first_instability_anchor') for f in [.05,.1] for s in SELECTORS for l in ['SA','AS','held2','refresh2','matched_change2','held4','refresh4']]
            cells.append(dict(id=row['id'],seed=seed,benchmark=row['benchmark'],anchor=a,branches=branches,labels=labels))
    save(ROOT/'branch_manifest.json',dict(frozen_before_sparse_outcomes=True,cells=cells,input_hashes=inputs,unique_branches=sum(len(c['branches']) for c in cells),scientific_labels=sum(len(c['labels']) for c in cells),protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':main()
