"""Same realized model-support sequences must reproduce native complete outcomes."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
S=Path(__file__).resolve().parent;R=S/'attempt002';OUT=S/'physical_alias_audit001';INPUTS={}
def read(p):INPUTS[str(p.relative_to(S))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def main():
    assert read(R/'runs/treatments/final_complete.json')['passed'];groups=defaultdict(list)
    for p in sorted((R/'branch_records').glob('*.json')):
        r=read(p);signature=tuple((e['step'],e['actual_support_sha256']) for e in r['events'] if e.get('intervention'));groups[r['id'],r['seed'],signature].append(r)
    rows=[];failures=[]
    for (qid,seed,sig),rs in groups.items():
        checks={k:len({json.dumps(r[k],sort_keys=True) for r in rs})==1 for k in ['output_sha256','N','C','T','correct','final_cpu_rng','final_cuda_rng','model_forwards']}
        row=dict(id=qid,seed=seed,realized_support_sequence_sha256=hashlib.sha256(json.dumps(sig).encode()).hexdigest(),branches=[r['branch_identity'] for r in rs],alias_count=len(rs),checks=checks)
        rows.append(row)
        if not all(checks.values()):failures.append(row)
    OUT.mkdir(exist_ok=False)
    for n,v in [('groups.json',rows),('input_hashes.json',INPUTS),('final_complete.json',dict(passed=not failures,unique_realized_policies=len(rows),repeated_physical_policy_groups=sum(r['alias_count']>1 for r in rows),duplicate_executed_branches=sum(r['alias_count']-1 for r in rows),failures=failures,scope='Realized treatment-sequence equality, not claimed global policy equivalence',native_forwards=0))]:
        with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    assert not failures,'Same realized native policies failed repetition'
    print(json.dumps(dict(passed=True,unique_realized=len(rows))))
if __name__=='__main__':main()
