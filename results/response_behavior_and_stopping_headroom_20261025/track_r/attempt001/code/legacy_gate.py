"""No new confirmation generation unless all exact legacy bindings reproduce."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];repo=root.parents[3]
    protocol=json.loads((root/'protocol.json').read_text())
    old=repo/'results/denoising_e2e_research_20261024/attempt003'
    original=json.loads((old/'e2e_protocol.json').read_text())
    old_scores={(x['arm'],x['id'],x['seed']):x for x in json.loads((old/'scoring001/official_scores.json').read_text())['outcomes']}
    inputs={};comparisons=[]
    def read(p):
        inputs[str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    for arm in protocol['legacy_arms']:
        leaf=root/'runs/legacy/all'/arm;marker=read(leaf/'final_complete.json');assert marker['passed']
        scores={(x['id'],x['seed']):x for x in read(leaf/'official_scores.json')['outcomes']}
        for index,identity in enumerate(protocol['legacy_reproduction']):
            original_index=next(i for i,x in enumerate(original['panel']['short']) if x==identity)
            a=read(leaf/f'audit{index:03d}.json');c=read(leaf/f'clean{index:03d}.json')
            oa=read(old/f'runs/panel/short/{arm}/audit{original_index:03d}.json')
            oc=read(old/f'runs/panel/short/{arm}/clean{original_index:03d}.json')
            checks={k:a[k]==c[k]==oa[k]==oc[k] for k in ['id','seed','output_sha256','N','C','T']}
            checks['official_score']=scores[identity['id'],identity['seed']]['correct']==old_scores[arm,identity['id'],identity['seed']]['correct']
            checks['audit_clean_RNG']=all(a[k]==c[k] for k in ['final_cpu_rng_sha256','final_cuda_rng_sha256'])
            checks['physical_GLOBAL']=a['work']['global_eligible_tiles']==oa['work']['global_eligible_tiles'] and a['work']['global_kept_tiles']==oa['work']['global_kept_tiles']
            checks['physical_LOCAL']=a['work']['native_local_eligible_rectangles']==oa['work']['native_local_eligible_rectangles'] and a['work']['native_local_skipped_tiles']==0
            if a['selector']:
                checks['support_and_lifecycle']=all(a['selector'][k]==oa['selector'][k] for k in ['physical_calls','selector_refresh_calls','held_mask_calls','carried_mask_calls'])
            checks['execution_mode']=a['work']['decode_cudagraph_modes']==oa['work']['decode_cudagraph_modes']
            comparisons.append(dict(arm=arm,**identity,checks=checks,passed=all(checks.values())))
    for index in range(len(protocol['legacy_reproduction'])):
        full=read(root/f'runs/legacy/all/dense_full_fix51994/audit{index:03d}.json')
        pw=read(root/f'runs/legacy/all/dense_piecewise/audit{index:03d}.json')
        assert all(full[k]==pw[k] for k in ['output_sha256','N','C','T']),'FULL/PIECEWISE disagreement'
    result=dict(passed=all(x['passed'] for x in comparisons),comparisons=comparisons,input_hashes=inputs,
        interpretation='Exact existing behavior reproduction; not an independent task sample or new selector.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root/'legacy_reproduction_receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(passed=result['passed'],comparisons=len(comparisons),failed=[x for x in comparisons if not x['passed']])))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
