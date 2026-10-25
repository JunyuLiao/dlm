"""Correct phase-denominator derivation from canonical prior execution leaves."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path


def main():
    study=Path(__file__).resolve().parent;repo=study.parents[1]
    prior=repo/'results/denoising_e2e_research_20261024/attempt003'
    out=study/'cpu_phase_addendum001';out.mkdir(exist_ok=False)
    source=json.loads((prior/'e2e_protocol.json').read_text());binding=json.loads((prior/'result_directory_bindings.json').read_text())
    rows=[];hashes={}
    for suite in ['short','long']:
        for arm in source['arms']:
            leaf=prior/binding['result_paths'].get(suite,{}).get(arm,f'runs/panel/{suite}/{arm}')
            marker=json.loads((leaf/'final_complete.json').read_text());assert marker['passed']
            for path in sorted(leaf.glob('audit???.json')):
                audit=json.loads(path.read_text());clean_path=path.with_name(path.name.replace('audit','clean'))
                clean=json.loads(clean_path.read_text());assert audit['output_sha256']==clean['output_sha256']
                for p in [path,clean_path]:hashes[str(p.relative_to(prior))]=hashlib.sha256(p.read_bytes()).hexdigest()
                w=audit['work'];phases=defaultdict(Counter)
                def add(name,e,k,n):phases[name].update(eligible=e,kept=k,skipped=e-k,calls=n)
                if 'adapter_receipt' in w:
                    for name,item in w['adapter_receipt']['value_phase_tiles'].items():add('v31_'+name,item['eligible_tiles'],item['kept_tiles'],item['calls'])
                elif audit['selector']:
                    s=audit['selector']
                    for call in s['physical_calls']:add('layer29_'+call['phase'],call['eligible'],call['eligible']-call['skipped'],1)
                    add('other_GLOBAL_dense',w['global_eligible_tiles']-s['target_eligible_tiles'],w['global_eligible_tiles']-s['target_eligible_tiles'],w['global_calls']-len(s['physical_calls']))
                else:add('native_GLOBAL_dense',w['global_eligible_tiles'],w['global_kept_tiles'],w['global_calls'])
                assert sum(x['eligible'] for x in phases.values())==w['global_eligible_tiles']
                assert sum(x['kept'] for x in phases.values())==w['global_kept_tiles']
                assert sum(x['skipped'] for x in phases.values())==w['global_eligible_tiles']-w['global_kept_tiles']
                rows.append(dict(id=audit['id'],seed=audit['seed'],arm=arm,suite=suite,
                    GLOBAL_phases={k:dict(v,sparsity=v['skipped']/v['eligible'] if v['eligible'] else 0) for k,v in phases.items()},
                    LOCAL=dict(eligible=w['native_local_eligible_rectangles'],skipped=w['native_local_skipped_tiles']),
                    warning='Raw value_phase_tiles legacy dense labels are superseded ONLY in this derived artifact; originals preserved.'))
    assert len(rows)==450
    result=dict(passed=True,cells=rows,denominator='Actual denoising Q128xKV64 GLOBAL/LOCAL rectangles; not prefill/commit or CUDA CTA counts',
        conclusions_changed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,data in [('corrected_phase_work.json',result),('input_hashes.json',hashes)]:
        (out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    (out/'final_complete.json').write_text(json.dumps(dict(passed=True,cells=450,hashed_audit_clean_inputs=len(hashes),
        summary_sha256=hashlib.sha256((out/'corrected_phase_work.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cells=450,hashed_inputs=len(hashes))))


if __name__=='__main__':main()
