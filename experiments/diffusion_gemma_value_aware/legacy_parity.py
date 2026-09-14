"""Full-token parity audit against matching legacy AIME26 aggressive runs."""
import json
from .protocol import ROOT,AIME,prepare,sha
from .run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def audit(root=ROOT):
    setup=prepare(root);prior=json.loads((AIME/'policy.json').read_text());records=[]
    for path in sorted((root/'verified_policies'/'aime26').glob('blasst_aggressive_*.json')):
        policy=json.loads(path.read_text());target=str(policy['target'])
        expected=prior['targets'][target]['log_scales']
        points=[p for p in policy['trace'] if all(p['policy'][k].get('log_scale')==expected[k] for k in ('local','global'))]
        if not points:continue
        source=points[0]['source_condition']
        for row in setup['calibration']:
            if row['benchmark']!='aime26':continue
            new=shard_path(root,'calibration',source,row['id']);old=AIME/f'blasst_s{int(policy["target"]*100)}'/f'{row["source_id"]}.json'
            a=json.loads(new.read_text());b=json.loads(old.read_text())
            assert all(a[k]==b[k]==row[k] for k in ('id','prompt_hash','seed','generation_budget'))
            records.append(dict(id=row['id'],target=policy['target'],exact_tokens=a['completion_tokens']==b['completion_tokens'],
                new=str(new),new_sha256=sha(new.read_bytes()),legacy=str(old),legacy_sha256=sha(old.read_bytes())))
    result=dict(passed=bool(records) and all(r['exact_tokens'] for r in records),comparisons=len(records),records=records,
        scope='Only completed calibration runs whose local/global log_scales exactly match the legacy policy; no heldout generations inspected')
    _write(root/'legacy_aggressive_parity.json',result)
    return result


if __name__=='__main__':
    result=audit();print({k:v for k,v in result.items() if k!='records'})
