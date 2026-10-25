"""Development headroom and predeclared rule choice; never read confirmation outcomes."""
import json
from pathlib import Path
import statistics
import hashlib


def resolve(folder,index,name):
    path=folder/f'cell{index:03d}_{name}/receipt.json'
    alias=folder/f'cell{index:03d}_{name}_alias.json'
    if not path.exists() and alias.exists():path=folder/json.loads(alias.read_text())['alias']/'receipt.json'
    return json.loads(path.read_text()) if path.exists() else None


def complete_endpoint(control,arms):
    return control['remaining_calls'] is not None and all(x is not None and x['remaining_calls'] is not None for x in arms.values())


def main():
    root=Path(__file__).resolve().parents[1];raw=json.loads((root/'manifest_identity.json').read_text())['identities']
    identities=[x for x in raw if x['split']=='development'];rows=[]
    assert json.loads((root/'executions/development/final_complete.json').read_text())['passed']
    for index,(identity,seed) in enumerate([(x,s) for x in identities for s in [42,43]]):
        ref=root/f'executions/references/cell{index:03d}_A_observed/receipt.json';a=json.loads(ref.read_text())
        arms={name:resolve(root/'executions/development',index,name) for name in ['current_error','support_sham','direction_high','direction_low']}
        rows.append(dict(id=identity['id'],seed=seed,control=a,arms=arms,completed=all(x is not None for x in arms.values()),
            available=complete_endpoint(a,arms)))
    options=[]
    for name in ['direction_high','direction_low']:
        available=[x for x in rows if x['available']]
        regressions=[dict(id=x['id'],seed=x['seed'],reference=b) for x in rows if x['completed'] for b,v in [('A',x['control']),('current_error',x['arms']['current_error']),('support_sham',x['arms']['support_sham'])] if v['correct'] and not x['arms'][name]['correct']]
        replicated=[]
        for identity in identities:
            pair=[x for x in available if x['id']==identity['id']]
            if len(pair)==2 and all(x['arms'][name]['remaining_calls']<=min(x['arms'][b]['remaining_calls'] for b in ['current_error','support_sham'])-1 for x in pair):replicated.append(identity['id'])
        mean=statistics.mean(x['arms'][name]['remaining_calls'] for x in available) if available else None
        options.append(dict(rule=name,available_cells=len(available),quality_regressions=regressions,replicated_development_questions=replicated,
            mean_remaining_calls=mean,eligible=bool(replicated) and not regressions))
    choices=[x for x in options if x['eligible']]
    selected=min(choices,key=lambda x:(x['mean_remaining_calls'],0 if x['rule']=='direction_high' else 1))['rule'] if choices else None
    result=dict(confirmation_authorized=selected is not None,selected_current_only_rule=selected,options=options,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Development may reveal hindsight headroom. Only a replicated current-information rule with no observed dense-correct regression advances; no confirmation data are read here.')
    with (root/'development_decision.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
