"""Operator-only identification coverage; does not read sparse future outcomes."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
R=Path(__file__).resolve().parent;E=R.parent/'experiment/attempt006';P=R.parent/'profiling/attempt002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    nr=load(E/'numerical_qualification.json');assert nr['passed'];numerical={(r['site'],r['drop_count']):r for r in nr['results']};inputs={str(p.relative_to(R.parent)):sha(p) for p in [E/'numerical_qualification.json',E/'branch_manifest.json',P/'manifest_identity.json']};rows=[];summary=[];coverage=defaultdict(list)
    for p in (E/'runs/references').glob('*seed*.json'):
        ref=load(p);inputs[str(p.relative_to(R.parent))]=sha(p);sites={}
        for name in ref['sites']:
            path=E/name;s=load(path);inputs[str(path.relative_to(R.parent))]=sha(path)
            for stratum in s['strata']:sites[stratum]=(name,s)
        for stratum in ['confidence','instability']:
            for fraction in [.05,.1]:
                row=dict(id=ref['id'],seed=ref['seed'],benchmark=ref['benchmark'],anchor=stratum,fraction=fraction,available=False,identifying_readout_vs_both=False)
                if stratum in sites:
                    name,site=sites[stratum];bank=next(b for b in site['banks'] if b['fraction']==fraction)
                    if 'alias_of' in bank:bank=site['banks'][bank['alias_of']]
                    if bank['screen']:
                        n=numerical[(name,bank['drop_count'])];pair={(min(x['i'],x['j']),max(x['i'],x['j'])):x for x in n['error_mass_matched_pairs']};ce=bank['selections']['current_error'];sh=bank['selections']['support_sham'];rd=bank['selections']['readout_margin'];matches=[pair.get((min(rd,i),max(rd,i)),{}) for i in [ce,sh]]
                        row.update(available=True,actual_drop_per_unit=bank['drop_count'],eligible_prefix_blocks=bank['blocks'],feasible_candidates=sum(x['feasible'] for x in bank['candidates']),executed_screen=len(bank['screen']),all_matched_pairs=len(pair),screen_matched_pairs=sum(x['selected_both'] for x in pair.values()),all_meaningful_pairs=sum(x['meaningful'] for x in pair.values()),screen_meaningful_pairs=sum(x['meaningful'] and x['selected_both'] for x in pair.values()),CE_readout_alias=ce==rd,sham_readout_alias=sh==rd,identifying_readout_vs_both=all(x.get('meaningful',False) for x in matches))
                    else:row.update(reason=bank['reason'],actual_drop_per_unit=bank['drop_count'])
                else:row['reason']='no_prespecified_anchor'
                rows.append(row);coverage[(stratum,fraction)].append(row)
    for (anchor,fraction),rs in coverage.items():
        byq=defaultdict(list)
        for r in rs:byq[r['id']].append(r)
        ids=[q for q,v in byq.items() if len(v)==2 and all(x['identifying_readout_vs_both'] for x in v)];tasks={r['benchmark'] for r in rs if r['id'] in ids}
        summary.append(dict(anchor=anchor,fraction=fraction,policy_cells=len(rs),available_cells=sum(r['available'] for r in rs),identifying_cells=sum(r['identifying_readout_vs_both'] for r in rs),both_seed_identifying_questions=ids,task_count=len(tasks),prospective_coverage_gate_possible=len(ids)>=4 and len(tasks)>=2,by_task=[dict(benchmark=task,policy_cells=sum(r['benchmark']==task for r in rs),available_cells=sum(r['benchmark']==task and r['available'] for r in rs),identifying_cells=sum(r['benchmark']==task and r['identifying_readout_vs_both'] for r in rs)) for task in ['aime26','humaneval','longbench_v2']]))
    plan=load(E/'branch_manifest.json');save(R/'cells.json',rows);save(R/'summary.json',dict(passed=True,operator_only=True,sparse_future_outcomes_read=False,configurations=summary,unique_native_branches=plan['unique_branches'],early_exclusions=plan['excluded'],all_pair_counts_not_independent_task_samples=True));save(R/'input_hashes.json',inputs);save(R/'final_complete.json',dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(E/'protocol.json'),summary_sha256=sha(R/'summary.json')))
if __name__=='__main__':main()
