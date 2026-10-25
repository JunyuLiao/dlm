"""Canonical, input-hashed CPU derivation; never edit the source screen."""
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matched(a, b, field='response'):
    error = abs(a['current_error']-b['current_error']) / max(a['current_error'], b['current_error'], 1e-6)
    mass = abs(a['deleted_mass']-b['deleted_mass'])
    separation = abs(a[field]-b[field]) / max(abs(a[field]), abs(b[field]), 1e-6)
    return error <= .05 and mass <= .01, separation >= .25


def main():
    root = Path(__file__).resolve().parents[1]
    prior = root.parents[1] / 'denoising_e2e_research_20261024/attempt003'
    out = root / 'cpu_addendum001'
    out.mkdir(exist_ok=False)
    inputs = {}
    def read(p):
        inputs[str(p.relative_to(prior))] = sha(p)
        return json.loads(p.read_text())
    protocol = read(prior/'e2e_protocol.json')
    assert sha(prior/'e2e_protocol.json') == '88e46d4088a094968ebbeaed05b0ce4b9cf6e824a643675eba257ac9920d35dd'
    assert read(prior/'final_complete.json')['passed']
    bindings = read(prior/'result_directory_bindings.json')
    score = read(prior/'scoring001/official_scores.json')
    expected = {(a, c['id'], c['seed']) for a in protocol['arms'] for suite in ['short','long'] for c in protocol['panel'][suite]}
    quality = {(r['arm'],r['id'],r['seed']):r for r in score['outcomes']}
    assert len(quality) == len(score['outcomes']) == 450 and set(quality) == expected
    work_rows, banks, selected, seen = [], [], [], set()
    for suite in ['short','long']:
        for arm in protocol['arms']:
            leaf = prior / bindings['result_paths'].get(suite,{}).get(arm, f'runs/panel/{suite}/{arm}')
            launch = prior / bindings['launch_paths'].get(suite,{}).get(arm, f'launches/{suite}/{arm}')
            marker = read(leaf/'final_complete.json')
            assert marker['passed'] and marker['cells'] == len(protocol['panel'][suite])
            assert read(launch/'exit_receipt.json')['returncode'] == 0
            lr = read(launch/'launch_receipt.json')
            binding_name = 'attempt005' if arm == 'v31_optimized' else 'attempt006' if suite == 'long' else None
            if binding_name:
                bp = prior.parent/binding_name/'execution_binding.json'
                assert lr['execution_binding_sha256'] == marker['execution_binding_sha256'] == sha(bp)
            for p in sorted(leaf.glob('clean???.json')):
                clean = read(p); audit = read(p.with_name(p.name.replace('clean','audit')))
                key = arm,clean['id'],clean['seed']
                assert key not in seen; seen.add(key)
                assert all(audit[k] == clean[k] for k in ['id','seed','output_sha256','N','C','T'])
                if binding_name: assert clean['execution_binding_sha256'] == audit['execution_binding_sha256'] == sha(bp)
                w = audit['work']; ge,gk,le = w['global_eligible_tiles'],w['global_kept_tiles'],w['native_local_eligible_rectangles']
                ls = w['native_local_skipped_tiles']; gs=ge-gk
                steps=[];tail=0
                for event in audit['native_trace']:
                    if event['commit']:steps.append(tail);tail=0
                    else:tail+=1
                assert len(steps)==clean['C'] and sum(steps)+tail==clean['N']
                sel=audit['selector']
                phase={'source':'adapter physical phases' if arm=='v31_optimized' else 'layer-specific selector calls' if sel else 'native dense',
                       'legacy_dense_phase_is_not_sparse_phase_accounting':bool(gs)}
                work_rows.append(dict(id=clean['id'],seed=clean['seed'],arm=arm,benchmark=clean['benchmark'],
                    GLOBAL=dict(eligible=ge,kept=gk,skipped=gs,sparsity=gs/ge),
                    LOCAL=dict(eligible=le,kept=le-ls,skipped=ls),
                    overall=dict(eligible=ge+le,kept=gk+le-ls,skipped=gs+ls,sparsity=(gs+ls)/(ge+le)),
                    stale_overall_sparsity=w['overall_rectangle_sparsity']!=(gs+ls)/(ge+le),phase_annotation=phase,
                    native_canvas_calls=steps,uncommitted_tail=tail,N=clean['N'],C=clean['C'],T=clean['T']))
                if sel:
                    for bi, bank in enumerate(sel['first_two_candidate_banks']):
                        for unit in bank['units']:
                            candidates=[x for x in unit['candidates'] if x['feasible']]
                            counts=Counter()
                            for a,b in itertools.combinations(candidates,2):
                                if len(a['tiles']) != len(b['tiles']):continue
                                ok,full=matched(a,b); _,res=matched(a,b,'residual_response')
                                counts['all_feasible_pairs']+=1
                                counts['error_mass_matched']+=ok
                                counts['full_response_separated']+=ok and full
                                counts['residual_response_separated']+=ok and res
                            banks.append(dict(id=clean['id'],seed=clean['seed'],arm=arm,bank_index=bi,
                                evidence='first_boundary' if bi==0 else 'second_saved_bank_only',prefix=bank['prefix'],step=bank['step'],
                                head=unit['head'],qb=unit['query_block'],feasible=len(candidates),**counts))
    assert seen==expected
    totals=defaultdict(Counter)
    for x in banks:
        totals[x['arm']+'|'+str(x['bank_index'])].update({k:x.get(k,0) for k in ['all_feasible_pairs','error_mass_matched','full_response_separated','residual_response_separated']})
    old_selected=read(prior/'support_analysis001/summary.json')
    prior_analysis=read(prior/'analysis001/summary.json')
    trajectory=read(prior/'paired_trajectory001/summary.json')
    comparisons=[]
    for task in ['aime26','humaneval','longbench_v2']:
        for reference in ['current_error_fixed4','sham_fixed4','dense_full_fix51994']:
            pairs=[]
            for key,r in quality.items():
                if key[0]!='response_guard_qdirection' or r['benchmark']!=task:continue
                b=quality[reference,key[1],key[2]]
                pairs.append(dict(id=key[1],seed=key[2],reference_correct=b['correct'],response_correct=r['correct'],
                    regression=b['correct'] and not r['correct'],improvement=r['correct'] and not b['correct']))
            comparisons.append(dict(task=task,reference=reference,transitions=pairs,
                official_paired=prior_analysis['paired'][task]['response_guard_qdirection__vs__'+reference],
                trajectory=next(x for x in trajectory['comparisons'] if x['task']==task and x['arm']=='response_guard_qdirection' and x['reference']==reference)))
    result=dict(passed=True,source_sha256=sha(__file__),work=work_rows,banks=banks,
        exhaustive_totals={k:dict(v) for k,v in totals.items()},selected_pairs=old_selected,
        quality_and_trajectory=comparisons,corrected_work_does_not_change_published_aggregates=True,
        scope='Only two saved banks, not every intervention. All pairs within each unit, never pooled as independent questions. Full response and residual response separated.')
    for name,data in [('summary.json',result),('input_hashes.json',inputs)]:
        (out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    (out/'final_complete.json').write_text(json.dumps(dict(passed=True,consumed_inputs=len(inputs),unique_cells=450,
        summary_sha256=sha(out/'summary.json'),inputs_sha256=sha(out/'input_hashes.json')),indent=2)+'\n')
    print(json.dumps(dict(cells=450,stale_work_receipts=sum(x['stale_overall_sparsity'] for x in work_rows),totals={k:dict(v) for k,v in totals.items()})))


if __name__=='__main__':main()
