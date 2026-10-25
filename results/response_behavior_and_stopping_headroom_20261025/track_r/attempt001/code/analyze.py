"""Frozen full-panel question-clustered quality/work readout."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];out=root/'analysis001';out.mkdir(exist_ok=False)
    inputs={}
    def read(p):inputs[str(p.relative_to(root))]=sha(p);return json.loads(p.read_text())
    protocol=read(root/'protocol.json');manifest=read(root/'manifest_identity.json');assert read(root/'legacy_reproduction_receipt.json')['passed']
    identities={x['id']:x for x in manifest['identities']};byarm=defaultdict(dict);rows=[]
    for arm in protocol['arms']:
        for task in ['aime26','humaneval']:
            leaf=root/'runs/panel'/task/arm;marker=read(leaf/'final_complete.json');assert marker['passed'] and marker['cells']==len(protocol['panel'][task])
            scores={(x['id'],x['seed']):x for x in read(leaf/'official_scores.json')['outcomes']}
            for i,identity in enumerate(protocol['panel'][task]):
                audit=read(leaf/f'audit{i:03d}.json');clean=read(leaf/f'clean{i:03d}.json');receipt=read(leaf/f'cell{i:03d}_receipt.json')
                assert receipt['passed'] and all(audit[k]==clean[k] for k in ['id','seed','N','C','T','output_sha256','final_cpu_rng_sha256','final_cuda_rng_sha256'])
                assert (clean['id'],clean['seed'])==(identity['id'],identity['seed'])
                key=identity['id'],identity['seed'];assert key not in byarm[arm]
                score=scores[key];work=audit['work'];selector=audit['selector'];ge=work['global_eligible_tiles'];gk=work['global_kept_tiles'];le=work['native_local_eligible_rectangles'];ls=work['native_local_skipped_tiles']
                counts=[];tail=0
                for event in audit['native_trace']:
                    if event['commit']:counts.append(tail);tail=0
                    else:tail+=1
                assert len(counts)==clean['C'] and sum(counts)+tail==clean['N'] and ls==0
                row={k:clean[k] for k in ['id','seed','arm','benchmark','N','C','T','N_per_C','W','S','S_per_N','finish_reason','timing_valid','output_sha256','model_forwards']}
                row.update(score,stratum=identities[key[0]]['stratum'],A1_exposed=identities[key[0]]['A1_exposed'],
                    native_canvas_calls=counts,uncommitted_tail_calls=tail,canvas48_reached=sum(x==48 for x in counts),output_cap=clean['T']>=8192,
                    GLOBAL=dict(eligible=ge,kept=gk,skipped=ge-gk),LOCAL=dict(eligible=le,kept=le-ls,skipped=ls),
                    overall=dict(eligible=ge+le,kept=gk+le-ls,skipped=ge-gk+ls),selector=({k:v for k,v in selector.items() if k not in ['first_two_candidate_banks','physical_calls']} if selector else None),
                    audit_path=str((leaf/f'audit{i:03d}.json').relative_to(root)),clean_path=str((leaf/f'clean{i:03d}.json').relative_to(root)))
                byarm[arm][key]=row;rows.append(row)
    assert len(rows)==700 and all(len(x)==140 for x in byarm.values())
    def interval(values,label,ratio=False,primary=False):
        # Values are seed-cell deltas/log-ratios. Bootstrap QUESTIONS, not seeds.
        groups=defaultdict(list)
        for identity,value in values:groups[identity].append(value)
        a=np.array([statistics.mean(v) for _,v in sorted(groups.items())])
        seed=20261025+int(hashlib.sha256(label.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed);index=rng.integers(0,len(a),(50000,len(a)));draw=a[index].mean(1)
        transform=np.exp if ratio else lambda x:x
        return dict(estimate=float(transform(a.mean())),ci95=transform(np.quantile(draw,[.025,.975])).tolist(),
            simultaneous_ci99375=transform(np.quantile(draw,[.003125,.996875])).tolist() if primary else None,
            question_clusters=len(a),seed_cells=sum(map(len,groups.values())),bootstrap_draws=50000)
    comparisons=[];aggregates=[];transitions=[];content=[]
    references=[x for x in protocol['arms'] if x!='response_guard_qdirection'];response=byarm['response_guard_qdirection']
    for task in ['aime26','humaneval']:
        for stratum in ['all','legacy_response','new_to_response','new_to_response_without_A1']:
            keys=[k for k,x in response.items() if x['benchmark']==task and (stratum=='all' or x['stratum']==stratum or stratum=='new_to_response_without_A1' and x['stratum']=='new_to_response' and not x['A1_exposed'])]
            assert keys
            for arm,values in byarm.items():
                selected=[values[k] for k in keys]
                aggregates.append(dict(benchmark=task,stratum=stratum,arm=arm,questions=len({x['id'] for x in selected}),cells=len(selected),correct=sum(x['correct'] for x in selected),
                    totals={k:sum(x[k] for x in selected) for k in ['N','C','T','W','S']},means={k:statistics.mean(x[k] for x in selected) for k in ['N','C','T','N_per_C','W','S','S_per_N']},
                    output_caps=sum(x['output_cap'] for x in selected),canvas48_reached=sum(x['canvas48_reached'] for x in selected),no_final=sum(x['no_final_response'] for x in selected),
                    extraction_failures=sum(not x['quality_detail']['parsed'] for x in selected),clean_timing_valid_cells=sum(x['timing_valid'] for x in selected)))
            for reference in references:
                paired=[(response[k],byarm[reference][k]) for k in keys]
                primary=stratum=='new_to_response';quality_primary=primary and reference in ['current_error_fixed4','sham_fixed4'];work_primary=primary and reference in ['dense_full_fix51994','allkept_layer29']
                quality=interval([(a['id'],int(a['correct'])-int(b['correct'])) for a,b in paired],task+stratum+reference+'quality',primary=quality_primary)
                ratios={}
                for metric in ['N','C','T','N_per_C','W','S','S_per_N']:
                    usable=[(a,b) for a,b in paired if metric not in ['W','S','S_per_N'] or a['timing_valid'] and b['timing_valid']]
                    if usable:ratios[metric]=interval([(a['id'],math.log(a[metric]/b[metric])) for a,b in usable],task+stratum+reference+metric,ratio=True,primary=work_primary and metric=='N')
                comparisons.append(dict(benchmark=task,stratum=stratum,method='response_guard_qdirection',reference=reference,quality_difference=quality,ratios=ratios,
                    correct_to_wrong=sum(b['correct'] and not a['correct'] for a,b in paired),wrong_to_correct=sum(a['correct'] and not b['correct'] for a,b in paired)))
                if stratum=='all':
                    for a,b in paired:
                        if a['correct']!=b['correct']:
                            transitions.append(dict(id=a['id'],seed=a['seed'],benchmark=task,reference=reference,response_correct=a['correct'],reference_correct=b['correct'],
                                response_N=a['N'],reference_N=b['N'],response_C=a['C'],reference_C=b['C'],response_T=a['T'],reference_T=b['T'],
                                response_text_metrics=a['text_metrics'],reference_text_metrics=b['text_metrics'],response_quality_detail=a['quality_detail'],reference_quality_detail=b['quality_detail'],
                                interpretation='Private-output programmatic rubric inspection; no reasoning-token correctness label.'))
                    for subset in ['all','correct_in_both_post_treatment']:
                        selected=[(a,b) for a,b in paired if subset=='all' or a['correct'] and b['correct']]
                        if not selected:continue
                        props={}
                        for m in ['duplicate_line_fraction','repeated_8gram_fraction','revision_vocabulary_count','repeated_function_definitions','final_channel_character','first_closed_box_in_final_character']:
                            valid=[(a,b) for a,b in selected if a['text_metrics'][m] is not None and b['text_metrics'][m] is not None]
                            if valid:props[m]=dict(response_mean=statistics.mean(a['text_metrics'][m] for a,b in valid),reference_mean=statistics.mean(b['text_metrics'][m] for a,b in valid),cells=len(valid),questions=len({a['id'] for a,b in valid}))
                        content.append(dict(benchmark=task,reference=reference,subset=subset,properties=props,
                            note='Descriptive text properties and retrospective character checkpoints; not native termination or causal redundancy evidence.'))
    decisions=[]
    for task in ['aime26','humaneval']:
        primary=[x for x in comparisons if x['benchmark']==task and x['stratum']=='new_to_response']
        quality=all(x['quality_difference']['simultaneous_ci99375'][0]>0 for x in primary if x['reference'] in ['current_error_fixed4','sham_fixed4'])
        work=all(x['ratios']['N']['simultaneous_ci99375'][1]<1 for x in primary if x['reference'] in ['dense_full_fix51994','allkept_layer29'])
        observed_loss=[x['reference'] for x in primary if x['quality_difference']['estimate']<0]
        point_quality=any(x['quality_difference']['estimate']>0 for x in primary if x['reference'] in ['current_error_fixed4','sham_fixed4'])
        point_work=any(x['ratios']['N']['estimate']<1 for x in primary if x['reference'] in ['dense_full_fix51994','allkept_layer29'])
        decisions.append(dict(benchmark=task,clear_quality_lead=quality,clear_total_work_lead=work,observed_quality_losses=observed_loss,
            disposition='quality/work tradeoff' if observed_loss and (point_work or point_quality) else 'clear replicated quality/work benefit' if (quality or work) and not observed_loss else 'suggestive/uncertain benefit' if point_quality or point_work else 'earlier favorable direction not reproduced',
            accuracy_noninferiority_established=False,mechanism_established=False))
    result=dict(passed=True,scope='Complete unchanged Response short-task behavior panel, not Stage B/C',protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(Path(__file__)),
        rows=rows,aggregates=aggregates,comparisons=comparisons,correctness_transitions=transitions,content_descriptions=content,decisions=decisions,
        primary_comparison_family=8,interval_policy='question-cluster bootstrap, both seeds retained; simultaneous99.375% primary and descriptive95% secondary',
        no_new_LongBench_behavior_panel=True,quality_preservation_and_speed_require_separate_evidence=True)
    for name,value in [('summary.json',result),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
    with (root/'readout_frozen.json').open('x') as f:json.dump(dict(passed=True,complete=True,summary_sha256=sha(out/'summary.json'),source_sha256=sha(Path(__file__)),panel_cells=700,protocol_sha256=sha(root/'protocol.json')),f,indent=2)
    print(json.dumps(dict(cells=700,decisions=decisions,quality_regressions=len(transitions))))


if __name__=='__main__':main()
