"""Native stopping endpoints, official quality and explicit gated missingness."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import statistics
from decide import resolve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];out=root/'analysis001';out.mkdir(exist_ok=False)
    inputs={}
    def read(p):inputs[str(p.relative_to(root))]=sha(p);return json.loads(p.read_text())
    manifest=read(root/'manifest_identity.json')['identities'];protocol=read(root/'protocol.json')
    full_marker=read(root/'executions/full/final_complete.json');ref_marker=read(root/'executions/references/final_complete.json')
    assert full_marker['passed'] and ref_marker['passed']
    read(root/'executions/development/final_complete.json');decision=read(root/'development_decision.json')
    if decision['confirmation_authorized']:assert read(root/'executions/confirmation/final_complete.json')['passed']
    rows=[];coverage=[];quality_transitions=[];selected=[];eng=[];resources=[];eng_seen=set()
    all_cells=[(r,s) for split in ['development','confirmation'] for r in manifest if r['split']==split for s in [42,43]]
    for i,(identity,seed) in enumerate(all_cells):
        qualification=read(root/f'executions/references/cell{i:03d}_qualification.json');assert qualification['passed']
        refs={name:read(root/f'executions/references/cell{i:03d}_{name}/receipt.json') for name in ['piecewise','A_unobserved','A_observed','A_repeat']}
        refs['full']=read(root/f'executions/full/cell{i:03d}_full/receipt.json')
        control=refs['A_observed'];bank_path=root/f'executions/references/cell{i:03d}_A_observed/operators/selections.json'
        bank=read(bank_path) if bank_path.exists() else None
        group_index=i if identity['split']=='development' else i-4
        group=identity['split'];launched=group=='development' or decision['confirmation_authorized']
        arms={};aliases={}
        for name in ['current_error','support_sham','direction_high','direction_low','A_after']:
            folder=root/f'executions/{group}';path=folder/f'cell{group_index:03d}_{name}/receipt.json';alias=folder/f'cell{group_index:03d}_{name}_alias.json'
            if path.exists():arms[name]=read(path)
            elif alias.exists():
                link=read(alias);arms[name]=read(folder/link['alias']/'receipt.json');aliases[name]=link['alias']
        coverage.append(dict(id=identity['id'],seed=seed,split=identity['split'],anchor=control['anchor'],
            matched_directional_bank=bool(bank and bank['bank']['qualified']),directional_pair=bank['bank']['directional_contrast'] if bank else None,
            pulse_stage_launched=launched,available_arms=list(arms),
            target_canvas_endpoint_missing_arms=[name for name,v in arms.items() if v['remaining_calls'] is None],
            missing_reason=None if len(arms)==5 else 'development_gate_not_passed' if not launched else 'absent_anchor_or_matched_direction_pair'))
        def compact(v,name):
            calls=[];tail=0
            for x in v['native_trace']:
                if x['commit']:calls.append(tail);tail=0
                else:tail+=1
            assert len(calls)==v['C'] and sum(calls)+tail==v['N']
            assert all(x['LOCAL_skipped']==0 and x['protected_skipped']==0 and x['current_QKV'] for x in v['runtime']['physical_calls'])
            for check in v['runtime']['GQA_qualification']:
                assert check['error']<=.03
                unique=(identity['id'],seed,v['label'],check['head'],check['qb'])
                if unique not in eng_seen:eng.append(dict(id=identity['id'],seed=seed,arm=name,**check));eng_seen.add(unique)
            return dict(id=identity['id'],seed=seed,split=identity['split'],arm=name,
                **{k:v[k] for k in ['N','C','T','N_per_C','remaining_calls','correct','answer_parsed','no_final_response','finish_reason','anchor','model_forwards','W','S','S_per_N','timing_qualified']},
                physical_execution_alias=aliases.get(name),output_capped=v['T']>=16384,canvas48_reached=sum(x==48 for x in calls),native_canvas_calls=calls,uncommitted_tail=tail,
                physical_work=v['work'],operator_work=v['runtime']['diagnostic_work'],pulse_calls=v['runtime']['pulse_calls'])
        for name,v in (refs|arms).items():rows.append(compact(v,name))
        for name,v in arms.items():
            if name=='A_after':continue
            assert v['pre_intervention_native_state']==control['pre_intervention_native_state']
            for refname,ref in [('A',control),('FULL',refs['full']),('PIECEWISE',refs['piecewise']),('current_error',arms.get('current_error')),('support_sham',arms.get('support_sham'))]:
                if ref is not None and ref['correct']!=v['correct']:
                    quality_transitions.append(dict(id=identity['id'],seed=seed,split=identity['split'],arm=name,reference=refname,reference_correct=ref['correct'],method_correct=v['correct'],method_answer_valid=v['answer_parsed']))
        rule=decision['selected_current_only_rule']
        if identity['split']=='confirmation' and rule in arms:
            v=arms[rule];ce=arms['current_error'];sham=arms['support_sham']
            complete=all(x['remaining_calls'] is not None for x in [v,ce,sham])
            selected.append(dict(id=identity['id'],seed=seed,rule=rule,remaining=v['remaining_calls'],current_error_remaining=ce['remaining_calls'],sham_remaining=sham['remaining_calls'],
                endpoint_complete=complete,gain_vs_current_error=ce['remaining_calls']-v['remaining_calls'] if complete else None,gain_vs_sham=sham['remaining_calls']-v['remaining_calls'] if complete else None,
                reference_correct=control['correct'],current_error_correct=ce['correct'],sham_correct=sham['correct'],method_correct=v['correct'],quality_regression=(control['correct'] or ce['correct'] or sham['correct']) and not v['correct']))
    for job in ['full','references','development','confirmation']:
        p=root/f'executions/{job}/final_complete.json'
        if p.exists():resources.append(read(p))
    confirmation_pass=len(selected)==4 and all(x['endpoint_complete'] and x['gain_vs_current_error']>=1 and x['gain_vs_sham']>=1 and not x['quality_regression'] for x in selected)
    observed_native_headroom=[]
    for c in coverage:
        group=[x for x in rows if (x['id'],x['seed'])==(c['id'],c['seed'])];a=next(x for x in group if x['arm']=='A_observed')
        effects=[dict(arm=x['arm'],remaining_gain=a['remaining_calls']-x['remaining_calls'] if a['remaining_calls'] is not None and x['remaining_calls'] is not None else None,
            target_canvas_endpoint_complete=a['remaining_calls'] is not None and x['remaining_calls'] is not None,correct=x['correct']) for x in group if x['arm'] in ['direction_high','direction_low']]
        observed_native_headroom.append(dict(id=c['id'],seed=c['seed'],split=c['split'],effects=effects,
            interpretation='Best future arm is oracle headroom only; no online rule is selected from this cell.'))
    result=dict(passed=True,scope='Gated later-canvas single-pulse native stopping-headroom diagnostic, not Response replication or Stage B/C',
        protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(Path(__file__)),coverage=coverage,rows=rows,GQA_checks=eng,
        development_decision=decision,confirmation_selected_rule_results=selected,confirmation_gate_passed=confirmation_pass,
        quality_transitions=quality_transitions,oracle_headroom=observed_native_headroom,resources=resources,
        no_quality_noninferiority_claim=True,clean_latency_qualified=False,AA_SA_AS_SS_not_run=True,
        conclusion='Replicated current-rule stopping headroom observed; independent review required' if confirmation_pass else 'No qualified confirmed stopping selector; inspect absent contrasts, native effects, and quality limitations separately')
    for name,v in [('summary.json',result),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    with (root/'readout_frozen.json').open('x') as f:json.dump(dict(passed=True,summary_sha256=sha(out/'summary.json'),protocol_sha256=sha(root/'protocol.json'),confirmation_gate_passed=confirmation_pass,unique_questions=4,seed_cells=8),f,indent=2)
    print(json.dumps(dict(questions=4,anchor_cells=sum(x['anchor'] is not None for x in coverage),matched_cells=sum(x['matched_directional_bank'] for x in coverage),confirmation_pass=confirmation_pass,regressions=len(quality_transitions))))


if __name__=='__main__':main()
