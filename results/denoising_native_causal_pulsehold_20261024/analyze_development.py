"""Read frozen pulse/hold public receipts after selector-choice artifacts exist."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root.resolve();out=r/'analysis001'
    if (out/'analysis_complete.json').exists():raise RuntimeError('immutable analysis already exists')
    p=read(r/'protocol.json');m=read(r/'execution001/final_complete.json')
    if not m['passed'] or m['protocol_sha256']!=sha(r/'protocol.json'):raise RuntimeError('incomplete/mismatched execution')
    summaries=[];input_hashes={};all_branches=[]
    for item in sorted((r/'execution001').glob('item*')):
        landmarks=read(item/'reference_landmarks.json')
        for anchor in sorted(item.glob('anchor*')):
            receipt=read(anchor/'receipt.json');bank=read(anchor/'candidate_bank.json');choices=read(anchor/'selections.json');controls=read(anchor/'controls.json')
            by_hash={x['support_hash']:x for x in bank['candidates']}
            feasible=[x for x in bank['candidates'] if x['feasible']]
            branches=[read(f) for f in sorted(anchor.glob('branch*.json'))]
            aa=controls['allkept'];aa_n=aa['remaining_denoise'];rows=[]
            for x in branches:
                ex=x['execution'];n=ex['remaining_denoise'];first=ex['trace'][0] if ex['trace'] else None
                second=ex['trace'][1] if len(ex['trace'])>1 else None
                dose=sum(call['action']=='S' for call in ex['physical'])
                candidate=by_hash[x['support_hash']]
                expected_skip=candidate['physical_eligible']-candidate['physical_kept']
                if any(call['skipped']!=(expected_skip if call['action']=='S' else 0) for call in ex['physical']):raise RuntimeError('support work mismatch')
                if ex['forwards']!=ex['steps']:raise RuntimeError('extra forward')
                if not ex['committed']:raise RuntimeError('did not reach native commit')
                row=dict(id=receipt['id'],seed=receipt['seed'],anchor=receipt['anchor'],L=receipt['L'],
                    support_hash=x['support_hash'],schedule=x['schedule'],remaining_calls=n,AA_remaining_calls=aa_n,delta_calls=n-aa_n,
                    first_call_converged=bool(first['converged']) if first else None,
                    second_call_available=second is not None,second_entropy_mean=second['entropy_mean'] if second else None,
                    terminal_cap=bool(ex['trace'][-1]['cap_reached']),
                    terminal_confident=bool(ex['trace'][-1]['confidence']),
                    second_native_winner_mismatch=x['versus_AA']['events'][1]['actual_native_winner_mismatch'] if len(x['versus_AA']['events'])>1 else None,
                    second_canvas_mismatch=x['versus_AA']['events'][1]['canvas_mismatch'] if len(x['versus_AA']['events'])>1 else None,
                    second_self_condition_l2=x['versus_AA']['events'][1]['self_condition_l2'] if len(x['versus_AA']['events'])>1 else None,
                    realized_sparse_calls=dose,global_eligible=ex['global_eligible_tiles'],global_kept=ex['global_kept_tiles'],global_skipped=ex['global_skipped_tiles'],
                    local_eligible=ex['local_eligible_tiles'],local_skipped=ex['local_skipped_tiles'],
                    current_error=candidate['current_error'],response=candidate['response_score'],
                    scores_selecting=[name for name,h in choices['selections'].items() if h==x['support_hash']])
                rows.append(row);all_branches.append(row)
            contrasts=[]
            for support in sorted({x['support_hash'] for x in rows}):
                schedules={x['schedule']:x for x in rows if x['support_hash']==support}
                if set(schedules)=={'SA','AS','SS'}:
                    sa,ass,ss=[schedules[x]['remaining_calls'] for x in ['SA','AS','SS']]
                    contrasts.append(dict(support_hash=support,endpoint='remaining_first_canvas_calls',SA_minus_AA=sa-aa_n,AS_minus_AA=ass-aa_n,
                        SS_minus_SA=ss-sa,scalar_interaction=ss-sa-ass+aa_n,
                        interpretation='factorial contrast of scalar stopping outcome; not decomposition of logit norms'))
            comparisons=[]
            for name in ['response','decision','mass','sham']:
                for schedule in ['SA','AS','SS']:
                    current=next((x for x in rows if x['support_hash']==choices['selections'].get('current_error') and x['schedule']==schedule),None)
                    candidate=next((x for x in rows if x['support_hash']==choices['selections'].get(name) and x['schedule']==schedule),None)
                    if current and candidate:
                        comparisons.append(dict(score=name,schedule=schedule,distinct_support=candidate['support_hash']!=current['support_hash'],
                            calls_minus_current_error=candidate['remaining_calls']-current['remaining_calls']))
            # Short probes are stratified by deleted routing unit. Cross-unit matches remain allocation contrasts.
            def deleted_unit(candidate):return next((i for i,unit in enumerate(candidate['support']) if not all(unit)),None)
            pairs=bank['matched_pairs']
            same_unit=sum(x['matched'] and deleted_unit(by_hash[x['left']])==deleted_unit(by_hash[x['right']]) for x in pairs)
            row=dict(**receipt,AA_remaining_calls=aa_n,AA_cap=bool(aa['trace'][-1]['cap_reached']),
                candidate_count=len(bank['candidates']),feasible_count=len(feasible),
                matched_pairs_count=sum(x['matched'] for x in pairs),same_unit_matched_pairs=same_unit,
                response_identifying_pairs=sum(x['response_identifying_pair'] for x in pairs),
                selected_hashes=choices['selections'],response_distinct_current_error=choices['selections'].get('response')!=choices['selections'].get('current_error'),
                decision_distinct_current_error=choices['selections'].get('decision')!=choices['selections'].get('current_error'),
                branches=rows,contrasts=contrasts,score_comparisons=comparisons,
                restoration_passed=controls['native_first_canvas_parity'] and controls['allkept_first_canvas_parity'] and controls['measurement_parity'],
                full_request_qualified=controls['full_qualification'] is not None,
                GQA_qualification=controls['gqa_qualification'])
            summaries.append(row)
            for f in anchor.glob('*.json'):input_hashes[str(f.relative_to(r))]=sha(f)
        input_hashes[str((item/'reference_landmarks.json').relative_to(r))]=sha(item/'reference_landmarks.json')
    identifying=[x for x in summaries if x['feasible_count']>=4 and x['matched_pairs_count']>0 and (x['response_distinct_current_error'] or x['decision_distinct_current_error'])]
    summary=dict(protocol_sha256=sha(r/'protocol.json'),anchors=summaries,anchor_count=len(summaries),identifying_anchor_count=len(identifying),
        total_candidates=sum(x['candidate_count'] for x in summaries),total_feasible=sum(x['feasible_count'] for x in summaries),
        total_matched_pairs=sum(x['matched_pairs_count'] for x in summaries),same_unit_matched_pairs=sum(x['same_unit_matched_pairs'] for x in summaries),
        total_response_identifying_pairs=sum(x['response_identifying_pairs'] for x in summaries),
        distinct_response_selections=sum(x['response_distinct_current_error'] for x in summaries),
        distinct_decision_selections=sum(x['decision_distinct_current_error'] for x in summaries),
        branch_count=len(all_branches),fewer_first_canvas_calls=sum(x['delta_calls']<0 for x in all_branches),more_first_canvas_calls=sum(x['delta_calls']>0 for x in all_branches),
        development_gate_passed=len(identifying)>=2,method_holdout_complete=False,official_task_outcomes=None,method_promoted=False,
        exposure='previously dense/intervention-exposed development identities seed42; not confirmation',
        limits=['four short development anchors only','candidate outcomes single run per schedule; native/all-kept bookends repeated exactly','acceptance reconstructed, not native buffer','no task-quality or full sparse trajectory claim','long-context and confirmation deferred before generation'],
        resource=read(r/'exit_receipt.json'))
    write(out/'summary.json',summary);write(out/'input_hashes.json',input_hashes)
    # Scientific plots use public numeric receipts only.
    os.environ.setdefault('MPLCONFIGDIR',str(r/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    colors={'SA':'tab:blue','AS':'tab:orange','SS':'tab:green'}
    for i,anchor in enumerate(summaries):
        for branch in anchor['branches']:
            axs[0].scatter(i,branch['delta_calls'],color=colors[branch['schedule']],alpha=.7)
            if branch['second_native_winner_mismatch'] is not None:
                axs[1].scatter(branch['current_error'],branch['second_native_winner_mismatch'],color=colors[branch['schedule']],alpha=.7)
    axs[0].axhline(0,color='black',linewidth=.7);axs[0].set_xticks(range(len(summaries)),[x['id'].split('/')[-1]+' a'+str(x['anchor']) for x in summaries]);axs[0].set_ylabel('Remaining first-canvas calls minus AA')
    axs[1].set_xlabel('Current worst-row attention error');axs[1].set_ylabel('Second-call native winner mismatch /256');axs[1].set_xscale('symlog',linthresh=1e-6)
    fig.suptitle('Blue SA pulse; orange AS second pulse; green SS hold — development only');fig.tight_layout();fig.savefig(out/'pulse_hold.png',dpi=180);fig.savefig(out/'pulse_hold.pdf');plt.close(fig)
    write(out/'analysis_complete.json',dict(passed=True,summary_sha256=sha(out/'summary.json'),protocol_sha256=sha(r/'protocol.json'),analysis_source_sha256=sha(Path(__file__)),scope='descriptive bounded development readout only'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['anchors','resource']},indent=2))
if __name__=='__main__':main()
