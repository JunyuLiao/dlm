"""Derived H scope, native pulse effects and resource closeout; no new gates."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

STUDY=Path(__file__).resolve().parents[1];REPO=STUDY.parents[1]
H=STUDY/'track_h/attempt001'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)


def main():
    assert json.loads((H/'execution_complete.json').read_text())['passed']
    marker=json.loads((H/'readout_frozen.json').read_text())
    assert marker['summary_sha256']==sha(H/'analysis001/summary.json')
    out=H/'closeout001';out.mkdir(exist_ok=False);inputs={}
    def read(p):inputs[str(p.relative_to(H))]=sha(p);return json.loads(p.read_text())
    analysis=read(H/'analysis001/summary.json');protocol=read(H/'protocol.json');binding=read(H/'binding_receipt.json')
    resources=Counter();jobs=[];requests=[]
    for leaf in sorted((H/'launches').iterdir()):
        launch=read(leaf/'launch_receipt.json');exit=read(leaf/'exit_receipt.json');assert exit['returncode']==0
        jobs.append(dict(job=leaf.name,started_unix=launch['started_unix'],**exit))
        if exit['GPU']:resources['allocated_GPU_worker_seconds']+=exit['allocated_seconds'];resources['GPU_jobs']+=1
    for job in ['full','references','development','confirmation']:
        folder=H/'executions'/job;path=folder/'final_complete.json'
        if not path.exists():continue
        value=read(path);assert value['passed']
        records=[read(p) for p in sorted(folder.glob('*/receipt.json'))]
        assert sum(x['model_forwards'] for x in records)+value['initialization_engine_forwards']==value['model_executions']
        resources['engine_execute_forwards']+=value['model_executions'];resources['startup_model_forwards']+=value['startup_forwards']
        resources['initialization_engine_forwards']+=value['initialization_engine_forwards']
        resources['complete_native_requests']+=len(records);resources['warm_requests']+=sum(x['label']=='warm' for x in records)
        resources['initialized_engines']+=not value.get('no_model_initialized',False)
        for record in records:requests.append(dict(job=job,**{k:record[k] for k in ['label','id','seed','model_forwards','N','C','T','correct']}))
    resources['explicit_total_model_forwards']=resources['engine_execute_forwards']+resources['startup_model_forwards']
    save(out/'resources.json',dict(**resources,jobs=jobs,requests=requests,GPU_failures=0,
        no_resource_ceilings=True,allocation_note='Launch through observed child exit, including startup/warmup/cleanup;15s polling resolution. Audit time is not clean latency.'))
    controls={(x['id'],x['seed']):x for x in analysis['rows'] if x['arm']=='A_observed'}
    effects=[];coverage=[];consumer_controls=[]
    manifest=read(H/'manifest_identity.json')['identities'];cells=[(x,s) for split in ['development','confirmation'] for x in manifest if x['split']==split for s in [42,43]]
    for i,(identity,seed) in enumerate(cells):
        control=read(H/f'executions/references/cell{i:03d}_A_observed/receipt.json')
        full=read(H/f'executions/full/cell{i:03d}_full/receipt.json')
        piecewise=read(H/f'executions/references/cell{i:03d}_piecewise/receipt.json')
        consumer_controls.append(dict(id=identity['id'],seed=seed,split=identity['split'],
            full={k:full[k] for k in ['N','C','T','correct','answer_parsed','no_final_response','anchor','remaining_calls']},
            piecewise={k:piecewise[k] for k in ['N','C','T','correct','answer_parsed','no_final_response','anchor','remaining_calls']},
            A={k:control[k] for k in ['N','C','T','correct','answer_parsed','no_final_response','anchor','remaining_calls']},
            FULL_PIECEWISE_exact_output=full['output_sha256']==piecewise['output_sha256'],
            A_FULL_exact_output=full['output_sha256']==control['output_sha256']))
        bank_path=H/f'executions/references/cell{i:03d}_A_observed/operators/selections.json'
        bank=read(bank_path)['bank'] if bank_path.exists() else None
        details=[]
        if bank:
            for unit in bank['units']:
                reasons=Counter()
                for x in unit['candidates']:
                    if not x['feasible']:
                        reasons['error_and_mass' if x['current_error']>.1 and x['deleted_mass']>.2 else 'error' if x['current_error']>.1 else 'mass']+=1
                details.append(dict(head=unit['head'],qb=unit['qb'],prefix_blocks=unit['blocks'],delete=unit['drop_count'],candidates=len(unit['candidates']),feasible=unit['feasible'],
                    error_mass_matched_pairs=len(unit['matched_pairs']),nonzero_direction_pairs=sum(x['nonzero'] and x['cosine']<=.5 for x in unit['matched_pairs']),exclusions=dict(reasons)))
        coverage.append(dict(id=identity['id'],seed=seed,split=identity['split'],anchor=control['anchor'],remaining_calls=control['remaining_calls'],
            bank_qualified=bank['qualified'] if bank else False,missing_reason=bank['reason'] if bank else 'no_qualifying_anchor',units=details,
            directional_contrast=bank['directional_contrast'] if bank else None,
            distinct_selected_supports=len({x['support_sha256'] for x in bank['selections'].values()}) if bank else 0))
        reference={x['native_step']:x for x in control['native_trace'] if not x['commit'] and control['anchor'] is not None and x['canvas']==control['anchor']['canvas'] and x['native_step']>control['anchor']['after_completed_calls']}
        group=identity['split'];j=i if group=='development' else i-4
        for arm in ['current_error','support_sham','direction_high','direction_low']:
            path=H/f'executions/{group}/cell{j:03d}_{arm}/receipt.json'
            alias=H/f'executions/{group}/cell{j:03d}_{arm}_alias.json';alias_target=None
            if not path.exists() and alias.exists():alias_target=read(alias)['alias'];path=H/f'executions/{group}'/alias_target/'receipt.json'
            if not path.exists():continue
            value=read(path);aligned=[]
            for x in value['native_trace']:
                if x['commit'] or value['anchor'] is None or x['canvas']!=value['anchor']['canvas'] or x['native_step'] not in reference:continue
                ref=reference[x['native_step']];fp=x['native_state_fingerprint'];rfp=ref['native_state_fingerprint']
                aligned.append(dict(native_step=x['native_step'],entropy_delta=x['entropy_mean']-ref['entropy_mean'],entropy_slack=x['entropy_slack'],
                    confidence=x['confidence'],reference_confidence=ref['confidence'],stable=x['stable'],reference_stable=ref['stable'],
                    changed={k:fp[k]!=rfp[k] for k in ['logits','canvas','self_condition','history','native_winner','cpu_rng','cuda_rng']},
                    worst_margin_erosion=x.get('worst_margin_erosion'),reference_winner_crossings=x.get('reference_winner_crossings'),
                    signed_margin_change_quantiles=x.get('signed_margin_change_quantiles')))
            kept=value['runtime']['physical_calls'];deleted=sum(x['skipped'] for x in kept)
            assert sum(x['phase']=='one_pulse' for x in kept)==1 and value['runtime']['pulse_calls']==1
            assert all(x['LOCAL_skipped']==0 and x['protected_skipped']==0 for x in kept)
            effects.append(dict(id=identity['id'],seed=seed,split=group,arm=arm,physical_execution_alias=alias_target,
                N=value['N'],C=value['C'],T=value['T'],reference_N=control['N'],reference_C=control['C'],reference_T=control['T'],
                remaining_calls=value['remaining_calls'],reference_remaining_calls=control['remaining_calls'],
                remaining_gain=control['remaining_calls']-value['remaining_calls'] if control['remaining_calls'] is not None and value['remaining_calls'] is not None else None,
                correct=value['correct'],reference_correct=control['correct'],answer_parsed=value['answer_parsed'],no_final_response=value['no_final_response'],
                GLOBAL_deleted_rectangles=deleted,LOCAL_deleted_rectangles=0,protected_deleted_rectangles=0,
                aligned_native_calls=aligned,native_comparisons_are_not_intermediate_task_correctness=True))
    assert all(sha(REPO/p)==h for p,h in protocol['sources'].items())
    save(out/'source_verification.json',dict(passed=True,pinned_sources=len(protocol['sources']),source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(H/'protocol.json')))
    save(out/'native_pulse_effects.json',dict(coverage=coverage,effects=effects,consumer_controls=consumer_controls,
        scope='One pulse then common all-kept; noAA/SA/AS/SS factorial, no holds, no restoration',acceptance_exact_counts_qualified=False,
        raw_marker_annotation='final_complete.one_pulse=True names the study contract, not actual exposure. Actual runtime.pulse_calls is0 for controls and1 for a deleting branch.'))
    save(out/'input_hashes.json',inputs)
    lines=['# Later-canvas signed-bias stopping-headroom diagnostic','',
        'Separate from unchanged Response replication and the failed A1 attention-motion forecasts. Four LongBench question clusters, two paired seeds. This is a targeted diagnostic cohort, not representative LongBench accuracy. Source/protocol and scientific source-preparation seal are public. No prompt truncation, native sampler change, restoration or resource ceiling.','',
        '|Question|Seed|FULL correct; N/C/T|PIECEWISE correct; N/C/T|Common A correct; N/C/T|FULL/PW output equal|',
        '|---|---:|---|---|---|---|']
    for x in consumer_controls:
        def ref(name):
            v=x[name];return f"{v['correct']}; {v['N']}/{v['C']}/{v['T']}"
        lines.append(f"|{x['id']}|{x['seed']}|{ref('full')}|{ref('piecewise')}|{ref('A')}|{x['FULL_PIECEWISE_exact_output']}|")
    lines+=['','Consumer substitutions are systematic treatments, not noise to subtract. Native remaining-call effects use the qualified common A state; FULL/PW anchors are separate trajectories.','',
        '|Question|Seed|Split|Common anchor (zero-based canvas / completed calls)|A remaining|Directional bank|Distinct selected supports|',
        '|---|---:|---|---|---:|---|---:|']
    for x in coverage:lines.append(f"|{x['id']}|{x['seed']}|{x['split']}|{x['anchor']}|{x['remaining_calls']}|{x['bank_qualified']}|{x['distinct_selected_supports']}|")
    lines+=['',f"Matched directional banks: {sum(x['bank_qualified'] for x in coverage)}/8cells; native anchors: {sum(x['anchor'] is not None for x in coverage)}/8. Missing anchors and contrasts are not replaced. Routing units and masks are not independent task samples.",'',
        '|Question|Seed|Arm|Remaining calls|Gain versus A|Complete N/C/T|Correct / A correct|Alias|',
        '|---|---:|---|---:|---:|---|---|---|']
    for x in effects:lines.append(f"|{x['id']}|{x['seed']}|{x['arm']}|{x['remaining_calls']}|{x['remaining_gain']}|{x['N']}/{x['C']}/{x['T']}|{x['correct']} / {x['reference_correct']}|{x['physical_execution_alias']}|")
    if not effects:lines+=['','No deleting branch qualified; native deleting-consumer or headroom evidence is therefore absent. Common-control/operator qualification must not be described as a successful deleting path.']
    lines+=['',f"Development decision: {analysis['development_decision']}",
        '',f"Confirmation gate passed: {analysis['confirmation_gate_passed']}. Every executed complete request is officially scored; all quality transitions versus A,FULL,PIECEWISE,current-error and sham are in analysis001/summary.json.",
        '', 'A wrong/capped reference cannot certify preserving a correct answer. A hindsight-best direction is oracle headroom only. Similar maximum current error/mass does not equate every position-wise perturbation. The primary direction pair varies one routing unit; the whole-composite angle and current-error matching status are recorded separately.',
        '', 'Repeated fresh common controls qualify full native output,N,C,T,RNG,counters and target-window fingerprints. Each executed deleting path additionally requires actual current K/V-tail equality, four GQA-route comparisons against explicit current masked attention, correct physical mask and zero protected/LOCAL deletion. Restored output assembly remains unqualified and unused.',
        '',f"Actual model forwards: {int(resources['explicit_total_model_forwards'])}; complete native requests including warmups: {int(resources['complete_native_requests'])}; allocated GPU-worker seconds: {resources['allocated_GPU_worker_seconds']:.3f}. No failed H GPU process recorded.",
        '', 'All H timing is audit instrumentation and is not clean policy latency. No quality-noninferiority, speedup, full-trajectory acceleration or optimal-layer claim follows. This is layer29,four routing units,one5%prefix pulse at a qualified later confidence stall; other locations/strengths and repeated policies were not tested. No Stage B/C or kernel campaign launched.']
    (H/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    labels=[x['id'].split('/')[-1][:8]+'/'+str(x['seed']) for x in coverage]
    ax.bar(range(8),[x['remaining_calls'] or 0 for x in coverage],color='gray',label='Common A reference')
    for arm,color in [('direction_high','tab:blue'),('direction_low','tab:orange')]:
        vals={(x['id'],x['seed']):x['remaining_calls'] for x in effects if x['arm']==arm}
        points=[(i,vals[(x['id'],x['seed'])]) for i,x in enumerate(coverage) if vals.get((x['id'],x['seed'])) is not None]
        if points:ax.scatter([x[0] for x in points],[x[1] for x in points],color=color,label=arm,zorder=3)
    ax.set_xticks(range(8),labels,rotation=35,ha='right');ax.set_ylabel('Remaining calls to the SAME native canvas commit');ax.legend()
    ax.set_title('Later-canvas headroom; missing anchors/contrasts remain missing (zero-height placeholders)')
    fig.savefig(out/'remaining_calls.png',dpi=160);plt.close(fig)
    save(H/'final_complete.json',dict(passed=True,gated_leaf_complete=True,protocol_sha256=sha(H/'protocol.json'),source_checkpoint=binding['source_checkpoint'],
        readout_sha256=sha(H/'readout_frozen.json'),summary_sha256=sha(H/'summary.md'),matched_bank_cells=sum(x['bank_qualified'] for x in coverage),
        confirmation_gate_passed=analysis['confirmation_gate_passed'],clean_policy_latency_qualified=False,accuracy_noninferiority_established=False,Stage_BC=False))


if __name__=='__main__':main()
