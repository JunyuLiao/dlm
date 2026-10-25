"""Derived closeout only; never changes generation or frozen primary analysis."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY/'track_r/attempt001'
REPO=STUDY.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def main():
    assert json.loads((ROOT/'execution_complete.json').read_text())['passed']
    marker=json.loads((ROOT/'readout_frozen.json').read_text())
    assert marker['summary_sha256']==sha(ROOT/'analysis001/summary.json')
    out=ROOT/'closeout001';out.mkdir(exist_ok=False)
    analysis=json.loads((ROOT/'analysis001/summary.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    inputs=dict(json.loads((ROOT/'analysis001/input_hashes.json').read_text()))
    corrected=[];selector_totals=defaultdict(Counter);selector_timers=defaultdict(Counter)
    for row in analysis['rows']:
        audit_path=ROOT/row['audit_path'];inputs[row['audit_path']]=sha(audit_path)
        audit=json.loads(audit_path.read_text());work=audit['work'];selector=audit['selector']
        phases={};calls=selector['physical_calls'] if selector else []
        for call in calls:
            phase='layer29_'+call['phase']
            totals=phases.setdefault(phase,dict(calls=0,eligible=0,kept=0,skipped=0))
            totals['calls']+=1
            for name in ['eligible','skipped']:totals[name]+=call[name]
            totals['kept']+=call['eligible']-call['skipped']
        ge=work['global_eligible_tiles'];skipped=work['global_skipped_tiles']
        layer_eligible=sum(v['eligible'] for v in phases.values())
        phases['native_global_dense']=dict(calls=work['global_calls']-len(calls),eligible=ge-layer_eligible,kept=ge-layer_eligible,skipped=0)
        assert sum(v['eligible'] for v in phases.values())==ge
        assert sum(v['kept'] for v in phases.values())==work['global_kept_tiles']
        assert sum(v['skipped'] for v in phases.values())==skipped
        le=work['native_local_eligible_rectangles'];assert work['native_local_skipped_tiles']==0
        corrected.append(dict(id=row['id'],seed=row['seed'],arm=row['arm'],benchmark=row['benchmark'],
            input_sha256=sha(audit_path),GLOBAL=row['GLOBAL'],LOCAL=row['LOCAL'],overall=row['overall'],
            overall_rectangle_sparsity=skipped/(ge+le),phases=phases,
            original_overall_sparsity=work['overall_rectangle_sparsity'],
            original_dense_phase_retained_unchanged=True,
            inherited_stale_labels=bool(skipped and (work['overall_rectangle_sparsity']==0 or work['value_phase_tiles'].get('dense',{}).get('skipped_tiles')==0))))
        if selector:
            for key,value in selector.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):selector_totals[(row['benchmark'],row['arm'])][key]+=value
            for phase,value in selector['selector_phase_gpu'].items():
                for key,number in value.items():
                    if isinstance(number,(int,float)) and not isinstance(number,bool):selector_timers[(row['benchmark'],row['arm'])][phase+'|'+key]+=number
    # Parent allocation includes startup, warmup and cleanup (15s polling resolution).
    jobs=[];resource=Counter();requests=0
    for leaf in sorted((ROOT/'launches').iterdir()):
        launch=json.loads((leaf/'launch_receipt.json').read_text());exit=json.loads((leaf/'exit_receipt.json').read_text())
        assert exit['returncode']==0
        jobs.append(dict(job=leaf.name,**exit,started_unix=launch['started_unix']))
        if exit['GPU']:
            resource['allocated_gpu_worker_seconds']+=exit['allocated_seconds'];resource['GPU_workers']+=1
    for p in sorted((ROOT/'runs').glob('*/*/*/final_complete.json')):
        value=json.loads(p.read_text());assert value['passed']
        resource['engine_execute_forwards']+=value['model_executions']
        resource['startup_model_forwards']+=value['startup_model_forwards']
        resource['engine_initialization_execute_forwards']+=value['engine_initialization_executions']
        records=[json.loads(q.read_text()) for q in p.parent.glob('*.json') if q.stem.startswith(('audit','clean','warm'))]
        resource['complete_request_forwards']+=sum(v['model_forwards'] for v in records)
        resource['warm_requests']+=sum(v['pass_name']=='warm' for v in records)
        requests+=len(records)
        inputs[str(p.relative_to(ROOT))]=sha(p)
    assert resource['GPU_workers']==16 and requests==1464 and resource['warm_requests']==16
    assert resource['engine_execute_forwards']==resource['complete_request_forwards']+resource['engine_initialization_execute_forwards']
    resource['explicit_total_model_forwards']=resource['engine_execute_forwards']+resource['startup_model_forwards']
    resource['complete_native_requests']=requests
    save(out/'resources.json',dict(**resource,jobs=jobs,failed_attempts=0,no_resource_ceilings=True,
        allocation_note='Parent launch through observed process exit; includes startup/warmup/cleanup, quantized by15s polling. Model startup calls and engine executions are disjoint counters.'))
    save(out/'physical_work_corrections.json',dict(rows=corrected,corrected_rows=sum(x['inherited_stale_labels'] for x in corrected),
        note='Derived phase and overall sparsity repair; original receipts untouched. Frozen primary analysis already recomputed aggregate sparsity correctly. Native prefill/commit remain dense and are outside this denoising rectangle denominator.'))
    save(out/'selector_accounting.json',[dict(benchmark=k[0],arm=k[1],totals=dict(v),phase_timer_totals=dict(selector_timers[k]),
        note='Audit diagnostic/selector work, separate from physical model-consumer work. Timers may nest; never add them to request W/S.') for k,v in sorted(selector_totals.items())])
    verified={p:sha(REPO/p)==h for p,h in protocol['sources'].items()};assert all(verified.values())
    closure=json.loads((ROOT/'scorer_source_closure001.json').read_text())
    helpers={entry['file']:sha(REPO/entry['file'])==entry['sha256'] for entry in [closure['final_response_source'],closure['numeric_score_source']]};assert all(helpers.values())
    save(out/'source_verification.json',dict(passed=True,source_pins=len(verified),all_frozen_sources_unchanged=True,scorer_helper_closure=helpers,
        protocol_sha256=sha(ROOT/'protocol.json'),readout_sha256=sha(ROOT/'readout_frozen.json'),source_checkpoint=json.loads((ROOT/'binding_receipt.json').read_text())['source_checkpoint']))
    save(out/'input_hashes.json',inputs)
    lines=['# Unchanged Response behavior replication','',
        'Complete exploratory short-task panel:30 AIME26 and40 HumanEval questions, paired seeds42/43, five arms. Original Response selector/runtime bytes are unchanged. This is not Stage B/C.','',
        'Legacy reproduction passed24 arm/cell comparisons, including HumanEval/71 seed43. All700 audit/clean cell pairs matched output,N,C,T and final CPU/CUDA RNG.','',
        '|Task / exposure|Arm|Correct / cells|N total|C total|T total|Mean N/C|Mean clean W (s)|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for a in analysis['aggregates']:
        if a['stratum'] not in ['all','new_to_response']:continue
        lines.append(f"|{a['benchmark']} / {a['stratum']}|{a['arm']}|{a['correct']}/{a['cells']}|{a['totals']['N']}|{a['totals']['C']}|{a['totals']['T']}|{a['means']['N_per_C']:.3f}|{a['means']['W']:.3f}|")
    lines+=['','Primary confirmation retains questions as clusters, with both seeds. Eight prespecified comparisons use simultaneous99.375% bootstrap intervals;95% intervals and legacy/full/strict-no-A1 sensitivities are descriptive. No noninferiority margin was declared.','',
        '|Task|Response versus|Quality difference (simultaneous interval when primary)|Paired N ratio (simultaneous interval when primary)|Paired N/C ratio|Paired T ratio|Paired clean W ratio|',
        '|---|---|---|---|---:|---:|---:|']
    for c in analysis['comparisons']:
        if c['stratum']!='new_to_response':continue
        def fmt(x):
            interval=x.get('simultaneous_ci99375') or x['ci95']
            return f"{x['estimate']:.4f} [{interval[0]:.4f},{interval[1]:.4f}]"
        q=c['quality_difference'];r=c['ratios']
        lines.append(f"|{c['benchmark']}|{c['reference']}|{fmt(q)}|{fmt(r['N'])}|{r['N_per_C']['estimate']:.4f}|{r['T']['estimate']:.4f}|{r['W']['estimate']:.4f}|")
    lines+=['','Decisions:']
    for d in analysis['decisions']:lines.append(f"- {d['benchmark']}: {d['disposition']}; clear quality lead={d['clear_quality_lead']}, clear total-work lead={d['clear_total_work_lead']}. Observed quality losses versus: {', '.join(d['observed_quality_losses']) or 'none' }.")
    lines+=['','Every correctness transition is preserved in analysis001/summary.json. New-to-Response is not universally unseen: all70 identities match some known historical prompt-tokenization hash. The AIME primary20-question stratum includes two A1-exposed questions; the strict18-question sensitivity is separate. HumanEval30 new questions exclude A1.',
        '',f"Explicit model forwards: {int(resource['explicit_total_model_forwards'])}; complete native requests: {requests}; allocated GPU-worker seconds: {resource['allocated_gpu_worker_seconds']:.3f}. No failed GPU process is recorded. Allocation includes startup, qualification, warmup and cleanup.",
        '', 'Quality, total calls, calls/canvas, clean latency and causal mechanism are separate findings. Audit/operator timing is not clean policy latency. Physical rectangle accounting is not CUDA CTA timing. Prefix extraction and correct-in-both subsets are retrospective descriptions, not a native stopping policy or causal redundancy explanation.',
        '', 'Track H is separately frozen and gated. Previous unfavorable LongBench Response evidence remains; no new LongBench Response panel was launched. No production optimization or new selector was introduced.']
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
    # Standard exported scientific figures; no private text/tensors.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,task in zip(axes,['aime26','humaneval']):
        entries=[a for a in analysis['aggregates'] if a['benchmark']==task and a['stratum']=='new_to_response']
        ax.barh([a['arm'] for a in entries],[a['correct']/a['cells'] for a in entries]);ax.set_xlim(0,1);ax.set_xlabel('Official accuracy, paired seeds (question clusters)');ax.set_title(task+' — new to original Response panel')
    fig.savefig(out/'quality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,task in zip(axes,['aime26','humaneval']):
        entries=[c for c in analysis['comparisons'] if c['benchmark']==task and c['stratum']=='new_to_response']
        for i,c in enumerate(entries):
            x=c['ratios']['N'];ci=x.get('simultaneous_ci99375') or x['ci95']
            ax.errorbar(x['estimate'],i,xerr=[[x['estimate']-ci[0]],[ci[1]-x['estimate']]],fmt='o')
        ax.set_yticks(range(len(entries)),[c['reference'] for c in entries]);ax.axvline(1,color='gray',linestyle='--');ax.set_xlabel('Response / control total N, question-weighted');ax.set_title(task)
    fig.savefig(out/'paired_total_calls.png',dpi=160);plt.close(fig)
    save(ROOT/'final_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),source_checkpoint=json.loads((ROOT/'binding_receipt.json').read_text())['source_checkpoint'],
        readout_sha256=sha(ROOT/'readout_frozen.json'),summary_sha256=sha(ROOT/'summary.md'),completed_cells=700,independent_questions=dict(aime26=30,humaneval=40),quality_preservation_established=False,clean_speed_claim=False,
        Track_H_included=False,Stage_BC_launched=False))


if __name__=='__main__':main()
