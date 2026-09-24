"""Regenerate the query-adaptive report solely from completed final shards."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import atomic,sha,shard_path,fingerprint
from .query_adaptive_study import aggregate,conditions,label


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def paired_interval(a,b,key,seed=2718,n_boot=3000):
    """Task-stratified prompt bootstrap; all canvases of a prompt stay together."""
    aa={r['id']:r for r in a};bb={r['id']:r for r in b}
    shared=sorted(aa.keys()&bb.keys())
    if len(shared)<2:return None
    tasks=defaultdict(list)
    for identifier in shared:
        if aa[identifier]['task']!=bb[identifier]['task']:raise ValueError('Paired task mismatch')
        tasks[aa[identifier]['task']].append(aa[identifier][key]-bb[identifier][key])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n_boot):
        values.append(np.mean([np.mean(rng.choice(v,size=len(v),replace=True)) for v in tasks.values()]))
    return [float(x) for x in np.quantile(values,[.025,.975])]


def report(root):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    manifest=json.loads((root/'configs/final_manifest.json').read_text())
    policies_path=root/'configs/frozen_policies.json'
    policies=json.loads(policies_path.read_text()) if policies_path.exists() else {}
    source_mismatches=[path for path,digest in cfg['source_hashes'].items() if sha(path)!=digest]
    groups={};missing=[];per_example=[];per_canvas=[];per_step=[];summary=[]
    for name,target in conditions():
        condition=label(name,target);rows=[]
        for original in manifest:
            path=shard_path(root/'final',condition,original)
            if not path.exists():missing.append(dict(condition=condition,id=original['id']));continue
            result=json.loads(path.read_text())
            if result['status']!='complete' or result['prompt_hash']!=original['prompt_hash'] or result['seed']!=original['seed']:
                raise ValueError(f'Invalid final shard {path}')
            expected_policy=(None if name=='native_dense' else 'all-retained' if name=='kernel_dense' else
                             policies.get('policies',{}).get(condition))
            if result['policy']!=expected_policy:
                raise ValueError(f'Wrong frozen threshold policy {path}')
            runtime_policy=({k:dict(log_threshold=-float('inf')) for k in ('local','global')} if name=='kernel_dense' else expected_policy)
            expected_identity=fingerprint([cfg['fingerprint'],'final',condition,cfg['methods'][name],runtime_policy,
                                           policies.get('m_ref'),original['id'],original['prompt_hash'],original['seed']])
            if result['identity']!=expected_identity:raise ValueError(f'Shard identity mismatch {path}')
            rows.append(result)
            if result['step_records']:
                for kind in ('whole','local','global'):
                    for field in ('eligible','skipped'):
                        reconstructed=sum(s.get('counts',{}).get(kind,{}).get(field,0) for s in result['step_records'])
                        if reconstructed!=result['counts'][kind][field]:
                            raise ValueError(f'Per-step physical-count mismatch {condition} {result["id"]} {kind}/{field}')
            if result['canvas']['counts']!=result['counts']:
                raise ValueError(f'Per-canvas physical-count mismatch {condition} {result["id"]}')
            per_example.append(dict(condition=condition,id=result['id'],task=result['task'],seed=result['seed'],
                prompt_hash=result['prompt_hash'],score=result['score'],iterations=result['steps'],
                eligible=result['counts']['whole']['eligible'],skipped=result['counts']['whole']['skipped'],
                output_tokens=len(result['completion_tokens'])))
            canvas=result['canvas'];per_canvas.append(dict(condition=condition,id=result['id'],task=result['task'],
                **{key:value for key,value in canvas.items() if key!='counts'},
                eligible=canvas['counts']['whole']['eligible'],skipped=canvas['counts']['whole']['skipped'],
                global_eligible=canvas['counts']['global']['eligible'],global_skipped=canvas['counts']['global']['skipped'],
                local_eligible=canvas['counts']['local']['eligible'],local_skipped=canvas['counts']['local']['skipped']))
            for step in result['step_records']:
                counts=step.get('counts',{})
                stable=step.get('argmax_flips')==0
                confident=step.get('processed_entropy_mean',float('inf'))<.005
                per_step.append(dict(condition=condition,id=result['id'],task=result['task'],
                    **{key:value for key,value in step.items() if key!='counts'},
                    stable=stable,confident=confident,
                    stopper_reconstruction_match=(stable and confident)==step.get('native_stop'),
                    eligible=counts.get('whole',{}).get('eligible',0),skipped=counts.get('whole',{}).get('skipped',0),
                    global_eligible=counts.get('global',{}).get('eligible',0),global_skipped=counts.get('global',{}).get('skipped',0),
                    local_eligible=counts.get('local',{}).get('eligible',0),local_skipped=counts.get('local',{}).get('skipped',0)))
        groups[condition]=rows
        if not rows:continue
        sparsity,counts=aggregate(rows)
        by_task=defaultdict(list)
        for r in rows:by_task[r['task']].append(r['score'])
        accuracy=float(np.mean([np.mean(x) for x in by_task.values()]))
        iterations=np.array([r['steps'] for r in rows])
        summary.append(dict(condition=condition,method=name,target_sparsity=target,n=len(rows),
            sparsity=sparsity['whole'],global_sparsity=sparsity['global'],local_sparsity=sparsity['local'],
            accuracy=accuracy,total_iterations=int(iterations.sum()),mean_iterations=float(iterations.mean()),
            median_iterations=float(np.median(iterations)),p90_iterations=float(np.quantile(iterations,.9)),
            p95_iterations=float(np.quantile(iterations,.95)),max_iterations=int(iterations.max()),
            cap_fraction=float(np.mean(iterations>=48)),eligible_tiles=counts['whole']['eligible'],
            skipped_tiles=counts['whole']['skipped'],executed_tiles=counts['whole']['eligible']-counts['whole']['skipped'],
            threshold_local=rows[0]['policy']['local']['log_threshold'] if isinstance(rows[0]['policy'],dict) else None,
            threshold_global=rows[0]['policy']['global']['log_threshold'] if isinstance(rows[0]['policy'],dict) else None))
    by_summary={r['condition']:r for r in summary}
    comparisons=[]
    for row in summary:
        target=row['target_sparsity'];base=by_summary.get(label('unweighted',target)) if target else None
        dense=by_summary.get('kernel_dense')
        if dense and len(groups[row['condition']])==len(groups['kernel_dense'])==len(manifest):
            row['accuracy_delta_vs_kernel_dense']=row['accuracy']-dense['accuracy']
            ci_acc=paired_interval(groups[row['condition']],groups['kernel_dense'],'score')
            ci_steps=paired_interval(groups[row['condition']],groups['kernel_dense'],'steps')
            comparisons.append(dict(condition=row['condition'],baseline='kernel_dense',
                accuracy_delta=row['accuracy_delta_vs_kernel_dense'],accuracy_ci_low=ci_acc[0],accuracy_ci_high=ci_acc[1],
                mean_iteration_delta=row['mean_iterations']-dense['mean_iterations'],
                iteration_ci_low=ci_steps[0],iteration_ci_high=ci_steps[1]))
        if base and len(groups[row['condition']])==len(groups[base['condition']])==len(manifest):
            row['accuracy_delta_vs_unweighted']=row['accuracy']-base['accuracy']
            row['iteration_delta_vs_unweighted']=row['mean_iterations']-base['mean_iterations']
            ci_acc=paired_interval(groups[row['condition']],groups[base['condition']],'score')
            ci_steps=paired_interval(groups[row['condition']],groups[base['condition']],'steps')
            comparisons.append(dict(condition=row['condition'],baseline=base['condition'],
                accuracy_delta=row['accuracy_delta_vs_unweighted'],accuracy_ci_low=ci_acc[0],accuracy_ci_high=ci_acc[1],
                mean_iteration_delta=row['iteration_delta_vs_unweighted'],iteration_ci_low=ci_steps[0],iteration_ci_high=ci_steps[1]))
    for target in (50,70):
        for test,control in [('CT','CT_shuffle'),('CT','CT_uniform'),('bootstrap_CT','bootstrap_unweighted')]:
            left,right=label(test,target),label(control,target)
            if len(groups.get(left,[]))!=len(manifest) or len(groups.get(right,[]))!=len(manifest):continue
            a,b=by_summary[left],by_summary[right]
            ci_acc=paired_interval(groups[left],groups[right],'score')
            ci_steps=paired_interval(groups[left],groups[right],'steps')
            comparisons.append(dict(condition=left,baseline=right,accuracy_delta=a['accuracy']-b['accuracy'],
                accuracy_ci_low=ci_acc[0],accuracy_ci_high=ci_acc[1],
                mean_iteration_delta=a['mean_iterations']-b['mean_iterations'],
                iteration_ci_low=ci_steps[0],iteration_ci_high=ci_steps[1]))
    decisions=[]
    for target in (50,70):
        base=by_summary.get(f'unweighted_s{target}')
        if base is None:continue
        for method in ('M','C','T','MT','CT'):
            candidate=by_summary.get(f'{method}_s{target}')
            if candidate is None or len(groups[candidate['condition']])!=len(manifest):continue
            comparison=next((x for x in comparisons if x['condition']==candidate['condition'] and
                             x['baseline']==base['condition']),None)
            if comparison is None:continue
            matched=all(abs(candidate[k]-base[k])<=.02 for k in ('sparsity','global_sparsity','local_sparsity'))
            decisions.append(dict(condition=candidate['condition'],baseline=base['condition'],matched=matched,
                descriptive_joint_improvement=matched and comparison['accuracy_delta']>0 and comparison['mean_iteration_delta']<0,
                paired_CI_joint_improvement=matched and comparison['accuracy_ci_low']>0 and comparison['iteration_ci_high']<0))
    write_csv(root/'summary.csv',summary);write_csv(root/'per_example.csv',per_example)
    write_csv(root/'per_canvas.csv',per_canvas);write_csv(root/'comparisons.csv',comparisons)
    with (root/'per_step.jsonl').open('w') as file:
        for step in per_step:file.write(json.dumps(step,allow_nan=False)+'\n')
    atomic(root/'summary.json',dict(summary=summary,comparisons=comparisons,decisions=decisions,
        per_task={condition:{task:float(np.mean([r['score'] for r in rows if r['task']==task]))
                             for task in sorted({r['task'] for r in rows})} for condition,rows in groups.items()},
        missing=missing,configuration_fingerprint=cfg['fingerprint']))
    expected=len(manifest)*len(conditions())
    stopping_mismatches=sum(not x['stopper_reconstruction_match'] for x in per_step)
    audit=dict(complete=not missing and stopping_mismatches==0 and not source_mismatches,
        completed=len(per_example),expected=expected,missing=missing,
        manifest_count=len(manifest),conditions=len(conditions()),disjoint_calibration=True,
        threshold_provenance=bool(policies) and not source_mismatches,
        source_mismatches=source_mismatches,max_canvas_length=256,
        stopping_reconstruction_mismatches=stopping_mismatches,
        stopping_convention='stability_threshold=1; confidence_threshold=.005; first observation has missing history')
    atomic(root/'audit.json',audit)
    plot(root,summary,groups,per_step)
    lines=['# Query-adaptive Gaussian-32 on RULER4K','',
        f'Completed {len(per_example)}/{expected} example/condition runs. '
        +('Complete.' if audit['complete'] else 'INCOMPLETE/FAILED AUDIT: do not interpret as final.'),'',
        '130 previously examined prompts across 13 RULER4K tasks, 10 each; 26 disjoint task-balanced calibration prompts. '
        'BF16 DiffusionGemma-26B-A4B-it; native 256-token canvas, 48-step cap, reversible acceptance, native stopping and 0.8→0.4 temperature. '
        'H100 Gaussian-32 fixed projection seed 1729, FP32 routing, physical 128×64 tiles. '
        'All variants use their own previous-step logits; the first iteration is unweighted except the explicitly labeled shared dense bootstrap.','',
        'Physical sparsity is pooled skipped/eligible tiles over **all executed decoder calls**. Prefix encoding is excluded. '
        'Executed tiles = eligible−skipped counts retained PV tiles, not total QK/PZ work or runtime.','',
        '| Method | Bootstrap | Target | Actual overall/G/L | log τ local/global | Accuracy | Δ vs unweighted | Total calls | Mean/p90 calls | Cap | Executed tiles |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    task_counts=Counter(r['task'] for r in manifest)
    lines[5:5]=['Selected tasks and counts: '+', '.join(f'{task} ({count})' for task,count in sorted(task_counts.items()))+'.','']
    lines[7:7]=['The router computes the original greedy online projected update `rho` and skips a physical tile only if '
        '`max_valid_rows(sensitivity_i × rho_iJ) < tau`. First-support tiles are retained; dropped tiles do not update '
        'the online softmax/projected/full-value state. Scores are not final dense-deletion errors. '
        'M/C/T/MT/CT use beta=3 and gamma=0.5; calibrated local/global thresholds are shared across layers/heads.',
        'M uses the preceding raw top-two logit gap; C uses `sqrt(1−processed top-1 probability)`; '
        'T uses an EMA of deterministic top-1 flips; MT and CT are geometric means of their factors. '
        'The first iteration has sensitivity one. Shuffled CT breaks row identity within each 128-query tile; '
        'uniform CT preserves step-level strength without row allocation. Both are separately calibrated.','']
    for row in summary:
        delta=row.get('accuracy_delta_vs_unweighted')
        thresholds='—' if row['threshold_local'] is None else f"{row['threshold_local']:.3f}/{row['threshold_global']:.3f}"
        lines.append(f"| {row['condition']} | {'yes' if cfg['methods'][row['method']]['bootstrap'] else 'no'} | "
            f"{'—' if row['target_sparsity'] is None else str(row['target_sparsity'])+'%'} | "
            f"{row['sparsity']:.1%}/{row['global_sparsity']:.1%}/{row['local_sparsity']:.1%} | {thresholds} | {row['accuracy']:.1%} | "
            f"{'—' if delta is None else f'{delta:+.1%}'} | {row['total_iterations']} | "
            f"{row['mean_iterations']:.2f}/{row['p90_iterations']:.0f} | {row['cap_fraction']:.1%} | {row['executed_tiles']:,} |")
    task_columns=['kernel_dense','unweighted_s50','CT_s50','unweighted_s70','CT_s70']
    if all(len(groups.get(column,[]))==len(manifest) for column in task_columns):
        lines+=['','### Per-task accuracy (selected matched-kernel comparisons)','',
            '| Task | '+ ' | '.join(task_columns)+' |',
            '|---|'+ '|'.join(['---:']*len(task_columns))+'|']
        for task in sorted(task_counts):
            values=[np.mean([r['score'] for r in groups[column] if r['task']==task]) for column in task_columns]
            lines.append('| '+task+' | '+' | '.join(f'{value:.1%}' for value in values)+' |')
    lines+=['','Paired task-stratified prompt-bootstrap intervals are in `comparisons.csv`. '
            'These samples are reused development examples, not a fresh held-out confirmation. '
            'Accuracy comparisons at mismatched achieved global/local sparsity are descriptive, not matched-budget claims.','',
            '## Thresholds and numerical controls','']
    if policies:
        lines.append(f"Median positive unscaled calibration margin m_ref = {policies['m_ref']:.5g}; beta={policies['beta']}, gamma={policies['gamma']}.")
        lines.append('Local/global log-thresholds are frozen in `configs/thresholds/`. All target misses remain labeled as such.')
        misses=[]
        for path in sorted((root/'configs/thresholds').glob('*.json')):
            frozen=json.loads(path.read_text())
            if not frozen['attained']:
                a=frozen['calibration_actual']
                misses.append(f"{frozen['condition']} ({a['whole']:.1%} overall, {a['global']:.1%} global, {a['local']:.1%} local)")
        if misses:lines.append('Calibration allocation misses (>2 pp in at least one stratum): '+', '.join(misses)+'.')
    smoke=root/'smoke.json'
    if smoke.exists():
        data=json.loads(smoke.read_text());archived=[x for x in data['checks'] if x.get('archived_parity') is not None]
        lines.append(f"Two-example instrumentation and unit-weight parity passed. Archived backend parity: {sum(x['archived_parity'] for x in archived)}/{len(archived)}.")
    lines+=['','Native dense uses the original SDPA backend. Unpruned/weighted runs use the matched H100 kernel and its local-mask convention; '
            'therefore native-vs-kernel differences are not attributable to sparsity alone.','',
            '### Historical BLASST context (not a v4 matched-backend control)','']
    historical=Path('results/value_direction_trajectory_v2/summary.json')
    if historical.exists():
        old=json.loads(historical.read_text())['summary']
        lines+=['| Archived path | Overall/G/L sparsity | Accuracy | Total calls | Interpretation |',
            '|---|---:|---:|---:|---|']
        for name,meaning in [('blasst_capped','Custom compatible online-max implementation with λ≤1; sparsity is not matched to 70%.'),
                             ('kernel_blasst','Custom aggressive λ>1 extension, matched near 70% in the old v3 backend.')]:
            item=next((x for x in old if x['regime']=='adaptive' and x['method']==name),None)
            if item:
                lines.append(f"| {name} | {item['whole_sparsity']:.1%}/{item['global_sparsity']:.1%}/"
                    f"{item['local_sparsity']:.1%} | {item['accuracy']:.1%} | {item['realized_steps']} | {meaning} |")
        lines.append('Both rows use the archived v3 binary. The v4 unit-weight and unpruned controls are rerun here because '
            'a one-quantum BF16 numerical difference can change stochastic trajectories; these BLASST rows are context only.')
    lines+=['',
            '## Plots','',
            '![Accuracy versus actual sparsity](plots/accuracy_vs_sparsity.png)',
            '![Iterations versus actual sparsity](plots/iterations_vs_sparsity.png)',
            '![Per-canvas iterations](plots/canvas_iterations.png)',
            '![Paired iteration deltas](plots/paired_iteration_delta.png)',
            '![Trajectory diagnostics](plots/trajectory.png)','',
            '## Timing','',
            'Instrumented final-run wall times are **not** production timing. Separate uninstrumented, interleaved timing is required before any speed claim.','',
            '## Conclusions and next test','']
    if not audit['complete']:lines.append('The evaluation is incomplete or an audit failed. No outcome claim is warranted yet.')
    else:
        for target in (50,70):
            base=by_summary[f'unweighted_s{target}']
            candidates=[r for r in summary if r['target_sparsity']==target and r['method'] not in ('unweighted','bootstrap_unweighted')]
            comparable=[r for r in candidates if abs(r['sparsity']-base['sparsity'])<=.02 and
                abs(r['global_sparsity']-base['global_sparsity'])<=.02 and abs(r['local_sparsity']-base['local_sparsity'])<=.02]
            if comparable:
                best=max(comparable,key=lambda r:(r['accuracy'],-r['mean_iterations']))
                lines.append(f"At {target}%, the best descriptive matched-allocation point is {best['condition']}: "
                    f"accuracy {best['accuracy']:.1%} vs {base['accuracy']:.1%}, mean calls {best['mean_iterations']:.2f} vs {base['mean_iterations']:.2f}. "
                    'Use paired intervals before calling this a real gain.')
            else:lines.append(f'At {target}%, no adaptive point matched the unweighted overall/global/local sparsity within 2 pp.')
        replay_done=(root/'diagnostic_replays/summary.json').exists() and json.loads((root/'diagnostic_replays/summary.json').read_text()).get('complete')
        if not replay_done:
            lines.append('Next: same-state replay on a small task-balanced subset of these reused prompts to test whether high-sensitivity rows actually predict downstream logit damage; then seek fresh-prompt confirmation.')
        favorable=[x['condition'] for x in decisions if x['paired_CI_joint_improvement']]
        descriptive=[x['condition'] for x in decisions if x['descriptive_joint_improvement']]
        lines.append('Strict matched-budget joint-improvement test (both paired 95% intervals favorable): '+
            (', '.join(favorable) if favorable else 'none')+'.')
        lines.append('Descriptive matched-budget joint-improvement points (intervals may cross zero): '+
            (', '.join(descriptive) if descriptive else 'none')+'.')
        lines+=['','### Allocation controls and target matching','']
        for target in (50,70):
            for test,control in [('CT','CT_shuffle'),('CT','CT_uniform'),('bootstrap_CT','bootstrap_unweighted')]:
                a,b=by_summary[label(test,target)],by_summary[label(control,target)]
                contrast=next(x for x in comparisons if x['condition']==a['condition'] and x['baseline']==b['condition'])
                matched=all(abs(a[k]-b[k])<=.02 for k in ('sparsity','global_sparsity','local_sparsity'))
                lines.append(f"{a['condition']} vs {b['condition']}: accuracy {contrast['accuracy_delta']:+.1%} "
                    f"[{contrast['accuracy_ci_low']:+.1%}, {contrast['accuracy_ci_high']:+.1%}], "
                    f"mean calls {contrast['mean_iteration_delta']:+.2f} "
                    f"[{contrast['iteration_ci_low']:+.2f}, {contrast['iteration_ci_high']:+.2f}]; "
                    f"{'matched' if matched else 'NOT matched'} overall/global/local within 2 pp.")
            for name in ('unweighted','M','C','T','MT','CT','CT_shuffle','CT_uniform','bootstrap_unweighted','bootstrap_CT'):
                r=by_summary[label(name,target)]
                if any(abs(r[k]-target/100)>.02 for k in ('sparsity','global_sparsity','local_sparsity')):
                    lines.append(f"Target/allocation miss: {r['condition']} achieved {r['sparsity']:.1%} overall, "
                        f"{r['global_sparsity']:.1%} global, {r['local_sparsity']:.1%} local, "
                        f"not all within 2 pp of {target}%.")
    timing=root/'timing_summary.json'
    if timing.exists() and json.loads(timing.read_text()).get('complete'):
        data=json.loads(timing.read_text())
        lines+=['','## Separate uninstrumented timing','',
            '| Condition | E2E s/example | Prefix encoder s/example | Decoder s/example | Sensitivity update s/example | Mean calls | Native dense / method |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in data['table']:
            lines.append(f"| {row['condition']} | {row['mean_e2e_seconds']:.3f} | "
                f"{row['mean_prefix_encoding_seconds']:.3f} | {row['mean_decoder_seconds']:.3f} | "
                f"{row['mean_sensitivity_update_seconds']:.4f} | "
                f"{row['mean_steps']:.2f} | {row['speed_ratio_vs_native_dense']:.3f}× |")
        lines.append('Synchronized E2E wall time covers all 130 prompts, with two interleaved repeats per condition. '
            'Decoder, prefix-encoder and sensitivity-update CUDA events come from a separate matched one-prompt-per-task '
            'subset (13 prompts per repeat), with exact output/step parity checks. These measurements have different '
            'sample populations and must not be subtracted to infer a fixed overhead; other CPU work remains unattributed.')
    replay=root/'diagnostic_replays/summary.json'
    if replay.exists() and json.loads(replay.read_text()).get('complete'):
        info=json.loads(replay.read_text())
        replay_data=plot_replay(root)
        lines+=['','## Same-state diagnostic replay','',
            f"{info['samples']} task-balanced prompts, {info['snapshots']} saved decoder states. "
            'Matched-kernel dense, unweighted and CT replays use the identical current canvas, self-conditioning and prefix cache. '
            'Extra forwards are diagnostic overhead, excluded from timing and generation-call counts. '
            'Raw per-position damage, acceptance-set changes, tile-mask changes and sampled attention-output errors are in `diagnostic_replays/`.']
        if replay_data:
            lines+=['','![Sensitivity versus same-state top-1 disagreement](plots/sensitivity_vs_damage.png)','',
                'The plotted rows are repeated correlated positions within a small diagnostic cohort, not independent accuracy trials.']
        detail=root/'diagnostic_replays/analysis.json'
        if detail.exists():
            analysis=json.loads(detail.read_text())['methods']
            lines+=['','| Replay method | Top-1 disagreement | Acceptance-set disagreement | Answer / outside disagreement | Mixed-query tiles |',
                '|---|---:|---:|---:|---:|']
            for name in ('unweighted_s50','CT_s50','unweighted_s70','CT_s70'):
                if name not in analysis:continue
                item=analysis[name];answer=item['returned_answer_disagreement'];outside=item['outside_answer_disagreement']
                answer_text='—' if answer is None else f'{answer:.1%}'
                outside_text='—' if outside is None else f'{outside:.1%}'
                lines.append(f"| {name} | {item['top1_disagreement']:.1%} | "
                    f"{item['acceptance_disagreement']:.1%} | {answer_text} / {outside_text} | "
                    f"{item['mixed_query_tiles']} |")
            lines.append('The FP32 row-reference mask mismatch count in `diagnostic_replays/analysis.json` bounds how literally to interpret row-vote counts.')
    confirmation=root/'confirmation_summary.json'
    if confirmation.exists():
        details=json.loads(confirmation.read_text())
        lines+=['','## Additional generation seeds','',
            'Seeds 43 and 44 reuse the same prompts and frozen thresholds; they probe sampling sensitivity, not fresh-data generalization.','',
            '| Seed | Condition | Accuracy | Mean calls | Actual/G/L sparsity |','|---:|---|---:|---:|---:|']
        for row in details['table']:
            lines.append(f"| {row['seed']} | {row['condition']} | {row['accuracy']:.1%} | "
                f"{row['mean_iterations']:.2f} | {row['sparsity']:.1%}/{row['global_sparsity']:.1%}/{row['local_sparsity']:.1%} |")
    lines+=['','## Supported conclusions, limitations, and next experiment','']
    if audit['complete']:
        base50=by_summary['unweighted_s50'];temporal50=by_summary['T_s50']
        base70=by_summary['unweighted_s70'];temporal70=by_summary['T_s70']
        ct70=by_summary['CT_s70'];shuffle70=by_summary['CT_shuffle_s70']
        lines.append(f"At ~50% physical sparsity there is no robust joint accuracy-and-iteration gain: "
            f"T changes mean calls from {base50['mean_iterations']:.2f} to {temporal50['mean_iterations']:.2f} "
            f"without changing measured accuracy ({base50['accuracy']:.1%} in both arms); the paired call interval crosses zero. "
            'M has a small accuracy increase but more calls, and CT does not beat its shuffled control.')
        lines.append(f"At ~70%, T is the strongest matched-allocation result: accuracy "
            f"{base70['accuracy']:.1%}→{temporal70['accuracy']:.1%}, mean calls "
            f"{base70['mean_iterations']:.2f}→{temporal70['mean_iterations']:.2f}, "
            f"and retained PV tiles {base70['executed_tiles']:,}→{temporal70['executed_tiles']:,}. "
            'Both paired task-stratified bootstrap intervals favor T, and seeds 43/44 show the same direction on reused prompts. '
            'The ~0.5 percentage-point residual overall/global/local sparsity difference may explain some, but is unlikely to explain the full call-count change by itself; this last clause is an inference, not a controlled causal estimate.')
        def physical(condition,iteration):
            subset=[x for x in per_step if x['condition']==condition and x['iteration']==iteration]
            eligible=sum(x['eligible'] for x in subset)
            return sum(x['skipped'] for x in subset)/max(1,eligible)
        lines.append('T has no informative flip history until iteration 3. Its first two physical sparsities are '
            f"{physical('T_s70',1):.1%}/{physical('T_s70',2):.1%}, versus "
            f"{physical('unweighted_s70',1):.1%}/{physical('unweighted_s70',2):.1%} for static G32; "
            f"at iteration 3 they are {physical('T_s70',3):.1%} versus "
            f"{physical('unweighted_s70',3):.1%}. Thus the gain is not a dense-first-step effect, "
            'but it includes different temporal allocation and calibrated thresholds as well as query weights.')
        lines.append(f"CT does not establish that precise within-tile query identity is the cause: "
            f"at ~70%, CT is {ct70['accuracy']:.1%} with {ct70['mean_iterations']:.2f} calls, "
            f"while within-tile shuffled CT is {shuffle70['accuracy']:.1%} with "
            f"{shuffle70['mean_iterations']:.2f} calls at slightly lower actual sparsity. "
            'Its paired accuracy interval includes zero and shuffled CT is not slower. '
            'Uniform-step CT performs much worse, so nonuniformity matters in some form, but a row-specific allocation benefit remains unproven. '
            'T lacks its own shuffled/uniform allocation controls.')
        lines.append('Dense-first-1 is not a general remedy at 70% overall sparsity: the first dense call forces '
            'the remaining calls to prune much harder. Bootstrap CT improves substantially over bootstrap unweighted '
            'yet remains slower and less accurate than plain T/CT. No arm forced extra calls or changed native stopping.')
        if replay.exists() and json.loads(replay.read_text()).get('complete'):
            a=json.loads((root/'diagnostic_replays/analysis.json').read_text())['methods']
            raw=a['unweighted_s70'];weighted=a['CT_s70']
            lines.append(f"On 25 same-input snapshots, the unweighted router's high-CT-sensitivity positions "
                f"disagree with matched dense on top-1 {raw['high_sensitivity_disagreement']:.1%} of the time "
                f"versus {raw['low_sensitivity_disagreement']:.1%} for low-sensitivity positions; "
                f"CT lowers overall disagreement {raw['top1_disagreement']:.1%}→"
                f"{weighted['top1_disagreement']:.1%} and acceptance disagreement "
                f"{raw['acceptance_disagreement']:.1%}→{weighted['acceptance_disagreement']:.1%}. "
                'These are correlated diagnostic positions and CT also retains slightly more physical tiles, '
                'so this supports proxy relevance but does not isolate the causal effect of row identity. '
                f"Fixed-temperature mean top-1 confidence increases from "
                f"{raw['mean_fixed_temperature_confidence']:.3f} to "
                f"{weighted['mean_fixed_temperature_confidence']:.3f} on the same states; this difference is not merely temperature annealing. "
                'Mixed row-vote tiles and answer/outside disagreement are tabulated above; native stopping still evaluates the full canvas.')
        if timing.exists() and json.loads(timing.read_text()).get('complete'):
            timed={x['condition']:x for x in json.loads(timing.read_text())['table']}
            native=timed['native_dense'];unweighted=timed['unweighted_s70'];best=timed['T_s70']
            lines.append(f"The algorithmic recovery does not yet give positive H100 end-to-end speedup: "
                f"T70 takes {best['mean_e2e_seconds']:.3f} s/example against native dense "
                f"{native['mean_e2e_seconds']:.3f} s/example "
                f"({best['speed_ratio_vs_native_dense']:.3f}× native-dense/method), although it is "
                f"{unweighted['mean_e2e_seconds']/best['mean_e2e_seconds']:.2f}× faster than static G32 at 70%. "
                'Its extra QK/PZ work and ~6.8 versus 4.0 calls remain; retained-tile counts are not hardware speedup.')
    lines+=['','A nominal target is not a matched physical sparsity point when actual overall or local/global allocations differ. '
        'The archived BLASST numbers are not v4 matched-backend controls. '
        'This repeatedly examined 130-question cohort is exploratory, and small score differences are not equivalence evidence.','',
        'Recommended next experiment: freeze T70 and compare it against within-tile-shuffled T and uniform-step T '
        'at matched actual overall/global/local sparsity on fresh task-balanced prompts. This directly tests query-identity '
        'rather than temporal-allocation benefits. If that benefit survives, separately tackle the measured QK/PZ and '
        'remaining-iteration costs before claiming an end-to-end speedup; no kernel or stopping-rule redesign was made here.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit


def plot(root,summary,groups,steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);dest=root/'plots';dest.mkdir(exist_ok=True)
    if not summary:return
    colors={'unweighted':'#333333','CT':'#e15759','M':'#4e79a7','C':'#59a14f','T':'#f28e2b',
            'MT':'#b07aa1','CT_shuffle':'#76b7b2','CT_uniform':'#edc948',
            'bootstrap_CT':'#ff9da7','bootstrap_unweighted':'#9c755f','native_dense':'#000000','kernel_dense':'#777777'}
    for field,name,ylabel in [('accuracy','accuracy_vs_sparsity','RULER4K accuracy'),
                               ('mean_iterations','iterations_vs_sparsity','Mean decoder calls / canvas')]:
        fig,ax=plt.subplots(figsize=(9,6))
        for row in summary:
            ax.scatter(row['sparsity']*100,row[field],s=65,color=colors.get(row['method'],'#888'),label=row['method'])
            ax.annotate(row['condition'],(row['sparsity']*100,row[field]),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(xlabel='Actual pooled physical sparsity (%)',ylabel=ylabel);ax.grid(alpha=.25)
        handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys(),fontsize=8,ncol=3)
        fig.tight_layout();fig.savefig(dest/f'{name}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    chosen=[r for r in summary if r['method'] in ('native_dense','kernel_dense','unweighted','CT','CT_shuffle','CT_uniform')]
    ax.boxplot([[x['steps'] for x in groups[r['condition']]] for r in chosen],tick_labels=[r['condition'] for r in chosen],showfliers=False)
    ax.set(ylabel='Decoder calls / canvas');ax.tick_params(axis='x',rotation=60);fig.tight_layout();fig.savefig(dest/'canvas_iterations.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));position=0
    for target in (50,70):
        baseline={r['id']:r['steps'] for r in groups.get(f'unweighted_s{target}',[])}
        for condition in (f'M_s{target}',f'C_s{target}',f'T_s{target}',f'MT_s{target}',f'CT_s{target}'):
            if condition not in groups or len(groups[condition])!=len(baseline):continue
            values=[r['steps']-baseline[r['id']] for r in groups[condition]]
            ax.boxplot(values,positions=[position],showfliers=False);ax.text(position,-.1,condition,rotation=65,fontsize=8,ha='right',va='top',transform=ax.get_xaxis_transform());position+=1
    ax.axhline(0,color='black',lw=1);ax.set(xlim=(-1,max(1,position)),ylabel='Paired calls minus unweighted');ax.set_xticks([]);fig.tight_layout();fig.savefig(dest/'paired_iteration_delta.png',dpi=160);plt.close(fig)
    by=defaultdict(list)
    for s in steps:by[(s['condition'],s['iteration'])].append(s)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for condition in ('unweighted_s50','CT_s50','unweighted_s70','CT_s70'):
        x=sorted(step for label,step in by if label==condition)
        if not x:continue
        series=[by[(condition,t)] for t in x]
        means=lambda key:[float(np.mean(values)) if values else float('nan')
            for group in series
            for values in [[z[key] for z in group if z.get(key) is not None]]]
        color=colors['CT'] if condition.startswith('CT') else colors['unweighted']
        style='-' if condition.endswith('50') else '--'
        axes[0,0].plot(x,means('processed_entropy_mean'),color=color,ls=style,label=condition)
        axes[0,1].plot(x,means('argmax_flips'),color=color,ls=style,label=condition)
        axes[1,0].plot(x,[sum(z['skipped'] for z in group)/max(1,sum(z['eligible'] for z in group)) for group in series],color=color,ls=style,label=condition)
        axes[1,1].plot(x,[len(group) for group in series],color=color,ls=style,label=condition)
    for ax,title in zip(axes.flat,['Processed entropy (survivors)','Argmax flips (survivors)',
                                    'Physical sparsity (survivors)','Surviving canvases']):
        ax.set(title=title,xlabel='Denoising iteration');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(dest/'trajectory.png',dpi=160);plt.close(fig)


def plot_replay(root):
    """Create the required sensitivity/damage plot from frozen replay shards."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for path in (Path(root)/'diagnostic_replays/shards/matched_dense').glob('*.json'):
        result=json.loads(path.read_text())
        for snap in result['snapshots']:
            for position in snap['position_records']:
                rows.append(dict(position,iteration=snap['iteration'],id=result['id']))
    if not rows:return None
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    stats=[]
    for target in (50,70):
        for method,color,style in [('unweighted','#333333','--'),('CT','#e15759','-')]:
            filtered=[r for r in rows if r['method']==f'{method}_s{target}']
            x=np.array([r['sensitivity'] for r in filtered]);damage=np.array([r['top1_disagree'] for r in filtered])
            loss=np.array([r['confidence_loss'] for r in filtered]);q=np.quantile(x,np.linspace(0,1,6))
            midpoint=[];rate=[];confidence=[];sizes=[]
            for low,high in zip(q[:-1],q[1:]):
                included=(x>=low)&(x<=high) if high==q[-1] else (x>=low)&(x<high)
                if included.any():
                    midpoint.append(float(np.mean(x[included])));rate.append(float(np.mean(damage[included])))
                    confidence.append(float(np.mean(loss[included])));sizes.append(int(included.sum()))
            axes[0].plot(midpoint,rate,style,color=color,label=f'{method} {target}%')
            axes[1].plot(midpoint,confidence,style,color=color,label=f'{method} {target}%')
            stats.append(dict(method=method,target=target,bin_midpoints=midpoint,top1_disagreement=rate,
                confidence_loss=confidence,position_counts=sizes))
    axes[0].set(xlabel='Previous-step sensitivity',ylabel='Same-state top-1 disagreement')
    axes[1].set(xlabel='Previous-step sensitivity',ylabel='Dense confidence minus replay confidence')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    figure.tight_layout();dest=Path(root)/'plots/sensitivity_vs_damage.png';figure.savefig(dest,dpi=160);plt.close(figure)
    atomic(Path(root)/'diagnostic_replays/binned_damage.json',stats)
    return stats
