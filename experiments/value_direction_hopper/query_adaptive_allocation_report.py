"""Shard-only audit and report for the fresh temporal-allocation controls."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import atomic, fingerprint, sha, shard_path
from .query_adaptive_study import aggregate
from .query_adaptive_allocation import METHODS, TARGETS, conditions, label


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def interval(left,right,key,*,seed=2718,draws=4000):
    a={x['id']:x for x in left};b={x['id']:x for x in right}
    if a.keys()!=b.keys() or len(a)!=130:raise ValueError('Unpaired full-set comparison')
    by_task=defaultdict(list)
    for identifier in sorted(a):
        if a[identifier]['task']!=b[identifier]['task']:raise ValueError('Task mismatch')
        by_task[a[identifier]['task']].append(a[identifier][key]-b[identifier][key])
    rng=np.random.default_rng(seed);draw=[]
    for _ in range(draws):
        draw.append(np.mean([np.mean(rng.choice(values,len(values),replace=True))
            for values in by_task.values()]))
    return [float(x) for x in np.quantile(draw,[.025,.975])]


def _summary(rows,condition,name,target):
    physical,counts=aggregate(rows)
    by_task=defaultdict(list)
    for r in rows:by_task[r['task']].append(r['score'])
    steps=np.array([r['steps'] for r in rows])
    return dict(condition=condition,method=name,target=target,n=len(rows),
        actual_overall=physical['whole'],actual_global=physical['global'],actual_local=physical['local'],
        accuracy=float(np.mean([np.mean(v) for v in by_task.values()])),
        total_steps=int(steps.sum()),mean_steps=float(steps.mean()),median_steps=float(np.median(steps)),
        p90_steps=float(np.quantile(steps,.9)),p95_steps=float(np.quantile(steps,.95)),
        cap_rate=float(np.mean(steps>=48)),eligible_tiles=counts['whole']['eligible'],
        skipped_tiles=counts['whole']['skipped'],executed_tiles=counts['whole']['eligible']-counts['whole']['skipped'])


def plots(root,summary,groups,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'plots';dest.mkdir(exist_ok=True)
    styles={'unweighted':('gray','o'),'T':('#e15759','o'),
        'T_shuffle':('#4e79a7','s'),'T_uniform':('#59a14f','^')}
    for field,title,ylabel,out in [('accuracy','Accuracy versus actual physical sparsity','RULER4K accuracy','accuracy_vs_sparsity.png'),
                                    ('mean_steps','Denoising steps versus actual physical sparsity','Mean calls/canvas','steps_vs_sparsity.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for method,(color,marker) in styles.items():
            rows=[s for s in summary if s['method']==method]
            ax.plot([r['actual_overall']*100 for r in rows],[r[field]*100 if field=='accuracy' else r[field] for r in rows],
                color=color,marker=marker,label=method)
        for name in ('native_dense','kernel_dense'):
            row=next(s for s in summary if s['condition']==name)
            ax.scatter([0],[row[field]*100 if field=='accuracy' else row[field]],
                label=name,marker='x' if name=='native_dense' else '+')
        ax.set(xlabel='Actual overall physical sparsity (%)',ylabel=ylabel,title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/out,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for axis,target in zip(axes,TARGETS):
        for method,(color,_) in styles.items():
            row=next(s for s in summary if s['condition']==label(method,target))
            axis.plot(['overall','global','local'],[row['actual_overall']*100,row['actual_global']*100,
                row['actual_local']*100],color=color,marker='o',label=method)
        axis.axhline(target,color='black',ls=':',lw=1)
        axis.set(title=f'Target {target}%',ylabel='Achieved physical sparsity (%)')
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'target_vs_achieved.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for axis,target in zip(axes,TARGETS):
        base={r['id']:r for r in groups[label('T',target)]}
        for method,color in (('T_shuffle','#4e79a7'),('T_uniform','#59a14f')):
            rows=groups[label(method,target)]
            values=[r['steps']-base[r['id']]['steps'] for r in rows]
            axis.hist(values,bins=range(-48,50,3),alpha=.5,label=method,color=color)
        axis.axvline(0,color='black',lw=1);axis.set(title=f'{target}% target',xlabel='Calls minus T (paired prompt)',ylabel='Examples')
        axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'paired_iteration_deltas.png',dpi=160);plt.close(fig)


def report(root):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    manifest=json.loads((root/'configs/final_manifest.json').read_text())
    calibration=json.loads((root/'configs/calibration_manifest.json').read_text())
    previous=json.loads((Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3'/'configs/final_manifest.json').read_text())
    source_mismatches=[path for path,digest in cfg['source_hashes'].items() if sha(path)!=digest]
    disjoint=all(not ({x[k] for x in manifest}&{x[k] for x in calibration}) and
        not ({x[k] for x in manifest}&{x[k] for x in previous}) for k in ('id','source_id','prompt_hash'))
    frozen_path=root/'configs/frozen_policies.json'
    frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {'policies':{}}
    groups={};missing=[];summary=[];examples=[];canvases=[];steps=[]
    for name,target in conditions():
        condition=label(name,target);rows=[]
        for original in manifest:
            path=shard_path(root/'final',condition,original)
            if not path.exists():missing.append(dict(condition=condition,id=original['id']));continue
            r=json.loads(path.read_text())
            policy=(None if name=='native_dense' else 'all-retained' if name=='kernel_dense' else
                frozen['policies'].get(condition))
            runtime_policy=({k:dict(log_threshold=-float('inf')) for k in ('local','global')}
                if name=='kernel_dense' else policy)
            identity=fingerprint([cfg['fingerprint'],'final',condition,cfg['methods'][name],runtime_policy,
                cfg['m_ref'],original['id'],original['prompt_hash'],original['seed']])
            if (r['status']!='complete' or r['identity']!=identity or r['policy']!=policy or
                r['prompt_hash']!=original['prompt_hash'] or r['seed']!=original['seed']):
                raise ValueError(f'Invalid shard/provenance {path}')
            for kind in ('whole','local','global'):
                for field in ('eligible','skipped'):
                    total=sum(s.get('counts',{}).get(kind,{}).get(field,0) for s in r['step_records'])
                    if total!=r['counts'][kind][field]:raise ValueError(f'Per-step tile mismatch {path}')
            if r['canvas']['counts']!=r['counts'] or r['canvas']['iterations']!=r['steps']:
                raise ValueError(f'Canvas mismatch {path}')
            rows.append(r)
            examples.append(dict(condition=condition,id=r['id'],task=r['task'],seed=r['seed'],
                prompt_hash=r['prompt_hash'],score=r['score'],iterations=r['steps'],
                eligible=r['counts']['whole']['eligible'],skipped=r['counts']['whole']['skipped'],
                output_tokens=len(r['completion_tokens'])))
            canvases.append(dict(condition=condition,id=r['id'],task=r['task'],
                **{k:v for k,v in r['canvas'].items() if k!='counts'},
                eligible=r['counts']['whole']['eligible'],skipped=r['counts']['whole']['skipped'],
                global_eligible=r['counts']['global']['eligible'],global_skipped=r['counts']['global']['skipped'],
                local_eligible=r['counts']['local']['eligible'],local_skipped=r['counts']['local']['skipped']))
            for step in r['step_records']:
                counts=step['counts'];steps.append(dict(condition=condition,id=r['id'],task=r['task'],
                    **{k:v for k,v in step.items() if k!='counts'},
                    eligible=counts['whole']['eligible'],skipped=counts['whole']['skipped'],
                    global_eligible=counts['global']['eligible'],global_skipped=counts['global']['skipped'],
                    local_eligible=counts['local']['eligible'],local_skipped=counts['local']['skipped']))
        groups[condition]=rows
        if rows:
            item=_summary(rows,condition,name,target)
            if target is not None:
                selection=json.loads((root/'configs/thresholds'/f'{condition}.json').read_text())
                item.update(threshold_local=selection['policy']['local']['log_threshold'],
                    threshold_global=selection['policy']['global']['log_threshold'],
                    calibration_goal=selection['goal'],calibration_actual=selection['calibration_actual'],
                    calibration_attained=selection['attained'])
            summary.append(item)
    index={r['condition']:r for r in summary};comparisons=[]
    if not missing:
        for target in TARGETS:
            reference=label('T',target);base=index[reference]
            for method in ('unweighted','T_shuffle','T_uniform'):
                condition=label(method,target);other=index[condition]
                matched=all(abs(other[k]-base[k])<=cfg['comparison_tolerance'] for k in
                    ('actual_overall','actual_global','actual_local'))
                acc=interval(groups[condition],groups[reference],'score')
                calls=interval(groups[condition],groups[reference],'steps')
                comparisons.append(dict(condition=condition,baseline=reference,
                    matched=matched,actual_overall_delta=other['actual_overall']-base['actual_overall'],
                    actual_global_delta=other['actual_global']-base['actual_global'],
                    actual_local_delta=other['actual_local']-base['actual_local'],
                    accuracy_delta=other['accuracy']-base['accuracy'],accuracy_ci_low=acc[0],accuracy_ci_high=acc[1],
                    mean_step_delta=other['mean_steps']-base['mean_steps'],
                    step_ci_low=calls[0],step_ci_high=calls[1]))
    csv_write(root/'summary.csv',summary);csv_write(root/'per_example.csv',examples)
    csv_write(root/'per_canvas.csv',canvases);csv_write(root/'comparisons.csv',comparisons)
    with (root/'per_step.jsonl').open('w') as f:
        for row in steps:f.write(json.dumps(row,allow_nan=False)+'\n')
    atomic(root/'summary.json',dict(summary=summary,comparisons=comparisons,missing=missing,
        per_task={condition:{task:float(np.mean([r['score'] for r in rows if r['task']==task]))
            for task in sorted({r['task'] for r in rows})} for condition,rows in groups.items()}))
    audit=dict(complete=not missing and not source_mismatches and disjoint,
        expected=len(manifest)*len(conditions()),completed=len(examples),missing=missing,
        disjoint_calibration_and_previous=disjoint,source_mismatches=source_mismatches,
        threshold_provenance=not missing and len(frozen['policies'])==len(METHODS)*len(TARGETS),
        task_counts=dict(Counter(r['task'] for r in manifest)))
    audit['complete']=audit['complete'] and audit['threshold_provenance']
    atomic(root/'audit.json',audit)
    if summary:plots(root,summary,groups,comparisons)
    confirmation=[]
    if audit['complete']:
        for seed in cfg['confirmation_seeds']:
            for target in TARGETS:
                for name in ('T','T_shuffle','T_uniform'):
                    condition=label(name,target);rows=[]
                    for original in manifest:
                        row=dict(original,seed=seed);path=shard_path(root/f'confirmation/seed{seed}',condition,row)
                        if path.exists():
                            result=json.loads(path.read_text())
                            expected=fingerprint([cfg['fingerprint'],f'confirmation/seed{seed}',condition,
                                cfg['methods'][name],frozen['policies'][condition],cfg['m_ref'],
                                row['id'],row['prompt_hash'],seed])
                            if result['identity']!=expected:raise ValueError('Confirmation provenance mismatch')
                            rows.append(result)
                    if len(rows)==len(manifest):
                        confirmation.append(dict(seed=seed,**_summary(rows,condition,name,target)))
    if confirmation:atomic(root/'confirmation_summary.json',confirmation)
    lines=['# Temporal query allocation: T versus shuffled-T and uniform-step-T','',
        f"Primary audit: {len(examples)}/{audit['expected']} complete; {'PASS' if audit['complete'] else 'INCOMPLETE'}. ",
        'The final 130 prompts are 10 per official RULER4K task, selected from ranks 15–24 of the pinned 50-per-task pool. '
        'Calibration uses ranks 13–14, and development smoke uses rank 25. They are disjoint from each other and '
        'from the earlier v16 and query-adaptive splits. The pool itself was already generated and may have appeared in other research; '
        'this is fresh relative to the current query-adaptive study, not a claim of globally unseen data.','',
        'DiffusionGemma-26B-A4B-it BF16, Gaussian-32 seed 1729, FP32 routing, 128×64 physical tiles, '
        'native 256-position canvas, reversible acceptance, 48-step maximum and unchanged 0.8→0.4 temperature/stopping. '
        'T uses its own previous deterministic top-1 flip EMA (beta=3, gamma=0.5), with first and second iteration weights one. '
        'Shuffled-T permutes weights within each 128-query physical tile using a separate RNG stream; '
        'uniform-step-T replaces all positions with their current canvas-step mean. No decoder rule or sparse kernel was changed.','',
        'Policies are calibrated on all 26 disjoint calibration trajectories: T toward nominal 50%/70%; '
        'unweighted, shuffled-T and uniform-step-T toward T’s measured calibration overall/global/local sparsity. '
        'Selection uses at most 10 full-generation policy points. Frozen thresholds are in `configs/thresholds/`. '
        'Final measurements never select a threshold. Physical sparsity pools skipped/eligible tile counts across all '
        'executed denoising calls; extra calls increase total eligible work. Prefix encoding is excluded.','',
        '| Method | Target | Actual overall / global / local | Accuracy | Total calls | Mean / p90 calls | Cap rate | Executed PV tiles | log tau L / G |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        threshold='—' if r['target'] is None else f"{r['threshold_local']:.3f} / {r['threshold_global']:.3f}"
        lines.append(f"| {r['condition']} | {'—' if r['target'] is None else str(r['target'])+'%'} | "
            f"{r['actual_overall']:.1%} / {r['actual_global']:.1%} / {r['actual_local']:.1%} | "
            f"{r['accuracy']:.1%} | {r['total_steps']} | {r['mean_steps']:.2f} / {r['p90_steps']:.0f} | "
            f"{r['cap_rate']:.1%} | {r['executed_tiles']:,} | {threshold} |")
    lines+=['','## Matched physical-sparsity comparisons','',
        'Differences below are control minus T, with task-stratified paired prompt-bootstrap 95% intervals. '
        '“Matched” requires actual overall, global and local sparsity each within 2 percentage points; '
        'a target label alone does not establish matching.','',
        '| Control vs T | Matched | Δ overall / global / local sparsity | Δ accuracy [95% CI] | Δ mean calls [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['condition']} | {'yes' if r['matched'] else 'NO'} | "
            f"{r['actual_overall_delta']:+.1%} / {r['actual_global_delta']:+.1%} / "
            f"{r['actual_local_delta']:+.1%} | {r['accuracy_delta']:+.1%} "
            f"[{r['accuracy_ci_low']:+.1%}, {r['accuracy_ci_high']:+.1%}] | "
            f"{r['mean_step_delta']:+.2f} [{r['step_ci_low']:+.2f}, {r['step_ci_high']:+.2f}] |")
    lines+=['','Native dense uses original SDPA, while kernel dense and all sparse arms use the matched H100 kernel and '
        'its local-mask convention. Native/kernel numerical differences must not be attributed to sparsity. '
        'These are still development-benchmark results, and a CI crossing zero is not proof of equivalence.','',
        '## Plots','',
        '![Accuracy versus actual sparsity](plots/accuracy_vs_sparsity.png)',
        '![Steps versus actual sparsity](plots/steps_vs_sparsity.png)',
        '![Target versus achieved sparsity](plots/target_vs_achieved.png)',
        '![Paired iteration changes](plots/paired_iteration_deltas.png)']
    if confirmation:
        lines+=['','## Two additional sampling seeds on the same fresh prompts','',
            '| Seed | Method | Actual overall / global / local | Accuracy | Mean calls |',
            '|---:|---|---:|---:|---:|']
        for r in confirmation:
            lines.append(f"| {r['seed']} | {r['condition']} | {r['actual_overall']:.1%} / "
                f"{r['actual_global']:.1%} / {r['actual_local']:.1%} | "
                f"{r['accuracy']:.1%} | {r['mean_steps']:.2f} |")
    lines+=['','## Interpretation','',
        'Only compare policies when actual overall/global/local allocations are sufficiently close. '
        'Shuffled-T preserves each tile’s sensitivity values but breaks which query owns them; '
        'uniform-step-T preserves the step’s average sensitivity while removing spatial allocation. '
        'A T advantage over uniform alone could come from nonuniformity without correct query identity. '
        'A T advantage over both controls at matched sparsity is stronger evidence for query-specific allocation, '
        'but remaining numerical and repeated-development-benchmark limits still apply.','',
        'End-to-end latency is not measured by these instrumented final runs; prior-cohort timing must not be '
        'presented as speedup on this cohort.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit
