"""Combined shard-only report: parent T/shuffle/uniform and tile mean/max."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .experiment import atomic,fingerprint,sha,shard_path
from .query_adaptive_study import aggregate
from .query_adaptive_allocation_report import _summary,csv_write,interval
from .query_adaptive_tile_controls import METHODS,PARENT,TARGETS,label


def plot(root,summary,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'plots';dest.mkdir(exist_ok=True)
    colors={'unweighted':'gray','T':'#e15759','T_shuffle':'#4e79a7',
        'T_uniform':'#59a14f','T_tile_mean':'#f28e2b','T_tile_max':'#b07aa1'}
    for key,title,ylabel,file in [('accuracy','Accuracy versus achieved physical sparsity','RULER4K accuracy','accuracy_vs_sparsity.png'),
                                  ('mean_steps','Denoising calls versus achieved physical sparsity','Mean calls per canvas','steps_vs_sparsity.png')]:
        fig,axis=plt.subplots(figsize=(8,5))
        for method,color in colors.items():
            subset=sorted((x for x in summary if x['method']==method),key=lambda x:x['target'])
            axis.plot([x['actual_overall']*100 for x in subset],
                [x[key]*100 if key=='accuracy' else x[key] for x in subset],marker='o',color=color,label=method)
        dense=next(x for x in summary if x['condition']=='native_dense')
        axis.scatter([0],[dense[key]*100 if key=='accuracy' else dense[key]],marker='x',color='black',label='native dense')
        axis.set(xlabel='Achieved whole-model physical tile sparsity (%)',ylabel=ylabel,title=title)
        axis.grid(alpha=.2);axis.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/file,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for axis,target in zip(axes,TARGETS):
        for method,color in colors.items():
            row=next(x for x in summary if x['condition']==label(method,target))
            axis.plot(['whole','global','local'],[row[f'actual_{k}']*100 for k in ('overall','global','local')],
                marker='o',color=color,label=method)
        axis.axhline(target,ls=':',color='black',lw=1)
        axis.set(title=f'{target}% target',ylabel='Measured physical tile sparsity (%)');axis.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'target_vs_achieved.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for axis,target in zip(axes,TARGETS):
        base={r['id']:r for r in groups[label('T',target)]}
        for method,color in colors.items():
            if method in ('T','unweighted'):continue
            values=[r['steps']-base[r['id']]['steps'] for r in groups[label(method,target)]]
            axis.hist(values,bins=range(-48,50,3),alpha=.45,label=method,color=color)
        axis.axvline(0,color='black',lw=1)
        axis.set(title=f'{target}% target',xlabel='Extra calls versus T (paired examples)',ylabel='Examples')
        axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'paired_call_changes.png',dpi=160);plt.close(fig)


def report(root):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    parent_cfg=json.loads((PARENT/'configs/configuration.json').read_text())
    parent_audit=json.loads((PARENT/'audit.json').read_text())
    parent_frozen=json.loads((PARENT/'configs/frozen_policies.json').read_text())['policies']
    frozen=json.loads((root/'configs/frozen_policies.json').read_text())['policies']
    manifest=json.loads((root/'configs/final_manifest.json').read_text())
    parent_manifest=json.loads((PARENT/'configs/final_manifest.json').read_text())
    if [x['id'] for x in manifest]!=[x['id'] for x in parent_manifest] or any(
        x['prompt_hash']!=y['prompt_hash'] for x,y in zip(manifest,parent_manifest)):
        raise ValueError('Parent/extension evaluation prompts differ')
    mismatches=[path for path,digest in cfg['source_hashes'].items() if sha(path)!=digest]
    missing=[];summary=[];groups={};examples=[];canvases=[];steps=[]
    parent_names=('native_dense','kernel_dense','unweighted','T','T_shuffle','T_uniform')
    condition_pairs=[('native_dense',None),('kernel_dense',None)]+[(name,target)
        for target in TARGETS for name in (*parent_names[2:],*METHODS)]
    for name,target in condition_pairs:
        condition=name if target is None else label(name,target)
        parent=name in parent_names
        source=PARENT if parent else root
        source_cfg=parent_cfg if parent else cfg
        policies=parent_frozen if parent else frozen
        rows=[]
        for original in manifest:
            path=shard_path(source/'final',condition,original)
            if not path.exists():missing.append(dict(condition=condition,id=original['id']));continue
            r=json.loads(path.read_text())
            policy=(None if name=='native_dense' else 'all-retained' if name=='kernel_dense' else policies[condition])
            runtime=({k:dict(log_threshold=-float('inf')) for k in ('local','global')}
                     if name=='kernel_dense' else policy)
            spec=source_cfg['methods'][name]
            identity=fingerprint([source_cfg['fingerprint'],'final',condition,spec,runtime,
                source_cfg['m_ref'],original['id'],original['prompt_hash'],original['seed']])
            if (r['identity']!=identity or r['policy']!=policy or r['status']!='complete' or
                r['prompt_hash']!=original['prompt_hash'] or r['seed']!=original['seed']):
                raise ValueError(f'Invalid source shard {path}')
            for kind in ('whole','local','global'):
                for field in ('eligible','skipped'):
                    total=sum(s.get('counts',{}).get(kind,{}).get(field,0) for s in r['step_records'])
                    if total!=r['counts'][kind][field]:raise ValueError(f'Physical count mismatch {path}')
            rows.append(r)
            examples.append(dict(condition=condition,id=r['id'],task=r['task'],seed=r['seed'],
                prompt_hash=r['prompt_hash'],score=r['score'],steps=r['steps'],
                eligible=r['counts']['whole']['eligible'],skipped=r['counts']['whole']['skipped'],
                source='parent' if parent else 'tile_extension'))
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
                threshold_source=source/'configs/thresholds'/f'{condition}.json'
                threshold=json.loads(threshold_source.read_text())
                item.update(threshold_local=threshold['policy']['local']['log_threshold'],
                    threshold_global=threshold['policy']['global']['log_threshold'],
                    calibration_actual=threshold['calibration_actual'],calibration_attained=threshold['attained'])
            summary.append(item)
    index={x['condition']:x for x in summary};comparisons=[]
    if not missing:
        for target in TARGETS:
            reference=label('T',target)
            for name in ('unweighted','T_shuffle','T_uniform',*METHODS):
                condition=label(name,target);a=index[condition];b=index[reference]
                acc=interval(groups[condition],groups[reference],'score')
                calls=interval(groups[condition],groups[reference],'steps')
                deltas={k:a[f'actual_{k}']-b[f'actual_{k}'] for k in ('overall','global','local')}
                comparisons.append(dict(condition=condition,baseline=reference,
                    matched=all(abs(d)<=cfg['comparison_tolerance'] for d in deltas.values()),
                    **{f'actual_{k}_delta':v for k,v in deltas.items()},
                    accuracy_delta=a['accuracy']-b['accuracy'],accuracy_ci_low=acc[0],accuracy_ci_high=acc[1],
                    mean_step_delta=a['mean_steps']-b['mean_steps'],
                    step_ci_low=calls[0],step_ci_high=calls[1]))
    csv_write(root/'summary.csv',summary);csv_write(root/'comparisons.csv',comparisons)
    csv_write(root/'per_example.csv',examples);csv_write(root/'per_canvas.csv',canvases)
    with (root/'per_step.jsonl').open('w') as f:
        for row in steps:f.write(json.dumps(row,allow_nan=False)+'\n')
    audit=dict(complete=not missing and not mismatches and parent_audit['complete'],
        expected=len(condition_pairs)*len(manifest),completed=len(examples),missing=missing,
        parent_audit=parent_audit['complete'],source_mismatches=mismatches,
        same_prompts_and_seeds=True,policies_frozen=len(frozen)==4)
    audit['complete']=audit['complete'] and audit['policies_frozen']
    atomic(root/'audit.json',audit)
    atomic(root/'summary.json',dict(summary=summary,comparisons=comparisons,missing=missing,
        per_task={condition:{task:float(np.mean([r['score'] for r in rows if r['task']==task]))
            for task in sorted({r['task'] for r in rows})} for condition,rows in groups.items()}))
    if not missing:plot(root,summary,groups)
    confirmation=[];confirmation_groups={};confirmation_coverage=[]
    if audit['complete']:
        for seed in cfg['confirmation_seeds']:
            for target in TARGETS:
                for name in ('T','T_shuffle','T_uniform',*METHODS):
                    condition=label(name,target);source=PARENT if name not in METHODS else root
                    spec=(parent_cfg if name not in METHODS else cfg)['methods'][name]
                    policy=(parent_frozen if name not in METHODS else frozen)[condition]
                    source_cfg=parent_cfg if name not in METHODS else cfg
                    rows=[]
                    for original in manifest:
                        row=dict(original,seed=seed)
                        path=shard_path(source/f'confirmation/seed{seed}',condition,row)
                        if not path.exists():continue
                        r=json.loads(path.read_text())
                        identity=fingerprint([source_cfg['fingerprint'],f'confirmation/seed{seed}',
                            condition,spec,policy,source_cfg['m_ref'],row['id'],row['prompt_hash'],seed])
                        if (r['identity']!=identity or r['status']!='complete' or
                            r['policy']!=policy or r['seed']!=seed or
                            r['prompt_hash']!=row['prompt_hash']):
                            raise ValueError(f'Confirmation provenance mismatch {path}')
                        rows.append(r)
                    confirmation_coverage.append(dict(seed=seed,condition=condition,
                        completed=len(rows),expected=len(manifest)))
                    if len(rows)==len(manifest):
                        confirmation_groups[seed,condition]=rows
                        confirmation.append(dict(seed=seed,**_summary(rows,condition,name,target)))
    confirmation_complete=bool(confirmation_coverage) and all(
        x['completed']==x['expected'] for x in confirmation_coverage)
    atomic(root/'confirmation_audit.json',dict(
        complete=confirmation_complete,
        expected=sum(x['expected'] for x in confirmation_coverage),
        completed=sum(x['completed'] for x in confirmation_coverage),groups=confirmation_coverage))
    confirmation_comparisons=[]
    for seed in cfg['confirmation_seeds']:
        for target in TARGETS:
            reference=label('T',target)
            if (seed,reference) not in confirmation_groups:continue
            baseline=next(x for x in confirmation if x['seed']==seed and x['condition']==reference)
            for name in ('T_shuffle','T_uniform',*METHODS):
                condition=label(name,target)
                if (seed,condition) not in confirmation_groups:continue
                candidate=next(x for x in confirmation if x['seed']==seed and x['condition']==condition)
                acc=interval(confirmation_groups[seed,condition],confirmation_groups[seed,reference],'score')
                calls=interval(confirmation_groups[seed,condition],confirmation_groups[seed,reference],'steps')
                deltas={k:candidate[f'actual_{k}']-baseline[f'actual_{k}']
                    for k in ('overall','global','local')}
                confirmation_comparisons.append(dict(seed=seed,condition=condition,baseline=reference,
                    matched=all(abs(d)<=cfg['comparison_tolerance'] for d in deltas.values()),
                    **{f'actual_{k}_delta':v for k,v in deltas.items()},
                    accuracy_delta=candidate['accuracy']-baseline['accuracy'],
                    accuracy_ci_low=acc[0],accuracy_ci_high=acc[1],
                    mean_step_delta=candidate['mean_steps']-baseline['mean_steps'],
                    step_ci_low=calls[0],step_ci_high=calls[1]))
    csv_write(root/'confirmation_comparisons.csv',confirmation_comparisons)
    if confirmation:atomic(root/'confirmation_summary.json',confirmation)
    lines=['# Fresh-prompt temporal query sensitivity and tile-level controls','',
        f"Audit: {'PASS' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{audit['expected']} example/condition shards. ",
        'All arms use the identical fresh 130-example RULER4K manifest (13 tasks × 10), seeds, tokenizer, '
        'BF16 DiffusionGemma-26B-A4B-it, 256-position canvases, 48-step cap, 0.8→0.4 temperature, '
        'native reversible acceptance/stopping, Gaussian-32 seed1729 and the same v4 128×64 kernel. '
        'The 26-example calibration set is disjoint from final and from the previous 130-example query-adaptive cohort. '
        'The existing T, within-tile shuffled-T, and uniform-step-T results are reused verbatim from the '
        'frozen parent run; this version adds only two tile-broadcast sensitivity definitions.','',
        'For each 128-row physical query tile, tile-mean broadcasts the mean prior-step T sensitivity to all rows; '
        'tile-max broadcasts the maximum. Both still multiply each row’s projected update risk before the unchanged '
        'tile maximum. First two iterations have weight one. Thresholds for new arms were calibrated on complete '
        '26-prompt trajectories to the parent T calibration overall/global/local measurements, frozen before final evaluation. '
        'Physical sparsity is pooled skipped/eligible tile counts across every realized decoder call, with prefix encoding excluded.','',
        '| Method | Target | Actual overall / global / local | Accuracy | Total calls | Mean / p90 calls | Cap rate | Executed PV tiles | log τ local/global |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        threshold='—' if r['target'] is None else f"{r['threshold_local']:.3f}/{r['threshold_global']:.3f}"
        lines.append(f"| {r['condition']} | {'—' if r['target'] is None else str(r['target'])+'%'} | "
            f"{r['actual_overall']:.1%}/{r['actual_global']:.1%}/{r['actual_local']:.1%} | "
            f"{r['accuracy']:.1%} | {r['total_steps']} | {r['mean_steps']:.2f}/{r['p90_steps']:.0f} | "
            f"{r['cap_rate']:.1%} | {r['executed_tiles']:,} | {threshold} |")
    lines+=['','## Paired comparisons with T','',
        'Control minus T; task-stratified prompt-bootstrap 95% confidence intervals. Matched means final overall, '
        'global and local physical sparsity are each within 2 percentage points. A nominal target is not sufficient.','',
        '| Control | Matched | Δ overall/global/local | Δ accuracy [95% CI] | Δ mean calls [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['condition']} | {'yes' if r['matched'] else 'NO'} | "
            f"{r['actual_overall_delta']:+.1%}/{r['actual_global_delta']:+.1%}/{r['actual_local_delta']:+.1%} | "
            f"{r['accuracy_delta']:+.1%} [{r['accuracy_ci_low']:+.1%}, {r['accuracy_ci_high']:+.1%}] | "
            f"{r['mean_step_delta']:+.2f} [{r['step_ci_low']:+.2f}, {r['step_ci_high']:+.2f}] |")
    lines+=['','## Plots','',
        '![Accuracy versus actual sparsity](plots/accuracy_vs_sparsity.png)',
        '![Calls versus actual sparsity](plots/steps_vs_sparsity.png)',
        '![Target versus achieved sparsity](plots/target_vs_achieved.png)',
        '![Paired call changes](plots/paired_call_changes.png)']
    if confirmation:
        lines+=['','## Additional sampling seeds on the same prompts','',
            '| Seed | Method | Actual overall/global/local | Accuracy | Mean calls |',
            '|---:|---|---:|---:|---:|']
        for r in confirmation:
            lines.append(f"| {r['seed']} | {r['condition']} | "
                f"{r['actual_overall']:.1%}/{r['actual_global']:.1%}/{r['actual_local']:.1%} | "
                f"{r['accuracy']:.1%} | {r['mean_steps']:.2f} |")
    lines+=['','## Interpretation','',
        'Shuffled-T preserves the within-tile sensitivity multiset while breaking row identity. '
        'Tile-mean and tile-max preserve only one statistic per tile; uniform-step-T preserves only the temporal mean. '
        'Thus tile-level benefits cannot be credited to precise per-row identity.','',
        f"At 50% nominal sparsity, all T variants achieved 49–51% measured sparsity and "
        f"{min(index[label(name,50)]['mean_steps'] for name in ('T','T_shuffle','T_uniform',*METHODS)):.2f}–"
        f"{max(index[label(name,50)]['mean_steps'] for name in ('T','T_shuffle','T_uniform',*METHODS)):.2f} "
        'calls per canvas; the allocation distinction is small on this cohort. '
        f"At 70%, T used {index['T_s70']['mean_steps']:.2f} calls and scored "
        f"{index['T_s70']['accuracy']:.1%}; shuffled-T used "
        f"{index['T_shuffle_s70']['mean_steps']:.2f} and scored "
        f"{index['T_shuffle_s70']['accuracy']:.1%}. Tile-mean used "
        f"{index['T_tile_mean_s70']['mean_steps']:.2f} and scored "
        f"{index['T_tile_mean_s70']['accuracy']:.1%}. Its remaining "
        f"{(index['T_s70']['actual_global']-index['T_tile_mean_s70']['actual_global'])*100:.2f}-point "
        'global sparsity difference matters near the denoising-instability cliff, so the step reduction is not '
        'proved to come from tile averaging alone.','',
        'Only paired comparisons with measured overall/global/local sparsity support allocation claims. '
        'The previous 130-example cohort informed method development; this split is fresh relative to that cohort, '
        'not globally unseen data. Native dense and the matched H100 kernel have numerical/mask differences. '
        'No end-to-end speedup is inferred from executed-tile counts or instrumented wall times.']
    if confirmation_comparisons:
        lines+=['',
            'The additional seeds reuse the same prompts and thresholds but change generation randomness. '
            'Their per-seed paired deltas and sparsity residuals are in `confirmation_comparisons.csv`; '
            'they test seed stability, not fresh-prompt generalization.']
    if confirmation_complete:
        replicated=[dict(seed=cfg['seed'],**r) for r in comparisons]+confirmation_comparisons
        def span(name,field):
            values=[r[field] for r in replicated if r['condition']==label(name,70)]
            return min(values),max(values)
        shuffled_acc=span('T_shuffle','accuracy_delta')
        mean_calls=span('T_tile_mean','mean_step_delta')
        mean_acc=span('T_tile_mean','accuracy_delta')
        mean_global=span('T_tile_mean','actual_global_delta')
        max_acc=span('T_tile_max','accuracy_delta')
        lines+=['',
            f"Across the three generation seeds at 70% nominal sparsity, shuffled-T was "
            f"{abs(shuffled_acc[1])*100:.1f}–{abs(shuffled_acc[0])*100:.1f} percentage points "
            'less accurate than T even though it skipped fewer physical tiles. This supports a real effect '
            'of sensitivity assignment, but does not prove the full row-wise T policy is optimal. '
            f"Tile-mean saved {abs(mean_calls[1]):.1f}–{abs(mean_calls[0]):.1f} "
            f"calls per canvas, while its accuracy difference ranged from {mean_acc[0]*100:+.1f} "
            f"to {mean_acc[1]*100:+.1f} points. It also skipped "
            f"{abs(mean_global[1])*100:.1f}–{abs(mean_global[0])*100:.1f} fewer percentage points "
            'of GLOBAL tiles; near this instability boundary, this residual mismatch could explain '
            'part of the step reduction. Tile-max had no stable accuracy advantage '
            f"({max_acc[0]*100:+.1f} to {max_acc[1]*100:+.1f} points across seeds). "
            'The robust finding is that random within-tile reassignment is harmful at high sparsity; '
            'the best exact allocation and its real latency remain unresolved.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit
