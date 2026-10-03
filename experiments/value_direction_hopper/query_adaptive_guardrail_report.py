"""Shard-only audit, paired analysis and report for the guardrail study."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .experiment import atomic, fingerprint, sha, shard_path
from . import query_adaptive_guardrail as core


def _csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(field for row in rows for field in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def _load_group(root,stage,condition,manifest,cfg,policy,method,target):
    rows=[];missing=[]
    for prompt in manifest:
        path=shard_path(root/stage,condition,prompt)
        if not path.exists():missing.append(prompt['id']);continue
        row=json.loads(path.read_text())
        expected=fingerprint([cfg['fingerprint'],stage,condition,cfg['methods'][method],
            policy,cfg['m_ref'],prompt['id'],prompt['prompt_hash'],prompt['seed']])
        if row['status']!='complete' or row['identity']!=expected or row['policy']!=policy or (
            row['id'],row['prompt_hash'],row['seed'])!=(
            prompt['id'],prompt['prompt_hash'],prompt['seed']):
            raise ValueError(f'Invalid shard provenance: {path}')
        if row.get('routing_path') and sha(row['routing_path'])!=row['routing_sha256']:
            raise ValueError(f'Corrupt routing counts: {path}')
        if row['canvas']['iterations']!=row['steps'] or row['canvas']['counts']!=row['counts']:
            raise ValueError(f'Canvas/call mismatch: {path}')
        for kind in ('whole','local','global'):
            for field in ('eligible','skipped'):
                if sum(step['counts'][kind][field] for step in row['step_records'])!=row['counts'][kind][field]:
                    raise ValueError(f'Per-step physical count mismatch: {path}')
        for step in row['step_records']:
            expected_phase='early' if step['iteration']<=2 else 'late'
            if step.get('threshold_phase')!=expected_phase:
                raise ValueError(f'Wrong threshold phase: {path}')
            if step['iteration']<=2 and step.get('weight_active'):
                raise ValueError(f'Early sensitivity was not unit weight: {path}')
        rows.append(row)
    return rows,missing


def _summarize(rows,method,target,source='new'):
    info=core.profile(rows)
    tasks=defaultdict(list)
    for row in rows:tasks[row['task']].append(row['score'])
    early_tails={}
    for iteration in (1,2):
        rates=[]
        for row in rows:
            step=next((s for s in row['step_records'] if s['iteration']==iteration),None)
            if step and step['counts']['whole']['eligible']:
                value=step['counts']['whole']
                rates.append(value['skipped']/value['eligible'])
        early_tails[f'step{iteration}_p95_sparsity']=float(np.quantile(rates,.95)) if rates else None
    return dict(condition=core.label(method,target) if target is not None else method,
        method=method,target=target,source=source,n=info['n'],
        actual_overall=info['overall']['whole'],actual_global=info['overall']['global'],
        actual_local=info['overall']['local'],accuracy=float(np.mean([np.mean(v) for v in tasks.values()])),
        total_calls=info['total_steps'],mean_calls=info['mean_steps'],
        median_calls=info['median_steps'],p90_calls=info['p90_steps'],
        p95_calls=info['p95_steps'],cap_rate=info['cap_count']/len(rows),
        eligible_tiles=None if method=='native_dense' else info['counts']['whole']['eligible'],
        skipped_tiles=None if method=='native_dense' else info['counts']['whole']['skipped'],
        executed_tiles=None if method=='native_dense' else info['executed_tiles'],
        step1_accepted_mean=info['accepted_mean']['step1'],
        step2_accepted_mean=info['accepted_mean']['step2'],
        step1_sparsity=info['phase_sparsity']['step1']['whole'],
        step1_global_sparsity=info['phase_sparsity']['step1']['global'],
        step1_local_sparsity=info['phase_sparsity']['step1']['local'],
        step2_sparsity=info['phase_sparsity']['step2']['whole'],
        step2_global_sparsity=info['phase_sparsity']['step2']['global'],
        step2_local_sparsity=info['phase_sparsity']['step2']['local'],
        later_sparsity=info['phase_sparsity']['later']['whole'],**early_tails)


def _bootstrap(left,right,field,draws=3000):
    a={r['id']:r for r in left};b={r['id']:r for r in right}
    if a.keys()!=b.keys() or len(a)!=130:raise ValueError('Unmatched prompt comparison')
    tasks=defaultdict(list)
    for key in sorted(a):
        if a[key]['task']!=b[key]['task']:raise ValueError('Task mismatch')
        tasks[a[key]['task']].append(a[key][field]-b[key][field])
    rng=np.random.default_rng(12977)
    distribution=[float(np.mean([np.mean(rng.choice(v,len(v),replace=True))
        for v in tasks.values()])) for _ in range(draws)]
    return [float(x) for x in np.quantile(distribution,[.025,.975])]


def _token_agreement(rows,dense_tokens):
    matches=positions=exact=0
    for row in rows:
        left=row['completion_tokens'];right=dense_tokens[row['id']]
        matches+=sum(a==b for a,b in zip(left,right))
        positions+=max(len(left),len(right))
        exact+=int(left==right)
    return (matches/positions if positions else 1.,exact/len(rows))


def _plots(root,summaries,groups,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=root/'plots';destination.mkdir(exist_ok=True)
    palette={'T':'#e15759','C':'#4e79a7','M':'#59a14f','unweighted':'#777777',
        'T_uniform':'#af7aa1','T_tile_mean':'#f28e2b','T_shuffle':'#76b7b2'}
    for field,ylabel,name in (('accuracy','RULER4K score (%)','accuracy_vs_sparsity.png'),
        ('token_agreement_vs_kernel_dense','Token agreement with matched dense (%)',
         'agreement_vs_sparsity.png'),
        ('mean_calls','Mean denoising calls/canvas','calls_vs_sparsity.png'),
        ('executed_tiles','Total executed eligible PV tiles','work_vs_sparsity.png')):
        fig,ax=plt.subplots(figsize=(9,5.5))
        percentage=field in ('accuracy','token_agreement_vs_kernel_dense')
        for method,color in palette.items():
            values=sorted((s for s in summaries if s['method']==method
                           and s['condition'] not in ('archived_T_s70','prior_fresh_T_s70',
                                                      'phase_ablation_v1')),
                          key=lambda s:s['actual_overall'])
            if not values:continue
            ax.plot([s['actual_overall']*100 for s in values],
                [s[field]*100 if percentage else s[field] for s in values],
                marker='o',label=method,color=color)
            for s in values:
                ax.annotate(f"{s['target']}%",(s['actual_overall']*100,
                    s[field]*100 if percentage else s[field]),fontsize=7)
        for method,marker in (('native_dense','x'),('kernel_dense','+')):
            value=next((s for s in summaries if s['method']==method),None)
            if value and not (field=='executed_tiles' and method=='native_dense'):
                ax.scatter([0],[value[field]*100 if percentage else value[field]],
                marker=marker,s=75,label=method)
        archived=next((s for s in summaries if s['condition']=='archived_T_s70'),None)
        if archived:
            ax.scatter([archived['actual_overall']*100],
                [archived[field]*100 if percentage else archived[field]],
                marker='*',s=100,color='#e15759',label='archived T70 reproduction')
        previous=next((s for s in summaries if s['condition']=='prior_fresh_T_s70'),None)
        if previous:
            ax.scatter([previous['actual_overall']*100],
                [previous[field]*100 if percentage else previous[field]],
                marker='D',s=45,color='#9c755f',label='previous failed T70 calibration')
        ablation=next((s for s in summaries if s['condition']=='phase_ablation_v1'),None)
        if ablation:
            ax.scatter([ablation['actual_overall']*100],
                [ablation[field]*100 if percentage else ablation[field]],
                marker='s',s=55,color='#edc948',label='early-only v1 phase ablation')
        ax.set(xlabel='Actual pooled physical sparsity (%)',ylabel=ylabel)
        ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
        fig.savefig(destination/name,dpi=160)
        if field=='mean_calls':
            ax.set_ylim(0,12)
            fig.tight_layout();fig.savefig(destination/'calls_vs_sparsity_zoom.png',dpi=160)
        plt.close(fig)
    fig,axis=plt.subplots(figsize=(9,5.5))
    for method,color in palette.items():
        values=[s for s in summaries if s['method']==method and
                s['condition'] not in ('archived_T_s70','prior_fresh_T_s70',
                                       'phase_ablation_v1')]
        for index,item in enumerate(values):
            axis.scatter(item['mean_calls'],item['accuracy']*100,color=color,
                         label=method if index==0 else None)
            axis.annotate(f"{item['actual_overall']:.0%}",
                (item['mean_calls'],item['accuracy']*100),fontsize=7)
    for method,marker in (('native_dense','x'),('kernel_dense','+')):
        item=next((s for s in summaries if s['method']==method),None)
        if item:
            axis.scatter(item['mean_calls'],item['accuracy']*100,
                marker=marker,s=90,label=method,color='black')
    axis.set(xlabel='Mean denoising calls per canvas',ylabel='RULER4K score (%)',
             title='Accuracy versus trajectory length; labels are actual sparsity')
    axis.grid(alpha=.2)
    if axis.get_legend_handles_labels()[0]:axis.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(destination/'accuracy_vs_calls.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for summary in summaries:
        if summary['method'] in ('native_dense','kernel_dense') or summary['condition'] in (
            'archived_T_s70','prior_fresh_T_s70','phase_ablation_v1'):continue
        ax.plot(['call 1','call 2','later','overall'],
            [summary['step1_sparsity']*100,summary['step2_sparsity']*100,
             summary['later_sparsity']*100,summary['actual_overall']*100],
            marker='o',label=summary['condition'])
    ax.set(ylabel='Pooled physical sparsity (%)',title='Trajectory allocation')
    ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7);fig.tight_layout()
    fig.savefig(destination/'phase_sparsity.png',dpi=160);plt.close(fig)
    for target in core.TARGETS:
        selected=[(name,groups[core.label(name,target)]) for name in core.METHODS
                  if core.label(name,target) in groups]
        if not selected:continue
        fig,ax=plt.subplots(figsize=(9,5))
        ax.boxplot([[r['steps'] for r in rows] for _,rows in selected],
            tick_labels=[name for name,_ in selected],showfliers=True)
        ax.set(ylabel='Denoising calls per canvas',title=f'{target}% target: per-canvas distribution')
        ax.grid(axis='y',alpha=.2);fig.tight_layout()
        fig.savefig(destination/f'calls_distribution_s{target}.png',dpi=160);plt.close(fig)
    if comparisons:
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for axis,target in zip(axes,core.TARGETS):
            subset=[c for c in comparisons if c['target']==target]
            if subset:
                axis.barh([c['method'] for c in subset],[c['mean_call_delta'] for c in subset])
            axis.axvline(0,color='black',lw=1)
            axis.set(title=f'{target}%',xlabel='Mean calls minus T on paired examples')
        fig.tight_layout();fig.savefig(destination/'paired_call_deltas.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for axis,target in zip(axes,core.TARGETS):
        anchor=groups.get(core.label('unweighted',target))
        if not anchor:continue
        baseline={row['id']:row['steps'] for row in anchor}
        for method,color in (('T','#e15759'),('C','#4e79a7'),('M','#59a14f'),
                             ('T_uniform','#af7aa1'),('T_tile_mean','#f28e2b')):
            rows=groups.get(core.label(method,target))
            if not rows:continue
            changes=[row['steps']-baseline[row['id']] for row in rows]
            axis.hist(changes,bins=range(-48,50,4),alpha=.36,color=color,
                      label=method)
        axis.axvline(0,color='black',lw=1)
        axis.set(title=f'{target}%: calls minus unweighted',xlabel='Paired per-canvas call difference')
        axis.grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(destination/'paired_call_histograms.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for axis,target in zip(axes,core.TARGETS):
        for method,color in palette.items():
            rows=groups.get(core.label(method,target))
            if not rows:continue
            maximum=max(row['steps'] for row in rows)
            axis.step(range(1,maximum+1),
                [sum(row['steps']>=step for row in rows) for step in range(1,maximum+1)],
                where='post',color=color,label=method)
        axis.set(title=f'{target}% target',xlabel='Denoising call',
                 ylabel='Canvases still active')
        axis.grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(destination/'surviving_canvases_vs_call.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for column,target in enumerate(core.TARGETS):
        for row_index,(field,title) in enumerate((
            ('processed_entropy_mean','Processed entropy'),
            ('argmax_flips','Argmax flips'))):
            axis=axes[row_index,column]
            for method,color in palette.items():
                rows=groups.get(core.label(method,target))
                if not rows:continue
                by=defaultdict(list)
                for example in rows:
                    for step in example['step_records']:
                        if step['iteration']<=8 and step.get(field) is not None:
                            by[step['iteration']].append(step[field])
                if by:
                    axis.plot(sorted(by),[np.mean(by[k]) for k in sorted(by)],
                        marker='.',color=color,label=method)
            axis.set(title=f'{target}%: {title}; surviving canvases only',
                     xlabel='Denoising call',ylabel=title)
            axis.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if not handles:handles,labels=axes[0,1].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=7)
    fig.tight_layout();fig.savefig(destination/'entropy_churn_by_call.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for axis,target in zip(axes,core.TARGETS):
        for method,color in palette.items():
            rows=groups.get(core.label(method,target))
            if not rows:continue
            by=defaultdict(list)
            for example in rows:
                for step in example['step_records']:
                    if step['iteration']<=8 and step.get('sensitivity_mean') is not None:
                        by[step['iteration']].append(step['sensitivity_mean'])
            if by:
                axis.plot(sorted(by),[np.mean(by[k]) for k in sorted(by)],
                          marker='.',color=color,label=method)
        axis.set(title=f'{target}% target; surviving canvases only',
                 xlabel='Denoising call',ylabel='Mean routing sensitivity')
        axis.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=7)
    fig.tight_layout();fig.savefig(destination/'sensitivity_by_call.png',dpi=160)
    plt.close(fig)


def report(root):
    root=Path(root)
    cfg,manifests=core.prepare(root)
    final=manifests['final'];groups={};missing=[];summary=[]
    parent_manifest=json.loads((core.PARENT/'configs/final_manifest.json').read_text())
    parent_cfg=json.loads((core.PARENT/'configs/configuration.json').read_text())
    if cfg['parent_fingerprint']!=parent_cfg['fingerprint']:
        raise ValueError('Reused dense parent configuration fingerprint changed')
    if [(r['id'],r['prompt_hash'],r['seed']) for r in final] != [
        (r['id'],r['prompt_hash'],r['seed']) for r in parent_manifest]:
        raise ValueError('Parent dense controls are not prompt/seed matched')
    parent_audit=json.loads((core.PARENT/'audit.json').read_text())
    if not parent_audit['complete']:raise ValueError('Parent dense baseline audit incomplete')
    # Parent baselines are explicitly identified as reused and retain their
    # own execution provenance. They are not recast as new-kernel shards.
    for method in ('native_dense','kernel_dense'):
        rows=[]
        for prompt in final:
            path=shard_path(core.PARENT/'final',method,prompt)
            if not path.exists():raise ValueError(f'Missing audited parent control: {path}')
            row=json.loads(path.read_text())
            if (row['id'],row['prompt_hash'],row['seed']) != (
                prompt['id'],prompt['prompt_hash'],prompt['seed']):
                raise ValueError(f'Parent control mismatch: {path}')
            rows.append(row)
        groups[method]=rows
        summary.append(_summarize(rows,method,None,source='audited_parent'))
    parent_t_rows=[]
    for prompt in final:
        path=shard_path(core.PARENT/'final','T_s70',prompt)
        if not path.exists():raise ValueError(f'Missing audited parent T70 shard: {path}')
        row=json.loads(path.read_text())
        if (row['id'],row['prompt_hash'],row['seed'])!=(
            prompt['id'],prompt['prompt_hash'],prompt['seed']):
            raise ValueError(f'Parent T70 prompt mismatch: {path}')
        parent_t_rows.append(row)
    groups['prior_fresh_T_s70']=parent_t_rows
    prior_summary=_summarize(parent_t_rows,'T',70,source='audited_parent_failed_calibration')
    parent_policy=json.loads((core.PARENT/'configs/thresholds/T_s70.json').read_text())['policy']
    prior_summary.update(condition='prior_fresh_T_s70',
        log_tau_early_local=parent_policy['local']['log_threshold'],
        log_tau_early_global=parent_policy['global']['log_threshold'],
        log_tau_late_local=parent_policy['local']['log_threshold'],
        log_tau_late_global=parent_policy['global']['log_threshold'])
    summary.append(prior_summary)
    threshold_paths=sorted((root/'configs/thresholds').glob('*.json'))
    frozen={path.stem:json.loads(path.read_text()) for path in threshold_paths}
    for target in core.TARGETS:
        for method in core.METHODS:
            condition=core.label(method,target)
            policy=frozen.get(condition)
            if not policy or 'policy' not in policy:continue
            rows,gaps=_load_group(root,'final',condition,final,cfg,policy['policy'],method,target)
            if gaps:
                missing.extend(dict(condition=condition,id=identifier) for identifier in gaps)
                continue
            groups[condition]=rows
            item=_summarize(rows,method,target)
            item.update(log_tau_early_local=policy['policy']['early']['local']['log_threshold'],
                log_tau_early_global=policy['policy']['early']['global']['log_threshold'],
                log_tau_late_local=policy['policy']['late']['local']['log_threshold'],
                log_tau_late_global=policy['policy']['late']['global']['log_threshold'],
                calibration_status=policy['status'])
            summary.append(item)
    archived_policy=core.phase_policy(cfg['old_t70'])
    archived_rows,archived_missing=_load_group(root,'archived_reproduction','T_s70',
        final,cfg,archived_policy,'T',70)
    if not archived_missing:
        groups['archived_T_s70']=archived_rows
        item=_summarize(archived_rows,'T',70,source='predeclared_archived_thresholds')
        item.update(condition='archived_T_s70',
            log_tau_early_local=archived_policy['early']['local']['log_threshold'],
            log_tau_early_global=archived_policy['early']['global']['log_threshold'],
            log_tau_late_local=archived_policy['late']['local']['log_threshold'],
            log_tau_late_global=archived_policy['late']['global']['log_threshold'])
        summary.append(item)
    ablation_config_path=root/'configs/phase_ablation_v1.json'
    if ablation_config_path.exists():
        from . import query_adaptive_guardrail_ablation
        ablation_config=json.loads(ablation_config_path.read_text())
        if (ablation_config['core_fingerprint']!=cfg['fingerprint'] or
            ablation_config['runner_sha256']!=sha(Path(query_adaptive_guardrail_ablation.__file__)) or
            ablation_config['source_policy_sha256']!=sha(query_adaptive_guardrail_ablation.SOURCE)):
            raise ValueError('Phase ablation provenance changed')
        ablation_policy=ablation_config['policy']
        ablation_rows,ablation_missing=_load_group(root,'phase_ablation_v1','T_s70',
            final,cfg,ablation_policy,'T',70)
        if not ablation_missing:
            groups['phase_ablation_v1']=ablation_rows
            item=_summarize(ablation_rows,'T',70,source='predeclared_v1_phase_policy')
            item.update(condition='phase_ablation_v1',
                log_tau_early_local=ablation_policy['early']['local']['log_threshold'],
                log_tau_early_global=ablation_policy['early']['global']['log_threshold'],
                log_tau_late_local=ablation_policy['late']['local']['log_threshold'],
                log_tau_late_global=ablation_policy['late']['global']['log_threshold'])
            summary.append(item)
    dense_tokens={row['id']:row['completion_tokens'] for row in groups['kernel_dense']}
    for item in summary:
        agreement,exact=_token_agreement(groups[item['condition']],dense_tokens)
        item['token_agreement_vs_kernel_dense']=agreement
        item['sequence_exact_vs_kernel_dense']=exact
    index={s['condition']:s for s in summary};comparisons=[]
    early_parity=[]
    for target in core.TARGETS:
        anchor=core.label('T',target)
        if anchor not in groups:continue
        reference={row['id']:row for row in groups[anchor]}
        for method in core.METHODS:
            condition=core.label(method,target)
            if method=='T' or condition not in groups:continue
            for row in groups[condition]:
                original=reference[row['id']]
                if min(2,len(original['step_records']))!=min(2,len(row['step_records'])):
                    raise ValueError(f'Different early-phase call counts: {condition} {row["id"]}')
                for call_index in range(min(2,len(original['step_records']))):
                    a,b=original['step_records'][call_index],row['step_records'][call_index]
                    if (a['counts']!=b['counts'] or a['accepted']!=b['accepted'] or
                        a['renoised']!=b['renoised'] or
                        abs(a['processed_entropy_mean']-b['processed_entropy_mean'])>1.e-7):
                        raise ValueError(f'Shared early-phase execution mismatch: {condition} {row["id"]} call {call_index+1}')
            early_parity.append(condition)
    for target in core.TARGETS:
        anchor=core.label('T',target)
        if anchor not in groups:continue
        for method in core.METHODS:
            condition=core.label(method,target)
            if method=='T' or condition not in groups:continue
            item=index[condition];reference=index[anchor]
            delta={kind:item[f'actual_{kind}']-reference[f'actual_{kind}']
                   for kind in ('overall','global','local')}
            ci_acc=_bootstrap(groups[condition],groups[anchor],'score')
            ci_calls=_bootstrap(groups[condition],groups[anchor],'steps')
            comparisons.append(dict(condition=condition,method=method,target=target,
                matched=all(abs(x)<=.02 for x in delta.values()),
                sparsity_delta_overall=delta['overall'],
                sparsity_delta_global=delta['global'],
                sparsity_delta_local=delta['local'],
                accuracy_delta=item['accuracy']-reference['accuracy'],
                accuracy_ci_low=ci_acc[0],accuracy_ci_high=ci_acc[1],
                mean_call_delta=item['mean_calls']-reference['mean_calls'],
                mean_call_ci_low=ci_calls[0],mean_call_ci_high=ci_calls[1]))
    baseline_comparisons=[]
    for target in core.TARGETS:
        reference=core.label('unweighted',target)
        if reference not in groups:continue
        for method in ('C','M','T','T_uniform','T_tile_mean','T_shuffle'):
            condition=core.label(method,target)
            if condition not in groups:continue
            a,b=index[condition],index[reference]
            delta={kind:a[f'actual_{kind}']-b[f'actual_{kind}']
                   for kind in ('overall','global','local')}
            accuracy_ci=_bootstrap(groups[condition],groups[reference],'score')
            call_ci=_bootstrap(groups[condition],groups[reference],'steps')
            baseline_comparisons.append(dict(condition=condition,reference=reference,
                target=target,matched=all(abs(x)<=.02 for x in delta.values()),
                sparsity_delta_overall=delta['overall'],
                sparsity_delta_global=delta['global'],
                sparsity_delta_local=delta['local'],
                accuracy_delta=a['accuracy']-b['accuracy'],
                accuracy_ci_low=accuracy_ci[0],accuracy_ci_high=accuracy_ci[1],
                mean_call_delta=a['mean_calls']-b['mean_calls'],
                mean_call_ci_low=call_ci[0],mean_call_ci_high=call_ci[1],
                executed_tile_delta=a['executed_tiles']-b['executed_tiles']))
    reference_comparisons=[]
    if 'T_s70' in groups:
        for reference in ('prior_fresh_T_s70','archived_T_s70',
                          'phase_ablation_v1','kernel_dense'):
            if reference not in groups:continue
            acc=_bootstrap(groups['T_s70'],groups[reference],'score')
            calls=_bootstrap(groups['T_s70'],groups[reference],'steps')
            a=index['T_s70'];b=index[reference]
            reference_comparisons.append(dict(reference=reference,
                accuracy_delta=a['accuracy']-b['accuracy'],
                accuracy_ci_low=acc[0],accuracy_ci_high=acc[1],
                mean_call_delta=a['mean_calls']-b['mean_calls'],
                mean_call_ci_low=calls[0],mean_call_ci_high=calls[1],
                sparsity_delta_overall=a['actual_overall']-b['actual_overall'],
                sparsity_delta_global=a['actual_global']-b['actual_global'],
                sparsity_delta_local=a['actual_local']-b['actual_local']))
    confirmation=[]
    t70=frozen.get('T_s70')
    if t70 and 'policy' in t70:
        for seed in cfg['confirmation_seeds']:
            seeded=[dict(row,seed=seed) for row in final]
            rows,gaps=_load_group(root,f'confirmation/seed{seed}','T_s70',
                seeded,cfg,t70['policy'],'T',70)
            if not gaps:
                confirmation.append(dict(seed=seed,
                    **_summarize(rows,'T',70,source='predeclared_seed_confirmation')))
    examples=[];canvases=[];steps=[]
    per_task_rows=[]
    for condition,rows in groups.items():
        task_groups=defaultdict(list)
        for row in rows:task_groups[row['task']].append(row)
        for task,subset in sorted(task_groups.items()):
            eligible=sum(r['counts']['whole']['eligible'] for r in subset)
            skipped=sum(r['counts']['whole']['skipped'] for r in subset)
            per_task_rows.append(dict(condition=condition,task=task,n=len(subset),
                score=float(np.mean([r['score'] for r in subset])),
                mean_calls=float(np.mean([r['steps'] for r in subset])),
                physical_sparsity=skipped/eligible if eligible else None,
                eligible_tiles=eligible if eligible else None,
                skipped_tiles=skipped if eligible else None))
        for row in rows:
            counts=row['counts'];examples.append(dict(condition=condition,id=row['id'],
                task=row['task'],seed=row['seed'],score=row['score'],calls=row['steps'],
                eligible=counts['whole']['eligible'],skipped=counts['whole']['skipped']))
            canvases.append(dict(condition=condition,id=row['id'],task=row['task'],
                **{k:v for k,v in row['canvas'].items() if k!='counts'},
                eligible=counts['whole']['eligible'],skipped=counts['whole']['skipped']))
            for step in row['step_records']:
                steps.append(dict(condition=condition,id=row['id'],task=row['task'],
                    **{k:v for k,v in step.items() if k!='counts'},
                    eligible=step['counts']['whole']['eligible'],
                    skipped=step['counts']['whole']['skipped'],
                    local_eligible=step['counts']['local']['eligible'],
                    local_skipped=step['counts']['local']['skipped'],
                    global_eligible=step['counts']['global']['eligible'],
                    global_skipped=step['counts']['global']['skipped']))
    _csv(root/'summary.csv',summary)
    _csv(root/'per_example.csv',examples)
    _csv(root/'per_canvas.csv',canvases)
    _csv(root/'per_task.csv',per_task_rows)
    _csv(root/'comparisons.csv',comparisons)
    _csv(root/'baseline_comparisons.csv',baseline_comparisons)
    _csv(root/'reference_comparisons.csv',reference_comparisons)
    layer_totals=defaultdict(Counter)
    routing_fields=('eligible','skipped','prefix_eligible','prefix_skipped',
        'canvas_eligible','canvas_skipped','boundary_eligible','boundary_skipped')
    for condition,rows in groups.items():
        for row in rows:
            path=row.get('routing_path')
            if not path:continue
            with gzip.open(path,'rt') as stream:
                routing=json.load(stream)
            for entry in routing:
                phase='step1' if entry['step']==0 else 'step2' if entry['step']==1 else 'later'
                key=(condition,int(entry['layer']),entry['attention_type'],phase)
                total=layer_totals[key]
                for field in routing_fields:
                    total[field]+=int(entry[field])
    layer_rows=[]
    for (condition,layer,kind,phase),counts in sorted(layer_totals.items()):
        layer_rows.append(dict(condition=condition,layer=layer,
            attention_type=kind,phase=phase,**{field:counts[field] for field in routing_fields},
            physical_sparsity=counts['skipped']/counts['eligible'] if counts['eligible'] else None))
    for condition,rows in groups.items():
        if condition=='native_dense':continue
        observed=sum(item['eligible'] for item in layer_rows if item['condition']==condition)
        expected=sum(row['counts']['whole']['eligible'] for row in rows)
        if observed!=expected:
            raise ValueError(f'Per-layer eligible tile reconciliation failed: {condition}')
    _csv(root/'per_layer_phase.csv',layer_rows)
    trajectory=defaultdict(list)
    for row in steps:
        trajectory[(row['condition'],row['iteration'])].append(row)
    trajectory_rows=[]
    for (condition,iteration),cohort in sorted(trajectory.items()):
        eligible=sum(row['eligible'] for row in cohort)
        skipped=sum(row['skipped'] for row in cohort)
        def mean(field):
            values=[row[field] for row in cohort if row.get(field) is not None]
            return float(np.mean(values)) if values else None
        trajectory_rows.append(dict(condition=condition,iteration=iteration,
            surviving_canvases=len(cohort),eligible=eligible,skipped=skipped,
            physical_sparsity=skipped/eligible if eligible else None,
            mean_accepted=mean('accepted'),mean_renoised=mean('renoised'),
            mean_processed_entropy=mean('processed_entropy_mean'),
            mean_argmax_flips=mean('argmax_flips'),
            mean_sensitivity=mean('sensitivity_mean')))
    _csv(root/'trajectory_aggregate.csv',trajectory_rows)
    with (root/'per_step.jsonl').open('w') as stream:
        for row in steps:stream.write(json.dumps(row,allow_nan=False)+'\n')
    atomic(root/'summary.json',dict(summary=summary,comparisons=comparisons,
        baseline_comparisons=baseline_comparisons,
        reference_comparisons=reference_comparisons,
        confirmation=confirmation,
        trajectory_aggregate=trajectory_rows,
        missing=missing,per_task={condition:{task:float(np.mean([
            r['score'] for r in rows if r['task']==task]))
            for task in sorted({r['task'] for r in rows})}
            for condition,rows in groups.items()}))
    source_mismatches=[path for path,digest in cfg['source_hashes'].items()
                       if sha(path)!=digest]
    smoke_path=root/'smoke.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    smoke_passed=bool(smoke.get('passed') and
        smoke.get('fingerprint')==cfg['fingerprint'])
    if not smoke_passed:source_mismatches.append('smoke_or_fingerprint')
    refinement_path=root/'configs/refinement_configuration.json'
    refinement=json.loads(refinement_path.read_text()) if refinement_path.exists() else None
    if refinement:
        from . import query_adaptive_guardrail_refine
        if (refinement['core_fingerprint']!=cfg['fingerprint'] or
            refinement['source_sha256']!=sha(Path(query_adaptive_guardrail_refine.__file__)) or
            refinement['v1_calibration_trace_sha256']!=sha(
                Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1'/'calibration_traces'/'T_s70.json')):
            source_mismatches.append('refinement_configuration')
    matrix_path=root/'configs/matrix_configuration.json'
    matrix=json.loads(matrix_path.read_text()) if matrix_path.exists() else None
    if matrix:
        from . import query_adaptive_guardrail_matrix
        if (matrix['core_fingerprint']!=cfg['fingerprint'] or
            matrix['source_sha256']!=sha(Path(query_adaptive_guardrail_matrix.__file__))):
            source_mismatches.append('matrix_configuration')
    timing_path=root/'timing_summary.json'
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else []
    timing_config_path=root/'configs/timing.json'
    if timing and timing_config_path.exists():
        from . import query_adaptive_guardrail_timing
        timing_config=json.loads(timing_config_path.read_text())
        if (timing_config['core_fingerprint']!=cfg['fingerprint'] or
            timing_config['source_sha256']!=sha(Path(query_adaptive_guardrail_timing.__file__))):
            source_mismatches.append('timing_configuration')
    expected=[core.label(method,target) for target in core.TARGETS
              for method in core.METHODS]
    incomplete=[condition for condition in expected if condition not in groups]
    target_misses=[name for name,value in frozen.items()
                  if value['status']!='attained']
    audit=dict(complete=not source_mismatches and not missing and not incomplete,
        source_mismatches=source_mismatches,missing=missing,incomplete=incomplete,
        report_source_sha256=sha(Path(__file__)),
        completed_conditions=sorted(groups),final_rows=len(final),
        task_counts=dict(Counter(r['task'] for r in final)),
        parent_dense_audit=parent_audit['complete'],
        smoke_passed=smoke_passed,
        timing_complete=bool(timing) and all(
            item['completion_or_call_mismatches']==0 for item in timing),
        shared_first_two_call_parity=early_parity,
        refinement_provenance=refinement['fingerprint'] if refinement else None,
        matrix_provenance=matrix['fingerprint'] if matrix else None,
        threshold_statuses={name:value['status'] for name,value in frozen.items()},
        target_misses=target_misses,
        archived_reproduction_complete=not archived_missing)
    atomic(root/'audit.json',audit)
    if len(summary)>2:_plots(root,summary,groups,comparisons)
    lines=['# Trajectory-guarded Gaussian-32 query-sensitivity comparison','',
        f"Audit: {'complete' if audit['complete'] else 'incomplete'}; "
        f"{sum(c in groups for c in expected)}/{len(expected)} sparse conditions complete.",'',
        'The exact candidate grids, ranking, guardrails, validation rule and all '
        'frozen log-thresholds are specified in [CALIBRATION_RULES.md]'
        '(CALIBRATION_RULES.md); [REPRODUCE.md](REPRODUCE.md) gives the run order.','',
        'All methods reuse DiffusionGemma-26B-A4B-it BF16, Gaussian-32 seed 1729, '
        '128×64 physical tiles, the exact 130 RULER4K prompts, 256-position canvas, '
        'native reversible acceptance, annealed temperature and 48-call maximum. '
        'Accepted positions can be renoised later and are not permanently committed. '
        'The final cohort has been examined in prior studies and is exploratory. '
        'Calibration and validation each use 26 new task-balanced prompts, disjoint '
        'from earlier recorded manifests and from these 130 prompts. '
        f'Model path: `{cfg["model"]}`; revision: `{cfg["revision"]}`.','',
        'The first two denoising calls use unit query weights and one common local/global '
        'threshold pair per target. Later calls use each method’s own previous-step '
        'sensitivity, with separate frozen local/global thresholds. Thresholds were '
        'selected on full calibration trajectories using pooled physical skipped/eligible '
        'counts, first-two-call ceilings, and call-count guardrails, then checked on '
        'disjoint validation prompts. Final scores did not select policies.','',
        'This shared early phase is deliberate: T has no observed top-1 flip '
        'until two predictions exist, and allowing C/M to act one call earlier '
        'would confound a matched first-two-call comparison. No method sees '
        'current-step logits before its attention calculation or performs a '
        'hidden extra forward pass.','',
        'The T anchor must match nominal whole/global/local pooled sparsity '
        'within ±2 points on calibration (±3 on validation). Other methods '
        'match T’s measured calibration whole/global/local rates within the '
        'same tolerances, so query allocation is compared at similar physical '
        'budgets. For 70%, calls 1–2 must each stay at '
        'or below 75% whole and 77% local/global skipped tiles; mean calls ≤8, '
        'p90 ≤12, and at most one 48-call cap among 26 prompts. For 50%, the '
        'corresponding ceilings are 55% whole, 57% local/global, mean ≤5.5, '
        'p90 ≤8, and zero caps. The validation score must not fall more than '
        '3 points below matched-kernel dense. Thresholds are frozen before '
        'the 130-prompt run. Sparsity pools skipped/eligible physical 128×64 '
        'tiles across all executed decoder calls, including valid prefix and '
        'canvas KV tiles; the separate dense prefix encoding is excluded.','',
        'For calls ≥3, C weights rows by prior processed confidence '
        '`1+3√(1−c)`, M by prior raw-logit margin '
        '`1+3m_ref/(m+m_ref)`, and T by a prior top-1-flip EMA '
        '`1+3u` with γ=0.5. T-uniform assigns the canvas-step mean to every '
        'query; T-tile-mean assigns each 128-query tile mean to its rows; '
        'T-shuffle permutes weights within each tile using an independent RNG. '
        'The kernel multiplies each query’s value-direction risk *before* '
        'the physical-tile maximum; a larger weight can only favor retaining '
        'a tile at a fixed state and threshold. Calibration selects just one '
        'late scalar threshold per local/global attention type, not per-query '
        'thresholds or learned weights. All controls share the same first-two-call '
        'thresholds and unchanged native decoder. '
        f'The frozen margin reference m_ref={cfg["m_ref"]:.5g} and Gaussian-32 '
        'projection seed 1729 are inherited from the prior audited study.','',
        'Native dense uses original SDPA; matched-kernel dense and sparse methods use the '
        'H100 kernel and its local-mask/numerical convention. These parent dense controls '
        'were reused only after prompt/seed and prior-audit checks.','',
        '| Method | Target | Policy gate | Actual O/G/L | Accuracy | Token agree | Mean/p90 calls | Cap | Call1/2 sparsity | Call1/2 accepted | Executed tiles | Early log τ L/G | Late log τ L/G |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if refinement:
        lines[10:10]=['Version 2 records a targeted refinement of the T70 late thresholds '
            'after v1 missed its validation p90/global allocation gates. The refinement '
            'grid was defined using only calibration and validation traces, never final '
            'answers; its source and v1 trace hashes are frozen in '
            '`configs/refinement_configuration.json`. Because the same 26 validation '
            'prompts informed v2, its validation pass is a development check, not an '
            'unbiased held-out confirmation. The previously examined final 130 are '
            'also exploratory.','']
    for item in summary:
        early='—' if item['target'] is None else f"{item['log_tau_early_local']:.3f}/{item['log_tau_early_global']:.3f}"
        late='—' if item['target'] is None else f"{item['log_tau_late_local']:.3f}/{item['log_tau_late_global']:.3f}"
        work='—' if item['method']=='native_dense' else f"{item['executed_tiles']:,}"
        lines.append(f"| {item['condition']} | {item['target'] or '—'} | "
            f"{item.get('calibration_status','reference')} | "
            f"{item['actual_overall']:.1%}/{item['actual_global']:.1%}/{item['actual_local']:.1%} | "
            f"{item['accuracy']:.1%} | {item['token_agreement_vs_kernel_dense']:.1%} | "
            f"{item['mean_calls']:.2f}/{item['p90_calls']:.1f} | "
            f"{item['cap_rate']:.1%} | {item['step1_sparsity']:.1%}/{item['step2_sparsity']:.1%} | "
            f"{item['step1_accepted_mean']:.1f}/{item['step2_accepted_mean']:.1f} | "
            f"{work} | {early} | {late} |")
    lines+=['','Token agreement compares generated token IDs positionally against '
        'matched-kernel dense over the full returned sequence. Positions after '
        'the first divergence still count; missing or extra positions are '
        'disagreements. Full-sequence exact-match rates are in `summary.csv`.']
    lines+=['','## Matched-T paired comparisons','',
        'Differences are method minus T on the same 130 prompts, with task-stratified '
        'paired bootstrap 95% intervals. Matched requires actual overall, global and '
        'local sparsity each within 2 percentage points. Intervals crossing zero are '
        'inconclusive, not evidence of equivalence. The intervals are exploratory '
        'and unadjusted for multiple method comparisons.','',
        '| Method | Target | Matched O/G/L? | Δ accuracy [95% CI] | Δ mean calls [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for item in comparisons:
        lines.append(f"| {item['method']} | {item['target']} | "
            f"{'yes' if item['matched'] else 'no'} | "
            f"{item['accuracy_delta']:+.1%} [{item['accuracy_ci_low']:+.1%}, {item['accuracy_ci_high']:+.1%}] | "
            f"{item['mean_call_delta']:+.2f} [{item['mean_call_ci_low']:+.2f}, {item['mean_call_ci_high']:+.2f}] |")
    if baseline_comparisons:
        lines+=['','## Query sensitivity versus unweighted Gaussian-32','',
            'Each difference is method minus the same-target unweighted router '
            'on paired prompts. Interpret only sparsity-matched rows as an '
            'allocation test; the executed-tile difference also reflects '
            'trajectory length.','',
            '| Method | Target | Matched O/G/L? | Δ accuracy [95% CI] | '
            'Δ mean calls [95% CI] | Δ executed tiles |',
            '|---|---:|---:|---:|---:|---:|']
        for item in baseline_comparisons:
            lines.append(f"| {item['condition']} | {item['target']} | "
                f"{'yes' if item['matched'] else 'no'} | "
                f"{item['accuracy_delta']:+.1%} "
                f"[{item['accuracy_ci_low']:+.1%}, {item['accuracy_ci_high']:+.1%}] | "
                f"{item['mean_call_delta']:+.2f} "
                f"[{item['mean_call_ci_low']:+.2f}, {item['mean_call_ci_high']:+.2f}] | "
                f"{item['executed_tile_delta']:+,} |")
    old_summary=json.loads((core.ARCHIVE/'summary.json').read_text())['summary']
    old_t=next(x for x in old_summary if x['condition']=='T_s70')
    parent_summary=json.loads((core.PARENT/'summary.json').read_text())['summary']
    fresh_t=next(x for x in parent_summary if x['condition']=='T_s70')
    lines+=['','## Pre-existing T70 references','',
        '| Cohort/policy | Actual sparsity | Accuracy | Mean calls |',
        '|---|---:|---:|---:|',
        f"| Earlier 130 prompts, earlier T70 policy | {old_t['sparsity']:.1%} | {old_t['accuracy']:.1%} | {old_t['mean_iterations']:.2f} |",
        f"| Current 130 prompts, previous unconstrained calibration | {fresh_t['actual_overall']:.1%} | {fresh_t['accuracy']:.1%} | {fresh_t['mean_steps']:.2f} |"]
    archived=next((x for x in summary if x['condition']=='archived_T_s70'),None)
    if archived:
        lines.append(f"| Current 130 prompts, exact earlier T70 thresholds | {archived['actual_overall']:.1%} | {archived['accuracy']:.1%} | {archived['mean_calls']:.2f} |")
    current=index.get('T_s70')
    if current:
        lines.append(f"| Current 130 prompts, new guarded T70 policy | {current['actual_overall']:.1%} | {current['accuracy']:.1%} | {current['mean_calls']:.2f} |")
    lines+=['','The earlier cohort differs from the current 130 prompts; its accuracy '
        'must not be treated as a paired improvement target. The previous '
        'unconstrained calibration and the guarded policy are prompt matched.']
    if reference_comparisons:
        lines+=['','## Paired guarded-T70 differences on the current 130 prompts','',
            'Differences are guarded T70 minus the named reference; task-stratified '
            'paired prompt-bootstrap 95% intervals. Differences in actual local/global '
            'sparsity remain possible confounders.','',
            '| Reference | Δ accuracy [95% CI] | Δ mean calls [95% CI] | Δ O/G/L sparsity |',
            '|---|---:|---:|---:|']
        for item in reference_comparisons:
            lines.append(f"| {item['reference']} | {item['accuracy_delta']:+.1%} "
                f"[{item['accuracy_ci_low']:+.1%}, {item['accuracy_ci_high']:+.1%}] | "
                f"{item['mean_call_delta']:+.2f} "
                f"[{item['mean_call_ci_low']:+.2f}, {item['mean_call_ci_high']:+.2f}] | "
                f"{item['sparsity_delta_overall']:+.1%}/"
                f"{item['sparsity_delta_global']:+.1%}/"
                f"{item['sparsity_delta_local']:+.1%} |")
    if 'phase_ablation_v1' in groups:
        lines+=['','The `phase_ablation_v1` arm uses the *same* safer first-two-call '
            'thresholds as guarded T70 but the older late pair. Comparing it to '
            '`archived_T_s70` isolates the early-phase change at those frozen '
            'thresholds; comparing guarded T70 to it isolates the late-global '
            'threshold adjustment. Full trajectories and actual sparsity can still '
            'differ nonlinearly, so neither comparison is a controlled same-state '
            'attention-output intervention.']
    if confirmation:
        lines+=['','## Predeclared generation-seed confirmations','',
            'These repeat the same 130 prompts and frozen T70 thresholds with '
            'different sampling seeds; they do not constitute new held-out prompts. '
            'Dense was not rerun at these seeds, so they do not establish '
            'seed-matched accuracy differences against dense.','',
            '| Seed | Actual O/G/L | Accuracy | Mean/p90 calls |',
            '|---:|---:|---:|---:|']
        for item in confirmation:
            lines.append(f"| {item['seed']} | {item['actual_overall']:.1%}/"
                f"{item['actual_global']:.1%}/{item['actual_local']:.1%} | "
                f"{item['accuracy']:.1%} | {item['mean_calls']:.2f}/"
                f"{item['p90_calls']:.1f} |")
    if timing:
        lines+=['','## Measured H100 latency','',
            'Untraced end-to-end timing uses complete generation, one warmup '
            'per condition, matched prompts, and rotating/interleaved order. '
            'Decoder/prefix/policy-update event times are measured in separate '
            'profile passes and must not be subtracted from the E2E runs. '
            'The policy event estimate excludes host-side dispatch overhead.','',
            '| Condition | Mean E2E seconds | Speed ratio vs native dense | '
            'Decoder / prefix / policy seconds (profiled) | Output/call mismatches |',
            '|---|---:|---:|---:|---:|']
        for item in timing:
            lines.append(f"| {item['condition']} | {item['e2e_mean_seconds']:.3f} | "
                f"{item['speed_ratio_vs_native_dense']:.3f}× | "
                f"{item['decoder_mean_profiled_seconds']:.3f} / "
                f"{item['prefix_mean_profiled_seconds']:.3f} / "
                f"{item['policy_update_mean_profiled_seconds']:.3f} | "
                f"{item['completion_or_call_mismatches']} |")
        if any(item['completion_or_call_mismatches'] for item in timing):
            lines+=['','Timing output/step mismatches require investigation; '
                'do not interpret these speed ratios as validated.']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(8,4.5))
        axis.bar([item['condition'] for item in timing],
                 [item['e2e_mean_seconds'] for item in timing])
        axis.set(ylabel='Mean complete-generation seconds per prompt',
                 title='Measured interleaved H100 end-to-end latency')
        axis.tick_params(axis='x',rotation=30)
        fig.tight_layout();fig.savefig(root/'plots/latency_comparison.png',dpi=160)
        plt.close(fig)
    lines+=['','## Plots','',
        '![Accuracy versus actual sparsity](plots/accuracy_vs_sparsity.png)',
        '![Token agreement versus actual sparsity](plots/agreement_vs_sparsity.png)',
        '![Mean calls versus actual sparsity](plots/calls_vs_sparsity.png)',
        '![Calls versus sparsity, short-trajectory zoom](plots/calls_vs_sparsity_zoom.png)',
        '![Accuracy versus calls](plots/accuracy_vs_calls.png)',
        '![Executed physical attention work](plots/work_vs_sparsity.png)',
        '![Paired call-count changes](plots/paired_call_histograms.png)',
        '![First-two-call allocation](plots/phase_sparsity.png)',
        '![Early entropy and churn](plots/entropy_churn_by_call.png)','',
        '![Mean query sensitivity by call](plots/sensitivity_by_call.png)','',
        '![Surviving canvases by call](plots/surviving_canvases_vs_call.png)','',
        'The plotted late-call means include only surviving canvases; this selection '
        'must be considered when interpreting recovery. Physical sparsity is not a '
        'runtime speedup measurement.']
    if timing:
        lines+=['','![Measured complete-generation latency](plots/latency_comparison.png)']
    else:
        lines+=['','No end-to-end speedup is claimed without a separate '
            'uninstrumented, matched timing study.']
    if incomplete:
        lines+=['','Incomplete or unattainable conditions: '+', '.join(incomplete)+'.']
    if target_misses:
        lines+=['','Policies evaluated despite missing calibration/validation guardrails: '
            +', '.join(target_misses)+'. These are measured operating points, '
            'not claimed target-sparsity successes.']
    if not archived_missing:
        lines+=['','`archived_T_s70` reruns the previously published T70 log-threshold '
            'pair on the current 130-prompt cohort. It is a reproduction reference, '
            'not a threshold chosen on final scores.']
    lines+=['','Per-task scores and calls are in `per_task.csv`; '
        '`per_layer_phase.csv` pools physical counts by layer, attention type '
        'and early/later phase, including prefix/canvas/boundary tile categories. '
        'Paired outputs, per-canvas counts, per-call diagnostics and calibration '
        'traces remain machine-readable in this result bundle.']
    if all(name in index for name in ('T_s50','T_s70','unweighted_s70',
                                      'T_uniform_s70','T_tile_mean_s70',
                                      'T_shuffle_s70','kernel_dense')):
        t50,t70=(index[f'T_s{target}'] for target in (50,70))
        baseline=index['kernel_dense']; unweighted=index['unweighted_s70']
        uniform=index['T_uniform_s70']; tile=index['T_tile_mean_s70']
        shuffled=index['T_shuffle_s70']
        lines+=['','## Supported conclusions and remaining limitations','',
            f"At the 50% operating point, query-specific T skips {t50['actual_overall']:.1%} "
            f"of eligible physical tiles and averages {t50['mean_calls']:.2f} calls/canvas "
            f"versus {baseline['mean_calls']:.2f} for matched-kernel dense. "
            'Its score and call count must be compared with all same-target controls above; '
            'a near-dense trajectory here is not evidence that 70% is equally safe.','',
            f"At 70%, T skips {t70['actual_overall']:.1%} overall "
            f"({t70['actual_global']:.1%} global; {t70['actual_local']:.1%} local), "
            f"scores {t70['accuracy']:.1%}, and averages {t70['mean_calls']:.2f} "
            f"calls, versus {unweighted['mean_calls']:.2f} calls and "
            f"{unweighted['accuracy']:.1%} for the matched-budget unweighted router. "
            f"The uniform-step and tile-mean controls take {uniform['mean_calls']:.2f} "
            f"and {tile['mean_calls']:.2f} calls; within-tile shuffled T takes "
            f"{shuffled['mean_calls']:.2f}. Their measured overall/global/local "
            'rates, paired intervals and calibration-gate statuses appear above. '
            'The shuffled control is the cleanest test of position-specific '
            'allocation because it preserves the within-tile weight distribution.','',
            f"The guarded T trajectory is substantially shorter than the previous "
            f"unconstrained calibration, but still exceeds matched dense by "
            f"{t70['mean_calls']-baseline['mean_calls']:.2f} calls/canvas. "
            'Thus this experiment mitigates, but does not eliminate, denoising-step '
            'inflation at ~70% physical sparsity. A 50% result should not be '
            'extrapolated to 70%.']
        if 'phase_ablation_v1' in index and 'archived_T_s70' in index:
            a=index['phase_ablation_v1']; old=index['archived_T_s70']
            lines+=['',f"The fixed-threshold phase ablation (safer first two calls, "
                f"old late thresholds) yields {a['actual_overall']:.1%} sparsity, "
                f"{a['accuracy']:.1%} score and {a['mean_calls']:.2f} calls, "
                f"compared with {old['mean_calls']:.2f} calls for the old uniform "
                f"phase policy and {t70['mean_calls']:.2f} for the new guarded "
                'policy. Early-phase protection alone is therefore not the full '
                'explanation; late local/global threshold allocation matters too.']
        if timing:
            best=next((item for item in timing if item['condition']=='T_s70'),None)
            if best and best['completion_or_call_mismatches']==0:
                lines+=['',f"The independently measured T70 end-to-end speed ratio "
                    f"against native dense is {best['speed_ratio_vs_native_dense']:.3f}× "
                    f"({1/best['speed_ratio_vs_native_dense']:.2f}× longer). "
                    'This is the complete-generation measurement, not a sparsity-derived '
                    'or attention-kernel-only estimate. The shorter guarded trajectory '
                    'does not yet produce a positive end-to-end speedup.']
        else:
            lines+=['','End-to-end runtime is pending; no speedup conclusion follows '
                'from executed-tile counts alone.']
        lines+=['','These 130 prompts have been used to develop the method, the '
            'validation cohort informed a v2 refinement, and the exploratory '
            'bootstrap intervals are unadjusted for multiple comparisons. '
            'A fresh task-balanced prompt set with frozen thresholds is needed '
            'before claiming generalization. Also, final per-example agreement '
            'and call counts do not by themselves prove which position caused a '
            'specific routing error.','',
            'Recommended next experiment: freeze the guarded T70 and its '
            'within-tile shuffled and unweighted controls, then evaluate on '
            'new RULER4K prompts and another long-context benchmark with '
            'matched whole/global/local physical sparsity and untraced timing. '
            'If the ~2.5 extra calls persist, use same-state query diagnostics '
            'to determine which early token positions still force extra '
            'iterations before changing the router or decoder.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=core.ROOT)
    args=parser.parse_args()
    print(json.dumps(report(args.root),indent=2))
