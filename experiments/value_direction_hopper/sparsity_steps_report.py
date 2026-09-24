"""Raw-only, canvas-weighted iteration/accuracy analysis of the frozen sweep."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .experiment import atomic, sha, shard_path
from .report import paired_ci, agreement


def write_csv(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def bootstrap(values,tasks,seed=20260921):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for value,task in zip(values,tasks):groups[task].append(value)
    draws=[]
    for group in groups.values():
        v=np.asarray(group);draws.append(v[rng.integers(0,len(v),(10000,len(v)))].mean(1))
    return [float(x) for x in np.quantile(np.mean(draws,axis=0),[.025,.975])]


def report(root):
    root=Path(root);config=json.loads((root/'configuration.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    snapshots=json.loads((root/'source_snapshots.json').read_text());policies=json.loads((root/'frozen_policies.json').read_text())
    for source,digest in config['source_hashes'].items():
        if sha(snapshots[source])!=digest:raise ValueError('Changed archived source: '+source)
    if {r['id'] for r in manifest}&set(config['calibration_ids']):raise ValueError('Calibration overlap')
    for family in ('gaussian32','blasst'):
        p=json.loads((root/'policies'/(family+'_s40.json')).read_text())
        if p['heldout_used'] or p['calibration_ids']!=config['calibration_ids'] or p['policy']!=policies[family+'_s40']:
            raise ValueError('Invalid40 calibration provenance')
    raw={};missing=[];per_canvas=[];per_step=[];settings=defaultdict(set);sources={}
    from .sparsity_steps import validate_cached
    from experiments.diffusion_gemma_ruler8k_jl import score
    for regime,labels in config['conditions'].items():
        for label in labels:
            for row in manifest:
                path=shard_path(root/regime,label,row)
                if not path.exists():missing.append(dict(regime=regime,condition=label,id=row['id']));continue
                r=validate_cached(path,config,row,label,regime);sources[str(path)]=sha(path)
                if r['score']!=score(row,r['prediction']):raise ValueError('Official scoring mismatch')
                if r['steps']!=r['metadata']['actual_denoising_step_count'] or len(r['drafts'])!=r['steps']:
                    raise ValueError('Step count mismatch')
                if r['canvases']!=1 or len({d['input_length'] for d in r['drafts']})!=1:
                    raise ValueError('Expected one canvas per prompt')
                if regime=='fixed4' and (r['steps']!=4 or [d['remaining_schedule_step'] for d in r['drafts']]!=[4,3,2,1]):
                    raise ValueError('Fixed4 is not exactly4 calls')
                if r['drafts'][-1]['score']!=r['score']:raise ValueError('Draft/final score mismatch')
                if label in policies and r['policy']!=policies[label]:raise ValueError('Unfrozen threshold')
                meta=r['metadata'];settings[regime].add(json.dumps([meta['denoising_configuration'],meta['sampling'],meta['thinking'],meta['native_canvas_length']],sort_keys=True))
                counts={k:dict(eligible=0,skipped=0) for k in ('whole','local','global')}
                if 'routing_path' in r:
                    with gzip.open(r['routing_path'],'rt') as f:routing=json.load(f)
                    coverage=defaultdict(set)
                    for call in routing:
                        coverage[call['step']].add((call['layer'],call['head']))
                        if call['attention_type']!=('global' if call['layer'] in (5,11,17,23,29) else 'local'):raise ValueError('Layer classification')
                        for k in ('whole',call['attention_type']):
                            for n in ('eligible','skipped'):counts[k][n]+=call[n]
                    if len(coverage)!=r['steps'] or any(v!={(l,h) for l in range(30) for h in range(16)} for v in coverage.values()):
                        raise ValueError('Layer/head/step coverage incomplete')
                if counts!=r['counts']:raise ValueError('Counts differ from raw routing')
                target=0. if '_s' not in label else int(label.rsplit('_s',1)[1])/100
                family='Gaussian32' if label.startswith('gaussian32') else 'Aggressive BLASST' if label.startswith('blasst') else label
                item=dict(regime=regime,condition=label,method=family,target_sparsity=target,id=row['id'],task=row['task'],
                    canvas_index=0,canvas_length=256,steps=r['steps'],accuracy=r['score'],output_tokens=len(r['completion_tokens']),
                    reached48=int(r['steps']==48),draft_regression=max(d['score'] for d in r['drafts'][:-1])>r['score']+1e-6 if len(r['drafts'])>1 else False,
                    first_would_stop=next((d['step'] for d in r['drafts'] if d['would_stop']),None))
                for kind in counts:
                    item[kind+'_eligible']=counts[kind]['eligible'];item[kind+'_skipped']=counts[kind]['skipped']
                    item[kind+'_sparsity']=counts[kind]['skipped']/max(1,counts[kind]['eligible'])
                per_canvas.append(item)
                for d in r['drafts']:
                    per_step.append(dict(regime=regime,condition=label,id=row['id'],task=row['task'],step=d['step'],
                        score=d['score'],would_stop=d['would_stop'],is_terminal=d['step']==r['steps']))
                if r.get('source'):
                    source=r['source']
                    if sha(source['path'])!=source['sha256'] or sha(source['trace_path'])!=source['trace_sha256']:
                        raise ValueError('Imported source changed')
                raw[regime,label,row['id']]=r
    if any(len(v)!=1 for v in settings.values()):raise ValueError('Unmatched decoding settings')
    first_draft_checks=0;first_token_checks=0
    for (regime,label,identifier),r in raw.items():
        if regime!='fixed4' or ('adaptive',label,identifier) not in raw:continue
        a=raw['adaptive',label,identifier]['drafts'][0];b=r['drafts'][0]
        if a['score']!=b['score']:raise ValueError('First adaptive/fixed4 draft score differs')
        first_draft_checks+=1
        if 'tokens' in a:
            if a['tokens']!=b['tokens']:raise ValueError('First adaptive/fixed4 token sequence differs')
            first_token_checks+=1
    summary=[];pairs=[];tasks=[];drafts=[];transitions=[];regime_pairs=[]
    for regime,labels in config['conditions'].items():
        for label in labels:
            group=[r for r in per_canvas if r['regime']==regime and r['condition']==label]
            if not group:continue
            steps=np.asarray([r['steps'] for r in group]);accuracy=np.asarray([r['accuracy'] for r in group]);tt=[r['task'] for r in group]
            step_ci=bootstrap(steps,tt);acc_ci=bootstrap(accuracy,tt)
            result=dict(regime=regime,condition=label,method=group[0]['method'],target_sparsity=group[0]['target_sparsity'],
                n=len(group),canvases=len(group),mean_steps=float(steps.mean()),median_steps=float(np.median(steps)),
                p90_steps=float(np.quantile(steps,.9)),p95_steps=float(np.quantile(steps,.95)),
                steps_lower=step_ci[0],steps_upper=step_ci[1],total_steps=int(steps.sum()),reached48=int((steps==48).sum()),
                accuracy=float(np.mean([np.mean([r['accuracy'] for r in group if r['task']==t]) for t in sorted(set(tt))])),
                accuracy_lower=acc_ci[0],accuracy_upper=acc_ci[1],draft_regression_count=sum(r['draft_regression'] for r in group))
            for kind in ('whole','global','local'):
                eligible=sum(r[kind+'_eligible'] for r in group);skipped=sum(r[kind+'_skipped'] for r in group)
                result[kind+'_eligible']=eligible;result[kind+'_skipped']=skipped;result[kind+'_sparsity']=skipped/max(1,eligible)
            baseline=[raw[regime,'kernel_dense',r['id']] for r in group if (regime,'kernel_dense',r['id']) in raw]
            comparisons=[agreement(raw[regime,label,r['id']]['completion_tokens'],raw[regime,'kernel_dense',r['id']]['completion_tokens'])
                         for r in group if (regime,'kernel_dense',r['id']) in raw]
            result['token_agreement']=sum(c['matches'] for c in comparisons)/max(1,sum(c['compared'] for c in comparisons))
            summary.append(result)
            for task in sorted(set(tt)):
                subset=[r for r in group if r['task']==task]
                tasks.append(dict(regime=regime,condition=label,task=task,n=len(subset),accuracy=float(np.mean([r['accuracy'] for r in subset])),
                    mean_steps=float(np.mean([r['steps'] for r in subset])),sparsity=sum(r['whole_skipped'] for r in subset)/max(1,sum(r['whole_eligible'] for r in subset))))
            for baseline_name in ('native_dense','kernel_dense'):
                if label==baseline_name:continue
                matched=[(r,raw[regime,label,r['id']],raw[regime,baseline_name,r['id']]) for r in manifest
                         if (regime,label,r['id']) in raw and (regime,baseline_name,r['id']) in raw]
                if matched:
                    pairs.append(dict(regime=regime,**paired_ci(matched,label,baseline_name)))
            if regime=='fixed4':
                for step in range(1,5):
                    scores=[raw[regime,label,r['id']]['drafts'][step-1]['score'] for r in group]
                    drafts.append(dict(condition=label,step=step,n=len(scores),accuracy=float(np.mean(scores)),
                        fully_correct=sum(s>=1-1e-6 for s in scores)))
                for a,b in ((1,2),(2,3),(3,4),(1,4),(2,4)):
                    changes=[(raw[regime,label,r['id']]['drafts'][a-1]['score'],raw[regime,label,r['id']]['drafts'][b-1]['score']) for r in group]
                    transitions.append(dict(condition=label,from_step=a,to_step=b,n=len(changes),
                        improved=sum(y>x+1e-6 for x,y in changes),worsened=sum(y<x-1e-6 for x,y in changes),
                        correct_to_incorrect=sum(x>=1-1e-6 and y<1-1e-6 for x,y in changes),
                        accuracy_delta=float(np.mean([y-x for x,y in changes]))))
    knees=[]
    for label in config['conditions']['fixed4']:
        matched=[(r,raw['fixed4',label,r['id']],raw['adaptive',label,r['id']]) for r in manifest
                 if ('fixed4',label,r['id']) in raw and ('adaptive',label,r['id']) in raw]
        if matched:
            regime_pairs.append(paired_ci(matched,'fixed4/'+label,'adaptive/'+label))
    for family in ('Gaussian32','Aggressive BLASST'):
        group=sorted([r for r in summary if r['regime']=='adaptive' and r['method']==family],key=lambda r:r['target_sparsity'])
        for a,b in zip(group,group[1:]):
            matched=[(r,raw['adaptive',b['condition'],r['id']],raw['adaptive',a['condition'],r['id']]) for r in manifest
                     if ('adaptive',b['condition'],r['id']) in raw and ('adaptive',a['condition'],r['id']) in raw]
            if not matched:continue
            ci=paired_ci(matched,b['condition'],a['condition'],field='steps')
            dx=100*(b['whole_sparsity']-a['whole_sparsity']);dy=b['mean_steps']-a['mean_steps']
            knees.append(dict(method=family,from_target=a['target_sparsity'],to_target=b['target_sparsity'],
                from_actual=a['whole_sparsity'],to_actual=b['whole_sparsity'],step_ratio=b['mean_steps']/a['mean_steps'],
                extra_steps=dy,slope_steps_per_sparsity_pp=dy/dx if dx>0 else None,
                paired_step_delta=ci['mean'],paired_lower=ci['lower'],paired_upper=ci['upper']))
    audit=dict(complete=not missing,completed=len(raw),expected=len(manifest)*sum(map(len,config['conditions'].values())),
        missing=missing,one_canvas_per_sample=True,canvas_length=256,all_fixed4_exact=True,counts_verified=True,
        identical_settings_within_regime=True,calibration_disjoint=True,source_hashes=sources)
    audit.update(first_adaptive_fixed4_draft_score_matches=first_draft_checks,
                 first_adaptive_fixed4_token_matches=first_token_checks)
    atomic(root/'audit.json',audit);atomic(root/'summary.json',dict(summary=summary,comparisons=pairs,regime_comparisons=regime_pairs,knee_intervals=knees,
        fixed4_drafts=drafts,fixed4_transitions=transitions,complete=audit['complete']))
    for name,rows in [('summary',summary),('per_canvas',per_canvas),('per_task',tasks),('per_step_scores',per_step),
                      ('paired_comparisons',pairs),('regime_comparisons',regime_pairs),('knee_intervals',knees),('fixed4_drafts',drafts),('fixed4_transitions',transitions)]:
        write_csv(root/(name+'.csv'),rows)
    plot(root,summary,per_canvas,drafts,tasks)
    lines=['# RULER4K: physical sparsity, denoising steps and exact-four-step accuracy','',
        f"{'Complete' if audit['complete'] else 'INTERIM'}: {len(raw)}/{audit['expected']} sample/configuration results.",'',
        'DiffusionGemma-26B-A4B-it, pinned revisionf7f5b7f5fa82ffc52addd066915886d497f5517b, oneH100, BF16,128Q x64KV physical tiles, Gaussian32 seed1729. '
        'Same130 prompts (13 official RULER4K tasks x10), per-prompt seed42, original official generation budgets30-128 tokens, prefix+canvas eligible. '
        'Every prompt uses exactly one256-token canvas, even when fewer than256 output tokens are returned. Therefore steps/problem equals steps/canvas here. '
        'No multi-canvas-position trend is inferred from this cohort.','',
        'Adaptive runs retain the native48-step maximum, entropy-bound sampler,0.8-to0.4 temperature schedule and stability/confidence stopping. '
        'Steps at48 are capped realized counts, not proof of convergence. Physical sparsity is total skipped eligible tiles / total eligible tiles over all routed decoder layers and steps; dense prefix encoding is excluded. '
        'Means of steps weight each canvas equally. Accuracy is official RULER partial-credit equal-task macro. Confidence intervals use paired task-stratified prompt bootstraps,10000 draws.','',
        'Existing frozen50/60/65/70 calibration thresholds are reused.40 is calibrated on26 disjoint calibration prompts, separately for local/global attention. '
        'Aggressive BLASST uses lambda=exp(log_scale)/valid_KV_length and permits lambda>1; no threshold uses final scores. '
        'The extra65 point localizes changes within the requested60-to70 interval. All plotted sparse points use the same frozen Hopper binary; earlier reference-kernel sweeps are not mixed into the curves. '
        'Native dense and unpruned kernel dense are both shown because their masks/arithmetic differ; unpruned kernel dense is the matched sparse-path control.','',
        '## Adaptive trade-off','',
        '|Method|Target %|Actual %|Global %|Local %|Mean steps/canvas [95% CI]|Median /p90|At48 /130|Accuracy %|',
        '|---|---:|---:|---:|---:|---|---|---:|---:|']
    for r in summary:
        if r['regime']=='adaptive':
            lines.append(f"|{r['method']}|{100*r['target_sparsity']:.0f}|{100*r['whole_sparsity']:.2f}|{100*r['global_sparsity']:.2f}|{100*r['local_sparsity']:.2f}|{r['mean_steps']:.2f} [{r['steps_lower']:.2f},{r['steps_upper']:.2f}]|{r['median_steps']:.1f} /{r['p90_steps']:.1f}|{r['reached48']}|{100*r['accuracy']:.2f}|")
    lines+=['','![Trade-off](plots/tradeoff.png)','', '## Where the iterations increase sharply','',
        'We report observed intervals, not a precisely identified phase transition. The table gives adjacent-target paired mean-step differences and slopes against achieved sparsity. '
        'The65 point is an additional measurement, not a threshold tuned for accuracy.','',
        '|Method|Target interval %|Measured interval %|Step ratio|Additional steps [paired95% CI]|Slope steps /sparsity pp|',
        '|---|---|---|---:|---|---:|']
    for r in knees:
        slope='n/a' if r['slope_steps_per_sparsity_pp'] is None else f"{r['slope_steps_per_sparsity_pp']:.2f}"
        lines.append(f"|{r['method']}|{100*r['from_target']:.0f}-{100*r['to_target']:.0f}|{100*r['from_actual']:.2f}-{100*r['to_actual']:.2f}|{r['step_ratio']:.2f}x|{r['paired_step_delta']:.2f} [{r['paired_lower']:.2f},{r['paired_upper']:.2f}]|{slope}|")
    for family in ('Gaussian32','Aggressive BLASST'):
        candidates=[r for r in knees if r['method']==family and r['slope_steps_per_sparsity_pp'] is not None]
        if candidates:
            best=max(candidates,key=lambda r:r['slope_steps_per_sparsity_pp'])
            lines+=['',f"{family}: steepest measured rise is between targets{100*best['from_target']:.0f}% and{100*best['to_target']:.0f}% "
                f"(achieved{100*best['from_actual']:.2f}% to{100*best['to_actual']:.2f}%): {best['step_ratio']:.2f}x as many steps, "
                f"+{best['extra_steps']:.2f} per canvas. This is descriptive; no universal critical threshold is established."]
        dense=next((r for r in summary if r['regime']=='adaptive' and r['condition']=='kernel_dense'),None)
        group=sorted([r for r in summary if r['regime']=='adaptive' and r['method']==family],key=lambda r:r['target_sparsity'])
        if dense:
            first=next((r for r in group if r['mean_steps']>=2*dense['mean_steps']),None)
            if first:
                lines+=['',f"Using 'at least twice the unpruned-kernel mean' as an explicit descriptive marker, {family} first crosses it at "
                    f"target{100*first['target_sparsity']:.0f}% (actual{100*first['whole_sparsity']:.2f}%), "
                    f"{first['mean_steps']:.2f} versus{dense['mean_steps']:.2f} calls/canvas. This marker is not a fitted change point."]
    lines+=['','![Per-canvas spread](plots/per_canvas.png)','',
        '## Exactly four steps per problem','',
        'Every method performs4 decoder calls per problem (520 calls across130 problems). Adaptive stopping is disabled; the native4-step schedule uses temperatures0.8,0.7,0.6,0.5. '
        'This compresses annealing compared with the adaptive48-step configuration, so comparison across regimes changes both stopping and temperature progression. '
        'Within fixed4 all methods have identical settings, and thresholds are unchanged from the adaptive sweep. Actual sparsity may change with the trajectory.','',
        '|Method|Target %|Actual %|Global %|Local %|Accuracy %|Delta vs matched dense pp [95% CI]|',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in summary:
        if r['regime']!='fixed4':continue
        p=next((p for p in pairs if p['regime']=='fixed4' and p['first']==r['condition'] and p['second']=='kernel_dense'),None)
        delta='--' if p is None else f"{100*p['mean']:+.2f} [{100*p['lower']:+.2f},{100*p['upper']:+.2f}]"
        lines.append(f"|{r['method']}|{100*r['target_sparsity']:.0f}|{100*r['whole_sparsity']:.2f}|{100*r['global_sparsity']:.2f}|{100*r['local_sparsity']:.2f}|{100*r['accuracy']:.2f}|{delta}|")
    lines+=['','|Condition|Fixed4 minus adaptive accuracy pp [paired95% CI]|', '|---|---|']
    for p in regime_pairs:
        lines.append(f"|{p['first'].split('/',1)[1]}|{100*p['mean']:+.2f} [{100*p['lower']:+.2f},{100*p['upper']:+.2f}]|")
    lines+=['','![Four-step draft scores](plots/fixed4_drafts.png)','',
        '## Can later iterations damage an earlier good draft?','',
        'All four drafts are scored using the same official output budget/EOS extraction. A fully correct-to-incorrect change is direct evidence of draft damage on that prompt. '
        'The retrospective best-draft count is descriptive, not an implementable early-stopping rule; it uses answer labels. '
        'Paired transition counts avoid suggesting that a rising average prevents regressions on individual prompts.','',
        '|Condition|Transition|Improved|Worsened|Fully correct to incorrect|Mean score delta pp|',
        '|---|---|---:|---:|---:|---:|']
    for r in transitions:
        if r['from_step'] in (2,3) and r['to_step']==4:
            lines.append(f"|{r['condition']}|{r['from_step']}->{r['to_step']}|{r['improved']}|{r['worsened']}|{r['correct_to_incorrect']}|{100*r['accuracy_delta']:+.2f}|")
    if audit['complete']:
        for family in ('gaussian32','blasst'):
            for target in (50,70):
                label=f'{family}_s{target}';r=next(r for r in summary if r['regime']=='fixed4' and r['condition']==label)
                lines+=['',f"{label}: {r['draft_regression_count']}/130 final drafts score lower than an earlier draft within the four calls. "
                    +('The long-run regression phenomenon is also observable within four steps.' if r['draft_regression_count'] else
                      'No within-four-step regression was observed in this cohort; this does not contradict damage after longer forced continuation.')]
    lines+=['','No kernel optimization or routing redesign is introduced. All raw generations, counts, draft scores, thresholds, source snapshots, per-task and per-canvas tables are retained. '
        'Previously examined examples are not fresh held-out evidence. No hardware-speedup claim is made from this sweep.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    atomic(root/'report_provenance.json',{str(p.resolve()):sha(p) for p in (Path(__file__),Path(__file__).with_name('report.py'))})
    return audit


def plot(root,summary,per_canvas,drafts,tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'plots';out.mkdir(exist_ok=True)
    colors={'Gaussian32':'#a32930','Aggressive BLASST':'#2366a5'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for family,color in colors.items():
        group=sorted([r for r in summary if r['regime']=='adaptive' and r['method']==family],key=lambda r:r['target_sparsity'])
        if not group:continue
        x=np.array([100*r['whole_sparsity'] for r in group]);y=np.array([r['mean_steps'] for r in group])
        axes[0].errorbar(x,y,yerr=[y-np.array([r['steps_lower'] for r in group]),np.array([r['steps_upper'] for r in group])-y],
                         marker='o',capsize=3,color=color,label=family)
        axes[1].plot(x,[100*r['accuracy'] for r in group],'-o',color=color,label=family)
        for r,xx,yy in zip(group,x,y):axes[0].annotate(f"{100*r['target_sparsity']:.0f}%",(xx,yy),xytext=(0,8),textcoords='offset points',fontsize=8)
    for label,style in [('native_dense','--'),('kernel_dense',':')]:
        dense=next((r for r in summary if r['regime']=='adaptive' and r['condition']==label),None)
        if dense:
            axes[0].axhline(dense['mean_steps'],color='0.4',ls=style,label=label+' (0% reference)')
            axes[1].axhline(100*dense['accuracy'],color='0.4',ls=style,label=label+' (0% reference)')
    for ax in axes:ax.set_xlabel('Measured physical sparsity (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean denoising calls per 256-token canvas');axes[1].set_ylabel('Official RULER4K accuracy (%)')
    axes[0].set_ylim(bottom=0);axes[1].set_ylim(0,100)
    fig.suptitle('130 matched RULER4K canvases; adaptive maximum 48 calls');fig.tight_layout()
    fig.savefig(out/'tradeoff.png',dpi=190);fig.savefig(out/'tradeoff.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(family,color) in zip(axes,colors.items()):
        for target in (.4,.5,.6,.65,.7):
            rows=[r for r in per_canvas if r['regime']=='adaptive' and r['method']==family and r['target_sparsity']==target]
            if rows:ax.scatter([100*r['whole_sparsity'] for r in rows],[r['steps'] for r in rows],s=12,alpha=.45,label=f'{100*target:.0f}% target')
        ax.axhline(48,color='0.4',ls='--',label='48-step cap');ax.set(title=family,xlabel='Measured sparsity per canvas (%)',ylabel='Denoising calls');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'per_canvas.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for condition in dict.fromkeys(r['condition'] for r in drafts):
        rows=sorted([r for r in drafts if r['condition']==condition],key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[100*r['accuracy'] for r in rows],'-o',label=condition)
    ax.set(xlabel='Executed denoising iteration',ylabel='Official draft accuracy (%)',xticks=[1,2,3,4],ylim=(0,100))
    if drafts:ax.legend(fontsize=8)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'fixed4_drafts.png',dpi=190);fig.savefig(out/'fixed4_drafts.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,family in zip(axes,('gaussian32','blasst')):
        for task in sorted({r['task'] for r in tasks}):
            rows=sorted([r for r in tasks if r['regime']=='adaptive' and r['condition'].startswith(family)],key=lambda r:float(r['condition'].rsplit('_s',1)[1]))
            rows=[r for r in rows if r['task']==task]
            if rows:ax.plot([100*r['sparsity'] for r in rows],[r['mean_steps'] for r in rows],'-o',ms=3,label=task)
        ax.set(title=family,xlabel='Measured task sparsity (%)',ylabel='Mean steps/canvas');ax.legend(fontsize=5,ncol=2)
    fig.tight_layout();fig.savefig(out/'per_task_steps.png',dpi=180);plt.close(fig)
