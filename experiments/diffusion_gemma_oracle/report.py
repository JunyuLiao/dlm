"""Regenerate scores, selection diagnostics and figures from immutable shards."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_dataset import score
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate, csv_write, token_counts
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_runner import shard_path
from .run import conditions, fingerprint
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci


def summed(records):
    v={k:sum(r[k] for r in records) for k in ('eligible','skipped','mass_sum','rows','error_sq','dense_sq')}
    return dict(v,sparsity=v['skipped']/max(v['eligible'],1),mass=v['mass_sum']/max(v['rows'],1),
        relative_output_error=(v['error_sq']/max(v['dense_sq'],1e-30))**.5)


def ranking_metrics(record):
    scores={s:np.array(v,dtype=float) for s,v in record['scores'].items()}
    centered={s:rankdata(v)-((len(v)+1)/2) for s,v in scores.items()}
    correlations={}
    for signal, ranks in centered.items():
        base=centered['blasst'];denominator=float(np.sqrt(np.dot(base,base)*np.dot(ranks,ranks)))
        correlations[signal]=float(np.dot(base,ranks)/denominator) if denominator else None
    rows=[]
    for target, kept in record['retained'].items():
        kept=np.array(kept,dtype=bool); k=int(kept.sum()); n=len(kept)
        for signal, values in scores.items():
            order=np.argsort(-values,kind='stable'); best=set(order[:k]); actual=set(np.flatnonzero(kept))
            correlation=correlations[signal]
            mass=scores['mass']; optimal_mass=mass[order[:k]].sum()
            rows.append(dict(layer=record['layer'],head=record['head'],step=record['step'],
                attention_type=record['attention_type'],target=float(target),signal=signal,
                blocks=n,retained=k,rank_correlation=correlation,
                nontrivial_budget=0<k<n,
                budget_overlap=len(best & actual)/k if k else None,
                retained_mass=float(mass[kept].sum()/max(mass.sum(),1e-30)),
                ranked_budget_mass=float(optimal_mass/max(mass.sum(),1e-30))))
    return rows


def report(root):
    setup=json.loads((root/'setup.json').read_text()); expected_fp=fingerprint(root)
    samples={r['id']:r for r in setup['samples']}; results=[]; selection=[]; hetero=[]; missing=[]
    allrank=[]; violations=[]; completed=0
    for identity,row in samples.items():
        cache={}
        for name,ref in setup['cached'][identity].items():
            p=Path(ref['path'])
            if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']: raise RuntimeError('cache changed')
            cache[name]=json.loads(p.read_text())
        dense=cache['dense']; dense_score=score(row,dense['prediction'])
        generated=dict(cache)
        for name in conditions():
            p=shard_path(root,name,identity)
            if not p.exists(): missing.append(dict(condition=name,id=identity));continue
            d=json.loads(p.read_text());completed+=1
            if d['fingerprint']!=expected_fp: violations.append(f'fingerprint: {p}')
            if d['id']!=identity or d['condition']!=name: violations.append(f'shard identity: {p}')
            if not d['completion_tokens'] or not d['records']: violations.append(f'empty result: {p}')
            if {r['attention_type'] for r in d['records']}!={'local','global'}: violations.append(f'attention types: {p}')
            if {r['layer'] for r in d['records']}!=set(range(30)): violations.append(f'layer coverage: {p}')
            if {r['head'] for r in d['records']}!=set(range(16)): violations.append(f'head coverage: {p}')
            for r in d['records']:
                vals=[r[k] for k in ('eligible','skipped','mass_sum','rows','error_sq','dense_sq')]
                if not np.isfinite(vals).all() or not 0<=r['skipped']<=r['eligible'] or not 0<=r['mass_sum']<=r['rows']+1e-3:
                    violations.append(f'invalid counters: {p}');break
            if name=='dense_observer':
                if d['completion_tokens']!=dense['completion_tokens']: violations.append(f'dense parity: {identity}')
                expected_probes={'dense'} | {f'{s}_topk_{t}' for s in ('qk','mass','contribution','blasst') for t in (.25,.5,.75,.9)} | {f'blasst_actual_{t}' for t in (.25,.5,.75,.9)}
                if {r['probe'] for r in d['records']}!=expected_probes: violations.append(f'counterfactual probes: {p}')
                if not d['rankings']: violations.append(f'missing block rankings: {p}')
                for r in d['rankings']:
                    if len(r['tiles'])!=len(r['scores']['mass']) or not all(len(v)==len(r['tiles']) for v in r['retained'].values()):
                        violations.append(f'ranking dimensions: {p}')
                    selection.extend(dict(id=identity,benchmark=row['benchmark'],**m) for m in ranking_metrics(r))
                allrank.extend(dict(id=identity,benchmark=row['benchmark'],**r) for r in d['rankings'])
            generated[name]=d
        for name,d in generated.items():
            measured=score(row,d['prediction']); m,n=token_counts(dense['completion_tokens'],d['completion_tokens'])
            result=dict(id=identity,benchmark=row['benchmark'],task=row['task'],condition=name,
                accuracy=measured,dense_accuracy=dense_score,delta=measured-dense_score,matching=m,compared=n)
            if 'records' in d:
                records=[r for r in d['records'] if name!='dense_observer' or r['probe']=='dense']
                for kind in ('overall','global','local'):
                    mm=summed([r for r in records if kind=='overall' or r['attention_type']==kind])
                    result.update({kind+'_'+k:v for k,v in mm.items()})
                hetero.extend(dict(id=identity,benchmark=row['benchmark'],condition=name,**r) for r in d['records'])
            else:
                agg=aggregate(d['calls'])
                for kind in ('overall','global','local'):
                    result[kind+'_sparsity']=agg[kind]['full_tile_sparsity']
                    # Use source calls directly to preserve count weights.
                    calls=[c for c in d['calls'] if kind=='overall' or c['attention_type']==kind]
                    result[kind+'_eligible']=sum(c.get('physical_total_tiles',c.get('eligible_tiles',0)) for c in calls)
                    result[kind+'_skipped']=sum(c.get('physical_skipped_tiles',c.get('skipped_tiles',0)) for c in calls)
                    result[kind+'_rows']=sum(c.get('retained_attention_mass_rows',c.get('valid_rows',0)) for c in calls)
                    result[kind+'_mass_sum']=sum(c.get('retained_dense_attention_mass',1)*c.get('retained_attention_mass_rows',c.get('valid_rows',0)) for c in calls)
            results.append(result)
    def summarize(group):
        task=defaultdict(list)
        for r in group: task[r['task']].append(r['accuracy'])
        out=dict(samples=len(group),accuracy=float(np.mean([np.mean(v) for v in task.values()])),
            token_agreement=sum(r['matching'] for r in group)/max(sum(r['compared'] for r in group),1),
            paired_delta=float(np.mean([r['delta'] for r in group])))
        out['paired_delta_ci95']=paired_bootstrap_ci([r['delta'] for r in group])
        out['improved_samples']=sum(r['delta']>1e-12 for r in group)
        out['worsened_samples']=sum(r['delta']< -1e-12 for r in group)
        out['unchanged_samples']=len(group)-out['improved_samples']-out['worsened_samples']
        out['dense_full_score_samples']=sum(r['dense_accuracy']>=1-1e-12 for r in group)
        out['dense_full_score_preserved']=sum(r['dense_accuracy']>=1-1e-12 and r['accuracy']>=1-1e-12 for r in group)
        for kind in ('overall','global','local'):
            e=sum(r.get(kind+'_eligible',0) for r in group);s=sum(r.get(kind+'_skipped',0) for r in group)
            out[kind+'_sparsity']=s/max(e,1)
        rows=sum(r.get('overall_rows',0) for r in group)
        out['retained_mass']=sum(r.get('overall_mass_sum',0) for r in group)/max(rows,1)
        out['relative_output_error']=None
        if all('overall_error_sq' in r for r in group):
            out['relative_output_error']=(sum(r['overall_error_sq'] for r in group)/max(sum(r['overall_dense_sq'] for r in group),1e-30))**.5
        return out
    groups=defaultdict(list);tasks=defaultdict(list)
    for r in results:
        groups[r['benchmark'],r['condition']].append(r);tasks[r['benchmark'],r['task'],r['condition']].append(r)
    summary=[dict(benchmark=b,condition=c,**summarize(v)) for (b,c),v in groups.items()]
    per_task=[dict(benchmark=b,task=t,condition=c,**summarize(v)) for (b,t,c),v in tasks.items()]
    csv_write(root/'summary.csv',summary);csv_write(root/'per_task.csv',per_task)
    csv_write(root/'selection_quality.csv',selection);csv_write(root/'heterogeneity.csv',hetero)
    _write(root/'summary.json',summary);_write(root/'per_sample.json',results)
    quality_groups=defaultdict(list)
    for r in selection: quality_groups[r['benchmark'],r['attention_type'],r['target'],r['signal']].append(r)
    quality=[]
    for (b,kind,t,s),rs in quality_groups.items():
        q=dict(benchmark=b,attention_type=kind,target=t,signal=s,ranked_rows=len(rs),
            nontrivial_rows=sum(r['nontrivial_budget'] for r in rs))
        for field in ('rank_correlation','budget_overlap','retained_mass','ranked_budget_mass'):
            vals=[r[field] for r in rs if r[field] is not None]
            q[field]=float(np.mean(vals)) if vals else None
        vals=[r['budget_overlap'] for r in rs if r['nontrivial_budget']]
        q['nontrivial_budget_overlap']=float(np.mean(vals)) if vals else None
        quality.append(q)
    csv_write(root/'selection_summary.csv',quality)
    dimension_quality=[]
    for dimension in ('step','layer','head'):
        bins=defaultdict(list)
        for r in selection:
            if r['signal']=='mass': bins[r['benchmark'],r['target'],r[dimension]].append(r)
        for (benchmark,target,index),rs in bins.items():
            correlations=[r['rank_correlation'] for r in rs if r['rank_correlation'] is not None]
            overlaps=[r['budget_overlap'] for r in rs if r['nontrivial_budget']]
            dimension_quality.append(dict(benchmark=benchmark,target=target,dimension=dimension,index=index,
                ranked_rows=len(rs),nontrivial_rows=len(overlaps),
                rank_correlation=float(np.mean(correlations)) if correlations else None,
                nontrivial_budget_overlap=float(np.mean(overlaps)) if overlaps else None,
                retained_mass=float(np.mean([r['retained_mass'] for r in rs])),
                same_budget_mass_gap=float(np.mean([r['ranked_budget_mass']-r['retained_mass'] for r in rs]))))
    csv_write(root/'routing_quality_by_dimension.csv',dimension_quality)
    diagnostic_groups=defaultdict(list)
    for r in hetero:
        if r['condition']=='dense_observer': diagnostic_groups[r['benchmark'],r['probe']].append(r)
    diagnostics=[dict(benchmark=b,probe=p,**summed(rs)) for (b,p),rs in diagnostic_groups.items()]
    csv_write(root/'dense_counterfactual.csv',diagnostics)
    comparisons=[]
    for a in summary:
        if not any(a['condition'].startswith(s) for s in ('qk_','mass_','contribution_')):continue
        other=[r for r in summary if r['benchmark']==a['benchmark'] and r['condition'].startswith(('blasst_length','sol_gaussian'))]
        for family in ('blasst_length','sol_gaussian'):
            candidates=[r for r in other if r['condition'].startswith(family)]
            if not candidates:continue
            b=min(candidates,key=lambda r:abs(r['overall_sparsity']-a['overall_sparsity']))
            gap=a['overall_sparsity']-b['overall_sparsity']
            comparisons.append(dict(benchmark=a['benchmark'],oracle=a['condition'],baseline=b['condition'],
                oracle_sparsity=a['overall_sparsity'],baseline_sparsity=b['overall_sparsity'],
                sparsity_gap=gap,comparable_within_5pp=abs(gap)<=.05,accuracy_delta=a['accuracy']-b['accuracy']))
    csv_write(root/'actual_sparsity_comparisons.csv',comparisons)
    dispersion=[]
    for (b,probe),rs in diagnostic_groups.items():
        if not probe.endswith('topk_0.5'):continue
        for dimension in ('step','layer','head'):
            bins=defaultdict(list)
            for r in rs:bins[r[dimension]].append(r)
            values=[summed(v)['relative_output_error'] for v in bins.values()]
            dispersion.append(dict(benchmark=b,probe=probe,dimension=dimension,groups=len(values),
                minimum=min(values),median=float(np.median(values)),maximum=max(values),
                maximum_over_minimum=max(values)/max(min(values),1e-30)))
    csv_write(root/'heterogeneity_summary.csv',dispersion)
    # Later steps are reached by fewer prompts. Normalize within each prompt
    # before comparing steps to avoid treating changing sample mix as dynamics.
    step_bins=defaultdict(list)
    for (b,probe),rs in diagnostic_groups.items():
        if probe!='mass_topk_0.5':continue
        for r in rs:step_bins[b,r['id'],r['step']].append(r)
    step_values={key:summed(rs)['relative_output_error'] for key,rs in step_bins.items()}
    step_pairs=[]
    for (b,identity,step),value in step_values.items():
        initial=step_values.get((b,identity,0))
        if initial is not None and initial>0:
            step_pairs.append(dict(benchmark=b,id=identity,step=step,error=value,initial_error=initial,
                relative_to_initial=value/initial))
    csv_write(root/'step_within_prompt.csv',step_pairs)
    tolerance_rows=attention_tolerance_frontiers(hetero)
    csv_write(root/'attention_tolerance_frontiers.csv',tolerance_rows)
    smoke=json.loads((root/'smoke.json').read_text()) if (root/'smoke.json').exists() else None
    if not smoke or not smoke['passed'] or smoke['fingerprint']!=expected_fp:violations.append('missing/mismatched CUDA smoke')
    audit=dict(complete=not missing and not violations,completed=completed,expected=len(samples)*len(conditions()),
        missing=missing,violations=violations,cached_baselines_verified=len(samples)*9,
        gpu_smoke=smoke)
    _write(root/'audit.json',audit)
    plots(root,summary,selection,hetero,allrank)
    routing_plots(root,dimension_quality)
    tolerance_plots(root,tolerance_rows)
    lines=['# DiffusionGemma: safe block sparsity and routing quality','',
        f"Status: {'complete' if audit['complete'] else 'INCOMPLETE'}; {completed}/{audit['expected']} new runs, with {len(samples)*9} verified cached baseline runs.",'',
        'Sample selection: 10 LongBench (2 each qasper, hotpotqa, gov_report, trec, passage_retrieval_en), 6 AIME24. See setup.json for exact IDs, unchanged prompts, budgets, seeds, pinned model and native denoising schedule. Temperature=0 preserves the adapter’s native stochastic temperature schedule; it does not mean greedy decoding.','',
        'All routes operate on physical 64×64 tiles, including prefix and canvas. Top-k retains n−floor(s·n) blocks per query block/head; rounding and nonempty-row rescues are counted in achieved sparsity. Top-p retains the shortest ranked prefix covering p of mass or contribution norm; its achieved sparsity is measured, not fixed. QK ranks maximum valid token score. Mass ranks summed normalized probabilities. Contribution ranks the Frobenius norm of P_tile V_tile, including vector cancellation within a tile. BLASST ranks the maximum row-wise margin against the preceding running maximum. Existing calibrated BLASST and fixed-Gaussian Sol results are reused without threshold changes.','',
        'Reused Sol thresholds are β = −0.674490, 0, 0.674490, 1.281552 for targets 25/50/75/90%. Reused BLASST applies its established local/global length rule λ=min(1, α·exp(γ·s)/L), with λ=1 for targets previously marked unattainable. The exact fitted parameters are in the [inherited policy](../diffusion_gemma_solattn_blasst_multibench_controlled/calibration_policy.json); effective thresholds remain in the hash-verified cached sample shards. No final example was used to select new thresholds.','',
        'Output error is ||O_mask−O_dense||_F / ||O_dense||_F, with exact softmax renormalization on the same Q/K/V (FP32 diagnostic arithmetic). Sparse-trajectory mass/error compare against dense attention at that sparse state. Dense-observer counterfactuals share the cached dense trajectory. Counterfactuals sample the first call per layer at steps 0,1,4,12,24,47; steps not reached are absent. All sparse execution calls contribute to achieved sparsity and step/layer/head aggregates. Ratios use summed eligible/skipped counts and summed probability mass/valid rows.','',
        'Ranking correlations and same-budget overlaps are descriptive means over sampled query-block/head observations; these observations are dependent, not additional independent benchmark examples. The token metric continues comparing equal output positions after divergence, counting missing/extra positions as disagreements.','',
        '| Benchmark | Method | N | Accuracy | Actual sparsity | Global | Local | Mass | Token agreement |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+str(r['benchmark'])+' | '+r['condition']+' | '+str(r['samples'])+' | '+' | '.join(f'{100*r[k]:.2f}%' for k in ('accuracy','overall_sparsity','global_sparsity','local_sparsity','retained_mass','token_agreement'))+' |')
    if diagnostics:
        lines += ['', '## Dense-state block-selection check', '',
            '| Benchmark | Probe | Actual sparsity | Retained mass | Relative output error |',
            '|---|---|---:|---:|---:|']
        for r in diagnostics:
            if r['probe'].endswith('_0.5'):
                lines.append(f"| {r['benchmark']} | {r['probe']} | {r['sparsity']:.3f} | {r['mass']:.3f} | {r['relative_output_error']:.3f} |")
    if quality:
        lines += ['', '## BLASST selection versus mass ranking', '',
            '| Benchmark | Layers | Target | Spearman | Same-budget overlap (nontrivial masks) |',
            '|---|---|---:|---:|---:|']
        for r in quality:
            if r['signal']=='mass' and r['target']==.5:
                lines.append(f"| {r['benchmark']} | {r['attention_type']} | {r['target']} | {r['rank_correlation']} | {r['nontrivial_budget_overlap']} |")
    lines += ['','## Interpretation gates','',
        'A routing rule is empirically accuracy-safe on this subset only if its paired benchmark score does not decrease at the measured sparsity; small samples cannot establish population-level equivalence. Stronger ranking success where BLASST fails at comparable actual sparsity supports a routing-quality limitation. Failure of all tested rankings does not prove impossibility: these are diagnostic heuristics, not an exhaustive downstream oracle.','',
        'Compare selection_quality.csv (Spearman and same-budget overlap, actual versus ranked-budget mass) with heterogeneity.csv (step/layer/head error and mass). Large same-budget error variation supports testing dynamic allocation, but does not demonstrate its downstream benefit. Contribution norm ignores interactions between multiple dropped tiles; exact post-mask error is the check.','',
        'attention_tolerance_frontiers.csv reports the largest tested sparsity below each attention-output-error tolerance (1/5/10/20%); it does not label downstream accuracy as safe. step_within_prompt.csv compares later steps to step zero within each prompt, since fewer prompts reach late steps. Layer/head heatmaps retain the interaction between these dimensions instead of averaging away head specialization.','',
        'The four requested empirical conclusions remain unresolved until the new CUDA runs finish.' if not audit['complete'] else empirical_conclusions(summary,diagnostics,quality,comparisons,dispersion),
        '', 'No latency or deployment-speedup claim is made.']
    if diagnostics:
        lines += ['', '## Figures and detailed tables', '',
            '- [Accuracy versus physical sparsity](figures/accuracy.png), [retained mass](figures/retained_mass.png), [output error](figures/relative_output_error.png).',
            '- BLASST ranking overlays: [LongBench local](figures/longbench_local_ranking_overlay.png), [LongBench global](figures/longbench_global_ranking_overlay.png), [AIME local](figures/aime24_local_ranking_overlay.png), [AIME global](figures/aime24_global_ranking_overlay.png). Each is the median retained-fraction observation for its benchmark/type, not a selected failure case; exact source records accompany the plots.',
            '- [Layer/head error: LongBench](figures/longbench_layer_head_error.png), [AIME](figures/aime24_layer_head_error.png), [attention-error tolerance distribution](figures/attention_tolerance_cdf.png).',
            '- [Step/layer/head selection overlap](figures/routing_nontrivial_budget_overlap.png), [rank correlation](figures/routing_rank_correlation.png), [same-budget mass gap](figures/routing_same_budget_mass_gap.png).',
            '- [Per-task scores](per_task.csv), [full condition metrics and paired intervals](summary.csv), [matched-actual-sparsity comparisons](actual_sparsity_comparisons.csv).']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit


def routing_plots(root,rows):
    if not rows:return
    import matplotlib.pyplot as plt
    for metric in ('rank_correlation','nontrivial_budget_overlap','same_budget_mass_gap'):
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,dimension in zip(axes,('step','layer','head')):
            for benchmark in ('longbench','aime24'):
                values=sorted([r for r in rows if r['benchmark']==benchmark and r['dimension']==dimension
                    and r['target']==.5 and r[metric] is not None],key=lambda r:r['index'])
                if values:ax.plot([r['index'] for r in values],[r[metric] for r in values],'o-',label=benchmark)
            ax.set(xlabel=dimension,ylabel=metric);ax.legend()
        fig.suptitle('Calibrated BLASST s50 versus normalized attention-mass ranking')
        fig.tight_layout();fig.savefig(root/'figures'/('routing_'+metric+'.png'));plt.close(fig)


def attention_tolerance_frontiers(records):
    """Largest *tested* top-k sparsity under an attention-error tolerance.

    This is a same-state diagnostic, not a downstream accuracy-safety label or
    an interpolated upper bound. Each cell is one prompt/layer/head/step.
    """
    bins=defaultdict(list)
    for r in records:
        if r['condition']!='dense_observer' or '_topk_' not in r['probe']:continue
        signal=r['probe'].split('_topk_')[0]
        bins[r['benchmark'],r['id'],r['attention_type'],r['layer'],r['head'],r['step'],signal].append(r)
    rows=[]
    for (b,identity,kind,layer,head,step,signal),values in bins.items():
        for tolerance in (.01,.05,.1,.2):
            eligible=[r for r in values if summed([r])['relative_output_error']<=tolerance]
            best=max(eligible,key=lambda r:r['skipped']/max(r['eligible'],1)) if eligible else None
            rows.append(dict(benchmark=b,id=identity,attention_type=kind,layer=layer,head=head,step=step,
                signal=signal,relative_error_tolerance=tolerance,
                largest_tested_sparsity=best['skipped']/max(best['eligible'],1) if best else 0.,
                qualifying_sparse_points=len(eligible),tested_sparse_points=len(values)))
    return rows


def tolerance_plots(root,rows):
    if not rows:return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,b in zip(axes,('longbench','aime24')):
        for kind in ('global','local'):
            values=[r['largest_tested_sparsity'] for r in rows if r['benchmark']==b and r['attention_type']==kind
                and r['signal']=='mass' and r['relative_error_tolerance']==.05]
            if values:
                x=np.sort(values);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=kind)
        ax.set(title=b,xlabel='Largest tested physical sparsity with ≤5% attention-output error',ylabel='Fraction of prompt/layer/head/step cells')
        ax.legend()
    fig.tight_layout();fig.savefig(root/'figures'/'attention_tolerance_cdf.png');plt.close(fig)


def empirical_conclusions(summary,diagnostics,quality,comparisons,dispersion):
    lines=[]
    signal_errors=defaultdict(list)
    for r in diagnostics:
        if r['probe'].endswith('_0.5') and r['probe'].split('_topk_')[0] in ('qk','mass','contribution','blasst'):
            signal_errors[r['probe'].split('_topk_')[0]].append(r['relative_output_error'])
    if signal_errors:
        means={signal:float(np.mean(values)) for signal,values in signal_errors.items()}
        best=min(means,key=means.get)
        lines.append(f"Across the two benchmarks at the matched 50% top-k budget, {best} has the lowest mean same-state attention-output error ({means[best]:.3f}); signal means are " + ', '.join(f'{s}={v:.3f}' for s,v in sorted(means.items())) + ". This identifies the best tested predictor for local attention output preservation, not a proof of downstream safety.")
    for benchmark in ('longbench','aime24'):
        dense=next(r for r in summary if r['benchmark']==benchmark and r['condition']=='dense')
        candidates=[r for r in summary if r['benchmark']==benchmark and r['condition'].startswith(('mass_','qk_','contribution_','blasst_topk'))]
        safe=[r for r in candidates if r['accuracy']>=dense['accuracy']-1e-12]
        if safe:
            best=max(safe,key=lambda r:r['overall_sparsity'])
            lines.append(f"{benchmark}: observed score-preserving block sparsity exists on this subset: {best['condition']} reaches {100*best['overall_sparsity']:.1f}% sparsity with score {100*best['accuracy']:.2f}% versus dense {100*dense['accuracy']:.2f}%. The paired confidence interval is {best['paired_delta_ci95']}; this is not an equivalence test.")
        else:
            lines.append(f"{benchmark}: none of the tested stronger rankings preserves the dense aggregate score on this subset. This limits the tested rules, not every possible oracle.")
        diags=[r for r in diagnostics if r['benchmark']==benchmark and r['probe'].endswith('topk_0.5')]
        if diags:
            best=min(diags,key=lambda r:r['relative_output_error'])
            lines.append(f"At the matched 50% top-k budget, {best['probe']} has the lowest aggregate dense-state output error ({best['relative_output_error']:.3f}). Downstream accuracy must be checked separately; lowest local attention error need not imply the best rollout.")
        matched=[r for r in comparisons if r['benchmark']==benchmark and r['baseline'].startswith('blasst_length') and r['comparable_within_5pp']]
        if matched:
            best=max(matched,key=lambda r:r['accuracy_delta'])
            lines.append(f"Largest stronger-ranking advantage within 5 sparsity points of calibrated BLASST: {best['oracle']} vs {best['baseline']}, score delta {100*best['accuracy_delta']:+.2f} points, actual-sparsity gap {100*best['sparsity_gap']:+.2f} points. This comparison is approximate, not an exact density match.")
    if dispersion:
        largest=max(dispersion,key=lambda r:r['maximum_over_minimum'])
        lines.append(f"The largest aggregated sensitivity spread is across {largest['dimension']} for {largest['benchmark']} {largest['probe']}: relative error {largest['minimum']:.3f}–{largest['maximum']:.3f}. Such variation motivates testing dynamic allocation if it is stable across samples; it is not evidence that an adaptive policy already preserves accuracy.")
    return '\n\n'.join(lines)


def plots(root,summary,selection,hetero,rankings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    for metric in ('accuracy','retained_mass','token_agreement','relative_output_error'):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,b in zip(axes,('longbench','aime24')):
            for family in ('dense','sol_gaussian','blasst_length','qk_topk','mass_topk','contribution_topk','blasst_topk','mass_topp','contribution_topp'):
                data=sorted([r for r in summary if r['benchmark']==b and r['condition'].startswith(family) and r.get(metric) is not None],key=lambda r:r['overall_sparsity'])
                if data: ax.plot([r['overall_sparsity'] for r in data],[r[metric] for r in data],'o-',label=family)
            ax.set(title=b,xlabel='Measured skipped / eligible physical tiles',ylabel=metric);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(directory/(metric+'.png'));plt.close(fig)
    for b in ('longbench','aime24'):
      for kind in ('global','local'):
        candidates=[r for r in rankings if r['benchmark']==b and r['attention_type']==kind and len(r['tiles'])>4]
        # Median achieved retained fraction, not a hand-picked failure case.
        candidates.sort(key=lambda r:(np.mean(r['retained']['0.5']),r['id'],r['layer'],r['head'],r['step']))
        chosen=candidates[len(candidates)//2] if candidates else None
        if chosen:
            fig,axes=plt.subplots(4,1,figsize=(12,10))
            for ax,sig in zip(axes,('qk','mass','contribution','blasst')):
                v=np.array(chosen['scores'][sig],dtype=float);order=np.argsort(-v,kind='stable')
                retained=np.array(chosen['retained']['0.5'])[order]
                display=v[order].copy();finite=display[np.isfinite(display)]
                # +inf denotes a new running-max record; put it above finite
                # margins for display, preserving the exact saved ranking.
                if (~np.isfinite(display)).any():
                    display[np.isposinf(display)]=(float(finite.max())+1) if len(finite) else 1
                    display[np.isneginf(display)]=(float(finite.min())-1) if len(finite) else -1
                ax.scatter(np.arange(len(v)),display,c=np.where(retained,'tab:blue','tab:red'),s=12)
                ax.set(ylabel=sig,xlabel='Importance rank; blue retained / red skipped by calibrated BLASST s50')
            fig.suptitle(f"{b} {kind}: median retained fraction; layer {chosen['layer']}, head {chosen['head']}, step {chosen['step']}")
            fig.tight_layout();fig.savefig(directory/(b+'_'+kind+'_ranking_overlay.png'));plt.close(fig)
            _write(directory/(b+'_'+kind+'_ranking_source.json'),chosen)
    for b in ('longbench','aime24'):
        cells=defaultdict(list)
        for r in hetero:
            if r['benchmark']==b and r['condition']=='dense_observer' and r['probe']=='mass_topk_0.5':
                cells[r['layer'],r['head']].append(r)
        if cells:
            array=np.full((30,16),np.nan)
            for (layer,head),records in cells.items():array[layer,head]=summed(records)['relative_output_error']
            fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(array,aspect='auto',cmap='magma')
            ax.set(xlabel='Head',ylabel='Layer',title=f'{b}: mass top-k s50 relative attention-output error')
            fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(directory/(b+'_layer_head_error.png'));plt.close(fig)
    for dimension in ('step','layer','head'):
        groups=defaultdict(list)
        for r in hetero:
            if r['condition']=='dense_observer' and r['probe'].endswith('topk_0.5'):
                groups[r['probe'],r[dimension]].append(r)
        if groups:
            fig,ax=plt.subplots(figsize=(9,4))
            for signal in sorted({k[0] for k in groups}):
                data=sorted((dim,summed(v)) for (s,dim),v in groups.items() if s==signal)
                ax.plot([d for d,_ in data],[v['relative_output_error'] for _,v in data],'o-',label=signal)
            ax.set(xlabel=dimension,ylabel='Same-state relative output error at s50');ax.legend()
            fig.tight_layout();fig.savefig(directory/(dimension+'_error.png'));plt.close(fig)
