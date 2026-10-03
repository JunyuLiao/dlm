"""Raw-shard intervention audit, Pareto analysis and plots; no inference."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .experiment import atomic,sha,shard_path,fingerprint
from .report import paired_ci
from .intervention_study import BASE,summarize


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def pareto(rows):
    return [r for r in rows if not any(
        other['actual_sparsity']>=r['actual_sparsity'] and other['accuracy']>=r['accuracy'] and other['mean_steps']<=r['mean_steps'] and
        (other['actual_sparsity']>r['actual_sparsity'] or other['accuracy']>r['accuracy'] or other['mean_steps']<r['mean_steps'])
        for other in rows if other is not r)]


def report(root):
    root=Path(root);cfg=json.loads((root/'configuration.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    policies=json.loads((root/'frozen_policies.json').read_text());selection=json.loads((root/'selection.json').read_text())
    snapshot=json.loads((root/'source_snapshots.json').read_text());limits=json.loads((root/'trigger_thresholds.json').read_text())['values']
    for source,digest in cfg['source_hashes'].items():
        if sha(snapshot[source])!=digest:raise ValueError('Archived source mismatch')
    baseline=json.loads((BASE/'summary.json').read_text())['summary']
    baseline=[r for r in baseline if r['regime']=='adaptive']
    dense={r['id']:json.loads(shard_path(BASE/'adaptive','kernel_dense',r).read_text()) for r in manifest}
    final={};missing=[];rawhash={};per_sample=[];per_step=[];summary=[];comparisons=[];settings=set();timings=[];shared_acceptance=[]
    for label,p in policies.items():
        if p['heldout_used'] or p['calibration_ids']!=cfg['calibration_ids']:raise ValueError('Policy uses wrong calibration')
        for row in manifest:
            path=shard_path(root/'final',label,row)
            if not path.exists():missing.append(str(path));continue
            r=json.loads(path.read_text());rawhash[str(path)]=sha(path)
            expected=fingerprint([cfg['fingerprint'],p['design']['policy'],p['thresholds'],limits,row['id'],row['seed'],row['prompt_hash']])
            if r['identity']!=expected:raise ValueError('Wrong frozen policy or sample')
            for key in ('id','prompt_hash','seed','generation_budget'):
                if row[key]!=r[key]:raise ValueError('Sample mismatch')
            if sha(r['routing_path'])!=r['routing_sha256']:raise ValueError('Corrupt routing')
            with gzip.open(r['routing_path'],'rt') as f:routing=json.load(f)
            count={k:dict(eligible=0,skipped=0) for k in ('whole','local','global')};coverage=defaultdict(set)
            mode={t['step']:t['dense'] for t in r['trajectory']};by_step=defaultdict(lambda:dict(eligible=0,skipped=0))
            for call in routing:
                step=call['step']+1
                if call['attention_type']!=('global' if call['layer'] in (5,11,17,23,29) else 'local'):raise ValueError('Wrong layer type')
                if not 0<=call['skipped']<=call['eligible']:raise ValueError('Counts invalid')
                if mode[step] and call['skipped']:raise ValueError('Dense iteration skipped tiles')
                coverage[step].add((call['layer'],call['head']))
                for k in ('whole',call['attention_type']):
                    for n in ('eligible','skipped'):count[k][n]+=call[n]
                for n in ('eligible','skipped'):by_step[step][n]+=call[n]
            if count!=r['counts'] or len(coverage)!=r['steps'] or len(r['trajectory'])!=r['steps']:raise ValueError('Step/count disagreement')
            if any(v!={(l,h) for l in range(30) for h in range(16)} for v in coverage.values()):raise ValueError('Missing layer/head')
            m=r['metadata'];settings.add(json.dumps([m['denoising_configuration'],m['sampling'],m['thinking'],m['native_canvas_length']],sort_keys=True))
            if r['steps']!=m['actual_denoising_step_count']:raise ValueError('Native/direct calls differ')
            from experiments.diffusion_gemma_ruler8k_jl import score
            if r['score']!=score(row,r['prediction']):raise ValueError('Official score mismatch')
            first_rescue=next((t['step'] for t in r['trajectory'] if t['dense']),None)
            protected=protected_disagreement=accepted_disagreement=accepted_total=0
            for t in r['trajectory']:
                ref=dense[row['id']]['completion_tokens'];added=t.get('added_positions',[])
                marked=t.get('protected_positions',[]);stored=t.get('protected_tokens',[])
                for i in range(min(len(ref),len(marked))):
                    if marked[i]:protected+=1;protected_disagreement+=stored[i]!=ref[i]
                for i in range(min(len(ref),len(added))):
                    if added[i]:accepted_total+=1;accepted_disagreement+=t['top1'][i]!=ref[i]
                pool=t.get('candidate_positions',[])
                if pool:
                    quota=min(p['design']['policy'].get('k',0),len(pool));known=[i for i in pool if i<len(ref)]
                    ranked=sorted(pool,key=lambda i:(-t['confidence_per_position'][i],i))[:quota]
                    known_ranked=[i for i in ranked if i<len(ref)]
                    shared_acceptance.append(dict(condition=label,id=row['id'],step=t['step'],pool=len(pool),k=quota,
                        random_expected_compared=len(known)*quota/len(pool),
                        random_expected_disagreement=sum(t['top1'][i]!=ref[i] for i in known)*quota/len(pool),
                        ranked_compared=len(known_ranked),ranked_disagreement=sum(t['top1'][i]!=ref[i] for i in known_ranked)))
                per_step.append(dict(condition=label,id=row['id'],task=row['task'],step=t['step'],dense=t['dense'],
                    accepted=t['accepted'],native_accepted=t['native_accepted'],extra_accepted=t['extra_accepted'],
                    protected=t['protected'],entropy=t['entropy'],confidence=t['confidence'],churn=t['churn'],
                    draft_score=t['draft_score'],terminated=t['terminated'],**by_step[t['step']]))
            eligible=count['whole']['eligible'];dense_eligible=sum(c['eligible'] for s,c in by_step.items() if mode[s])
            item=dict(condition=label,id=row['id'],task=row['task'],score=r['score'],steps=r['steps'],dense_steps=r['dense_steps'],
                actual_sparsity=count['whole']['skipped']/max(1,eligible),eligible=eligible,skipped=count['whole']['skipped'],
                local_eligible=count['local']['eligible'],local_skipped=count['local']['skipped'],
                global_eligible=count['global']['eligible'],global_skipped=count['global']['skipped'],
                dense_eligible=dense_eligible,conditional_max_sparsity=1-dense_eligible/max(1,eligible),
                reached4=r['steps']>=4,first_dense_step=first_rescue,
                stops_after_first_dense=first_rescue is not None and first_rescue==r['steps'],
                pre_rescue_score=None if first_rescue is None or first_rescue==1 else r['trajectory'][first_rescue-2]['draft_score'],
                first_rescue_score=None if first_rescue is None else r['trajectory'][first_rescue-1]['draft_score'],
                protected_compared=protected,protected_dense_disagreement=protected_disagreement,
                extra_compared=accepted_total,extra_dense_disagreement=accepted_disagreement,
                draft_regressed=any(t['draft_score']>r['score']+1e-6 for t in r['trajectory']))
            final[label,row['id']]=r;per_sample.append(item)
        group=[r for r in per_sample if r['condition']==label]
        if not group:continue
        steps=np.array([r['steps'] for r in group]);total_steps=int(steps.sum());dense_steps=sum(r['dense_steps'] for r in group)
        item=dict(policy=label,target_sparsity=p['design']['target'],n=len(group),actual_sparsity=sum(r['skipped'] for r in group)/sum(r['eligible'] for r in group),
            global_sparsity=sum(r['global_skipped'] for r in group)/sum(r['global_eligible'] for r in group),
            local_sparsity=sum(r['local_skipped'] for r in group)/sum(r['local_eligible'] for r in group),
            mean_steps=float(steps.mean()),median=float(np.median(steps)),p95=float(np.quantile(steps,.95)),cap48_fraction=float((steps==48).mean()),
            accuracy=float(np.mean([np.mean([r['score'] for r in group if r['task']==t]) for t in sorted({r['task'] for r in group})])),
            dense_steps=dense_steps,dense_step_fraction=dense_steps/total_steps,dense_steps_per_example=dense_steps/len(group),
            fraction_reaches4=float(np.mean([r['reached4'] for r in group])),
            fraction_stops_after_first_dense=float(np.mean([r['stops_after_first_dense'] for r in group])),
            instrumented_seconds=sum(final[label,r['id']]['seconds'] for r in group),
            calibrated_target_attained=p['attained'],calibration_actual=p['selected']['summary']['sparsity']['whole'],
            highest_observed_pilot_sparsity=p['maximum_observed'],
            protected_dense_disagreement=sum(r['protected_dense_disagreement'] for r in group)/max(1,sum(r['protected_compared'] for r in group)),
            protected_compared=sum(r['protected_compared'] for r in group),
            extra_dense_disagreement=sum(r['extra_dense_disagreement'] for r in group)/max(1,sum(r['extra_compared'] for r in group)),
            regressed_prompts=sum(r['draft_regressed'] for r in group))
        relevant=[t for r in group for t in final[label,r['id']]['trajectory']]
        item.update(mean_protected_fraction=float(np.mean([t['protected']/256 for t in relevant])),
                    mean_accepted=float(np.mean([t['accepted'] for t in relevant])),
                    mean_entropy=float(np.mean([t['entropy'] for t in relevant])),
                    mean_churn=float(np.mean([t['churn'] for t in relevant if t['churn'] is not None])),
                    extra_accepted=sum(t['extra_accepted'] for t in relevant),
                    pre_rescue_accuracy=float(np.mean([r['pre_rescue_score'] for r in group if r['pre_rescue_score'] is not None])) if any(r['pre_rescue_score'] is not None for r in group) else None,
                    first_rescue_accuracy=float(np.mean([r['first_rescue_score'] for r in group if r['pre_rescue_score'] is not None])) if any(r['pre_rescue_score'] is not None for r in group) else None,
                    fraction_rescued=float(np.mean([r['dense_steps']>0 for r in group])),
                    conditional_sparse_upper_bound=1-sum(r['dense_eligible'] for r in group)/sum(r['eligible'] for r in group))
        summary.append(item)
        pairs=[(r,final[label,r['id']],dense[r['id']]) for r in manifest if (label,r['id']) in final]
        comparisons.append(dict(policy=label,**paired_ci(pairs,label,'kernel_dense')))
        # Match against baseline at nearest measured sparsity, explicitly retaining
        # the gap rather than attributing a counterfactual number to rescue alone.
        nearest=min([b for b in baseline if b['method']=='Gaussian32'],key=lambda b:abs(b['whole_sparsity']-item['actual_sparsity']))
        base_steps={r['id']:json.loads(shard_path(BASE/'adaptive',nearest['condition'],r).read_text())['steps'] for r in manifest}
        item.update(nearest_baseline=nearest['condition'],baseline_sparsity_gap=item['actual_sparsity']-nearest['whole_sparsity'],
                    paired_steps_saved=float(np.mean([base_steps[r['id']]-r['steps'] for r in group])),
                    apparent_steps_saved_per_dense=sum(base_steps[r['id']]-r['steps'] for r in group)/dense_steps if dense_steps else None)
    if len(settings)!=1:raise ValueError('Generation settings diverged')
    timing_missing=[]
    for label in ['native_dense','kernel_dense']+selection['timing_selected']:
        records=[]
        for repeat in range(2):
            for row in manifest:
                path=shard_path(root/f'timing/{repeat}',label,row)
                if not path.exists():timing_missing.append(str(path));continue
                r=json.loads(path.read_text());records.append(r)
                if label not in ('native_dense','kernel_dense'):
                    instrumented=final[label,row['id']]
                    if r['completion_tokens']!=instrumented['completion_tokens'] or r['steps']!=instrumented['steps']:raise ValueError('Timed path differs from diagnostic generation')
        if records:timings.append(dict(policy=label,n=len(records),total_seconds=sum(r['seconds'] for r in records),
            mean_seconds=float(np.mean([r['seconds'] for r in records])),steps=sum(r['steps'] for r in records)))
    native=next((t for t in timings if t['policy']=='native_dense'),None)
    for t in timings:t['speedup_vs_native']=native['mean_seconds']/t['mean_seconds'] if native else None
    frontier=pareto(summary)
    audit=dict(complete=not missing and not timing_missing and len(policies)>=len(cfg['designs']),
        completed=len(final),expected=len(policies)*130,missing=missing,timing_missing=timing_missing,
        all_requested_designs_present=set(cfg['designs'])<=set(policies),source_hashes=rawhash,
        calibration_ids_disjoint=not bool(set(cfg['calibration_ids'])&set(cfg['ids'])),
        dense_steps_count_in_denominator=True,normal_timing_generation_parity=True)
    atomic(root/'audit.json',audit);atomic(root/'summary.json',dict(summary=summary,timing=timings,pareto=[r['policy'] for r in frontier],paired_accuracy=comparisons,audit_complete=audit['complete']))
    for name,data in [('summary',summary),('per_sample',per_sample),('per_step',per_step),('timing',timings),('paired_accuracy',comparisons),('shared_acceptance_diagnostic',shared_acceptance)]:write_csv(root/(name+'.csv'),data)
    figures(root,summary,baseline,per_step,selection['timing_selected'],per_sample)
    lines=['# Preventing sparse denoising tails','',
        f"{'Complete' if audit['complete'] else 'INCOMPLETE'}: {len(final)}/{len(policies)*130} final generations. Baseline reused without repeating the completed sweep.",'',
        'Same130 RULER4K prompts,13tasks x10,one256-token canvas, native48-step maximum/temperature/entropy/stability settings, BF16, frozen Gaussian32 projection and128x64 attention kernel. '
        'Dense interventions retain every tile using the matched kernel/mask path. They are not numerically identical to the native dense baseline. '
        'All dynamic targets refer to summed skipped eligible tiles / summed eligible tiles across every step, including dense rescue calls. '
        'Physical sparsity counts decoder attention only; native prefix encoding remains dense.','',
        'Thresholds were calibrated on26 disjoint prompts; reactive thresholds derive from dense pilot transition quantiles. Final scores did not choose thresholds or combinations. '
        'A common offset to the previously validated local/global log thresholds is swept over a full grid, with up to3 additional points. '
        'Changing trajectory length can make the aggregate response nonmonotonic. A missed target is labelled outside the tested range, not proven physically impossible. '
        'Maximum observed pilot sparsity is retained. High sparsity achieved only by prolonged sparse iterations does not count as healthy decoding.','',
        'Protection modifies accepted input tokens and renoising, while all attention queries and native stopping remain evaluated. '
        'Adaptive acceptance relaxes the native entropy budget as the accepted fraction increases. Random/ranked diagnostics use the same near-boundary candidate definition and min(4,pool size) quota; '
        'independent trajectories can have different candidate sets/quotas. On each saved candidate state we additionally compare exact uniform-random expected disagreement with confidence-ranked disagreement at identical k; see shared_acceptance_diagnostic.csv. This is a selection diagnostic, not a counterfactual full rollout. Random selection uses an independent RNG and does not consume native sampling randomness. '
        'Protected/added-token disagreement uses matched-dense output IDs within its returned span, a proxy rather than ground-truth token correctness.','',
        '|Policy|Target %|Actual %|Global /Local %|Mean|Median|P95|Cap48 %|Accuracy %|Dense steps/example|E2E seconds/example|',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        t=next((t for t in timings if t['policy']==r['policy']),None)
        seconds='not timed' if t is None else f"{t['mean_seconds']:.3f}"
        lines.append(f"|{r['policy']}|{100*r['target_sparsity']:.0f}|{100*r['actual_sparsity']:.2f}|{100*r['global_sparsity']:.2f} /{100*r['local_sparsity']:.2f}|{r['mean_steps']:.2f}|{r['median']:.1f}|{r['p95']:.1f}|{100*r['cap48_fraction']:.1f}|{100*r['accuracy']:.2f}|{r['dense_steps_per_example']:.2f}|{seconds}|")
    lines+=['','![Trade-off](plots/tradeoff.png)','', 'Empirical final Pareto frontier (uncertainty may reverse small differences): '+', '.join(r['policy'] for r in frontier), '',
        'Timing configurations were selected on development before final evaluation, not selected from final-test winners. '
        'Two repeats of all130 prompts, policy signals required for execution retained, diagnostic traces/draft scoring/routing counters disabled; model generation wall time includes signal overhead. '
        'Warmup uses one completed prompt per condition. No theoretical-sparsity speedup claim.','',
        '|Timed policy|Mean seconds/example|Native dense / policy time|','|---|---:|---:|']
    for t in timings:lines.append(f"|{t['policy']}|{t['mean_seconds']:.3f}|{t['speedup_vs_native']:.3f}x|")
    lines+=['','## Questions to interpret from the measured frontier','',
        'Dense-first and dense-late must be compared at achieved overall sparsity, not the threshold assigned to sparse calls. The per-sample tables include first dense iteration, pre/post-rescue draft score, and immediate termination after rescue. '
        'Step savings per dense call are observational comparisons to the nearest Gaussian32 sparsity point, with the sparsity gap explicitly reported; they are not a causal count of iterations eliminated.','',
        'Reactive corrections never occur on consecutive iterations: a dense correction returns to sparse before another can trigger. Churn/entropy/acceptance traces diagnose which signals associate with tails. '
        'Acceptance is reversible, not permanent commitment. The native exit test remains whole-canvas stability plus entropy, so extra accepted positions do not directly imply earlier termination.','',
        'Protection may sustain a wrong prediction; dense-token disagreement is only a proxy. The official prompt accuracy, draft regression counts, and paired intervals remain the outcome criteria. '
        'Random/ranked rollout differences cannot isolate selection quality without considering diverged states and actual quotas. No acceptance or stopping rule is declared safe solely from confidence.','',
        'Previously examined final prompts are reused. These are exploratory comparisons with multiple hypotheses, not fresh held-out confirmation. '
        'A final recommendation requires completed audits and the achieved-sparsity/accuracy/step/latency frontier.']
    healthy=[r for r in summary if r['accuracy']>=next(b['accuracy'] for b in baseline if b['condition']=='kernel_dense')-.02 and r['mean_steps']<=6]
    if healthy:
        best=max(healthy,key=lambda r:r['actual_sparsity'])
        lines+=['',f"Using the explicit exploratory definition accuracy within2pp of matched dense and mean<=6 calls, highest observed overall sparsity is {100*best['actual_sparsity']:.2f}% for {best['policy']} "
            f"({100*best['accuracy']:.2f}% accuracy, {best['mean_steps']:.2f} calls). This tolerance is a reporting convention, not a statistical equivalence claim."]
    lines+=answers(summary,baseline,timings,comparisons,shared_acceptance)
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return audit


def answers(rows,baseline,timings,comparisons,shared):
    lines=['','## Explicit research answers','']
    dense=next(r for r in baseline if r['condition']=='kernel_dense')
    for prefix,title in [('dense_first','Does a dense first call move the instability boundary?'),
                         ('dense_late','Does dense-late rescue remove the tail efficiently?'),
                         ('reactive','Is reactive correction better than a fixed schedule?')]:
        group=sorted([r for r in rows if r['policy'].startswith(prefix)],key=lambda r:r['target_sparsity'])
        lines+=['',title,'']
        for r in group:
            lines.append(f"- {r['policy']}: actual{100*r['actual_sparsity']:.2f}%, mean{r['mean_steps']:.2f}, accuracy{100*r['accuracy']:.2f}%, "
                f"dense{r['dense_steps_per_example']:.2f} calls/example; paired step saving{r['paired_steps_saved']:+.2f} relative to {r['nearest_baseline']} "
                f"(sparsity gap{100*r['baseline_sparsity_gap']:+.2f}pp).")
        feasible=[r for r in group if r['actual_sparsity']>=.65 and r['mean_steps']<=6 and r['accuracy']>=dense['accuracy']-.02]
        lines+=['',('At least one measured configuration meets the exploratory high-sparsity, near-dense accuracy and <=6-call criterion.' if feasible else
            'No measured configuration in this family simultaneously reaches >=65% overall sparsity, <=6 calls, and accuracy within2pp of matched dense.'),
            'Rescue can lower realized sparsity by shortening the sparse tail; target labels alone cannot establish a shifted boundary.']
    lines+=['','Dense-late rescue details:','',
        '|Policy|Reach step4 %|Terminate on first dense % of all examples|Before rescue accuracy %|After first rescue accuracy %|',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['policy'].startswith('dense_late'):
            pre='n/a' if r['pre_rescue_accuracy'] is None else f"{100*r['pre_rescue_accuracy']:.2f}"
            post='n/a' if r['first_rescue_accuracy'] is None else f"{100*r['first_rescue_accuracy']:.2f}"
            lines.append(f"|{r['policy']}|{100*r['fraction_reaches4']:.2f}|{100*r['fraction_stops_after_first_dense']:.2f}|{pre}|{post}|")
    lines+=['','Are the tails mostly prediction churn or conservative acceptance?','',
        'Acceptance interventions and dense correction address different mechanisms, but each also changes future states. Compare the observed outcomes below; '
        'no causal percentage of the tail can be assigned to churn versus acceptance from these rollouts alone. Native stopping still requires stability and low entropy.','',
        '|Policy|Actual %|Mean steps|Accuracy %|Mean accepted|Mean churn|Extra accepted observations|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['policy'].startswith(('accept_','extra_')):
            lines.append(f"|{r['policy']}|{100*r['actual_sparsity']:.2f}|{r['mean_steps']:.2f}|{100*r['accuracy']:.2f}|{r['mean_accepted']:.1f}|{r['mean_churn']:.4f}|{r['extra_accepted']}|")
    lines+=['','Can protection prevent good drafts deteriorating?','',
        '|Policy|Protected position-time %|Disagreement with dense within compared protected output %|Regressed prompts /130|Accuracy %|',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['policy'].startswith('protect'):
            lines.append(f"|{r['policy']}|{100*r['mean_protected_fraction']:.2f}|{100*r['protected_dense_disagreement']:.2f}|{r['regressed_prompts']}|{100*r['accuracy']:.2f}|")
    lines+=['','The disagreement column is not a count of proven wrong protected tokens: dense predictions are not token-level ground truth. '
        'Protection success requires both fewer regressions and acceptable final task scores at comparable actual sparsity. Reopening remains enabled.','',
        'How much can adaptive acceptance reduce steps before accuracy declines?','']
    for target in (.6,.65,.7):
        group=[r for r in rows if r['policy'].startswith('accept_') and r['target_sparsity']==target]
        for r in group:
            ci=next(p for p in comparisons if p['policy']==r['policy'])
            lines.append(f"- {r['policy']}: {r['mean_steps']:.2f} calls at{100*r['actual_sparsity']:.2f}% sparsity, accuracy delta "
                f"{100*ci['mean']:+.2f}pp [{100*ci['lower']:+.2f},{100*ci['upper']:+.2f}] against matched dense.")
    if shared:
        random_den=sum(r['random_expected_compared'] for r in shared);rank_den=sum(r['ranked_compared'] for r in shared)
        lines+=['',f"On identical saved candidate states and quotas, random selection's expected dense-token disagreement is "
            f"{100*sum(r['random_expected_disagreement'] for r in shared)/max(1,random_den):.2f}%, versus "
            f"{100*sum(r['ranked_disagreement'] for r in shared)/max(1,rank_den):.2f}% for confidence ranking. "
            'These compare positions within the returned dense span; rank selection can change that coverage. Full rollout results remain the decisive quality test.']
    lines+=['','Which gives the best end-to-end trade-off?','']
    sparse_times=[t for t in timings if t['policy'] not in ('native_dense','kernel_dense')]
    if sparse_times:
        best=min(sparse_times,key=lambda t:t['mean_seconds']);r=next(r for r in rows if r['policy']==best['policy'])
        lines+=[f"Fastest of the two development-selected timed interventions: {best['policy']}, {best['mean_seconds']:.3f}s/example, "
            f"{best['speedup_vs_native']:.3f}x native-dense speed ratio, {100*r['actual_sparsity']:.2f}% actual sparsity, "
            f"{100*r['accuracy']:.2f}% accuracy and {r['mean_steps']:.2f} calls. "
            'Only measured configurations are compared; no latency is imputed for the untimed frontier. Timing order is grouped by configuration, so clock/thermal drift is a limitation.']
    high=[r for r in rows if r['actual_sparsity']>=.65 and r['mean_steps']<=6 and r['accuracy']>=dense['accuracy']-.02]
    lines+=['','Recommended next direction: '+('validate the high-sparsity, short-trajectory Pareto candidates on fresh prompts and additional seeds before specializing kernels.' if high else
        'the measured interventions have not yet established the requested >=65% sparsity with near-dense quality and <=6 calls. Prioritize preserving the prediction trajectory at the measured attainable frontier; increasing sparse-step thresholds merely to offset dense corrections is not evidence of progress.'),
        'This is an exploratory recommendation from measured outcomes, with paired uncertainty; the shared130 prompts have been repeatedly examined.']
    return lines


def figures(root,summary,baseline,steps,selected,samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'plots';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for family in ('Gaussian32','Aggressive BLASST'):
        rows=sorted([r for r in baseline if r['method']==family],key=lambda r:r['whole_sparsity'])
        for ax,key in zip(axes,('mean_steps','accuracy')):
            ax.plot([100*r['whole_sparsity'] for r in rows],[r[key]*(100 if key=='accuracy' else 1) for r in rows],'-o',label=family)
    families=sorted({r['policy'].rsplit('_s',1)[0] for r in summary})
    for family in families:
        rows=sorted([r for r in summary if r['policy'].rsplit('_s',1)[0]==family],key=lambda r:r['actual_sparsity'])
        axes[0].plot([100*r['actual_sparsity'] for r in rows],[r['mean_steps'] for r in rows],'-o',ms=3,label=family)
        axes[1].plot([100*r['actual_sparsity'] for r in rows],[100*r['accuracy'] for r in rows],'-o',ms=3,label=family)
    for label,ls in [('native_dense','--'),('kernel_dense',':')]:
        b=next(r for r in baseline if r['condition']==label)
        axes[0].axhline(b['mean_steps'],ls=ls,color='0.4',label=label);axes[1].axhline(100*b['accuracy'],ls=ls,color='0.4')
    for ax in axes:ax.set_xlabel('Actual whole-generation physical sparsity (%)');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    axes[0].set_ylabel('Mean denoising calls per canvas');axes[1].set_ylabel('Official accuracy (%)')
    fig.tight_layout();fig.savefig(out/'tradeoff.png',dpi=190);fig.savefig(out/'tradeoff.pdf');plt.close(fig)
    for metric in ('entropy','churn','accepted'):
        fig,ax=plt.subplots(figsize=(7,4))
        for label in selected:
            group=defaultdict(list)
            for r in steps:
                if r['condition']==label and r[metric] is not None:group[r['step']].append(r[metric])
            x=sorted(group);ax.plot(x,[np.mean(group[t]) for t in x],label=label)
        ax.set(xlabel='Step (surviving trajectories)',ylabel=metric);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/(metric+'_vs_step.png'),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for label in selected:
        x=sorted(r['steps'] for r in samples if r['condition']==label)
        if x:ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label)
    ax.set(xlabel='Denoising steps',ylabel='Empirical CDF');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'step_distributions.png',dpi=160);plt.close(fig)
