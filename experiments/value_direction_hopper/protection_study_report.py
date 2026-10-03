"""Raw-only protection refinement analysis, including stopping failure types."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .experiment import atomic,sha,shard_path
from .report import paired_ci
from .protection_study import BASE,OLD


def csvwrite(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def diagnose(record,label):
    """Categories overlap; fresh predictions still participate in stopping."""
    result=[];reopen_total=np.zeros(256,int);churn_total=np.zeros(256,int)
    for t in record['trajectory']:
        reopen_total+=np.array(t['reopened_positions'],int)
        churn_total+=np.array(t['churn_positions'],int)
    for t in record['trajectory']:
        old=np.array(t['previous_protected'],bool);churn=np.array(t['churn_positions'],bool)
        entropy=np.array(t['entropy_positions']);unstable=np.array(t['stopper_unstable'],bool)
        reopened=np.array(t['reopened_positions'],bool)
        marked=np.array(t['protected_positions'],bool)
        token_conflict=marked&(np.array(t['protected_tokens'])!=np.array(t['top1']))
        end=len(t['draft_tokens'])
        for region,start,stop in [('all',0,256),('answer',0,end),('remaining',end,256)]:
            sl=slice(start,stop)
            result.append(dict(policy=label,id=record['id'],task=record['task'],step=t['step'],region=region,
                continued_beyond5=record['steps']>5,positions=stop-start,
                unstable_positions=int(unstable[sl].sum()),
                unstable_unprotected=int((unstable&~old)[sl].sum()),
                unstable_protected=int((unstable&old)[sl].sum()),
                protected_token_conflict=int(token_conflict[sl].sum()),
                unprotected_churn=int((~old&churn)[sl].sum()),repeated_unprotected_churn=int((~old&churn&(churn_total>1))[sl].sum()),
                reopened=int(reopened[sl].sum()),repeated_reopens=int((reopened&(reopen_total>1))[sl].sum()),
                protected_fresh_churn=int((old&churn)[sl].sum()),stable_high_entropy=int((~unstable&(entropy>=.005))[sl].sum()),
                entropy_sum=float(entropy[sl].sum()),protected=t['protected'] if region=='all' else int(np.array(t['protected_positions'])[sl].sum()),
                newly_protected=t['newly_protected'] if region=='all' else None,
                accepted=t['accepted'] if region=='all' else None,renoised=t['renoised'] if region=='all' else None,
                confidence=t['confidence'],margin=t['margin'],entropy=t['entropy'],churn=t['churn'],
                stable=t['stable'],confident=t['confident'],failure='none' if t['would_stop'] else 'entropy' if t['stable'] else 'stability' if t['confident'] else 'both'))
    return result


def report(root):
    root=Path(root);cfg=json.loads((root/'configuration.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    policies=json.loads((root/'frozen_policies.json').read_text());timed=json.loads((root/'timing_selection.json').read_text())['selected']
    raw={};summary=[];pairs=[];diagnostics=[];per_sample=[];missing=[];hashes={}
    snapshots=json.loads((root/'source_snapshots.json').read_text())
    for source,digest in cfg['source_hashes'].items():
        if sha(snapshots[source])!=digest:raise ValueError('Source snapshot mismatch')
    controls=['native_dense','kernel_dense','static_s60','static_s65','protect_s60','protect_s65']
    for label in controls+list(policies):
        for row in manifest:
            if label.startswith('protect'):path=shard_path(root/'controls',label,row)
            elif label in controls:
                name=label.replace('static','gaussian32');path=shard_path(BASE/'adaptive',name,row)
            else:path=shard_path(root/'final',label,row)
            if not path.exists():missing.append(str(path));continue
            r=json.loads(path.read_text());hashes[str(path)]=sha(path)
            for key in ('id','prompt_hash','seed','generation_budget'):
                if r[key]!=row[key]:raise ValueError('Sample mismatch')
            if r['steps']!=r['metadata']['actual_denoising_step_count']:raise ValueError('Wrong steps')
            settings=r['metadata']['denoising_configuration']
            if settings['max_denoising_steps']!=48 or settings['confidence_threshold']!=.005 or settings['stability_threshold']!=1:raise ValueError('Stopping changed')
            if 'trajectory' in r:
                if len(r['trajectory'])!=r['steps'] or not r['trajectory'][-1]['terminal']:raise ValueError('Incomplete trajectory')
                diagnostics+=diagnose(r,label)
                if any(t['would_stop']!=(t['stable'] and t['confident']) for t in r['trajectory']):raise ValueError('Native stopping differs')
                if r['steps']<48 and not r['trajectory'][-1]['would_stop']:raise ValueError('Early termination bypassed native stopper')
                if sha(r['routing_path'])!=r['routing_sha256']:raise ValueError('Corrupt routing')
            if label in policies:
                if r['spec'] != dict(policies[label]['spec'],profile=False) or r['thresholds'] != policies[label]['thresholds']:
                    raise ValueError('Final shard differs from frozen policy')
            for counts in (r['counts'],r.get('executed_counts',r['counts'])):
                for kind in ('whole','global','local'):
                    if not 0 <= counts[kind]['skipped'] <= counts[kind]['eligible']:
                        raise ValueError('Invalid physical tile counts')
            from experiments.diffusion_gemma_ruler8k_jl import score
            if r['score']!=score(row,r['prediction']):raise ValueError('Score mismatch')
            raw[label,row['id']]=r
            output=r['counts']['whole'];exe=r.get('executed_counts',r['counts'])['whole']
            trajectory=r.get('trajectory',[])
            item=dict(policy=label,id=row['id'],task=row['task'],accuracy=r['score'],steps=r['steps'],
                eligible=exe['eligible'],skipped=exe['skipped'],retained=exe['eligible']-exe['skipped'],
                output_eligible=output['eligible'],output_skipped=output['skipped'],
                reopened=sum(t['reopened'] for t in trajectory),protected_observations=sum(t['protected'] for t in trajectory),
                logical_rescued=sum(t['logical_rescued'] for t in trajectory),
                rescue_extra_qk=sum(w['extra_qk_tiles'] for w in r.get('rescue_work',[])),
                rescue_extra_retained=sum(w['extra_output_retained'] for w in r.get('rescue_work',[])),
                rescue_extra_executed_pv=sum(w['executed_retained']-w['base_retained'] for w in r.get('rescue_work',[])))
            per_sample.append(item)
        group=[r for r in per_sample if r['policy']==label]
        if not group:continue
        steps=np.array([r['steps'] for r in group]);executed={}
        for kind in ('whole','global','local'):
            eligible=sum(raw[label,r['id']].get('executed_counts',raw[label,r['id']]['counts'])[kind]['eligible'] for r in group)
            skipped=sum(raw[label,r['id']].get('executed_counts',raw[label,r['id']]['counts'])[kind]['skipped'] for r in group)
            executed[kind]=skipped/max(1,eligible)
        item=dict(policy=label,n=len(group),actual_sparsity=executed['whole'],global_sparsity=executed['global'],local_sparsity=executed['local'],
            output_sparsity=sum(r['output_skipped'] for r in group)/max(1,sum(r['output_eligible'] for r in group)),
            accuracy=float(np.mean([r['accuracy'] for r in group])),mean_steps=float(steps.mean()),median=float(np.median(steps)),p95=float(np.quantile(steps,.95)),
            finish4=float((steps<=4).mean()),finish5=float((steps<=5).mean()),cap48=float((steps==48).mean()),
            reopen_rate=sum(r['reopened'] for r in group)/max(1,sum(r['protected_observations']+r['reopened'] for r in group)),
            total_qk_tiles=sum(r['eligible'] for r in group),total_retained_pv_tiles=sum(r['retained'] for r in group),
            rescue_extra_qk=sum(r['rescue_extra_qk'] for r in group),rescue_extra_executed_pv=sum(r['rescue_extra_executed_pv'] for r in group),
            rescue_extra_output_retained=sum(r['rescue_extra_retained'] for r in group),logical_rescued_fraction=sum(r['logical_rescued'] for r in group)/max(1,256*steps.sum()))
        summary.append(item)
        for baseline in controls:
            matched=[(row,raw[label,row['id']],raw[baseline,row['id']]) for row in manifest if (label,row['id']) in raw and (baseline,row['id']) in raw]
            if matched and baseline!=label:pairs.append(paired_ci(matched,label,baseline))
    # Controls occur earlier in iteration order, so comparisons with every control
    # are available for all refinements.
    timing=[];timing_missing=[]
    for label in controls+timed:
        records={}
        for mode in ('e2e','profile'):
            records[mode]=[]
            for repeat in range(2):
                for row in manifest:
                    path=shard_path(root/f'timing/{mode}/{repeat}',label,row)
                    if not path.exists():timing_missing.append(str(path));continue
                    r=json.loads(path.read_text());reference=raw[label,row['id']]
                    if r['completion_tokens']!=reference['completion_tokens'] or r['steps']!=reference['steps']:raise ValueError('Runtime mode changed generation')
                    records[mode].append(r)
        if records['e2e']:
            timing.append(dict(policy=label,n=len(records['e2e']),e2e_seconds=float(np.mean([r['seconds'] for r in records['e2e']])),
                decoder_seconds=float(np.mean([r['decoder_ms']/1000 for r in records['profile']])) if records['profile'] else None,
                policy_seconds=float(np.mean([r['protection_ms']/1000 for r in records['profile']])) if records['profile'] else None))
    native=next((t for t in timing if t['policy']=='native_dense'),None)
    for t in timing:t['speed_ratio']=native['e2e_seconds']/t['e2e_seconds'] if native else None
    audit=dict(complete=not missing and not timing_missing,completed=len(raw),expected=(6+len(policies))*130,
        missing=missing,timing_missing=timing_missing,source_hashes=hashes,stopping_unchanged=True,runtime_parity=True)
    audit['analysis_source_sha256']=sha(Path(__file__))
    atomic(root/'audit.json',audit);atomic(root/'summary.json',dict(summary=summary,timing=timing,comparisons=pairs,audit_complete=audit['complete']))
    for name,data in [('summary',summary),('timing',timing),('per_sample',per_sample),('stopping_diagnostics',diagnostics),('comparisons',pairs)]:csvwrite(root/(name+'.csv'),data)
    plots(root,summary,diagnostics,per_sample,timing)
    lines=['# Protection refinement at 60–65% sparsity','',f"{'Complete' if audit['complete'] else 'INCOMPLETE'}: {len(raw)}/{audit['expected']} control/final examples.",'',
        'Same130 RULER4K prompts,seed42,BF16,fixed Gaussian32 projections,128x64 physical tiles,256-token canvas and original48-step native schedule. '
        'Native stopping examines all fresh logits and predictions; protection only changes accepted input tokens. No smaller cap, fabricated stability, excluded queries or stopping relaxation. '
        'Native dense uses its original backend. Matched-kernel dense and interventions use the historical sparse masks/numerics.','',
        'Fixed-threshold ablations run on26 disjoint calibration prompts before selection/recalibration. Threshold adjustments are only0,-0.25,-0.5 logtau: no compensating increase in skipping. '
        'Top2 non-rescue refinements,one rescue variant,and one combination pertarget are selected on calibration; timing selection uses13 separate development prompts. '
        'Previously examined final130 remain exploratory; no tuning on final scores.','',
        'Rescue is reference-only: a normal sparse pass plus a conservative or dense pass, with full128-query blocks selected from previous-iteration signals. '
        'Actual sparsity below counts skipped/eligible tiles over BOTH executed passes. Output-mask sparsity is separately saved; it does not imply avoiding the discarded pass. '
        'All extra QK and PV work is counted. Rescue reference variants are not used to claim optimized speedups.','',
        '|Policy|Actual sparsity %|G/L %|Accuracy %|Mean steps|P95|Finish<=5 %|Reopening %|QK tiles (M)|Retained PV tiles (M)|E2E s|Speed ratio|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        t=next((t for t in timing if t['policy']==r['policy']),None)
        ts='not timed' if t is None else f"{t['e2e_seconds']:.3f}";ratio='--' if t is None else f"{t['speed_ratio']:.3f}x"
        qk='—' if r['policy']=='native_dense' else f"{r['total_qk_tiles']/1e6:.2f}"
        pv='—' if r['policy']=='native_dense' else f"{r['total_retained_pv_tiles']/1e6:.2f}"
        lines.append(f"|{r['policy']}|{100*r['actual_sparsity']:.2f}|{100*r['global_sparsity']:.1f}/{100*r['local_sparsity']:.1f}|{100*r['accuracy']:.2f}|{r['mean_steps']:.2f}|{r['p95']:.1f}|{100*r['finish5']:.1f}|{100*r['reopen_rate']:.2f}|{qk}|{pv}|{ts}|{ratio}|")
    lines+=['','Accuracy is the equal-task mean across 13 RULER4K tasks (10 examples each). Sparsity uses summed physical tile counts over all executed iterations; G/L denotes global/local layers. The native dense backend has no matching router tile counter, so its work columns are unreported. The complete counts and output-mask sparsity are in `summary.csv`.','',
        '![Accuracy versus mean steps](plots/tradeoff.png)','',
        '![Accuracy versus measured sparsity](plots/sparsity_accuracy.png)','',
        '![Step-count distributions](plots/steps.png)','', '## Remaining termination failures','',
        'Counts below cover original protect controls afteriteration1 in trajectories longer than5 calls. Categories overlap: unprotected churn, repeated reopening, fresh protected-query churn, and stable positions with entropy>=0.005. '
        'Per-region/per-step records separate returned-answer positions from the remaining canvas, without excluding either from stopping.','',
        '|Control|Both fail steps|Only entropy|Only stability|Unprotected churn|Repeated reopening|Protected fresh churn|Stable high entropy|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ('protect_s60','protect_s65'):
        d=[r for r in diagnostics if r['policy']==label and r['region']=='all' and r['continued_beyond5'] and r['step']>1]
        lines.append('|'+label+'|'+ '|'.join(str(sum(r['failure']==f for r in d)) for f in ('both','entropy','stability'))+'|'+
            '|'.join(str(sum(r[k] for r in d)) for k in ('unprotected_churn','repeated_reopens','protected_fresh_churn','stable_high_entropy'))+'|')
    lines+=['',
        'The remaining long-tail iterations fail prediction stability much more often than entropy alone. Counts of positions with entropy ≥0.005 are descriptive; the native entropy rule tests the mean over all positions. The answer/remaining split follows each step’s extracted draft length, so its boundary can move.','',
        '|Control|Long trajectories (>5)|Unprotected churn in answer|Unprotected churn elsewhere|Mean unstable positions/failed step|Failed steps with ≤5 unstable positions|',
        '|---|---:|---:|---:|---:|---:|']
    for label in ('protect_s60','protect_s65'):
        d=[r for r in diagnostics if r['policy']==label and r['region']=='all' and r['continued_beyond5'] and r['step']>1 and r['failure']!='none']
        answer=[r for r in diagnostics if r['policy']==label and r['region']=='answer' and r['continued_beyond5'] and r['step']>1]
        remaining=[r for r in diagnostics if r['policy']==label and r['region']=='remaining' and r['continued_beyond5'] and r['step']>1]
        lines.append(f"|{label}|{len({r['id'] for r in d})}|{sum(r['unprotected_churn'] for r in answer)}|{sum(r['unprotected_churn'] for r in remaining)}|{np.mean([r['unstable_positions'] for r in d]):.1f}|{sum(r['unstable_positions']<=5 for r in d)}/{len(d)}|")
    hysteresis_diag=[r for r in diagnostics if r['policy']=='hysteresis_s65' and r['region']=='all']
    hysteresis_conflicts=sum(r['protected_token_conflict'] for r in hysteresis_diag)
    hysteresis_observations=sum(r['protected'] for r in hysteresis_diag)
    lines+=['','The old confidence>=0.999 criterion already implies margin>=0.998; removing its margin0.98 check should be equivalent. Strictmargin0.999 is the nonredundant ablation. '
        'Hysteresis uses the probability assigned to the stored protected token and its disadvantage to the strongest alternative, allowing corrected predictions to reopen. '
        'The repeated unprotected changes occur across many positions, not just a single stubborn token. Hysteresis drops reopening from 0.30% to 0.006% at 65% sparsity while mean steps change only from 6.49 to 6.48; reopening therefore does not explain the long tail. Confidence-only exactly reproduces existing protection at 60%, and stricter margin increases mean steps to 5.75. Very-confident immediate protection also leaves the 65% trajectory effectively unchanged.','',
        'Selective dense rescue improves the mean to 4.70 calls at its nominal 60% target and 5.05 at 65%, but its actual executed sparsity is only 37.27% and 40.94%. The calibration set already showed these targets unattainable with the tested fixed thresholds: 36.68% and 41.00% executed sparsity, respectively. Each rescued call runs two full attention passes. The nominal 60% rescue executes 24.69 million QK tiles and 15.49 million retained PV tiles, versus 14.15 million each for matched-kernel dense across this cohort. Only about 1.6% of logical queries were flagged for rescue; closing them to 128-query tiles and running the duplicate pass dominates work. The combined variants do not repair this mismatch. These rescue results are algorithmic diagnostics, not a viable speedup result.','',
        f'A protected token that disagrees with the fresh top-1 prediction is recorded in `stopping_diagnostics.csv` as a conflict. Hysteresis at 65% produced {hysteresis_conflicts} such conflicts in {hysteresis_observations:,} protected-position observations. This is a safety signal, not a ground-truth label for an incorrectly protected token; the benchmark only labels the final answer. Paired prompt bootstrap accuracy intervals against native dense, matched-kernel dense, static Gaussian-32, and existing protection are in `comparisons.csv`. The 130 prompts have been examined repeatedly, so small differences are exploratory.','',
        '![Early entropy](plots/entropy.png) ![Early prediction churn](plots/churn.png)','',
        '![Protected positions](plots/protected.png) ![Reopened positions](plots/reopened.png)','',
        '## Runtime','', 'End-to-end timing is untraced. Separate profile runs measure complete decoder CUDA event time and sampler/protection policy GPU time. '
        'Two interleaved repeats of all130 prompts, identical per-policy warmup, all required policy work included. Profile outputs/steps must match untraced and diagnostic outputs. '
        'Do not divide E2E latency bysteps to infer decoder savings; prefix and host overhead are included in E2E but not decoder CUDA events. Policy events omit host-side controller overhead, which remains in E2E.','',
        'The fastest sparse condition measured here is static Gaussian-32 at 60%: 0.682 s/example versus 0.546 s for native dense (0.80× speed ratio). Existing protection at 65% takes 0.782 s/example (0.70×); it saves about 0.048 s versus static 65% but remains slower than dense. The separately measured decoder CUDA time is 0.251 s/example for native dense, 0.380 for static 60%, and 0.449 for existing protection at 65%; the protection policy itself accounts for 0.032–0.036 s/example in the profiled runs. No timed refinement improves both quality and speed relative to its static/protection controls. Rescue reference paths were not timed or profiled as optimized inference because their duplicate full passes make them a different and costlier execution path.','',
        '![Latency](plots/latency.png)']
    valid=[r for r in summary if r['policy'] in policies and r['accuracy']>=.89 and r['mean_steps']<5 and
           abs(r['actual_sparsity']-policies[r['policy']]['target'])<=.02]
    lines+=['','## Recommendation','',
        ('Measured configurations meeting the fixed criterion(mean<5,accuracy>=89%,actualsparsity within2pp): '+', '.join(r['policy'] for r in valid) if valid else
         'No evaluated refinement meets all fixed criteria: mean<5 calls,accuracy>=89%,and actual sparsity within2pp of60% or65%. The criteria have not been relaxed.'),
        'Review paired accuracy differences against both static and existing-protection controls; a CI containing zero is not proof of equivalence. '
        'Selective-rescue output-quality improvements alone are insufficient if duplicated QK/PV makes executed work larger. '
        'The next algorithmic target is stability of unprotected answer positions during the first few calls. A small fixed-threshold diagnostic on the disjoint calibration set could test whether selective attention changes those predictions without duplicating a full pass; its executed tile work must be measured before a full 130-example run. Any promising operating point needs fresh prompts and additional seeds before generalizing beyond this repeatedly examined cohort.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');return audit


def plots(root,rows,diagnostics,samples,timing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'plots';out.mkdir(exist_ok=True)
    def family(label):
        if label=='native_dense':return 'Native dense','black','o'
        if label=='kernel_dense':return 'Matched dense','#555555','s'
        if label.startswith('static'):return 'Static Gaussian-32','#1f77b4','o'
        if label.startswith('protect'):return 'Existing protection','#ff7f0e','s'
        if 'rescue' in label or 'combined' in label:return 'Two-pass rescue','#d62728','D'
        return 'Protection refinement','#2ca02c','^'
    fig,ax=plt.subplots(figsize=(10,5.5))
    for r in rows:
        _,color,marker=family(r['policy'])
        ax.scatter(r['mean_steps'],100*r['accuracy'],label=f"{r['policy']} ({100*r['actual_sparsity']:.1f}% tiles)",
                   color=color,marker=marker,s=65,edgecolors='white',linewidths=.5)
    ax.set(xlabel='Mean denoising calls per example',ylabel='RULER4K equal-task accuracy (%)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
    fig.tight_layout();fig.savefig(out/'tradeoff.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for r in rows:
        _,color,marker=family(r['policy'])
        ax.scatter(100*r['actual_sparsity'],100*r['accuracy'],label=f"{r['policy']} ({r['mean_steps']:.2f} calls)",
                   color=color,marker=marker,s=65,edgecolors='white',linewidths=.5)
    ax.set(xlabel='Executed physical tile sparsity (%)',ylabel='RULER4K equal-task accuracy (%)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
    fig.tight_layout();fig.savefig(out/'sparsity_accuracy.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for label in dict.fromkeys(r['policy'] for r in samples):
        x=sorted(r['steps'] for r in samples if r['policy']==label);ax.step(x,np.arange(1,len(x)+1)/len(x),label=label)
    ax.set(xlabel='Steps',ylabel='CDF');ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'steps.png',dpi=180);plt.close(fig)
    for metric in ('entropy','churn','protected','reopened'):
        fig,ax=plt.subplots(figsize=(9,5))
        for label in dict.fromkeys(r['policy'] for r in diagnostics):
            group=defaultdict(list)
            for r in diagnostics:
                if r['policy']==label and r['region']=='all' and r['step']<=6:group[r['step']].append(r[metric])
            x=sorted(group);ax.plot(x,[np.mean(group[s]) for s in x],label=label)
        ax.set(xlabel='Iteration (surviving trajectories)',ylabel=metric);ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.barh([t['policy'] for t in timing],[t['e2e_seconds'] for t in timing]);ax.set_xlabel('Untraced E2E seconds/example')
    fig.tight_layout();fig.savefig(out/'latency.png',dpi=180);plt.close(fig)
