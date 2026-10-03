"""Raw-shard-only accuracy, count-weighted sparsity and performance reporting."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def agreement(a,b):
    return dict(matches=sum(x==y for x,y in zip(a,b)),compared=max(len(a),len(b)),exact=a==b)


def paired_ci(rows,first,second,field='score'):
    groups=defaultdict(list)
    for row,a,b in rows:groups[row['task']].append(float(a[field])-float(b[field]))
    if not groups:return None
    rng=np.random.default_rng(20260920);draws=[]
    for values in groups.values():
        values=np.asarray(values);draws.append(values[rng.integers(0,len(values),(10000,len(values)))].mean(1))
    bootstrap=np.stack(draws).mean(0)
    return dict(first=first,second=second,field=field,mean=float(np.mean([np.mean(v) for v in groups.values()])),
        lower=float(np.quantile(bootstrap,.025)),upper=float(np.quantile(bootstrap,.975)),
        convention='paired prompt bootstrap within each task, equal-task mean, 10000 draws; fixed seed 20260920')


def report(root):
    from .experiment import atomic,sha,shard_path,validate_shard
    contract=json.loads((root/'configuration.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    violations=[];missing=[];raw={};counts=defaultdict(lambda:defaultdict(lambda:dict(eligible=0,skipped=0)))
    snapshots=json.loads((root/'source_snapshots.json').read_text()) if (root/'source_snapshots.json').exists() else {}
    if [r['id'] for r in manifest]!=contract['ids']:violations.append('manifest IDs changed')
    for name,wanted in contract['source_hashes'].items():
        stored=Path(snapshots.get(name,name))
        if not stored.exists() or sha(stored)!=wanted:violations.append('changed or missing archived source: '+name)
    for method in contract['methods']:
        for row in manifest:
            path=shard_path(root,method,row)
            if not path.exists():missing.append(dict(method=method,id=row['id']));continue
            try:
                value=json.loads(path.read_text());validate_shard(value,row,contract,method)
                if not 0<=value['score']<=1 or value['wall_seconds']<=0:raise ValueError('Invalid score/timing')
                if method!='native_dense':
                    records=Path(value['routing_path'])
                    if sha(records)!=value['routing_sha256']:raise ValueError('Routing checksum mismatch')
                    with gzip.open(records,'rt') as f:stats=json.load(f)
                    if {s['layer'] for s in stats}!=set(range(30)):raise ValueError('Incomplete layer coverage')
                    if {s['head'] for s in stats}!=set(range(16)):raise ValueError('Incomplete head coverage')
                    for s in stats:
                        if not 0<=s['skipped']<=s['eligible']:raise ValueError('Invalid physical tile count')
                        if s['attention_type'] not in ('local','global'):raise ValueError('Invalid layer classification')
                        if sum(s[r+'_eligible'] for r in ('prefix','canvas','boundary'))!=s['eligible']:raise ValueError('Region count mismatch')
                        for group in ('whole',s['attention_type']):
                            for key in ('eligible','skipped'):counts[method][group][key]+=s[key]
                raw[method,row['id']]=value
            except Exception as exc:violations.append(f'{path}: {exc!r}')
    cached={};source=Path(contract['source'])
    for method,old in [('cached_dense','dense'),('cached_gaussian32','jl_gaussian_r32_s70'),('cached_blasst','blasst_s70')]:
        # Historical dense is cached under dense/dense, not final/dense.
        folder=source/('dense' if old=='dense' else 'final')/old/'shards'
        for path in folder.glob('*.json'):
            value=json.loads(path.read_text());cached[method,value['id']]=value
    summary=[];per_task=[];comparisons=[];pairwise=[]
    for method in contract['methods']:
        samples=[raw[method,row['id']] for row in manifest if (method,row['id']) in raw]
        if not samples:continue
        task_scores=defaultdict(list)
        for sample in samples:task_scores[sample['task']].append(sample['score'])
        for task,values in sorted(task_scores.items()):per_task.append(dict(method=method,task=task,n=len(values),accuracy=float(np.mean(values))))
        seconds=sum(s['wall_seconds'] for s in samples);tokens=sum(len(s['completion_tokens']) for s in samples)
        item=dict(method=method,n=len(samples),accuracy=float(np.mean([np.mean(v) for v in task_scores.values()])),
                  generation_seconds=seconds,generated_tokens=tokens,tokens_per_second=tokens/seconds,
                  denoising_steps=sum(s['metadata']['actual_denoising_step_count'] for s in samples),
                  median_seconds=float(np.median([s['wall_seconds'] for s in samples])),
                  thresholds=contract['policies'].get(method),target_sparsity=0. if method=='native_dense' else .70)
        for group in ('whole','local','global'):
            c=counts[method][group]
            item[group+'_eligible']=c['eligible'];item[group+'_skipped']=c['skipped']
            item[group+'_sparsity']=c['skipped']/c['eligible'] if c['eligible'] else (0. if method=='native_dense' else None)
        for baseline in ('native_dense','cached_dense','cached_gaussian32','cached_blasst'):
            pairs=[];metrics=[]
            for row in manifest:
                a=raw.get((method,row['id']));b=(raw if baseline=='native_dense' else cached).get((baseline,row['id']))
                if a is None or b is None:continue
                if a['prompt_hash']!=b['prompt_hash'] or a['seed']!=b['seed'] or a['generation_budget']!=b['generation_budget']:
                    violations.append(f'Incompatible comparison {method}/{baseline}/{row["id"]}');continue
                metrics.append(agreement(a['completion_tokens'],b['completion_tokens']));pairs.append((row,a,b))
            if pairs:
                matches=sum(m['matches'] for m in metrics);compared=sum(m['compared'] for m in metrics)
                comparison=dict(method=method,baseline=baseline,n=len(pairs),matches=matches,compared=compared,
                    token_agreement=matches/compared if compared else 1.,sequence_exact=float(np.mean([m['exact'] for m in metrics])),
                    accuracy_delta_ci=paired_ci(pairs,method,baseline))
                if baseline=='native_dense':
                    comparison['total_wall_speedup']=sum(b['wall_seconds'] for _,a,b in pairs)/sum(a['wall_seconds'] for _,a,b in pairs)
                    item['native_dense_wall_speedup']=comparison['total_wall_speedup'];item['native_dense_token_agreement']=comparison['token_agreement']
                comparisons.append(comparison)
        summary.append(item)
    audit=dict(passed=not missing and not violations,expected=len(manifest)*len(contract['methods']),completed=len(raw),
        missing=missing,violations=violations,scope='completion/provenance/count audit, not a production-readiness or speedup certificate')
    result=dict(configuration=contract,summary=summary,per_task=per_task,comparisons=comparisons,audit=audit,
        agreement_definition='Positional token IDs after and before divergence; missing/extra positions are disagreements. Micro ratio of summed matches/compared.',
        sparsity_definition='sum(skipped eligible 128x64 tiles)/sum(eligible tiles); no mean of per-call percentages',
        timing_caveat='Wall time includes model, host scheduling, projection/refresh and output decoding. Generated lengths and denoising steps can differ. No profiler events unless configuration.profile is true.',
        retained_mass_scope='Shared-state diagnostics are in the engineering benchmark bundle, not recomputed on every final generation')
    atomic(root/'summary.json',result);atomic(root/'audit.json',audit)
    for name,rows in [('summary.csv',summary),('per_task.csv',per_task),('comparisons.csv',comparisons)]:
        if not rows:continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    def pct(x):return '—' if x is None else f'{100*x:.2f}'
    lines=['# H100 value-direction attention evaluation','',
           f"Audit: {'complete' if audit['passed'] else 'INCOMPLETE'}; {audit['completed']}/{audit['expected']} completed outputs.",'',
           f"Frozen {contract['split']} set: {len(manifest)} prompts; original budgets, seed, model revision and thresholds. Gaussian32 seed1729; 128Q×64KV; BF16 QKV/FP32 routing.",'',
           '**This report is not a production-readiness claim.** Original native SDPA ignores a local-window argument that the historical sparse reference enforces. The D512 baseline has not been established as FA3. Speed comparisons must carry these qualifications.','',
           '|Method|N|Accuracy %|Whole sparsity %|Global %|Local %|Wall s|Native wall speedup|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        lines.append(f"|{x['method']}|{x['n']}|{pct(x['accuracy'])}|{pct(x['whole_sparsity'])}|{pct(x['global_sparsity'])}|{pct(x['local_sparsity'])}|{x['generation_seconds']:.2f}|{x.get('native_dense_wall_speedup',float('nan')):.3f}×|")
    lines+=['','Sparsity uses summed physical tile counts. Latency is synchronized generation wall time, not theoretical savings. Different output lengths/denoising counts affect end-to-end comparisons. No timings from historical cached outputs are used as contemporary speed baselines.','',
            '## Matched accuracy and output comparisons','',
            '|Method|Reference|N|Accuracy delta (pp), paired 95% CI|Token agreement %|Sequence exact %|',
            '|---|---|---:|---:|---:|---:|']
    for x in comparisons:
        c=x['accuracy_delta_ci'];lines.append(f"|{x['method']}|{x['baseline']}|{x['n']}|{100*c['mean']:+.2f} [{100*c['lower']:+.2f}, {100*c['upper']:+.2f}]|{pct(x['token_agreement'])}|{pct(x['sequence_exact'])}|")
    lines+=['','## Per-task results','', '|Task|Method|N|Accuracy %|','|---|---|---:|---:|']
    lines += [f"|{x['task']}|{x['method']}|{x['n']}|{pct(x['accuracy'])}|" for x in per_task]
    lines+=['','Thresholds, effective per-call λ, projection hashes, raw generations and compressed per-layer/head/step routing counts are saved with the shards. BLASST uses the previously frozen inverse-valid-length policy, including its existing permission to exceed λ=1; it is not mislabeled original λ≤1 BLASST.','',
            'Accuracy uncertainty is a paired, task-stratified prompt bootstrap. These are previously examined examples, not fresh held-out confirmation. See configuration.json, comparisons.csv and audit.json for exact provenance and completeness.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # Standard static scientific figure; no interactive app or visualization skill.
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        for x in summary:
            if x['whole_sparsity'] is not None:ax.scatter(x['whole_sparsity']*100,x['accuracy']*100,label=x['method'])
        ax.set(xlabel='Measured physical sparsity (%)',ylabel='RULER accuracy (%)');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(root/'accuracy_actual_sparsity.png',dpi=160);plt.close(fig)
    print(json.dumps(audit),flush=True)
    return result
